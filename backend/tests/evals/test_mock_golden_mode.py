r"""The load run's mock provider in golden-set mode.

`infra/load/mock-provider/main.py` stands in for the model in the eval gate. With the
environment variable `MOCK_GOLDEN_SET` pointing at a golden `.jsonl`, a `json_schema`
call named `IntentExtraction` or `ToolPlanAnswer` is answered from the golden case
whose LAST user message equals the request's last user message: its expected intent,
then its expected tool calls, then an empty plan. No match is a 400 that names the
schema. With the variable unset nothing changes.

The module sits outside the `app` package, so it is loaded by file path, the way
`infra/tests/load/test_mock_provider.py` does. It is loaded AFTER the environment is
set, so the case passes whether the mock reads the variable at import or per request.
Delays go through the module's own `sleep`, replaced here, so nothing waits. No
socket: `TestClient` only.

One thing this file assumes about the order of the two answers: a `ToolPlanAnswer`
request that already carries a `tool` message (the calls were made and answered) gets
the empty plan, and one without gets the calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_PATH = REPO_ROOT / "infra" / "load" / "mock-provider" / "main.py"

SET_ENV = "MOCK_GOLDEN_SET"

SHARED_OPENING: list[dict[str, str]] = [
    {"role": "user", "content": "I need help"},
    {"role": "assistant", "content": "With what?"},
]
"""Both golden cases start the same way, so a match on the FIRST user message cannot tell
them apart and only the last user message can."""

VPN_MESSAGE = "my VPN is broken"
PRINTER_MESSAGE = "the printer is jammed"


def golden_line(
    case_id: str, last_message: str, intent: str, calls: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": "helpdesk",
        "tenant": "nordlicht",
        "principal_user": "dana.okafor",
        "messages": [*SHARED_OPENING, {"role": "user", "content": last_message}],
        "expected": {"intent": intent, "tool_calls": calls},
    }


GOLDEN_LINES = [
    golden_line(
        "vpn-1",
        VPN_MESSAGE,
        "troubleshoot_vpn",
        [{"name": "lookup_vpn_status", "args_subset": {"site": "berlin-office"}}],
    ),
    golden_line(
        "printer-1",
        PRINTER_MESSAGE,
        "report_printer_fault",
        [{"name": "open_printer_ticket", "args_subset": {"floor": "third-floor"}}],
    ),
]


def load_mock() -> ModuleType:
    spec = importlib.util.spec_from_file_location("golden_mode_mock_provider", MOCK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_client(monkeypatch: pytest.MonkeyPatch, golden_path: Path | None) -> Iterator[TestClient]:
    if golden_path is None:
        monkeypatch.delenv(SET_ENV, raising=False)
    else:
        monkeypatch.setenv(SET_ENV, str(golden_path))
    mock = load_mock()

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(mock, "sleep", no_sleep)
    with TestClient(mock.app) as client:
        yield client


@pytest.fixture
def golden_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    path = tmp_path / "helpdesk-v1.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in GOLDEN_LINES) + "\n", encoding="utf-8")
    yield from make_client(monkeypatch, path)


@pytest.fixture
def plain_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from make_client(monkeypatch, None)


def schema_body(
    schema_name: str, last_message: str, *, extra: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A chat request asking for a structured answer, as the help desk's agent sends it."""
    return {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are the help desk agent."},
            *SHARED_OPENING,
            {"role": "user", "content": last_message},
            *(extra or []),
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }


def content_of(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    return content


def strings_in(value: object) -> list[str]:
    """Every string anywhere in a parsed JSON document, keys excluded.

    A string that is itself a JSON object is opened too: the help desk's
    `ToolPlanAnswer` carries each call's `arguments` encoded as one string."""
    if isinstance(value, str):
        if value.startswith("{"):
            try:
                return [value, *strings_in(json.loads(value))]
            except ValueError:
                pass
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings_in(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings_in(item)]
    return []


def test_intent_extraction_answers_the_matching_cases_expected_intent(
    golden_client: TestClient,
) -> None:
    """Catches a mock that answers every request with the first case, one that matches on the
    first user message (both cases share it), and one that answers a fixed intent. The answer
    must be valid JSON in the message content and carry the intent of the case whose LAST user
    message matched, and not the other case's."""
    vpn = content_of(
        golden_client.post(
            "/v1/chat/completions", json=schema_body("IntentExtraction", VPN_MESSAGE)
        )
    )
    printer = content_of(
        golden_client.post(
            "/v1/chat/completions", json=schema_body("IntentExtraction", PRINTER_MESSAGE)
        )
    )

    assert "troubleshoot_vpn" in strings_in(json.loads(vpn))
    assert "report_printer_fault" not in strings_in(json.loads(vpn))
    assert "report_printer_fault" in strings_in(json.loads(printer))
    assert "troubleshoot_vpn" not in strings_in(json.loads(printer))


def test_tool_plan_answers_the_expected_calls_then_an_empty_plan(
    golden_client: TestClient,
) -> None:
    """Catches a mock that plans the same calls forever, which would loop the agent until its
    step cap, and one that plans nothing at all. The first request must carry the expected
    tool name and its argument values; the follow-up, after a tool result is in the messages,
    must carry none of them."""
    first = json.loads(
        content_of(
            golden_client.post(
                "/v1/chat/completions", json=schema_body("ToolPlanAnswer", VPN_MESSAGE)
            )
        )
    )
    follow_up = json.loads(
        content_of(
            golden_client.post(
                "/v1/chat/completions",
                json=schema_body(
                    "ToolPlanAnswer",
                    VPN_MESSAGE,
                    extra=[{"role": "tool", "tool_call_id": "call-1", "content": "{}"}],
                ),
            )
        )
    )

    assert "lookup_vpn_status" in strings_in(first)
    assert "berlin-office" in strings_in(first)
    assert "open_printer_ticket" not in strings_in(first)
    assert "lookup_vpn_status" not in strings_in(follow_up)
    assert "berlin-office" not in strings_in(follow_up)


@pytest.mark.parametrize("schema_name", ["IntentExtraction", "ToolPlanAnswer"])
def test_an_unmatched_message_answers_400_naming_the_schema(
    golden_client: TestClient, schema_name: str
) -> None:
    """Catches a mock that falls back to a canned answer when no case matches: the gate would
    then score the agent on an answer no golden case asked for and stay green. The request
    must be refused with 400 and the body must name the schema that could not be answered."""
    response = golden_client.post(
        "/v1/chat/completions", json=schema_body(schema_name, "a message no case contains")
    )

    assert response.status_code == 400
    assert schema_name in response.text


def test_without_the_golden_set_setting_the_fixed_reply_is_unchanged(
    plain_client: TestClient,
) -> None:
    """Catches a golden mode that is always on: with `MOCK_GOLDEN_SET` unset, the same
    structured request must still get the load run's fixed reply with status 200, not a 400
    and not an intent."""
    response = plain_client.post(
        "/v1/chat/completions", json=schema_body("IntentExtraction", VPN_MESSAGE)
    )

    assert content_of(response) == "This is a fixed answer from the load-test mock provider."


def test_a_plain_chat_call_is_still_the_fixed_reply_in_golden_mode(
    golden_client: TestClient,
) -> None:
    """Catches a golden mode that takes over every call: only `json_schema` calls are
    answered from the set, so the compose step and the judge-free paths keep the fixed
    reply, even when the last user message matches no case."""
    response = golden_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "no case says this"}],
        },
    )

    assert content_of(response) == "This is a fixed answer from the load-test mock provider."
