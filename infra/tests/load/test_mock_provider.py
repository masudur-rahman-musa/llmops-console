r"""`infra/load/mock-provider/main.py`: the fixed-delay OpenAI-compatible
provider the k6 load run puts behind the real gateway.

The module lives outside `opsconsole` on purpose (its image installs only FastAPI
and uvicorn), so it is loaded by file path, the way
`tests/evals/test_eval_target_stub.py` loads the kind e2e's stub.

Every delay goes through the module-level `sleep`, looked up at call time. The
cases below replace it with an async fake that records the seconds asked for
and returns at once, so the suite never waits and the delays themselves are
still asserted exactly. No backing service, no socket: `TestClient` only.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_DIR = REPO_ROOT / "infra" / "load" / "mock-provider"
MOCK_PATH = MOCK_DIR / "main.py"
DOCKERFILE = MOCK_DIR / "Dockerfile"

CHAT_BODY: dict[str, Any] = {
    "model": "load-echo-model-7",
    "messages": [{"role": "user", "content": "how do I reset my password?"}],
}
EMBED_BODY: dict[str, Any] = {
    "model": "load-echo-embed-3",
    "input": "a sentence to embed",
}


def _load_mock() -> ModuleType:
    """Load `main.py` by path, the way a script outside every package has to be.

    Registered in `sys.modules` before it executes, as uvicorn's own import of
    `main` would be: Pydantic resolves a request model's string annotations
    (`from __future__ import annotations`) through the module's entry there, and
    without it FastAPI refuses the body model at the first request."""
    spec = importlib.util.spec_from_file_location("load_mock_provider", MOCK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mock() -> ModuleType:
    return _load_mock()


@pytest.fixture
def slept(mock: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every delay the mock asks for, in order; the fake returns immediately."""
    calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(mock, "sleep", fake_sleep)
    return calls


@pytest.fixture
def client(mock: ModuleType, slept: list[float]) -> Iterator[TestClient]:
    """A client over the mock app with `sleep` already patched (via `slept`)."""
    with TestClient(mock.app) as test_client:
        yield test_client


def is_int(value: object) -> bool:
    """A JSON integer, refusing `bool` (an `int` subclass in Python)."""
    return type(value) is int


def sse_events(body: str) -> list[str]:
    """The `data:` payloads of an SSE body, in order. Every non-empty event must be
    a `data: ` line; anything else fails here rather than being skipped."""
    events: list[str] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), f"not an SSE data event: {block!r}"
        events.append(block[len("data: ") :])
    return events


def stream_events(client: TestClient) -> tuple[str, list[str]]:
    response = client.post("/v1/chat/completions", json={**CHAT_BODY, "stream": True})
    assert response.status_code == 200
    return response.headers["content-type"], sse_events(response.text)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CHAT_DELAY_S", 0.4),
        ("STREAM_CHUNKS", 10),
        ("STREAM_CHUNK_DELAY_S", 0.04),
        ("EMBED_DELAY_S", 0.02),
        ("EMBED_DIM", 384),
        ("PORT", 8090),
    ],
)
def test_the_mock_constants_hold_their_agreed_values(
    mock: ModuleType, name: str, value: float
) -> None:
    """Criteria 2-4: catches a mock whose delay, chunk count,
    vector width or port drifts from the value the README's load numbers assume."""
    assert getattr(mock, name) == value


def test_health_answers_ok(client: TestClient) -> None:
    """Criterion 1: catches a mock with no `/health`, which the compose healthcheck
    needs before the gateway may start."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("extra", [{}, {"stream": False}], ids=["stream-absent", "stream-false"])
def test_non_streaming_chat_returns_one_choice_and_a_usage_block(
    client: TestClient, extra: dict[str, Any]
) -> None:
    """Criteria 1-2: catches a response missing `usage` or its total, one that
    hard-codes the model instead of echoing it, or one that treats an explicit
    `stream: false` as a stream."""
    response = client.post("/v1/chat/completions", json={**CHAT_BODY, **extra})
    assert response.status_code == 200
    document = response.json()
    assert isinstance(document["id"], str)
    assert document["object"] == "chat.completion"
    assert is_int(document["created"])
    assert document["model"] == CHAT_BODY["model"]
    assert len(document["choices"]) == 1
    choice = document["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
    usage = document["usage"]
    assert all(is_int(usage[key]) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_non_streaming_chat_sleeps_the_chat_delay_exactly_once(
    client: TestClient, slept: list[float]
) -> None:
    """Criterion 2: catches a mock that skips the delay, calls `asyncio.sleep`
    around the seam, sleeps twice, or sleeps a stream-chunk delay instead."""
    response = client.post("/v1/chat/completions", json=CHAT_BODY)
    assert response.status_code == 200
    assert slept == [0.4]


def test_a_stream_is_ten_content_chunks_then_one_usage_chunk_then_done(
    client: TestClient,
) -> None:
    """Criterion 3: catches a stream with the wrong chunk count, a missing or
    misplaced usage chunk, a `finish_reason` on the wrong chunk, or no final
    `[DONE]` (which leaves a streaming client waiting)."""
    content_type, events = stream_events(client)
    assert content_type.startswith("text/event-stream")
    assert len(events) == 12
    assert events[-1] == "[DONE]"
    assert events.count("[DONE]") == 1

    chunks = [json.loads(event) for event in events[:10]]
    for position, chunk in enumerate(chunks):
        assert chunk["object"] == "chat.completion.chunk"
        assert len(chunk["choices"]) == 1
        choice = chunk["choices"][0]
        assert choice["index"] == 0
        assert isinstance(choice["delta"]["content"], str) and choice["delta"]["content"]
        expected_finish = "stop" if position == 9 else None
        assert choice["finish_reason"] == expected_finish, f"chunk {position}"

    usage_chunk = json.loads(events[10])
    assert usage_chunk["choices"] == []
    usage = usage_chunk["usage"]
    assert all(is_int(usage[key]) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_a_stream_sleeps_the_chunk_delay_once_per_content_chunk(
    client: TestClient, slept: list[float]
) -> None:
    """Criterion 3: catches a stream that sleeps once in total, adds the
    non-streaming 0.4 s on top, or sleeps before the usage chunk as well."""
    stream_events(client)
    assert slept == [0.04] * 10


@pytest.mark.parametrize(
    "inputs",
    [["first text", "second text", "third text"], "a bare string"],
    ids=["list-of-three", "bare-string"],
)
def test_embeddings_return_one_finite_384_wide_vector_per_input(
    client: TestClient, inputs: str | list[str]
) -> None:
    """Criterion 4: catches a mock returning one vector for a list, indices out of
    order, the wrong width, a non-finite value, a hard-coded model, or no usage."""
    expected_count = 1 if isinstance(inputs, str) else len(inputs)
    response = client.post("/v1/embeddings", json={"model": EMBED_BODY["model"], "input": inputs})
    assert response.status_code == 200
    document = response.json()
    assert document["object"] == "list"
    assert document["model"] == EMBED_BODY["model"]
    assert [item["index"] for item in document["data"]] == list(range(expected_count))
    for item in document["data"]:
        assert item["object"] == "embedding"
        vector = item["embedding"]
        assert len(vector) == 384
        assert all(isinstance(value, float) and math.isfinite(value) for value in vector)
    assert is_int(document["usage"]["prompt_tokens"])
    assert is_int(document["usage"]["total_tokens"])


def test_embeddings_sleep_the_embed_delay_once_per_request(
    client: TestClient, slept: list[float]
) -> None:
    """Criterion 4: two inputs in one request, so catches a mock that sleeps per
    input, skips the delay, or sleeps the chat delay instead."""
    response = client.post(
        "/v1/embeddings", json={"model": EMBED_BODY["model"], "input": ["one", "two"]}
    )
    assert response.status_code == 200
    assert slept == [0.02]


def test_the_same_text_embeds_to_the_same_vector_and_different_texts_do_not(
    client: TestClient,
) -> None:
    """Criterion 4 (deterministic): catches random vectors (the repeat differs) and
    a constant vector (two different texts collide), either of which would make
    the gateway's semantic cache behave unlike it does in production."""

    def embed(text: str) -> list[float]:
        response = client.post("/v1/embeddings", json={"model": "m", "input": text})
        assert response.status_code == 200
        vector: list[float] = response.json()["data"][0]["embedding"]
        return vector

    first = embed("the repeated prompt")
    assert embed("the repeated prompt") == first
    assert embed("a different prompt") != first


@pytest.mark.parametrize(
    ("path", "body", "missing"),
    [
        ("/v1/chat/completions", CHAT_BODY, "model"),
        ("/v1/chat/completions", CHAT_BODY, "messages"),
        ("/v1/embeddings", EMBED_BODY, "model"),
        ("/v1/embeddings", EMBED_BODY, "input"),
    ],
)
def test_a_request_missing_a_required_field_is_422(
    client: TestClient, path: str, body: dict[str, Any], missing: str
) -> None:
    """Unhappy-path criterion: each row drops exactly one field from an otherwise
    valid body, so catches a mock that gives that one field a default."""
    partial = {key: value for key, value in body.items() if key != missing}
    response = client.post(path, json=partial)
    assert response.status_code == 422


@pytest.mark.parametrize("path", [MOCK_PATH, DOCKERFILE], ids=["main.py", "Dockerfile"])
def test_the_mock_and_its_image_carry_nothing_of_opsconsole(path: Path) -> None:
    """Criterion 1: catches a mock importing gateway types, or an image installing
    the package, which would let load pass through code the gateway also runs."""
    assert "opsconsole" not in path.read_text(encoding="utf-8")


def dockerfile_instructions() -> list[str]:
    r"""The Dockerfile's instructions with `\` continuations joined and comments
    dropped, so a pin split across lines is still one instruction."""
    joined: list[str] = []
    pending = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        joined.append(pending + line)
        pending = ""
    if pending:
        joined.append(pending)
    return joined


def test_the_mock_image_is_python_312_slim() -> None:
    """Criterion 1: catches a different or unpinned base image (every `FROM` is
    scanned, so a second stage cannot hide one)."""
    froms = [line for line in dockerfile_instructions() if line.upper().startswith("FROM ")]
    assert froms, "the Dockerfile has no FROM"
    for line in froms:
        assert line.split()[1] == "python:3.12-slim", line


def test_the_mock_image_pins_fastapi_and_uvicorn_to_the_lockfile_versions() -> None:
    r"""a project rule: catches an install of a version other than `uv.lock`'s, or any
    unpinned package on a `pip install` line. Arguments starting with `-` are
    flags; every other argument must carry `==`."""
    installs = [
        line
        for line in dockerfile_instructions()
        if re.search(r"\bpip\s+install\b", line) and not line.lstrip().startswith("#")
    ]
    assert installs, "the Dockerfile installs nothing"
    packages: list[str] = []
    for line in installs:
        for command in re.split(r"&&|;", line):
            match = re.search(r"\bpip\s+install\s+(.*)$", command)
            if match is None:
                continue
            for argument in match.group(1).split():
                if argument.startswith("-"):
                    continue
                packages.append(argument.strip("'\""))
    assert "fastapi==0.141.1" in packages
    assert "uvicorn==0.52.4" in packages
    unpinned = [package for package in packages if "==" not in package]
    assert unpinned == []
