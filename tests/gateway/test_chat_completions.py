"""`POST /v1/chat/completions`: the OpenAI wire shape, both mappings, and the failures.

Three things are under test here and they are kept apart on purpose.

* The ANSWER: the `chat.completion` envelope, the `finish_reason` table, `usage`
  carried through, and the alias-versus-provider-model split (the body names the
  alias the caller used; only `X-Gateway-Model` carries the provider's id, which
  `test_response_headers.py` holds).
* The OUTBOUND request: what the route hands the adapter. `StubAdapter` records
  every `ChatRequest`, so the wire-to-neutral mapping is asserted field by field
  rather than inferred from a successful answer. The one case that has to prove
  the whole path - a tool result reaching Anthropic in the caller's order - runs
  a real `AnthropicAdapter` over `httpx.MockTransport` and asserts the JSON that
  went out.
* The REFUSALS: the coherence table on `messages`, an unknown body field, and
  the four routing failures. Each row is arranged so the control
  it names is the only thing that can reject it; a row rejected by two rules at
  once would stay green after its own rule was deleted.

No socket is opened, no provider key is read and no backing service is started:
the successful cases answer from `StubAdapter`, the two adapter-level cases from
recorded payloads under `tests/providers/fixtures/anthropic/` (#66).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from providers.conftest import PROVIDER_DETAIL_MARKER, fail, make_adapter, ok

from opsconsole.core.errors import ProviderError
from opsconsole.core.settings import Settings
from opsconsole.gateway.routes.chat import COMPLETION_ID_BYTES, COMPLETION_ID_PREFIX
from opsconsole.providers.base import (
    Adapter,
    ChatMessage,
    ChatRequest,
    ToolCall,
    ToolChoice,
    ToolSpec,
)

from .conftest import (
    CHAT,
    STUB_CONTENT,
    STUB_RESPONSE_MODEL,
    STUB_TOOL_CALL,
    STUB_USAGE,
    TEST_KEYS,
    StubAdapter,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    post_chat,
    stub_response,
)

NORDLICHT_KEY = TEST_KEYS["nordlicht"]

# The whole id: the published prefix, then `COMPLETION_ID_BYTES` bytes as hex.
COMPLETION_ID_HEX_PATTERN = re.compile(f"{re.escape(COMPLETION_ID_PREFIX)}[0-9a-f]+")

# The alias the shipped policy resolves, for `evals` under `internal`, onto the
# US `anthropic` provider. `test_response_headers.py` derives the provider model
# id from `resolve()`; here only the alias matters, because the alias is what the
# response body must name.
ALIAS = "agent-default"

WEATHER_TOOL_WIRE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "severity": {"type": "integer"}},
    "required": ["title", "severity"],
}

RESPONSE_FORMAT_WIRE: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "ticket_summary", "schema": TICKET_SCHEMA},
}

# One complete tool loop as an OpenAI client sends it back: the assistant turn
# that asked for the call, then the result of that call.
ASSISTANT_TOOL_CALL_WIRE: dict[str, Any] = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "toolu_01WeatherCall00000000",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
        }
    ],
}

TOOL_RESULT_WIRE: dict[str, Any] = {
    "role": "tool",
    "tool_call_id": "toolu_01WeatherCall00000000",
    "content": "18C, clear",
}

TOOL_LOOP_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Weather in Berlin?"},
    ASSISTANT_TOOL_CALL_WIRE,
    TOOL_RESULT_WIRE,
]


def ok_body(response: Response) -> dict[str, Any]:
    """Assert a 200 and return its parsed body, so a failure shows the wire text."""
    assert response.status_code == 200, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


# --- the protocol the stub stands in for -------------------------------------


def test_stub_adapter_satisfies_the_protocol() -> None:
    """Catches a stub that has drifted from the real `Adapter` (a `complete` instead of a
    `chat`, or no `price_key`), which would let every success case in this file pass
    against a contract no shipped adapter implements."""
    stub = StubAdapter(name="anthropic")

    assert isinstance(stub, Adapter)
    assert stub.name == "anthropic"
    assert stub.price_key("some-model") == "anthropic/some-model"


# --- the answer --------------------------------------------------------------


def test_successful_completion_is_openai_shaped(chat_client: TestClient) -> None:
    """Catches a route answering the neutral `ChatResponse` straight out (`content`,
    `finish_reason`, `usage` at the top level), which no OpenAI-compatible client can
    read: it looks for `choices[0].message.content`."""
    body = ok_body(post_chat(chat_client, chat_body()))

    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert len(body["id"]) > len("chatcmpl-")
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert set(choice) == {"index", "message", "finish_reason"}
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == STUB_CONTENT


def test_completion_id_is_random_and_the_pinned_length(chat_client: TestClient) -> None:
    """Catches a shortened, counted or per-process completion id. `COMPLETION_ID_BYTES` is a
    bound this plan introduced, and the ledger and the trace store key rows by this id: a
    counter restarting with the process would merge two tenants' calls into one row, and
    every other assertion in this file would stay green."""
    first = ok_body(post_chat(chat_client, chat_body()))["id"]
    second = ok_body(post_chat(chat_client, chat_body()))["id"]

    assert first != second
    assert COMPLETION_ID_HEX_PATTERN.fullmatch(first) is not None
    assert COMPLETION_ID_HEX_PATTERN.fullmatch(second) is not None
    assert COMPLETION_ID_BYTES == 12
    assert len(first) == len(COMPLETION_ID_PREFIX) + 2 * COMPLETION_ID_BYTES


def test_created_is_unix_seconds_not_milliseconds(chat_client: TestClient) -> None:
    """Catches `created` built from `time.time_ns()` or `* 1000`: a client rendering it as
    a date would show a timestamp fifty thousand years out, and the field is one an
    OpenAI-compatible SDK parses rather than passes through."""
    before = int(time.time())

    body = ok_body(post_chat(chat_client, chat_body()))

    assert isinstance(body["created"], int)
    assert before <= body["created"] <= int(time.time()) + 1


def test_response_model_is_the_alias_the_caller_named(chat_client: TestClient) -> None:
    """Catches a route echoing the adapter's `ChatResponse.model` (or the resolved
    provider model id) into the body: that discloses which provider served the request and
    hands back a name the caller cannot send again, since only aliases are routable."""
    body = ok_body(post_chat(chat_client, chat_body(model=ALIAS)))

    assert body["model"] == ALIAS
    assert body["model"] != STUB_RESPONSE_MODEL
    assert STUB_RESPONSE_MODEL not in json.dumps(body)


def test_usage_is_carried_through_unchanged(chat_client: TestClient) -> None:
    """Catches the zero-filled or recomputed usage (`total = prompt`, or a `0` stand-in
    until the ledger lands): the ledger prices from these three numbers, so a wrong one
    books a real, billed call at the wrong cost."""
    body = ok_body(post_chat(chat_client, chat_body()))

    assert body["usage"] == {
        "prompt_tokens": STUB_USAGE.prompt_tokens,
        "completion_tokens": STUB_USAGE.completion_tokens,
        "total_tokens": STUB_USAGE.total_tokens,
    }
    assert body["usage"]["prompt_tokens"] == 137
    assert body["usage"]["completion_tokens"] == 29
    assert body["usage"]["total_tokens"] == 166


# The whole table, unknown row included. The neutral contract
# carries the provider's own stop reason; OpenAI publishes a different set, and a
# client branches on it to decide whether to run a tool.
FINISH_REASON_CASES: list[tuple[str, str, str]] = [
    ("a finished answer", "end_turn", "stop"),
    ("a stop sequence", "stop_sequence", "stop"),
    ("the token ceiling", "max_tokens", "length"),
    ("a tool call", "tool_use", "tool_calls"),
    ("an unknown reason passes through", "refusal_not_yet_mapped", "refusal_not_yet_mapped"),
]


@pytest.mark.parametrize(
    ("case", "provider_reason", "expected"),
    FINISH_REASON_CASES,
    ids=[case for case, *_ in FINISH_REASON_CASES],
)
def test_finish_reason_table(
    case: str, provider_reason: str, expected: str, settings: Settings
) -> None:
    """Catches a route passing the provider's stop reason through unmapped: a client that
    branches on `tool_calls` would never run a tool, and one that branches on `length`
    would never notice a truncated answer. The unknown row catches the opposite mistake, a
    mapping that collapses everything it does not recognise into "stop"."""
    stub = StubAdapter(name="anthropic", response=stub_response(finish_reason=provider_reason))

    with build_chat_client(settings, {"anthropic": stub}) as client:
        body = ok_body(post_chat(client, chat_body()))

    assert body["choices"][0]["finish_reason"] == expected, case


def test_returned_tool_calls_are_openai_shaped(settings: Settings) -> None:
    """Catches a route returning the neutral `ToolCall` verbatim (`{"id", "name",
    "arguments"}`): an OpenAI client reads the name from `function.name`, so it would
    dispatch nothing and see a message with neither content nor a callable tool."""
    stub = StubAdapter(
        name="anthropic",
        response=stub_response(content="", tool_calls=(STUB_TOOL_CALL,), finish_reason="tool_use"),
    )

    with build_chat_client(settings, {"anthropic": stub}) as client:
        body = ok_body(post_chat(client, chat_body()))

    message = body["choices"][0]["message"]
    assert message["tool_calls"] == [
        {
            "id": STUB_TOOL_CALL.id,
            "type": "function",
            "function": {"name": STUB_TOOL_CALL.name, "arguments": STUB_TOOL_CALL.arguments},
        }
    ]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Berlin"}
    assert body["choices"][0]["finish_reason"] == "tool_calls"


# --- the outbound request ----------------------------------------------------


def last_request(stub: StubAdapter) -> ChatRequest:
    """The `ChatRequest` the route handed the adapter on the most recent call."""
    assert stub.requests, "the route never called the adapter"
    return stub.requests[-1]


def test_the_adapter_is_handed_the_resolved_provider_model_not_the_alias(
    chat_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a route passing the caller's alias straight into `ChatRequest.model`, which
    would make the provider 404 on a name only our routing policy knows."""
    ok_body(post_chat(chat_client, chat_body(model=ALIAS)))

    sent = last_request(chat_stub)
    assert sent.model != ALIAS
    assert sent.model == "claude-haiku-4-5"


def test_messages_reach_the_adapter_in_order(
    chat_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a route that rebuilds `messages` from a dict keyed by role, or that lifts
    the system turn itself: both reorder the conversation, and placing the system turn is
    the adapter's job, not the wire layer's."""
    ok_body(
        post_chat(
            chat_client,
            chat_body(
                messages=[
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Which city?"},
                    {"role": "assistant", "content": "Berlin."},
                    {"role": "user", "content": "And the country?"},
                ]
            ),
        )
    )

    sent = last_request(chat_stub)
    assert [(m.role, m.content) for m in sent.messages] == [
        ("system", "You are terse."),
        ("user", "Which city?"),
        ("assistant", "Berlin."),
        ("user", "And the country?"),
    ]


def test_a_tool_loop_maps_into_the_neutral_contract(
    chat_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a wire mapping that drops the two halves of a tool loop - an assistant turn
    whose content is empty but whose `tool_calls` are the answer, and the `tool` result
    turn that follows - leaving the provider a conversation in which a tool was never
    called and never answered."""
    ok_body(post_chat(chat_client, chat_body(messages=TOOL_LOOP_MESSAGES)))

    sent = last_request(chat_stub)
    assert sent.messages == (
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="Weather in Berlin?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(
                ToolCall(
                    id="toolu_01WeatherCall00000000",
                    name="get_weather",
                    arguments='{"city": "Berlin"}',
                ),
            ),
        ),
        ChatMessage(role="tool", content="18C, clear", tool_call_id="toolu_01WeatherCall00000000"),
    )


def test_tools_reach_the_adapter_as_neutral_tool_specs(
    chat_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a route forwarding OpenAI's `{"type": "function", "function": {...}}`
    wrapper into the neutral contract, which would push one provider's wire shape through
    every adapter that reads `ToolSpec`."""
    ok_body(post_chat(chat_client, chat_body(tools=[WEATHER_TOOL_WIRE])))

    sent = last_request(chat_stub)
    assert sent.tools == (
        ToolSpec(
            name="get_weather",
            description="Look up the weather in a city.",
            parameters=WEATHER_TOOL_WIRE["function"]["parameters"],
        ),
    )


def test_response_format_reaches_the_adapter(
    chat_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a route that accepts `response_format` and drops it: the answer would come
    back as prose while the caller's client parses it as JSON, which is the failure a
    schema demand exists to prevent."""
    ok_body(post_chat(chat_client, chat_body(response_format=RESPONSE_FORMAT_WIRE)))

    sent = last_request(chat_stub)
    assert sent.response_format is not None
    assert sent.response_format.name == "ticket_summary"
    assert dict(sent.response_format.json_schema) == TICKET_SCHEMA


ToolChoiceWire = str | dict[str, Any] | None

TOOL_CHOICE_CASES: list[tuple[str, ToolChoiceWire, ToolChoice | None]] = [
    ("omitted stays unset", None, None),
    ("auto", "auto", ToolChoice(mode="auto")),
    ("none", "none", ToolChoice(mode="none")),
    ("required", "required", ToolChoice(mode="required")),
    (
        "a named function",
        {"type": "function", "function": {"name": "get_weather"}},
        ToolChoice(mode="tool", name="get_weather"),
    ),
]


@pytest.mark.parametrize(
    ("case", "wire", "expected"),
    TOOL_CHOICE_CASES,
    ids=[case for case, *_ in TOOL_CHOICE_CASES],
)
def test_tool_choice_maps_into_the_neutral_contract(
    case: str,
    wire: ToolChoiceWire,
    expected: ToolChoice | None,
    chat_client: TestClient,
    chat_stub: StubAdapter,
) -> None:
    """Catches a route that forwards OpenAI's `tool_choice` union unchanged, and the
    narrower mistake of collapsing `"required"` and `"auto"` into one mode - the first
    leaves the model free to answer in prose when the caller demanded a call."""
    body = chat_body(tools=[WEATHER_TOOL_WIRE])
    if wire is not None:
        body["tool_choice"] = wire

    ok_body(post_chat(chat_client, body))

    assert last_request(chat_stub).tool_choice == expected, case


def test_tool_result_turn_reaches_the_provider_in_order(settings: Settings) -> None:
    """The whole path, through the real adapter: catches any break between the wire body
    and Anthropic's own JSON - a tool result rendered as a `tool` role Anthropic rejects,
    a result placed before the assistant turn that asked for it, or an empty text block
    prepended to a tool-call-only assistant turn (Anthropic 400s on empty text)."""
    adapter, scripted = make_adapter([ok("tool_result_followup")], name="anthropic")

    with build_chat_client(settings, {"anthropic": adapter}) as client:
        body = ok_body(post_chat(client, chat_body(messages=TOOL_LOOP_MESSAGES)))

    assert scripted.request_count == 1
    sent = scripted.body()
    assert sent["system"] == "You are terse."
    assert sent["messages"] == [
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01WeatherCall00000000",
                    "name": "get_weather",
                    "input": {"city": "Berlin"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01WeatherCall00000000",
                    "content": "18C, clear",
                }
            ],
        },
    ]
    assert body["choices"][0]["message"]["content"] == "It is 18C in Berlin."
    assert body["usage"] == {
        "prompt_tokens": 152,
        "completion_tokens": 11,
        "total_tokens": 163,
    }


# --- refusals: the request body ----------------------------------------------


def test_stream_false_is_accepted(chat_client: TestClient) -> None:
    """Catches a route that takes the streaming branch on any `stream` key at all: a
    client sending `stream: false` explicitly, as the OpenAI SDK does, would be handed
    server-sent events it is not reading. `stream: true` is served, and asserted, in
    tests/gateway/test_chat_streaming.py."""
    ok_body(post_chat(chat_client, chat_body(stream=False)))


def test_unknown_body_field_is_422(chat_client: TestClient) -> None:
    """Catches a request model left on Pydantic's default `extra="ignore"`: a misspelled
    `max_token` would be dropped and the call served with the default ceiling, truncating
    an answer the caller thought it had paid to widen (decision F)."""
    response = post_chat(chat_client, chat_body(max_token=16))

    assert_error_body(response, status=422, code="VALIDATION")
    assert "max_token" in response.json()["error"]["message"]


def test_tool_choice_with_response_format_is_422(chat_client: TestClient) -> None:
    """Catches a route accepting both: each demands a different forced tool call, so
    whichever is applied last silently wins and the caller gets an answer shaped by the
    field it did not think was in charge."""
    response = post_chat(
        chat_client,
        chat_body(
            tools=[WEATHER_TOOL_WIRE],
            tool_choice="required",
            response_format=RESPONSE_FORMAT_WIRE,
        ),
    )

    assert_error_body(response, status=422, code="VALIDATION")


def test_empty_messages_is_422(chat_client: TestClient) -> None:
    """Catches a route that forwards an empty conversation, which every provider answers
    with its own 400 - a provider error for what is plainly the caller's mistake."""
    response = post_chat(chat_client, chat_body(messages=[]))

    assert_error_body(response, status=422, code="VALIDATION")


# Each row fails exactly one coherence rule and satisfies the others, so deleting
# the rule it names is the only edit that turns the row green. There is
# deliberately no "a tool turn with no content" row: the general "content is
# required" rule already rejects it, so the row would be green with the
# tool-specific clause deleted and would pin nothing of its own.
INCOHERENT_MESSAGES: list[tuple[str, dict[str, Any]]] = [
    ("a tool turn with no tool_call_id", {"role": "tool", "content": "18C, clear"}),
    (
        "a user turn carrying tool_call_id",
        {"role": "user", "content": "hi", "tool_call_id": "toolu_01WeatherCall00000000"},
    ),
    (
        "a system turn carrying tool_call_id",
        {"role": "system", "content": "be terse", "tool_call_id": "toolu_01WeatherCall00000000"},
    ),
    (
        "a user turn carrying tool_calls",
        {"role": "user", "content": "hi", "tool_calls": ASSISTANT_TOOL_CALL_WIRE["tool_calls"]},
    ),
    (
        "a tool turn carrying tool_calls",
        {
            "role": "tool",
            "content": "18C, clear",
            "tool_call_id": "toolu_01WeatherCall00000000",
            "tool_calls": ASSISTANT_TOOL_CALL_WIRE["tool_calls"],
        },
    ),
    ("an assistant turn with neither content nor tool_calls", {"role": "assistant"}),
    ("a user turn with no content", {"role": "user"}),
    # Empty, not absent. Anthropic rejects an empty text block, so a route that
    # only checks for `None` turns the caller's own mistake into a 502 carrying
    # the provider's refusal.
    ("a user turn with empty content", {"role": "user", "content": ""}),
    (
        "a tool turn with empty content",
        {"role": "tool", "content": "", "tool_call_id": "toolu_01WeatherCall00000000"},
    ),
    ("an unknown role", {"role": "function", "content": "18C, clear"}),
]


@pytest.mark.parametrize(
    ("case", "message"), INCOHERENT_MESSAGES, ids=[case for case, _ in INCOHERENT_MESSAGES]
)
def test_incoherent_message_is_422(
    case: str, message: dict[str, Any], chat_client: TestClient
) -> None:
    """Catches a route that types `messages` loosely and lets the provider judge them: a
    tool result with no id, or a tool call attached to a user turn, becomes a provider 400
    (or worse, a silently mangled conversation) instead of a 422 naming the caller's
    mistake."""
    response = post_chat(chat_client, chat_body(messages=[message]))

    assert_error_body(response, status=422, code="VALIDATION")


def test_a_complete_tool_loop_is_accepted(chat_client: TestClient) -> None:
    """The anchor for the table above: catches a coherence check so strict it rejects the
    legal shapes too - an assistant turn whose content is absent because `tool_calls` are
    the answer, and the `tool` turn that follows it."""
    ok_body(post_chat(chat_client, chat_body(messages=TOOL_LOOP_MESSAGES)))


# --- refusals: routing and the provider --------------------------------------


def test_unknown_alias_is_404(chat_client: TestClient) -> None:
    """Catches a route that hands an unknown model name to an adapter anyway, turning a
    routing failure the caller can fix into a provider error it cannot."""
    response = post_chat(chat_client, chat_body(model="no-such-alias"))

    assert_error_body(response, status=404, code="ALIAS_UNKNOWN")


def test_a_malformed_alias_is_422_not_404(chat_client: TestClient) -> None:
    """Catches a route that looks the alias up in the policy dict before the identifier
    pre-check runs: a name with a trailing newline would come back as 404 "unknown alias",
    telling the caller the name does not exist rather than that it is not a legal name."""
    response = post_chat(chat_client, chat_body(model="agent-default\n"))

    assert_error_body(response, status=422, code="VALIDATION")


def test_eu_tenant_on_a_us_alias_is_451_with_no_call(
    chat_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a route that calls the provider and then checks residency, or that widens
    the region set when nothing is allowed: `agent-strong` is US-only while `anthropic_eu`
    is disabled, so an EU tenant's request must not leave the process at all."""
    before = len(chat_stub.requests)

    response = post_chat(
        chat_client,
        chat_body(model="agent-strong"),
        key=NORDLICHT_KEY,
        tenant="nordlicht",
    )

    assert_error_body(response, status=451, code="RESIDENCY_UNAVAILABLE")
    assert len(chat_stub.requests) == before, "the provider was called for a denied request"


def test_no_adapter_for_any_candidate_is_503(chat_client: TestClient) -> None:
    """Catches a route that ignores the resolver's health seam and calls the first
    candidate regardless: `nordlicht` overrides `agent-default` onto `local`, which has no
    adapter, so the answer is an unavailability the operator can act on rather than a
    `KeyError` served as a 500."""
    response = post_chat(chat_client, chat_body(model=ALIAS), key=NORDLICHT_KEY, tenant="nordlicht")

    assert_error_body(response, status=503, code="ALL_PROVIDERS_FAILED")


def test_provider_failure_is_502_without_the_provider_message(settings: Settings) -> None:
    """Catches a route that renders the provider's own words into the client body: the
    recorded 400 fixture carries a marker string, and a client that can read an upstream
    request id or quota wording is being told about our provider relationship."""
    adapter, scripted = make_adapter([fail(400, "error_400")], name="anthropic")

    with build_chat_client(settings, {"anthropic": adapter}) as client:
        response = post_chat(client, chat_body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert scripted.request_count == 1, "a 400 is not retried"
    assert PROVIDER_DETAIL_MARKER not in response.text


def test_an_adapter_raising_provider_error_is_502(settings: Settings) -> None:
    """Catches a route that lets a `ProviderError` reach the last-resort handler as an
    untyped bug: the caller would get a 500 with `code: null` instead of the 502 its
    retry logic is written against."""
    stub = StubAdapter(
        name="anthropic",
        error=ProviderError(
            "anthropic: the provider answered 503",
            retryable=True,
            provider_message=PROVIDER_DETAIL_MARKER,
        ),
    )

    with build_chat_client(settings, {"anthropic": stub}) as client:
        response = post_chat(client, chat_body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert PROVIDER_DETAIL_MARKER not in response.text


def test_chat_requires_auth(chat_client: TestClient) -> None:
    """Catches a route declared without the request context dependency: it would serve an
    unauthenticated caller, and with no tenant there is no residency rule to apply."""
    response = chat_client.post(CHAT, json=chat_body())

    assert_error_body(response, status=401, code="UNAUTHORIZED")


def test_chat_rejects_a_tenant_that_is_not_the_keys(chat_client: TestClient) -> None:
    """Catches a route that resolves on the `X-Tenant` header without the key/tenant
    match: one tenant could then route under another's residency rules, which is the
    residency guarantee bypassed by a header."""
    response = chat_client.post(
        CHAT,
        json=chat_body(),
        headers=auth_headers(key=NORDLICHT_KEY, tenant="evals", data_class="internal"),
    )

    assert_error_body(response, status=403, code="TENANT_MISMATCH")
