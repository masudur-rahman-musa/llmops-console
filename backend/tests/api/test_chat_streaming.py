"""`POST /v1/chat/completions` with `stream: true`: the SSE frames the client reads.

Everything here goes through `TestClient` against an app `create_app` built, so
the claims about status, content type, headers and frame text are settled by a
request rather than by reading the route. The adapter behind
it is `StubAdapter`, whose `stream_chat` plays a scripted list of NEUTRAL events:
the Anthropic SSE dialect is tested one layer down, in
tests/providers/test_anthropic_streaming.py, and nothing of it may appear here.

Three properties carry the plan's risk and each has its own case. The stream is
opened and its first event pulled inside the route, so a provider failure is
still a typed 502 and not a 200 carrying an error frame
(`test_a_provider_failure_before_the_first_event_is_502_with_no_stream`). `usage`
is present on every stream and carried from the provider, never zero-filled
(`test_usage_chunk_is_last_and_carries_the_fixture_numbers_exactly`). And a
failure after the first event, which cannot change a status already sent, emits
one error frame and then ends WITHOUT `data: [DONE]`, which is the only thing
that tells a client a stream was truncated
(`test_a_failure_after_the_first_event_emits_an_error_chunk_and_no_done`).

The stub's streamed numbers and strings differ from its non-streaming ones, so a
route rendering a stream out of the non-streaming path is visible here rather
than accidentally right. No socket is opened, no provider key is read and no
backing service is started (#66).
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.core.errors import ProviderError
from app.providers.base import Adapter, ChatMessage, ChatRequest, ToolCallDelta
from app.schemas.chat import CHUNK_OBJECT
from app.services.streaming import DONE_SENTINEL, SSE_HEADERS, SSE_MEDIA_TYPE, sse_frames
from app.services.wire_mapping import COMPLETION_ID_PREFIX

from .conftest import (
    STREAM_ANSWER,
    STREAM_FRAGMENTS,
    STREAM_TOOL_CALL,
    STREAM_TOOL_INDEX,
    STREAM_USAGE,
    STUB_CONTENT,
    STUB_RESPONSE_MODEL,
    STUB_USAGE,
    TEST_KEYS,
    StubAdapter,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    post_chat,
    stream_events,
)

# The alias the shipped policy resolves, for `evals` under `internal`, onto the
# US `anthropic` provider, and the provider model id it resolves to. The body may
# name only the first; the second is an operator's fact and travels in a header.
ALIAS = "agent-default"
PROVIDER_MODEL = "claude-haiku-5-5"


DONE_FRAME = "data: [DONE]"

# The headers `post_chat` sends, for the one case that has to drive the request
# itself: `TestClient.stream` is the only way to abandon a response part way.
STREAM_HEADERS = auth_headers(key=TEST_KEYS["evals"], tenant="evals", data_class="internal")


def stream_body(**overrides: object) -> dict[str, Any]:
    """The baseline request body with `stream: true`."""
    return chat_body(stream=True, **overrides)


def frame_lines(response: Response) -> list[str]:
    """Every non-empty line of a streamed 200, with the media type asserted.

    A `text/plain` or `application/json` answer would be read as frames by this
    helper and pass several of the assertions below, so the content type is
    checked here once rather than in each case that parses a body.
    """
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";")[0] == "text/event-stream"
    return [line for line in response.text.splitlines() if line.strip()]


def data_frames(response: Response) -> list[str]:
    """The payload of every `data:` frame, in order, the `[DONE]` sentinel included."""
    return [
        line.removeprefix("data:").strip()
        for line in frame_lines(response)
        if line.startswith("data:")
    ]


def chunks(response: Response) -> list[dict[str, Any]]:
    """Every frame parsed as JSON, without the sentinel."""
    parsed: list[dict[str, Any]] = []
    for payload in data_frames(response):
        if payload == "[DONE]":
            continue
        parsed.append(json.loads(payload))
    return parsed


def deltas(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `delta` object of every chunk that carries one choice."""
    return [chunk["choices"][0]["delta"] for chunk in parsed if chunk["choices"]]


# --- the protocol the stub stands in for -------------------------------------


def test_stub_adapter_satisfies_the_protocol() -> None:
    """Catches a stub that has `chat` but not the `stream_chat` the protocol grew in this
    plan: every streaming case in this file would then pass against a contract no shipped
    adapter implements, and the first real adapter would fail at runtime."""
    stub = StubAdapter(name="anthropic")

    assert isinstance(stub, Adapter)
    assert hasattr(stub, "stream_chat")
    assert callable(stub.stream_chat)


# --- the frames --------------------------------------------------------------


def test_a_stream_ends_with_done(chat_client: TestClient) -> None:
    """Catches a stream that stops after its last chunk: an OpenAI client reads until the
    sentinel, so a missing `data: [DONE]` leaves it waiting on a connection that will
    never say another word. Also catches a sentinel spelled `data:[DONE]` or `[DONE]`,
    which the SDKs do not recognise."""
    response = post_chat(chat_client, stream_body())

    lines = frame_lines(response)
    assert lines[-1] == DONE_FRAME
    assert lines.count(DONE_FRAME) == 1
    assert DONE_SENTINEL == "[DONE]"
    assert SSE_MEDIA_TYPE == "text/event-stream"


def test_every_chunk_carries_the_alias_and_one_id(chat_client: TestClient) -> None:
    """Catches a stream naming the provider model id, which hands back a name that cannot
    be sent again and discloses which provider served the request, and catches a new id
    minted per chunk, which the ledger and the trace store would file as one row per
    fragment of a single answer."""
    response = post_chat(chat_client, stream_body(model=ALIAS))

    parsed = chunks(response)
    assert len(parsed) > 1, "a single chunk cannot prove the id is shared"
    assert {chunk["model"] for chunk in parsed} == {ALIAS}
    assert {chunk["object"] for chunk in parsed} == {"chat.completion.chunk"}
    assert CHUNK_OBJECT == "chat.completion.chunk"

    ids = {chunk["id"] for chunk in parsed}
    assert len(ids) == 1
    assert ids.pop().startswith(COMPLETION_ID_PREFIX)
    assert PROVIDER_MODEL not in response.text
    assert STUB_RESPONSE_MODEL not in response.text


def test_first_chunk_carries_the_assistant_role(chat_client: TestClient) -> None:
    """Catches a route that puts `role` on every chunk or on none: a client builds the
    message from the first delta's role, so an absent one leaves it with a message of no
    role, and a repeated one is re-read as a second message."""
    response = post_chat(chat_client, stream_body())

    seen = deltas(chunks(response))
    assert seen[0]["role"] == "assistant"
    assert [delta.get("role") for delta in seen[1:]] == [None] * (len(seen) - 1)


def test_text_deltas_arrive_in_order(chat_client: TestClient) -> None:
    """Catches a route that reorders or coalesces the provider's fragments: the answer a
    client assembles is the concatenation of these deltas in this order, so a set, a dict
    or a reversed buffer renders words the model never wrote."""
    response = post_chat(chat_client, stream_body())

    content = [delta["content"] for delta in deltas(chunks(response)) if delta.get("content")]
    assert content == list(STREAM_FRAGMENTS)
    assert "".join(content) == STREAM_ANSWER
    assert STUB_CONTENT not in response.text, "the non-streaming answer leaked into the stream"


def test_usage_chunk_is_last_and_carries_the_fixture_numbers_exactly(
    chat_client: TestClient,
) -> None:
    """Catches a stream that zero-fills usage, that recomputes the total from its own
    fragment count, or that omits the usage chunk entirely: the ledger prices from these
    three numbers, so any of the three books a real, billed call wrongly. The scripted
    numbers are not the non-streaming stub's, so a usage copied from that path fails
    here too."""
    parsed = chunks(post_chat(chat_client, stream_body()))

    last = parsed[-1]
    assert last["choices"] == []
    assert last["usage"] == {
        "prompt_tokens": 431,
        "completion_tokens": 52,
        "total_tokens": 483,
    }
    assert STREAM_USAGE.prompt_tokens == 431
    assert STREAM_USAGE.completion_tokens == 52
    assert STREAM_USAGE.total_tokens == 483
    assert STUB_USAGE.total_tokens != STREAM_USAGE.total_tokens
    assert [chunk.get("usage") for chunk in parsed[:-1]] == [None] * (len(parsed) - 1)


def test_streamed_tool_call_is_one_whole_delta(settings: Settings) -> None:
    """Catches a route that drops the streamed tool call, that renders the neutral
    `ToolCall` flat (an OpenAI client reads the name from `function.name` and would
    dispatch nothing), or that hardcodes `index` to 0, which merges two calls of one
    answer into one when a second arrives."""
    stub = StubAdapter(
        name="anthropic",
        events=stream_events(
            fragments=(),
            tool_calls=(ToolCallDelta(index=STREAM_TOOL_INDEX, call=STREAM_TOOL_CALL),),
            finish_reason="tool_use",
        ),
    )

    with build_chat_client(settings, {"anthropic": stub}) as client:
        response = post_chat(client, stream_body())
        parsed = chunks(response)

    seen = deltas(parsed)
    assert seen[0]["role"] == "assistant", (
        "a tool-only answer still opens with the role; a role emitted lazily on the first "
        "text delta would leave this stream with none"
    )
    assert seen[0].get("tool_calls") is None, "the opening frame carries the role and nothing else"

    calls = [call for delta in deltas(parsed) for call in (delta.get("tool_calls") or [])]
    assert calls == [
        {
            "index": STREAM_TOOL_INDEX,
            "id": STREAM_TOOL_CALL.id,
            "type": "function",
            "function": {
                "name": STREAM_TOOL_CALL.name,
                "arguments": STREAM_TOOL_CALL.arguments,
            },
        }
    ]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Berlin", "units": "metric"}
    assert STREAM_TOOL_INDEX != 0


# The same table the non-streaming path is pinned against, unknown row included
#. A client branches on this field to decide whether to run a
# tool or to report a truncated answer.
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
def test_finish_reason_table_on_the_stream(
    case: str, provider_reason: str, expected: str, settings: Settings
) -> None:
    """Catches a streaming branch that reimplements the mapping instead of reusing
    `FINISH_REASON_BY_STOP_REASON`: a client streaming a tool call would never run it, and
    one streaming a truncated answer would never notice. The unknown row catches the
    opposite mistake, a mapping that collapses what it does not recognise into "stop"."""
    stub = StubAdapter(name="anthropic", events=stream_events(finish_reason=provider_reason))

    with build_chat_client(settings, {"anthropic": stub}) as client:
        parsed = chunks(post_chat(client, stream_body()))

    finished = [
        chunk["choices"][0]["finish_reason"]
        for chunk in parsed
        if chunk["choices"] and chunk["choices"][0].get("finish_reason") is not None
    ]
    assert finished == [expected], case


# --- the headers -------------------------------------------------------------

EXPECTED_GATEWAY_HEADERS = {
    "x-gateway-provider": "anthropic",
    "x-gateway-model": PROVIDER_MODEL,
    "x-gateway-cache": "miss",
    "x-gateway-redacted": "false",
}


def test_headers_on_a_streamed_success(chat_client: TestClient) -> None:
    """Catches a `StreamingResponse` built without the header set the non-streaming path
    carries, which would leave an operator unable to say which provider served
    a streamed request; catches a missing `Cache-Control: no-cache` or
    `X-Accel-Buffering: no`, either of which lets a proxy buffer the whole answer and
    deliver it as one block; and catches a cost header on a stream, which is not a
    number the gateway is withholding but one that does not exist yet: these headers
    are on the wire before the first frame, and the usage a cost is priced from arrives
    with the last one. Five here and six on a non-streamed answer is the
    permanent shape, not a stage of it."""
    response = post_chat(chat_client, stream_body(model=ALIAS))

    assert response.status_code == 200, response.text
    headers = {name.lower(): value for name, value in response.headers.items()}
    for name, value in EXPECTED_GATEWAY_HEADERS.items():
        assert headers[name] == value
    assert len(headers["x-gateway-trace-id"]) == 32
    assert len([name for name in headers if name.startswith("x-gateway-")]) == 5
    assert "x-gateway-cost-usd" not in headers

    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"
    assert dict(SSE_HEADERS) == {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    assert headers["content-type"].split(";")[0] == "text/event-stream"


# --- the failures ------------------------------------------------------------


def test_a_provider_failure_before_the_first_event_is_502_with_no_stream(
    settings: Settings,
) -> None:
    """Catches a route that returns the `StreamingResponse` before pulling the first
    event: the status would already be 200 by the time the adapter failed, so a caller
    whose request never reached the provider would be told it succeeded and handed an
    error frame its SDK reports as a parse failure."""
    stub = StubAdapter(
        name="anthropic",
        stream_error=ProviderError("anthropic: the provider answered 503", retryable=True),
        stream_error_at=0,
    )

    with build_chat_client(settings, {"anthropic": stub}) as client:
        response = post_chat(client, stream_body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert response.headers["content-type"].split(";")[0] == "application/json"
    assert DONE_FRAME not in response.text


def test_a_stream_that_opens_and_yields_nothing_is_502(settings: Settings) -> None:
    """Catches a route that calls `next()` on an empty stream and lets `StopIteration`
    escape: the caller gets a 500 with `code: null` from the last-resort handler instead
    of the typed 502 its retry logic reads, and inside a generator that exception is the
    one Python turns into a `RuntimeError`."""
    stub = StubAdapter(name="anthropic", events=())

    with build_chat_client(settings, {"anthropic": stub}) as client:
        response = post_chat(client, stream_body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert response.headers["content-type"].split(";")[0] == "application/json"


ROUTING_FAILURE_CASES: list[tuple[str, str, str, str, int, str]] = [
    ("an unknown alias", "no-such-alias", TEST_KEYS["evals"], "evals", 404, "ALIAS_UNKNOWN"),
    (
        "no adapter for any candidate",
        "embed",
        TEST_KEYS["evals"],
        "evals",
        503,
        "ALL_PROVIDERS_FAILED",
    ),
]


@pytest.mark.parametrize(
    ("case", "alias", "key", "tenant", "status", "code"),
    ROUTING_FAILURE_CASES,
    ids=[case for case, *_ in ROUTING_FAILURE_CASES],
)
def test_a_routing_failure_on_a_stream_is_the_ordinary_typed_error(
    case: str,
    alias: str,
    key: str,
    tenant: str,
    status: int,
    code: str,
    chat_client: TestClient,
) -> None:
    """Catches a streaming branch taken before the resolver runs, which would answer 200
    and an error frame to a request that never had a provider to reach."""
    response = post_chat(chat_client, stream_body(model=alias), key=key, tenant=tenant)

    assert_error_body(response, status=status, code=code)
    assert response.headers["content-type"].split(";")[0] == "application/json", case


def test_a_failure_after_the_first_event_emits_an_error_chunk_and_no_done(
    settings: Settings,
) -> None:
    """Catches a stream that ends silently on a mid-stream failure, which a client cannot
    tell from a complete answer, and catches one that still sends `data: [DONE]` after the
    error frame, which tells the client the truncated answer was finished. The status is
    already 200 and cannot be changed, so the frame and the missing sentinel are the whole
    report (decision E)."""
    stub = StubAdapter(
        name="anthropic",
        stream_error=ProviderError("anthropic: the connection dropped", retryable=True),
        stream_error_at=2,
    )

    with build_chat_client(settings, {"anthropic": stub}) as client:
        response = post_chat(client, stream_body())

    lines = frame_lines(response)
    assert DONE_FRAME not in lines
    payloads = data_frames(response)
    error_frames = [json.loads(payload) for payload in payloads if "error" in json.loads(payload)]
    assert len(error_frames) == 1
    error = error_frames[0]["error"]
    assert set(error) == {"message", "type", "code", "param"}
    assert error["code"] == "PROVIDER_ERROR"
    assert json.loads(payloads[-1]) == error_frames[0], "the error frame is the last one"
    assert len(payloads) > 1, "the error frame must follow the events already delivered"


def test_every_frame_ends_with_a_blank_line(chat_client: TestClient) -> None:
    """Catches a `_frame` simplified to a single newline. Every other case in this file
    reads the response through `frame_lines`, which drops blank lines and so cannot see
    the separator at all - so without this the terminator could be deleted with the whole
    suite staying green, while a real SDK client received one unterminated event and
    waited on it until its read timeout."""
    response = post_chat(chat_client, stream_body())
    body = response.text

    assert body.endswith(f"data: {DONE_SENTINEL}\n\n")
    assert "\n\n" in body
    frames = [frame for frame in body.split("\n\n") if frame]
    assert len(frames) == len(data_frames(response))
    assert all(frame.startswith("data: ") for frame in frames)
    assert "\n\n\n" not in body, "a frame must be separated by exactly one blank line"


def test_an_untyped_failure_mid_stream_is_not_rendered_as_a_frame(settings: Settings) -> None:
    """Pins the deliberate limit of decision E rather than a behaviour. An untyped
    exception is a bug, there is nothing safe to say about one in a frame, and nothing in
    this gateway turns it into a body: it escapes the generator and the ASGI server logs
    it, which reaches a real client as a cut stream with no sentinel. Catches a bare
    `except Exception` added beside the `except GatewayError`, which would report a bug to
    the client as a provider failure carrying a code the closed `ErrorCode` set does not
    contain.

    `TestClient` re-raises what escapes rather than cutting a socket, so the escape itself
    is what this asserts; the cut connection is Starlette's documented behaviour once a
    response has started, not something this suite can observe."""
    stub = StubAdapter(
        name="anthropic",
        stream_error=RuntimeError("a bug in the adapter"),
        stream_error_at=2,
    )

    with (
        build_chat_client(settings, {"anthropic": stub}) as client,
        pytest.raises(RuntimeError, match="a bug in the adapter"),
    ):
        post_chat(client, stream_body())


def test_closing_the_frame_generator_releases_the_provider_stream() -> None:
    """Pins `sse_frames`' `finally` against the thing that otherwise fakes it. When a
    client disconnects, Starlette drops the suspended generator and CPython's refcounting
    closes the adapter's generator a moment later anyway - so an end-to-end disconnect
    test passes with the `finally` deleted and proves nothing. This test keeps its own
    reference to `events` alive across the close, so refcounting cannot stand in: only the
    explicit `events.close()` can set the flag.

    Deleting the `finally` block fails this test. Without it, a later refactor that keeps
    any reference to the generator - a span, a traceback, a wrapper - leaves the streamed
    `httpx` response checked out of the pool, and a browser reloading a slow answer
    exhausts `max_connections` until every request to that provider blocks."""
    stub = StubAdapter(name="anthropic")
    events = stub.stream_chat(
        ChatRequest(model=PROVIDER_MODEL, messages=(ChatMessage(role="user", content="Hi"),))
    )
    first = next(events)
    frames: Generator[str, None, None] = sse_frames(
        first, events, ALIAS, "chatcmpl-000000000000000000000000"
    )

    assert next(frames).startswith("data: ")
    assert not stub.stream_closed, "nothing is released while the stream is still running"

    frames.close()

    assert stub.stream_closed, "the provider stream was never released"
    assert events is not None, "the reference this case depends on must outlive the close"


# --- stream_options ----------------------------------------------------------


def test_include_usage_false_is_422(chat_client: TestClient) -> None:
    """Catches a route that accepts `include_usage: false` and then streams usage anyway,
    which is a promise silently broken, or one that honours it, which breaks the design's
    guarantee that every stream reports what it consumed (decision D)."""
    response = post_chat(chat_client, stream_body(stream_options={"include_usage": False}))

    assert_error_body(response, status=422, code="VALIDATION")


def test_include_usage_true_is_accepted(chat_client: TestClient) -> None:
    """The other half, so the row above cannot be passed by refusing `stream_options`
    outright: catches a route that rejects the field whatever its value, which breaks the
    OpenAI SDK, since it sends `stream_options` on every streamed request it makes."""
    response = post_chat(chat_client, stream_body(stream_options={"include_usage": True}))

    parsed = chunks(response)
    assert parsed[-1]["usage"]["total_tokens"] == 483
    assert frame_lines(response)[-1] == DONE_FRAME
