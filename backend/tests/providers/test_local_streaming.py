"""The streaming half of the OpenAI-compatible adapter: the SSE reader and the open.

Two halves, kept apart the way the mapping is kept apart from the HTTP.
`from_openai_events` is pure - a function over the decoded lines
`httpx.Response.iter_lines()` yields - so every reader case runs against a `.sse`
file with no transport at all. `stream_chat` adds the request, the 120s read
timeout and the retried open, and those cases run through `httpx.MockTransport`:
no socket is opened and no provider key is read.

Every `.sse` file here is HAND-WRITTEN to the published OpenAI streaming wire
shape and says so in its own leading comment lines: no llama.cpp server was
available when these tests were written, which is a deliberate departure from
the usual recorded fixtures.

The token counts are asserted as exact literals and are
distinct per fixture, so a reader that fills usage from the wrong chunk is
visible rather than accidentally right.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
import pytest

from app.core.errors import ProviderError
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCallDelta,
    UsageTotal,
)
from app.providers.openai_compatible import (
    CHAT_COMPLETIONS_PATH,
    OpenAICompatibleAdapter,
    from_openai_events,
)
from app.providers.transport import CONNECT_TIMEOUT, READ_TIMEOUT, STREAM_READ_TIMEOUT

from .conftest import (
    LOCAL_BASE_URL,
    LOCAL_FIXTURE_DIR,
    LOCAL_PROVIDER_NAME,
    FixedJitter,
    SleepRecorder,
    local_ok,
    make_local_adapter,
)

REQUEST = ChatRequest(
    model="qwen2.5-7b-instruct",
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)

# --- what the hand-written streams contain, as exact literals ----------------

TEXT_FRAGMENTS = ("Berlin", " is the capital", " of Germany.")
TEXT_ANSWER = "Berlin is the capital of Germany."
TEXT_PROMPT_TOKENS = 613
TEXT_COMPLETION_TOKENS = 57
TEXT_TOTAL_TOKENS = 670

TOOL_TEXT = "Checking the weather."
TOOL_CALL_ID = "call_local_stream_0001"
TOOL_CALL_NAME = "get_weather"
TOOL_CALL_ARGUMENTS = '{"city": "Berlin", "units": "metric"}'
TOOL_PROMPT_TOKENS = 419
TOOL_COMPLETION_TOKENS = 74
TOOL_TOTAL_TOKENS = 493

TWO_TOOL_PROMPT_TOKENS = 528
TWO_TOOL_COMPLETION_TOKENS = 91
TWO_TOOL_TOTAL_TOKENS = 619

# The whole event sequence, as types, so an extra or a missing event is visible
# rather than only a wrong value inside one of them.
TEXT_EVENT_TYPES: list[type[object]] = [
    StreamStart,
    TextDelta,
    TextDelta,
    TextDelta,
    FinishReason,
    UsageTotal,
]
TRUNCATED_EVENT_TYPES: list[type[object]] = [StreamStart, TextDelta, TextDelta]
TWO_TOOL_EVENT_TYPES: list[type[object]] = [
    StreamStart,
    ToolCallDelta,
    ToolCallDelta,
    FinishReason,
    UsageTotal,
]


def sse_lines(name: str) -> list[str]:
    """One hand-written stream as `iter_lines()` yields it: decoded, without newlines."""
    return (LOCAL_FIXTURE_DIR / f"{name}.sse").read_text(encoding="utf-8").splitlines()


def sse_body(name: str) -> str:
    """One hand-written stream as the bytes a server would put on the wire."""
    return (LOCAL_FIXTURE_DIR / f"{name}.sse").read_text(encoding="utf-8")


# One scripted step: an SSE body served as a 200, a (status, body) pair for a
# failure, or a transport-level exception class to raise instead.
SseStep = str | tuple[int, Mapping[str, Any]] | type[httpx.RequestError]

SSE_CONTENT_TYPE = "text/event-stream"


class SseTransport:
    """A `MockTransport` that answers with SSE bodies and records every request.

    The last step repeats once the script runs out, so a retry loop that makes
    more attempts than allowed gets a plausible answer rather than an
    IndexError; the exact request counts are what catch it.
    """

    def __init__(self, steps: Sequence[SseStep]) -> None:
        assert steps, "a scripted transport needs at least one step"
        self._steps = list(steps)
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self._steps[min(len(self.requests) - 1, len(self._steps) - 1)]
        if isinstance(step, type):
            raise step(f"scripted {step.__name__}", request=request)
        if isinstance(step, str):
            return httpx.Response(
                200,
                content=step.encode("utf-8"),
                headers={"content-type": SSE_CONTENT_TYPE},
                request=request,
            )
        status, body = step
        return httpx.Response(status, json=dict(body), request=request)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def body(self, index: int = 0) -> dict[str, Any]:
        """The JSON the adapter sent on the request at `index`."""
        sent: Any = json.loads(self.requests[index].content.decode("utf-8"))
        assert isinstance(sent, dict), "the adapter must send a JSON object"
        return sent


def make_local_sse_adapter(
    steps: Sequence[SseStep],
    *,
    sleep: SleepRecorder | None = None,
    jitter: float = 1.0,
) -> tuple[OpenAICompatibleAdapter, SseTransport]:
    """A local adapter wired to a scripted SSE transport, with no key and no clock."""
    scripted = SseTransport(steps)
    adapter = OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=scripted.transport,
        sleep=sleep if sleep is not None else SleepRecorder(),
        jitter=FixedJitter(jitter),
    )
    return adapter, scripted


# --- the pure reader ---------------------------------------------------------


def test_the_reader_yields_the_start_then_the_text_then_the_finish_and_usage_last() -> None:
    """Catches a reader that emits an event per chunk regardless of content (the empty
    opening delta would arrive as a `TextDelta("")`), one that drops the usage chunk
    because its `choices` is empty, and one that emits usage before the finish reason -
    each of which hands the route a sequence it cannot render in order."""
    events = list(from_openai_events(sse_lines("stream_text")))

    assert isinstance(events[0], StreamStart)
    assert events[0].role == "assistant"

    texts = [event for event in events if isinstance(event, TextDelta)]
    assert tuple(event.text for event in texts) == TEXT_FRAGMENTS
    assert "".join(event.text for event in texts) == TEXT_ANSWER

    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "stop"
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == TEXT_PROMPT_TOKENS
    assert events[-1].usage.completion_tokens == TEXT_COMPLETION_TOKENS
    assert events[-1].usage.total_tokens == TEXT_TOTAL_TOKENS
    assert [type(event) for event in events] == TEXT_EVENT_TYPES


def test_the_reader_stops_at_the_done_sentinel() -> None:
    """Catches a reader that treats `[DONE]` as JSON (a `JSONDecodeError` on every single
    stream) and one that keeps reading past it: the fixture carries a truncated line after
    the sentinel, so a reader that does not stop answers a 502 instead of a finished
    answer."""
    events = list(from_openai_events(sse_lines("stream_after_done")))

    assert [type(event) for event in events] == TEXT_EVENT_TYPES
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.total_tokens == TEXT_TOTAL_TOKENS


def test_a_streamed_tool_call_arrives_as_one_whole_delta() -> None:
    """Catches a reader that forwards each `delta.tool_calls[].function.arguments`
    fragment as its own event: `{"city": "Ber` is not JSON any client can parse, and
    the stream promises one whole call. Also catches a reader that emits the call on its
    opening fragment, whose arguments would be the empty string the server opens with."""
    events = list(from_openai_events(sse_lines("stream_tool_call")))

    calls = [event for event in events if isinstance(event, ToolCallDelta)]
    assert len(calls) == 1
    assert calls[0].index == 0
    assert calls[0].call.id == TOOL_CALL_ID
    assert calls[0].call.name == TOOL_CALL_NAME
    assert calls[0].call.arguments == TOOL_CALL_ARGUMENTS

    texts = [event for event in events if isinstance(event, TextDelta)]
    assert [event.text for event in texts] == [TOOL_TEXT]
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "tool_calls"
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == TOOL_PROMPT_TOKENS
    assert events[-1].usage.completion_tokens == TOOL_COMPLETION_TOKENS
    assert events[-1].usage.total_tokens == TOOL_TOTAL_TOKENS


def test_the_reader_emits_every_buffered_call_before_the_finish_reason() -> None:
    """Catches a reader that flushes a call when the NEXT index opens (a rejected
    heuristic): the fixture interleaves the two calls, so call 0's last fragment
    arrives after call 1 has opened and a flush-on-open reader hands the client
    `{"city": "Ber` as a whole call. Also catches a reader that buffers against whichever
    call opened last, which merges the two argument strings into one."""
    events = list(from_openai_events(sse_lines("stream_two_tools")))

    calls = [event for event in events if isinstance(event, ToolCallDelta)]
    assert [call.index for call in calls] == [0, 1], "buffered calls are emitted in index order"
    assert [call.call.id for call in calls] == ["call_local_first_0001", "call_local_second_0002"]
    assert [call.call.name for call in calls] == ["get_weather", "get_time"]
    assert [call.call.arguments for call in calls] == [
        '{"city": "Berlin"}',
        '{"zone": "Europe/Berlin"}',
    ]

    types = [type(event) for event in events]
    assert types.index(FinishReason) > max(
        position for position, event in enumerate(events) if isinstance(event, ToolCallDelta)
    ), "every call is delivered before the finish reason"
    assert types == TWO_TOOL_EVENT_TYPES
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == TWO_TOOL_PROMPT_TOKENS
    assert events[-1].usage.completion_tokens == TWO_TOOL_COMPLETION_TOKENS
    assert events[-1].usage.total_tokens == TWO_TOOL_TOTAL_TOKENS


def test_a_stream_that_ends_without_usage_raises_rather_than_zero_filling() -> None:
    """Catches a reader that finishes a usage-less stream with `Usage(0, 0, 0)`: the
    ledger prices from these numbers, so a real, billed answer would be booked as free -
    and this is the stream a server sends when `stream_options.include_usage` was left
    off, which is precisely the mistake the outbound-body test exists to prevent."""
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in from_openai_events(sse_lines("stream_no_usage")):
            events.append(event)

    assert raised.value.retryable is False
    assert raised.value.http_status == 502
    assert not [event for event in events if isinstance(event, UsageTotal)]
    assert not [event for event in events if isinstance(event, FinishReason)]
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "Billed but uncounted."
    ]


def test_a_stream_cut_off_mid_answer_raises() -> None:
    """Catches a reader that treats the end of the line iterator as a finished stream: a
    dropped connection has no finish reason and no usage, and completing it would hand the
    client a truncated answer that looks whole and bills at zero."""
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in from_openai_events(sse_lines("stream_truncated")):
            events.append(event)

    assert raised.value.retryable is False
    assert [type(event) for event in events] == TRUNCATED_EVENT_TYPES


def test_a_stream_that_renders_nothing_raises() -> None:
    """Catches the streaming path diverging from `from_openai`, which already refuses an
    answer it can render nothing from: without this guard the same reply is a 502 on the
    non-streaming path and a 200 with an empty answer, a finish reason and a usage chunk
    the ledger prices on the streaming one."""
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in from_openai_events(sse_lines("stream_no_content")):
            events.append(event)

    assert raised.value.retryable is False
    assert raised.value.http_status == 502
    assert [type(event) for event in events] == [StreamStart]


def test_an_unreadable_data_line_is_a_provider_error() -> None:
    """Catches the `except (KeyError, TypeError, ValueError)` around the reader being
    narrowed or missing: a truncated `data:` line would escape as a raw `JSONDecodeError`,
    which is not a `GatewayError`, so the route's error frame is never written and the
    client gets a silently cut stream instead of a typed 502."""
    with pytest.raises(ProviderError) as raised:
        list(from_openai_events(sse_lines("stream_unreadable")))

    assert raised.value.retryable is False
    assert raised.value.http_status == 502
    assert "JSONDecodeError" in (raised.value.provider_message or "")


def test_a_data_line_that_is_not_an_object_is_a_provider_error() -> None:
    """Catches a reader calling `.get("choices")` on whatever JSON a `data:` line holds: a
    list answers with an `AttributeError`, which is neither a `ProviderError` nor a
    `GatewayError`, so it escapes both the adapter's contract and the route's error frame.
    Built here rather than in a fixture because no server sends this - it is what a proxy
    rewriting the body produces."""
    with pytest.raises(ProviderError) as raised:
        list(from_openai_events(['data: ["not", "an", "object"]']))

    assert raised.value.retryable is False
    assert "TypeError" in (raised.value.provider_message or "")


# --- the stream open ---------------------------------------------------------


def test_the_streamed_body_asks_for_usage_as_well_as_the_stream() -> None:
    """The acceptance criterion, asserted on the OUTBOUND body rather than on the parsed
    result: catches an adapter that sends `stream: true` alone, which an OpenAI-compatible
    server answers with no usage chunk at all - every streamed call would then be booked
    at zero, or refused by the reader, depending on the day. Also catches a `stream` flag
    put in the query string or the headers instead of the body."""
    adapter, scripted = make_local_sse_adapter([sse_body("stream_text")])

    list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert sent.url.path == "/v1/chat/completions"
    body = scripted.body()
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["model"] == REQUEST.model


def test_the_non_streaming_body_carries_neither_stream_flag() -> None:
    """Catches an adapter that sets `stream` and `stream_options` in the shared mapping
    instead of on the streamed call: an ordinary completion would come back as SSE the
    non-streaming path cannot parse, so every plain request would 502."""
    adapter, scripted = make_local_adapter([local_ok("plain")])

    adapter.chat(REQUEST)

    body = scripted.body()
    assert "stream" not in body
    assert "stream_options" not in body


def test_stream_chat_yields_the_events_over_the_transport() -> None:
    """Catches a `stream_chat` that reads the response with `.json()` or `.text` instead
    of iterating lines: the whole body would arrive as one unparsable blob, so nothing
    would reach the caller until the server finished - which on a CPU box is minutes."""
    adapter, _ = make_local_sse_adapter([sse_body("stream_text")])

    events = list(adapter.stream_chat(REQUEST))

    assert [type(event) for event in events] == TEXT_EVENT_TYPES
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == TEXT_ANSWER


def test_the_streamed_read_timeout_is_120_and_the_clients_stays_60() -> None:
    """Catches an adapter that streams through the plain `send` path or with the client's
    own 60s: a local CPU model can take longer than that between events, so a long answer
    would be cut off mid-sentence. Catches the opposite fix too - raising the CLIENT's
    timeout would let every ordinary completion hold a worker twice as long."""
    adapter, scripted = make_local_sse_adapter([sse_body("stream_text")])

    list(adapter.stream_chat(REQUEST))

    timeout = scripted.requests[0].extensions["timeout"]
    assert timeout["read"] == 120.0
    assert timeout["connect"] == 5.0
    assert adapter.client.timeout.read == 60.0
    assert STREAM_READ_TIMEOUT == 120.0
    assert READ_TIMEOUT == 60.0
    assert CONNECT_TIMEOUT == 5.0


def test_the_stream_open_is_retried_but_a_started_answer_is_not(sleeps: SleepRecorder) -> None:
    """Catches both halves at once: an open that gives up on the 503 a busy llama.cpp slot
    answers with, and a retry wrapped around the ITERATION, which would reopen the call
    and replay sentences the client has already been shown."""
    adapter, scripted = make_local_sse_adapter(
        [(503, {"error": {"message": "all slots busy"}}), sse_body("stream_text")], sleep=sleeps
    )

    events = list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 2
    assert len(sleeps.calls) == 1, "one retried open means one wait"
    assert isinstance(events[-1], UsageTotal)

    mid_sleeps = SleepRecorder()
    mid_adapter, mid_scripted = make_local_sse_adapter(
        [sse_body("stream_truncated")], sleep=mid_sleeps
    )
    delivered: list[StreamEvent] = []

    with pytest.raises(ProviderError):
        for event in mid_adapter.stream_chat(REQUEST):
            delivered.append(event)

    assert mid_scripted.request_count == 1, "a stream that failed mid-body was reopened"
    assert mid_sleeps.calls == []
    assert [type(event) for event in delivered] == TRUNCATED_EVENT_TYPES


def test_a_failure_status_on_the_open_never_starts_a_stream(sleeps: SleepRecorder) -> None:
    """Catches an open that ignores the status and hands the error body to the SSE reader:
    a 400 would surface as an unreadable stream rather than as the non-retryable
    `ProviderError` the route turns into a typed 502."""
    adapter, scripted = make_local_sse_adapter(
        [(400, {"error": {"message": "bad request"}})], sleep=sleeps
    )
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in adapter.stream_chat(REQUEST):
            events.append(event)

    assert events == []
    assert scripted.request_count == 1, "a 400 is not retried"
    assert raised.value.retryable is False


def test_the_local_server_being_down_never_starts_a_stream(sleeps: SleepRecorder) -> None:
    """Catches a raw `httpx.ConnectError` escaping `stream_chat` when the model container
    is not running: the route would answer 500 with a traceback instead of the 502 a
    client's retry logic reads, and it must not be retried, because a box that is down is
    down."""
    adapter, scripted = make_local_sse_adapter([httpx.ConnectError], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 1
    assert sleeps.calls == []
    assert raised.value.retryable is False
    assert raised.value.http_status == 502


def test_the_hand_written_fixtures_are_the_shape_this_module_claims() -> None:
    """Catches a fixture edited into agreement with a wrong reader: the counts, the
    fragmentation and the missing chunks are asserted against the text itself, so a
    `stream_no_usage` that grew a usage chunk fails here rather than quietly making the
    zero-fill test vacuous."""
    no_usage = sse_body("stream_no_usage")
    assert '"usage"' not in no_usage, "the usage-less stream must carry no usage chunk"
    assert '"finish_reason":"stop"' in no_usage, "the guard must fire on a FINISHED stream"
    assert "data: [DONE]" in no_usage

    no_content = sse_body("stream_no_content")
    assert '"content":"' not in no_content.replace('"content":""', "")
    assert "tool_calls" not in no_content, "a tool call would make the no-content guard vacuous"
    assert '"total_tokens":64' in no_content

    truncated = sse_body("stream_truncated")
    assert 'finish_reason":"' not in truncated
    assert '"usage"' not in truncated
    assert "data: [DONE]" not in truncated, "a cut connection sends no sentinel"

    text = sse_body("stream_text")
    assert f'"prompt_tokens":{TEXT_PROMPT_TOKENS}' in text
    assert f'"completion_tokens":{TEXT_COMPLETION_TOKENS}' in text
    assert '"choices":[],"usage"' in text, "usage arrives on a chunk with no choices"

    after_done = sse_body("stream_after_done")
    assert after_done.index("data: [DONE]") < after_done.index('{"truncated'), (
        "the poison line must sit AFTER the sentinel, or the test proves nothing"
    )

    tool = sse_body("stream_tool_call")
    assert TOOL_CALL_ARGUMENTS not in tool, "the arguments must arrive as fragments"
    assert f'"prompt_tokens":{TOOL_PROMPT_TOKENS}' in tool

    two = sse_body("stream_two_tools")
    assert two.count('"type":"function"') == 2
    assert '{"city": "Berlin"}' not in two, "the arguments must arrive as fragments"
    first_open = two.index("call_local_first_0001")
    second_open = two.index("call_local_second_0002")
    last_first_fragment = two.index('lin\\"}')
    assert first_open < second_open < last_first_fragment, (
        "the two calls must INTERLEAVE, or a flush-on-next-open reader passes"
    )

    assert CHAT_COMPLETIONS_PATH == "/chat/completions"


def test_a_transport_failure_mid_stream_is_a_provider_error_and_is_not_retried(
    sleeps: SleepRecorder,
) -> None:
    """Catches a raw `httpx.ReadTimeout` escaping `stream_chat` once the body has started,
    which is this provider's most likely failure of all - a CPU box going quiet for longer
    than the 120s between-events timeout part way through an answer. The exception is not
    a `GatewayError`, so the route writes no error frame and the client sees a silently cut
    stream instead of a typed 502. Catches a retry wrapped around the ITERATION too, which
    would reopen the call and replay sentences the client has already been shown."""

    def cut_off() -> Iterator[bytes]:
        yield sse_body("stream_truncated").encode("utf-8")
        raise httpx.ReadTimeout("scripted mid-stream cut", request=None)

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, content=cut_off(), headers={"content-type": SSE_CONTENT_TYPE}, request=request
        )

    adapter = OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=httpx.MockTransport(handle),
        sleep=sleeps,
        jitter=FixedJitter(1.0),
    )
    delivered: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in adapter.stream_chat(REQUEST):
            delivered.append(event)

    assert [type(event) for event in delivered] == TRUNCATED_EVENT_TYPES
    assert len(requests) == 1, "a stream that failed mid-body was reopened"
    assert sleeps.calls == []
    assert raised.value.retryable is True, "a read timeout is the one retryable transport failure"
    assert raised.value.http_status == 502
