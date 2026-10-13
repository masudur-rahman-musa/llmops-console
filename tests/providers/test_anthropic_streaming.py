"""The streaming half of the Anthropic adapter: the SSE reader and the stream open.

Two halves, kept apart the way an earlier change kept `from_anthropic` apart from the HTTP.
`from_anthropic_events` is pure - a function over the decoded lines
`httpx.Response.iter_lines()` yields - so every reader case runs against a
recorded `.sse` file with no transport at all. `stream_chat` adds the request,
the 120s read timeout and the retried open, and those cases run through
`httpx.MockTransport`: no socket is opened and no provider key is read (#66).

The recorded fixtures carry deliberately distinct, non-round token counts, and
they are asserted here as exact literals. `message_start` in
each of them reports a small `output_tokens` of its own, which Anthropic really
does send: a reader that takes the completion count from the wrong event books
the wrong number, and the ledger prices from it.

The SSE transport helper is local to this module rather than in
`tests/providers/conftest.py` because it is the only thing in the suite that
needs a streaming body; the scripted JSON transport there answers everything
else.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
import pytest

from opsconsole.core.errors import ProviderError
from opsconsole.providers.anthropic import (
    CONNECT_TIMEOUT,
    MAX_RETRIES,
    READ_TIMEOUT,
    STREAM_READ_TIMEOUT,
    AnthropicAdapter,
    from_anthropic_events,
)
from opsconsole.providers.base import (
    ChatMessage,
    ChatRequest,
    FinishReason,
    ResponseFormat,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCallDelta,
    ToolChoice,
    ToolSpec,
    UsageTotal,
)

from .conftest import (
    FIXTURE_DIR,
    TEST_API_KEY,
    TEST_PROVIDER_NAME,
    FixedJitter,
    SleepRecorder,
)

REQUEST = ChatRequest(
    model="claude-haiku-4-5",
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)

# --- what the recorded streams contain, as exact literals --------------------

TEXT_FRAGMENTS = ("Berlin", " is the capital", " of Germany.")
TEXT_ANSWER = "Berlin is the capital of Germany."
TEXT_PROMPT_TOKENS = 613
TEXT_COMPLETION_TOKENS = 57
TEXT_TOTAL_TOKENS = 670

TOOL_TEXT = "Checking the weather."
TOOL_CALL_ID = "toolu_01StreamWeatherCall0"
TOOL_CALL_NAME = "get_weather"
TOOL_CALL_ARGUMENTS = '{"city": "Berlin", "units": "metric"}'
TOOL_BLOCK_INDEX = 1
TOOL_PROMPT_TOKENS = 419
TOOL_COMPLETION_TOKENS = 74
TOOL_TOTAL_TOKENS = 493

# The whole event sequence, as types, so an extra or a missing event is visible
# rather than only a wrong value inside one of them. Annotated because the
# comparison's left side is a list of the union's base class.
TEXT_EVENT_TYPES: list[type[object]] = [
    StreamStart,
    TextDelta,
    TextDelta,
    TextDelta,
    FinishReason,
    UsageTotal,
]
TRUNCATED_EVENT_TYPES: list[type[object]] = [StreamStart, TextDelta, TextDelta]


def sse_lines(name: str) -> list[str]:
    """One recorded stream as `iter_lines()` yields it: decoded, without newlines."""
    path = FIXTURE_DIR / f"{name}.sse"
    return path.read_text(encoding="utf-8").splitlines()


def sse_body(name: str) -> str:
    """One recorded stream as the bytes a provider would put on the wire."""
    return (FIXTURE_DIR / f"{name}.sse").read_text(encoding="utf-8")


# One scripted step: a recorded SSE body served as a 200, a (status, body) pair
# for a failure, or a transport-level exception class to raise instead.
SseStep = str | tuple[int, Mapping[str, Any]] | type[httpx.RequestError]

SSE_CONTENT_TYPE = "text/event-stream"


class SseTransport:
    """A `MockTransport` that answers with recorded SSE bodies and records every request.

    The last step repeats once the script runs out, so a retry loop that makes
    more attempts than the plan allows gets a plausible answer rather than an
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


def make_sse_adapter(
    steps: Sequence[SseStep],
    *,
    sleep: SleepRecorder | None = None,
    jitter: float = 1.0,
) -> tuple[AnthropicAdapter, SseTransport]:
    """An adapter wired to a scripted SSE transport, with an explicit key and no clock."""
    scripted = SseTransport(steps)
    adapter = AnthropicAdapter(
        name=TEST_PROVIDER_NAME,
        api_key=TEST_API_KEY,
        base_url="https://api.anthropic.test",
        transport=scripted.transport,
        sleep=sleep if sleep is not None else SleepRecorder(),
        jitter=FixedJitter(jitter),
    )
    return adapter, scripted


# --- the pure reader ---------------------------------------------------------


def test_the_reader_yields_the_start_then_the_text_then_the_finish_and_usage_last() -> None:
    """Catches a reader that emits Anthropic's own event names, drops the `ping` or the
    `content_block_start` into the answer text, or emits usage before the stream is
    finished - each of which hands the route a sequence it cannot render in order."""
    events = list(from_anthropic_events(sse_lines("stream_text")))

    assert isinstance(events[0], StreamStart)
    assert events[0].role == "assistant"

    texts = [event for event in events if isinstance(event, TextDelta)]
    assert tuple(event.text for event in texts) == TEXT_FRAGMENTS
    assert "".join(event.text for event in texts) == TEXT_ANSWER

    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "end_turn"
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == TEXT_PROMPT_TOKENS
    assert events[-1].usage.completion_tokens == TEXT_COMPLETION_TOKENS
    assert events[-1].usage.total_tokens == TEXT_TOTAL_TOKENS
    assert [type(event) for event in events] == TEXT_EVENT_TYPES


def test_the_reader_assembles_a_tool_use_block_from_its_json_fragments() -> None:
    """Catches a reader that forwards Anthropic's `input_json_delta` fragments one by one:
    `{"city": "Ber` is not JSON any client can parse, and decision B promises one whole
    call. Also catches a reader that emits the block when it OPENS, whose arguments would
    be the empty `input` object Anthropic starts with."""
    events = list(from_anthropic_events(sse_lines("stream_tool_use")))

    calls = [event for event in events if isinstance(event, ToolCallDelta)]
    assert len(calls) == 1
    assert calls[0].index == TOOL_BLOCK_INDEX
    assert calls[0].call.id == TOOL_CALL_ID
    assert calls[0].call.name == TOOL_CALL_NAME
    assert calls[0].call.arguments == TOOL_CALL_ARGUMENTS

    texts = [event for event in events if isinstance(event, TextDelta)]
    assert [event.text for event in texts] == [TOOL_TEXT]
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == TOOL_PROMPT_TOKENS
    assert events[-1].usage.completion_tokens == TOOL_COMPLETION_TOKENS
    assert events[-1].usage.total_tokens == TOOL_TOTAL_TOKENS
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "tool_use"


def test_a_truncated_stream_raises_rather_than_zero_filling_usage() -> None:
    """Catches a reader that ends a cut-off stream with `Usage(0, 0, 0)` or with the
    `output_tokens: 1` of `message_start`: the ledger prices from these numbers, so a
    connection dropped mid-answer would book a real, billed call as free."""
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in from_anthropic_events(sse_lines("stream_truncated")):
            events.append(event)

    assert raised.value.retryable is False
    assert raised.value.http_status == 502
    assert not [event for event in events if isinstance(event, UsageTotal)], (
        "a truncated stream produced a usage total"
    )
    assert not [event for event in events if isinstance(event, FinishReason)]
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "The connection",
        " dropped before",
    ]


def test_a_stream_with_no_readable_block_raises() -> None:
    """Catches the streaming reader diverging from `from_anthropic`, which already refuses
    an answer it could render nothing from. Anthropic's extended thinking sends only
    `thinking_delta` blocks; without this guard the same reply is a 502 on the
    non-streaming path and a 200 with an empty answer, a finish reason and a usage chunk
    the ledger prices on the streaming one."""
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in from_anthropic_events(sse_lines("stream_thinking_only")):
            events.append(event)

    assert raised.value.retryable is False
    assert raised.value.http_status == 502
    assert [type(event) for event in events] == [StreamStart]
    assert not [event for event in events if isinstance(event, UsageTotal)]
    assert "thinking" in (raised.value.provider_message or "")


def test_two_tool_calls_keep_their_own_indices_and_arguments() -> None:
    """Catches a reader that buffers `input_json_delta` fragments against whichever tool
    block opened last rather than against the block the fragment names: the two calls'
    JSON runs together into one `arguments` string, and a client dispatches one malformed
    call instead of two. Also catches a reader that emits one `ToolCallDelta` per answer."""
    events = list(from_anthropic_events(sse_lines("stream_two_tools")))

    calls = [event for event in events if isinstance(event, ToolCallDelta)]
    assert [call.index for call in calls] == [0, 1]
    assert [call.call.id for call in calls] == [
        "toolu_01StreamFirstCall000",
        "toolu_01StreamSecondCall00",
    ]
    assert [call.call.name for call in calls] == ["get_weather", "get_time"]
    assert [call.call.arguments for call in calls] == [
        '{"city": "Berlin"}',
        '{"zone": "Europe/Berlin"}',
    ]
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == 528
    assert events[-1].usage.completion_tokens == 91
    assert events[-1].usage.total_tokens == 619


def test_interleaved_tool_fragments_stay_with_the_block_that_names_them() -> None:
    """Pins BOTH `payload["index"] == tool_index` checks, which the sequential
    `stream_two_tools.sse` cannot: there each block opens after the previous one closed,
    so the guard never decides anything and removing it changes no output. Here block 0's
    fragment arrives AFTER block 1 has opened. Without the check it is appended to block
    1's buffer, and the client is handed `{"zone": "Europe/Berlin"}lin"}` as one call's
    arguments - a malformed tool call dispatched in place of two valid ones.

    Hand-built rather than recorded, like the not-an-object case below: it is what a proxy
    reordering block deltas produces, not a stream Anthropic is known to send."""
    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":90,"output_tokens":1}}}',
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"tool_use","id":"toolu_01First","name":"get_weather","input":{}}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"input_json_delta","partial_json":"{\\"city\\": \\"Ber"}}',
        'data: {"type":"content_block_start","index":1,"content_block":'
        '{"type":"tool_use","id":"toolu_01Second","name":"get_time","input":{}}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"input_json_delta","partial_json":"lin\\"}"}}',
        'data: {"type":"content_block_delta","index":1,"delta":'
        '{"type":"input_json_delta","partial_json":"{\\"zone\\": \\"UTC\\"}"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":44}}',
    ]

    calls = [event for event in from_anthropic_events(lines) if isinstance(event, ToolCallDelta)]

    assert [call.call.id for call in calls] == ["toolu_01Second", "toolu_01First"], (
        "each call is emitted when ITS block closes, in that order"
    )
    assert [call.index for call in calls] == [1, 0]
    assert {call.call.id: call.call.arguments for call in calls} == {
        "toolu_01First": '{"city": "Berlin"}',
        "toolu_01Second": '{"zone": "UTC"}',
    }


def test_a_stream_cut_off_before_any_content_reports_the_truncation() -> None:
    """Catches the two end-of-stream guards being ordered the other way round. A
    connection dropped after `content_block_start` but before the first `text_delta` has
    rendered nothing AND has no stop reason; reporting it as an empty answer sends an
    operator hunting a content bug instead of the cut connection that actually happened.
    Both are a non-retryable 502, so `provider_message` is the only thing that says
    which."""
    lines = [
        line for line in sse_lines("stream_truncated") if '"type":"content_block_delta"' not in line
    ]

    with pytest.raises(ProviderError) as raised:
        list(from_anthropic_events(lines))

    message = raised.value.provider_message or ""
    assert "input_tokens=251" in message, "the truncation guard names what was missing"
    assert "no readable content block" not in raised.value.message
    assert raised.value.message == "the provider ended the stream before it finished"


def test_an_unreadable_data_line_is_a_provider_error() -> None:
    """Catches the `except (KeyError, TypeError, ValueError)` around the reader being
    narrowed: a truncated `data:` line would then escape as a raw `JSONDecodeError`, which
    is not a `GatewayError`, so the route's error frame would never be written and the
    client would get a silently cut stream instead of a typed 502."""
    with pytest.raises(ProviderError) as raised:
        list(from_anthropic_events(sse_lines("stream_unreadable")))

    assert raised.value.retryable is False
    assert raised.value.http_status == 502
    assert "JSONDecodeError" in (raised.value.provider_message or "")


def test_a_data_line_that_is_not_an_object_is_a_provider_error() -> None:
    """Catches a reader that calls `.get("type")` on whatever JSON a `data:` line holds: a
    list or a bare string answers with an `AttributeError`, which is neither a
    `ProviderError` nor a `GatewayError`, so it escapes both the adapter's contract and
    the route's error frame. The lines are built here rather than recorded because no
    provider sends this - it is what a proxy rewriting the body would produce."""
    with pytest.raises(ProviderError) as raised:
        list(from_anthropic_events(["event: message_start", 'data: ["not", "an", "object"]']))

    assert raised.value.retryable is False
    assert "TypeError" in (raised.value.provider_message or "")


# --- a response_format answer streamed as content ------------------
#
# `stream_json_schema_tool.sse` is Anthropic answering a schema request: one forced
# `tool_use` block, its input split over three `input_json_delta` fragments. With
# `forced_tool` naming that block, the fragments are the answer TEXT (decision A).

FORCED_NAME = "ticket_summary"
FORCED_FRAGMENTS = (
    '{"title": "Card dec',
    'lined", "severity": 3, ',
    '"tags": ["billing", "eu"]}',
)
FORCED_INPUT: dict[str, Any] = {"title": "Card declined", "severity": 3, "tags": ["billing", "eu"]}
FORCED_PROMPT_TOKENS = 210
FORCED_COMPLETION_TOKENS = 64
FORCED_TOTAL_TOKENS = 274

FORCED_EVENT_TYPES: list[type[object]] = [
    StreamStart,
    TextDelta,
    TextDelta,
    TextDelta,
    FinishReason,
    UsageTotal,
]

JSON_FRAGMENT_MARKER = '"type":"input_json_delta"'
EMPTY_FRAGMENT_LINE = (
    'data: {"type":"content_block_delta","index":0,"delta":'
    '{"type":"input_json_delta","partial_json":""}}'
)


def forced_lines_with_an_empty_fragment_inserted() -> list[str]:
    """The recorded forced stream with one empty `partial_json` after the first fragment."""
    lines = sse_lines("stream_json_schema_tool")
    first = next(i for i, line in enumerate(lines) if JSON_FRAGMENT_MARKER in line)
    return [*lines[: first + 1], EMPTY_FRAGMENT_LINE, *lines[first + 1 :]]


FORCED_STREAM_CASES: list[tuple[str, list[str]]] = [
    ("the recorded stream", sse_lines("stream_json_schema_tool")),
    ("an empty fragment among the three", forced_lines_with_an_empty_fragment_inserted()),
]


@pytest.mark.parametrize(
    ("case", "lines"), FORCED_STREAM_CASES, ids=[case for case, _ in FORCED_STREAM_CASES]
)
def test_the_reader_streams_forced_tool_fragments_as_text(case: str, lines: list[str]) -> None:
    """Catches a reader that keeps the forced block as a `ToolCallDelta`, one that buffers
    it into a single `TextDelta` at block stop (decision A promises them live), one that
    forwards an empty fragment as an empty delta, and one that leaves `tool_use` as the
    finish reason, which the route maps to `tool_calls`."""
    events = list(from_anthropic_events(lines, forced_tool=FORCED_NAME))

    assert [type(event) for event in events] == FORCED_EVENT_TYPES, case
    assert not [event for event in events if isinstance(event, ToolCallDelta)], case
    texts = [event.text for event in events if isinstance(event, TextDelta)]
    assert tuple(texts) == FORCED_FRAGMENTS, case
    assert json.loads("".join(texts)) == FORCED_INPUT, case
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "end_turn", case
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.prompt_tokens == FORCED_PROMPT_TOKENS, case
    assert events[-1].usage.completion_tokens == FORCED_COMPLETION_TOKENS, case
    assert events[-1].usage.total_tokens == FORCED_TOTAL_TOKENS, case


EMPTY_FORCED_CASES: list[tuple[str, list[str]]] = [
    (
        "no fragment at all",
        [line for line in sse_lines("stream_json_schema_tool") if JSON_FRAGMENT_MARKER not in line],
    ),
    (
        "one empty fragment",
        [
            EMPTY_FRAGMENT_LINE if JSON_FRAGMENT_MARKER in line else line
            for line in sse_lines("stream_json_schema_tool")
        ],
    ),
]


@pytest.mark.parametrize(
    ("case", "lines"), EMPTY_FORCED_CASES, ids=[case for case, _ in EMPTY_FORCED_CASES]
)
def test_a_forced_block_with_no_fragments_yields_an_empty_object(
    case: str, lines: list[str]
) -> None:
    """Catches a reader that emits nothing for an empty forced input: the stream would
    either raise as unreadable or finish with empty content, where the non-streaming path
    gives `"{}"` for the same answer."""
    events = list(from_anthropic_events(lines, forced_tool=FORCED_NAME))

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["{}"], case
    assert not [event for event in events if isinstance(event, ToolCallDelta)], case
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "end_turn", case
    assert isinstance(events[-1], UsageTotal)


def test_held_text_is_emitted_when_no_forced_block_opens() -> None:
    """Catches a reader that holds text while `forced_tool` is set and never lets it go
    when no forced block arrives: a plain-text refusal would be lost, and the stream would
    raise as unreadable or finish empty."""
    events = list(from_anthropic_events(sse_lines("stream_text"), forced_tool=FORCED_NAME))

    texts = [event.text for event in events if isinstance(event, TextDelta)]
    assert "".join(texts) == TEXT_ANSWER
    finish_at = next(i for i, event in enumerate(events) if isinstance(event, FinishReason))
    assert all(i < finish_at for i, event in enumerate(events) if isinstance(event, TextDelta)), (
        "held text must come out before the finish"
    )
    assert isinstance(events[0], StreamStart)
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "end_turn"
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.total_tokens == TEXT_TOTAL_TOKENS


def text_then_forced_block_lines() -> list[str]:
    """A text block at index 0, then the recorded forced block moved to index 1."""
    recorded = sse_lines("stream_json_schema_tool")
    start = next(i for i, line in enumerate(recorded) if '"type":"message_start"' in line)
    text_block = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"text_delta","text":"Here it is."}}',
        'data: {"type":"content_block_stop","index":0}',
    ]
    rest = [line.replace('"index":0', '"index":1') for line in recorded[start + 1 :]]
    return [*recorded[: start + 1], *text_block, *rest]


def test_text_before_a_forced_block_is_dropped_on_the_stream() -> None:
    """Catches a reader that forwards text live while `forced_tool` is set: the preamble
    reaches the client before the JSON, and the joined content no longer parses."""
    events = list(from_anthropic_events(text_then_forced_block_lines(), forced_tool=FORCED_NAME))

    texts = [event.text for event in events if isinstance(event, TextDelta)]
    assert tuple(texts) == FORCED_FRAGMENTS
    assert json.loads("".join(texts)) == FORCED_INPUT
    assert not [event for event in events if isinstance(event, ToolCallDelta)]
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "end_turn"


UNFORCED_TOOL_CASES: list[tuple[str, str | None]] = [
    ("no forced_tool", None),
    ("a forced_tool naming no block", FORCED_NAME),
]


@pytest.mark.parametrize(
    ("case", "forced_tool"), UNFORCED_TOOL_CASES, ids=[case for case, _ in UNFORCED_TOOL_CASES]
)
def test_a_stream_tool_call_is_unchanged_without_forced_tool(
    case: str, forced_tool: str | None
) -> None:
    """Catches a reader that turns a caller's own `tool_use` block into text: with no
    `forced_tool`, or one naming a different block, `get_weather` must stay a whole call,
    its preamble text must survive, and the finish must stay `tool_use`."""
    events = list(from_anthropic_events(sse_lines("stream_tool_use"), forced_tool=forced_tool))

    calls = [event for event in events if isinstance(event, ToolCallDelta)]
    assert len(calls) == 1, case
    assert calls[0].index == TOOL_BLOCK_INDEX, case
    assert calls[0].call.id == TOOL_CALL_ID, case
    assert calls[0].call.name == TOOL_CALL_NAME, case
    assert calls[0].call.arguments == TOOL_CALL_ARGUMENTS, case
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == TOOL_TEXT
    assert isinstance(events[-2], FinishReason)
    assert events[-2].reason == "tool_use", case
    assert isinstance(events[-1], UsageTotal)
    assert events[-1].usage.total_tokens == TOOL_TOTAL_TOKENS, case
    if forced_tool is None:
        # Today's order exactly; the held-text rule may only move text when a name is set.
        assert [type(event) for event in events] == [
            StreamStart,
            TextDelta,
            ToolCallDelta,
            FinishReason,
            UsageTotal,
        ]


# --- the stream open ---------------------------------------------------------


def test_stream_read_timeout_is_exactly_120() -> None:
    """Catches the read timeout left at the non-streaming 60s, which cuts off a long
    answer mid-sentence, and catches the opposite fix of raising the CLIENT's timeout to
    120s, which would let every ordinary completion hold a worker twice as long."""
    assert STREAM_READ_TIMEOUT == 120.0
    assert CONNECT_TIMEOUT == 5.0

    adapter, scripted = make_sse_adapter([sse_body("stream_text")])
    list(adapter.stream_chat(REQUEST))

    timeout = scripted.requests[0].extensions["timeout"]
    assert timeout["read"] == 120.0
    assert timeout["connect"] == 5.0
    assert adapter.client.timeout.read == 60.0
    assert READ_TIMEOUT == 60.0


def test_the_stream_body_asks_the_provider_to_stream() -> None:
    """Catches an adapter that opens `POST /v1/messages` without `stream: true` in the
    body: Anthropic answers one complete JSON message, and the reader sees no SSE at all
    while the caller waits for events."""
    adapter, scripted = make_sse_adapter([sse_body("stream_text")])

    list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert sent.url.path == "/v1/messages"
    payload = json.loads(sent.content.decode("utf-8"))
    assert payload["stream"] is True
    assert payload["model"] == REQUEST.model
    assert sent.headers["x-api-key"] == TEST_API_KEY


def test_stream_chat_yields_the_recorded_events_over_the_transport() -> None:
    """Catches a `stream_chat` that reads the response with `.json()` or `.text` instead
    of iterating lines: the whole body would arrive as one unparsable blob, so nothing
    would reach the caller until the provider finished."""
    adapter, _ = make_sse_adapter([sse_body("stream_text")])

    events = list(adapter.stream_chat(REQUEST))

    assert [type(event) for event in events] == TEXT_EVENT_TYPES
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == TEXT_ANSWER


def test_stream_open_retries_only_before_the_first_event(sleeps: SleepRecorder) -> None:
    """Catches both halves of decision C at once: an open that gives up on a 500 the next
    attempt would have served, and a retry wrapped around the whole iteration, which would
    replay an answer whose first sentences the client has already been shown."""
    adapter, scripted = make_sse_adapter(
        [(500, {"error": {"message": "overloaded"}}), sse_body("stream_text")], sleep=sleeps
    )

    events = list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 2
    assert len(sleeps.calls) == 1, "one retried open means one wait"
    assert isinstance(events[-1], UsageTotal)
    assert MAX_RETRIES == 2

    mid_sleeps = SleepRecorder()
    mid_adapter, mid_scripted = make_sse_adapter([sse_body("stream_truncated")], sleep=mid_sleeps)
    delivered: list[StreamEvent] = []

    with pytest.raises(ProviderError):
        for event in mid_adapter.stream_chat(REQUEST):
            delivered.append(event)

    assert mid_scripted.request_count == 1, "a stream that failed mid-body was reopened"
    assert mid_sleeps.calls == []
    assert [type(event) for event in delivered] == TRUNCATED_EVENT_TYPES


def test_a_transport_failure_mid_stream_is_a_provider_error_and_is_not_retried(
    sleeps: SleepRecorder,
) -> None:
    """Catches a raw `httpx.ReadTimeout` escaping `stream_chat` once the body has started -
    the route would answer 500 with a traceback instead of the error frame decision E
    promises - and catches a retry wrapped around the iteration, which would reopen the
    call and replay sentences the client has already been shown."""

    def cut_off() -> Iterator[bytes]:
        yield sse_body("stream_truncated").encode("utf-8")
        raise httpx.ReadTimeout("scripted mid-stream cut", request=None)

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, content=cut_off(), headers={"content-type": SSE_CONTENT_TYPE}, request=request
        )

    adapter = AnthropicAdapter(
        name=TEST_PROVIDER_NAME,
        api_key=TEST_API_KEY,
        base_url="https://api.anthropic.test",
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
    assert raised.value.http_status == 502
    assert raised.value.retryable is True


def test_a_stream_that_never_opens_is_a_provider_error(sleeps: SleepRecorder) -> None:
    """Catches an unwrapped `httpx.ReadTimeout` escaping `stream_chat`: the route would
    answer 500 with a traceback instead of the 502 a client's retry logic reads, and plan
    10's fallback would have no `retryable` flag to act on."""
    adapter, scripted = make_sse_adapter([httpx.ConnectTimeout], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 3
    assert len(sleeps.calls) == 2
    assert raised.value.retryable is True
    assert raised.value.http_status == 502


def test_a_failure_status_on_the_open_never_starts_a_stream(sleeps: SleepRecorder) -> None:
    """Catches an open that ignores the status and hands the error body to the SSE reader:
    a 400 would surface as an unreadable stream rather than as the non-retryable
    `ProviderError` the route turns into a typed 502."""
    adapter, scripted = make_sse_adapter(
        [(400, {"error": {"message": "bad request"}})], sleep=sleeps
    )
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as raised:
        for event in adapter.stream_chat(REQUEST):
            events.append(event)

    assert events == []
    assert scripted.request_count == 1, "a 400 is not retried"
    assert raised.value.retryable is False


def test_the_recorded_fixtures_are_the_shape_this_module_claims() -> None:
    """Catches a fixture edited into agreement with a wrong reader: the token counts and
    the truncation are asserted against the recorded text itself, so a stream_truncated
    that grew a `message_delta` fails here rather than quietly making the zero-fill test
    vacuous."""
    truncated = sse_body("stream_truncated")
    assert "message_delta" not in truncated
    assert "message_stop" not in truncated
    assert '"input_tokens":251' in truncated

    text = sse_body("stream_text")
    assert f'"input_tokens":{TEXT_PROMPT_TOKENS}' in text
    assert f'"output_tokens":{TEXT_COMPLETION_TOKENS}' in text
    assert '"output_tokens":1' in text, "message_start must report its own count"

    thinking = sse_body("stream_thinking_only")
    assert "text_delta" not in thinking, "a text delta would make the no-content guard vacuous"
    assert "tool_use" not in thinking
    assert '"stop_reason":"end_turn"' in thinking, "the guard must fire on a FINISHED stream"

    two = sse_body("stream_two_tools")
    assert two.count('"type":"tool_use"') == 2
    assert '{"city": "Berlin"}' not in two, "the arguments must arrive as fragments"
    assert '{"zone": "Europe/Berlin"}' not in two

    tool = sse_body("stream_tool_use")
    assert f'"input_tokens":{TOOL_PROMPT_TOKENS}' in tool
    assert f'"output_tokens":{TOOL_COMPLETION_TOKENS}' in tool
    assert TOOL_CALL_ARGUMENTS not in tool, "the arguments must arrive as fragments"

    forced = sse_body("stream_json_schema_tool")
    assert f'"input_tokens":{FORCED_PROMPT_TOKENS}' in forced
    assert f'"output_tokens":{FORCED_COMPLETION_TOKENS}' in forced
    assert '"stop_reason":"tool_use"' in forced, "the rewrite to end_turn must have work to do"
    assert forced.count(JSON_FRAGMENT_MARKER) == len(FORCED_FRAGMENTS)
    whole = json.dumps(FORCED_INPUT)
    assert whole not in forced, "the input must arrive as fragments"
    assert json.dumps(whole)[1:-1] not in forced, "not even escaped inside one fragment"


TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "severity": {"type": "integer"}},
    "required": ["title", "severity"],
}

# Both rows send the SAME forced `tool_choice` to Anthropic: `to_anthropic` renders
# a `response_format` and a caller's named `tool_choice` identically on the wire.
# So only `request.response_format` can tell them apart.
STREAM_CHAT_FORCED_CASES: list[tuple[str, ChatRequest, bool]] = [
    (
        "a response_format request",
        REQUEST.model_copy(
            update={"response_format": ResponseFormat(name=FORCED_NAME, json_schema=TICKET_SCHEMA)}
        ),
        True,
    ),
    (
        "a caller's own forced tool, no response_format",
        REQUEST.model_copy(
            update={
                "tools": (ToolSpec(name=FORCED_NAME, parameters=TICKET_SCHEMA),),
                "tool_choice": ToolChoice(mode="tool", name=FORCED_NAME),
            }
        ),
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "request_", "as_content"),
    STREAM_CHAT_FORCED_CASES,
    ids=[case for case, *_ in STREAM_CHAT_FORCED_CASES],
)
def test_stream_chat_passes_the_response_format_name_as_forced_tool(
    case: str, request_: ChatRequest, as_content: bool
) -> None:
    """Catches a `stream_chat` that never passes `forced_tool` (the first row stays a tool
    call), and one that reads it from the outbound `tool_choice` instead of
    `response_format`: a caller who forced its own tool would get the call back as text."""
    adapter, scripted = make_sse_adapter([sse_body("stream_json_schema_tool")])

    events = list(adapter.stream_chat(request_))

    assert scripted.request_count == 1
    sent = json.loads(scripted.requests[0].content.decode("utf-8"))
    assert sent["tool_choice"] == {"type": "tool", "name": FORCED_NAME}, case
    texts = [event.text for event in events if isinstance(event, TextDelta)]
    calls = [event for event in events if isinstance(event, ToolCallDelta)]
    assert isinstance(events[-2], FinishReason)
    if as_content:
        assert tuple(texts) == FORCED_FRAGMENTS, case
        assert calls == [], case
        assert events[-2].reason == "end_turn", case
    else:
        assert texts == [], case
        assert [call.call.name for call in calls] == [FORCED_NAME], case
        assert events[-2].reason == "tool_use", case
