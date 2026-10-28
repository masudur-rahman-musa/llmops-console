"""SSE framing for a streamed chat completion."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain

from app.core.errors import GatewayError, ProviderError
from app.core.http_errors import error_body
from app.model_routing.resolver import Resolved
from app.providers.base import Adapter, StreamEvent
from app.redaction import StreamRehydrator, rehydrate
from app.schemas.chat import ChatCompletionRequest
from app.schemas.provider import FinishReason, TextDelta, ToolCallDelta, Usage, UsageTotal
from app.services.wire_mapping import (
    COMPLETION_ID_BYTES,
    COMPLETION_ID_PREFIX,
    to_chunk,
    to_neutral,
)

SSE_MEDIA_TYPE = "text/event-stream"
"""The content type of a streamed answer."""


DONE_SENTINEL = "[DONE]"
"""The payload of the frame that ends a complete stream, spelled as OpenAI
publishes it. An SDK reads until it arrives, so its ABSENCE is what reports a
truncated stream. Tests pin the
whole final line and its absence after a failure."""


SSE_HEADERS: Mapping[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@dataclass
class StreamPlan:
    """What `_streamed` hands back so the HTTP layer can build the response.

    `frames` is the generator `sse_frames` produces; `media_type` and `headers`
    are what the route needs to build an HTTP streaming response. Keeping them
    together lets the route hold one thing instead of three, and keeps
    the HTTP response type out of the service layer entirely.
    """

    frames: Generator[str, None, None]
    media_type: str
    headers: dict[str, str]


"""The two headers a streamed answer needs beyond the six of #9.

Both exist to stop something between the gateway and the client holding the
frames back and delivering them as one block, which turns a stream into a slow
non-streaming answer: `Cache-Control` for caches, `X-Accel-Buffering` for nginx
and the ingress controllers that read it. """


def _completion_id() -> str:
    """One fresh completion id, shared by every frame of one streamed answer."""
    return f"{COMPLETION_ID_PREFIX}{secrets.token_hex(COMPLETION_ID_BYTES)}"


def _frame(payload: str) -> str:
    """One SSE frame: `data: `, the payload, and the blank line that ends it.

    The blank line is not decoration - it is what terminates an event, so a
    client whose stream never carries one hands nothing to its parser and waits
    until its read timeout. A test asserts the
    raw bytes of a real response, because the helpers the other cases use split
    on lines and cannot see a separator at all.
    """
    return f"data: {payload}\n\n"


def sse_frames(
    first: StreamEvent,
    events: Iterator[StreamEvent],
    alias: str,
    completion_id: str,
    on_usage: Callable[[Usage], None] | None = None,
    *,
    mapping: Mapping[str, str] | None = None,
    on_unreported_usage: Callable[[int], None] | None = None,
    on_close: Callable[[], None] | None = None,
    on_finish: Callable[[str], None] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> Generator[str, None, None]:
    """Render the neutral events as SSE frames, ending with `[DONE]` if they finish.

    A `GatewayError` raised part way through becomes ONE frame carrying the same
    `{"error": {...}}` body every non-streaming failure uses, and the stream then
    ends with no sentinel: the status is already 200 and cannot be changed, so
    the frame says what went wrong and the missing `[DONE]` is what lets a client
    tell a truncated answer from a finished one.

    Only a TYPED failure is rendered, and the limit is deliberate rather than an
    oversight. An untyped one is a bug, and there is nothing safe to say about a
    bug in a frame; it reaches the client as a cut stream with no sentinel, which
    is the same "truncated" report a typed failure gets, minus the
    explanation. Nothing in this gateway turns it into a body: the `Exception`
    handler registered in `app/core/http_errors.py` is served by Starlette's
    `ServerErrorMiddleware`, which re-raises unchanged once a response has
    started, so for a crash inside this generator that handler never runs and the
    traceback is logged by the ASGI server rather than by us.

    The provider's connection is released in a `finally` rather than left to the
    garbage collector. A client that navigates away mid-answer leaves this
    generator suspended, and the streamed `httpx` response underneath it stays
    checked out of the pool until it is closed.

    Typed as a `Generator` rather than an `Iterator` because a caller has to be
    able to `close()` it: that is what releases the provider stream when the
    client goes away, and an `Iterator` does not promise the method.

    A sync generator, deliberately: the chat route runs it inside
    `asyncio.to_thread`, so a provider that pauses between events
    holds a worker thread rather than the event loop.

    `on_unreported_usage` is called with the characters this generator actually
    SENT, and only when the stream ended without a `UsageTotal` event - a typed
    failure after the first frame, or a client that went away mid-answer. It runs
    in the `finally` beside the release above, so the abandoned stream reaches it
    too, and it is what makes this true: every served request
    books a row, and the one nobody reported usage for books an estimated one
    rather than nothing at all.
    A stream that DID report usage has already booked from `on_usage`, so the
    flag below is what keeps a completed stream from booking twice.

    The count is of text this generator EMITTED, not of what the provider sent:
    text still held in the rehydrator was never part of an answer anyone
    received, and counting it would price a fragment the client never saw.

    `mapping` is the rehydration the caller asked for, and `None` - the shape a
    request without `X-Gateway-Rehydrate` takes - leaves every frame exactly as
    it was before rehydration existed.
    With a mapping, a `TextDelta` goes through a `StreamRehydrator` and becomes
    ZERO or one frame: a fragment that could still be completing a placeholder is
    held rather than sent as an empty delta. A `ToolCallDelta`
    carries a COMPLETE call, so its `arguments` need no
    buffer and are substituted whole.

    The held text is flushed before EVERY non-text event, before the error frame
    and before `[DONE]` (the rules), so a `finish_reason` can never
    overtake the last of the answer and nothing the provider produced is dropped.

    `on_close` runs last in the `finally`, after every frame this generator
    sends, which is what ends a stream's generation span with the stream itself.

    `on_finish` gets the provider's finish reason when its event arrives, for
    the trace's Output note.

    `on_text` gets each provider text piece BEFORE placeholders are put back,
    which is what a tenant with `trace_content` on sends to Langfuse as the
    answer.

    The wrap runs INSIDE this generator rather than around the iterator it is
    handed: the `finally` below closes `events` by name, and closing
    a wrapper instead would leave the provider's stream to the garbage collector
    - which a test exists to catch.
    """
    created = int(time.time())
    # An empty map stands in for "no rehydration asked for": both `rehydrate` and
    # `StreamRehydrator` are the identity over one, so the frames of a request
    # without the header are what they were before this parameter existed, and
    # the two paths do not need two copies of the loop below.
    restore: Mapping[str, str] = mapping if mapping is not None else {}
    rehydrator = StreamRehydrator(restore)
    sent_chars = 0
    usage_reported = False

    def rendered(event: StreamEvent) -> str:
        return _frame(
            to_chunk(
                event, completion_id=completion_id, created=created, alias=alias
            ).model_dump_json()
        )

    def held() -> Generator[str, None, None]:
        """The frame carrying whatever is still buffered, or no frame at all."""
        nonlocal sent_chars
        text = rehydrator.flush()
        if text:
            sent_chars += len(text)
            yield rendered(TextDelta(text=text))

    try:
        for event in chain((first,), events):
            if isinstance(event, TextDelta):
                if on_text is not None:
                    on_text(event.text)
                emitted = rehydrator.feed(event.text)
                if not emitted and event.text:
                    # Everything the provider sent is still held: no frame at all.
                    # `event.text` guards the other way round - a
                    # provider delta that was empty to begin with is rendered
                    # exactly as it was before this parameter existed.
                    continue
                sent_chars += len(emitted)
                event = TextDelta(text=emitted)
            else:
                yield from held()
                if isinstance(event, ToolCallDelta):
                    event = ToolCallDelta(
                        index=event.index,
                        call=event.call.model_copy(
                            update={"arguments": rehydrate(event.call.arguments, restore)}
                        ),
                    )
            if isinstance(event, FinishReason) and on_finish is not None:
                on_finish(event.reason)
            if isinstance(event, UsageTotal):
                usage_reported = True
                if on_usage is not None:
                    on_usage(event.usage)
            yield rendered(event)
        yield from held()
    except GatewayError as error:
        yield from held()
        yield _frame(
            json.dumps(
                error_body(
                    message=error.message,
                    code=error.code,
                    http_status=error.http_status,
                    param=error.param,
                )
            )
        )
        return
    finally:
        if isinstance(events, Generator):
            # `itertools.chain` has no `close`, so the adapter's own generator is
            # what has to be closed; `_streamed` passes it separately for that
            # reason rather than chaining the first event onto it here.
            events.close()
        if on_unreported_usage is not None and not usage_reported:
            on_unreported_usage(sent_chars)
        if on_close is not None:
            on_close()
    yield _frame(DONE_SENTINEL)


def streamed(
    adapter: Adapter,
    body: ChatCompletionRequest,
    chosen: Resolved,
    headers: dict[str, str],
    on_usage: Callable[[Usage], None],
    *,
    mapping: Mapping[str, str] | None = None,
    on_unreported_usage: Callable[[int], None] | None = None,
    on_close: Callable[[], None] | None = None,
    on_finish: Callable[[str], None] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> StreamPlan:
    """Open the stream, pull its first event, and return the plan for the route.

    The eager first event is the whole point. The HTTP layer sends the status
    the moment the response is returned, so anything the adapter raises after
    that can only be an error frame inside a 200 - which an SDK reports to its
    caller as a parse failure rather than as the provider outage it was. Pulling
    one event here keeps every pre-stream failure a typed HTTP error the existing
    handlers render.

    A stream that opens and ends without a single event is a provider failure
    too, not a `StopIteration` for the last-resort handler to turn into a 500
    with no code.

    Returns a `StreamPlan` so the route builds the HTTP response and the
    response type stays out of the service layer.
    """
    events = adapter.stream_chat(to_neutral(body, chosen.model))
    first = next(events, None)
    if first is None:
        raise ProviderError(
            f"{chosen.provider}: the provider opened a stream and sent nothing",
            retryable=False,
        )

    return StreamPlan(
        frames=sse_frames(
            first,
            events,
            body.model,
            _completion_id(),
            on_usage=on_usage,
            mapping=mapping,
            on_unreported_usage=on_unreported_usage,
            on_close=on_close,
            on_finish=on_finish,
            on_text=on_text,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers={**headers, **SSE_HEADERS},
    )
