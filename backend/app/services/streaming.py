"""SSE framing for a streamed chat completion."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from itertools import chain

from fastapi.responses import StreamingResponse

from app.core.errors import GatewayError, ProviderError
from app.core.http_errors import error_body
from app.model_routing.resolver import Resolved
from app.providers.base import Adapter, StreamEvent, TextDelta, ToolCallDelta, Usage, UsageTotal
from app.redaction import StreamRehydrator, rehydrate
from app.schemas.chat import ChatCompletionRequest
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
truncated stream; `test_a_stream_ends_with_done` pins the
whole final line and `test_a_failure_after_the_first_event_emits_an_error_chunk_and_no_done`
pins the absence."""


SSE_HEADERS: Mapping[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
"""The two headers a streamed answer needs beyond the six of #9.

Both exist to stop something between the gateway and the client holding the
frames back and delivering them as one block, which turns a stream into a slow
non-streaming answer: `Cache-Control` for caches, `X-Accel-Buffering` for nginx
and the ingress controllers that read it. Pinned by
`test_headers_on_a_streamed_success`."""


def _completion_id() -> str:
    """One fresh completion id, shared by every frame of one streamed answer."""
    return f"{COMPLETION_ID_PREFIX}{secrets.token_hex(COMPLETION_ID_BYTES)}"


def _frame(payload: str) -> str:
    """One SSE frame: `data: `, the payload, and the blank line that ends it.

    The blank line is not decoration - it is what terminates an event, so a
    client whose stream never carries one hands nothing to its parser and waits
    until its read timeout. `test_every_frame_ends_with_a_blank_line` asserts the
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
) -> Generator[str, None, None]:
    """Render the neutral events as SSE frames, ending with `[DONE]` if they finish.

    A `GatewayError` raised part way through becomes ONE frame carrying the same
    `{"error": {...}}` body every non-streaming failure uses, and the stream then
    ends with no sentinel: the status is already 200 and cannot be changed, so
    the frame says what went wrong and the missing `[DONE]` is what lets a client
    tell a truncated answer from a finished one (an earlier change decision E,
    `test_a_failure_after_the_first_event_emits_an_error_chunk_and_no_done`).

    Only a TYPED failure is rendered, and the limit is deliberate rather than an
    oversight. An untyped one is a bug, and there is nothing safe to say about a
    bug in a frame; it reaches the client as a cut stream with no sentinel, which
    is the same "truncated" report decision E gives a typed failure, minus the
    explanation. Nothing in this gateway turns it into a body: the `Exception`
    handler registered in `gateway/errors.py` is served by Starlette's
    `ServerErrorMiddleware`, which re-raises unchanged once a response has
    started, so for a crash inside this generator that handler never runs and the
    traceback is logged by the ASGI server rather than by us
    (`test_an_untyped_failure_mid_stream_is_not_rendered_as_a_frame`).

    The provider's connection is released in a `finally` rather than left to the
    garbage collector. A client that navigates away mid-answer leaves this
    generator suspended, and the streamed `httpx` response underneath it stays
    checked out of the pool until it is closed
    (`test_closing_the_frame_generator_releases_the_provider_stream`).

    Typed as a `Generator` rather than an `Iterator` because a caller has to be
    able to `close()` it: that is what releases the provider stream when the
    client goes away, and an `Iterator` does not promise the method.

    A sync generator, deliberately: FastAPI iterates it in a worker thread, so a
    provider that pauses between events holds a thread rather than the event
    loop, exactly as the non-streaming path is already handled.

    `on_unreported_usage` is called with the characters this generator actually
    SENT, and only when the stream ended without a `UsageTotal` event - a typed
    failure after the first frame, or a client that went away mid-answer. It runs
    in the `finally` beside the release above, so the abandoned stream reaches it
    too, and it is what makes an earlier change decision E true: every served request
    books a row, and the one nobody reported usage for books an estimated one
    rather than nothing at all
    (`test_a_stream_that_fails_after_its_first_frame_books_one_estimated_row`).
    A stream that DID report usage has already booked from `on_usage`, so the
    flag below is what keeps a completed stream from booking twice
    (`test_a_stream_with_usage_books_exactly_one_row`).

    The count is of text this generator EMITTED, not of what the provider sent:
    text still held in the rehydrator was never part of an answer anyone
    received, and counting it would price a fragment the client never saw.

    `mapping` is the rehydration the caller asked for, and `None` - the shape a
    request without `X-Gateway-Rehydrate` takes - leaves every frame exactly as
    it was before an earlier change (`test_a_stream_without_the_header_is_what_it_is_today`).
    With a mapping, a `TextDelta` goes through a `StreamRehydrator` and becomes
    ZERO or one frame: a fragment that could still be completing a placeholder is
    held rather than sent as an empty delta (decision D). A `ToolCallDelta`
    carries a COMPLETE call, so its `arguments` need no
    buffer and are substituted whole
    (`test_a_streamed_tool_calls_arguments_are_rehydrated`).

    The held text is flushed before EVERY non-text event, before the error frame
    and before `[DONE]` (decisions E and F), so a `finish_reason` can never
    overtake the last of the answer and nothing the provider produced is dropped
    (`test_held_text_is_flushed_before_the_finish_reason`,
    `test_held_text_is_flushed_before_an_error_frame`,
    `test_a_trailing_partial_placeholder_is_flushed_before_done`).

    The wrap runs INSIDE this generator rather than around the iterator it is
    handed (decision C): the `finally` below closes `events` by name, and closing
    a wrapper instead would leave the provider's stream to the garbage collector
    - which is what `test_closing_the_frame_generator_releases_the_provider_stream`
    exists to catch.
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
                emitted = rehydrator.feed(event.text)
                if not emitted and event.text:
                    # Everything the provider sent is still held: no frame at all
                    # (decision D). `event.text` guards the other way round - a
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
    yield _frame(DONE_SENTINEL)


def _streamed(
    adapter: Adapter,
    body: ChatCompletionRequest,
    chosen: Resolved,
    headers: dict[str, str],
    on_usage: Callable[[Usage], None],
    *,
    mapping: Mapping[str, str] | None = None,
    on_unreported_usage: Callable[[int], None] | None = None,
) -> StreamingResponse:
    """Open the stream, pull its first event, and only then answer 200.

    The eager first event is the whole point. `StreamingResponse` sends the
    status the moment it is returned, so anything the adapter raises after that
    can only be an error frame inside a 200 - which an SDK reports to its caller
    as a parse failure rather than as the provider outage it was. Pulling one
    event here keeps every pre-stream failure a typed HTTP error the existing
    handlers render (`test_a_provider_failure_before_the_first_event_is_502_with_no_stream`).

    A stream that opens and ends without a single event is a provider failure
    too, not a `StopIteration` for the last-resort handler to turn into a 500
    with no code (`test_a_stream_that_opens_and_yields_nothing_is_502`).
    """
    events = adapter.stream_chat(to_neutral(body, chosen.model))
    first = next(events, None)
    if first is None:
        raise ProviderError(
            f"{chosen.provider}: the provider opened a stream and sent nothing",
            retryable=False,
        )

    return StreamingResponse(
        sse_frames(
            first,
            events,
            body.model,
            _completion_id(),
            on_usage=on_usage,
            mapping=mapping,
            on_unreported_usage=on_unreported_usage,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers={**headers, **SSE_HEADERS},
    )
