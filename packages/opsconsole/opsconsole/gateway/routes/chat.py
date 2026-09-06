"""`POST /v1/chat/completions`: the OpenAI wire shape, and the join it makes.

An earlier change left three things that did not touch each other: a request could be
authenticated, resolved to a provider and a model id, and a provider could be
called. This module is that join, plus the one control that has to happen before
the outbound call: inline redaction. No cache, no ledger, no fallback.

The OpenAI wire shape lives here and stops here. `providers/base.py` keeps the
gateway's own vocabulary, so the two mappings - a wire request into a
`ChatRequest`, a `ChatResponse` back into a `chat.completion` - are one file
rather than a shape spread through the package. That is why an OpenAI tool call
is `{"type": "function", "function": {...}}` in this module and a flat `ToolCall`
everywhere behind it.

Two published details worth knowing before reading the code:

* The body names the ALIAS the caller sent, never the provider model id. Aliases
  are the only routable names (the design), so an answer naming a provider model
  would hand back a name that cannot be sent again and disclose which provider
  served the request. The provider model id is an operator's fact and travels in
  `X-Gateway-Model` (`test_response_model_is_the_alias_the_caller_named`).
* `stream: true` answers `text/event-stream`: the same resolution and the same
  provider call, rendered as `chat.completion.chunk` frames and closed with
  `data: [DONE]`. The stream is OPENED and its first event pulled inside the
  route, before anything is returned, so a failure with nothing yet delivered is
  still a typed HTTP error rather than a 200 carrying an error frame
  (`test_a_provider_failure_before_the_first_event_is_502_with_no_stream`). Once
  the status is sent it cannot be changed, so a later failure is one error frame
  and no sentinel.
* When the tenant's rule for the request's data class carries `redact: true`,
  every message `content` is redacted before the wire request becomes a
  `ChatRequest`, so the adapter can only ever be handed placeholders
  (`test_the_chat_adapter_receives_only_what_the_class_allows`). Tool call
  ARGUMENTS are left alone: they are a machine payload the caller's own tool
  parses, and a placeholder inside that JSON is a value the tool cannot act on.

Both shapes are built here, from one set of wire models: a `chat.completion` and
the `chat.completion.chunk` frames are the same vocabulary rendered twice, so a
tool call cannot be OpenAI-shaped on one path and flat on the other.

Every rejection the wire models make is a 422 `VALIDATION` raised before any
outbound call, by construction: FastAPI validates the body before the route body
runs. The routing and provider failures - 404, 451, 503, 502 - are raised inside
the route by `resolve()` and the adapter, and reach the client through the
handlers in `gateway/errors.py`.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from itertools import chain
from typing import Any, Literal, assert_never

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.types import Scope

from opsconsole.core.errors import GatewayError, ProviderError
from opsconsole.core.metrics import label_resolution, record_redactions, record_tokens
from opsconsole.core.tracing import SpanAttributes, annotate, stage_span
from opsconsole.gateway.context import Context, RequestContext
from opsconsole.gateway.errors import error_body
from opsconsole.gateway.routes.redact import (
    REDACT_SPAN,
    redact_or_reject,
    redaction_applies,
)
from opsconsole.providers.base import (
    Adapter,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ResponseFormat,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSpec,
    Usage,
    UsageTotal,
)
from opsconsole.routing.policy import Policy, PolicyStore
from opsconsole.routing.resolver import Resolved, resolve

COMPLETION_ID_PREFIX = "chatcmpl-"
"""What every completion id starts with, as OpenAI publishes it. Pinned by
`test_successful_completion_is_openai_shaped`."""

COMPLETION_ID_BYTES = 12
"""Random bytes behind each completion id, rendered as 24 hex characters.

The value is pinned exactly by `test_completion_id_is_random_and_the_pinned_length`,
which also asserts that two completions do not share an id. Both matter to a
component that has not been written yet: the ledger and the trace store key rows
by this id, so a counter, a shortened id or a per-process constant would merge
two tenants' calls into one row."""

COMPLETION_OBJECT: Literal["chat.completion"] = "chat.completion"
"""The `object` discriminator an OpenAI-compatible client branches on."""

CHUNK_OBJECT: Literal["chat.completion.chunk"] = "chat.completion.chunk"
"""The same discriminator for a streamed frame. A client reads the answer out of
`choices[].delta` rather than `choices[].message` when it sees this
(`test_every_chunk_carries_the_alias_and_one_id`)."""

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

FINISH_REASON_BY_STOP_REASON: Mapping[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}
"""Provider stop reasons in OpenAI's words, the whole table
(`test_finish_reason_table`).

A reason not in the table passes through unchanged rather than collapsing into
`stop`: a client branches on this field to decide whether to run a tool or to
report a truncated answer, and silently calling an unmapped reason a clean finish
is the one wrong answer that looks right.
"""

CACHE_STATUS_MISS = "miss"
"""`X-Gateway-Cache` on every response here. Nothing is cached yet, and `miss` is
the truthful value for a call that went to the provider."""

REDACTED_TRUE = "true"
REDACTED_FALSE = "false"
"""The two words `X-Gateway-Redacted` is written with. Lowercase, because a
console sums this string and `True` would count as neither
(`test_the_two_header_values_are_the_published_words`).

Which one a response carries is decided per REQUEST, not per route: the header is
`true` exactly when at least one value was actually replaced (an earlier change decision
C), so a redact class carrying no personal data answers `false`."""


class _Wire(BaseModel):
    """A model on the published API boundary: frozen, and unknown fields rejected.

    Rejected rather than ignored: a misspelled `max_token`
    becomes a 422 naming the field instead of a silently applied default
    (`test_unknown_body_field_is_422`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class FunctionSpec(_Wire):
    """The body of an OpenAI tool definition."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = {}


class WireTool(_Wire):
    """One tool offered to the model, in OpenAI's `{"type": "function", ...}` wrapper."""

    type: Literal["function"]
    function: FunctionSpec


class NamedFunction(_Wire):
    """The function named by an object-form `tool_choice`."""

    name: str


class NamedToolChoice(_Wire):
    """`tool_choice: {"type": "function", "function": {"name": ...}}`."""

    type: Literal["function"]
    function: NamedFunction


class WireToolCallFunction(_Wire):
    """The call the model made: a name, and its arguments as a JSON string."""

    name: str
    arguments: str


class WireToolCall(_Wire):
    """One tool call, in both directions: replayed by the caller and returned by us."""

    id: str
    type: Literal["function"] = "function"
    function: WireToolCallFunction


class WireMessage(_Wire):
    """One turn as an OpenAI client sends it, with the coherence rules enforced.

    The rules exist because the alternative is letting the provider judge the
    conversation: a tool result with no id, or a tool call hung off a user turn,
    would come back as a provider 400 the caller cannot act on instead of a 422
    naming what is wrong (`test_incoherent_message_is_422`).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[WireToolCall, ...] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> WireMessage:
        """Reject the role and field combinations no provider can be told."""
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("a tool turn needs the tool_call_id it answers")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id belongs only on a tool turn")

        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls belong only on an assistant turn")

        if not self.content and not self.tool_calls:
            # Empty, not merely absent: Anthropic rejects an empty text block, so
            # `{"role": "user", "content": ""}` would leave as a 502 carrying the
            # provider's refusal of what is plainly the caller's own mistake
            # (`test_incoherent_message_is_422`, the empty-content rows).
            raise ValueError("content is required unless the turn carries tool_calls")
        return self


class StreamOptions(_Wire):
    """`stream_options: {"include_usage": bool}`, accepted so an ordinary SDK works.

    The OpenAI SDK sends this field on every streamed request, so refusing it
    outright through `extra="forbid"` would break a client that did nothing
    wrong. `false` is refused instead, with a 422 that says why: usage is
    returned on every stream (the design) because the ledger prices from it, so
    honouring `false` would be a promise the gateway does not keep (an earlier change
    decision D). Both halves are pinned - `test_include_usage_false_is_422` and
    `test_include_usage_true_is_accepted`.
    """

    include_usage: bool = True


class JsonSchemaSpec(_Wire):
    """The `json_schema` body of a `response_format`.

    `schema` is a Python attribute name Pydantic reserves, so the field is
    `json_schema` here and aliased back to `schema` on the wire.
    """

    name: str
    json_schema: Mapping[str, Any] = Field(alias="schema")


class WireResponseFormat(_Wire):
    """`response_format: {"type": "json_schema", "json_schema": {...}}`."""

    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatCompletionRequest(_Wire):
    """One chat completion request, exactly as an OpenAI-compatible client sends it.

    `model` is an alias, not a provider model id; the resolver checks its shape
    before any lookup, so a malformed one is a 422 rather than a 404 claiming the
    name does not exist (`test_a_malformed_alias_is_422_not_404`).
    """

    model: str
    messages: tuple[WireMessage, ...] = Field(min_length=1)
    tools: tuple[WireTool, ...] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    response_format: WireResponseFormat | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def _supported(self) -> ChatCompletionRequest:
        """Refuse the two requests this gateway cannot serve honestly."""
        if self.stream_options is not None and not self.stream_options.include_usage:
            raise ValueError("stream_options.include_usage: usage is reported on every stream")
        if self.tool_choice is not None and self.response_format is not None:
            # Each demands a different forced call, so whichever is applied last
            # silently wins and the caller gets an answer shaped by the field it
            # did not think was in charge.
            raise ValueError("tool_choice and response_format cannot be combined")
        return self


class CompletionMessage(BaseModel):
    """The assistant turn we answer with. `content` is null when the answer was a call."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: tuple[WireToolCall, ...] | None = None


class Choice(BaseModel):
    """The single choice this gateway returns. `n > 1` is not offered."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    message: CompletionMessage
    finish_reason: str


class CompletionUsage(BaseModel):
    """What the call consumed, carried through from the adapter and never zero-filled.

    The ledger prices from these three numbers, so a zero books a real, billed
    call as free (`test_usage_is_carried_through_unchanged`).
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """The `chat.completion` envelope. `model` is the alias the caller named."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion"] = COMPLETION_OBJECT
    created: int
    model: str
    choices: tuple[Choice, ...]
    usage: CompletionUsage


class WireToolCallChunk(_Wire):
    """One tool call inside a streamed delta: `WireToolCall` plus the position.

    `index` is what an OpenAI client assembles a multi-call answer by, and it is
    the provider's own position rather than this frame's: hardcoding 0 merges two
    calls of one answer into one (`test_streamed_tool_call_is_one_whole_delta`).
    """

    index: int
    id: str
    type: Literal["function"] = "function"
    function: WireToolCallFunction


class ChunkDelta(BaseModel):
    """What one frame adds to the message being assembled. Every field optional.

    Optional because a frame carries exactly one kind of news: the opening role,
    a fragment of text, a completed tool call, or - on the frame that closes the
    answer - nothing at all beside the `finish_reason` above it. `role` rides the
    first frame only; repeating it is read as a second message
    (`test_first_chunk_carries_the_assistant_role`).
    """

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: tuple[WireToolCallChunk, ...] | None = None


class ChunkChoice(BaseModel):
    """The single choice a frame carries. `n > 1` is not offered, as on the completion."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """One `chat.completion.chunk` frame. `model` is the alias, as on the completion.

    `usage` is null on every frame but the last, which carries it with an EMPTY
    `choices`: that is where an OpenAI-compatible client looks for it, and it is
    the number the ledger will price from, so it is carried from the provider and
    never zero-filled
    (`test_usage_chunk_is_last_and_carries_the_fixture_numbers_exactly`).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion.chunk"] = CHUNK_OBJECT
    created: int
    model: str
    choices: tuple[ChunkChoice, ...]
    usage: CompletionUsage | None = None


ROUTE_SPAN = "route"
"""The stage span around `resolve()`. Shared with the embeddings route, which
imports it from here for the same reason it imports `gateway_headers`: one name
in one place, so two routes cannot disagree about what a stage is called. Pinned
by `test_the_emitted_span_names_are_exactly_the_stages_that_exist`."""

PROVIDER_SPAN_PREFIX = "provider:"
"""What a provider span's name starts with. The suffix is the RESOLVED provider,
never the alias the caller sent: an alias names a policy entry and a provider
names what actually served the request, which is the one question this span
exists to answer (`test_the_provider_span_names_the_resolved_provider`)."""


def provider_span_name(provider: str) -> str:
    """The stage span name for one resolved provider."""
    return f"{PROVIDER_SPAN_PREFIX}{provider}"


def resolution_attributes(alias: str, resolved: Resolved) -> SpanAttributes:
    """What the resolver decided, as span labels.

    Set only AFTER `resolve()` returns, which is what keeps a refused request's
    alias off its trace: a 451 names the alias and the regions a tenant may not
    reach, and a residency refusal is not the place to write either down
    (`test_a_residency_refusal_records_the_code_on_the_span`).
    """
    return SpanAttributes(alias=alias, provider=resolved.provider, region=resolved.region)


def gateway_headers(resolved: Resolved, trace_id: str, *, redacted: bool) -> dict[str, str]:
    """The six `X-Gateway-*` headers describing one served request (#9).

    `redacted` is keyword-only and has no default on purpose. A default would let
    a route report "nothing was redacted" for a request it did redact simply by
    forgetting the argument, and that is the one value here a caller might act on
    (`test_the_redaction_outcome_is_keyword_only_and_required`).

    Six, and only six: `X-Gateway-Cost-USD` is deliberately absent until the
    ledger can price a call, because a header reading
    `0.00` is a wrong number a dashboard would happily sum. The set is asserted
    whole by `test_the_header_set_is_exactly_six` and its absence by
    `test_cost_header_is_absent`.

    Only the success path carries them. An error answers through the handlers in
    `gateway/errors.py`, which run outside the route and have no resolver result
    or context to describe - the seam an earlier change opens for tracing.
    """
    return {
        "X-Gateway-Provider": resolved.provider,
        "X-Gateway-Region": resolved.region,
        "X-Gateway-Model": resolved.model,
        "X-Gateway-Cache": CACHE_STATUS_MISS,
        "X-Gateway-Redacted": REDACTED_TRUE if redacted else REDACTED_FALSE,
        "X-Gateway-Trace-Id": trace_id,
    }


def _neutral_messages(messages: tuple[WireMessage, ...]) -> tuple[ChatMessage, ...]:
    """Map the conversation into the neutral contract, in the caller's order.

    The order is the mapping's whole job here: a tool result means nothing except
    immediately after the assistant turn that asked for it, and placing the
    system turn is the adapter's business rather than this layer's
    (`test_messages_reach_the_adapter_in_order`).
    """
    return tuple(
        ChatMessage(
            role=message.role,
            content=message.content if message.content is not None else "",
            tool_calls=tuple(
                ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
                for call in (message.tool_calls or ())
            ),
            tool_call_id=message.tool_call_id,
        )
        for message in messages
    )


def _neutral_tool_choice(
    choice: Literal["auto", "none", "required"] | NamedToolChoice | None,
) -> ToolChoice | None:
    """Map OpenAI's `tool_choice` union onto the neutral one.

    The string forms carry across unchanged; the object form becomes the `tool`
    mode carrying the name. `required` stays `required` - translating it into a
    provider's spelling is the adapter's job
    (`test_tool_choice_maps_into_the_neutral_contract`).
    """
    if choice is None:
        return None
    if isinstance(choice, NamedToolChoice):
        return ToolChoice(mode="tool", name=choice.function.name)
    return ToolChoice(mode=choice)


def to_neutral(body: ChatCompletionRequest, model: str) -> ChatRequest:
    """Map a wire request onto the neutral contract, against the resolved model id.

    `model` is the provider's own id from the resolver, never the alias the body
    names: an adapter is handed a name its provider knows
    (`test_the_adapter_is_handed_the_resolved_provider_model_not_the_alias`).
    """
    return ChatRequest(
        model=model,
        messages=_neutral_messages(body.messages),
        tools=tuple(
            ToolSpec(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters,
            )
            for tool in (body.tools or ())
        ),
        tool_choice=_neutral_tool_choice(body.tool_choice),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        response_format=(
            ResponseFormat(
                name=body.response_format.json_schema.name,
                json_schema=body.response_format.json_schema.json_schema,
            )
            if body.response_format is not None
            else None
        ),
    )


def to_wire(answer: ChatResponse, alias: str) -> ChatCompletion:
    """Map a neutral answer into the `chat.completion` envelope.

    `alias` is what the body names, not `answer.model`: see the module docstring.
    Tool calls are re-wrapped in OpenAI's `function` shape, because a client reads
    the name from `function.name` and would dispatch nothing against the flat
    neutral one (`test_returned_tool_calls_are_openai_shaped`).
    """
    calls = tuple(
        WireToolCall(
            id=call.id,
            function=WireToolCallFunction(name=call.name, arguments=call.arguments),
        )
        for call in answer.tool_calls
    )
    return ChatCompletion(
        id=f"{COMPLETION_ID_PREFIX}{secrets.token_hex(COMPLETION_ID_BYTES)}",
        created=int(time.time()),
        model=alias,
        choices=(
            Choice(
                message=CompletionMessage(
                    content=answer.content if answer.content else None,
                    tool_calls=calls if calls else None,
                ),
                finish_reason=FINISH_REASON_BY_STOP_REASON.get(
                    answer.finish_reason, answer.finish_reason
                ),
            ),
        ),
        usage=CompletionUsage(
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
            total_tokens=answer.usage.total_tokens,
        ),
    )


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


def to_chunk(
    event: StreamEvent, *, completion_id: str, created: int, alias: str
) -> ChatCompletionChunk:
    """Map one neutral stream event into the frame that carries it.

    The counterpart of `to_wire`, and it reuses the same wire models and the same
    `FINISH_REASON_BY_STOP_REASON`, so a streamed tool call cannot be shaped
    differently from a returned one and a streamed stop reason cannot be mapped
    by a second table that drifts from the first
    (`test_finish_reason_table_on_the_stream`).

    `alias` is what every frame names, never the provider model id, for the
    reason the module docstring gives
    (`test_every_chunk_carries_the_alias_and_one_id`).
    """
    envelope: dict[str, Any] = {"id": completion_id, "created": created, "model": alias}

    if isinstance(event, StreamStart):
        return ChatCompletionChunk(
            **envelope, choices=(ChunkChoice(delta=ChunkDelta(role=event.role)),)
        )
    if isinstance(event, TextDelta):
        return ChatCompletionChunk(
            **envelope, choices=(ChunkChoice(delta=ChunkDelta(content=event.text)),)
        )
    if isinstance(event, ToolCallDelta):
        return ChatCompletionChunk(
            **envelope,
            choices=(
                ChunkChoice(
                    delta=ChunkDelta(
                        tool_calls=(
                            WireToolCallChunk(
                                index=event.index,
                                id=event.call.id,
                                function=WireToolCallFunction(
                                    name=event.call.name, arguments=event.call.arguments
                                ),
                            ),
                        )
                    )
                ),
            ),
        )
    if isinstance(event, FinishReason):
        return ChatCompletionChunk(
            **envelope,
            choices=(
                ChunkChoice(
                    delta=ChunkDelta(),
                    finish_reason=FINISH_REASON_BY_STOP_REASON.get(event.reason, event.reason),
                ),
            ),
        )
    if isinstance(event, UsageTotal):
        return ChatCompletionChunk(
            **envelope,
            choices=(),
            usage=CompletionUsage(
                prompt_tokens=event.usage.prompt_tokens,
                completion_tokens=event.usage.completion_tokens,
                total_tokens=event.usage.total_tokens,
            ),
        )
    # Spelled out rather than left as a fall-through: `assert_never` makes mypy
    # fail this function the day `StreamEvent` grows a sixth member, instead of
    # letting the new event be rendered as whatever the last branch happened to
    # build. No request can reach it - the union is closed and every member is
    # handled above.
    assert_never(event)


def sse_frames(
    first: StreamEvent,
    events: Iterator[StreamEvent],
    alias: str,
    completion_id: str,
    on_usage: Callable[[Usage], None] | None = None,
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
    """
    created = int(time.time())
    try:
        for event in chain((first,), events):
            if isinstance(event, UsageTotal) and on_usage is not None:
                on_usage(event.usage)
            chunk = to_chunk(event, completion_id=completion_id, created=created, alias=alias)
            yield _frame(chunk.model_dump_json())
    except GatewayError as error:
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
    yield _frame(DONE_SENTINEL)


router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
def create_chat_completion(
    request: Request,
    context: Context,
    body: ChatCompletionRequest,
    response: Response,
) -> ChatCompletion | StreamingResponse:
    """Resolve the alias, call the first allowed healthy provider, answer OpenAI-shaped.

    A provider with no adapter - its key unset, or its kind not yet built - is
    invisible here, through the resolver's own health seam rather than a second
    unavailability path. So a request whose only allowed
    candidate has no adapter is the resolver's 503 `ALL_PROVIDERS_FAILED`
    (`test_no_adapter_for_any_candidate_is_503`), which is the path an earlier change's
    circuit breaker fills in.

    The first candidate is called and its failure is returned. There is no
    fallback in this plan: `ProviderError` carries `retryable` for an earlier change to
    read, and until then a provider failure is one 502
    (`test_provider_failure_is_502_without_the_provider_message`).

    A streamed request takes the same path to the same provider and differs only
    in what it is handed back. The resolution happens first either way, so an
    unknown alias, a residency refusal and a candidate with no adapter are the
    ordinary typed errors on both
    (`test_a_routing_failure_on_a_stream_is_the_ordinary_typed_error`).

    `response_model=None` because the return type is a union: FastAPI would
    otherwise try to validate a `StreamingResponse` against `ChatCompletion`. The
    non-streaming body is a built model either way, so nothing is left unchecked.

    Declared `def` rather than `async def` on purpose. The adapters are
    synchronous `httpx` clients, so FastAPI runs this in a worker thread; an
    `async def` would block the event loop for the whole provider call, and would
    hold it for the whole length of a stream.
    """
    store: PolicyStore = request.app.state.policy
    adapters: Mapping[str, Adapter] = request.app.state.adapters
    scope = request.scope

    with stage_span(scope, ROUTE_SPAN):
        candidates = resolve(
            store.policy,
            context.tenant,
            body.model,
            context.data_class.value,
            is_healthy=lambda provider: provider in adapters,
        )
        chosen = candidates[0]
    annotate(scope, resolution_attributes(body.model, chosen))
    label_resolution(
        scope,
        alias=body.model,
        provider=chosen.provider,
        region=chosen.region,
        cache=CACHE_STATUS_MISS,
    )
    adapter = adapters[chosen.provider]

    outbound, redacted = _redacted(scope, store.policy, context, body)
    annotate(scope, SpanAttributes(redacted=redacted))
    headers = gateway_headers(chosen, context.trace_id, redacted=redacted)

    with stage_span(scope, provider_span_name(chosen.provider)):
        if body.stream:
            return _streamed(
                adapter, outbound, chosen, headers, _token_sink(context.tenant, chosen)
            )
        answer = adapter.chat(to_neutral(outbound, chosen.model))

    record_tokens(
        tenant=context.tenant,
        provider=chosen.provider,
        model=chosen.model,
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
    )
    response.headers.update(headers)
    return to_wire(answer, body.model)


def _token_sink(tenant: str, chosen: Resolved) -> Callable[[Usage], None]:
    """The callback a stream books its tokens through when the usage chunk arrives.

    A stream reports its usage in a `UsageTotal` event, delivered from inside the
    frame generator long after this route returned - so the labels the counter
    needs have to be captured here, where they are known, rather than looked up
    there. It is a closure over the resolver's answer for the same reason the
    non-streamed call above passes `chosen.model`: the model label is the resolved
    id, never the alias and never what the provider called it
    (`test_tokens_are_recorded_from_the_streams_usage_chunk`,
    `test_the_token_model_label_is_the_resolved_model_not_the_alias`).
    """

    def sink(usage: Usage) -> None:
        record_tokens(
            tenant=tenant,
            provider=chosen.provider,
            model=chosen.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    return sink


def _redacted(
    scope: Scope, policy: Policy, context: RequestContext, body: ChatCompletionRequest
) -> tuple[ChatCompletionRequest, bool]:
    """The request the adapter should see, and whether anything was actually replaced.

    Returns the body unchanged and `False` when the tenant's rule for this data
    class does not ask for redaction, which is what keeps a `public` caller from
    getting placeholders it never asked for and keeps the 16384-character cap off
    a route that had none (`test_the_redaction_cap_does_not_apply_on_a_non_redact_class`).

    Every turn goes through ONE `redact_or_reject` call, so the numbering is
    shared across the conversation: the same address in turn 0 and turn 2 is one
    placeholder, which is what #19 means by stable within a request
    (`test_every_chat_message_is_redacted_under_one_numbering`).

    A turn with no `content` (an assistant turn that is only tool calls) keeps its
    `None` rather than becoming an empty string: `_neutral_messages` reads the two
    the same way, but a body rebuilt with the wrong one would be a silent shape
    change on the way to an earlier change's admin surface.
    """
    if not redaction_applies(policy, context.tenant, context.data_class):
        return body, False

    with stage_span(scope, REDACT_SPAN):
        batch = redact_or_reject(
            tuple(message.content or "" for message in body.messages), param="messages"
        )
    record_redactions(tenant=context.tenant, counts=batch.counts)
    rebuilt = tuple(
        message.model_copy(update={"content": content if message.content is not None else None})
        for message, content in zip(body.messages, batch.texts, strict=True)
    )
    return body.model_copy(update={"messages": rebuilt}), batch.redacted


def _streamed(
    adapter: Adapter,
    body: ChatCompletionRequest,
    chosen: Resolved,
    headers: dict[str, str],
    on_usage: Callable[[Usage], None],
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
        sse_frames(first, events, body.model, _completion_id(), on_usage=on_usage),
        media_type=SSE_MEDIA_TYPE,
        headers={**headers, **SSE_HEADERS},
    )
