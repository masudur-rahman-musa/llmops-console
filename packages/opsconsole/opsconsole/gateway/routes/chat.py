"""`POST /v1/chat/completions`: the OpenAI wire shape, and the join it makes.

An earlier change left three things that did not touch each other: a request could be
authenticated, resolved to a provider and a model id, and a provider could be
called. This module is that join and nothing else - no cache, no ledger, no
redaction, no fallback, no streaming.

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
* `stream: true` is refused with a 422 rather than ignored. Streaming is plan
  05b; until it lands, a client that asked for events must not be handed one
  complete JSON body it will mis-parse (`test_stream_true_is_422_validation`).

Every rejection the wire models make is a 422 `VALIDATION` raised before any
outbound call, by construction: FastAPI validates the body before the route body
runs. The routing and provider failures - 404, 451, 503, 502 - are raised inside
the route by `resolve()` and the adapter, and reach the client through the
handlers in `gateway/errors.py`.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opsconsole.gateway.context import Context
from opsconsole.providers.base import (
    Adapter,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from opsconsole.routing.policy import PolicyStore
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

REDACTED_FALSE = "false"
"""`X-Gateway-Redacted` on every response here. Redaction lands in an earlier change; until
then nothing is redacted and the header says so."""


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

    @model_validator(mode="after")
    def _supported(self) -> ChatCompletionRequest:
        """Refuse the two requests this plan cannot serve honestly."""
        if self.stream:
            raise ValueError("stream: streaming is not supported yet")
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


def gateway_headers(resolved: Resolved, trace_id: str) -> dict[str, str]:
    """The six `X-Gateway-*` headers describing one served request (#9).

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
        "X-Gateway-Redacted": REDACTED_FALSE,
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


router = APIRouter()


@router.post("/v1/chat/completions")
def create_chat_completion(
    request: Request,
    context: Context,
    body: ChatCompletionRequest,
    response: Response,
) -> ChatCompletion:
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

    Declared `def` rather than `async def` on purpose. The adapters are
    synchronous `httpx` clients, so FastAPI runs this in a worker thread; an
    `async def` would block the event loop for the whole provider call.
    """
    store: PolicyStore = request.app.state.policy
    adapters: Mapping[str, Adapter] = request.app.state.adapters

    candidates = resolve(
        store.policy,
        context.tenant,
        body.model,
        context.data_class.value,
        is_healthy=lambda provider: provider in adapters,
    )
    chosen = candidates[0]

    answer = adapters[chosen.provider].chat(to_neutral(body, chosen.model))

    response.headers.update(gateway_headers(chosen, context.trace_id))
    return to_wire(answer, body.model)
