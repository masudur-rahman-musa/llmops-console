"""Mapping between the OpenAI wire shape and the neutral provider contract."""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping
from typing import Any, Literal, assert_never

from app.core.http_errors import as_gateway_error
from app.core.validation import IdentifierInvalid
from app.providers.base import (
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
    UsageTotal,
)
from app.redaction import MAP_ID_PATTERN, RedactionMapStore, RedactionMapStoreUnavailable, rehydrate
from app.schemas.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    ChunkDelta,
    CompletionMessage,
    CompletionUsage,
    NamedToolChoice,
    WireMessage,
    WireToolCall,
    WireToolCallChunk,
    WireToolCallFunction,
)

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


REHYDRATE_HEADER = "X-Gateway-Rehydrate"
"""The request header a caller puts a `redaction_map_id` in to ask for its values back.

A request header rather than a body field (decision A): the body is the OpenAI
wire shape with `extra="forbid"`, so a gateway control there would be a field
every OpenAI-shaped client had to be taught, sitting next to `model` as if the
provider understood it. Read on this route and nowhere else (decision F) - it
means nothing on `/v1/redact` or on embeddings, and parsing it in the shared
`context` dependency would 422 requests that never rehydrate.

Its ABSENCE is what "off by default" means: there is no flag and no value that
means off; `tests/gateway/test_rehydration.py` covers that in
`test_no_header_answers_exactly_what_it_answers_today`, and the spelling is
pinned by every case in that module, which sends this exact name."""


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
                strict=body.response_format.json_schema.strict,
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


def requested_mapping(
    store: RedactionMapStore, tenant: str, header: str | None
) -> Mapping[str, str] | None:
    """The map this request asked to rehydrate through, or `None` when it asked for none.

    Everything here happens BEFORE a provider is dialled, which is the point: a
    caller holding a stale id pays for no tokens, and every refusal below is
    asserted against an adapter that recorded zero calls
    (`test_a_malformed_map_id_is_422_with_no_outbound_call`,
    `test_an_unknown_map_id_is_422_with_no_outbound_call`,
    `test_an_unreachable_map_store_is_502_with_no_outbound_call`).

    The header is tested against `None`, never for truthiness: an empty value is a
    caller that sent the header with nothing in it, and that is a malformed id
    rather than a request that never asked (the empty row of
    `test_the_map_id_rejection_table`).

    The answer is the same on a streamed request as on a returned one: an earlier change
    gave the frame renderer a boundary-safe buffer, so `stream: true` with the
    header no longer refuses and each refusal below is re-asserted on a streamed
    body (`test_a_malformed_map_id_on_a_stream_is_422_with_no_outbound_call`,
    `test_an_unknown_map_id_on_a_stream_is_422_with_no_outbound_call`,
    `test_an_id_live_for_another_tenant_is_absent_on_a_stream`,
    `test_an_unreachable_map_store_on_a_stream_is_502_with_no_outbound_call`).

    Three refusals, in the order they are cheapest to make:

    * A malformed id is 422 before the store is asked, against the store's own
      `MAP_ID_PATTERN` rather than a second copy of the rule.
      `fullmatch`, because `$` in Python also matches before a trailing newline; the inclusive boundary and the whitespace rows are
      `test_a_thirty_two_character_hex_id_is_accepted` and
      `test_the_map_id_rejection_table`.
    * An unreachable store is a 502 `PROVIDER_ERROR` in this function's OWN
      words, exactly as `filed_map_id` renders the write side: the client's text
      names an internal host and port and stays in the chained cause
      (`test_a_map_store_failure_does_not_put_the_clients_words_in_front_of_a_caller`).
    * An id with no live map is 422 (decision B). The map is read for the
      AUTHENTICATED tenant, so an id live for another tenant is simply absent and
      answers the identical body - a message naming the id, or saying whose it
      was, would hand a caller an oracle for which ids exist
      (`test_an_id_live_for_another_tenant_is_absent`,
      `test_an_unknown_map_id_is_422_with_no_outbound_call`).
    """
    if header is None:
        return None
    if MAP_ID_PATTERN.fullmatch(header) is None:
        raise IdentifierInvalid(f"{REHYDRATE_HEADER}: not a valid map id", param=REHYDRATE_HEADER)
    try:
        mapping = store.get(tenant, header)
    except RedactionMapStoreUnavailable as exc:
        raise as_gateway_error(exc, "the redaction map store is unavailable") from exc
    if mapping is None:
        raise IdentifierInvalid(f"{REHYDRATE_HEADER}: no live map", param=REHYDRATE_HEADER)
    return mapping


def rehydrated(answer: ChatResponse, mapping: Mapping[str, str] | None) -> ChatResponse:
    """The answer with the map's placeholders restored, or the answer itself.

    Applied to the NEUTRAL response before `to_wire` (decision C), because the
    streamed path's `TextDelta` and `ToolCallDelta` are neutral too and one
    substitution covers both; on the wire model the two would need two
    implementations of the same thing. `sse_frames` is where the streamed half
    runs, over a `StreamRehydrator` rather than this function, because a
    placeholder can be split across two frames.

    Both `content` and each tool call's `arguments` (decision D). A tool call
    handed `<EMAIL_1>` is a broken tool call, and the content of an answer whose
    only placeholder sits in a tool call looks entirely correct
    (`test_a_placeholder_in_a_tool_calls_arguments_is_rehydrated`). The call is
    copied rather than rebuilt, so its `id` and `name` are what the provider sent.
    """
    if mapping is None:
        return answer
    return answer.model_copy(
        update={
            "content": rehydrate(answer.content, mapping),
            "tool_calls": tuple(
                call.model_copy(update={"arguments": rehydrate(call.arguments, mapping)})
                for call in answer.tool_calls
            ),
        }
    )
