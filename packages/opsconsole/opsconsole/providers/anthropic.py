"""The Anthropic adapter: the Messages API behind the neutral contract.

The mapping and the transport are kept apart, on purpose. `to_anthropic`,
`from_anthropic` and `from_anthropic_events` are pure functions - over
dictionaries for the completion pair, over the decoded SSE lines for the stream -
so every direction is tested against recorded payloads without a transport;
`AnthropicAdapter` adds the HTTP, the fixed timeouts and the retry loop around
them, with `sleep` and `jitter` injected so the retry table costs microseconds to
test and produces exact delays rather than a range. No provider SDK is imported:
the mapping is the thing we own, and it stops here, so no Anthropic field name
and no SSE event name reaches the rest of the gateway.

Every PROVIDER failure that leaves this module is a `ProviderError` carrying
`retryable`, including a timeout and a 200 whose body we cannot read, so plan
10's fallback reads one flag instead of re-deriving a verdict from a status it
can no longer see. The caller's own bad input is the deliberate exception: it
leaves as `IdentifierInvalid` (422) before any call is made, because that is the
caller's error to fix and not a provider outage to fall back from. Seven inputs
reach that path: the six `checked_tool_name` call sites this module has
(`grep -n checked_tool_name providers/anthropic.py`, which also returns the
import and these two lines) - a
tool call's id and name, a `tool_call_id` on a result turn, a `tools[].name`, a
forced `tool_choice.name` and a `response_format.name` - and the one direct
raise below, for tool call `arguments` that will not parse as JSON.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from opsconsole.core.errors import ProviderError
from opsconsole.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageTotal,
    checked_tool_name,
)
from opsconsole.providers.transport import (
    CONNECT_TIMEOUT,
    MAX_RETRIES,
    PROVIDER_MESSAGE_LIMIT,
    READ_TIMEOUT,
    RETRY_BASE_DELAY,
    STREAM_READ_TIMEOUT,
    ProviderTransport,
    unreadable,
)
from opsconsole.routing.errors import IdentifierInvalid

__all__ = [
    # Re-exported from `providers.transport`, which owns the HTTP policy every
    # adapter shares (the design). Named here for one reason only: an earlier change's tests
    # import them from this module, and the an earlier change extraction that moved them
    # was meant to change no behaviour and edit no test
    # (`grep -n "from opsconsole.providers.anthropic import" -A 10 tests/providers/`
    # shows both `test_anthropic_transport.py` and `test_anthropic_streaming.py`
    # reading them here). Nothing in this module uses them.
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_TOOL_CHOICE_BY_MODE",
    "ANTHROPIC_VERSION",
    "CONNECT_TIMEOUT",
    "MAX_RETRIES",
    "MESSAGES_PATH",
    "PROVIDER_MESSAGE_LIMIT",
    "READ_TIMEOUT",
    "RETRY_BASE_DELAY",
    "STREAM_READ_TIMEOUT",
    "AnthropicAdapter",
    "from_anthropic",
    "from_anthropic_events",
    "to_anthropic",
]

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
MESSAGES_PATH = "/v1/messages"

ANTHROPIC_TOOL_CHOICE_BY_MODE = {"auto": "auto", "none": "none", "required": "any"}
"""The neutral `ToolChoice.mode` in Anthropic's own words.

Anthropic spells `required` as `any`, and this table is the only place that
translation happens (`grep -rn ANTHROPIC_TOOL_CHOICE_BY_MODE packages/`), which
is the whole reason the neutral contract keeps its own word for it. The `tool`
mode is absent from the table because it carries a name and is built separately.
Pinned row by row by `test_tool_choice_maps_to_the_anthropic_table`."""


def _tool_use_block(call: ToolCall, field: str) -> dict[str, Any]:
    """Render one tool call the model made, as the block Anthropic replays it as.

    Both ids are checked here rather than at the contract, so an invalid one
    raises before any outbound call is built
    (`test_bad_tool_ids_raise_before_any_outbound_call`). `arguments` is the JSON
    *string* the published API hands its clients, and Anthropic's `input` is an
    object, so it is parsed rather than passed through; a string that will not
    parse is the caller's error and not a provider outage, which is why it leaves
    as `IdentifierInvalid` (`test_tool_call_arguments_that_are_not_json_raise`).
    """
    try:
        arguments = json.loads(call.arguments)
    except ValueError as exc:
        raise IdentifierInvalid(
            f"{field}.arguments: not valid JSON", param=f"{field}.arguments"
        ) from exc
    return {
        "type": "tool_use",
        "id": checked_tool_name(call.id, f"{field}.id"),
        "name": checked_tool_name(call.name, f"{field}.name"),
        "input": arguments,
    }


def _rendered_turn(index: int, message: ChatMessage) -> dict[str, Any]:
    """Render one non-system turn as Anthropic carries it.

    Anthropic has neither a `tool` role nor a `tool_calls` field: a tool RESULT is
    a `tool_result` block inside a **user** turn, and a tool CALL is a `tool_use`
    block inside an assistant turn. That asymmetry is the mapping this function
    exists for (`test_tool_result_turn_renders_as_a_user_turn_in_order`).

    An assistant turn stays a plain string unless it carries calls
    (`test_a_plain_assistant_turn_stays_a_plain_string`), and a turn whose only
    answer was a call gets no text block at all: Anthropic rejects an empty
    `text` (`test_a_tool_call_only_assistant_turn_has_no_text_block`).
    """
    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    # `or ""` rather than a branch: an absent id fails the same
                    # pre-check as a malformed one, with the same 422 naming the
                    # same field (`test_a_tool_turn_without_a_tool_call_id_raises`).
                    "tool_use_id": checked_tool_name(
                        message.tool_call_id or "", f"messages[{index}].tool_call_id"
                    ),
                    "content": message.content,
                }
            ],
        }

    if message.tool_calls:
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        blocks.extend(
            _tool_use_block(call, f"messages[{index}].tool_calls[{position}]")
            for position, call in enumerate(message.tool_calls)
        )
        return {"role": message.role, "content": blocks}

    return {"role": message.role, "content": message.content}


def to_anthropic(request: ChatRequest) -> dict[str, Any]:
    """Render a normalised request as an Anthropic Messages request body.

    Every caller-supplied tool name is checked first, so an invalid one raises
    before any outbound call is built
    (`test_bad_tool_name_raises_before_any_outbound_call`).

    The system turn is lifted out of `messages` into the top-level `system`
    field, because Anthropic has no `system` role and rejects one; the remaining
    turns keep the caller's order. Several system messages join with a blank
    line, which is the only reading that neither drops one nor runs two
    instructions together.
    """
    tools: list[dict[str, Any]] = [
        {
            "name": checked_tool_name(tool.name, f"tools[{index}].name"),
            "description": tool.description,
            "input_schema": dict(tool.parameters),
        }
        for index, tool in enumerate(request.tools)
    ]

    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS,
        "messages": [
            _rendered_turn(index, message)
            for index, message in enumerate(request.messages)
            if message.role != "system"
        ],
    }

    system = "\n\n".join(
        message.content for message in request.messages if message.role == "system"
    )
    if system:
        body["system"] = system

    if request.temperature is not None:
        # `is not None`, not truthiness: an explicit temperature of 0 is the
        # setting an eval run depends on, and dropping it would change the answer.
        body["temperature"] = request.temperature

    if request.tool_choice is not None:
        choice = request.tool_choice
        if choice.mode == "tool":
            # `or ""` for the same reason as the tool result id above: a `tool`
            # mode with no name fails the pre-check rather than sending `null`.
            body["tool_choice"] = {
                "type": "tool",
                "name": checked_tool_name(choice.name or "", "tool_choice.name"),
            }
        else:
            body["tool_choice"] = {"type": ANTHROPIC_TOOL_CHOICE_BY_MODE[choice.mode]}

    if request.response_format is not None:
        # A JSON schema becomes one forced tool. Anthropic has no response_format,
        # and a tool the model is required to call is the mechanism that makes the
        # schema binding rather than a suggestion the prose can ignore.
        # Applied after `tool_choice` on purpose, so the schema demand wins if
        # both are somehow set: the forced call is what makes a schema binding,
        # and a plain `auto` overriding it would leave the model free to answer
        # in prose (`test_response_format_wins_over_tool_choice`). The API
        # boundary refuses the pair outright, so this is only the tie-break.
        forced = checked_tool_name(request.response_format.name, "response_format.name")
        tools.append(
            {
                "name": forced,
                "description": "Return the answer as this schema.",
                "input_schema": dict(request.response_format.json_schema),
            }
        )
        body["tool_choice"] = {"type": "tool", "name": forced}

    if tools:
        # Omitted entirely rather than sent empty: an empty `tools` is a field
        # Anthropic validates, not one it ignores.
        body["tools"] = tools

    return body


def from_anthropic(payload: Mapping[str, Any]) -> ChatResponse:
    """Read an Anthropic Messages response into the normalised response.

    Text blocks concatenate into `content` and `tool_use` blocks become
    `tool_calls` whose `arguments` is a JSON *string*, which is the shape the
    published OpenAI-compatible API promises its clients. A block of any other
    type is passed over rather than guessed at, and an answer that leaves nothing
    behind - no text and no tool call - is a `ProviderError` rather than an empty
    string a tenant is billed for (`test_an_unknown_block_type_is_ignored` and
    `test_an_answer_with_no_readable_block_raises`).

    Usage is required. Anthropic's `input_tokens` and `output_tokens` become
    `prompt_tokens` and `completion_tokens` with `total_tokens` their sum, and a
    response missing any of it raises rather than defaulting to zero: the ledger
    prices from these numbers, so a silent zero books a real, billed call as free
    (`test_a_response_without_usage_raises_provider_error`).
    """
    try:
        text: list[str] = []
        calls: list[ToolCall] = []
        for block in payload["content"]:
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=json.dumps(block["input"], sort_keys=True),
                    )
                )
            # Any other block type is deliberately not rendered. Anthropic adds
            # block types over time (`thinking` is one), and a block we do not
            # know is not one we can claim is answer text. Dropping it silently
            # would be the same failure as a zero usage, so the guard below
            # refuses an answer that ends up carrying nothing at all.

        usage = payload["usage"]
        prompt_tokens = usage["input_tokens"]
        completion_tokens = usage["output_tokens"]

        content = "".join(text)
        if not content and not calls:
            # Not `transport.unreadable`: the body parsed fine, it simply carried nothing
            # this adapter can render. Saying so is what stops an operator
            # chasing a parse bug that is not there.
            raise ProviderError(
                "the provider answered with no readable content block",
                retryable=False,
                provider_message=(
                    f"stop_reason={payload.get('stop_reason')!r}, "
                    f"block types={[block.get('type') for block in payload['content']]}"
                ),
            )

        return ChatResponse(
            model=payload["model"],
            content=content,
            tool_calls=tuple(calls),
            finish_reason=payload["stop_reason"],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise unreadable(exc) from exc


@dataclass
class _OpenToolBlock:
    """One `tool_use` block being assembled, held until its `content_block_stop`.

    Kept in a dict keyed by the block's own index rather than in a single "the
    open block" variable, because Anthropic may have more than one block open at
    a time: a fragment names the index it belongs to, and matching on that is
    what stops one call's JSON being appended to another's
    (`test_interleaved_tool_fragments_stay_with_the_block_that_names_them`).
    """

    id: str
    name: str
    fragments: list[str]


SSE_DATA_PREFIX = "data: "
"""The only SSE field this reader looks at.

An `event:` line names what the `data:` line beneath it already carries in its
`type`, so reading both would be two sources for one fact; comments, `id:` lines
and the blank separators are skipped for the same reason a `ping` is
(`test_the_reader_yields_the_start_then_the_text_then_the_finish_and_usage_last`)."""


def _sse_payload(line: str) -> dict[str, Any] | None:
    """The JSON object on one `data:` line, or `None` for a line that carries none."""
    if not line.startswith(SSE_DATA_PREFIX):
        return None
    payload: Any = json.loads(line[len(SSE_DATA_PREFIX) :])
    if not isinstance(payload, dict):
        raise TypeError(f"an SSE data line carried {type(payload).__name__}, not an object")
    return payload


def from_anthropic_events(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Read Anthropic's Messages SSE lines into the neutral stream events.

    Pure, and pure on purpose: it takes the decoded lines `iter_lines()` yields
    and never touches a transport, so every shape it has to handle is a recorded
    `.sse` fixture rather than a mocked socket. It is the streaming twin of
    `from_anthropic`, and it draws the same boundary - no Anthropic event name
    leaves this module.

    Text arrives as `text_delta` fragments and is forwarded in the provider's
    order. A `tool_use` block is ASSEMBLED: its `input_json_delta` fragments are
    buffered and emitted as one whole `ToolCallDelta` when the block closes,
    because `{"city": "Ber` is not JSON any client can parse (an earlier change decision
    B, `test_the_reader_assembles_a_tool_use_block_from_its_json_fragments`).
    The fragments are concatenated rather than parsed and re-serialised, so what
    the model wrote is what the client is handed.

    Usage is split across two events - `input_tokens` on `message_start`,
    `output_tokens` on `message_delta` - and both are required, as is the stop
    reason. A stream that ends without them raises rather than yielding a total
    the ledger would price: `message_start` reports an `output_tokens` of its own
    that is not the answer's, so zero-filling is not the only wrong number
    available (`test_a_truncated_stream_raises_rather_than_zero_filling_usage`).
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    stop_reason: str | None = None
    open_tools: dict[int, _OpenToolBlock] = {}
    rendered = False
    seen_blocks: set[str] = set()

    try:
        for line in lines:
            payload = _sse_payload(line)
            if payload is None:
                continue
            kind = payload.get("type")

            if kind == "message_start":
                prompt_tokens = payload["message"]["usage"]["input_tokens"]
                yield StreamStart()
            elif kind == "content_block_start":
                block = payload["content_block"]
                seen_blocks.add(block["type"])
                if block["type"] == "tool_use":
                    open_tools[payload["index"]] = _OpenToolBlock(
                        id=block["id"], name=block["name"], fragments=[]
                    )
            elif kind == "content_block_delta":
                delta = payload["delta"]
                if delta["type"] == "text_delta":
                    rendered = True
                    yield TextDelta(text=delta["text"])
                elif delta["type"] == "input_json_delta":
                    fragment_of = open_tools.get(payload["index"])
                    if fragment_of is not None:
                        fragment_of.fragments.append(delta["partial_json"])
            elif kind == "content_block_stop":
                closed = open_tools.pop(payload["index"], None)
                if closed is not None:
                    rendered = True
                    yield ToolCallDelta(
                        index=payload["index"],
                        call=ToolCall(
                            id=closed.id,
                            name=closed.name,
                            arguments="".join(closed.fragments),
                        ),
                    )
            elif kind == "message_delta":
                stop_reason = payload["delta"]["stop_reason"]
                completion_tokens = payload["usage"]["output_tokens"]
            # Every other event - `ping`, `message_stop`, and whatever Anthropic
            # adds next - is passed over. An event we do not know is not one we
            # can claim is answer text, which is the same rule `from_anthropic`
            # applies to an unknown content block.
    except (KeyError, TypeError, ValueError) as exc:
        raise unreadable(exc) from exc

    if stop_reason is None or prompt_tokens is None or completion_tokens is None:
        raise ProviderError(
            "the provider ended the stream before it finished",
            retryable=False,
            provider_message=(
                f"stop_reason={stop_reason!r}, input_tokens={prompt_tokens!r}, "
                f"output_tokens={completion_tokens!r}"
            ),
        )

    if not rendered:
        # The streaming twin of `from_anthropic`'s no-readable-block guard, and
        # it exists for the same reason: a `thinking`-only answer renders nothing
        # this adapter can hand a client, and finishing the stream with a usage
        # chunk would bill the tenant for an empty answer. Refusing it on one
        # path and serving it on the other would be two behaviours for one
        # provider reply (`test_a_stream_with_no_readable_block_raises`).
        #
        # Second, not first: a stream cut off before its first text delta
        # renders nothing EITHER, and reporting that as an empty answer would
        # point an operator at a content bug instead of the dropped connection
        # it was (`test_a_stream_cut_off_before_any_content_reports_the_truncation`).
        raise ProviderError(
            "the provider streamed no readable content block",
            retryable=False,
            provider_message=f"stop_reason={stop_reason!r}, block types={sorted(seen_blocks)}",
        )

    yield FinishReason(reason=stop_reason)
    yield UsageTotal(
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    )


class AnthropicAdapter:
    """One Anthropic account, addressed by its policy provider name.

    `name` is that policy name rather than the vendor, so two Anthropic accounts
    in one policy stay distinguishable everywhere they are counted; `price_key`
    is built from it (`test_price_key_is_provider_name_slash_model`).

    The key travels only in the `x-api-key` header
    (`test_the_key_travels_only_in_the_x_api_key_header`). This module logs
    nothing at all; what reaches a log on a failure is
    `ProviderError.provider_message`, which holds the provider's answer or the
    transport's own exception text, never the request body or its headers.
    """

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str = ANTHROPIC_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.name = name
        self.http = ProviderTransport(
            name=name,
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            transport=transport,
            sleep=sleep,
            jitter=jitter,
        )

    @property
    def client(self) -> httpx.Client:
        """The underlying HTTP client, so the fixed timeouts stay observable here.

        Asserted on by `test_client_timeouts_are_fixed_at_five_and_sixty` and
        `test_close_releases_the_client`.
        """
        return self.http.client

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run one non-streaming completion."""
        body = to_anthropic(request)
        response = self.http.send(MESSAGES_PATH, body)
        try:
            payload = response.json()
        except ValueError as exc:
            # A 200 that is not JSON at all - a proxy's HTML error page, a
            # truncated body. Pinned by `test_a_non_json_success_body_is_a_provider_error`.
            raise unreadable(exc) from exc
        return from_anthropic(payload)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Run one streamed completion, yielding neutral events as they arrive.

        The open and the read are deliberately separate.
        `ProviderTransport.open_stream` owns the retry loop, so the retry cannot
        reach around the iteration: once one event has been yielded the client
        has already been shown part of an answer, and nothing that cannot be
        un-sent is resent (an earlier change decision C,
        `test_stream_open_retries_only_before_the_first_event`).

        A transport failure part way through the body is wrapped here rather than
        escaping raw, so the module's promise holds on this path too - every
        provider failure that leaves it is a `ProviderError` carrying `retryable`
        (`test_a_transport_failure_mid_stream_is_a_provider_error_and_is_not_retried`).
        """
        body = to_anthropic(request)
        # Copied rather than mutated in place: `to_anthropic` is the shape the
        # non-streaming path sends, and the one difference belongs to this call.
        streamed = {**body, "stream": True}
        with self.http.open_stream(MESSAGES_PATH, streamed) as response:
            try:
                yield from from_anthropic_events(response.iter_lines())
            except httpx.RequestError as exc:
                raise self.http.transport_error(exc) from exc

    def healthcheck(self) -> bool:
        """Always true, and it calls nothing.

        The seam is still unfilled, and naming what filled it elsewhere would be
        wrong: `providers/breaker.py` learns from the outcomes of REAL requests,
        through the routes' `is_healthy`, and calls this method nowhere. So the
        `provider` readiness probe still proves only that an adapter was built.

        Probing the provider here would put a network call - and a live key in
        CI - on the readiness path, which
        `test_healthcheck_is_true_without_calling_the_provider` forbids.
        """
        return True

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        return f"{self.name}/{model}"

    def close(self) -> None:
        """Release the connection pool."""
        self.http.close()
