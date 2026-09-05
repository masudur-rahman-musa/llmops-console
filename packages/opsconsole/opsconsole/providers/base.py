"""The provider-neutral adapter contract.

Everything a provider adapter takes and returns is defined here, in the
gateway's own vocabulary. The models are OpenAI-shaped because that is the API
this product publishes, but they are ours: no provider SDK is imported, and no
provider's field names appear, so `providers/openai_compatible.py` implements it unchanged
rather than translating into one provider's dialect, and the OpenAI and Azure
adapters of an earlier change are meant to do the same. `test_base_contract_carries_no_provider_vocabulary`
holds that by walking this module's names and every model's fields.

The models are frozen and reject unknown fields. Frozen because an adapter must
not mutate the caller's request while mapping it - an earlier change retries the same
request against a second provider, and it has to be the same request; unknown
fields rejected because a misspelled `max_token` should be an error rather than
a silently applied default. Both are pinned by
`test_contract_models_are_frozen` and `test_contract_models_forbid_unknown_fields`.

`embed()` is deliberately absent: Anthropic has no embeddings endpoint, and a
method that exists only to raise is worse than a second protocol. An earlier change adds
an embedding protocol next to this one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from opsconsole.routing.errors import IdentifierInvalid

DEFAULT_MAX_TOKENS = 4096
"""What `max_tokens` becomes when the caller omits it.

Anthropic requires the field, so there has to be a default; 4096 is large enough
that it is not felt as a cap on an ordinary answer. Pinned exactly by
`test_default_max_tokens_is_exactly_4096`.
"""

MAX_TOOL_NAME_LENGTH = 128
"""Longest tool name accepted. Pinned by `test_max_tool_name_length_is_exactly_128`."""

TOOL_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,128}")
"""The character set a tool name may use, matched with `re.fullmatch`.

`.match` with `^...$` would accept a trailing newline, and this string is sent
to a provider as the identifier the model calls back by name. The rejection
table in `test_tool_name_rejection_table` covers a trailing newline, a trailing
CRLF and a trailing space; the accepted set (uppercase included) is held by
`test_tool_name_acceptance_table`.
"""


def checked_tool_name(value: str, field: str) -> str:
    r"""Validate one caller-supplied identifier before it reaches a provider.

    Named for its first caller and now used for every identifier in a tool loop.
    `grep -rnE "checked_tool_name\(" packages/` returns 13 lines as of an earlier change:
    this definition and TWELVE call sites, the same six in each of the two
    adapters that build an outbound body - `providers/anthropic.py` and
    `providers/openai_compatible.py`. The six are a tool call's id and its name,
    the `tool_call_id` on a result turn, a `tools[].name`, a forced
    `tool_choice.name` and a `response_format.name`. They share a cap and a
    character set because they are the same kind of string - one the model calls
    back by name - and one rule is one place to change it.

    Raises `IdentifierInvalid` (code VALIDATION, status 422) naming `field`, so a
    bad value is the caller's error with a pointer to it rather than a provider
    400 the caller cannot act on. Adapters call this before the outbound request
    is built, which `test_bad_tool_name_raises_before_any_outbound_call` and
    `test_bad_tool_ids_raise_before_any_outbound_call` prove by asserting no
    request was made.
    """
    if len(value) > MAX_TOOL_NAME_LENGTH:
        raise IdentifierInvalid(
            f"{field}: longer than {MAX_TOOL_NAME_LENGTH} characters", param=field
        )
    if TOOL_NAME_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: must match {TOOL_NAME_PATTERN.pattern}", param=field)
    return value


class _Contract(BaseModel):
    """Frozen, and unknown fields rejected. See the module docstring for why."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolCall(_Contract):
    """A tool the model asked to call. `arguments` is a JSON string, as OpenAI publishes it.

    Defined above `ChatMessage` because an assistant turn carries these when it is
    replayed back to a provider: the request half of a tool loop.
    """

    id: str
    name: str
    arguments: str


class ChatMessage(_Contract):
    """One turn. `system` is a role here; adapters place it wherever their provider wants it.

    Four roles, closed (`test_role_set_stays_closed`). `tool` is the result of a
    call the model asked for, and it carries `tool_call_id` to say which call it
    answers; an `assistant` turn carries `tool_calls` when the model's answer was
    a call rather than words, so `content` defaults to empty rather than being
    required (`test_a_tool_call_only_assistant_turn_needs_no_content`). Which
    combinations are coherent is settled at the API boundary, in
    `gateway/routes/chat.py`, where an incoherent one is a 422 the caller can
    read; the ids themselves are checked by the adapter, before any outbound
    call (`test_bad_tool_ids_raise_before_any_outbound_call`).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


class ToolSpec(_Contract):
    """A tool the model may call. `parameters` is a JSON schema, passed through as data."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = {}


class ResponseFormat(_Contract):
    """A demand that the answer conform to a JSON schema.

    `name` is caller-supplied and becomes an identifier on the outbound call, so
    adapters run it through `checked_tool_name` like any other tool name.
    """

    name: str
    json_schema: Mapping[str, Any]


class ToolChoice(_Contract):
    """How freely the model may choose whether to call a tool.

    Four modes, closed, in the gateway's own words
    (`test_tool_choice_mode_set_stays_closed`): `auto` leaves it to the model,
    `none` forbids a call, `required` demands one of the offered tools, and
    `tool` demands the one `name` names. `required` is deliberately not spelled
    the way any provider spells it - Anthropic calls it `any` - so that word
    stays inside the adapter.

    `name` is meaningful only with `mode="tool"` and is not required by the
    model: it is caller-supplied and becomes an identifier on the outbound call,
    so the adapter runs it through `checked_tool_name`
    (`test_tool_choice_name_is_checked_before_any_outbound_call`). A `tool` mode
    with no name at all fails the same check as a 422 before anything is sent
    (`test_a_named_tool_choice_with_no_name_raises_before_any_outbound_call`).
    """

    mode: Literal["auto", "none", "required", "tool"]
    name: str | None = None


class ChatRequest(_Contract):
    """One completion request, already resolved to a provider model id.

    `model` is the provider's own id from the routing policy, never an alias: the
    resolver has already run by the time an adapter sees this.
    """

    model: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: ResponseFormat | None = None


class Usage(_Contract):
    """What the call consumed. The ledger prices from these three numbers."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(_Contract):
    """One completion. `content` is empty rather than absent when the answer is only tool calls."""

    model: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str
    usage: Usage


class StreamStart(_Contract):
    """The answer has begun. Carries the role and nothing else.

    A separate event rather than a flag on the first text delta, because an
    answer that is only a tool call has no first text delta and still has to
    open. `test_first_chunk_carries_the_assistant_role` pins the role on an
    ordinary text answer and `test_streamed_tool_call_is_one_whole_delta` pins it
    on a tool-only one, which is the case a lazily emitted role would break.
    """

    role: Literal["assistant"] = "assistant"


class TextDelta(_Contract):
    """One fragment of the answer, in the order the provider produced it."""

    text: str


class ToolCallDelta(_Contract):
    """One COMPLETE tool call, delivered whole rather than as argument fragments.

    Partial JSON is unusable to a client until the call closes, and forwarding
    fragments would mean owning a second incremental-arguments shape in the wire
    mapping. The cost is recorded: a client cannot render
    arguments as they are typed. `index` is the call's position in the answer,
    which is what an OpenAI-compatible client assembles by
    (`test_streamed_tool_call_is_one_whole_delta`).
    """

    index: int
    call: ToolCall


class FinishReason(_Contract):
    """Why the answer ended, in the provider's own word.

    Mapped into OpenAI's vocabulary at the API boundary, through the same table
    the non-streaming path uses (`test_finish_reason_table_on_the_stream`), so a
    provider's spelling stops at the adapter here as it does everywhere else.
    """

    reason: str


class UsageTotal(_Contract):
    """What the streamed call consumed. Always sent, and never zero-filled.

    A stream carries usage exactly like a completion does (the design): the ledger
    prices from these numbers, so an adapter that cannot read them raises rather
    than defaulting to zero and booking a real, billed call as free
    (`test_a_truncated_stream_raises_rather_than_zero_filling_usage`).
    """

    usage: Usage


StreamEvent = StreamStart | TextDelta | ToolCallDelta | FinishReason | UsageTotal
"""Everything a streamed answer is made of, in the gateway's own vocabulary.

Five events, closed. No SSE and no provider event name reaches past an adapter,
which is the same boundary `ChatResponse` draws for the non-streaming path.
"""


@runtime_checkable
class Adapter(Protocol):
    """What every provider adapter offers the gateway.

    `name` is the policy provider name (the key in `routing.yaml`'s `providers`
    and in the factory's mapping), not the vendor: two accounts with the same
    vendor are two providers, and the ledger, the metric label and the resolver
    all have to address them apart. `price_key` is built from it for that reason,
    pinned by `test_price_key_is_provider_name_slash_model`.
    """

    name: str

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run one non-streaming completion. Every failure is a `ProviderError`."""
        ...

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Run one streamed completion. Every failure is a `ProviderError`.

        On the one protocol rather than a separate `StreamingAdapter` (an earlier change
        decision A): every provider in the policy speaks SSE, so a second
        protocol would buy a capability check on every request and a "this
        provider cannot stream" path nothing would reach.
        """
        ...

    def healthcheck(self) -> bool:
        """Whether this provider should be offered traffic."""
        ...

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        ...
