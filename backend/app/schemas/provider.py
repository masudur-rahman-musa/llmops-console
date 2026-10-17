"""The gateway's own request and response shapes: the neutral provider contract.

These models are the vocabulary every part of the gateway uses to talk to and
about provider adapters. They are defined here, in `schemas/`, because they are
shared across `providers/`, `services/`, `cache/` and `limits/` — the same
reason every other schema lives under `schemas/` rather than inside the module
that first needed it.

`providers/base.py` imports and re-exports everything here for the callers that
reached it from there before this move; new code should import directly from
`app.schemas.provider`.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool


class _Contract(BaseModel):
    """Frozen, and unknown fields rejected. See `providers/base.py` for why."""

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

    Four roles, closed. `tool` is the result of a
    call the model asked for, and it carries `tool_call_id` to say which call it
    answers; an `assistant` turn carries `tool_calls` when the model's answer was
    a call rather than words, so `content` defaults to empty rather than being
    required. Which
    combinations are coherent is settled at the API boundary, in
    `schemas/chat.py`, where an incoherent one is a 422 the caller can
    read; the ids themselves are checked by the adapter, before any outbound
    call.
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

    `strict` is the caller's own flag, `None` when they left it out. A wire with
    a strict mode forwards it as given and never adds it. `StrictBool`,
    so `"yes"` is refused rather than read as true.
    """

    name: str
    json_schema: Mapping[str, Any]
    strict: StrictBool | None = None


class ToolChoice(_Contract):
    """How freely the model may choose whether to call a tool.

    Four modes, closed, in the gateway's own words: `auto` leaves it to the model,
    `none` forbids a call, `required` demands one of the offered tools, and
    `tool` demands the one `name` names. `required` is deliberately not spelled
    the way any provider spells it - Anthropic calls it `any` - so that word
    stays inside the adapter.

    `name` is meaningful only with `mode="tool"` and is not required by the
    model: it is caller-supplied and becomes an identifier on the outbound call,
    so the adapter runs it through `checked_tool_name`. A `tool` mode
    with no name at all fails the same check as a 422 before anything is sent.
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
    open. The role is sent on an ordinary text answer and on a tool-only one,
    which is the case a lazily emitted role would break.
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
    which is what an OpenAI-compatible client assembles by.
    """

    index: int
    call: ToolCall


class FinishReason(_Contract):
    """Why the answer ended, in the provider's own word.

    Mapped into OpenAI's vocabulary at the API boundary, through the same table
    the non-streaming path uses, so a
    provider's spelling stops at the adapter here as it does everywhere else.
    """

    reason: str


class UsageTotal(_Contract):
    """What the streamed call consumed. Always sent, and never zero-filled.

    A stream carries usage exactly like a completion does: the ledger
    prices from these numbers, so an adapter that cannot read them raises rather
    than defaulting to zero and booking a real, billed call as free.
    """

    usage: Usage


StreamEvent = StreamStart | TextDelta | ToolCallDelta | FinishReason | UsageTotal
"""Everything a streamed answer is made of, in the gateway's own vocabulary.

Five events, closed. No SSE and no provider event name reaches past an adapter,
which is the same boundary `ChatResponse` draws for the non-streaming path.
"""


# --- embeddings ---------------------------------------------------------------
#
# A contract of its own beside the chat one, for the same reason `EmbeddingAdapter`
# is a protocol of its own: an embedding request shares no field with a completion
# beyond the model id, and folding the two together would mean a `ChatRequest`
# with a `messages` nobody fills.


class EmbeddingRequest(_Contract):
    """One embedding request, already resolved to a provider model id.

    `model` is the provider's own id from the routing policy, never an alias, as
    on `ChatRequest`: the resolver has already run by the time an adapter sees
    this.

    `input` is always a tuple, even for a caller who sent a bare string. The
    published wire accepts either, and an adapter handed a `str` would iterate it
    character by character somewhere below - one vector per letter, no error.

    There is deliberately no `encoding_format` here. The gateway renders base64
    itself from the float vector and always asks the provider for floats, so the field
    stops at the API boundary and this contract gives it nowhere to go.
    """

    model: str
    input: tuple[str, ...]


class EmbeddingUsage(_Contract):
    """What an embedding call consumed. Two counts, not the three of `Usage`.

    An embedding has no completion tokens, on this wire or in the response this
    product publishes, so reusing `Usage` could only mean writing a zero into a
    field with no meaning - and a zero-filled count is exactly what the rule
    against zero-filling usage exists to prevent, hidden behind a field that
    always looks populated.
    """

    prompt_tokens: int
    total_tokens: int


class Embedding(_Contract):
    """One vector, and the position of the input it was computed from.

    `index` is carried rather than implied by tuple position because it is the
    caller's only way back to its own rows: a provider that serialises its
    answers out of order is reordered by this field, and one that numbers them in
    any way other than `0..n-1` is refused.
    """

    index: int
    vector: tuple[float, ...]


class EmbeddingResponse(_Contract):
    """One embedding call's answer. `embeddings` is ordered, `index` 0..n-1."""

    model: str
    embeddings: tuple[Embedding, ...]
    usage: EmbeddingUsage
