"""Pydantic wire and response models for /v1/chat/completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

COMPLETION_OBJECT: Literal["chat.completion"] = "chat.completion"
"""The `object` discriminator an OpenAI-compatible client branches on."""


CHUNK_OBJECT: Literal["chat.completion.chunk"] = "chat.completion.chunk"
"""The same discriminator for a streamed frame. A client reads the answer out of
`choices[].delta` rather than `choices[].message` when it sees this."""


class _Wire(BaseModel):
    """A model on the published API boundary: frozen, and unknown fields rejected.

    Rejected rather than ignored: a misspelled `max_token`
    becomes a 422 naming the field instead of a silently applied default.
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
    naming what is wrong.
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
            raise ValueError("content is required unless the turn carries tool_calls")
        return self


class StreamOptions(_Wire):
    """`stream_options: {"include_usage": bool}`, accepted so an ordinary SDK works.

    The OpenAI SDK sends this field on every streamed request, so refusing it
    outright through `extra="forbid"` would break a client that did nothing
    wrong. `false` is refused instead, with a 422 that says why: usage is
    returned on every stream because the ledger prices from it, so
    honouring `false` would be a promise the gateway does not keep.
    """

    include_usage: bool = True


class JsonSchemaSpec(_Wire):
    """The `json_schema` body of a `response_format`.

    `schema` is a Python attribute name Pydantic reserves, so the field is
    `json_schema` here and aliased back to `schema` on the wire.
    """

    name: str
    json_schema: Mapping[str, Any] = Field(alias="schema")
    strict: StrictBool | None = None


class WireResponseFormat(_Wire):
    """`response_format: {"type": "json_schema", "json_schema": {...}}`."""

    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatCompletionRequest(_Wire):
    """One chat completion request, exactly as an OpenAI-compatible client sends it.

    `model` is an alias, not a provider model id; the resolver checks its shape
    before any lookup, so a malformed one is a 422 rather than a 404 claiming the
    name does not exist.
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
    call as free.
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
    calls of one answer into one.
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
    first frame only; repeating it is read as a second message.
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
    never zero-filled.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion.chunk"] = CHUNK_OBJECT
    created: int
    model: str
    choices: tuple[ChunkChoice, ...]
    usage: CompletionUsage | None = None
