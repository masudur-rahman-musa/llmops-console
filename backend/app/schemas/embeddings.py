"""Pydantic wire and response models for /v1/embeddings, with the input caps."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

LIST_OBJECT: Literal["list"] = "list"


EMBEDDING_OBJECT: Literal["embedding"] = "embedding"
"""The two `object` discriminators, in the words the published API uses. A client
branches on these strings to decide how to read the body, so a near-miss is a
body that parses and then hands the client nothing
(`test_the_two_object_discriminators_are_the_published_words`)."""


MAX_EMBEDDING_INPUTS = 64
"""Most inputs one request may carry. Pinned exactly by
`test_the_input_caps_are_exactly_sixty_four_and_eight_thousand_one_hundred_ninety_two`,
with the inclusive boundary either side in `test_an_input_at_the_boundary_is_accepted`
and `test_a_rejected_input_is_422_before_any_outbound_call`.

An unbounded batch holds a CPU box for minutes, and this endpoint's whole point
is that it runs on that box."""


MAX_EMBEDDING_INPUT_CHARS = 8192
"""Longest single input, in characters. Same pinning and the same reason as the
item cap above. Characters rather than tokens because the gateway has no
tokenizer."""


class _Wire(BaseModel):
    """A model on the published API boundary: frozen, and unknown fields rejected.

    The same base the chat route's wire models use, and restated here rather
    than imported so this module's boundary does not move when that one does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbeddingsRequest(_Wire):
    """One embeddings request, exactly as an OpenAI-compatible client sends it.

    `model` is an alias, not a provider model id; the resolver checks its shape
    before any lookup, so a malformed one is a 422 rather than a 404 claiming
    the name does not exist (`test_a_malformed_alias_is_422_not_404`).

    `input` accepts a bare string or a list of strings and nothing else.
    OpenAI's token-array forms (`int[]`, `int[][]`) are refused by this
    annotation: the gateway has no tokenizer, and neither redaction
    nor pricing can see inside a token array, so passing one through would be a
    shape this product cannot honour its guarantees on (an earlier change decision C,
    `test_a_rejected_input_is_422_before_any_outbound_call`).
    """

    model: str
    input: str | tuple[str, ...]
    encoding_format: Literal["float", "base64"] = "float"

    @field_validator("input")
    @classmethod
    def _within_caps(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        """Refuse an input no CPU box should be asked to embed, before anything is sent.

        On the FIELD rather than on the model, so the shared
        `handle_request_validation_error` reports `input` as the offending
        parameter: with up to 64 items in a batch, "the request is not valid"
        leaves the caller nothing to look at
        (`test_a_capped_input_names_the_field`).

        An empty string is refused alongside the sizes because it is not a text
        to embed: the provider would answer a vector for nothing, or a 400 the
        caller cannot act on.
        """
        items = (value,) if isinstance(value, str) else value
        if not items:
            raise ValueError("input: at least one text is required")
        if len(items) > MAX_EMBEDDING_INPUTS:
            raise ValueError(f"input: at most {MAX_EMBEDDING_INPUTS} inputs per request")
        for position, item in enumerate(items):
            if not item:
                raise ValueError(f"input[{position}]: an empty string is not a text to embed")
            if len(item) > MAX_EMBEDDING_INPUT_CHARS:
                raise ValueError(
                    f"input[{position}]: longer than {MAX_EMBEDDING_INPUT_CHARS} characters"
                )
        return value

    @property
    def texts(self) -> tuple[str, ...]:
        """The inputs as the neutral contract carries them: always a tuple, in order.

        A bare string becomes a one-item tuple here rather than being passed
        down as itself, which something below would iterate character by
        character - one vector per letter, and no error anywhere
        (`test_a_bare_string_input_reaches_the_adapter_as_a_one_item_tuple`).
        """
        return (self.input,) if isinstance(self.input, str) else self.input


class WireEmbedding(BaseModel):
    """One vector as this API returns it: floats, or the base64 string of those floats."""

    model_config = ConfigDict(frozen=True)

    object: Literal["embedding"] = EMBEDDING_OBJECT
    index: int
    embedding: tuple[float, ...] | str


class WireEmbeddingUsage(BaseModel):
    """What the call consumed. Two counts, for the reason `EmbeddingUsage` gives."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    """The `list` envelope an OpenAI-compatible client reads.

    `model` is the ALIAS the caller sent, never the provider model id: aliases
    are the only routable names (the design), so an answer naming a provider model
    would hand back a name that cannot be sent again and disclose which provider
    served the request. The provider model id travels in `X-Gateway-Model`
    (`test_response_model_is_the_alias_the_caller_named`).
    """

    model_config = ConfigDict(frozen=True)

    object: Literal["list"] = LIST_OBJECT
    data: tuple[WireEmbedding, ...]
    model: str
    usage: WireEmbeddingUsage
