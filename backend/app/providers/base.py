"""The provider-neutral adapter contract.

Everything a provider adapter takes and returns is defined in
`app.schemas.provider`, in the gateway's own vocabulary.  The models are
OpenAI-shaped because that is the API this product publishes, but they are ours:
no provider SDK is imported, and no provider's field names appear, so
`providers/openai_compatible.py` implements it unchanged rather than translating
into one provider's dialect, and the OpenAI and Azure adapters of an earlier change are
meant to do the same.  `test_base_contract_carries_no_provider_vocabulary` holds
that by walking this module's names and every model's fields.

The models are frozen and reject unknown fields.  Frozen because an adapter must
not mutate the caller's request while mapping it - the routes' fallback loop
sends the same request to a second provider, and it has to be the SAME request;
unknown fields rejected because a misspelled `max_token` should be an error
rather than a silently applied default.  Both are pinned by
`test_contract_models_are_frozen` and `test_contract_models_forbid_unknown_fields`.

`embed()` is deliberately absent from `Adapter`: Anthropic has no embeddings
endpoint, and a method that exists only to raise is worse than a second
protocol.  `EmbeddingAdapter` at the foot of this module is that second protocol,
and the embeddings route turns it into a capability the resolver's health seam
can test.

`healthcheck()` is not on the `Adapter` protocol.  The always-true
stub it held proved only that at least one adapter was built, which the
`provider` readiness probe now asserts directly by checking that the adapter
mapping is non-empty.  No network call belongs on the readiness path, and a
method that does nothing is worse than no method at all.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from app.core.validation import IdentifierInvalid

# Re-export every model from schemas.provider so callers that import them from
# here continue to work without a change (rule 13: a moved symbol updates its
# references in the same diff, but this re-export covers the large number of
# test and app imports that predate the move).
from app.schemas.provider import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
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

# Keep the re-exported names visible to ruff so it does not strip them.
__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Embedding",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "FinishReason",
    "ResponseFormat",
    "StreamEvent",
    "StreamStart",
    "TextDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolChoice",
    "ToolSpec",
    "Usage",
    "UsageTotal",
]

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


class HttpAdapter:
    """The shared behaviour every HTTP-based adapter provides.

    Subclasses assign `self.name` (the policy provider name) and `self.http` (a
    `ProviderTransport`) in their own `__init__`.  This base class supplies the
    two methods that are identical across all HTTP adapters: `price_key`, built
    from `self.name`, and `close`, which delegates to `self.http`.

    `AzureOpenAIAdapter` inherits through `OpenAICompatibleAdapter`, so the chain
    is `AzureOpenAIAdapter → OpenAICompatibleAdapter → HttpAdapter` (an earlier change,
    agreed by auto-mode).

    `test_both_adapters_subclass_the_shared_base` pins this structure.
    """

    name: str
    """The policy provider name.  Set by each subclass `__init__`."""

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        return f"{self.name}/{model}"

    def close(self) -> None:
        """Release the connection pool."""
        self.http.close()  # type: ignore[attr-defined]


@runtime_checkable
class Adapter(Protocol):
    """What every provider adapter offers the gateway.

    `name` is the policy provider name (the key in `routing.yaml`'s `providers`
    and in the factory's mapping), not the vendor: two accounts with the same
    vendor are two providers, and the ledger, the metric label and the resolver
    all have to address them apart. `price_key` is built from it for that reason,
    pinned by `test_price_key_is_provider_name_slash_model`.

    `healthcheck()` was removed in an earlier change.  The `provider` readiness probe now
    reads True when the adapter mapping is non-empty and False when empty; it
    calls no adapter method.  `test_the_provider_probe_is_ready_when_an_adapter_exists`
    pins the new behaviour and `test_no_adapter_or_protocol_exposes_healthcheck`
    confirms the method is gone.
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

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        ...


# --- embeddings ---------------------------------------------------------------
#
# A contract of its own beside the chat one, for the same reason `EmbeddingAdapter`
# is a protocol of its own: an embedding request shares no field with a completion
# beyond the model id, and folding the two together would mean a `ChatRequest`
# with a `messages` nobody fills.


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """A provider that can embed. Separate from `Adapter`, and a capability, not a method.

    Every provider in the policy can chat; not every one can embed, so this is
    the thing the embeddings route asks `isinstance` about and hands to the
    resolver as its health seam. An alias whose candidates are all chat-only is
    then the resolver's typed 503 rather than an `AttributeError` served as a 500.
    The two ends of that choice are pinned by
    `test_the_openai_compatible_adapter_satisfies_the_embedding_protocol` and
    `test_the_anthropic_adapter_does_not_satisfy_the_embedding_protocol`.

    `name` is the policy provider name, for the same reason it is on `Adapter`.
    """

    name: str

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed every input. Every failure is a `ProviderError`.

        The guarantee the embeddings route relies on, in three parts, so an
        implementation that keeps two of them has kept none:

        1. `embeddings` carries exactly `len(request.input)` items;
        2. their `index` values are exactly `0..n-1`;
        3. every `vector` is non-empty and every component is a finite number
           within float32 range, because the route packs them as float32 when the
           caller asks for base64.

        None of the three is implied by another. An answer numbered `0..62` for a
        64-item request satisfies (2) and is still missing a row; an empty vector
        satisfies (1) and (2) and renders as `[]` with a 200, which the caller
        stores against a real row. The caller has no way to find any of this out
        for itself - it matches vectors back to its own rows by index and by
        nothing else.

        The OpenAI-compatible implementation checks the numbering and the vectors
        while reading the payload, and the count in `embed()`, where the request
        is still in hand
        (`test_indices_that_are_not_zero_to_n_minus_one_raise`,
        `test_an_empty_vector_raises`,
        `test_a_non_finite_vector_component_raises`,
        `test_a_vector_component_out_of_float32_range_raises`,
        `test_an_answer_with_fewer_vectors_than_inputs_raises`).
        """
        ...

    def price_key(self, model: str) -> str:
        """The stable `"<provider>/<model>"` string the ledger stores."""
        ...
