"""`POST /v1/embeddings`: the OpenAI wire shape, the input caps, and base64.

The endpoint exists so that embeddings never leave the box. Nothing in this
module enforces that: the `embed` alias, the tenant data class rules and the
resolver do, exactly as they do for a completion. That is the point of routing
this through `resolve()` rather than reaching for an in-process model - residency, health and, later, budgets and the ledger apply to this
route because they apply to the resolver.

Three things here that the chat route does not have to think about:

* Order is the contract. The caller matches vectors back to its own rows by
  `index` and by nothing else, so this route renders the vectors in the order
  the adapter returned them and renumbers nothing
  (`test_a_list_input_returns_one_vector_per_item_in_request_order`). That the
  ADAPTER hands back a complete, correctly numbered answer is the guarantee
  `EmbeddingAdapter.embed` states; one adapter implements it today, and its two
  halves are pinned by `test_indices_that_are_not_zero_to_n_minus_one_raise`
  and `test_an_answer_with_fewer_vectors_than_inputs_raises`. A second
  embedding-capable adapter owes the same two.
* `encoding_format` stops here. The gateway renders base64 itself, from the
  float vector, and the outbound request always asks the provider for floats. The neutral `EmbeddingRequest` has nowhere to put the
  field for that reason.
* The caps are on the request model, not in the route body. A 64-item batch of
  8192-character texts is minutes of a CPU box, so the request that a cap
  refuses must be refused before anything is sent
  (`test_a_rejected_input_is_422_before_any_outbound_call`). A `field_validator`
  rather than a `model_validator` because that is what makes the shared
  validation handler name `input` as the offending parameter
  (`test_a_capped_input_names_the_field`); a cross-field cap would have to be a
  model validator and would name the body instead.

The six `X-Gateway-*` headers come from `gateway.routes.chat.gateway_headers`,
imported rather than rebuilt: two header sets would drift the first time either
changed (`test_the_header_set_is_exactly_six`).

Inline redaction applies here for the same reason it applies to a completion, and
the reason is this route specifically: `confidential` allows region `eu`, so an
embedding input can leave the box, and this is the route whose batches are
largest (an earlier change decision B,
`test_the_embedding_adapter_receives_only_what_the_class_allows`). Every input in
one request shares one map and one numbering, so two batch rows holding the same
address embed the same placeholder.

Two caps now stand in front of this route and they are not the same cap. The
8192-character one above is the wire shape's, applies to every request, and is
refused by the request model. `MAX_REDACTION_INPUT_CHARS` (16384) belongs to the
redaction engine and applies only to a class that is redacted - and since 8192 is
the smaller, the wire cap always answers first here
(`test_an_embedding_input_over_the_routes_own_cap_is_422_first`).
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Mapping
from typing import Literal, cast

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, field_validator

from opsconsole.core.errors import ProviderError
from opsconsole.core.metrics import record_redactions, record_tokens
from opsconsole.core.tracing import SpanAttributes, annotate, stage_span
from opsconsole.gateway.context import Context
from opsconsole.gateway.routes.chat import (
    ROUTE_SPAN,
    absorb_failure,
    gateway_headers,
    health_seam,
    label_candidate,
    provider_span_name,
    record_served,
)
from opsconsole.gateway.routes.redact import (
    REDACT_SPAN,
    redact_or_reject,
    redaction_applies,
    tenant_directory,
)
from opsconsole.providers.base import Adapter, EmbeddingAdapter, EmbeddingRequest
from opsconsole.providers.breaker import CircuitBreaker
from opsconsole.routing.errors import AllProvidersUnavailable
from opsconsole.routing.policy import PolicyStore
from opsconsole.routing.resolver import resolve

EMBEDDINGS = "/v1/embeddings"
"""The published route path, spelled as an OpenAI SDK posts it. Pinned by
`test_the_route_path_is_the_published_one`."""

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

FLOAT32_LITTLE_ENDIAN = "<{count}f"
"""The `struct` format base64 vectors are packed with: little-endian float32.

Named rather than inlined because the byte order is this route's one silent
corruption risk - a big-endian pack, or a float64 one, returns a base64 string
that decodes to plausible numbers rather than to an error. Pinned against the
IEEE-754 bytes of `1.0` by `test_to_base64_is_little_endian_float32`."""


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


def to_base64(vector: tuple[float, ...]) -> str:
    """Render one vector as the OpenAI SDK decodes it: little-endian float32, base64.

    The SDK sends `encoding_format: "base64"` by default whenever numpy is
    installed, so this is the format most real clients will actually receive. Any other byte order or width decodes cleanly into
    wrong numbers, which is why the encoding is pinned against the bytes of
    `1.0` rather than only round-tripped (`test_to_base64_is_little_endian_float32`).
    """
    packed = struct.pack(FLOAT32_LITTLE_ENDIAN.format(count=len(vector)), *vector)
    return base64.b64encode(packed).decode("ascii")


router = APIRouter()


@router.post(EMBEDDINGS)
def create_embeddings(
    request: Request,
    context: Context,
    body: EmbeddingsRequest,
    response: Response,
) -> EmbeddingsResponse:
    """Resolve the alias, then walk the allowed providers that can embed until one does.

    The health seam is "this provider has an embedding adapter AND its breaker
    allows it", not "this provider has an adapter": every provider in the policy
    can chat and only some can embed, so an alias whose candidates are all
    chat-only has to be the resolver's typed 503 `ALL_PROVIDERS_FAILED` rather
    than an `AttributeError` served as a 500 with no code (an earlier change decision B,
    `test_a_candidate_without_an_embedding_adapter_is_503`). The breaker half is
    the same one the chat route uses, built by the same `health_seam`, so a
    provider taken out of rotation is out of rotation on both
    (`test_the_embeddings_route_skips_a_provider_whose_breaker_is_open`).
    Residency is decided inside `resolve()` and therefore BEFORE that seam, so a
    request no region allows is a 451 and not a 503
    (`test_an_eu_tenant_on_a_us_only_alias_is_451_with_no_call`).

    The fallback loop is the chat route's, over the tuple `resolve()` answered and
    never a second resolution: a retryable failure moves to the next candidate,
    anything else is returned from the one that raised it
    (`test_the_embeddings_route_falls_back_on_a_retryable_failure`). Redaction
    happens once, before the loop, so a fallback sends the second provider the
    same placeholders and not a second numbering.

    Redaction happens between the resolution and the outbound call, so the adapter
    is handed placeholders and never the raw inputs - the tests read the stub's
    recorded `EmbeddingRequest` rather than the answer, because a route that
    redacted its response and sent the raw text would look identical from the body.

    Declared `def` rather than `async def` for the same reason the chat route is:
    the adapters are synchronous `httpx` clients, so FastAPI runs this in a
    worker thread instead of blocking the event loop for the whole provider call.
    """
    store: PolicyStore = request.app.state.policy
    adapters: Mapping[str, Adapter] = request.app.state.adapters
    breaker: CircuitBreaker = request.app.state.breaker
    scope = request.scope

    with stage_span(scope, ROUTE_SPAN):
        candidates = resolve(
            store.policy,
            context.tenant,
            body.model,
            context.data_class.value,
            is_healthy=health_seam(
                breaker,
                lambda provider: isinstance(adapters.get(provider), EmbeddingAdapter),
            ),
        )
    label_candidate(scope, body.model, candidates[0])

    texts = body.texts
    redacted = False
    if redaction_applies(store.policy, context.tenant, context.data_class):
        # After the check, never before it: a class that is not redacted must not
        # depend on the directory store at all, or a Redis incident would take
        # down `public` traffic that never reads a name. Compiled once for the
        # whole batch, which is why it is not inside
        # `redact_batch`.
        directory = tenant_directory(request.app.state.directory_store, context.tenant)
        with stage_span(scope, REDACT_SPAN):
            batch = redact_or_reject(texts, param="input", directory=directory)
        texts, redacted = batch.texts, batch.redacted
        record_redactions(tenant=context.tenant, counts=batch.counts)
    annotate(scope, SpanAttributes(redacted=redacted))

    last = len(candidates) - 1
    for attempt, chosen in enumerate(candidates):
        label_candidate(scope, body.model, chosen)
        # The seam above is what makes this cast safe: the resolver only returned
        # providers whose adapter satisfies the protocol.
        adapter = cast(EmbeddingAdapter, adapters[chosen.provider])
        try:
            with stage_span(scope, provider_span_name(chosen.provider)):
                answer = adapter.embed(EmbeddingRequest(model=chosen.model, input=texts))
        except ProviderError as exc:
            absorb_failure(breaker, chosen.provider, exc)
            if not exc.retryable or attempt == last:
                raise
            continue
        break
    else:
        # Unreachable while `resolve()` keeps its contract, and stated as its own
        # typed error for the reason the chat route states it: a resolver that
        # one day answered an empty tuple is a 503, never a 500 with no code.
        raise AllProvidersUnavailable(
            f"no candidate was tried for tenant {context.tenant!r}, alias {body.model!r}"
        )

    record_served(
        scope,
        breaker,
        tenant=context.tenant,
        alias=body.model,
        chosen=chosen,
        failed=attempt,
        policy=store.policy,
        data_class=context.data_class,
        log=request.app.state.residency_log,
    )
    record_tokens(
        tenant=context.tenant,
        provider=chosen.provider,
        model=chosen.model,
        prompt_tokens=answer.usage.prompt_tokens,
    )
    response.headers.update(gateway_headers(chosen, context.trace_id, redacted=redacted))
    return EmbeddingsResponse(
        data=tuple(
            WireEmbedding(
                index=embedding.index,
                embedding=(
                    to_base64(embedding.vector)
                    if body.encoding_format == "base64"
                    else embedding.vector
                ),
            )
            for embedding in answer.embeddings
        ),
        model=body.model,
        usage=WireEmbeddingUsage(
            prompt_tokens=answer.usage.prompt_tokens,
            total_tokens=answer.usage.total_tokens,
        ),
    )
