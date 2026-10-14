"""POST /v1/embeddings: the thin route over the embedding flow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from fastapi import APIRouter, Request, Response

from app.core.errors import ProviderError
from app.core.metrics import record_redactions, record_tokens
from app.core.tracing import SpanAttributes, annotate, stage_span
from app.gateway.context import Context
from app.limits.budget import BUDGET_OVERRIDE_HEADER
from app.model_routing.errors import AllProvidersUnavailable
from app.model_routing.policy import PolicyStore, ResidencyMode
from app.model_routing.resolver import resolve
from app.providers.base import Adapter, EmbeddingAdapter, EmbeddingRequest, Usage
from app.providers.breaker import CircuitBreaker
from app.schemas.embeddings import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    WireEmbedding,
    WireEmbeddingUsage,
)
from app.services.chat_pipeline import (
    COST_HEADER,
    ROUTE_SPAN,
    absorb_failure,
    book_cost,
    check_budget,
    check_rate_limit,
    gateway_headers,
    health_seam,
    label_candidate,
    provider_span_name,
    record_served,
)
from app.services.embedding_pipeline import to_base64
from app.services.redaction_flow import (
    REDACT_SPAN,
    redact_or_reject,
    redaction_applies,
    tenant_directory,
)

EMBEDDINGS = "/v1/embeddings"
"""The published route path, spelled as an OpenAI SDK posts it. Pinned by
`test_the_route_path_is_the_published_one`."""


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

    The tenant's RATE LIMIT is checked first of all, through the chat route's
    `check_rate_limit`, and its monthly budget immediately after, through the same
    route's `check_budget` - both before the resolution, so a tenant over either
    is the same typed 429 here as on a completion and no embedding provider is
    dialled for it (`test_the_embeddings_route_refuses_over_rate_too`,
    `test_the_embeddings_route_refuses_over_cap_too`). The rate check is the
    earlier of the two by criterion 6: a tenant over both is told to slow down
    rather than to top up. The debit that follows is
    inside `book_cost` below, so this route needs no second call to stay in step
    with the counter.

    The cost row is booked at the end, through the chat route's `book_cost` and
    after `record_served`, so an embedding is priced, counted and filed exactly as
    a completion is. `EmbeddingUsage` becomes the ledger's `Usage` with
    `completion_tokens=0`: an embedding has no output direction, and the zero here
    is arithmetic rather than a zero-filled report - `record_tokens` above still
    omits the direction rather than sampling it
    (`test_a_served_request_books_exactly_one_row`,
    `test_the_booked_row_names_what_served`).

    Declared `def` rather than `async def` for the same reason the chat route is:
    the adapters are synchronous `httpx` clients, so FastAPI runs this in a
    worker thread instead of blocking the event loop for the whole provider call.
    """
    check_rate_limit(request.app.state, context)
    check_budget(request.app.state, context, override=request.headers.get(BUDGET_OVERRIDE_HEADER))

    store: PolicyStore = request.app.state.policy
    adapters: Mapping[str, Adapter] = request.app.state.adapters
    breaker: CircuitBreaker = request.app.state.breaker
    mode: ResidencyMode = request.app.state.settings.residency_mode
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
            mode=mode,
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
    response.headers[COST_HEADER] = book_cost(
        request.app.state,
        context,
        alias=body.model,
        chosen=chosen,
        usage=Usage(
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=0,
            total_tokens=answer.usage.total_tokens,
        ),
        cached=False,
    )
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
