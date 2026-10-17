"""Base64 rendering for embedding vectors, and the embeddings service entry point."""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import cast

from starlette.types import Scope

from app.core.context import RequestContext
from app.core.metrics import record_redactions, record_tokens
from app.core.tracing import SpanAttributes, annotate, stage_span
from app.model_routing.policy import ResidencyMode
from app.model_routing.resolver import Resolved, resolve
from app.providers.base import (
    EmbeddingAdapter,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)
from app.schemas.embeddings import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    WireEmbedding,
    WireEmbeddingUsage,
)
from app.services.chat_pipeline import (
    COST_HEADER,
    ROUTE_SPAN,
    Clients,
    book_cost,
    check_budget,
    check_rate_limit,
    gateway_headers,
    health_seam,
    label_candidate,
    provider_span_name,
    record_served,
    walk_candidates,
)
from app.services.redaction_flow import (
    REDACT_SPAN,
    redact_or_reject,
    redaction_applies,
    tenant_directory,
)


@dataclass
class EmbeddingsResult:
    """What `embed_texts` hands back so the HTTP layer can build the response.

    `body` is the response the route returns; `headers` are all seven
    `X-Gateway-*` headers including `X-Gateway-Cost-USD`.  The route does
    `response.headers.update(result.headers)` and returns `result.body`.
    """

    body: EmbeddingsResponse
    headers: dict[str, str]


def embed_texts(
    clients: Clients,
    context: RequestContext,
    body: EmbeddingsRequest,
    scope: Scope,
    *,
    budget_override: str | None = None,
) -> EmbeddingsResult:
    """Run the full embeddings pipeline and return a plain result.

    The route calls this once, then wraps the result into an HTTP response.
    All pipeline logic lives here: rate limit, budget, resolution, redaction,
    the fallback loop, cost booking.  The route adds nothing except the HTTP
    layer (`response.headers`, return value).

    Mirrors `complete_chat` in structure: same checks, same `walk_candidates`
    loop, same `record_served` + `book_cost` close.
    """
    check_rate_limit(clients, context)
    check_budget(clients, context, override=budget_override)

    mode: ResidencyMode = clients.settings.residency_mode

    with stage_span(scope, ROUTE_SPAN):
        candidates = resolve(
            clients.policy.policy,
            context.tenant,
            body.model,
            context.data_class.value,
            is_healthy=health_seam(
                clients.breaker,
                lambda provider: isinstance(clients.adapters.get(provider), EmbeddingAdapter),
            ),
            mode=mode,
        )
    label_candidate(scope, body.model, candidates[0])

    texts = body.texts
    redacted = False
    if redaction_applies(clients.policy.policy, context.tenant, context.data_class):
        directory = tenant_directory(clients.directory_store, context.tenant)
        with stage_span(scope, REDACT_SPAN):
            batch = redact_or_reject(texts, param="input", directory=directory)
        texts, redacted = batch.texts, batch.redacted
        record_redactions(tenant=context.tenant, counts=batch.counts)
    annotate(scope, SpanAttributes(redacted=redacted))

    _call_count: list[int] = [0]

    def provider_call(
        chosen: Resolved,
    ) -> tuple[EmbeddingResponse, Resolved, int]:
        failed_so_far = _call_count[0]
        _call_count[0] += 1
        label_candidate(scope, body.model, chosen)
        adapter = cast(EmbeddingAdapter, clients.adapters[chosen.provider])
        with stage_span(scope, provider_span_name(chosen.provider)):
            answer = adapter.embed(EmbeddingRequest(model=chosen.model, input=texts))
        return answer, chosen, failed_so_far

    answer, chosen, failed = walk_candidates(candidates, clients.breaker, provider_call)

    record_served(
        scope,
        clients.breaker,
        tenant=context.tenant,
        alias=body.model,
        chosen=chosen,
        failed=failed,
        policy=clients.policy.policy,
        data_class=context.data_class,
        log=clients.residency_log,
    )
    record_tokens(
        tenant=context.tenant,
        provider=chosen.provider,
        model=chosen.model,
        prompt_tokens=answer.usage.prompt_tokens,
    )
    headers = gateway_headers(chosen, context.trace_id, redacted=redacted)
    headers[COST_HEADER] = book_cost(
        clients,
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
    return EmbeddingsResult(
        body=EmbeddingsResponse(
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
        ),
        headers=headers,
    )


FLOAT32_LITTLE_ENDIAN = "<{count}f"
"""The `struct` format base64 vectors are packed with: little-endian float32.

Named rather than inlined because the byte order is this route's one silent
corruption risk - a big-endian pack, or a float64 one, returns a base64 string
that decodes to plausible numbers rather than to an error. Pinned against the
IEEE-754 bytes of `1.0` by `test_to_base64_is_little_endian_float32`."""


def to_base64(vector: tuple[float, ...]) -> str:
    """Render one vector as the OpenAI SDK decodes it: little-endian float32, base64.

    The SDK sends `encoding_format: "base64"` by default whenever numpy is
    installed, so this is the format most real clients will actually receive. Any other byte order or width decodes cleanly into
    wrong numbers, which is why the encoding is pinned against the bytes of
    `1.0` rather than only round-tripped (`test_to_base64_is_little_endian_float32`).
    """
    packed = struct.pack(FLOAT32_LITTLE_ENDIAN.format(count=len(vector)), *vector)
    return base64.b64encode(packed).decode("ascii")
