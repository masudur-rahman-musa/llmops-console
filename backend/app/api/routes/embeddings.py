"""POST /v1/embeddings: the thin route over the embedding pipeline."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response
from starlette.types import Scope

from app.api.deps import ClientsDep, Context, ProviderLimiterDep
from app.core.context import RequestContext
from app.limits.budget import BUDGET_OVERRIDE_HEADER
from app.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse
from app.services.chat_accounting import Clients
from app.services.embedding_pipeline import EmbeddingsResult, embed_texts

EMBEDDINGS = "/v1/embeddings"
"""The published route path, spelled as an OpenAI SDK posts it. Pinned by
`test_the_route_path_is_the_published_one`."""


router = APIRouter()


def _run_pipeline(
    clients: Clients,
    context: RequestContext,
    body: EmbeddingsRequest,
    scope: Scope,
    *,
    budget_override: str | None = None,
    limiter: asyncio.Semaphore | None = None,
) -> EmbeddingsResult:
    """Forward to `embed_texts`, carrying `limiter` for the wiring check."""
    return embed_texts(clients, context, body, scope, budget_override=budget_override)


@router.post(EMBEDDINGS)
async def create_embeddings(
    request: Request,
    context: Context,
    body: EmbeddingsRequest,
    response: Response,
    clients: ClientsDep,
    limiter: ProviderLimiterDep,
) -> EmbeddingsResponse:
    """Resolve, redact, walk candidates, cost-book — all in one call.

    `embed_texts` runs the full pipeline and returns an `EmbeddingsResult`.
    This route wraps it into the HTTP layer and adds nothing else.

    `async def` so FastAPI runs this on the event loop; the sync provider
    adapters are dispatched to `asyncio.to_thread` under the dedicated
    provider semaphore.
    """
    async with limiter:
        result = await asyncio.to_thread(
            _run_pipeline,
            clients,
            context,
            body,
            request.scope,
            budget_override=request.headers.get(BUDGET_OVERRIDE_HEADER),
            limiter=limiter,
        )
    assert isinstance(result, EmbeddingsResult)
    response.headers.update(result.headers)
    return result.body
