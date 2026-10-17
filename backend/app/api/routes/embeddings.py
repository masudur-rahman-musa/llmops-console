"""POST /v1/embeddings: the thin route over the embedding pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.api.deps import ClientsDep, Context
from app.limits.budget import BUDGET_OVERRIDE_HEADER
from app.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse
from app.services.embedding_pipeline import EmbeddingsResult, embed_texts

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
    clients: ClientsDep,
) -> EmbeddingsResponse:
    """Resolve, redact, walk candidates, cost-book — all in one call.

    `embed_texts` runs the full pipeline and returns an `EmbeddingsResult`.
    This route wraps it into the HTTP layer and adds nothing else.

    Declared `def` rather than `async def` for the same reason the chat route
    is: the adapters are synchronous `httpx` clients, so FastAPI runs this in
    a worker thread instead of blocking the event loop.
    """
    result = embed_texts(
        clients,
        context,
        body,
        request.scope,
        budget_override=request.headers.get(BUDGET_OVERRIDE_HEADER),
    )
    assert isinstance(result, EmbeddingsResult)
    response.headers.update(result.headers)
    return result.body
