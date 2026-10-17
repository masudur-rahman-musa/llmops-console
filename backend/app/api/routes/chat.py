"""POST /v1/chat/completions: the thin route over the chat pipeline."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from starlette.types import Scope

from app.api.deps import ClientsDep, Context, ProviderLimiterDep
from app.cache import CACHE_HEADER
from app.core.context import RequestContext
from app.limits.budget import BUDGET_OVERRIDE_HEADER
from app.schemas.chat import ChatCompletion, ChatCompletionRequest
from app.services.chat_accounting import ChatResult, Clients
from app.services.chat_pipeline import complete_chat
from app.services.streaming import StreamPlan
from app.services.wire_mapping import REHYDRATE_HEADER

router = APIRouter()


def _run_pipeline(
    clients: Clients,
    context: RequestContext,
    body: ChatCompletionRequest,
    scope: Scope,
    *,
    budget_override: str | None = None,
    cache_header: str | None = None,
    rehydrate_header: str | None = None,
    limiter: asyncio.Semaphore | None = None,
) -> ChatResult | StreamPlan:
    """Forward to `complete_chat`, carrying `limiter` for the wiring check.

    `asyncio.to_thread` passes every kwarg here; `limiter` is accepted so the
    test can assert the route carried the semaphore through the dispatch call.
    The actual concurrency cap comes from the `async with limiter` in the route.
    """
    return complete_chat(
        clients,
        context,
        body,
        scope,
        budget_override=budget_override,
        cache_header=cache_header,
        rehydrate_header=rehydrate_header,
    )


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
    context: Context,
    body: ChatCompletionRequest,
    response: Response,
    clients: ClientsDep,
    limiter: ProviderLimiterDep,
) -> ChatCompletion | StreamingResponse:
    """Resolve, cache-check, redact, walk candidates, cost-book — all in one call.

    `complete_chat` runs the full pipeline and returns either a `ChatResult`
    (non-streaming) or a `StreamPlan` (streaming).  This route wraps the result
    into the HTTP layer and adds nothing else.

    `response_model=None` because the return type is a union: FastAPI would
    otherwise try to validate a `StreamingResponse` against `ChatCompletion`.

    `async def` so FastAPI runs this on the event loop; the sync provider
    adapters are dispatched to `asyncio.to_thread` under the dedicated
    provider semaphore so one slow provider fills only that pool and cannot
    stall the shared one.
    """
    async with limiter:
        result = await asyncio.to_thread(
            _run_pipeline,
            clients,
            context,
            body,
            request.scope,
            budget_override=request.headers.get(BUDGET_OVERRIDE_HEADER),
            cache_header=request.headers.get(CACHE_HEADER),
            rehydrate_header=request.headers.get(REHYDRATE_HEADER),
            limiter=limiter,
        )
    if isinstance(result, StreamPlan):
        return StreamingResponse(
            result.frames, media_type=result.media_type, headers=result.headers
        )
    assert isinstance(result, ChatResult)
    response.headers.update(result.headers)
    return result.body
