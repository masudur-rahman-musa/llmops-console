"""POST /v1/chat/completions: the thin route over the chat pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import ClientsDep, Context
from app.limits.budget import BUDGET_OVERRIDE_HEADER
from app.schemas.chat import ChatCompletion, ChatCompletionRequest
from app.services.chat_pipeline import (
    CACHE_HEADER,
    ChatResult,
    complete_chat,
)
from app.services.streaming import StreamPlan
from app.services.wire_mapping import REHYDRATE_HEADER

router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
def create_chat_completion(
    request: Request,
    context: Context,
    body: ChatCompletionRequest,
    response: Response,
    clients: ClientsDep,
) -> ChatCompletion | StreamingResponse:
    """Resolve, cache-check, redact, walk candidates, cost-book — all in one call.

    `complete_chat` runs the full pipeline and returns either a `ChatResult`
    (non-streaming) or a `StreamPlan` (streaming).  This route wraps the result
    into the HTTP layer and adds nothing else.

    `response_model=None` because the return type is a union: FastAPI would
    otherwise try to validate a `StreamingResponse` against `ChatCompletion`.

    Declared `def` rather than `async def` on purpose.  The adapters are
    synchronous `httpx` clients, so FastAPI runs this in a worker thread; an
    `async def` would block the event loop for the whole provider call, and would
    hold it for the whole length of a stream.
    """
    result = complete_chat(
        clients,
        context,
        body,
        request.scope,
        budget_override=request.headers.get(BUDGET_OVERRIDE_HEADER),
        cache_header=request.headers.get(CACHE_HEADER),
        rehydrate_header=request.headers.get(REHYDRATE_HEADER),
    )
    if isinstance(result, StreamPlan):
        return StreamingResponse(
            result.frames, media_type=result.media_type, headers=result.headers
        )
    assert isinstance(result, ChatResult)
    response.headers.update(result.headers)
    return result.body
