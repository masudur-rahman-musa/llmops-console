"""POST /v1/redact: the thin route over the redaction flow."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import Context, DirectoryStoreDep, RedactionMapStoreDep
from app.core.metrics import record_redactions
from app.core.tracing import SpanAttributes, annotate, stage_span
from app.schemas.redact import RedactMessage, RedactRequest, RedactResponse
from app.services.redaction_flow import REDACT_SPAN, filed_map_id, redact_batch, tenant_directory

REDACT = "/v1/redact"
"""The published route path. Pinned against the mounted route by
`test_the_route_path_is_the_published_one`."""


router = APIRouter()


@router.post(REDACT)
def create_redaction(
    request: Request,
    context: Context,
    body: RedactRequest,
    maps: RedactionMapStoreDep,
    directory_store: DirectoryStoreDep,
) -> RedactResponse:
    """Redact the body under one map, file the map for the tenant, answer both.

    The tenant is the AUTHENTICATED one from `context`, never the `X-Tenant`
    header on its own: the map holds raw personal data, so filing it under a name
    the client merely claimed would be the store's one unacceptable failure. The
    header is checked against the key before this runs
    (`test_a_tenant_that_does_not_match_the_key_is_403`).

    The class is the body's when it carries one and `X-Data-Class` otherwise
    (decision E), and the answer echoes whichever applied. It is a label on this
    call rather than a control: what the class decides is the INLINE path, and
    this route redacts either way.

    Declared `def` rather than `async def` like every other route here. Redaction
    is CPU work over the whole body, so a coroutine would hold the event loop for
    the length of the scan instead of running in a worker thread.
    """
    directory = tenant_directory(directory_store, context.tenant)
    data_class = body.data_class if body.data_class is not None else context.data_class

    with stage_span(request.scope, REDACT_SPAN):
        if body.text is not None:
            batch = redact_batch((body.text,), directory)
            text: str | None = batch.texts[0]
            messages: tuple[RedactMessage, ...] | None = None
        else:
            turns = body.messages or ()
            batch = redact_batch(tuple(turn.content for turn in turns), directory)
            text = None
            messages = tuple(
                turn.model_copy(update={"content": content})
                for turn, content in zip(turns, batch.texts, strict=True)
            )
    annotate(request.scope, SpanAttributes(redacted=batch.redacted, data_class=data_class.value))
    record_redactions(tenant=context.tenant, counts=batch.counts)

    return RedactResponse(
        text=text,
        messages=messages,
        data_class=data_class,
        redaction_map_id=filed_map_id(maps, context.tenant, batch.mapping),
    )
