"""`POST /admin/incidents` and `PATCH /admin/incidents/{incident_id}`: the timeline's two writes.

The incidents table is one nothing else in the system produces: the console's
timeline is written by a human rather than by a served request, which is why it
arrives here rather than as a code path. What it stores, and why each field is
checked before any statement runs, is `app.storage.console_events`.

Behind the admin key (`admin_key`): a working tenant key is 401 on both, and an
unset admin key authorises nobody rather than everybody. Neither
route is counted by the request metrics (`COUNTED_PATH_PREFIX` is `/v1/`), both
are traced, and both are cluster-internal.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.api.deps import ConsoleEventsDep
from app.schemas.admin import IncidentCreate, IncidentCreated, IncidentResolve
from app.storage.console_events import (
    IncidentRow,
    checked_incident_id,
)

ADMIN_INCIDENTS = "/admin/incidents"
"""Where an incident is recorded."""

ADMIN_INCIDENT = "/admin/incidents/{incident_id}"
"""Where one is resolved. Pinned by the same case."""


router = APIRouter()


@router.post(ADMIN_INCIDENTS, status_code=201)
def record_incident(body: IncidentCreate, writer: ConsoleEventsDep) -> IncidentCreated:
    """Record one incident and answer the id it was filed under.

    201 rather than 200: this creates a record the caller can address afterwards,
    and the id in the body is that address. `uuid4().hex` rather than a slug the
    caller supplies, so two incidents filed in the same minute cannot
    collide on a name someone typed twice and no client string reaches a primary
    key.

    `resolved_at` is `None` and cannot be anything else - `IncidentCreate` has no
    such field. Every value the body did carry is pre-checked inside
    `write_incident` before any statement runs, and each refusal is an
    `IdentifierInvalid` naming its own field, which the error handler renders as
    422 `VALIDATION` with that field as `param`.

    Declared `def` rather than `async def`, like the two routes above: the write
    is blocking work over a pooled connection, so FastAPI running it in a worker
    thread is what keeps it off the event loop.
    """
    row = IncidentRow(
        id=uuid.uuid4().hex,
        started_at=body.started_at,
        resolved_at=None,
        severity=body.severity,
        title=body.title,
        summary=body.summary,
        postmortem_url=body.postmortem_url,
    )
    writer.write_incident(row)
    return IncidentCreated(id=row.id)


@router.patch(ADMIN_INCIDENT)
def resolve_incident(
    incident_id: str, body: IncidentResolve, writer: ConsoleEventsDep
) -> IncidentCreated:
    """Close one open incident, or refuse with the reason it could not be closed.

    The path parameter is the one place a client string takes the shape of an
    incident id, so it is pre-checked here as well as in the writer: a malformed
    id is refused before a connection is borrowed, exactly as `?tenant=` is on the
    view above.

    Both failures are 422 `VALIDATION` naming `id` and are
    told apart by their messages, which `storage.console_events` owns:
    `ErrorCode.VALIDATION` is fixed at 422, and a 404 or a 409 carrying that code
    would break the one-status-per-code table
    the error contract tests hold. Resolving twice is a refusal
    rather than a no-op, because the second call would otherwise
    overwrite the timestamp a postmortem's duration is computed from.

    Answers the id it resolved, so the response has the same shape as the create's
    and a caller scripting both reads one field from both.
    """
    checked_incident_id(incident_id)
    writer.resolve_incident(incident_id, body.resolved_at)
    return IncidentCreated(id=incident_id)
