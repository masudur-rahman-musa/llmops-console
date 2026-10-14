"""`POST /admin/releases`: the deploys record.

The releases table is written by CI rather than by a served request, which is why
it arrives here rather than as a code path. What it stores, and why each field is
checked before any statement runs, is `app.storage.console_events`.

Behind the admin key (`admin_key`): a working tenant key is 401, and an unset
admin key authorises nobody (`test_every_console_event_route_is_401_without_an_admin_key`,
`test_admin_routes_reject_every_caller_when_no_admin_key_is_configured`). Not
counted by the request metrics, traced, and cluster-internal.

The console event writer is read straight off `app.state` here rather than
through `incidents._console_events`: a route never imports a sibling route
(`tests/api/test_route_layering.py`), and the accessor is a single state read.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import admin_key
from app.model_routing.errors import IdentifierInvalid
from app.schemas.admin import ReleaseCreate
from app.storage.console_events import EVAL_RUN_ID_FIELD, ConsoleEventWriter, ReleaseRow

ADMIN_RELEASES = "/admin/releases"
"""Where a deploy is recorded. Pinned against the mounted route by
`test_the_admin_console_event_paths_and_statuses_are_pinned`."""


router = APIRouter(dependencies=[Depends(admin_key)])


@router.post(ADMIN_RELEASES, status_code=201)
def record_release(request: Request, body: ReleaseCreate) -> ReleaseCreate:
    """Record one deploy of one sha.

    A named `eval_run_id` is checked for EXISTENCE before the insert. The column is
    a foreign key, so without this check a CI job posting a run whose rows never
    stored would reach the constraint and the integrity error would leave the route
    as a 500 - a deploy step failing with a stack trace instead of a typed message
    naming the input at fault (`test_a_release_naming_an_unstored_run_is_422_
    validation_naming_eval_run_id`). The check is the writer's, so the in-memory
    writer answers True for every well-formed id and a local run keeps working.

    Two deploys of one sha are two rows (decision B): nothing here looks for an
    existing row, because a rollback re-deploys an earlier sha and an upsert would
    erase the original deploy.

    Echoes the body rather than an id: the key is `(git_sha, deployed_at)`, both of
    which the caller sent, so there is nothing here it does not already know.
    """
    writer: ConsoleEventWriter = request.app.state.console_events
    if body.eval_run_id is not None and not writer.eval_run_exists(body.eval_run_id):
        raise IdentifierInvalid(
            f"{EVAL_RUN_ID_FIELD}: no eval run carries this id", param=EVAL_RUN_ID_FIELD
        )
    writer.write_release(
        ReleaseRow(
            git_sha=body.git_sha,
            deployed_at=body.deployed_at,
            eval_run_id=body.eval_run_id,
            pr_url=body.pr_url,
            status=body.status,
        )
    )
    return body
