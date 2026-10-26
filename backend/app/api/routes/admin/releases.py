"""`POST /admin/releases`: the deploys record.

The releases table is written by CI rather than by a served request, which is why
it arrives here rather than as a code path. What it stores, and why each field is
checked before any statement runs, is `app.storage.console_events`.

Behind the admin key (`admin_key`): a working tenant key is 401, and an unset
admin key authorises nobody. Not
counted by the request metrics, traced, and cluster-internal.

The console event writer arrives through the `ConsoleEventsDep` dependency
(`app.api.deps.get_console_events`), so a
route never imports a sibling route and
never reaches into state by hand.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ConsoleEventsDep
from app.core.validation import IdentifierInvalid
from app.schemas.admin import ReleaseCreate
from app.storage.console_events import EVAL_RUN_ID_FIELD, ReleaseRow

ADMIN_RELEASES = "/admin/releases"
"""Where a deploy is recorded."""


router = APIRouter()


@router.post(ADMIN_RELEASES, status_code=201)
def record_release(body: ReleaseCreate, writer: ConsoleEventsDep) -> ReleaseCreate:
    """Record one deploy of one sha.

    A named `eval_run_id` is checked for EXISTENCE before the insert. The column is
    a foreign key, so without this check a CI job posting a run whose rows never
    stored would reach the constraint and the integrity error would leave the route
    as a 500 - a deploy step failing with a stack trace instead of a typed message
    naming the input at fault. The check is the writer's, so the in-memory
    writer answers True for every well-formed id and a local run keeps working.

    Two deploys of one sha are two rows: nothing here looks for an
    existing row, because a rollback re-deploys an earlier sha and an upsert would
    erase the original deploy.

    Echoes the body rather than an id: the key is `(git_sha, deployed_at)`, both of
    which the caller sent, so there is nothing here it does not already know.
    """
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
