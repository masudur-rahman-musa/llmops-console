r"""Releases: the console record CI writes by hand.

Every other table in `storage/` is filled by a code path. This one is filled
through `/admin/releases`, so the values arriving here are pasted by a CI job
rather than produced by the gateway, and each is held to one rule: it is
pre-checked against this module's own constant BEFORE any statement runs, and
each refusal names its field in `IdentifierInvalid.param`, which is what the
route copies into the `param` a 422 body carries. The inclusive boundaries are
pinned in the storage tests. Incidents are written up as postmortems in
`docs/postmortems/` instead, and their table is gone.

**`eval_run_exists` is a pre-check, not a caught error.** `releases.eval_run_id`
is a foreign key, so an id no run carries would surface as an integrity error the
route could only render as a 500. The route asks this question first; the writer
that can answer it is the Postgres one, and the in-memory writer answers True for
everything because it has no `eval_runs` to consult - the trade this makes,
so a laptop with no `database_url` keeps every admin route answering.
The FK stays on the table regardless, for the writers that never come through
this module.

The writer shape - Protocol, in-memory, Postgres over the app's one pool - is the
one `cost_ledger.py` fixes, and `_checked_length` below is this module's version
of its `_checked_identifier` idiom: one function per field, raising before
anything is borrowed.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from typing import Final, Protocol

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from app.core.validation import IdentifierInvalid
from app.storage.db import DEFAULT_SCHEMA, checked_schema

RELEASES: Final[str] = "releases"
"""The table `migrations/004_console_events.sql` creates. A test checks the name
against the migration."""

EVAL_RUNS: Final[str] = "eval_runs"
"""The table `releases.eval_run_id` references, and the one `eval_run_exists`
reads. Created by `migrations/003_eval_runs.sql` and named here rather than
imported because `storage/eval_store.py` is the layer ABOVE this one."""

RELEASE_STATUSES: Final[frozenset[str]] = frozenset({"deployed", "rolled_back", "blocked"})
"""The closed release vocabulary, matching `releases_status_check`."""

MAX_URL_LENGTH: Final[int] = 500
"""The cap on `pr_url`."""

URL_SCHEME: Final[str] = "https://"
"""The only scheme `pr_url` may carry. The console renders it as a link
an operator clicks, so `file://` and a plain `http://` are refused rather than
normalised; the check is a prefix test against this literal, so an uppercased
`HTTPS://` is refused too."""

MIN_GIT_SHA_LENGTH: Final[int] = 7
"""The shortest abbreviated sha a release may name. Six is ambiguous in a
repository of this size, and git's own default abbreviation is seven."""

MAX_GIT_SHA_LENGTH: Final[int] = 40
"""A full sha-1. Both bounds are pinned inclusively in the tests."""

RUN_ID_LENGTH: Final[int] = 32
"""A `uuid4` hex: 16 bytes, 32 characters."""

GIT_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]+")
"""Lowercase hex, matched with `re.fullmatch` (so a trailing newline is refused). The length rule
is the constants above rather than the pattern, because the field has two bounds
rather than one; the pattern is what refuses an uppercase sha and a trailing
newline, which a length check at the same total length cannot see."""

RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")
"""Exactly `RUN_ID_LENGTH` lowercase hex characters, with the length IN the
pattern as well as in the constant, for the reason `cost_ledger.TRACE_ID_PATTERN`
gives: a trailing newline, a CRLF and a space at the exact total length are
refused by the pattern and by nothing else."""

EVAL_RUN_ID_FIELD: Final[str] = "eval_run_id"
"""What a release naming an unstored run is refused as."""


class ReleaseRow(BaseModel):
    """One deploy of one sha.

    Not one sha: `(git_sha, deployed_at)` is the key, so a rollback
    that re-deploys an earlier sha is a second row rather than an overwrite of the
    first deploy's time.

    `eval_run_id` is nullable because a sha deployed before any eval has no run to
    name, and `pr_url` because a deploy can be made from a branch that never had a
    pull request.
    """

    model_config = ConfigDict(frozen=True)

    git_sha: str
    deployed_at: datetime
    eval_run_id: str | None
    pr_url: str | None
    status: str


class ConsoleEventWriter(Protocol):
    """Where a release goes, and the one question the route asks first.

    A Protocol rather than a base class, for the reason `CostLedgerWriter` gives:
    a caller's own writer satisfies it without importing anything from here to
    inherit from.

    Unlike `CostLedgerWriter.write`, these methods are NOT allowed to fail
    silently at the caller. A cost row is lost behind a request that already
    happened; a release that failed to file is a CI job waiting on the answer, so
    the route lets the failure become an answer.
    """

    def write_release(self, row: ReleaseRow) -> None:
        """File one deploy. Raises `IdentifierInvalid` on a value that fails a pre-check."""
        ...

    def eval_run_exists(self, eval_run_id: str) -> bool:
        """Is there an `eval_runs` row with this id, as far as this writer can tell."""
        ...


class InMemoryConsoleEventWriter:
    """Releases in this process's memory, in the order written.

    What a gateway with no `OPSCONSOLE_DATABASE_URL` builds, the same
    trade `InMemoryCostLedgerWriter` makes. It is
    not a store: a restart loses every release recorded on a laptop, which is the
    accepted cost of keeping the admin routes answering without Postgres.
    """

    def __init__(self) -> None:
        self._releases: list[ReleaseRow] = []
        self._lock = threading.Lock()

    @property
    def releases(self) -> tuple[ReleaseRow, ...]:
        """A snapshot, so a caller holding it does not see later rows appear in it."""
        with self._lock:
            return tuple(self._releases)

    def write_release(self, row: ReleaseRow) -> None:
        _checked_release(row)
        with self._lock:
            self._releases.append(row)

    def eval_run_exists(self, eval_run_id: str) -> bool:
        """True for any well-formed id: this writer has no `eval_runs` to consult.

        Deliberate rather than accidental. Answering False would turn
        every `POST /admin/releases` naming a run into a 422 on exactly the
        configuration this writer exists to serve. The shape is still checked, so
        a malformed id is refused here as it would be against Postgres.
        """
        checked_eval_run_id(eval_run_id)
        return True


class PostgresConsoleEventWriter:
    """One statement per call, over the pool the app already holds. No DDL, ever.

    Every pre-check runs BEFORE the borrow, so a refused value
    costs no connection and leaves nothing half-written. Each value then becomes a
    bound parameter; the checks are not what stands between these tables and an
    injection, they are what keeps the records an operator reads honest.
    """

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)

    def write_release(self, row: ReleaseRow) -> None:
        """Write one deploy. Raises only on a value that fails a pre-check."""
        _checked_release(row)
        with self._pool.connection() as conn:
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (git_sha, deployed_at, eval_run_id, pr_url, status) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(sql.Identifier(self._schema, RELEASES)),
                (row.git_sha, row.deployed_at, row.eval_run_id, row.pr_url, row.status),
            )

    def eval_run_exists(self, eval_run_id: str) -> bool:
        """Is there an `eval_runs` row with this id. The route's pre-check on the FK."""
        checked_eval_run_id(eval_run_id)
        with self._pool.connection() as conn:
            found = conn.execute(
                sql.SQL("SELECT 1 FROM {} WHERE id = %s").format(
                    sql.Identifier(self._schema, EVAL_RUNS)
                ),
                (eval_run_id,),
            ).fetchone()
        return found is not None


def checked_release_status(value: str) -> str:
    """One of `RELEASE_STATUSES`, or `IdentifierInvalid`."""
    if value not in RELEASE_STATUSES:
        raise IdentifierInvalid(
            f"status: not one of {', '.join(sorted(RELEASE_STATUSES))}", param="status"
        )
    return value


def _checked_length(value: str, field: str, *, maximum: int) -> str:
    """One free-text field, between one character and its own cap.

    Empty is a refusal rather than a default: a blank link renders as a gap that
    nothing distinguishes from a rendering fault.
    """
    if not value:
        raise IdentifierInvalid(f"{field}: empty", param=field)
    if len(value) > maximum:
        raise IdentifierInvalid(f"{field}: longer than {maximum} characters", param=field)
    return value


def checked_url(value: str, field: str) -> str:
    """A link the console renders, against its cap and against `URL_SCHEME`.

    `field` is passed rather than fixed, so each refusal names the input the
    caller actually sent - the console puts the message beside that field.
    """
    _checked_length(value, field, maximum=MAX_URL_LENGTH)
    if not value.startswith(URL_SCHEME):
        raise IdentifierInvalid(f"{field}: does not start with {URL_SCHEME}", param=field)
    return value


def checked_git_sha(value: str) -> str:
    """A commit sha, `MIN_GIT_SHA_LENGTH` to `MAX_GIT_SHA_LENGTH` lowercase hex.

    `re.fullmatch`, so a trailing newline is refused - which
    matters here more than for most fields, because CI posts this value from a
    shell and a captured `git rev-parse` carries one.
    """
    if not (MIN_GIT_SHA_LENGTH <= len(value) <= MAX_GIT_SHA_LENGTH):
        raise IdentifierInvalid(
            f"git_sha: not between {MIN_GIT_SHA_LENGTH} and {MAX_GIT_SHA_LENGTH} characters",
            param="git_sha",
        )
    if GIT_SHA_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("git_sha: not lowercase hex", param="git_sha")
    return value


def checked_eval_run_id(value: str) -> str:
    """An `eval_runs` id, which `evals.report` also generates as a `uuid4` hex."""
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(
            f"{EVAL_RUN_ID_FIELD}: not {RUN_ID_LENGTH} lowercase hex characters",
            param=EVAL_RUN_ID_FIELD,
        )
    return value


def _checked_release(row: ReleaseRow) -> ReleaseRow:
    """Every client-supplied value on a deploy, before anything is borrowed.

    `eval_run_id`'s EXISTENCE is the route's question, not this one's: here only
    its shape is checked."""
    checked_git_sha(row.git_sha)
    checked_release_status(row.status)
    if row.eval_run_id is not None:
        checked_eval_run_id(row.eval_run_id)
    if row.pr_url is not None:
        checked_url(row.pr_url, "pr_url")
    return row
