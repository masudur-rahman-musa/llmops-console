r"""Incidents and releases: the two console records a human and CI write by hand.

Every other table in `ledger/` is filled by a code path. These two are filled
through `/admin/incidents` and `/admin/releases`, so the values arriving here are
typed by an operator or pasted by a CI job rather than produced by the gateway,
and a project rule applies to all seven of them: each is pre-checked against
this module's own constant BEFORE any statement runs, and each refusal names its
field in `IdentifierInvalid.param`, which is what the route copies into the
`param` a 422 body carries. The inclusive boundaries are
tests/ledger/test_console_events.py's seven tables.

**An incident is born open** (the design). `resolved_at` is the one nullable
timestamp and a NULL there IS "open", so there is no status column that can
disagree with it. `write_incident` takes a row whose `resolved_at` is whatever
the caller built; the ROUTE is what never offers the field, and
`resolve_incident` is the only way one gets set
(`test_the_incident_post_returns_a_thirty_two_hex_id_and_never_sets_resolved_at`,
`test_an_open_incident_stores_a_null_resolved_at_and_the_patch_sets_it`).

**Resolving is not idempotent** (decision E). `resolve_incident` raises
`IdentifierInvalid` for an id no row carries and for an incident already
resolved, with a DIFFERENT message for each, because a mistyped id that answers
200 tells an operator an incident is closed while it is still open, and a second
resolve that succeeds overwrites the one timestamp a postmortem's duration is
computed from. The two messages are the only thing telling the failures apart:
both are `VALIDATION` naming `id`, since
`ErrorCode.VALIDATION` is fixed at 422 and `tests/gateway/test_error_contract.py`
pins one class and one status per code. Pinned by
`test_resolving_an_unknown_id_and_an_already_resolved_incident`.

**`eval_run_exists` is a pre-check, not a caught error.** `releases.eval_run_id`
is a foreign key, so an id no run carries would surface as an integrity error the
route could only render as a 500. The route asks this question first; the writer
that can answer it is the Postgres one, and the in-memory writer answers True for
everything because it has no `eval_runs` to consult - the trade decision C makes,
so a laptop with no `database_url` keeps every admin route answering
(`test_the_in_memory_writer_answers_that_every_eval_run_exists`,
`test_a_release_naming_an_unstored_run_is_422_validation_naming_eval_run_id`).
The FK stays on the table regardless, for the writers that never come through
this module (`test_a_release_naming_an_unstored_run_is_refused_by_the_foreign_key`).

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

from app.ledger.db import DEFAULT_SCHEMA, checked_schema
from app.model_routing.errors import IdentifierInvalid

INCIDENTS: Final[str] = "incidents"
"""One of the two tables `migrations/004_console_events.sql` creates. Pinned
against the migration by `test_the_migration_creates_both_tables_in_the_schema`."""

RELEASES: Final[str] = "releases"
"""The other one, pinned by the same case."""

EVAL_RUNS: Final[str] = "eval_runs"
"""The table `releases.eval_run_id` references, and the one `eval_run_exists`
reads. Created by `migrations/003_eval_runs.sql` and named here rather than
imported because `evals/store.py` is the layer ABOVE this one."""

SEVERITIES: Final[frozenset[str]] = frozenset({"sev1", "sev2", "sev3"})
"""The closed severity vocabulary, spelled in lowercase exactly as the table's
`incidents_severity_check` spells it. Compared as a set membership rather than
case-insensitively, so `SEV1` is refused here as it would be there
(`test_the_severity_table`)."""

RELEASE_STATUSES: Final[frozenset[str]] = frozenset({"deployed", "rolled_back", "blocked"})
"""The closed release vocabulary, matching `releases_status_check`
(`test_the_status_table`)."""

MAX_TITLE_LENGTH: Final[int] = 200
"""A headline an operator types, capped so an unbounded `text` column cannot take
whatever a paste carries. Pinned inclusively by
`test_the_title_length_boundary_is_inclusive`."""

MAX_SUMMARY_LENGTH: Final[int] = 4000
"""What happened, in a paragraph or several. Twenty times the title's cap and a
separate constant, because one shared checker would refuse a legal summary
(`test_the_summary_length_boundary_is_inclusive`)."""

MAX_URL_LENGTH: Final[int] = 500
"""The cap on `postmortem_url` and `pr_url` alike
(`test_the_url_length_boundary_is_inclusive`)."""

URL_SCHEME: Final[str] = "https://"
"""The only scheme either URL field may carry. The console renders both as links
an operator clicks, so `file://` and a plain `http://` are refused rather than
normalised; the check is a prefix test against this literal, so an uppercased
`HTTPS://` is refused too (`test_the_url_length_boundary_is_inclusive`)."""

MIN_GIT_SHA_LENGTH: Final[int] = 7
"""The shortest abbreviated sha a release may name. Six is ambiguous in a
repository of this size, and git's own default abbreviation is seven."""

MAX_GIT_SHA_LENGTH: Final[int] = 40
"""A full sha-1. Both bounds pinned inclusively by
`test_the_git_sha_boundaries_are_inclusive`."""

INCIDENT_ID_LENGTH: Final[int] = 32
"""A `uuid4` hex: 16 bytes, 32 characters. Pinned by
`test_the_incident_id_is_exactly_thirty_two_hex`."""

GIT_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]+")
"""Lowercase hex, matched with `re.fullmatch`. The length rule
is the constants above rather than the pattern, because the field has two bounds
rather than one; the pattern is what refuses an uppercase sha and a trailing
newline, which a length check at the same total length cannot see."""

INCIDENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")
"""Exactly `INCIDENT_ID_LENGTH` lowercase hex characters, with the length IN the
pattern as well as in the constant, for the reason `cost_ledger.TRACE_ID_PATTERN`
gives: the three sharpest rejection rows - a trailing newline, a CRLF and a space
at the exact total length - are refused by the pattern and by nothing else."""

INCIDENT_ID_FIELD: Final[str] = "id"
"""What every incident-id refusal names, on the write path and on the PATCH path
parameter alike. The route copies it into the wire body's `param`."""

EVAL_RUN_ID_FIELD: Final[str] = "eval_run_id"
"""What a release naming an unstored run is refused as."""

UNKNOWN_INCIDENT_MESSAGE: Final[str] = f"{INCIDENT_ID_FIELD}: no incident carries this id"
"""One of the two `PATCH` refusals. Both are 422 `VALIDATION` naming `id`
(deviation 1), so the MESSAGE is the only thing telling them apart on the wire;
`test_patching_a_resolved_incident_is_422_validation_with_its_own_message`
asserts the two are different rather than asserting either text."""

ALREADY_RESOLVED_MESSAGE: Final[str] = f"{INCIDENT_ID_FIELD}: this incident is already resolved"
"""The other one. Resolving twice is a mistake rather than a no-op (decision E):
the second call would otherwise overwrite the timestamp a postmortem's duration
is computed from."""


class IncidentRow(BaseModel):
    """One incident, as it is written: the design's columns and nothing else.

    Frozen, for the reason `CostLedgerRow` is: the same object is handed to a
    writer and read back from one, and `resolve_incident` therefore REPLACES a
    stored row with `model_copy` rather than mutating it in place.

    `resolved_at` and `postmortem_url` are the two nullable fields. An open
    incident has no resolution time by definition, and a postmortem is written
    days later or not at all.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    started_at: datetime
    resolved_at: datetime | None
    severity: str
    title: str
    summary: str
    postmortem_url: str | None


class ReleaseRow(BaseModel):
    """One deploy of one sha (the design).

    Not one sha: `(git_sha, deployed_at)` is the key (decision B), so a rollback
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
    """Where an incident or a release goes, and the one question the route asks first.

    A Protocol rather than a base class, for the reason `CostLedgerWriter` gives:
    a caller's own writer satisfies it without importing anything from here to
    inherit from.

    Unlike `CostLedgerWriter.write`, these methods are NOT allowed to fail
    silently at the caller. A cost row is lost behind a request that already
    happened; an incident that failed to file is a request an operator is waiting
    on, so the route lets the failure become an answer.
    """

    def write_incident(self, row: IncidentRow) -> None:
        """File one incident. Raises `IdentifierInvalid` on a value that fails a pre-check."""
        ...

    def resolve_incident(self, incident_id: str, resolved_at: datetime) -> None:
        """Close one open incident, or raise `IdentifierInvalid` naming `id`."""
        ...

    def write_release(self, row: ReleaseRow) -> None:
        """File one deploy. Raises `IdentifierInvalid` on a value that fails a pre-check."""
        ...

    def eval_run_exists(self, eval_run_id: str) -> bool:
        """Is there an `eval_runs` row with this id, as far as this writer can tell."""
        ...


class InMemoryConsoleEventWriter:
    """Both records in this process's memory, in the order written.

    What a gateway with no `OPSCONSOLE_DATABASE_URL` builds (decision C), the same
    trade `InMemoryCostLedgerWriter` and `InMemoryResidencyLogWriter` make. It is
    not a store: a restart loses every incident recorded on a laptop, which is the
    accepted cost of keeping the admin routes answering without Postgres.

    Two lists rather than one, kept apart so the timeline cannot render a deploy
    as an outage (`test_the_in_memory_writer_files_and_reads_back_both_rows`).
    Incidents are keyed by nothing: `resolve_incident` scans, because the list is
    a session's worth of rows and an index here would be state to keep correct for
    no gain.
    """

    def __init__(self) -> None:
        self._incidents: list[IncidentRow] = []
        self._releases: list[ReleaseRow] = []
        self._lock = threading.Lock()

    @property
    def incidents(self) -> tuple[IncidentRow, ...]:
        """A snapshot, so a caller holding it does not see later rows appear in it."""
        with self._lock:
            return tuple(self._incidents)

    @property
    def releases(self) -> tuple[ReleaseRow, ...]:
        """The same, for deploys."""
        with self._lock:
            return tuple(self._releases)

    def write_incident(self, row: IncidentRow) -> None:
        _checked_incident(row)
        with self._lock:
            self._incidents.append(row)

    def resolve_incident(self, incident_id: str, resolved_at: datetime) -> None:
        """Close one incident, refusing an unknown id and an already-resolved one.

        The stored row is frozen, so it is REPLACED at its position with a
        `model_copy` carrying the resolution time; mutating it would be a write to
        an object a caller may already be holding a snapshot of.
        """
        checked_incident_id(incident_id)
        with self._lock:
            for index, row in enumerate(self._incidents):
                if row.id != incident_id:
                    continue
                if row.resolved_at is not None:
                    raise IdentifierInvalid(ALREADY_RESOLVED_MESSAGE, param=INCIDENT_ID_FIELD)
                self._incidents[index] = row.model_copy(update={"resolved_at": resolved_at})
                return
        raise IdentifierInvalid(UNKNOWN_INCIDENT_MESSAGE, param=INCIDENT_ID_FIELD)

    def write_release(self, row: ReleaseRow) -> None:
        _checked_release(row)
        with self._lock:
            self._releases.append(row)

    def eval_run_exists(self, eval_run_id: str) -> bool:
        """True for any well-formed id: this writer has no `eval_runs` to consult.

        Deliberate rather than accidental (decision C). Answering False would turn
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

    `resolve_incident` is one conditional UPDATE followed by a SELECT of what the
    row now says, rather than a SELECT-then-UPDATE: two statements on a borrowed
    connection would let a second PATCH land between them and make "already
    resolved" a race. The UPDATE matching no row is what distinguishes the two
    failures, and the SELECT is what tells "no such incident" from "this one was
    already closed" (`test_an_open_incident_stores_a_null_resolved_at_and_the_patch_sets_it`).
    """

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)

    def write_incident(self, row: IncidentRow) -> None:
        """Write one incident. Raises only on a value that fails a pre-check."""
        _checked_incident(row)
        with self._pool.connection() as conn:
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} ("
                    "id, started_at, resolved_at, severity, title, summary, postmortem_url) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                ).format(sql.Identifier(self._schema, INCIDENTS)),
                (
                    row.id,
                    row.started_at,
                    row.resolved_at,
                    row.severity,
                    row.title,
                    row.summary,
                    row.postmortem_url,
                ),
            )

    def resolve_incident(self, incident_id: str, resolved_at: datetime) -> None:
        """Close one open incident, or raise `IdentifierInvalid` naming `id`."""
        checked_incident_id(incident_id)
        with self._pool.connection() as conn:
            updated = conn.execute(
                sql.SQL(
                    "UPDATE {} SET resolved_at = %s WHERE id = %s AND resolved_at IS NULL"
                ).format(sql.Identifier(self._schema, INCIDENTS)),
                (resolved_at, incident_id),
            ).rowcount
            if updated == 1:
                return
            found = conn.execute(
                sql.SQL("SELECT 1 FROM {} WHERE id = %s").format(
                    sql.Identifier(self._schema, INCIDENTS)
                ),
                (incident_id,),
            ).fetchone()
        if found is None:
            raise IdentifierInvalid(UNKNOWN_INCIDENT_MESSAGE, param=INCIDENT_ID_FIELD)
        raise IdentifierInvalid(ALREADY_RESOLVED_MESSAGE, param=INCIDENT_ID_FIELD)

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


def checked_severity(value: str) -> str:
    """One of `SEVERITIES`, or `IdentifierInvalid` before any statement runs."""
    if value not in SEVERITIES:
        raise IdentifierInvalid(
            f"severity: not one of {', '.join(sorted(SEVERITIES))}", param="severity"
        )
    return value


def checked_release_status(value: str) -> str:
    """One of `RELEASE_STATUSES`, or `IdentifierInvalid`."""
    if value not in RELEASE_STATUSES:
        raise IdentifierInvalid(
            f"status: not one of {', '.join(sorted(RELEASE_STATUSES))}", param="status"
        )
    return value


def _checked_length(value: str, field: str, *, maximum: int) -> str:
    """One free-text field, between one character and its own cap.

    Empty is a refusal rather than a default: a blank title renders as a gap on
    the timeline that nothing distinguishes from a rendering fault.
    """
    if not value:
        raise IdentifierInvalid(f"{field}: empty", param=field)
    if len(value) > maximum:
        raise IdentifierInvalid(f"{field}: longer than {maximum} characters", param=field)
    return value


def checked_title(value: str) -> str:
    """The incident headline, 1 to `MAX_TITLE_LENGTH` characters."""
    return _checked_length(value, "title", maximum=MAX_TITLE_LENGTH)


def checked_summary(value: str) -> str:
    """What happened, 1 to `MAX_SUMMARY_LENGTH` characters."""
    return _checked_length(value, "summary", maximum=MAX_SUMMARY_LENGTH)


def checked_url(value: str, field: str) -> str:
    """A link the console renders, against its cap and against `URL_SCHEME`.

    `field` is passed rather than fixed, because the same rule guards
    `postmortem_url` and `pr_url` and each refusal has to name the input the
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


def checked_incident_id(value: str) -> str:
    """An incident id, exactly `INCIDENT_ID_LENGTH` lowercase hex characters.

    Decision A makes the id server-generated, so the `PATCH` path parameter is the
    only place a client string takes this shape and the only place this runs on
    input someone typed.
    """
    if INCIDENT_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(
            f"{INCIDENT_ID_FIELD}: not {INCIDENT_ID_LENGTH} lowercase hex characters",
            param=INCIDENT_ID_FIELD,
        )
    return value


def checked_eval_run_id(value: str) -> str:
    """An `eval_runs` id, which `evals.report` also generates as a `uuid4` hex.

    The same shape as an incident id and a separate function, because the field it
    names is what the 422 carries: a release naming a malformed run must not report
    `id`, which the console shows beside the incident field.
    """
    if INCIDENT_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(
            f"{EVAL_RUN_ID_FIELD}: not {INCIDENT_ID_LENGTH} lowercase hex characters",
            param=EVAL_RUN_ID_FIELD,
        )
    return value


def _checked_incident(row: IncidentRow) -> IncidentRow:
    """Every client-supplied value on an incident, before anything is borrowed."""
    checked_incident_id(row.id)
    checked_severity(row.severity)
    checked_title(row.title)
    checked_summary(row.summary)
    if row.postmortem_url is not None:
        checked_url(row.postmortem_url, "postmortem_url")
    return row


def _checked_release(row: ReleaseRow) -> ReleaseRow:
    """The same for a deploy. `eval_run_id`'s EXISTENCE is the route's question, not
    this one's: here only its shape is checked."""
    checked_git_sha(row.git_sha)
    checked_release_status(row.status)
    if row.eval_run_id is not None:
        checked_eval_run_id(row.eval_run_id)
    if row.pr_url is not None:
        checked_url(row.pr_url, "pr_url")
    return row
