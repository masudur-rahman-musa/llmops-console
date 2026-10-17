r"""The residency decision log: one row per request a provider actually served.

What this answers is "where did this tenant's data go", per request, for ninety
days. What it does NOT do is enforce anything. The residency guarantee is the
resolver's, made before a provider is dialled and proved by
tests/routing/test_residency_property.py; this module records the answer the
resolver already gave, which is why `allowed` is expected to be `true` on every
row the gateway can produce and why a `false` one is worth an alert.

**`allowed` is computed here, never passed in.** `region_allowed` re-derives the
tenant's permitted regions from the policy at write time. A caller filling the
field would be recording its own belief rather than the policy's answer, and the
one shape that belief takes in practice is a literal `True` - which is exactly
what a log kept as evidence must not contain. Pinned by
`test_allowed_is_computed_from_the_policy_and_can_be_false`.

**A failed write changes nothing about the request** (an earlier change decision D, ADR
0004). `record_residency` logs at ERROR and returns. The evidence trail is
best-effort on purpose: this write sits on the hot path of every served request,
and a Postgres blip taking out a gateway whose residency behaviour is entirely
unchanged is the worse failure. the project notes
request-s-critical-path.md` says what would change that.

**Partitions are made on demand** (decision B), cached per process per day. A
refused `CREATE` is swallowed and the INSERT still runs, landing the row in the
migration's DEFAULT partition: a read-only role costs a badly-placed row, never a
lost one (`test_a_row_whose_partition_cannot_be_created_lands_in_the_default`).
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, date, datetime, timedelta
from typing import Final, Protocol

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from app.core.validation import checked_identifier
from app.model_routing.policy import DataClass
from app.storage.db import DEFAULT_SCHEMA, checked_schema

logger = logging.getLogger(__name__)

RESIDENCY_LOG_RETENTION_DAYS: Final[int] = 90
"""How long a residency row is kept, in days.

A data-protection commitment rather than a tuning knob, which is why
`test_the_retention_is_ninety_days` asserts it exactly: widened, the gateway
keeps request-level evidence for longer than it promised anyone; narrowed, an
auditor asking about last month has nothing to read. `make purge-residency`
drops every partition older than this, and the boundary is inclusive-keep -
`today - 90` stays, `today - 91` goes.
"""

RESIDENCY_LOG: Final[str] = "residency_log"
"""The parent table `migrations/001_residency_log.sql` creates."""

_PARTITION_PREFIX: Final[str] = f"{RESIDENCY_LOG}_"
PARTITION_DAY_FORMAT: Final[str] = "%Y%m%d"
"""`residency_log_20260907`. Pinned by `test_the_partition_name_is_the_pinned_format`.

The writer builds this name and `cli/ledger.py`'s purge parses it back, so the
two agree by construction only as long as nobody changes one of them; the format
is a constant here for that reason and asserted against literals there.
"""


def partition_name(day: date) -> str:
    """The partition one day's rows live in."""
    return f"{_PARTITION_PREFIX}{day.strftime(PARTITION_DAY_FORMAT)}"


class ResidencyLogRow(BaseModel):
    """One decision, as it is written: the request, and where it actually went.

    Frozen, because the same object is handed to a writer and to whatever logs
    it, and a row edited afterwards would make the in-process writer's contents
    disagree with what was written (`test_a_row_is_frozen`).

    `provider` and `region` name the candidate that SERVED. On a request that
    fell back, that is the second candidate and not the first - the field
    answering "where did the data go" has to name where the data went.
    """

    model_config = ConfigDict(frozen=True)

    ts: datetime
    tenant: str
    data_class: DataClass
    alias: str
    provider: str
    region: str
    allowed: bool


class ResidencyLogWriter(Protocol):
    """Where a decision row goes. `write` is the whole contract.

    A Protocol rather than a base class so that a caller's own writer - a test
    double, and later a batching one - satisfies it without importing anything
    from here to inherit from. `write` is allowed to RAISE: `record_residency` is
    what guards it, in one place, so no implementation carries a try/except of
    its own and no implementation can quietly decide not to.
    """

    def write(self, row: ResidencyLogRow) -> None:
        """File one decision row. May raise; the caller's guard is the only one."""
        ...


class InMemoryResidencyLogWriter:
    """Every row in this process's memory, in the order written.

    What a gateway with no `OPSCONSOLE_DATABASE_URL` builds, which is what lets
    the suite and a bare local run need no backing service - the same trade plan
    09a decision B made for the redaction map store. It is not a store: nothing
    reads it back, nothing expires, and a restart loses everything. That is the
    honest shape for "no database configured", and the reason `/readyz` reports
    `postgres` separately.
    """

    def __init__(self) -> None:
        self._rows: list[ResidencyLogRow] = []
        self._lock = threading.Lock()

    @property
    def rows(self) -> tuple[ResidencyLogRow, ...]:
        """A snapshot, so a caller holding it does not see later rows appear in it."""
        with self._lock:
            return tuple(self._rows)

    def write(self, row: ResidencyLogRow) -> None:
        with self._lock:
            self._rows.append(row)


class PostgresResidencyLogWriter:
    """One INSERT per row, into the day's partition, over a pooled connection.

    The identifier pre-checks run BEFORE any statement: the
    four `text` fields become a bound parameter each, but `provider`'s and
    `region`'s day is not the point - `tenant`, `alias`, `provider` and `region`
    are all strings a request can influence, and this table is the audit trail,
    so a value that could not be a routing identifier has no business being
    recorded as one. `test_write_refuses_a_malformed_identifier_before_any_sql`
    asserts no connection was even borrowed, which is the difference between a
    refused row and a row half-written under a partition name built from the
    same string.

    The `CREATE` and the `INSERT` take SEPARATE connections on purpose. In one
    transaction a refused `CREATE` aborts it, and the `INSERT` that follows fails
    with `InFailedSqlTransaction` - which would lose exactly the row decision B
    exists to keep. Two borrows cost one extra round trip on the first request of
    each day and nothing afterwards.
    """

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)
        self._lock = threading.Lock()
        self._days: set[date] = set()

    def write(self, row: ResidencyLogRow) -> None:
        """Ensure the day's partition, then write the row. Raises only on a bad identifier."""
        checked_identifier(row.tenant, "tenant")
        checked_identifier(row.alias, "alias")
        checked_identifier(row.provider, "provider")
        checked_identifier(row.region, "region")

        self._ensure_partition(row.ts.astimezone(UTC).date())
        with self._pool.connection() as conn:
            conn.execute(
                sql.SQL(
                    "INSERT INTO {} (ts, tenant, data_class, alias, provider, region, allowed) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                ).format(sql.Identifier(self._schema, RESIDENCY_LOG)),
                (
                    row.ts,
                    row.tenant,
                    row.data_class.value,
                    row.alias,
                    row.provider,
                    row.region,
                    row.allowed,
                ),
            )

    def _ensure_partition(self, day: date) -> None:
        """Create this day's partition once per process, and never fail because of it.

        The cache is keyed by the DAY, not by "have I ever created one": the
        second shape passes every same-day test and then sends every row after
        the first midnight into the DEFAULT partition, silently, forever
        (`test_a_row_on_the_next_day_creates_a_second_partition`, standing rule
        34).

        A refused create marks the day done ANYWAY. The alternative is a DDL
        statement per request for the rest of the day against a role that has
        already said no, each one taking an `ACCESS EXCLUSIVE` lock attempt on
        the parent; the DEFAULT partition is the backstop that makes swallowing
        it affordable.

        The `FOR VALUES` bounds are `sql.Literal`, not bound parameters.
        PostgreSQL refuses a placeholder in a partition bound ("there is no
        parameter $1"), and a `CREATE` that always failed sent every row to the
        DEFAULT partition, so the retention purge had whole-day partitions to
        drop and never dropped anything. `sql.Literal` over the two `datetime`
        values builds the bound with no string-joining, so there is nothing to
        inject. Pinned by `test_postgres_writer.py`'s three partition cases and
        `test_cli.py`'s eight purge cases.
        """
        with self._lock:
            if day in self._days:
                return
            self._days.add(day)
        name = partition_name(day)
        start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})"
                    ).format(
                        sql.Identifier(self._schema, name),
                        sql.Identifier(self._schema, RESIDENCY_LOG),
                        sql.Literal(start),
                        sql.Literal(start + timedelta(days=1)),
                    ),
                )
        except Exception:
            logger.exception("residency log partition %s could not be created", name)


def _day_of(partition: str) -> date | None:
    """The day a partition name encodes, or `None` if it does not encode one.

    `None` is what keeps `residency_log_default` alive: it is a child of the
    parent table with no day in its name, and a purge that dropped what it could
    not date would remove the backstop, so the next refused `CREATE` would lose
    its row instead of misfiling it
    (`test_the_default_partition_is_never_dropped`).
    """
    prefix = f"{RESIDENCY_LOG}_"
    if not partition.startswith(prefix):
        return None
    try:
        return datetime.strptime(partition[len(prefix) :], PARTITION_DAY_FORMAT).date()
    except ValueError:
        return None


def _partition_names(conn: Connection[TupleRow], schema: str) -> list[str]:
    """Every child of `residency_log` in this schema, the DEFAULT one included."""
    rows = conn.execute(
        """
        select child.relname
        from pg_inherits i
        join pg_class parent on parent.oid = i.inhparent
        join pg_class child on child.oid = i.inhrelid
        join pg_namespace n on n.oid = parent.relnamespace
        where n.nspname = %s and parent.relname = %s
        """,
        (schema, RESIDENCY_LOG),
    ).fetchall()
    return [str(row[0]) for row in rows]


def purge_partitions(
    conn: Connection[TupleRow],
    *,
    schema: str = DEFAULT_SCHEMA,
    today: date,
    retention_days: int = RESIDENCY_LOG_RETENTION_DAYS,
) -> tuple[str, ...]:
    """Drop the day partitions older than the retention, and answer what was dropped.

    The boundary is inclusive-keep: `today - retention_days` STAYS and
    `today - retention_days - 1` goes, asserted pair by pair in
    `test_the_purge_boundary_is_inclusive_keep`. Both directions of an off-by-one
    here are real damage - one destroys a day of evidence inside the retention
    the project promised, the other keeps everything forever.

    `today` is a parameter rather than a clock read so a caller can say which day
    it means; `retention_days` is one so an operator can run a shorter retention
    deliberately, and `make purge-residency` passes neither, which is why the
    default is checked against the published constant
    (`test_the_shipped_retention_is_the_purges_default`).

    Answers only the partitions it actually dropped, sorted: the CronJob that
    will call this logs the answer, and an operator asking what happened to last
    quarter's rows has to be told the truth.

    DROP rather than DELETE, which is the reason the table is partitioned at all:
    dropping ninety days is a catalogue update, deleting the same rows is a table
    rewrite and a vacuum on the hot write path.
    """
    checked = checked_schema(schema)
    cutoff = today - timedelta(days=retention_days)
    dropped: list[str] = []
    for name in _partition_names(conn, checked):
        day = _day_of(name)
        if day is None or day >= cutoff:
            continue
        conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(checked, name)))
        dropped.append(name)
    return tuple(sorted(dropped))
