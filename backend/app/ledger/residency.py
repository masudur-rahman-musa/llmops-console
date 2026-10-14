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
unchanged is the worse failure. `docs/adr/0004-residency-log-write-is-not-on-the-
request-s-critical-path.md` says what would change that.

**Partitions are made on demand** (decision B), cached per process per day. A
refused `CREATE` is swallowed and the INSERT still runs, landing the row in the
migration's DEFAULT partition: a read-only role costs a badly-placed row, never a
lost one (`test_a_row_whose_partition_cannot_be_created_lands_in_the_default`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Final, Protocol

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from app.core.metrics import record_residency_violation
from app.ledger.db import DEFAULT_SCHEMA, checked_schema
from app.routing.errors import IdentifierInvalid
from app.routing.policy import DataClass, Policy, Residency
from app.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH, Resolved

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

The writer builds this name and `ledger/cli.py`'s purge parses it back, so the
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


def region_allowed(policy: Policy, tenant: str, data_class: DataClass, region: str) -> bool:
    r"""Does this tenant's policy permit this data class in this region.

    The same narrowing `routing/resolver.py` performs when it builds the
    candidate list: the data class rule's `allow_regions`, intersected with
    `policy.eu_regions` when the tenant's residency is EU. Deny by default at
    both levels - an unknown tenant and a data class with no rule each allow
    nothing, never the union of anything else.

    Duplicated from the resolver rather than shared with it on purpose. Extracting
    a helper would mean editing `routing/resolver.py`, and that module is the one
    the residency property test proves; a log that could change the resolver in
    order to describe it is not evidence. What keeps the two honest is that this
    function is only ever asked about a region the resolver ALREADY returned, so
    a disagreement between them shows up as an `allowed=false` row and an alert
    rather than as a silent pass. The rows are
    `test_region_allowed_over_the_shipped_policy` and
    `test_a_data_class_with_no_rule_allows_nothing`.
    """
    tenant_config = policy.tenants.get(tenant)
    if tenant_config is None:
        return False
    rule = tenant_config.data_class_rules.get(data_class)
    allowed = set(rule.allow_regions) if rule is not None else set()
    if tenant_config.residency is Residency.EU:
        allowed &= set(policy.eu_regions)
    return region in allowed


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
        _checked_identifier(row.tenant, "tenant")
        _checked_identifier(row.alias, "alias")
        _checked_identifier(row.provider, "provider")
        _checked_identifier(row.region, "region")

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
                        "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM (%s) TO (%s)"
                    ).format(
                        sql.Identifier(self._schema, name),
                        sql.Identifier(self._schema, RESIDENCY_LOG),
                    ),
                    (start, start + timedelta(days=1)),
                )
        except Exception:
            logger.exception("residency log partition %s could not be created", name)


def _checked_identifier(value: str, field: str) -> str:
    """One value that reaches the audit trail, or `IdentifierInvalid` before any SQL.

    The same cap and the same pattern the resolver applies, imported from it
    rather than restated: these strings arrive from a request that the resolver
    already accepted, so a second, looser rule here would be a way into the table
    that the routing layer has no idea exists. `re.fullmatch`, so a trailing
    newline is refused.
    """
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(f"{field}: longer than {MAX_IDENTIFIER_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: not a lowercase slug")
    return value


def _utcnow() -> datetime:
    """Wall clock, in UTC and timezone-aware.

    The partition a row lands in is chosen from this, so a naive local stamp
    would file an evening request under the wrong day for half the world
    (`test_the_row_is_stamped_from_the_injected_clock`).
    """
    return datetime.now(tz=UTC)


def record_residency(
    log: ResidencyLogWriter,
    *,
    policy: Policy,
    tenant: str,
    data_class: DataClass,
    alias: str,
    chosen: Resolved,
    now: Callable[[], datetime] = _utcnow,
) -> None:
    """Write the row for the candidate that served. Never raises.

    Called once per served request, from `gateway/routes/chat.py::record_served`,
    which both the chat paths and `/v1/embeddings` go through at the moment the
    serving candidate is known - and, on a stream, before the first frame is on
    the wire.

    `allowed` is computed here from `policy`. A `false` row moves
    `gateway_residency_violations_total` and nothing else does: a 451 is the
    resolver refusing CORRECTLY and is counted by `gateway_requests_total`
    instead, which is what makes the violation counter
    permanently zero by construction and alertable at greater than zero.

    The write is guarded (decision D): an exception is logged and the request is
    unchanged on all three paths. The counter moves before the write, so a
    violation is still visible when the database is the thing that is broken.
    """
    row = ResidencyLogRow(
        ts=now(),
        tenant=tenant,
        data_class=data_class,
        alias=alias,
        provider=chosen.provider,
        region=chosen.region,
        allowed=region_allowed(policy, tenant, data_class, chosen.region),
    )
    if not row.allowed:
        record_residency_violation(tenant=tenant)
    try:
        log.write(row)
    except Exception:
        logger.exception(
            "residency log write failed for tenant %s, provider %s", tenant, chosen.provider
        )
