r"""The cost ledger: one row per request, attributing spend to a tenant, a model and a day.

What this answers is "what did this cost, and who spent it", per request, for as
long as the table is kept - which is forever, deliberately. It has no retention
at all, so it is a plain table and there is no purge command to go with it.

**The cost is computed HERE, never passed in**, like
the price version. `record_cost`
takes the `PriceTable` and the usage and prices the request through
`app.limits.cost.cost_for`. A caller filling `cost_usd` would be recording
its own belief rather than the price table's answer, and it would let the number
in the row disagree with the `price_version` beside it.

**A failed write changes nothing about the request** (the same rule
as for every write). The provider has already charged by the time this
runs, so a Postgres blip turning a served request into a 500 would lose the
answer as well as the row. A failed write is not quietly lost either:
a row nothing landed for is buffered and sent again, and a row that is dropped is
counted on `gateway_ledger_writes_total` under a `dropped_*` value, which fires
the `LedgerWriteDropped` alert. The ledger can still be a LOWER BOUND on spend
after such an alert, and the counter `gateway_cost_usd_total` is the number that
keeps moving regardless - the route moves it from the row `record_cost` ANSWERS.

**`model` has its own rule, not the resolver's**. The shipped
`routing.yaml` names `qwen2.5-3b-instruct` and `bge-small-en-v1.5`, and both fail
`routing.resolver.IDENTIFIER_PATTERN` because of the dots, so reusing it here
would refuse - and therefore silently lose - every ledger row the local provider
produced.  A test is what makes a routing change that adds an unrepresentable id fail in the suite
rather than in production.
"""

from __future__ import annotations

import enum
import logging
import queue
import re
import threading
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final, Protocol

import psycopg
from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from app.core.metrics import record_ledger_write
from app.core.validation import IdentifierInvalid, checked_identifier, checked_user_id
from app.limits.budget import to_micro_usd
from app.limits.cost import COST_QUANTUM
from app.storage.db import DEFAULT_SCHEMA, checked_schema

logger = logging.getLogger(__name__)

COST_LEDGER: Final[str] = "cost_ledger"
"""The table `migrations/002_cost_ledger.sql` creates."""

COST_LEDGER_DAILY: Final[str] = "cost_ledger_daily"
"""The materialised view the console reads, refreshed by `make refresh-rollup`.
Pinned by the same case."""

MAX_MODEL_LENGTH: Final[int] = 128
"""The cap on a provider's model id, and deliberately not the resolver's 64.

A model id is a vendor's string rather than one of ours: a fine-tune carries an
account and a suffix, and a hosted open-weights id carries a publisher prefix.
The caps and the inclusive boundary are pinned in the tests.
"""

MODEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
"""What a provider's model id may contain: letters, digits, `.`, `:`, `_`, `-`
and `/`, starting with a letter or a digit.

Wider than `routing.resolver.IDENTIFIER_PATTERN` on purpose and
still narrow enough that nothing here can reach a statement as anything but a
bound parameter - no quote, no semicolon, no whitespace, no backslash and no
percent. `re.fullmatch`, so a trailing newline is refused.
"""

MAX_PRICE_VERSION_LENGTH: Final[int] = 256
"""The cap on the version string a row records.

Built by `app.limits.prices.price_version` from a price key and a date
rather than by a request, so it is only as bounded as the key that fed it - which
is why it is capped anyway.
"""

TRACE_ID_LENGTH: Final[int] = 32
"""A W3C trace-context trace id, in hex: 16 bytes, 32 characters. The console
joins a ledger row to its trace on this, so a row carrying anything else joins to
nothing.."""

TRACE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")
"""Exactly `TRACE_ID_LENGTH` lowercase hex characters, matched with
`re.fullmatch`. The length is in the pattern as well as in the constant because
the three sharpest rejection rows - a trailing newline, CRLF and space at the
exact total length - are refused by the pattern and by nothing else."""

CACHED_COST_USD: Final[Decimal] = Decimal(0).quantize(COST_QUANTUM)
"""What a request served from a cache costs: zero, at the ledger's six places.

This is a statement about what one column MEANS rather
than a saving: `sum(cost_usd)` is spend, so it reconciles against an invoice with
no `WHERE cached = false` and a console page that forgot that filter cannot
overstate a tenant's bill. Nothing is lost - the row still carries the replayed
answer's tokens and the price version in force, so the cost the hit AVOIDED is
derivable from the row exactly. Quantized from `COST_QUANTUM` rather than written
as a literal, so the zero has the same exponent every other cost on the table
has.
"""

LEDGER_QUEUE_SIZE: Final[int] = 10000
"""How many rows the input queue, and separately the retry buffer, may hold.

Decided without asking: 10 000 absorbs a burst of concurrent requests, or a
database outage of a few minutes, without growing without bound. A row that fits
in neither is counted as `dropped_full` and the chat reply still succeeds.
Pinned by `test_the_buffer_holds_exactly_10000_rows`.
"""

LEDGER_RETRY_WAIT_START: Final[float] = 1.0
LEDGER_RETRY_WAIT_CAP: Final[float] = 30.0
"""The worker's wait before it tries the retry buffer again: 1s, doubling to a
30s cap, and back to 1s once a write lands."""


def next_retry_wait(current: float | None) -> float:
    """The wait after `current`: the start when there was none, else double, capped.

    Pure, so the sequence is tested without a clock
    (`test_the_retry_wait_doubles_from_one_second_to_a_thirty_second_cap`).
    """
    if current is None:
        return LEDGER_RETRY_WAIT_START
    return min(current * 2, LEDGER_RETRY_WAIT_CAP)


class _Outcome(enum.Enum):
    """What one INSERT attempt came to, sorted by where it failed."""

    LANDED = "ok"
    RETRY = "retry"
    INVALID = "dropped_invalid"
    UNSURE = "dropped_unsure"


_SENTINEL: object = object()
"""Put on the queue to tell the worker thread to drain and stop."""


class CostLedgerRow(BaseModel):
    """One request's cost, as it is written: the spec's columns and nothing else.

    Frozen, because the same object is handed to a writer and to whatever counts
    it, and a `cost_usd` edited afterwards would make the ledger disagree with
    the number already reported for that request.

    `provider` and `model` name what SERVED. On a request that fell back, that is
    the second candidate: the field answering "what did this cost" has to name
    the thing that charged for it.

    `session` and `prompt_version` are the nullable fields. A request with no
    session or prompt version header is still a request that cost money, and
    refusing it here would lose exactly those rows. `user_id` is nullable for the
    same reason: a request with no `X-User` is booked under no user.

    `status` is the HTTP answer (200 for a served request, 5xx for one that failed after
    routing) and `latency_ms` the time it took, null when it was not measured.
    """

    model_config = ConfigDict(frozen=True)

    ts: datetime
    tenant: str
    alias: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost_usd: Decimal
    price_version: str
    estimated: bool
    session: str | None
    trace_id: str
    prompt_version: str | None = None
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    status: int = 200
    latency_ms: int | None = None
    user_id: str | None = None


class CostLedgerWriter(Protocol):
    """Where a cost row goes. `write` is the whole contract.

    A Protocol rather than a base class so a caller's own writer - a test double,
    and later a batching one - satisfies it without importing anything from here
    to inherit from. `write` is allowed to RAISE: `record_cost` is what guards it,
    in one place, so no implementation carries a try/except of its own and no
    implementation can quietly decide not to.
    """

    def write(self, row: CostLedgerRow) -> None:
        """File one cost row. May raise; the caller's guard is the only one."""
        ...


class InMemoryCostLedgerWriter:
    """Every row in this process's memory, in the order written.

    What a gateway with no `OPSCONSOLE_DATABASE_URL` builds, the same trade made
    for the redaction map store. It is not a store: nothing reads it
    back, nothing expires, and
    a restart loses everything. Rows are kept in a LIST and never keyed, because
    spend is a sum over them and a writer that deduplicated by tenant or by trace
    id would under-report every tenant that asked the same thing twice.
    """

    def __init__(self) -> None:
        self._rows: list[CostLedgerRow] = []
        self._lock = threading.Lock()

    @property
    def rows(self) -> tuple[CostLedgerRow, ...]:
        """A snapshot, so a caller holding it does not see later rows appear in it."""
        with self._lock:
            return tuple(self._rows)

    def write(self, row: CostLedgerRow) -> None:
        with self._lock:
            self._rows.append(row)


class PostgresCostLedgerWriter:
    """One INSERT per row via a bounded background queue. No DDL, ever.

    `write` enqueues and returns at once so a slow or stuck database
    cannot hold a chat reply. A daemon thread drains the queue and does the
    actual INSERT. Where a failed INSERT failed decides what happens to the row
    (a design note, pinned in `tests/storage/test_ledger_retry.py`):

    * getting a connection, or the INSERT lost with its connection: nothing was
      committed, so the row goes to a retry buffer and is sent again;
    * the database refusing the row: it will never pass, `dropped_invalid`;
    * the commit failing: the row may be in the table, so it is `dropped_unsure`
      and never sent again, or the money would be counted twice.

    Rows land in the order written: while the buffer holds anything, a new row
    joins its end instead of going ahead of it. The worker tries the buffer
    after `next_retry_wait`, stopping at the first row that still cannot be
    sent. A row that fits in neither the queue nor the buffer is `dropped_full`.
    Every outcome increments `gateway_ledger_writes_total{result}`.

    Pre-checks still run BEFORE enqueuing. A row that fails
    a pre-check is still raised immediately, because a malformed row is a bug
    in the caller, not a database slowdown. `write` is the only method the
    `CostLedgerWriter` protocol requires; `flush` and `close` are extras the
    app's shutdown and the tests use.
    """

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)
        self._queue: queue.Queue[CostLedgerRow | object] = queue.Queue(maxsize=LEDGER_QUEUE_SIZE)
        self._buffer: deque[CostLedgerRow] = deque()
        """Rows waiting to be sent again. Touched only by the worker thread."""
        self._wait: float | None = None
        self._worker = threading.Thread(target=self._run, daemon=True, name="ledger-writer")
        self._worker.start()

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def write(self, row: CostLedgerRow) -> None:
        """Enqueue one row. Raises only on a value that fails a pre-check.

        Pre-checks run here on the caller's thread so a malformed row is
        refused immediately with no silent loss. The INSERT is deferred to
        the background worker, so this call returns without waiting for the
        database.
        """
        checked_identifier(row.tenant, "tenant")
        checked_identifier(row.alias, "alias")
        checked_identifier(row.provider, "provider")
        if row.session is not None:
            checked_identifier(row.session, "session")
        if row.prompt_version is not None:
            checked_identifier(row.prompt_version, "prompt_version")
        if row.user_id is not None:
            checked_user_id(row.user_id)
        _checked_model(row.model)
        _checked_trace_id(row.trace_id)
        _checked_price_version(row.price_version)

        try:
            self._queue.put_nowait(row)
        except queue.Full:
            logger.error("ledger queue full; row for tenant=%s dropped", row.tenant)
            record_ledger_write(result="dropped_full")

    def flush(self) -> None:
        """Block until every queued row is handled and the buffer was tried once.

        Sends a barrier and waits for it to come back through the worker, so
        every row ahead of it is inserted, buffered or counted before the worker
        tries the buffer once, with no wait, and lets this return. Useful in
        tests and in shutdown sequences where the caller needs a clean queue
        before reading back from the database.
        """
        done = threading.Event()
        self._queue.put(done)
        done.wait()

    def close(self) -> None:
        """Drain the queue, try the buffer once, then stop the worker thread.

        Sends the module-level sentinel and joins the thread, which means this
        call blocks until every pending row has been attempted. A row still
        buffered after that one try is counted `dropped_shutdown`; close never
        waits out the retry wait. Safe to call more than once; the second call
        joins a thread that is already done.
        """
        self._queue.put(_SENTINEL)
        self._worker.join()

    # ------------------------------------------------------------------
    # background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Drain the queue until the sentinel arrives, trying the buffer between rows."""
        while True:
            try:
                item = self._queue.get(timeout=self._wait if self._buffer else None)
            except queue.Empty:
                self._send_buffered()
                continue
            if item is _SENTINEL:
                self._send_buffered()
                for row in self._buffer:
                    logger.error("ledger writer stopped; row for tenant=%s dropped", row.tenant)
                    record_ledger_write(result="dropped_shutdown")
                self._buffer.clear()
                break
            if isinstance(item, threading.Event):
                # flush() barrier: try the buffer once, signal the waiter, keep running.
                self._send_buffered()
                item.set()
                continue
            if isinstance(item, CostLedgerRow):
                if self._buffer:
                    self._to_buffer(item)
                else:
                    self._settle(item, self._insert(item))

    def _send_buffered(self) -> None:
        """Send buffered rows in order, stopping at the first that still cannot go."""
        while self._buffer:
            row = self._buffer[0]
            outcome = self._insert(row)
            if outcome is _Outcome.RETRY:
                self._wait = next_retry_wait(self._wait)
                return
            self._buffer.popleft()
            self._settle(row, outcome)

    def _settle(self, row: CostLedgerRow, outcome: _Outcome) -> None:
        """Count one attempt's outcome, and buffer a row nothing landed for."""
        if outcome is _Outcome.RETRY:
            self._to_buffer(row)
            return
        if outcome is _Outcome.LANDED:
            self._wait = None
        record_ledger_write(result=outcome.value)

    def _to_buffer(self, row: CostLedgerRow) -> None:
        """Buffer a row to send again, counted `retry` once, or `dropped_full`."""
        if len(self._buffer) >= LEDGER_QUEUE_SIZE:
            logger.error("ledger retry buffer full; row for tenant=%s dropped", row.tenant)
            record_ledger_write(result="dropped_full")
            return
        if not self._buffer:
            logger.warning("ledger writes failing; buffering rows to send again")
            self._wait = next_retry_wait(None)
        self._buffer.append(row)
        record_ledger_write(result="retry")

    def _insert(self, row: CostLedgerRow) -> _Outcome:
        """Insert one row and sort the outcome by where it failed. Never raises.

        The stage is tracked by hand because the same exception class means
        different things at different places: an `OperationalError` from the
        INSERT is a lost connection with nothing committed, while one raised on
        leaving the block is a failed commit that may have landed
        (`test_a_failed_commit_is_dropped_unsure_and_never_sent_again`).
        """
        stage = "connect"
        try:
            with self._pool.connection() as conn:
                stage = "insert"
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} ("
                        "ts, tenant, alias, provider, model, input_tokens, "
                        "output_tokens, cached, cost_usd, price_version, estimated, "
                        "session, trace_id, prompt_version, cache_read_tokens, cache_write_tokens, "
                        "status, latency_ms, user_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                        "%s, %s, %s)"
                    ).format(sql.Identifier(self._schema, COST_LEDGER)),
                    (
                        row.ts,
                        row.tenant,
                        row.alias,
                        row.provider,
                        row.model,
                        row.input_tokens,
                        row.output_tokens,
                        row.cached,
                        row.cost_usd,
                        row.price_version,
                        row.estimated,
                        row.session,
                        row.trace_id,
                        row.prompt_version,
                        row.cache_read_tokens,
                        row.cache_write_tokens,
                        row.status,
                        row.latency_ms,
                        row.user_id,
                    ),
                )
                stage = "commit"
        except Exception as exc:
            lost = isinstance(exc, psycopg.OperationalError | psycopg.InterfaceError)
            if stage == "connect" or (stage == "insert" and lost):
                return _Outcome.RETRY
            outcome = _Outcome.INVALID if stage == "insert" else _Outcome.UNSURE
            logger.error(
                "ledger write %s; row for tenant=%s dropped: %s",
                outcome.value,
                row.tenant,
                f"{type(exc).__name__}: {exc}",
            )
            return outcome
        return _Outcome.LANDED


def _checked_model(value: str) -> str:
    """The provider's model id, against this module's own rule."""
    if len(value) > MAX_MODEL_LENGTH:
        raise IdentifierInvalid(f"model: longer than {MAX_MODEL_LENGTH} characters")
    if MODEL_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("model: not a provider model id")
    return value


def _checked_trace_id(value: str) -> str:
    """The trace id, against 32 lowercase hex characters."""
    if TRACE_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"trace_id: not {TRACE_ID_LENGTH} lowercase hex characters")
    return value


def _checked_price_version(value: str) -> str:
    """The price version, against its cap. It has no pattern: `limits.prices` builds
    it, and the two halves it is built from are checked where they arrive."""
    if len(value) > MAX_PRICE_VERSION_LENGTH:
        raise IdentifierInvalid(f"price_version: longer than {MAX_PRICE_VERSION_LENGTH} characters")
    return value


class LedgerUnreadable(Exception):
    """The ledger could not be read, so a month's spend is not known.

    Raised by `PostgresMonthSpendReader` and turned into 503 `BUDGET_UNAVAILABLE` by the
    budget check: counting from zero instead would hand the tenant a fresh budget."""


class MonthSpendReader(Protocol):
    """One UTC month's not-cached spend in micro-USD, the number a budget counter loads.

    `None` means there is no ledger to read (no database). The same rows the console's
    budget bars add up, so the two numbers agree."""

    def tenant_month_spend(self, tenant: str, month: str) -> int | None: ...

    def user_month_spend(self, tenant: str, user_id: str, month: str) -> int | None: ...


class NoDatabaseMonthSpendReader:
    """What a gateway with no database reads: nothing, so the counter stands alone."""

    def tenant_month_spend(self, tenant: str, month: str) -> int | None:
        return None

    def user_month_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        return None


class PostgresMonthSpendReader:
    """Sums `cost_usd` over one tenant's (or user's) not-cached rows of one UTC month.

    Any `psycopg.Error` becomes `LedgerUnreadable`. Pinned by
    `tests/storage/test_ledger_month_spend_postgres.py` and `test_ledger_month_spend.py`.
    """

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)

    def tenant_month_spend(self, tenant: str, month: str) -> int | None:
        checked_identifier(tenant, "tenant")
        return self._sum(month, sql.SQL(""), (tenant,))

    def user_month_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        checked_identifier(tenant, "tenant")
        checked_user_id(user_id)
        return self._sum(month, sql.SQL(" AND user_id = %s"), (tenant, user_id))

    def _sum(self, month: str, extra: sql.Composable, params: tuple[str, ...]) -> int:
        start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
        end = (
            start.replace(year=start.year + 1, month=1)
            if start.month == 12
            else (start.replace(month=start.month + 1))
        )
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    sql.SQL(
                        "SELECT coalesce(sum(cost_usd), 0) FROM {} "
                        "WHERE NOT cached AND tenant = %s{} AND ts >= %s AND ts < %s"
                    ).format(sql.Identifier(self._schema, COST_LEDGER), extra),
                    (*params, start, end),
                ).fetchone()
        except psycopg.Error as exc:
            raise LedgerUnreadable(f"{type(exc).__name__}: {exc}") from exc
        total = Decimal(row[0]) if row is not None else Decimal(0)
        return to_micro_usd(total)


def refresh_daily_rollup(
    pool: ConnectionPool[Connection[TupleRow]],
    *,
    schema: str = DEFAULT_SCHEMA,
) -> bool:
    """Refresh `cost_ledger_daily`, and answer whether it went CONCURRENTLY.

    Two shapes, and which one runs is not a preference. A materialised view that
    has never been populated CANNOT be refreshed concurrently, so a command that
    always used `CONCURRENTLY` would fail on exactly the database where it is run
    for the first time; one that never used it takes an `ACCESS EXCLUSIVE` lock
    for the length of every later refresh, and the console reads nothing while it
    is held. So the first refresh is the plain one and every later refresh is
    concurrent, decided from `pg_class.relispopulated` rather than from a flag
    someone has to set. Both are idempotent - the second answers the same rows as
    the first.

    Takes the POOL rather than a connection:
    `REFRESH MATERIALIZED VIEW CONCURRENTLY` cannot run inside a transaction
    block and psycopg opens one on the first statement, so the code that borrows
    the connection has to be the code that sets autocommit.

    The concurrent refresh is also what a UNIQUE index over the whole group key
    buys: without one PostgreSQL refuses it at runtime, and only at runtime.
    """
    checked = checked_schema(schema)
    with pool.connection() as conn:
        conn.autocommit = True
        row = conn.execute(
            """
            select c.relispopulated
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relname = %s
            """,
            (checked, COST_LEDGER_DAILY),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"{checked}.{COST_LEDGER_DAILY} does not exist; run `make migrate` first"
            )
        concurrently = bool(row[0])
        statement = (
            "REFRESH MATERIALIZED VIEW CONCURRENTLY {}"
            if concurrently
            else "REFRESH MATERIALIZED VIEW {}"
        )
        conn.execute(sql.SQL(statement).format(sql.Identifier(checked, COST_LEDGER_DAILY)))
    return concurrently
