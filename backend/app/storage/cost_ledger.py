r"""The cost ledger: one row per request, attributing spend to a tenant, a model and a day.

What this answers is "what did this cost, and who spent it", per request, for as
long as the table is kept - which is forever, deliberately. `residency_log` is
partitioned because it has a ninety-day retention and dropping a partition beats
deleting a quarter of a table; this one has no retention at all, so it is a plain
table and there is no purge command to go with it.

**The cost is computed HERE, never passed in** (decision C), exactly as
`record_residency` computes `allowed` rather than accepting it. `record_cost`
takes the `PriceTable` and the usage and prices the request through
`app.limits.cost.cost_for`. A caller filling `cost_usd` would be recording
its own belief rather than the price table's answer, and it would let the number
in the row disagree with the `price_version` beside it. Pinned by
`test_the_cost_is_priced_from_the_table_and_cannot_be_supplied_by_the_caller`.

**A failed write changes nothing about the request** (decision E, the same rule
as an earlier change decision D and a design note). `record_cost` logs at ERROR and returns.
The provider has already charged by the time this runs, so a Postgres blip
turning a served request into a 500 would lose the answer as well as the row.
What that costs is stated rather than hidden: a cost row can be lost, so the
ledger is a LOWER BOUND on spend during a database incident, and the counter
`gateway_cost_usd_total` is the number that keeps moving regardless - the route
moves it from the row this function ANSWERS, which is why a
lost write still shows up as spend on a dashboard. Pinned by
`test_a_failing_write_is_swallowed`.

**`model` has its own rule, not the resolver's** (decision D). The shipped
`routing.yaml` names `qwen2.5-3b-instruct` and `bge-small-en-v1.5`, and both fail
`routing.resolver.IDENTIFIER_PATTERN` because of the dots, so reusing it here
would refuse - and therefore silently lose - every ledger row the local provider
produced.  `test_every_model_the_shipped_policy_can_resolve_to_passes_the_pattern`
is what makes a routing change that adds an unrepresentable id fail in the suite
rather than in production.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from datetime import datetime
from decimal import Decimal
from typing import Final, Protocol

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from app.core.metrics import record_ledger_write
from app.core.validation import IdentifierInvalid, checked_identifier
from app.limits.cost import COST_QUANTUM
from app.storage.db import DEFAULT_SCHEMA, checked_schema

logger = logging.getLogger(__name__)

COST_LEDGER: Final[str] = "cost_ledger"
"""The table `migrations/002_cost_ledger.sql` creates. Pinned by
`test_the_table_names_are_the_ones_the_migration_creates`."""

COST_LEDGER_DAILY: Final[str] = "cost_ledger_daily"
"""The materialised view the console reads, refreshed by `make refresh-rollup`.
Pinned by the same case."""

MAX_MODEL_LENGTH: Final[int] = 128
"""The cap on a provider's model id, and deliberately not the resolver's 64.

A model id is a vendor's string rather than one of ours: a fine-tune carries an
account and a suffix, and a hosted open-weights id carries a publisher prefix.
Pinned exactly by `test_the_model_and_price_version_caps_are_pinned`, with the
inclusive boundary in `test_a_model_at_exactly_the_length_cap_is_accepted`.
"""

MODEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
"""What a provider's model id may contain: letters, digits, `.`, `:`, `_`, `-`
and `/`, starting with a letter or a digit.

Wider than `routing.resolver.IDENTIFIER_PATTERN` on purpose (decision D) and
still narrow enough that nothing here can reach a statement as anything but a
bound parameter - no quote, no semicolon, no whitespace, no backslash and no
percent. `re.fullmatch`, so a trailing newline is refused.
The two directions are `test_write_refuses_a_malformed_model_before_any_sql` and
`test_the_model_pattern_admits_the_shapes_provider_ids_actually_take`.
"""

MAX_PRICE_VERSION_LENGTH: Final[int] = 256
"""The cap on the version string a row records.

Built by `app.limits.prices.price_version` from a price key and a date
rather than by a request, so it is only as bounded as the key that fed it - which
is why it is capped anyway. Pinned exactly by
`test_the_model_and_price_version_caps_are_pinned`.
"""

TRACE_ID_LENGTH: Final[int] = 32
"""A W3C trace-context trace id, in hex: 16 bytes, 32 characters. The console
joins a ledger row to its trace on this, so a row carrying anything else joins to
nothing. Pinned by `test_a_trace_id_is_exactly_thirty_two_characters`."""

TRACE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")
"""Exactly `TRACE_ID_LENGTH` lowercase hex characters, matched with
`re.fullmatch`. The length is in the pattern as well as in the constant because
the three sharpest rejection rows - a trailing newline, CRLF and space at the
exact total length - are refused by the pattern and by nothing else
(`test_write_refuses_a_malformed_trace_id_before_any_sql`)."""

CACHED_COST_USD: Final[Decimal] = Decimal(0).quantize(COST_QUANTUM)
"""What a request served from a cache costs: zero, at the ledger's six places.

Decision B of an earlier change, and it is a statement about what one column MEANS rather
than a saving: `sum(cost_usd)` is spend, so it reconciles against an invoice with
no `WHERE cached = false` and a console page that forgot that filter cannot
overstate a tenant's bill. Nothing is lost - the row still carries the replayed
answer's tokens and the price version in force, so the cost the hit AVOIDED is
derivable from the row exactly. Quantized from `COST_QUANTUM` rather than written
as a literal, so the zero has the same exponent every other cost on the table
has. Pinned by `test_a_cached_request_is_booked_at_zero_with_the_price_version_in_force`
and `test_a_cached_row_keeps_the_replayed_answers_token_counts`.
"""

LEDGER_QUEUE_SIZE: Final[int] = 10000
"""How many rows the background queue may hold before new ones are dropped.

Decided without asking: 10 000 absorbs a burst of concurrent requests
without growing without bound. A full queue means the database cannot keep up;
the row is counted as `dropped` and the chat reply still succeeds. Pinned by
`test_the_ledger_queue_size_is_exactly_10000`.
"""

_SENTINEL: object = object()
"""Put on the queue to tell the worker thread to drain and stop."""


class CostLedgerRow(BaseModel):
    """One request's cost, as it is written: the design's columns and nothing else.

    Frozen, because the same object is handed to a writer and to whatever counts
    it, and a `cost_usd` edited afterwards would make the ledger disagree with
    the number already reported for that request (`test_a_row_is_frozen`).

    `provider` and `model` name what SERVED. On a request that fell back, that is
    the second candidate: the field answering "what did this cost" has to name
    the thing that charged for it.

    `session` is the one nullable field. A request with no session header is
    still a request that cost money, and refusing it here would lose exactly
    those rows.
    """

    model_config = ConfigDict(frozen=True)

    ts: datetime
    tenant: str
    alias: str
    provider: str
    model: str
    region: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost_usd: Decimal
    price_version: str
    estimated: bool
    session: str | None
    trace_id: str


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

    What a gateway with no `OPSCONSOLE_DATABASE_URL` builds, the same trade plan
    09a decision B made for the redaction map store and an earlier change for the
    residency log. It is not a store: nothing reads it back, nothing expires, and
    a restart loses everything. Rows are kept in a LIST and never keyed, because
    spend is a sum over them and a writer that deduplicated by tenant or by trace
    id would under-report every tenant that asked the same thing twice
    (`test_the_in_memory_writer_keeps_every_row_in_the_order_written`).
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

    An earlier change: `write` enqueues and returns at once so a slow or stuck database
    cannot hold a chat reply. A daemon thread drains the queue and does the
    actual INSERT; a full queue drops the row and counts it as `dropped` rather
    than blocking. Each outcome — ok, error, dropped — increments
    `gateway_ledger_writes_total{result}`.

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
        checked_identifier(row.region, "region")
        if row.session is not None:
            checked_identifier(row.session, "session")
        _checked_model(row.model)
        _checked_trace_id(row.trace_id)
        _checked_price_version(row.price_version)

        try:
            self._queue.put_nowait(row)
        except queue.Full:
            record_ledger_write(result="dropped")

    def flush(self) -> None:
        """Block until every row currently in the queue has been processed.

        Sends a sentinel and waits for it to come back through the worker,
        so every row ahead of it is either inserted or counted as an error
        before this returns. Useful in tests and in shutdown sequences where
        the caller needs a clean queue before reading back from the database.
        """
        done = threading.Event()
        self._queue.put(done)
        done.wait()

    def close(self) -> None:
        """Drain the queue then stop the worker thread.

        Sends the module-level sentinel and joins the thread, which means this
        call blocks until every pending row has been attempted. Safe to call
        more than once; the second call joins a thread that is already done.
        """
        self._queue.put(_SENTINEL)
        self._worker.join()

    # ------------------------------------------------------------------
    # background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Drain the queue until the sentinel arrives."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, threading.Event):
                # flush() barrier: signal the waiter and keep running.
                item.set()
                continue
            self._insert(item)

    def _insert(self, row: CostLedgerRow) -> None:
        """Insert one row and count the outcome. Never raises."""
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    sql.SQL(
                        "INSERT INTO {} ("
                        "ts, tenant, alias, provider, model, region, input_tokens, "
                        "output_tokens, cached, cost_usd, price_version, estimated, "
                        "session, trace_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(sql.Identifier(self._schema, COST_LEDGER)),
                    (
                        row.ts,
                        row.tenant,
                        row.alias,
                        row.provider,
                        row.model,
                        row.region,
                        row.input_tokens,
                        row.output_tokens,
                        row.cached,
                        row.cost_usd,
                        row.price_version,
                        row.estimated,
                        row.session,
                        row.trace_id,
                    ),
                )
            record_ledger_write(result="ok")
        except Exception:
            logger.exception("ledger write failed; row for tenant=%s dropped", row.tenant)
            record_ledger_write(result="error")


def _checked_model(value: str) -> str:
    """The provider's model id, against this module's own rule (decision D)."""
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
    the first - and the pair is
    `test_the_first_refresh_populates_and_the_second_runs_concurrently`.

    Takes the POOL rather than a connection, unlike `purge_partitions`:
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
