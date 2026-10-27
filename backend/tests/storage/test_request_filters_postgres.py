"""Status, latency and the request filters, against a real server.

The `db` file of an earlier change. The migration, the percentile and the filtered SQL are things only
PostgreSQL can show. Every case runs the SHIPPED migrations into a throwaway schema (the
fixtures in tests/storage/conftest.py, unchanged) and inserts rows with raw SQL, so a column
the SQL file and the reader misname between them is a failure here.

The contract these cases are written against, before it exists:

* `migrations/009_ledger_status_latency.sql`: `cost_ledger.status smallint NOT NULL DEFAULT
  200`, `cost_ledger.latency_ms integer` (null), and `cost_ledger_daily` rebuilt over the rows
  with `status < 500`, WITH data;
* `CostLedgerRow.status: int = 200` and `.latency_ms: int | None = None`, both written;
* `PostgresConsoleReader.recent_requests(*, page, filters, days, now)`; `RequestsRead` gains
  `slow_ms` (`percentile_cont(0.95)` of the window's non-null latencies), `tenants` and
  `models` (distinct, sorted, the window's rows); filters are ANDed and narrow only the list
  and its page count; errors is `status >= 500`; slow is `latency_ms > slow_ms`; day is the
  UTC date of `ts`;
* home requests today, the hit rate, requests per day, tokens per model, cost per alias and
  the cost days count rows with `status < 500` only.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

import pytest
from psycopg import sql

from app.limits.prices import PriceTable
from app.model_routing.provider_files import build_price_table, load_provider_files
from app.storage import CostLedgerRow, PostgresCostLedgerWriter
from app.storage.dashboard_reads import (
    PAGE_SIZE,
    PostgresConsoleReader,
    RequestFilters,
)
from app.storage.db import apply_migrations

from .conftest import Pool, requires_postgres

pytestmark = [requires_postgres, pytest.mark.db]

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_PROVIDERS = REPO_ROOT / "config" / "providers"

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
DAY = date(2026, 10, 27)
DAY_START = datetime(2026, 10, 27, tzinfo=UTC)
PREVIOUS_START = DAY_START - timedelta(days=1)
WINDOW = 30
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
NINE = "009_ledger_status_latency"


@pytest.fixture(scope="module")
def shipped_prices() -> PriceTable:
    """The shipped price table, read once."""
    return build_price_table(load_provider_files(SHIPPED_PROVIDERS))


def insert_row(
    pool: Pool,
    schema: str,
    /,
    *,
    ts: datetime,
    tenant: str = "acme",
    alias: str = "agent-default",
    provider: str = "anthropic",
    model: str = "m1",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached: bool = False,
    cost_usd: str = "0.001000",
    status: int = 200,
    latency_ms: int | None = None,
) -> None:
    """One `cost_ledger` row, with the two columns migration 009 adds."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id, "
                "status, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'v', false, null, %s, %s, %s)"
            ).format(sql.Identifier(schema, "cost_ledger")),
            (
                ts,
                tenant,
                alias,
                provider,
                model,
                input_tokens,
                output_tokens,
                cached,
                Decimal(cost_usd),
                TRACE_ID,
                status,
                latency_ms,
            ),
        )


def ids_of(pool: Pool, schema: str, filters: RequestFilters, *, page: int = 1) -> list[int]:
    """The `input_tokens` (used as a row id) of one page under the filters, sorted."""
    read = PostgresConsoleReader(pool, schema=schema).recent_requests(
        page=page, filters=filters, days=WINDOW, now=NOW
    )
    return sorted(row.input_tokens for row in read.rows)


# --- the migration ---------------------------------------------------------------------


def test_migration_keeps_old_rows_as_served(
    pool: Pool, schema: str, shipped_prices: PriceTable
) -> None:
    """Catches a migration that adds `status` without the default 200 (an old row reads
    null or fails the NOT NULL), makes `latency_ms` NOT NULL or gives it a default (old rows
    would read a made-up latency), types the columns wrongly, or lets 5xx rows into the cost
    days. A row is written under files 001 to 008 only, then 009 runs: the row must read 200
    and null and its day must still show in the cost days, read from the ledger. A 499 row is
    counted and a 500 row is not."""
    version_table = sql.Identifier(schema, "schema_migrations")
    with pool.connection() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(
            sql.SQL(
                "CREATE TABLE {} (version text PRIMARY KEY, "
                "applied_at timestamptz NOT NULL DEFAULT now())"
            ).format(version_table)
        )
        # Pretend file 009 ran, so the runner applies 001 to 008 and stops short of it.
        conn.execute(sql.SQL("INSERT INTO {} (version) VALUES (%s)").format(version_table), (NINE,))
        apply_migrations(conn, schema=schema)
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id) "
                "VALUES (%s, 'acme', 'agent-default', 'anthropic', 'm1', 1, 1, false, 0.5, "
                "'v', false, null, %s)"
            ).format(sql.Identifier(schema, "cost_ledger")),
            (DAY_START, TRACE_ID),
        )
        conn.execute(sql.SQL("DELETE FROM {} WHERE version = %s").format(version_table), (NINE,))

        applied = apply_migrations(conn, schema=schema)

    assert applied == (NINE,)
    with pool.connection() as conn:
        columns = conn.execute(
            "select column_name, data_type, is_nullable, column_default "
            "from information_schema.columns where table_schema = %s "
            "and table_name = 'cost_ledger' and column_name in ('status', 'latency_ms') "
            "order by column_name",
            (schema,),
        ).fetchall()
        old = conn.execute(
            sql.SQL("select status, latency_ms from {}").format(
                sql.Identifier(schema, "cost_ledger")
            )
        ).fetchall()
    assert [tuple(c) for c in columns] == [
        ("latency_ms", "integer", "YES", None),
        ("status", "smallint", "NO", "200"),
    ]
    assert [tuple(r) for r in old] == [(200, None)]
    reader = PostgresConsoleReader(pool, schema=schema)
    kept = reader.cost(days=WINDOW, tenant=None, now=NOW, prices=shipped_prices).days
    assert [(day.day, day.requests) for day in kept] == [(DAY, 1)]

    insert_row(pool, schema, ts=DAY_START, status=499)
    insert_row(pool, schema, ts=DAY_START, status=500)
    counted = reader.cost(days=WINDOW, tenant=None, now=NOW, prices=shipped_prices).days
    assert [(day.day, day.requests) for day in counted] == [(DAY, 2)]


# --- the writer ------------------------------------------------------------------------


def test_writer_stores_status_and_latency(pool: Pool, migrated_schema: str) -> None:
    """Catches an INSERT that leaves the new columns out of its list (every row reads 200
    and null), or binds them the wrong way round: a row with 503 and 412 reads back 503 and
    412, and a row built with no arguments for them reads 200 and null."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    fields: dict[str, object] = {
        "ts": NOW,
        "tenant": "acme",
        "alias": "agent-default",
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "input_tokens": 0,
        "output_tokens": 0,
        "cached": False,
        "cost_usd": Decimal("0.000000"),
        "price_version": "anthropic/claude-haiku-5-5@2026-10-01",
        "estimated": False,
        "session": None,
        "trace_id": TRACE_ID,
    }
    failed = CostLedgerRow.model_validate({**fields, "status": 503, "latency_ms": 412})
    plain = CostLedgerRow.model_validate({**fields, "ts": NOW - timedelta(minutes=1)})
    writer.write(failed)
    writer.write(plain)
    writer.flush()

    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select status, latency_ms from {} order by ts desc").format(
                sql.Identifier(migrated_schema, "cost_ledger")
            )
        ).fetchall()
    assert [tuple(r) for r in rows] == [(503, 412), (200, None)]


# --- the filters -----------------------------------------------------------------------

# id, start of its day, minutes after it, tenant, model, status, cached. The id is stored as
# `input_tokens` and the latency is id * 100, so the 20 rows carry 100 to 2000 in steps of 100.
#
# Row 20 matches all six filters. Row 19 differs from it only in being not slow. Rows 18 to 14
# each differ from row 20 in exactly one thing (the day, the tenant, the model, the status, the
# cache flag). Rows 1 to 13 are fillers that match nothing, except 12 (status 500) and 13
# (status 499, one second before midnight of DAY), which sit on both sides of the errors line
# and the day line.
FILTER_ROWS: list[tuple[int, datetime, int, str, str, int, bool]] = [
    (20, DAY_START, 20, "acme", "m1", 502, True),
    (19, DAY_START, 19, "acme", "m1", 502, True),
    (18, PREVIOUS_START, 18, "acme", "m1", 502, True),
    (17, DAY_START, 17, "bolt", "m1", 502, True),
    (16, DAY_START, 16, "acme", "m2", 502, True),
    (15, DAY_START, 15, "acme", "m1", 200, True),
    (14, DAY_START, 0, "acme", "m1", 502, False),
    (13, DAY_START - timedelta(seconds=1), 0, "bolt", "m2", 499, False),
    (12, PREVIOUS_START, 12, "bolt", "m2", 500, False),
    *[(i, PREVIOUS_START, i, "bolt", "m2", 200, False) for i in range(1, 12)],
]


def write_filter_rows(pool: Pool, schema: str) -> None:
    for row_id, start, minutes, tenant, model, status, cached in FILTER_ROWS:
        insert_row(
            pool,
            schema,
            ts=start + timedelta(minutes=minutes),
            tenant=tenant,
            model=model,
            status=status,
            cached=cached,
            input_tokens=row_id,
            latency_ms=row_id * 100,
        )


FILTER_CASES: list[tuple[str, RequestFilters, list[int]]] = [
    ("none", RequestFilters(), list(range(1, 21))),
    ("day", RequestFilters(day=DAY), [14, 15, 16, 17, 19, 20]),
    ("tenant", RequestFilters(tenant="acme"), [14, 15, 16, 18, 19, 20]),
    ("model", RequestFilters(model="m1"), [14, 15, 17, 18, 19, 20]),
    ("errors", RequestFilters(errors=True), [12, 14, 16, 17, 18, 19, 20]),
    ("slow", RequestFilters(slow=True), [20]),
    ("cached", RequestFilters(cached=True), [15, 16, 17, 18, 19, 20]),
]


def test_each_filter_narrows_the_list_and_they_combine(pool: Pool, migrated_schema: str) -> None:
    """Catches a filter that is ignored, one wired to another's column, an errors line at 499
    or 501 instead of 500, a day taken in the server's
    zone, a filter ORed instead of ANDed, and a combination that drops one member: the
    six alone give their sets, all six together keep row 20 only, and taking away any one
    filter adds back exactly the one row that differs from row 20 in that one thing."""
    write_filter_rows(pool, migrated_schema)

    for case, filters, expected in FILTER_CASES:
        got = ids_of(pool, migrated_schema, filters)
        assert got == expected, case

    everything = RequestFilters(
        day=DAY, tenant="acme", model="m1", errors=True, slow=True, cached=True
    )
    assert ids_of(pool, migrated_schema, everything) == [20]
    without = [
        ("slow", everything.model_copy(update={"slow": False}), [19, 20]),
        ("day", everything.model_copy(update={"day": None, "slow": False}), [18, 19, 20]),
        ("tenant", everything.model_copy(update={"tenant": None, "slow": False}), [17, 19, 20]),
        ("model", everything.model_copy(update={"model": None, "slow": False}), [16, 19, 20]),
        ("errors", everything.model_copy(update={"errors": False, "slow": False}), [15, 19, 20]),
        ("cached", everything.model_copy(update={"cached": False, "slow": False}), [14, 19, 20]),
    ]
    for case, filters, expected in without:
        assert ids_of(pool, migrated_schema, filters) == expected, case


def test_pages_follow_the_filtered_count(pool: Pool, migrated_schema: str) -> None:
    """Catches a page count taken from the whole ledger while the list is filtered (3 pages
    of which 2 are empty), and a page past the last that is not clamped under a filter:
    45 rows, 21 of them for `acme`, give 3 pages unfiltered and 2 under the tenant filter."""
    for i in range(45):
        insert_row(
            pool,
            migrated_schema,
            ts=DAY_START + timedelta(minutes=i),
            tenant="acme" if i < 21 else "bolt",
            input_tokens=i,
        )
    reader = PostgresConsoleReader(pool, schema=migrated_schema)
    assert PAGE_SIZE == 20

    plain = reader.recent_requests(page=1, filters=RequestFilters(), days=WINDOW, now=NOW)
    acme = reader.recent_requests(
        page=1, filters=RequestFilters(tenant="acme"), days=WINDOW, now=NOW
    )
    last = reader.recent_requests(
        page=99, filters=RequestFilters(tenant="acme"), days=WINDOW, now=NOW
    )

    assert (plain.page, plain.pages, len(plain.rows)) == (1, 3, 20)
    assert (acme.page, acme.pages, len(acme.rows)) == (1, 2, 20)
    assert (last.page, last.pages, len(last.rows)) == (2, 2, 1)
    assert {row.tenant for row in last.rows} == {"acme"}


# --- the slow line ---------------------------------------------------------------------


def test_slow_means_above_the_window_p95(pool: Pool, migrated_schema: str) -> None:
    """Catches a slow line taken from the max, the mean or the 90th percentile, one that
    counts rows outside the window (a row 40 days back at 99999 ms would move it), one that
    counts null latencies as zero (a null row is added, which would lower the line), a
    comparison of `>=` (the row exactly at the line is not slow, and none is here), and a
    null latency treated as slow. Latencies 100 to 2000 in steps of 100 give 1905.0 and only
    the 2000 row is slow."""
    for step in range(1, 21):
        insert_row(
            pool,
            migrated_schema,
            ts=DAY_START + timedelta(minutes=step),
            input_tokens=step,
            latency_ms=step * 100,
        )
    insert_row(pool, migrated_schema, ts=DAY_START + timedelta(minutes=30), input_tokens=77)
    insert_row(
        pool,
        migrated_schema,
        ts=NOW - timedelta(days=40),
        input_tokens=88,
        latency_ms=99999,
    )
    reader = PostgresConsoleReader(pool, schema=migrated_schema)

    read = reader.recent_requests(page=1, filters=RequestFilters(slow=True), days=WINDOW, now=NOW)

    assert read.slow_ms == 1905.0
    assert [row.input_tokens for row in read.rows] == [20]
    assert [row.latency_ms for row in read.rows] == [2000]


def test_no_latencies_give_a_null_line_and_slow_keeps_nothing(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches a slow line of 0 for a window with no latencies (every row would be slow) and
    a slow filter that fails or keeps everything when the line is null: old rows have no
    latency, so the line is null and slow keeps nothing."""
    insert_row(pool, migrated_schema, ts=DAY_START, input_tokens=1)
    insert_row(pool, migrated_schema, ts=DAY_START + timedelta(minutes=1), input_tokens=2)
    reader = PostgresConsoleReader(pool, schema=migrated_schema)

    plain = reader.recent_requests(page=1, filters=RequestFilters(), days=WINDOW, now=NOW)
    slow = reader.recent_requests(page=1, filters=RequestFilters(slow=True), days=WINDOW, now=NOW)

    assert plain.slow_ms is None
    assert len(plain.rows) == 2
    assert slow.slow_ms is None
    assert slow.rows == ()
    assert (slow.page, slow.pages) == (1, 1)


# --- failed rows change no served number -----------------------------------------------


class Served(NamedTuple):
    """Every served-only number the failed-row case compares."""

    today: int
    hit_rate: float | None
    last_day: tuple[date, int]
    models: list[tuple[str, float, float]]
    per_alias: list[tuple[str, float]]
    days: list[tuple[int, int]]


def test_failed_rows_change_no_served_number(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches any served-only read that still counts a 502 row: requests today, the cache
    hit rate (the failed row is not cached, so a count of all rows moves it), requests per
    day, tokens per model (a failed row on a model nothing served would list it with zeros),
    cost per request (a count of all rows lowers the average) and the cost days' requests.
    Every number is read before the failed row is stored and again after, and must not
    change; the first read is checked to hold real numbers so the equality is not empty."""
    insert_row(
        pool,
        migrated_schema,
        ts=NOW - timedelta(hours=2),
        model="claude-haiku-5-5",
        input_tokens=100,
        output_tokens=50,
        cost_usd="0.002000",
    )
    insert_row(
        pool,
        migrated_schema,
        ts=NOW - timedelta(hours=1),
        model="claude-haiku-5-5",
        input_tokens=10,
        output_tokens=5,
        cached=True,
        cost_usd="0.000000",
    )
    reader = PostgresConsoleReader(pool, schema=migrated_schema)

    def snapshot() -> Served:
        cost = reader.cost(days=WINDOW, tenant=None, now=NOW, prices=shipped_prices)
        return Served(
            today=reader.home(now=NOW).requests_today,
            hit_rate=reader.cache_hit_rate(WINDOW, NOW),
            last_day=reader.daily_requests(WINDOW, NOW)[-1],
            models=[(t.model, t.input, t.output) for t in reader.model_tokens(WINDOW, NOW)],
            per_alias=[(c.alias, c.cost_usd) for c in cost.cost_per_request],
            days=[(day.requests, day.cached_requests) for day in cost.days],
        )

    before = snapshot()
    assert before == Served(
        today=2,
        hit_rate=0.5,
        last_day=(DAY, 2),
        models=[("claude-haiku-5-5", 110.0, 55.0)],
        per_alias=[("agent-default", 0.002)],
        days=[(2, 1)],
    )

    insert_row(
        pool,
        migrated_schema,
        ts=NOW - timedelta(minutes=30),
        model="gpt-5-mini",
        provider="openai",
        input_tokens=0,
        output_tokens=0,
        cost_usd="0.000000",
        status=502,
        latency_ms=300,
    )

    assert snapshot() == before


# --- the options -----------------------------------------------------------------------


def test_filter_options_are_the_window_tenants_and_models(pool: Pool, migrated_schema: str) -> None:
    """Catches options that are not sorted (rows are stored out of order), repeated (two rows
    share a tenant), taken from the filtered list instead of the window (the options must
    not shrink when a filter is set), or read past the window: a row 31 days back has a
    tenant and a model of its own and appears in neither list."""
    stored = [
        ("nordlicht", "gpt-5-mini", 3),
        ("acme", "claude-haiku-5-5", 2),
        ("nordlicht", "claude-haiku-5-5", 1),
    ]
    for tenant, model, minutes in stored:
        insert_row(
            pool,
            migrated_schema,
            ts=NOW - timedelta(minutes=minutes),
            tenant=tenant,
            model=model,
        )
    insert_row(
        pool,
        migrated_schema,
        ts=NOW - timedelta(days=31),
        tenant="ancient",
        model="old-model",
    )
    reader = PostgresConsoleReader(pool, schema=migrated_schema)

    plain = reader.recent_requests(page=1, filters=RequestFilters(), days=WINDOW, now=NOW)
    narrowed = reader.recent_requests(
        page=1, filters=RequestFilters(tenant="acme"), days=WINDOW, now=NOW
    )

    assert plain.tenants == ("acme", "nordlicht")
    assert plain.models == ("claude-haiku-5-5", "gpt-5-mini")
    assert narrowed.tenants == plain.tenants
    assert narrowed.models == plain.models
    assert len(narrowed.rows) == 1
