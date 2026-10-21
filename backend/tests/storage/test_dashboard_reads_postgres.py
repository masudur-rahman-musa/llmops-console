"""`PostgresConsoleReader` against a real server: every console read, from raw rows.

Every case runs the SHIPPED migrations into a throwaway schema, inserts rows with
raw SQL, and reads them back through the reader. Raw inserts rather than the
writers on purpose: the reads are written against the migrations' columns, and a
writer in between would let a column both sides misname stay invisible here.

The rows are chosen so each boundary has a row on BOTH sides of it - a row at
UTC midnight and one a second before, a row exactly at the window's start and one
a second before - and each filter has a distractor that satisfies every other
condition, so removing that one filter is what changes the answer.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`. One
session pool, one throwaway schema per test - the fixtures in
tests/storage/conftest.py, unchanged.

The contract these cases are written against, before it exists:

* `PostgresConsoleReader(pool, *, schema)` with `home`, `cost`, `quality`,
  `residency`, `incidents` and `releases`, answering the frozen read models of
  `app.storage.dashboard_reads`;
* a window of `now - timedelta(days=days)`, compared with `>=` on timestamps and
  with `day >= since.date()` on the rollup; "today" is `[UTC midnight, +1 day)`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from app.limits.prices import PriceTable, load_prices
from app.schemas.dashboard import (
    AliasCost,
    CategoryCount,
    CostDay,
    EvalRunSummary,
    Incident,
    LastEval,
    RegionCount,
    Release,
    ReleaseMarker,
    Saving,
    TagCount,
)
from app.storage.dashboard_reads import (
    ROW_LIMIT,
    BlockedRelease,
    HomeRead,
    PostgresConsoleReader,
)

from .conftest import Pool, requires_postgres

pytestmark = requires_postgres

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_PRICES = REPO_ROOT / "config" / "prices.yaml"

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
MIDNIGHT = datetime(2026, 9, 16, 0, 0, tzinfo=UTC)
WINDOW_30 = NOW - timedelta(days=30)
SECOND = timedelta(seconds=1)

SHIPPED_PROVIDER = "anthropic"
SHIPPED_MODEL = "claude-haiku-4-5"


# --- raw inserts -----------------------------------------------------------------


def insert_cost(
    pool: Pool,
    schema: str,
    /,
    *,
    ts: datetime,
    tenant: str = "nordlicht",
    alias: str = "agent-default",
    provider: str = SHIPPED_PROVIDER,
    model: str = SHIPPED_MODEL,
    region: str = "us",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached: bool = False,
    cost_usd: str = "0.001000",
    price_version: str = "anthropic/claude-haiku-4-5@2025-10-15",
) -> None:
    """One `cost_ledger` row."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, region, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "cost_ledger")),
            (
                ts,
                tenant,
                alias,
                provider,
                model,
                region,
                input_tokens,
                output_tokens,
                cached,
                Decimal(cost_usd),
                price_version,
                False,
                None,
                "0af7651916cd43dd8448eb211c80319c",
            ),
        )


def refresh_rollup(pool: Pool, schema: str, /) -> None:
    """Populate `cost_ledger_daily` from the rows inserted so far."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL("REFRESH MATERIALIZED VIEW {}").format(
                sql.Identifier(schema, "cost_ledger_daily")
            )
        )


def insert_residency(
    pool: Pool,
    schema: str,
    /,
    *,
    ts: datetime,
    tenant: str,
    region: str,
    provider: str,
    allowed: bool,
) -> None:
    """One `residency_log` row."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, data_class, alias, provider, region, allowed) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "residency_log")),
            (ts, tenant, "internal", "agent-default", provider, region, allowed),
        )


def insert_run(
    pool: Pool,
    schema: str,
    /,
    *,
    run_id: str,
    started_at: datetime,
    metrics: dict[str, Any],
    passed: bool = True,
    set_name: str = "helpdesk",
    run_kind: str = "ci",
    git_sha: str | None = None,
    finished_at: datetime | None = None,
) -> None:
    """One `eval_runs` row; `finished_at` defaults to twenty minutes after the start."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (id, started_at, finished_at, set_name, set_sha, set_sha_source, "
                "run_kind, target, tenant, git_sha, metrics, passed, report_path) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "eval_runs")),
            (
                run_id,
                started_at,
                finished_at if finished_at is not None else started_at + timedelta(minutes=20),
                set_name,
                "0" * 64,
                "manifest",
                run_kind,
                "http://gateway.test",
                "evals",
                git_sha,
                Jsonb(metrics),
                passed,
                f"reports/{run_id}.json",
            ),
        )


def insert_case(
    pool: Pool,
    schema: str,
    /,
    *,
    run_id: str,
    case_id: str,
    category: str,
    passed: bool,
    tags: list[str],
) -> None:
    """One `eval_cases` row."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (run_id, case_id, category, passed, tags, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "eval_cases")),
            (run_id, case_id, category, passed, tags, Jsonb({})),
        )


def insert_incident(
    pool: Pool,
    schema: str,
    /,
    *,
    incident_id: str,
    started_at: datetime,
    resolved_at: datetime | None,
    severity: str = "sev2",
    title: str = "gateway 5xx on the eu pool",
    summary: str = "the eu pool answered 503 for eleven minutes",
    postmortem_url: str | None = None,
) -> None:
    """One `incidents` row."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (id, started_at, resolved_at, severity, title, summary, "
                "postmortem_url) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "incidents")),
            (incident_id, started_at, resolved_at, severity, title, summary, postmortem_url),
        )


def insert_release(
    pool: Pool,
    schema: str,
    /,
    *,
    git_sha: str,
    deployed_at: datetime,
    status: str = "deployed",
    eval_run_id: str | None = None,
    pr_url: str | None = None,
) -> None:
    """One `releases` row."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (git_sha, deployed_at, eval_run_id, pr_url, status) "
                "VALUES (%s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "releases")),
            (git_sha, deployed_at, eval_run_id, pr_url, status),
        )


def hex_id(number: int) -> str:
    """A distinct 32-hex id per number."""
    return f"{number:032x}"


@pytest.fixture(scope="module")
def shipped_prices() -> PriceTable:
    """The shipped price table, read once."""
    return load_prices(SHIPPED_PRICES)


# --- home ------------------------------------------------------------------------


def test_today_counts_only_the_current_utc_day(pool: Pool, migrated_schema: str) -> None:
    """Catches "today" computed as a rolling 24 hours (the 23:59:59 rows yesterday fall
    inside it), an exclusive lower bound at UTC midnight, an unbounded upper edge (a row at
    tomorrow's midnight), violations counted without `NOT allowed`, the last eval chosen by
    `finished_at` instead of `started_at`, and open incidents counted without
    `resolved_at IS NULL`."""
    schema = migrated_schema
    insert_cost(pool, schema, ts=MIDNIGHT, cost_usd="0.500000")
    insert_cost(pool, schema, ts=NOW - timedelta(hours=1), cost_usd="0.250000")
    insert_cost(pool, schema, ts=MIDNIGHT - SECOND, cost_usd="4.000000")
    insert_cost(pool, schema, ts=MIDNIGHT + timedelta(days=1), cost_usd="8.000000")

    insert_residency(
        pool, schema, ts=MIDNIGHT, tenant="cuan", region="us", provider="anthropic", allowed=False
    )
    insert_residency(
        pool,
        schema,
        ts=MIDNIGHT - SECOND,
        tenant="cuan",
        region="us",
        provider="anthropic",
        allowed=False,
    )
    insert_residency(
        pool, schema, ts=NOW, tenant="cuan", region="eu", provider="anthropic_eu", allowed=True
    )

    newest_start = NOW - timedelta(hours=2)
    insert_run(
        pool,
        schema,
        run_id="run-newest-start",
        started_at=newest_start,
        finished_at=newest_start + timedelta(minutes=5),
        metrics={"check_score": 0.91},
        passed=False,
    )
    insert_run(
        pool,
        schema,
        run_id="run-newest-finish",
        started_at=NOW - timedelta(hours=5),
        finished_at=NOW - timedelta(minutes=30),
        metrics={"check_score": 0.5},
    )

    insert_incident(pool, schema, incident_id=hex_id(1), started_at=NOW, resolved_at=None)
    insert_incident(
        pool, schema, incident_id=hex_id(2), started_at=NOW - timedelta(days=40), resolved_at=None
    )
    insert_incident(
        pool,
        schema,
        incident_id=hex_id(3),
        started_at=NOW - timedelta(hours=3),
        resolved_at=NOW - timedelta(hours=2),
    )

    read = PostgresConsoleReader(pool, schema=schema).home(now=NOW)

    assert read == HomeRead(
        requests_today=2,
        cost_usd_today=0.75,
        residency_violations_today=1,
        last_eval=LastEval(
            id="run-newest-start",
            finished_at=newest_start + timedelta(minutes=5),
            passed=False,
            check_score=0.91,
        ),
        open_incidents=2,
    )


def test_an_empty_ledger_answers_zeros_and_no_last_eval(pool: Pool, migrated_schema: str) -> None:
    """Catches `sum(cost_usd)` over no rows reaching the model as NULL (a validation error,
    so a 500 on a fresh install) instead of 0, and a missing eval run raising instead of
    answering `last_eval: None`; the one run stored here has no `check_score`, which
    must read as `None` rather than raise."""
    reader = PostgresConsoleReader(pool, schema=migrated_schema)
    assert reader.configured is True

    assert reader.home(now=NOW) == HomeRead(
        requests_today=0,
        cost_usd_today=0.0,
        residency_violations_today=0,
        last_eval=None,
        open_incidents=0,
    )

    insert_run(
        pool, migrated_schema, run_id="run-no-accuracy", started_at=NOW, metrics={"other": 1}
    )
    last = reader.home(now=NOW).last_eval
    assert last is not None
    assert last.check_score is None


# --- cost ------------------------------------------------------------------------


def test_an_unpopulated_rollup_is_empty_and_flagged(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches the rollup read without the `pg_matviews.ispopulated` check: PostgreSQL
    refuses a SELECT from a never-refreshed materialised view, so the cost page would be a
    500 on every fresh install until someone runs `make refresh-rollup`."""
    insert_cost(pool, migrated_schema, ts=NOW - timedelta(hours=1))

    read = PostgresConsoleReader(pool, schema=migrated_schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

    assert read.rollup_populated is False
    assert read.days == ()


def test_days_sum_the_rollup_by_day_tenant_and_provider(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches days not summed across the rollup's alias, model and region columns (two
    rows for one day, tenant and provider), a window compared by instant instead of by
    UTC date (the row at 01:00 on the window's first day would drop), a missing lower
    bound (the day before it), a wrong order, and a read of the live ledger instead of the
    rollup (a row inserted after the refresh must not appear)."""
    schema = migrated_schema
    first_day = WINDOW_30.date()
    day_one = datetime(first_day.year, first_day.month, first_day.day, 1, 0, tzinfo=UTC)
    insert_cost(pool, schema, ts=day_one, cost_usd="0.100000")
    insert_cost(pool, schema, ts=day_one - timedelta(hours=2), cost_usd="9.000000")
    today = NOW - timedelta(hours=1)
    insert_cost(pool, schema, ts=today, alias="agent-default", cost_usd="0.250000")
    insert_cost(
        pool,
        schema,
        ts=today,
        alias="judge",
        model="claude-opus-5",
        region="us",
        cached=True,
        cost_usd="0.500000",
    )
    insert_cost(
        pool, schema, ts=today, provider="local", model="qwen", region="self", cost_usd="0.000000"
    )
    insert_cost(pool, schema, ts=today, tenant="cuan", cost_usd="0.125000")
    refresh_rollup(pool, schema)
    insert_cost(pool, schema, ts=today, tenant="evals", cost_usd="1.000000")

    read = PostgresConsoleReader(pool, schema=schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

    assert read.rollup_populated is True
    assert read.days == (
        CostDay(
            day=first_day,
            tenant="nordlicht",
            provider="anthropic",
            requests=1,
            cached_requests=0,
            cost_usd=0.1,
        ),
        CostDay(
            day=NOW.date(),
            tenant="cuan",
            provider="anthropic",
            requests=1,
            cached_requests=0,
            cost_usd=0.125,
        ),
        CostDay(
            day=NOW.date(),
            tenant="nordlicht",
            provider="anthropic",
            requests=2,
            cached_requests=1,
            cost_usd=0.75,
        ),
        CostDay(
            day=NOW.date(),
            tenant="nordlicht",
            provider="local",
            requests=1,
            cached_requests=0,
            cost_usd=0.0,
        ),
    )


TWO_PRICE_KEY = "testprov/model-a"

TWO_PRICES = PriceTable.model_validate(
    {
        "prices": {
            TWO_PRICE_KEY: [
                {
                    "effective_from": "2026-08-01",
                    "input_usd_per_1m": "1.00",
                    "output_usd_per_1m": "2.00",
                },
                {
                    "effective_from": "2026-09-10",
                    "input_usd_per_1m": "10.00",
                    "output_usd_per_1m": "20.00",
                },
            ]
        }
    }
)
"""A key whose price changes inside the window, tenfold, so the instant a group is priced
at decides the answer."""


def insert_cached_test_row(
    pool: Pool,
    schema: str,
    ts: datetime,
    input_tokens: int,
    output_tokens: int,
    /,
    *,
    tenant: str = "nordlicht",
) -> None:
    """One cached `cost_ledger` row for `TWO_PRICE_KEY`, booked at zero as a hit is."""
    version = "2026-09-10" if ts.date() >= date(2026, 9, 10) else "2026-08-01"
    insert_cost(
        pool,
        schema,
        ts=ts,
        tenant=tenant,
        provider="testprov",
        model="model-a",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached=True,
        cost_usd="0",
        price_version=f"{TWO_PRICE_KEY}@{version}",
    )


def test_savings_price_cached_rows_at_the_price_in_force(pool: Pool, migrated_schema: str) -> None:
    """Catches savings priced at `now` (the 5 September group would cost ten times as much),
    read from the cached rows' own `cost_usd` (zero), summed over non-cached rows too, left
    unwindowed (a priced row a second before the window), not grouped per (day, tenant), or
    totalled wrongly."""
    schema = migrated_schema
    early = datetime(2026, 9, 5, 10, 0, tzinfo=UTC)
    late = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
    insert_cached_test_row(pool, schema, early, 1_000_000, 0)
    insert_cached_test_row(pool, schema, early + timedelta(hours=1), 0, 500_000)
    insert_cached_test_row(pool, schema, late, 1_000_000, 0)
    insert_cached_test_row(pool, schema, late, 0, 1_000_000, tenant="cuan")
    insert_cached_test_row(pool, schema, WINDOW_30 - SECOND, 1_000_000, 0)
    insert_cost(
        pool,
        schema,
        ts=late,
        provider="testprov",
        model="model-a",
        input_tokens=9_000_000,
        output_tokens=9_000_000,
        cached=False,
        cost_usd="270.000000",
        price_version=f"{TWO_PRICE_KEY}@2026-09-10",
    )
    refresh_rollup(pool, schema)

    read = PostgresConsoleReader(pool, schema=schema).cost(
        days=30, tenant=None, now=NOW, prices=TWO_PRICES
    )

    assert read.savings == (
        Saving(day=date(2026, 9, 5), tenant="nordlicht", saved_usd=2.0),
        Saving(day=date(2026, 9, 12), tenant="cuan", saved_usd=20.0),
        Saving(day=date(2026, 9, 12), tenant="nordlicht", saved_usd=10.0),
    )
    assert read.savings_usd_total == pytest.approx(32.0)


def test_a_cached_row_of_a_shipped_priced_provider_saves_more_than_zero(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches the savings key built from anything but the ledger's `provider` and `model`
    joined as `provider/model` - the provider kind, the alias, the region - which prices
    every group as `unpriced` zero, so the "money saved" figure reads 0 in production while
    a test table with a made-up key stays green."""
    key = f"{SHIPPED_PROVIDER}/{SHIPPED_MODEL}"
    assert key in shipped_prices.prices

    insert_cost(
        pool,
        migrated_schema,
        ts=NOW - timedelta(hours=1),
        cached=True,
        cost_usd="0",
        input_tokens=10_000,
        output_tokens=10_000,
        price_version=f"{key}@2025-10-15",
    )
    refresh_rollup(pool, migrated_schema)

    read = PostgresConsoleReader(pool, schema=migrated_schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

    assert read.savings_usd_total > 0
    assert [saving.tenant for saving in read.savings] == ["nordlicht"]


def test_cost_per_request_excludes_cached_rows(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches cached rows counted in the divisor (a free cache hit would halve the
    apparent cost of a request), an alias with only cached rows answered as a zero-cost
    alias, rows outside the window averaged in, and an order other than by alias."""
    schema = migrated_schema
    recent = NOW - timedelta(hours=1)
    insert_cost(pool, schema, ts=recent, alias="judge", cost_usd="1.000000")
    insert_cost(pool, schema, ts=recent, alias="agent-default", cost_usd="0.300000")
    insert_cost(pool, schema, ts=recent, alias="agent-default", cost_usd="0.500000")
    insert_cost(pool, schema, ts=recent, alias="agent-default", cached=True, cost_usd="0")
    insert_cost(pool, schema, ts=recent, alias="agent-default", cached=True, cost_usd="0")
    insert_cost(pool, schema, ts=recent, alias="embed", cached=True, cost_usd="0")
    insert_cost(pool, schema, ts=WINDOW_30 - SECOND, alias="agent-default", cost_usd="10.000000")
    refresh_rollup(pool, schema)

    read = PostgresConsoleReader(pool, schema=schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

    assert [row.alias for row in read.cost_per_request] == ["agent-default", "judge"]
    assert read.cost_per_request[0].cost_usd == pytest.approx(0.4)
    assert read.cost_per_request[1].cost_usd == pytest.approx(1.0)
    assert all(isinstance(row, AliasCost) for row in read.cost_per_request)


# --- residency -------------------------------------------------------------------


def test_residency_by_region_and_violations_over_the_window(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches the window compared with `>` (the row exactly at its start drops), no lower
    bound (the violation a second before it counts), a grouping missing `provider` (the two
    eu providers merge), violations counted over `allowed` rows, and a wrong order."""
    schema = migrated_schema
    recent = NOW - timedelta(days=1)
    rows = [
        (recent, "nordlicht", "eu", "anthropic_eu", True),
        (WINDOW_30, "nordlicht", "eu", "anthropic_eu", True),
        (recent, "nordlicht", "eu", "mistral_eu", True),
        (recent, "nordlicht", "self", "local", True),
        (recent, "cuan", "us", "anthropic", False),
        (WINDOW_30 - SECOND, "cuan", "us", "anthropic", False),
        (WINDOW_30 - SECOND, "evals", "us", "openai", True),
    ]
    for ts, tenant, region, provider, allowed in rows:
        insert_residency(
            pool, schema, ts=ts, tenant=tenant, region=region, provider=provider, allowed=allowed
        )

    read = PostgresConsoleReader(pool, schema=schema).residency(days=30, tenant=None, now=NOW)

    assert read.by_region == (
        RegionCount(tenant="cuan", region="us", provider="anthropic", requests=1),
        RegionCount(tenant="nordlicht", region="eu", provider="anthropic_eu", requests=2),
        RegionCount(tenant="nordlicht", region="eu", provider="mistral_eu", requests=1),
        RegionCount(tenant="nordlicht", region="self", provider="local", requests=1),
    )
    assert read.violations == 1


# --- quality ---------------------------------------------------------------------


def test_quality_runs_categories_taxonomy_and_markers(pool: Pool, migrated_schema: str) -> None:
    """Catches runs not windowed (the 40-day-old run), windowed with `>` (the run exactly at
    the window's start), or not newest first; categories counted over every run's cases
    instead of the newest run's (`zzz` belongs to an older run); a taxonomy not windowed
    (`stale`) or ordered by count alone; and release markers not windowed or not newest
    first."""
    schema = migrated_schema
    newest = NOW - timedelta(days=2)
    older = NOW - timedelta(days=10)
    outside = NOW - timedelta(days=40)
    insert_run(pool, schema, run_id="run-older", started_at=older, metrics={"a": 1}, git_sha="s2")
    insert_run(pool, schema, run_id="run-newest", started_at=newest, metrics={"a": 2}, passed=False)
    insert_run(
        pool, schema, run_id="run-at-start", started_at=WINDOW_30, metrics={}, run_kind="nightly"
    )
    insert_run(pool, schema, run_id="run-outside", started_at=outside, metrics={})

    cases: list[tuple[str, str, str, bool, list[str]]] = [
        ("run-newest", "c1", "routing", True, ["wrong_tool", "hallucination"]),
        ("run-newest", "c2", "routing", True, ["wrong_tool"]),
        ("run-newest", "c3", "refusal", False, []),
        ("run-newest", "c4", "refusal", True, []),
        ("run-older", "c1", "zzz", True, ["hallucination", "refusal_miss"]),
        ("run-outside", "c1", "yyy", True, ["stale", "stale", "stale"]),
    ]
    for run_id, case_id, category, passed, tags in cases:
        insert_case(
            pool,
            schema,
            run_id=run_id,
            case_id=case_id,
            category=category,
            passed=passed,
            tags=tags,
        )

    insert_release(pool, schema, git_sha="s1", deployed_at=NOW - timedelta(days=1))
    insert_release(
        pool, schema, git_sha="s2", deployed_at=NOW - timedelta(days=5), status="blocked"
    )
    insert_release(pool, schema, git_sha="s0", deployed_at=NOW - timedelta(days=45))

    read = PostgresConsoleReader(pool, schema=schema).quality(days=30, now=NOW)

    assert read.runs == (
        EvalRunSummary(
            id="run-newest",
            started_at=newest,
            run_kind="ci",
            git_sha=None,
            passed=False,
            metrics={"a": 2},
        ),
        EvalRunSummary(
            id="run-older",
            started_at=older,
            run_kind="ci",
            git_sha="s2",
            passed=True,
            metrics={"a": 1},
        ),
        EvalRunSummary(
            id="run-at-start",
            started_at=WINDOW_30,
            run_kind="nightly",
            git_sha=None,
            passed=True,
            metrics={},
        ),
    )
    assert read.categories == (
        CategoryCount(category="refusal", passed=1, total=2),
        CategoryCount(category="routing", passed=2, total=2),
    )
    assert read.taxonomy == (
        TagCount(tag="hallucination", count=2),
        TagCount(tag="wrong_tool", count=2),
        TagCount(tag="refusal_miss", count=1),
    )
    assert read.release_markers == (
        ReleaseMarker(git_sha="s1", deployed_at=NOW - timedelta(days=1), status="deployed"),
        ReleaseMarker(git_sha="s2", deployed_at=NOW - timedelta(days=5), status="blocked"),
    )


def insert_accuracy_run(
    pool: Pool,
    schema: str,
    run_id: str,
    days_ago: int,
    accuracy: float,
    /,
    *,
    passed: bool = True,
    set_name: str = "helpdesk",
) -> None:
    """One eval run carrying only `check_score`, started `days_ago` before `NOW`."""
    insert_run(
        pool,
        schema,
        run_id=run_id,
        started_at=NOW - timedelta(days=days_ago),
        metrics={"check_score": accuracy},
        passed=passed,
        set_name=set_name,
    )


def test_blocked_releases_carry_their_regressions(pool: Pool, migrated_schema: str) -> None:
    """Catches the comparison run chosen as the newest passed run overall (a later one
    exists), without `passed` (a newer failed run), without `set_name` (a newer passed run of
    another set), or as the oldest; catches non-blocked releases listed, an unlinked release
    raising instead of answering `None` twice, and a first-ever run given a previous run."""
    schema = migrated_schema
    insert_accuracy_run(pool, schema, "p-old", 20, 0.80)
    insert_accuracy_run(pool, schema, "p-prev", 10, 0.90)
    insert_accuracy_run(pool, schema, "failed-between", 6, 0.99, passed=False)
    insert_accuracy_run(pool, schema, "other-set", 4, 0.97, set_name="other")
    insert_accuracy_run(pool, schema, "linked", 3, 0.85, passed=False)
    insert_accuracy_run(pool, schema, "later-pass", 1, 0.95)

    linked_at = NOW - timedelta(days=2)
    unlinked_at = NOW - timedelta(days=4)
    first_at = NOW - timedelta(days=8)
    pr_url = "https://github.com/example/repo/pull/7"
    insert_release(
        pool,
        schema,
        git_sha="bad-linked",
        deployed_at=linked_at,
        status="blocked",
        eval_run_id="linked",
        pr_url=pr_url,
    )
    insert_release(pool, schema, git_sha="bad-unlinked", deployed_at=unlinked_at, status="blocked")
    insert_release(
        pool,
        schema,
        git_sha="bad-first",
        deployed_at=first_at,
        status="blocked",
        eval_run_id="p-old",
    )
    insert_release(
        pool, schema, git_sha="good", deployed_at=NOW - timedelta(hours=1), eval_run_id="later-pass"
    )

    read = PostgresConsoleReader(pool, schema=schema).quality(days=30, now=NOW)

    assert read.blocked == (
        BlockedRelease(
            git_sha="bad-linked",
            deployed_at=linked_at,
            pr_url=pr_url,
            eval_run_id="linked",
            metrics={"check_score": 0.85},
            previous_metrics={"check_score": 0.90},
        ),
        BlockedRelease(
            git_sha="bad-unlinked",
            deployed_at=unlinked_at,
            pr_url=None,
            eval_run_id=None,
            metrics=None,
            previous_metrics=None,
        ),
        BlockedRelease(
            git_sha="bad-first",
            deployed_at=first_at,
            pr_url=None,
            eval_run_id="p-old",
            metrics={"check_score": 0.80},
            previous_metrics=None,
        ),
    )


# --- incidents and releases --------------------------------------------------------


def test_incidents_are_newest_first_and_capped_at_100(pool: Pool, migrated_schema: str) -> None:
    """Catches a timeline read in insertion order (rows go in oldest first), without the
    `LIMIT` (101 rows are stored), with `open` not derived from `resolved_at IS NULL`, or
    with a column dropped on the way out."""
    assert ROW_LIMIT == 100
    schema = migrated_schema
    for minutes in range(100, -1, -1):
        insert_incident(
            pool,
            schema,
            incident_id=hex_id(minutes),
            started_at=NOW - timedelta(minutes=minutes),
            resolved_at=NOW if minutes % 2 else None,
            postmortem_url=f"https://example.test/pm/{minutes}" if minutes == 1 else None,
        )

    read = PostgresConsoleReader(pool, schema=schema).incidents()

    assert len(read) == 100
    assert [row.id for row in read] == [hex_id(minutes) for minutes in range(100)]
    assert read[0] == Incident(
        id=hex_id(0),
        started_at=NOW,
        resolved_at=None,
        severity="sev2",
        title="gateway 5xx on the eu pool",
        summary="the eu pool answered 503 for eleven minutes",
        postmortem_url=None,
        open=True,
    )
    assert read[1] == Incident(
        id=hex_id(1),
        started_at=NOW - timedelta(minutes=1),
        resolved_at=NOW,
        severity="sev2",
        title="gateway 5xx on the eu pool",
        summary="the eu pool answered 503 for eleven minutes",
        postmortem_url="https://example.test/pm/1",
        open=False,
    )


def test_releases_left_join_eval_runs_and_cap_at_100(pool: Pool, migrated_schema: str) -> None:
    """Catches an INNER join (the unlinked release vanishes), eval fields filled for an
    unlinked release, `check_score` raising when a linked run lacks it, a read in
    insertion order, and a missing `LIMIT` (103 releases are stored)."""
    schema = migrated_schema
    for hours in range(100, 0, -1):
        insert_release(
            pool,
            schema,
            git_sha=f"old{hours:04d}",
            deployed_at=NOW - timedelta(days=1, hours=hours),
        )
    ran_at = NOW - timedelta(hours=3)
    insert_run(pool, schema, run_id="run-scored", started_at=ran_at, metrics={"check_score": 0.88})
    insert_run(pool, schema, run_id="run-unscored", started_at=ran_at, metrics={}, passed=False)
    pr_url = "https://github.com/example/repo/pull/9"
    insert_release(
        pool,
        schema,
        git_sha="scored",
        deployed_at=NOW - timedelta(minutes=1),
        eval_run_id="run-scored",
        pr_url=pr_url,
    )
    insert_release(
        pool,
        schema,
        git_sha="unlinked",
        deployed_at=NOW - timedelta(minutes=2),
        status="rolled_back",
    )
    insert_release(
        pool,
        schema,
        git_sha="unscored",
        deployed_at=NOW - timedelta(minutes=3),
        eval_run_id="run-unscored",
        status="blocked",
    )

    read = PostgresConsoleReader(pool, schema=schema).releases()

    assert len(read) == 100
    assert read[:3] == (
        Release(
            git_sha="scored",
            deployed_at=NOW - timedelta(minutes=1),
            status="deployed",
            pr_url=pr_url,
            eval_run_id="run-scored",
            eval_passed=True,
            check_score=0.88,
        ),
        Release(
            git_sha="unlinked",
            deployed_at=NOW - timedelta(minutes=2),
            status="rolled_back",
            pr_url=None,
            eval_run_id=None,
            eval_passed=None,
            check_score=None,
        ),
        Release(
            git_sha="unscored",
            deployed_at=NOW - timedelta(minutes=3),
            status="blocked",
            pr_url=None,
            eval_run_id="run-unscored",
            eval_passed=False,
            check_score=None,
        ),
    )
    assert [row.git_sha for row in read[3:]] == [f"old{hours:04d}" for hours in range(1, 98)]


# --- the tenant filter -------------------------------------------------------------


def test_a_tenant_filter_narrows_cost_and_residency(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches `tenant` applied to one cost query and not the others - the rollup days, the
    savings and the per-request average each carry a `nordlicht` row that would leak into
    `cuan`'s page - and applied to residency's grouping but not its violation count."""
    schema = migrated_schema
    recent = NOW - timedelta(hours=1)
    for tenant, row_cost in (("nordlicht", "0.900000"), ("cuan", "0.100000")):
        insert_cost(pool, schema, ts=recent, tenant=tenant, cost_usd=row_cost)
        insert_cost(
            pool,
            schema,
            ts=recent,
            tenant=tenant,
            cached=True,
            cost_usd="0",
            input_tokens=10_000,
            output_tokens=10_000,
        )
        insert_residency(
            pool,
            schema,
            ts=recent,
            tenant=tenant,
            region="eu",
            provider="anthropic_eu",
            allowed=True,
        )
    insert_residency(
        pool,
        schema,
        ts=recent,
        tenant="nordlicht",
        region="us",
        provider="anthropic",
        allowed=False,
    )
    refresh_rollup(pool, schema)
    reader = PostgresConsoleReader(pool, schema=schema)

    everyone = reader.cost(days=30, tenant=None, now=NOW, prices=shipped_prices)
    cost = reader.cost(days=30, tenant="cuan", now=NOW, prices=shipped_prices)

    assert {row.tenant for row in everyone.days} == {"nordlicht", "cuan"}
    assert [row.tenant for row in cost.days] == ["cuan"]
    assert [row.tenant for row in cost.savings] == ["cuan"]
    assert cost.savings_usd_total > 0
    assert cost.savings_usd_total == pytest.approx(everyone.savings_usd_total / 2)
    assert [row.alias for row in cost.cost_per_request] == ["agent-default"]
    assert cost.cost_per_request[0].cost_usd == pytest.approx(0.1)

    residency = reader.residency(days=30, tenant="cuan", now=NOW)
    assert residency.by_region == (
        RegionCount(tenant="cuan", region="eu", provider="anthropic_eu", requests=1),
    )
    assert residency.violations == 0
    assert reader.residency(days=30, tenant=None, now=NOW).violations == 1
