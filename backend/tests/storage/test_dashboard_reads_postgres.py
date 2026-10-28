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

* `PostgresConsoleReader(pool, *, schema)` with `home`, `cost` and `quality`,
  answering the frozen read models of `app.storage.dashboard_reads`;
* a window of `now - timedelta(days=days)`, compared with `>=` on timestamps and
  with `day >= since.date()` on the cost days, which are summed from the ledger itself;
"today" is `[UTC midnight, +1 day)`. A range is `start` and `end`, both
  UTC dates with both ends kept, and it replaces the `days` window for the day rows.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from app.limits.prices import PriceTable
from app.model_routing.provider_files import build_price_table, load_provider_files
from app.schemas.dashboard import (
    AliasCost,
    CategoryCount,
    CostDay,
    EvalRunSummary,
    LastEval,
    ModelTokens,
    ReleaseMarker,
    Saving,
    TagCount,
    UserCost,
)
from app.storage.dashboard_reads import (
    BlockedRelease,
    HomeRead,
    PostgresConsoleReader,
)

from .conftest import Pool, requires_postgres

pytestmark = requires_postgres

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_PROVIDERS = REPO_ROOT / "config" / "providers"

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
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached: bool = False,
    cost_usd: str = "0.001000",
    price_version: str = "anthropic/claude-haiku-4-5@2025-10-15",
    status: int = 200,
) -> None:
    """One `cost_ledger` row."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id, "
                "status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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
                price_version,
                False,
                None,
                "0af7651916cd43dd8448eb211c80319c",
                status,
            ),
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


@pytest.fixture(scope="module")
def shipped_prices() -> PriceTable:
    """The shipped price table, read once."""
    return build_price_table(load_provider_files(SHIPPED_PROVIDERS))


# --- home ------------------------------------------------------------------------


def test_today_counts_only_the_current_utc_day(pool: Pool, migrated_schema: str) -> None:
    """Catches "today" computed as a rolling 24 hours (the 23:59:59 rows yesterday fall
    inside it), an exclusive lower bound at UTC midnight, an unbounded upper edge (a row at
    tomorrow's midnight), the last eval chosen by
    `finished_at` instead of `started_at`, and a home read that still carries an open-incident
    count (the `incidents` table is gone, so the read would fail on the migrated schema)."""
    schema = migrated_schema
    insert_cost(pool, schema, ts=MIDNIGHT, cost_usd="0.500000")
    insert_cost(pool, schema, ts=NOW - timedelta(hours=1), cost_usd="0.250000")
    insert_cost(pool, schema, ts=MIDNIGHT - SECOND, cost_usd="4.000000")
    insert_cost(pool, schema, ts=MIDNIGHT + timedelta(days=1), cost_usd="8.000000")

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

    read = PostgresConsoleReader(pool, schema=schema).home(now=NOW)

    assert read == HomeRead(
        requests_today=2,
        cost_usd_today=0.75,
        last_eval=LastEval(
            id="run-newest-start",
            finished_at=newest_start + timedelta(minutes=5),
            passed=False,
            check_score=0.91,
        ),
        last_eval_metrics={"check_score": 0.91},
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
        last_eval=None,
        last_eval_metrics=None,
    )

    insert_run(
        pool, migrated_schema, run_id="run-no-accuracy", started_at=NOW, metrics={"other": 1}
    )
    last = reader.home(now=NOW).last_eval
    assert last is not None
    assert last.check_score is None


def test_hit_rate_counts_cached_rows_inside_the_window(pool: Pool, migrated_schema: str) -> None:
    """Catches a hit rate that counts cached rows outside the window (two more cached rows sit a
    second and three days before the start, which would make it 3 of 6), an exclusive lower
    edge (the cached row exactly at the start is dropped), a rate over the cached rows only
    or over the rows outside the window, and an empty window answered as 0 instead of None."""
    schema = migrated_schema
    reader = PostgresConsoleReader(pool, schema=schema)
    assert reader.cache_hit_rate(7, NOW) is None

    start = NOW - timedelta(days=7)
    insert_cost(pool, schema, ts=start, cached=True, cost_usd="0")
    insert_cost(pool, schema, ts=start + SECOND)
    insert_cost(pool, schema, ts=NOW - timedelta(hours=1), tenant="cuan")
    insert_cost(pool, schema, ts=NOW - timedelta(hours=2))
    insert_cost(pool, schema, ts=start - SECOND, cached=True, cost_usd="0")
    insert_cost(pool, schema, ts=start - timedelta(days=3), cached=True, cost_usd="0")

    assert reader.cache_hit_rate(7, NOW) == 0.25


# --- cost ------------------------------------------------------------------------


def test_days_sum_the_ledger_by_day_tenant_and_provider(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches days not summed across the alias and model columns (two rows for one day,
    tenant and provider), a window compared by instant instead of by UTC date (the row at
    01:00 on the window's first day would drop), a missing lower bound (the day before it),
    a wrong order, and a stale copy (the `evals` row is booked last, with no refresh, and
    must be there)."""
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
        cached=True,
        cost_usd="0.500000",
    )
    insert_cost(pool, schema, ts=today, provider="local", model="qwen", cost_usd="0.000000")
    insert_cost(pool, schema, ts=today, tenant="cuan", cost_usd="0.125000")
    insert_cost(pool, schema, ts=today, tenant="evals", cost_usd="1.000000")

    read = PostgresConsoleReader(pool, schema=schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

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
            tenant="evals",
            provider="anthropic",
            requests=1,
            cached_requests=0,
            cost_usd=1.0,
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


RANGE_NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def test_today_shows_without_any_refresh(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches a day read from a copy that only a job refreshes (a view or table): a row
    booked a minute ago is in `days` for today at once, with nothing run after the insert."""
    insert_cost(pool, migrated_schema, ts=NOW - timedelta(minutes=1), cost_usd="0.250000")

    read = PostgresConsoleReader(pool, schema=migrated_schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

    assert [(day.day, day.requests, day.cost_usd) for day in read.days] == [(NOW.date(), 1, 0.25)]


def test_days_keep_from_and_to_and_leave_out_failed_rows(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches a range that ignores `start`/`end` and uses `days` (days=1 would show only
    the fifth), an exclusive edge at either end (the rows at 00:00:00 on the first day and
    at 23:59:59 on the last are kept), a missing upper bound (the 4 October row), a 502 row
    counted as a request, and cached requests left out. Every distractor satisfies the
    other conditions, so each removed condition changes the answer."""
    schema = migrated_schema
    insert_cost(pool, schema, ts=datetime(2026, 10, 1, 23, 59, 59, tzinfo=UTC), cost_usd="9.0")
    insert_cost(pool, schema, ts=datetime(2026, 10, 2, 0, 0, 0, tzinfo=UTC), cost_usd="0.100000")
    insert_cost(
        pool,
        schema,
        ts=datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC),
        cost_usd="5.000000",
        status=502,
    )
    insert_cost(pool, schema, ts=datetime(2026, 10, 3, 9, 0, 0, tzinfo=UTC), cost_usd="0.200000")
    insert_cost(
        pool,
        schema,
        ts=datetime(2026, 10, 3, 23, 59, 59, tzinfo=UTC),
        cached=True,
        cost_usd="0",
    )
    insert_cost(pool, schema, ts=datetime(2026, 10, 4, 0, 0, 0, tzinfo=UTC), cost_usd="7.0")
    insert_cost(pool, schema, ts=RANGE_NOW - timedelta(hours=1), cost_usd="3.0")

    read = PostgresConsoleReader(pool, schema=schema).cost(
        days=1,
        tenant=None,
        now=RANGE_NOW,
        prices=shipped_prices,
        start=date(2026, 10, 2),
        end=date(2026, 10, 3),
    )

    assert read.days == (
        CostDay(
            day=date(2026, 10, 2),
            tenant="nordlicht",
            provider="anthropic",
            requests=1,
            cached_requests=0,
            cost_usd=0.1,
        ),
        CostDay(
            day=date(2026, 10, 3),
            tenant="nordlicht",
            provider="anthropic",
            requests=2,
            cached_requests=1,
            cost_usd=0.2,
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
    joined as `provider/model` - the provider kind, the alias - which prices
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

    read = PostgresConsoleReader(pool, schema=schema).cost(
        days=30, tenant=None, now=NOW, prices=shipped_prices
    )

    assert [row.alias for row in read.cost_per_request] == ["agent-default", "judge"]
    assert read.cost_per_request[0].cost_usd == pytest.approx(0.4)
    assert read.cost_per_request[1].cost_usd == pytest.approx(1.0)
    assert all(isinstance(row, AliasCost) for row in read.cost_per_request)


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


def test_a_release_marker_names_its_eval_run(pool: Pool, migrated_schema: str) -> None:
    """Catches a markers select that leaves `eval_run_id` out (every marker reads back with no
    run), one that reads it from the wrong column, and a missing link turned into an empty
    string instead of `None`."""
    schema = migrated_schema
    insert_run(pool, schema, run_id="linked", started_at=NOW - timedelta(days=3), metrics={})
    insert_release(
        pool, schema, git_sha="with-run", deployed_at=NOW - timedelta(days=1), eval_run_id="linked"
    )
    insert_release(pool, schema, git_sha="no-run", deployed_at=NOW - timedelta(days=2))

    read = PostgresConsoleReader(pool, schema=schema).quality(days=30, now=NOW)

    assert read.release_markers == (
        ReleaseMarker(
            git_sha="with-run",
            deployed_at=NOW - timedelta(days=1),
            status="deployed",
            eval_run_id="linked",
        ),
        ReleaseMarker(
            git_sha="no-run",
            deployed_at=NOW - timedelta(days=2),
            status="deployed",
            eval_run_id=None,
        ),
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


# --- the tenant filter -------------------------------------------------------------


def test_a_tenant_filter_narrows_cost(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches `tenant` applied to one cost query and not the others - the day rows, the
    savings and the per-request average each carry a `nordlicht` row that would leak into
    `cuan`'s page -"""
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


# --- the month spend ---------------------------------------------------------------


def test_month_spend_sums_this_utc_month_not_cached(pool: Pool, migrated_schema: str) -> None:
    """Catches a month taken as the last 30 days (the row at 31 August 23:59:59 falls inside
    it), an exclusive lower edge (the row at 00:00:00 on the 1st is dropped), cached rows
    summed in (the cached row carries a cost on purpose), a `tenant` filter that is ignored
    (the `cuan` rows leak into `nordlicht`) or applied to the wrong side, and a tenant with
    no rows this month being invented instead of left out."""
    schema = migrated_schema
    month_start = datetime(2026, 9, 1, tzinfo=UTC)
    insert_cost(pool, schema, ts=month_start, cost_usd="1.500000")
    insert_cost(pool, schema, ts=NOW - timedelta(hours=1), cost_usd="0.250000")
    insert_cost(pool, schema, ts=month_start - SECOND, cost_usd="40.000000")
    insert_cost(pool, schema, ts=NOW - timedelta(hours=2), cached=True, cost_usd="7.000000")
    insert_cost(pool, schema, ts=NOW - timedelta(hours=1), tenant="cuan", cost_usd="3.000000")
    insert_cost(pool, schema, ts=month_start - SECOND, tenant="evals", cost_usd="9.000000")
    reader = PostgresConsoleReader(pool, schema=schema)

    assert reader.month_spend(tenant=None, now=NOW) == pytest.approx(
        {"nordlicht": 1.75, "cuan": 3.0}
    )
    assert reader.month_spend(tenant="nordlicht", now=NOW) == pytest.approx({"nordlicht": 1.75})
    assert reader.month_spend(tenant="evals", now=NOW) == {}


# --- requests per day and tokens per model ------------------------------------------


def test_daily_requests_counts_each_utc_day_and_fills_zeros(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches days cut on local or rolling-24-hour edges (the rows at 23:59 and 00:00 UTC
    must land on their own days), a day with no rows dropped instead of 0, a window of 6 or
    8 days instead of 7 (the row 8 days back and the one a second before the first day's
    midnight are left out, the one at that midnight is in), cached rows or other tenants
    left uncounted, and a result not oldest first."""
    schema = migrated_schema
    first_day = datetime(2026, 9, 10, tzinfo=UTC)
    insert_cost(pool, schema, ts=NOW - timedelta(days=8))
    insert_cost(pool, schema, ts=first_day - SECOND)
    insert_cost(pool, schema, ts=first_day)
    insert_cost(pool, schema, ts=datetime(2026, 9, 14, 23, 59, tzinfo=UTC))
    insert_cost(pool, schema, ts=datetime(2026, 9, 15, 0, 0, tzinfo=UTC))
    insert_cost(pool, schema, ts=datetime(2026, 9, 16, 8, 0, tzinfo=UTC), cached=True, cost_usd="0")
    insert_cost(pool, schema, ts=datetime(2026, 9, 16, 9, 0, tzinfo=UTC), tenant="cuan")
    reader = PostgresConsoleReader(pool, schema=schema)

    assert reader.daily_requests(7, NOW) == (
        (date(2026, 9, 10), 1, 0),
        (date(2026, 9, 11), 0, 0),
        (date(2026, 9, 12), 0, 0),
        (date(2026, 9, 13), 0, 0),
        (date(2026, 9, 14), 1, 0),
        (date(2026, 9, 15), 1, 0),
        (date(2026, 9, 16), 2, 0),
    )


def test_daily_requests_splits_answered_and_failed_at_500(pool: Pool, migrated_schema: str) -> None:
    """Catches a split at the wrong status (499 counted as failed, or 500 as answered), a
    failed count that ignores the UTC day edges (a 500 at 23:59 and one at 00:00 land on their
    own days), failed rows outside the window counted (the one a second before the first
    day), and an empty day dropped instead of `(day, 0, 0)`."""
    schema = migrated_schema
    first_day = datetime(2026, 9, 10, tzinfo=UTC)
    busy = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
    for status in (200, 499, 500, 503):
        insert_cost(pool, schema, ts=busy, status=status)
    insert_cost(pool, schema, ts=datetime(2026, 9, 13, 23, 59, tzinfo=UTC), status=500)
    insert_cost(pool, schema, ts=datetime(2026, 9, 14, 0, 0, tzinfo=UTC), status=500)
    insert_cost(pool, schema, ts=first_day - SECOND, status=500)
    reader = PostgresConsoleReader(pool, schema=schema)

    assert reader.daily_requests(7, NOW) == (
        (date(2026, 9, 10), 0, 0),
        (date(2026, 9, 11), 0, 0),
        (date(2026, 9, 12), 2, 2),
        (date(2026, 9, 13), 0, 1),
        (date(2026, 9, 14), 0, 1),
        (date(2026, 9, 15), 0, 0),
        (date(2026, 9, 16), 0, 0),
    )


def test_model_tokens_sum_the_window_by_model(pool: Pool, migrated_schema: str) -> None:
    """Catches tokens summed over every row instead of the window (the Anthropic row a second
    before the start is left out), an exclusive lower edge (the OpenAI row exactly at the
    start is counted), rows of one model not summed together, a provider other than
    Anthropic dropped (the OpenAI model shows), input and output swapped, and an order other
    than by model name."""
    schema = migrated_schema
    start = NOW - timedelta(days=7)
    insert_cost(pool, schema, ts=NOW - timedelta(hours=1), input_tokens=100, output_tokens=50)
    insert_cost(
        pool, schema, ts=NOW - timedelta(hours=2), input_tokens=30, output_tokens=20, tenant="cuan"
    )
    insert_cost(
        pool,
        schema,
        ts=start,
        provider="openai",
        model="gpt-5-mini",
        input_tokens=1000,
        output_tokens=400,
        price_version="openai/gpt-5-mini@2025-08-07",
    )
    insert_cost(pool, schema, ts=start - SECOND, input_tokens=9000, output_tokens=8000)
    reader = PostgresConsoleReader(pool, schema=schema)

    assert reader.model_tokens(7, NOW) == (
        ModelTokens(model=SHIPPED_MODEL, input=130.0, output=70.0),
        ModelTokens(model="gpt-5-mini", input=1000.0, output=400.0),
    )


# --- top users by cost ----------------------------------------------------


def insert_user_cost(
    pool: Pool,
    schema: str,
    /,
    *,
    ts: datetime,
    tenant: str,
    user_id: str | None,
    cost_usd: str,
    status: int = 200,
) -> None:
    """One `cost_ledger` row that names a user (or none) and a status."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id, "
                "user_id, status) "
                "VALUES (%s, %s, 'agent-default', %s, %s, 10, 5, false, %s, %s, false, null, "
                "%s, %s, %s)"
            ).format(sql.Identifier(schema, "cost_ledger")),
            (
                ts,
                tenant,
                SHIPPED_PROVIDER,
                SHIPPED_MODEL,
                Decimal(cost_usd),
                "anthropic/claude-haiku-4-5@2025-10-15",
                "0af7651916cd43dd8448eb211c80319c",
                user_id,
                status,
            ),
        )


def user_of(number: int) -> str:
    """A well-formed user id that differs from every other in its last digits."""
    return f"u_{number:032x}"


@pytest.mark.db
def test_top_users_ranks_per_tenant_and_keeps_ten(
    pool: Pool, migrated_schema: str, shipped_prices: PriceTable
) -> None:
    """Catches a read with no per-tenant limit (11 users come back), a limit applied over
    all tenants together (`nordlicht` loses its two to `cuan`'s ten), a wrong order (the
    lowest first), and each filter: a no-user row (ranked as one user), a
    5xx row (money that was never spent), a row before the window, and, with `tenant=cuan`,
    a `nordlicht` row. Every distractor satisfies every OTHER condition, so removing that
    one filter is what changes the answer. User N spends N cents, so the order is set by
    the cost alone and the 11th user (the cheapest) is the one cut."""
    schema = migrated_schema
    recent = NOW - timedelta(hours=1)
    for number in range(1, 12):
        cents = f"0.{number:02d}0000"
        user = user_of(number)
        insert_user_cost(pool, schema, ts=recent, tenant="cuan", user_id=user, cost_usd=cents)
        insert_user_cost(pool, schema, ts=recent, tenant="cuan", user_id=user, cost_usd="0.000001")
    for number in (101, 102):
        insert_user_cost(
            pool, schema, ts=recent, tenant="nordlicht", user_id=user_of(number), cost_usd="0.001"
        )
    # Distractors: each would rank first for `cuan` if its filter were missing.
    insert_user_cost(pool, schema, ts=recent, tenant="cuan", user_id=None, cost_usd="9.000000")
    insert_user_cost(
        pool, schema, ts=recent, tenant="cuan", user_id=user_of(50), cost_usd="9.000000", status=503
    )
    insert_user_cost(
        pool, schema, ts=WINDOW_30 - SECOND, tenant="cuan", user_id=user_of(51), cost_usd="9.000000"
    )
    reader = PostgresConsoleReader(pool, schema=schema)

    everyone = reader.cost(days=30, tenant=None, now=NOW, prices=shipped_prices).top_users
    only_cuan = reader.cost(days=30, tenant="cuan", now=NOW, prices=shipped_prices).top_users

    cuan_rows = [row for row in everyone if row.tenant == "cuan"]
    nordlicht_rows = [row for row in everyone if row.tenant == "nordlicht"]
    assert len(cuan_rows) == 10
    assert len(nordlicht_rows) == 2
    assert [row.user_id for row in cuan_rows] == [user_of(number) for number in range(11, 1, -1)]
    assert isinstance(cuan_rows[0], UserCost)
    assert cuan_rows[0].requests == 2
    assert cuan_rows[0].cost_usd == pytest.approx(0.110001)
    assert user_of(1) not in {row.user_id for row in everyone}
    assert {row.user_id for row in nordlicht_rows} == {user_of(101), user_of(102)}
    assert only_cuan == tuple(cuan_rows)
