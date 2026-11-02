"""Every Postgres read the console makes, one method per page.

The reader shape follows the writers in `storage/`: a Protocol, a Postgres
implementation over the app's one pool, and a stand-in for a gateway with no
`database_url`. The stand-in is `UnconfiguredConsoleReader`, and its
`configured = False` is what the routes read to answer `postgres: unconfigured`
without calling it at all.

**Windows.** A page's window starts at `now - timedelta(days=days)`, compared with
`>=` on a timestamp. "Today" on the home page is `[UTC midnight, +1 day)`. Every
query is bounded by its window, by `ROW_LIMIT`, or by `PAGE_SIZE` for the paged
request list.

**Cost days come straight from the ledger**: per UTC day, tenant and
provider, the served rows (`status < 500`), cached ones counted, from `start` to
`end` with both days kept. With no range, the last `days` days through today. So
today shows the moment a request is booked. A test pins this.

**Savings are estimated at read time**. A cache hit is booked at zero
with the replayed answer's own token counts, so what it saved is what those tokens
would have cost. Cached rows are grouped by `(day, tenant, provider, model,
price_version)` and each group is priced by `limits.cost.cost_for` at key
`provider/model` and at the group's earliest `ts`, the price in force when it was
served.

**Regressions** are computed from the metrics the reader returns, by
`regressed_metrics`, over `HIGHER_IS_BETTER_METRICS` only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Final, Protocol

from psycopg import Connection, sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from app.limits.cost import cost_for
from app.limits.prices import PriceTable
from app.schemas.dashboard import (
    AliasCost,
    CategoryCount,
    CostDay,
    EvalRunSummary,
    LastEval,
    ModelTokens,
    NeedsALookPlay,
    Regression,
    ReleaseMarker,
    RequestRow,
    Saving,
    TagCount,
    UserCost,
)
from app.schemas.provider import Usage
from app.storage.db import DEFAULT_SCHEMA, checked_schema

ROW_LIMIT: Final[int] = 100
"""The most rows any unpaged list on a page carries."""

PAGE_SIZE: Final[int] = 20
"""The rows one page of the request list carries."""

TOP_USERS_PER_TENANT: Final[int] = 10
"""How many users the Cost tab ranks for each tenant."""

HIGHER_IS_BETTER_METRICS: Final[tuple[str, ...]] = (
    "safety",
    "outcome_every_play",
    "outcome_at_least_once",
    "reply_quality",
)
"""The eval scores where lower is worse.

The rest of a report's metrics are counts, where lower is not a regression, so "a metric
lower than last time" would list a run with fewer cases as a regression."""


NOTHING_LISTED: Final[str] = "No rule broken, and the end state matches none that is listed."
"""Why a play needs a look when the judge flagged nothing: no right end state matched."""


class _Read(BaseModel):
    model_config = ConfigDict(frozen=True)


class RequestFilters(_Read):
    """What narrows the request list; every part is optional and they are joined with AND.

    `errors` means status 500 or above, the line the error rate uses. `slow` means latency
    above the window's p95. `day` is a UTC date. Values reach SQL as bound parameters."""

    day: date | None = None
    tenant: str | None = None
    model: str | None = None
    errors: bool = False
    slow: bool = False
    cached: bool = False


class RequestsRead(_Read):
    """One page of the request list, its number and how many pages there are (at least 1).

    `slow_ms`, `tenants` and `models` describe the page window, not the filtered list."""

    rows: tuple[RequestRow, ...]
    page: int
    pages: int
    slow_ms: float | None = None
    tenants: tuple[str, ...] = ()
    models: tuple[str, ...] = ()


class HomeRead(_Read):
    """Today's ledger, the newest run, and that run's metrics for the gate verdict.

    `last_eval.passed` here is the run's stored "every stable case passed"; the home route
    replaces it with the verdict on `last_eval_metrics`."""

    requests_today: int
    cost_usd_today: float
    last_eval: LastEval | None
    last_eval_metrics: dict[str, Any] | None = None


class CostRead(_Read):
    days: tuple[CostDay, ...]
    savings: tuple[Saving, ...]
    savings_usd_total: float
    cost_per_request: tuple[AliasCost, ...]
    top_users: tuple[UserCost, ...] = ()
    prompt_cache_saved_usd: float = 0.0


class BlockedRelease(_Read):
    """A blocked deploy with the raw metrics a regression is computed from.

    `metrics` is the linked run's, `previous_metrics` the newest EARLIER passed run's
    of the same `set_name`; either is `None` when there is no such run.
    """

    git_sha: str
    deployed_at: datetime
    pr_url: str | None
    eval_run_id: str | None
    metrics: dict[str, Any] | None
    previous_metrics: dict[str, Any] | None


class QualityRead(_Read):
    runs: tuple[EvalRunSummary, ...]
    release_markers: tuple[ReleaseMarker, ...]
    categories: tuple[CategoryCount, ...]
    taxonomy: tuple[TagCount, ...]
    blocked: tuple[BlockedRelease, ...]
    needs_a_look: tuple[NeedsALookPlay, ...] = ()


def _number(value: object) -> float | None:
    """A JSON number as a float; `None` for anything else, `bool` included."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def regressed_metrics(
    current: Mapping[str, object] | None, previous: Mapping[str, object] | None
) -> tuple[Regression, ...]:
    """Every higher-is-better metric strictly lower than in the previous run.

    Either side missing is no regressions, and so is a metric either run did not
    measure.
    """
    if current is None or previous is None:
        return ()
    found: list[Regression] = []
    for name in HIGHER_IS_BETTER_METRICS:
        value, before = _number(current.get(name)), _number(previous.get(name))
        if value is not None and before is not None and value < before:
            found.append(Regression(metric=name, value=value, previous=before))
    return tuple(found)


class ConsoleReader(Protocol):
    """What the `/console/*` routes read Postgres through."""

    configured: bool

    def home(self, *, now: datetime) -> HomeRead: ...

    def cost(
        self,
        *,
        days: int,
        tenant: str | None,
        now: datetime,
        prices: PriceTable,
        start: date | None = None,
        end: date | None = None,
    ) -> CostRead: ...

    def month_spend(self, *, tenant: str | None, now: datetime) -> dict[str, float]: ...

    def quality(self, *, days: int, now: datetime) -> QualityRead: ...

    def recent_requests(
        self, *, page: int, filters: RequestFilters, days: int, now: datetime
    ) -> RequestsRead: ...

    def cache_hit_rate(self, days: int, now: datetime) -> float | None: ...

    def daily_requests(self, days: int, now: datetime) -> tuple[tuple[date, int, int], ...]: ...

    def model_tokens(self, days: int, now: datetime) -> tuple[ModelTokens, ...]: ...


class UnconfiguredConsoleReader:
    """What a gateway with no `database_url` holds. The routes never call it.

    It answers empty reads anyway, so a caller that forgets to check `configured`
    renders an empty page rather than crashing.
    """

    configured = False

    def home(self, *, now: datetime) -> HomeRead:
        return HomeRead(
            requests_today=0,
            cost_usd_today=0.0,
            last_eval=None,
        )

    def cost(
        self,
        *,
        days: int,
        tenant: str | None,
        now: datetime,
        prices: PriceTable,
        start: date | None = None,
        end: date | None = None,
    ) -> CostRead:
        return CostRead(days=(), savings=(), savings_usd_total=0.0, cost_per_request=())

    def month_spend(self, *, tenant: str | None, now: datetime) -> dict[str, float]:
        return {}

    def quality(self, *, days: int, now: datetime) -> QualityRead:
        return QualityRead(runs=(), release_markers=(), categories=(), taxonomy=(), blocked=())

    def recent_requests(
        self, *, page: int, filters: RequestFilters, days: int, now: datetime
    ) -> RequestsRead:
        return RequestsRead(rows=(), page=1, pages=1)

    def cache_hit_rate(self, days: int, now: datetime) -> float | None:
        return None

    def daily_requests(self, days: int, now: datetime) -> tuple[tuple[date, int, int], ...]:
        return ()

    def model_tokens(self, days: int, now: datetime) -> tuple[ModelTokens, ...]:
        return ()


def _outcome(metrics: object) -> float | None:
    """`outcome_every_play` out of a stored metrics document, or `None`."""
    return _number(metrics.get("outcome_every_play")) if isinstance(metrics, dict) else None


def _prompt_cache_saving(prices: PriceTable, row: tuple[Any, ...]) -> Decimal:
    """What one row group's prompt cache saved: its price with every cache token billed
    as plain input, minus its real price, at the price in force at its first row.

    Below zero when writes cost more than reads saved; 0 for an unpriced model or one
    with no cache rates. Tests pin this.
    """
    provider, model, input_tokens, output_tokens, reads, writes, first_ts = row
    key = f"{provider}/{model}"
    plain, out, read, write = int(input_tokens), int(output_tokens), int(reads), int(writes)
    as_input = Usage(
        prompt_tokens=plain + read + write,
        completion_tokens=out,
        total_tokens=plain + read + write + out,
    )
    real = Usage(
        prompt_tokens=plain,
        completion_tokens=out,
        total_tokens=plain + read + write + out,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )
    return cost_for(prices, key, first_ts, as_input).usd - cost_for(prices, key, first_ts, real).usd


def _tenant_clause(tenant: str | None) -> tuple[sql.Composable, tuple[str, ...]]:
    """` AND tenant = %s` and its parameter, or nothing when no tenant is asked for."""
    if tenant is None:
        return sql.SQL(""), ()
    return sql.SQL(" AND tenant = %s"), (tenant,)


class PostgresConsoleReader:
    """Every read over the pool the app already holds. Read-only, no DDL.

    Each method borrows one connection for all of its statements. Values reach
    SQL as bound parameters; the schema is the one `checked_schema` accepted.
    """

    configured = True

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]],
        *,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        self._pool = pool
        self._schema = checked_schema(schema)

    def _table(self, name: str) -> sql.Identifier:
        return sql.Identifier(self._schema, name)

    def home(self, *, now: datetime) -> HomeRead:
        start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        with self._pool.connection() as conn:
            ledger = conn.execute(
                sql.SQL(
                    "SELECT count(*), coalesce(sum(cost_usd), 0) FROM {} "
                    "WHERE ts >= %s AND ts < %s AND status < 500"
                ).format(self._table("cost_ledger")),
                (start, end),
            ).fetchone()
            run = conn.execute(
                sql.SQL(
                    "SELECT id, finished_at, passed, metrics FROM {} "
                    "ORDER BY started_at DESC LIMIT 1"
                ).format(self._table("eval_runs"))
            ).fetchone()
        assert ledger is not None
        return HomeRead(
            requests_today=ledger[0],
            cost_usd_today=float(ledger[1]),
            last_eval=None
            if run is None
            else LastEval(
                id=run[0], finished_at=run[1], passed=run[2], outcome_every_play=_outcome(run[3])
            ),
            last_eval_metrics=None if run is None or not isinstance(run[3], dict) else run[3],
        )

    def cache_hit_rate(self, days: int, now: datetime) -> float | None:
        """Cached ledger rows over all ledger rows in `[now - days, now]`; `None` with no rows.

        Counted from the ledger, not Prometheus: `increase()` reads 0 for a counter series
        born at 1, so one cached request showed a 0% rate. A test pins this.
        """
        with self._pool.connection() as conn:
            row = conn.execute(
                sql.SQL(
                    "SELECT count(*) FILTER (WHERE cached), count(*) FROM {} "
                    "WHERE ts >= %s AND ts <= %s AND status < 500"
                ).format(self._table("cost_ledger")),
                (now - timedelta(days=days), now),
            ).fetchone()
        assert row is not None
        cached, total = row
        return None if total == 0 else cached / total

    def daily_requests(self, days: int, now: datetime) -> tuple[tuple[date, int, int], ...]:
        """`(day, answered, failed)` per UTC day for the last `days` UTC dates ending today,
        oldest first: ledger rows below status 500, then rows at 500 or above.

        Every day is present, a day with no rows as `(day, 0, 0)`, so the chart has no gaps.
        A test pins this.
        """
        today = now.astimezone(UTC).date()
        first = today - timedelta(days=days - 1)
        start = datetime(first.year, first.month, first.day, tzinfo=UTC)
        end = datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=1)
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT (ts AT TIME ZONE 'UTC')::date, "
                    "count(*) FILTER (WHERE status < 500), "
                    "count(*) FILTER (WHERE status >= 500) FROM {} "
                    "WHERE ts >= %s AND ts < %s GROUP BY 1"
                ).format(self._table("cost_ledger")),
                (start, end),
            ).fetchall()
        counts: dict[date, tuple[int, int]] = {row[0]: (row[1], row[2]) for row in rows}
        return tuple(
            (day, *counts.get(day, (0, 0)))
            for day in (first + timedelta(days=offset) for offset in range(days))
        )

    def model_tokens(self, days: int, now: datetime) -> tuple[ModelTokens, ...]:
        """Tokens in and out per served model over `[now - days, now]`, by model name.

        Read from the ledger like Daily cost, so the two agree: the Prometheus counter
        missed OpenAI rows. A test pins this.
        """
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT model, coalesce(sum(input_tokens), 0), "
                    "coalesce(sum(output_tokens), 0), coalesce(sum(cache_read_tokens), 0), "
                    "coalesce(sum(cache_write_tokens), 0) "
                    "FROM {} WHERE ts >= %s AND ts <= %s AND status < 500 "
                    "GROUP BY model ORDER BY model LIMIT %s"
                ).format(self._table("cost_ledger")),
                (now - timedelta(days=days), now, ROW_LIMIT),
            ).fetchall()
        return tuple(
            ModelTokens(
                model=row[0],
                input=float(row[1]),
                output=float(row[2]),
                cache_read=float(row[3]),
                cache_write=float(row[4]),
            )
            for row in rows
        )

    def cost(
        self,
        *,
        days: int,
        tenant: str | None,
        now: datetime,
        prices: PriceTable,
        start: date | None = None,
        end: date | None = None,
    ) -> CostRead:
        since = now - timedelta(days=days)
        first_day = since.astimezone(UTC).date() if start is None else start
        last_day = now.astimezone(UTC).date() if end is None else end
        where_tenant, tenant_params = _tenant_clause(tenant)
        with self._pool.connection() as conn:
            day_rows = conn.execute(
                sql.SQL(
                    "SELECT (ts AT TIME ZONE 'UTC')::date, tenant, provider, count(*), "
                    "count(*) FILTER (WHERE cached), sum(cost_usd) FROM {} "
                    "WHERE status < 500 AND ts >= %s AND ts < %s{} "
                    "GROUP BY 1, tenant, provider ORDER BY 1, tenant, provider"
                ).format(self._table("cost_ledger"), where_tenant),
                (
                    datetime.combine(first_day, time.min, UTC),
                    datetime.combine(last_day + timedelta(days=1), time.min, UTC),
                    *tenant_params,
                ),
            ).fetchall()
            cached_rows = conn.execute(
                sql.SQL(
                    "SELECT (ts AT TIME ZONE 'UTC')::date, tenant, provider, model, "
                    "sum(input_tokens), sum(output_tokens), min(ts) FROM {} "
                    "WHERE cached AND status < 500 AND ts >= %s{} "
                    "GROUP BY 1, tenant, provider, model, price_version"
                ).format(self._table("cost_ledger"), where_tenant),
                (since, *tenant_params),
            ).fetchall()
            prompt_cache_rows = conn.execute(
                sql.SQL(
                    "SELECT provider, model, sum(input_tokens), sum(output_tokens), "
                    "sum(cache_read_tokens), sum(cache_write_tokens), min(ts) FROM {} "
                    "WHERE NOT cached AND status < 500 AND ts >= %s AND ts <= %s{} "
                    "AND (cache_read_tokens > 0 OR cache_write_tokens > 0) "
                    "GROUP BY provider, model, price_version"
                ).format(self._table("cost_ledger"), where_tenant),
                (since, now, *tenant_params),
            ).fetchall()
            alias_rows = conn.execute(
                sql.SQL(
                    "SELECT alias, sum(cost_usd) / count(*) FROM {} "
                    "WHERE NOT cached AND status < 500 AND ts >= %s{} GROUP BY alias ORDER BY alias"
                ).format(self._table("cost_ledger"), where_tenant),
                (since, *tenant_params),
            ).fetchall()
            # Per tenant, the users with the highest served cost; rows with no user left out.
            user_rows = conn.execute(
                sql.SQL(
                    "SELECT tenant, user_id, requests, cost_usd FROM ("
                    "SELECT tenant, user_id, count(*) AS requests, sum(cost_usd) AS cost_usd, "
                    "row_number() OVER (PARTITION BY tenant "
                    "ORDER BY sum(cost_usd) DESC, user_id) AS rank FROM {} "
                    "WHERE user_id IS NOT NULL AND status < 500 AND ts >= %s{} "
                    "GROUP BY tenant, user_id) AS ranked "
                    "WHERE rank <= %s ORDER BY tenant, rank"
                ).format(self._table("cost_ledger"), where_tenant),
                (since, *tenant_params, TOP_USERS_PER_TENANT),
            ).fetchall()

        saved: defaultdict[tuple[date, str], Decimal] = defaultdict(Decimal)
        for day, row_tenant, provider, model, input_tokens, output_tokens, first_ts in cached_rows:
            usage = Usage(
                prompt_tokens=int(input_tokens),
                completion_tokens=int(output_tokens),
                total_tokens=int(input_tokens) + int(output_tokens),
            )
            saved[(day, row_tenant)] += cost_for(prices, f"{provider}/{model}", first_ts, usage).usd
        prompt_saved = sum(
            (_prompt_cache_saving(prices, row) for row in prompt_cache_rows), Decimal(0)
        )
        savings = tuple(
            Saving(day=day, tenant=row_tenant, saved_usd=float(usd))
            for (day, row_tenant), usd in sorted(saved.items())
        )
        return CostRead(
            days=tuple(
                CostDay(
                    day=row[0],
                    tenant=row[1],
                    provider=row[2],
                    requests=int(row[3]),
                    cached_requests=int(row[4]),
                    cost_usd=float(row[5]),
                )
                for row in day_rows
            ),
            savings=savings,
            savings_usd_total=float(sum(saved.values(), Decimal(0))),
            prompt_cache_saved_usd=float(prompt_saved),
            cost_per_request=tuple(
                AliasCost(alias=row[0], cost_usd=float(row[1])) for row in alias_rows
            ),
            top_users=tuple(
                UserCost(
                    tenant=row[0], user_id=row[1], requests=int(row[2]), cost_usd=float(row[3])
                )
                for row in user_rows
            ),
        )

    def month_spend(self, *, tenant: str | None, now: datetime) -> dict[str, float]:
        """USD per tenant from the not-cached ledger rows since the start of this UTC month.

        The budget bars' spend. A tenant with no rows is left out, so the
        caller tells "spent nothing" from "unknown". A test pins this.
        """
        month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        where_tenant, tenant_params = _tenant_clause(tenant)
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT tenant, sum(cost_usd) FROM {} "
                    "WHERE NOT cached AND ts >= %s{} GROUP BY tenant"
                ).format(self._table("cost_ledger"), where_tenant),
                (month_start, *tenant_params),
            ).fetchall()
        return {row[0]: float(row[1]) for row in rows}

    def quality(self, *, days: int, now: datetime) -> QualityRead:
        since = now - timedelta(days=days)
        runs_table = self._table("eval_runs")
        cases_table = self._table("eval_cases")
        releases_table = self._table("releases")
        with self._pool.connection() as conn:
            runs = conn.execute(
                sql.SQL(
                    "SELECT id, started_at, run_kind, git_sha, passed, metrics FROM {} "
                    "WHERE started_at >= %s ORDER BY started_at DESC LIMIT %s"
                ).format(runs_table),
                (since, ROW_LIMIT),
            ).fetchall()
            markers = conn.execute(
                sql.SQL(
                    "SELECT git_sha, deployed_at, status, eval_run_id FROM {} "
                    "WHERE deployed_at >= %s ORDER BY deployed_at DESC LIMIT %s"
                ).format(releases_table),
                (since, ROW_LIMIT),
            ).fetchall()
            categories = (
                conn.execute(
                    sql.SQL(
                        "SELECT category, count(*) FILTER (WHERE passed), count(*) FROM {} "
                        "WHERE run_id = %s GROUP BY category ORDER BY category"
                    ).format(cases_table),
                    (runs[0][0],),
                ).fetchall()
                if runs
                else []
            )
            taxonomy = conn.execute(
                sql.SQL(
                    "SELECT tag, count(*) FROM {} AS c JOIN {} AS r ON r.id = c.run_id, "
                    "unnest(c.tags) AS tag WHERE r.started_at >= %s "
                    "GROUP BY tag ORDER BY count(*) DESC, tag"
                ).format(cases_table, runs_table),
                (since,),
            ).fetchall()
            needs_a_look = (
                conn.execute(
                    sql.SQL(
                        "SELECT case_id, play, flagged FROM {} "
                        "WHERE run_id = %s AND result = 'needs_a_look' "
                        "ORDER BY case_id, play LIMIT %s"
                    ).format(self._table("eval_plays")),
                    (runs[0][0], ROW_LIMIT),
                ).fetchall()
                if runs
                else []
            )
            blocked = conn.execute(
                sql.SQL(
                    "SELECT rel.git_sha, rel.deployed_at, rel.pr_url, rel.eval_run_id, "
                    "run.metrics, ("
                    "SELECT prev.metrics FROM {runs} AS prev "
                    "WHERE prev.set_name = run.set_name AND prev.passed "
                    "AND prev.started_at < run.started_at "
                    "ORDER BY prev.started_at DESC LIMIT 1) "
                    "FROM {releases} AS rel LEFT JOIN {runs} AS run ON run.id = rel.eval_run_id "
                    "WHERE rel.status = 'blocked' AND rel.deployed_at >= %s "
                    "ORDER BY rel.deployed_at DESC LIMIT %s"
                ).format(runs=runs_table, releases=releases_table),
                (since, ROW_LIMIT),
            ).fetchall()
        return QualityRead(
            runs=tuple(
                EvalRunSummary(
                    id=row[0],
                    started_at=row[1],
                    run_kind=row[2],
                    git_sha=row[3],
                    passed=row[4],
                    metrics=row[5],
                )
                for row in runs
            ),
            release_markers=tuple(
                ReleaseMarker(git_sha=row[0], deployed_at=row[1], status=row[2], eval_run_id=row[3])
                for row in markers
            ),
            categories=tuple(
                CategoryCount(category=row[0], passed=row[1], total=row[2]) for row in categories
            ),
            taxonomy=tuple(TagCount(tag=row[0], count=row[1]) for row in taxonomy),
            blocked=tuple(
                BlockedRelease(
                    git_sha=row[0],
                    deployed_at=row[1],
                    pr_url=row[2],
                    eval_run_id=row[3],
                    metrics=row[4],
                    previous_metrics=row[5],
                )
                for row in blocked
            ),
            needs_a_look=tuple(
                NeedsALookPlay(case_id=row[0], play=row[1], why="; ".join(row[2]) or NOTHING_LISTED)
                for row in needs_a_look
            ),
        )

    def recent_requests(
        self, *, page: int, filters: RequestFilters, days: int, now: datetime
    ) -> RequestsRead:
        """One page of the ledger, newest first, each row with `trace_url` null.

        OFFSET paging with a `count(*)` on the same connection, since the console shows
        "Page X of Y" and jumps by page number. `trace_id` breaks a tie on `ts`, so no row
        shows on two pages. A page past the last serves the last,
        and an empty ledger is page 1 of 1.

        The filters narrow the list and its count only. `slow_ms` (the 95th percentile of
        the window's latencies), `tenants` and `models` are read over the whole window
        `[now - days, now]` and ignore the filters, so a filter list never shrinks to the
        choice already made. Tests pin this."""
        table = self._table("cost_ledger")
        since = now - timedelta(days=days)
        with self._pool.connection() as conn:
            window = conn.execute(
                sql.SQL(
                    "SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) "
                    "FROM {} WHERE ts >= %s AND ts <= %s AND latency_ms IS NOT NULL"
                ).format(table),
                (since, now),
            ).fetchone()
            slow_ms = None if window is None or window[0] is None else float(window[0])
            options = conn.execute(
                sql.SQL(
                    "SELECT 'tenant', tenant FROM {t} WHERE ts >= %s AND ts <= %s GROUP BY tenant "
                    "UNION ALL SELECT 'model', model FROM {t} WHERE ts >= %s AND ts <= %s "
                    "GROUP BY model ORDER BY 1, 2"
                ).format(t=table),
                (since, now, since, now),
            ).fetchall()
            clauses: list[sql.Composable] = []
            params: list[object] = []
            if filters.day is not None:
                start = datetime(filters.day.year, filters.day.month, filters.day.day, tzinfo=UTC)
                clauses.append(sql.SQL("ts >= %s AND ts < %s"))
                params += [start, start + timedelta(days=1)]
            if filters.tenant is not None:
                clauses.append(sql.SQL("tenant = %s"))
                params.append(filters.tenant)
            if filters.model is not None:
                clauses.append(sql.SQL("model = %s"))
                params.append(filters.model)
            if filters.errors:
                clauses.append(sql.SQL("status >= 500"))
            if filters.slow:
                # No latencies in the window means no line, and nothing is slow. Slow is
                # judged inside the window the line was taken from.
                if slow_ms is None:
                    clauses.append(sql.SQL("false"))
                else:
                    clauses.append(sql.SQL("latency_ms > %s AND ts >= %s"))
                    params += [slow_ms, since]
            if filters.cached:
                clauses.append(sql.SQL("cached"))
            where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
            counted = conn.execute(
                sql.SQL("SELECT count(*) FROM {}{}").format(table, where), params
            ).fetchone()
            total = 0 if counted is None else int(counted[0])
            pages = max(1, -(-total // PAGE_SIZE))
            page = min(max(page, 1), pages)
            rows = conn.execute(
                sql.SQL(
                    "SELECT ts, tenant, alias, provider, model, input_tokens, output_tokens, "
                    "cached, cost_usd, prompt_version, trace_id, status, latency_ms "
                    "FROM {}{} ORDER BY ts DESC, trace_id DESC LIMIT %s OFFSET %s"
                ).format(table, where),
                (*params, PAGE_SIZE, (page - 1) * PAGE_SIZE),
            ).fetchall()
        return RequestsRead(
            page=page,
            pages=pages,
            slow_ms=slow_ms,
            tenants=tuple(row[1] for row in options if row[0] == "tenant"),
            models=tuple(row[1] for row in options if row[0] == "model"),
            rows=tuple(
                RequestRow(
                    ts=row[0],
                    tenant=row[1],
                    alias=row[2],
                    provider=row[3],
                    model=row[4],
                    input_tokens=row[5],
                    output_tokens=row[6],
                    cached=row[7],
                    cost_usd=float(row[8]),
                    prompt_version=row[9],
                    trace_id=row[10],
                    trace_url=None,
                    status=row[11],
                    latency_ms=row[12],
                )
                for row in rows
            ),
        )
