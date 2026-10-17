"""Every Postgres read the console makes, one method per page.

The reader shape follows the writers in `storage/`: a Protocol, a Postgres
implementation over the app's one pool, and a stand-in for a gateway with no
`database_url`. The stand-in is `UnconfiguredConsoleReader`, and its
`configured = False` is what the routes read to answer `postgres: unconfigured`
without calling it at all.

**Windows.** A page's window starts at `now - timedelta(days=days)`, compared with
`>=` on a timestamp and with `day >= since.date()` on the daily rollup, whose days
are UTC dates. "Today" on the home page is `[UTC midnight, +1 day)` over the live
ledger rather than the rollup: the rollup refreshes only on
`make refresh-rollup`. Every query is bounded
by its window or by `ROW_LIMIT`.

**The rollup may be unpopulated.** PostgreSQL refuses a SELECT from a materialised
view that was never refreshed, so `cost` asks `pg_matviews.ispopulated` first and
answers `rollup_populated: False` with no days rather than raising.

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
from datetime import UTC, date, datetime, timedelta
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
    Incident,
    LastEval,
    RegionCount,
    Regression,
    Release,
    ReleaseMarker,
    Saving,
    TagCount,
)
from app.schemas.provider import Usage
from app.storage.db import DEFAULT_SCHEMA, checked_schema

ROW_LIMIT: Final[int] = 100
"""The most rows any list on a page carries."""

COST_ROLLUP: Final[str] = "cost_ledger_daily"
"""The materialised view `migrations/002_cost_ledger.sql` creates."""

HIGHER_IS_BETTER_METRICS: Final[tuple[str, ...]] = (
    "intent_accuracy",
    "tool_selection_accuracy",
    "tool_args_accuracy",
    "decision_accuracy",
    "risk_class_accuracy",
    "refusal_precision",
    "refusal_recall",
    "clarify_accuracy",
    "citation_accuracy",
    "injection_resistance",
)
"""The eval metrics where lower is worse, in `evals.metrics.METRIC_NAMES` order.

The rest of a report's metrics are counts, latencies and costs, where lower is an
improvement, so "a metric lower than last time" would list a faster, cheaper run
as a regression."""


class _Read(BaseModel):
    model_config = ConfigDict(frozen=True)


class HomeRead(_Read):
    requests_today: int
    cost_usd_today: float
    residency_violations_today: int
    last_eval: LastEval | None
    open_incidents: int


class CostRead(_Read):
    rollup_populated: bool
    days: tuple[CostDay, ...]
    savings: tuple[Saving, ...]
    savings_usd_total: float
    cost_per_request: tuple[AliasCost, ...]


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


class ResidencyRead(_Read):
    by_region: tuple[RegionCount, ...]
    violations: int


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
        self, *, days: int, tenant: str | None, now: datetime, prices: PriceTable
    ) -> CostRead: ...

    def quality(self, *, days: int, now: datetime) -> QualityRead: ...

    def residency(self, *, days: int, tenant: str | None, now: datetime) -> ResidencyRead: ...

    def incidents(self) -> tuple[Incident, ...]: ...

    def releases(self) -> tuple[Release, ...]: ...


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
            residency_violations_today=0,
            last_eval=None,
            open_incidents=0,
        )

    def cost(self, *, days: int, tenant: str | None, now: datetime, prices: PriceTable) -> CostRead:
        return CostRead(
            rollup_populated=False, days=(), savings=(), savings_usd_total=0.0, cost_per_request=()
        )

    def quality(self, *, days: int, now: datetime) -> QualityRead:
        return QualityRead(runs=(), release_markers=(), categories=(), taxonomy=(), blocked=())

    def residency(self, *, days: int, tenant: str | None, now: datetime) -> ResidencyRead:
        return ResidencyRead(by_region=(), violations=0)

    def incidents(self) -> tuple[Incident, ...]:
        return ()

    def releases(self) -> tuple[Release, ...]:
        return ()


def _accuracy(metrics: object) -> float | None:
    """`decision_accuracy` out of a stored metrics document, or `None`."""
    return _number(metrics.get("decision_accuracy")) if isinstance(metrics, dict) else None


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
                    "SELECT count(*), coalesce(sum(cost_usd), 0) FROM {} WHERE ts >= %s AND ts < %s"
                ).format(self._table("cost_ledger")),
                (start, end),
            ).fetchone()
            violations = conn.execute(
                sql.SQL(
                    "SELECT count(*) FROM {} WHERE NOT allowed AND ts >= %s AND ts < %s"
                ).format(self._table("residency_log")),
                (start, end),
            ).fetchone()
            run = conn.execute(
                sql.SQL(
                    "SELECT id, finished_at, passed, metrics FROM {} "
                    "ORDER BY started_at DESC LIMIT 1"
                ).format(self._table("eval_runs"))
            ).fetchone()
            open_incidents = conn.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE resolved_at IS NULL").format(
                    self._table("incidents")
                )
            ).fetchone()
        assert ledger is not None and violations is not None and open_incidents is not None
        return HomeRead(
            requests_today=ledger[0],
            cost_usd_today=float(ledger[1]),
            residency_violations_today=violations[0],
            last_eval=None
            if run is None
            else LastEval(
                id=run[0], finished_at=run[1], passed=run[2], decision_accuracy=_accuracy(run[3])
            ),
            open_incidents=open_incidents[0],
        )

    def cost(self, *, days: int, tenant: str | None, now: datetime, prices: PriceTable) -> CostRead:
        since = now - timedelta(days=days)
        where_tenant, tenant_params = _tenant_clause(tenant)
        with self._pool.connection() as conn:
            populated = conn.execute(
                "SELECT ispopulated FROM pg_matviews WHERE schemaname = %s AND matviewname = %s",
                (self._schema, COST_ROLLUP),
            ).fetchone()
            rollup_populated = populated is not None and bool(populated[0])
            day_rows = (
                conn.execute(
                    sql.SQL(
                        "SELECT day, tenant, provider, sum(requests), sum(cached_requests), "
                        "sum(cost_usd) FROM {} WHERE day >= %s{} "
                        "GROUP BY day, tenant, provider ORDER BY day, tenant, provider"
                    ).format(self._table(COST_ROLLUP), where_tenant),
                    (since.astimezone(UTC).date(), *tenant_params),
                ).fetchall()
                if rollup_populated
                else []
            )
            cached_rows = conn.execute(
                sql.SQL(
                    "SELECT (ts AT TIME ZONE 'UTC')::date, tenant, provider, model, "
                    "sum(input_tokens), sum(output_tokens), min(ts) FROM {} "
                    "WHERE cached AND ts >= %s{} "
                    "GROUP BY 1, tenant, provider, model, price_version"
                ).format(self._table("cost_ledger"), where_tenant),
                (since, *tenant_params),
            ).fetchall()
            alias_rows = conn.execute(
                sql.SQL(
                    "SELECT alias, sum(cost_usd) / count(*) FROM {} "
                    "WHERE NOT cached AND ts >= %s{} GROUP BY alias ORDER BY alias"
                ).format(self._table("cost_ledger"), where_tenant),
                (since, *tenant_params),
            ).fetchall()

        saved: defaultdict[tuple[date, str], Decimal] = defaultdict(Decimal)
        for day, row_tenant, provider, model, input_tokens, output_tokens, first_ts in cached_rows:
            usage = Usage(
                prompt_tokens=int(input_tokens),
                completion_tokens=int(output_tokens),
                total_tokens=int(input_tokens) + int(output_tokens),
            )
            saved[(day, row_tenant)] += cost_for(prices, f"{provider}/{model}", first_ts, usage).usd
        savings = tuple(
            Saving(day=day, tenant=row_tenant, saved_usd=float(usd))
            for (day, row_tenant), usd in sorted(saved.items())
        )
        return CostRead(
            rollup_populated=rollup_populated,
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
            cost_per_request=tuple(
                AliasCost(alias=row[0], cost_usd=float(row[1])) for row in alias_rows
            ),
        )

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
                    "SELECT git_sha, deployed_at, status FROM {} "
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
                ReleaseMarker(git_sha=row[0], deployed_at=row[1], status=row[2]) for row in markers
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
        )

    def residency(self, *, days: int, tenant: str | None, now: datetime) -> ResidencyRead:
        since = now - timedelta(days=days)
        where_tenant, tenant_params = _tenant_clause(tenant)
        table = self._table("residency_log")
        with self._pool.connection() as conn:
            regions = conn.execute(
                sql.SQL(
                    "SELECT tenant, region, provider, count(*) FROM {} WHERE ts >= %s{} "
                    "GROUP BY tenant, region, provider ORDER BY tenant, region, provider"
                ).format(table, where_tenant),
                (since, *tenant_params),
            ).fetchall()
            violations = conn.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE NOT allowed AND ts >= %s{}").format(
                    table, where_tenant
                ),
                (since, *tenant_params),
            ).fetchone()
        assert violations is not None
        return ResidencyRead(
            by_region=tuple(
                RegionCount(tenant=row[0], region=row[1], provider=row[2], requests=row[3])
                for row in regions
            ),
            violations=violations[0],
        )

    def incidents(self) -> tuple[Incident, ...]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT id, started_at, resolved_at, severity, title, summary, postmortem_url "
                    "FROM {} ORDER BY started_at DESC LIMIT %s"
                ).format(self._table("incidents")),
                (ROW_LIMIT,),
            ).fetchall()
        return tuple(
            Incident(
                id=row[0],
                started_at=row[1],
                resolved_at=row[2],
                severity=row[3],
                title=row[4],
                summary=row[5],
                postmortem_url=row[6],
                open=row[2] is None,
            )
            for row in rows
        )

    def releases(self) -> tuple[Release, ...]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT rel.git_sha, rel.deployed_at, rel.status, rel.pr_url, "
                    "rel.eval_run_id, run.passed, run.metrics "
                    "FROM {} AS rel LEFT JOIN {} AS run ON run.id = rel.eval_run_id "
                    "ORDER BY rel.deployed_at DESC LIMIT %s"
                ).format(self._table("releases"), self._table("eval_runs")),
                (ROW_LIMIT,),
            ).fetchall()
        return tuple(
            Release(
                git_sha=row[0],
                deployed_at=row[1],
                status=row[2],
                pr_url=row[3],
                eval_run_id=row[4],
                eval_passed=row[5],
                decision_accuracy=_accuracy(row[6]),
            )
            for row in rows
        )
