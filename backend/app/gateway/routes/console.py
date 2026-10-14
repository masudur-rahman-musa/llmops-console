"""The console's read surface: one route per page, every number a page shows.

Seven routes, all behind `routes.admin.admin_key`:

    GET /console/home          today's traffic, spend, latency and the last eval
    GET /console/cost          the daily rollup, estimated cache savings, budgets
    GET /console/quality       eval runs, categories, taxonomy, blocked releases
    GET /console/residency     where each tenant's data went, and redactions
    GET /console/reliability   latency and error-rate series, breakers, fallbacks
    GET /console/incidents     the timeline
    GET /console/releases      deploys, each with its eval result

The console reads through here and never touches Postgres or Prometheus itself.
A working tenant key is 401 on all seven, as on `/admin/*`
(`test_every_console_route_is_401_with_a_tenant_key`).

**Sources are flags, never errors** (decision C, a design note). Every body carries
`sources: {postgres, prometheus}`, each `ok`, `unconfigured` or `error`. A source
that is not configured is not asked; one that fails is logged at ERROR and its
fields answer `null` or `[]` while the route stays 200 and the other source's
fields still render (`test_a_failing_prometheus_is_200_with_prometheus_error_and_an_error_log`,
`test_a_failing_reader_is_200_with_postgres_error_and_an_error_log`). A page that
reads one source reports the other as `unconfigured`.

**Inputs.** `days` is an integer from `MIN_DAYS` to `MAX_DAYS`, refused as 422
`VALIDATION` naming `days` by the request validation handler before the route body
runs (`test_the_days_boundary_is_inclusive`,
`test_a_non_integer_days_is_422_naming_days`). `tenant` is pre-checked by admin's
`_checked_tenant` before any query, because it is interpolated into a PromQL
label matcher (`test_the_tenant_rejection_table`). The reliability page's SLO error
budgets read a fixed 30-day window and ignore `days` (`test_slos_ignore_days`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict

from app.gateway.console_reads import (
    AliasCost,
    CategoryCount,
    ConsoleReader,
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
    regressed_metrics,
)
from app.gateway.prometheus import (
    AVAILABILITY_ALLOWED_BAD,
    AVAILABILITY_BUDGET_QUERY,
    AVAILABILITY_OBJECTIVE,
    CIRCUITS_QUERY,
    ERROR_RATE_QUERY,
    LATENCY_ALLOWED_SLOW,
    LATENCY_BUDGET_QUERY,
    LATENCY_THRESHOLD_SECONDS,
    P95_QUERY,
    RESIDENCY_VIOLATIONS_30D_QUERY,
    PrometheusClient,
    PrometheusError,
    fallbacks_query,
    redactions_query,
)
from app.gateway.routes.admin import _checked_tenant, admin_key
from app.limits.budget import budget_ratio, current_month, to_micro_usd, to_usd
from app.limits.prices import PriceStore
from app.model_routing.policy import PolicyStore

logger = logging.getLogger(__name__)

CONSOLE_HOME: Final[str] = "/console/home"
CONSOLE_COST: Final[str] = "/console/cost"
CONSOLE_QUALITY: Final[str] = "/console/quality"
CONSOLE_RESIDENCY: Final[str] = "/console/residency"
CONSOLE_RELIABILITY: Final[str] = "/console/reliability"
CONSOLE_INCIDENTS: Final[str] = "/console/incidents"
CONSOLE_RELEASES: Final[str] = "/console/releases"

CONSOLE_PATHS: Final[tuple[str, ...]] = (
    CONSOLE_HOME,
    CONSOLE_COST,
    CONSOLE_QUALITY,
    CONSOLE_RESIDENCY,
    CONSOLE_RELIABILITY,
    CONSOLE_INCIDENTS,
    CONSOLE_RELEASES,
)
"""All seven, pinned as literals against the app's OpenAPI document by
`test_the_seven_console_paths_are_pinned` and kept off every public router by
`tests/infra/test_k8s_apps.py::test_no_public_router_matches_a_console_path`."""

DAYS_PARAM: Final[str] = "days"
MIN_DAYS: Final[int] = 1
MAX_DAYS: Final[int] = 90
DEFAULT_DAYS: Final[int] = 30

SECONDS_PER_DAY: Final[int] = 86_400

SLO_WINDOW: Final[str] = "30d"
"""The window every SLO row reports, matching the `[30d]` in each budget query; it
never follows `days` (`test_slos_rows_are_availability_latency_residency_in_order`)."""

CIRCUIT_STATE_NAMES: Final[dict[int, str]] = {0: "closed", 1: "half_open", 2: "open"}
"""`gateway_circuit_state`'s values as the page shows them. Any other value is
skipped rather than guessed at (`test_the_reliability_body_reads_the_named_series`)."""


class SourceState(StrEnum):
    OK = "ok"
    UNCONFIGURED = "unconfigured"
    ERROR = "error"


class _Body(BaseModel):
    model_config = ConfigDict(frozen=True)


class Sources(_Body):
    postgres: SourceState
    prometheus: SourceState


class HomePage(_Body):
    sources: Sources
    requests_today: int | None
    cost_usd_today: float | None
    p95_seconds: float | None
    error_rate: float | None
    residency_violations_today: int | None
    last_eval: LastEval | None
    open_incidents: int | None


class Budget(_Body):
    tenant: str
    cap_usd: float
    spent_usd: float | None
    ratio: float | None


class CostPage(_Body):
    sources: Sources
    rollup_populated: bool | None
    days: tuple[CostDay, ...]
    savings: tuple[Saving, ...]
    savings_usd_total: float | None
    estimated: bool
    budgets: tuple[Budget, ...]
    cost_per_request: tuple[AliasCost, ...]


class BlockedItem(_Body):
    git_sha: str
    deployed_at: datetime
    pr_url: str | None
    eval_run_id: str | None
    regressed: tuple[Regression, ...]


class QualityPage(_Body):
    sources: Sources
    runs: tuple[EvalRunSummary, ...]
    release_markers: tuple[ReleaseMarker, ...]
    categories: tuple[CategoryCount, ...]
    taxonomy: tuple[TagCount, ...]
    blocked: tuple[BlockedItem, ...]


class RedactionCount(_Body):
    tenant: str
    kind: str
    count: float


class ResidencyPage(_Body):
    sources: Sources
    by_region: tuple[RegionCount, ...]
    violations: int | None
    redactions: tuple[RedactionCount, ...]


class Circuit(_Body):
    provider: str
    state: str


class FallbackCount(_Body):
    tenant: str
    alias: str
    provider: str
    count: float


class Slo(_Body):
    """One SLO's error budget. `budget_remaining` is negative when overspent, never clamped."""

    name: str
    objective: float
    window: str
    observed: float | None
    budget_remaining: float | None


class ReliabilityPage(_Body):
    sources: Sources
    slos: tuple[Slo, ...]
    latency_p95: tuple[tuple[float, float], ...]
    error_rate: tuple[tuple[float, float], ...]
    circuits: tuple[Circuit, ...]
    fallbacks: tuple[FallbackCount, ...]


class IncidentsPage(_Body):
    sources: Sources
    incidents: tuple[Incident, ...]


class ReleasesPage(_Body):
    sources: Sources
    releases: tuple[Release, ...]


Days = Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)]


def _postgres[T](
    request: Request, page: str, read: Callable[[ConsoleReader], T]
) -> tuple[SourceState, T | None]:
    """One page's Postgres read, as a state and a value.

    Only `psycopg.Error` is a source failure; the pool's own timeout is one of
    them. Anything else is a bug and stays a 500.
    """
    reader: ConsoleReader = request.app.state.console_reader
    if not reader.configured:
        return SourceState.UNCONFIGURED, None
    try:
        return SourceState.OK, read(reader)
    except psycopg.Error as exc:
        logger.error("console %s: the postgres read failed: %s", page, type(exc).__name__)
        return SourceState.ERROR, None


def _prometheus[T](
    request: Request, page: str, read: Callable[[PrometheusClient], T]
) -> tuple[SourceState, T | None]:
    """One page's Prometheus reads, as a state and a value. One failure fails them all."""
    client: PrometheusClient | None = request.app.state.prometheus
    if client is None:
        return SourceState.UNCONFIGURED, None
    try:
        return SourceState.OK, read(client)
    except PrometheusError as exc:
        logger.error("console %s: the prometheus read failed: %s", page, exc)
        return SourceState.ERROR, None


def _first(samples: list[tuple[dict[str, str], float]]) -> float | None:
    return samples[0][1] if samples else None


router = APIRouter(dependencies=[Depends(admin_key)])


@router.get(CONSOLE_HOME)
def home(request: Request) -> HomePage:
    """Today from the live ledger (decision F), p95 and error rate from Prometheus."""
    now = datetime.now(UTC)
    postgres, read = _postgres(request, "home", lambda reader: reader.home(now=now))
    prometheus, rates = _prometheus(
        request,
        "home",
        lambda client: (
            _first(client.instant_query(P95_QUERY)),
            _first(client.instant_query(ERROR_RATE_QUERY)),
        ),
    )
    p95, error_rate = rates if rates is not None else (None, None)
    return HomePage(
        sources=Sources(postgres=postgres, prometheus=prometheus),
        requests_today=None if read is None else read.requests_today,
        cost_usd_today=None if read is None else read.cost_usd_today,
        p95_seconds=p95,
        error_rate=error_rate,
        residency_violations_today=None if read is None else read.residency_violations_today,
        last_eval=None if read is None else read.last_eval,
        open_incidents=None if read is None else read.open_incidents,
    )


def _budgets(request: Request, tenant: str | None) -> tuple[Budget, ...]:
    """Each tenant's cap from the policy against its spend counter for this UTC month.

    Read from neither source: the counter is the budget store the chat route
    debits, so this answers even with Postgres down. An unknown spend (a Redis
    outage) is `null` rather than zero (`test_budgets_read_the_policy_cap_and_the_store_spend`).
    """
    store: PolicyStore = request.app.state.policy
    policy = store.policy
    month = current_month()
    budgets: list[Budget] = []
    for name, tenant_policy in policy.tenants.items():
        if tenant is not None and name != tenant:
            continue
        cap = tenant_policy.budget_usd_month
        spend = request.app.state.budget_store.spend(name, month)
        budgets.append(
            Budget(
                tenant=name,
                cap_usd=float(cap),
                spent_usd=None if spend is None else float(to_usd(spend)),
                ratio=None
                if spend is None
                else budget_ratio(spend_micro_usd=spend, cap_micro_usd=to_micro_usd(cap)),
            )
        )
    return tuple(budgets)


@router.get(CONSOLE_COST)
def cost(request: Request, days: Days = DEFAULT_DAYS, tenant: str | None = None) -> CostPage:
    """The rollup's days, estimated cache savings, budgets and cost per request."""
    if tenant is not None:
        _checked_tenant(tenant)
    now = datetime.now(UTC)
    prices: PriceStore = request.app.state.prices
    postgres, read = _postgres(
        request,
        "cost",
        lambda reader: reader.cost(days=days, tenant=tenant, now=now, prices=prices.table),
    )
    return CostPage(
        sources=Sources(postgres=postgres, prometheus=SourceState.UNCONFIGURED),
        rollup_populated=None if read is None else read.rollup_populated,
        days=() if read is None else read.days,
        savings=() if read is None else read.savings,
        savings_usd_total=None if read is None else read.savings_usd_total,
        estimated=True,
        budgets=_budgets(request, tenant),
        cost_per_request=() if read is None else read.cost_per_request,
    )


@router.get(CONSOLE_QUALITY)
def quality(request: Request, days: Days = DEFAULT_DAYS) -> QualityPage:
    """Eval runs over the window, and what regressed behind each blocked release (decision G)."""
    now = datetime.now(UTC)
    postgres, read = _postgres(
        request, "quality", lambda reader: reader.quality(days=days, now=now)
    )
    if read is None:
        return QualityPage(
            sources=Sources(postgres=postgres, prometheus=SourceState.UNCONFIGURED),
            runs=(),
            release_markers=(),
            categories=(),
            taxonomy=(),
            blocked=(),
        )
    return QualityPage(
        sources=Sources(postgres=postgres, prometheus=SourceState.UNCONFIGURED),
        runs=read.runs,
        release_markers=read.release_markers,
        categories=read.categories,
        taxonomy=read.taxonomy,
        blocked=tuple(
            BlockedItem(
                git_sha=item.git_sha,
                deployed_at=item.deployed_at,
                pr_url=item.pr_url,
                eval_run_id=item.eval_run_id,
                regressed=regressed_metrics(item.metrics, item.previous_metrics),
            )
            for item in read.blocked
        ),
    )


@router.get(CONSOLE_RESIDENCY)
def residency(
    request: Request, days: Days = DEFAULT_DAYS, tenant: str | None = None
) -> ResidencyPage:
    """Requests per tenant, region and provider, violations, and redactions by kind."""
    if tenant is not None:
        _checked_tenant(tenant)
    now = datetime.now(UTC)
    postgres, read = _postgres(
        request, "residency", lambda reader: reader.residency(days=days, tenant=tenant, now=now)
    )
    prometheus, redactions = _prometheus(
        request,
        "residency",
        lambda client: tuple(
            RedactionCount(
                tenant=labels.get("tenant", ""), kind=labels.get("kind", ""), count=value
            )
            for labels, value in client.instant_query(redactions_query(days, tenant))
        ),
    )
    return ResidencyPage(
        sources=Sources(postgres=postgres, prometheus=prometheus),
        by_region=() if read is None else read.by_region,
        violations=None if read is None else read.violations,
        redactions=redactions or (),
    )


def _slos(client: PrometheusClient) -> tuple[Slo, ...]:
    """The three gateway-measured SLO budgets over the fixed `SLO_WINDOW`.

    No traffic makes both ratios `0 / 0`, which Prometheus answers `NaN` and the
    client drops, so an empty result is `null` rather than a full budget
    (`test_nan_gives_null_observed_and_budget`). An overspent budget stays negative
    (`test_an_overspent_budget_is_negative_and_unclamped`). A failure raises
    `PrometheusError` into the page's one `_prometheus` read, which empties `slos`
    with everything else
    (`test_a_failing_budget_query_is_200_with_empty_slos_prometheus_error_and_an_error_log`).
    """
    bad = _first(client.instant_query(AVAILABILITY_BUDGET_QUERY))
    under = _first(client.instant_query(LATENCY_BUDGET_QUERY))
    violations = _first(client.instant_query(RESIDENCY_VIOLATIONS_30D_QUERY))
    return (
        Slo(
            name="availability",
            objective=AVAILABILITY_OBJECTIVE,
            window=SLO_WINDOW,
            observed=None if bad is None else 1 - bad,
            budget_remaining=None if bad is None else 1 - bad / AVAILABILITY_ALLOWED_BAD,
        ),
        Slo(
            name="latency",
            objective=LATENCY_THRESHOLD_SECONDS,
            window=SLO_WINDOW,
            observed=under,
            budget_remaining=None if under is None else 1 - (1 - under) / LATENCY_ALLOWED_SLOW,
        ),
        Slo(
            name="residency",
            objective=0.0,
            window=SLO_WINDOW,
            observed=violations,
            budget_remaining=None,
        ),
    )


@router.get(CONSOLE_RELIABILITY)
def reliability(request: Request, days: Days = DEFAULT_DAYS) -> ReliabilityPage:
    """SLO budgets, hourly latency and error-rate series over the window, breakers and fallbacks."""
    end = time.time()
    start = end - days * SECONDS_PER_DAY

    def read(client: PrometheusClient) -> ReliabilityPage:
        return ReliabilityPage(
            sources=Sources(postgres=SourceState.UNCONFIGURED, prometheus=SourceState.OK),
            slos=_slos(client),
            latency_p95=tuple(client.range_query(P95_QUERY, start=start, end=end)),
            error_rate=tuple(client.range_query(ERROR_RATE_QUERY, start=start, end=end)),
            circuits=tuple(
                Circuit(provider=labels.get("provider", ""), state=CIRCUIT_STATE_NAMES[int(value)])
                for labels, value in client.instant_query(CIRCUITS_QUERY)
                if value in CIRCUIT_STATE_NAMES
            ),
            fallbacks=tuple(
                FallbackCount(
                    tenant=labels.get("tenant", ""),
                    alias=labels.get("alias", ""),
                    provider=labels.get("provider", ""),
                    count=value,
                )
                for labels, value in client.instant_query(fallbacks_query(days))
            ),
        )

    prometheus, page = _prometheus(request, "reliability", read)
    if page is not None:
        return page
    return ReliabilityPage(
        sources=Sources(postgres=SourceState.UNCONFIGURED, prometheus=prometheus),
        slos=(),
        latency_p95=(),
        error_rate=(),
        circuits=(),
        fallbacks=(),
    )


@router.get(CONSOLE_INCIDENTS)
def incidents(request: Request) -> IncidentsPage:
    """The newest `ROW_LIMIT` incidents, each marked `open` while unresolved."""
    postgres, read = _postgres(request, "incidents", lambda reader: reader.incidents())
    return IncidentsPage(
        sources=Sources(postgres=postgres, prometheus=SourceState.UNCONFIGURED),
        incidents=() if read is None else read,
    )


@router.get(CONSOLE_RELEASES)
def releases(request: Request) -> ReleasesPage:
    """The newest `ROW_LIMIT` deploys, each with its linked eval run's result or `null`."""
    postgres, read = _postgres(request, "releases", lambda reader: reader.releases())
    return ReleasesPage(
        sources=Sources(postgres=postgres, prometheus=SourceState.UNCONFIGURED),
        releases=() if read is None else read,
    )
