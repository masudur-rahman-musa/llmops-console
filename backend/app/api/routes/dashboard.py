"""The console's read surface: one route per page, every number a page shows.

Seven routes, all behind `app.api.deps.admin_key`:

    GET /console/home          today's traffic, spend, latency and the last eval
    GET /console/cost          the daily rollup, estimated cache savings, budgets
    GET /console/quality       eval runs, categories, taxonomy, blocked releases
    GET /console/residency     where each tenant's data went, and redactions
    GET /console/reliability   latency and error-rate series, breakers, fallbacks
    GET /console/incidents     the timeline
    GET /console/releases      deploys, each with its eval result

The console reads through here and never touches Postgres or Prometheus itself.
A working tenant key is 401 on all seven, as on `/admin/*`. The sums and source
helpers are `app.services.dashboard`; the page shapes are `app.schemas.dashboard`.

**Inputs.** `days` is an integer from `MIN_DAYS` to `MAX_DAYS`, refused as 422
`VALIDATION` naming `days` by the request validation handler before the route body
runs. `tenant` is pre-checked by
`app.api.deps.checked_tenant` before any query, because it is interpolated into a
PromQL label matcher. The reliability page's
SLO error budgets read a fixed 30-day window and ignore `days`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Annotated, Final

from fastapi import APIRouter, Query, Request

from app.api.deps import PricesDep, checked_tenant
from app.schemas.dashboard import (
    BlockedItem,
    Circuit,
    CostPage,
    FallbackCount,
    HomePage,
    IncidentsPage,
    QualityPage,
    RedactionCount,
    ReleasesPage,
    ReliabilityPage,
    ResidencyPage,
    Sources,
)
from app.schemas.dashboard import SourceState as SourceState
from app.services.dashboard import budgets, first, postgres, prometheus, slos
from app.services.prometheus_reads import (
    CIRCUITS_QUERY,
    ERROR_RATE_QUERY,
    P95_QUERY,
    PrometheusClient,
    fallbacks_query,
    redactions_query,
)
from app.storage.dashboard_reads import regressed_metrics

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
"""All seven, spelled as literals and kept off every public router."""

DAYS_PARAM: Final[str] = "days"
MIN_DAYS: Final[int] = 1
MAX_DAYS: Final[int] = 90
DEFAULT_DAYS: Final[int] = 30

SECONDS_PER_DAY: Final[int] = 86_400

CIRCUIT_STATE_NAMES: Final[dict[int, str]] = {0: "closed", 1: "half_open", 2: "open"}
"""`gateway_circuit_state`'s values as the page shows them. Any other value is
skipped rather than guessed at."""

Days = Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)]


router = APIRouter()


@router.get(CONSOLE_HOME)
def home(request: Request) -> HomePage:
    """Today from the live ledger, p95 and error rate from Prometheus."""
    now = datetime.now(UTC)
    pg, read = postgres(request, "home", lambda reader: reader.home(now=now))
    prom, rates = prometheus(
        request,
        "home",
        lambda client: (
            first(client.instant_query(P95_QUERY)),
            first(client.instant_query(ERROR_RATE_QUERY)),
        ),
    )
    p95, error_rate = rates if rates is not None else (None, None)
    return HomePage(
        sources=Sources(postgres=pg, prometheus=prom),
        requests_today=None if read is None else read.requests_today,
        cost_usd_today=None if read is None else read.cost_usd_today,
        p95_seconds=p95,
        error_rate=error_rate,
        residency_violations_today=None if read is None else read.residency_violations_today,
        last_eval=None if read is None else read.last_eval,
        open_incidents=None if read is None else read.open_incidents,
    )


@router.get(CONSOLE_COST)
def cost(
    request: Request, prices: PricesDep, days: Days = DEFAULT_DAYS, tenant: str | None = None
) -> CostPage:
    """The rollup's days, estimated cache savings, budgets and cost per request."""
    if tenant is not None:
        checked_tenant(tenant)
    now = datetime.now(UTC)
    pg, read = postgres(
        request,
        "cost",
        lambda reader: reader.cost(days=days, tenant=tenant, now=now, prices=prices.table),
    )
    return CostPage(
        sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
        rollup_populated=None if read is None else read.rollup_populated,
        days=() if read is None else read.days,
        savings=() if read is None else read.savings,
        savings_usd_total=None if read is None else read.savings_usd_total,
        estimated=True,
        budgets=budgets(request, tenant),
        cost_per_request=() if read is None else read.cost_per_request,
    )


@router.get(CONSOLE_QUALITY)
def quality(request: Request, days: Days = DEFAULT_DAYS) -> QualityPage:
    """Eval runs over the window, and what regressed behind each blocked release."""
    now = datetime.now(UTC)
    pg, read = postgres(request, "quality", lambda reader: reader.quality(days=days, now=now))
    if read is None:
        return QualityPage(
            sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
            runs=(),
            release_markers=(),
            categories=(),
            taxonomy=(),
            blocked=(),
        )
    return QualityPage(
        sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
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
        checked_tenant(tenant)
    now = datetime.now(UTC)
    pg, read = postgres(
        request, "residency", lambda reader: reader.residency(days=days, tenant=tenant, now=now)
    )
    prom, redactions = prometheus(
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
        sources=Sources(postgres=pg, prometheus=prom),
        by_region=() if read is None else read.by_region,
        violations=None if read is None else read.violations,
        redactions=redactions or (),
    )


@router.get(CONSOLE_RELIABILITY)
def reliability(request: Request, days: Days = DEFAULT_DAYS) -> ReliabilityPage:
    """SLO budgets, hourly latency and error-rate series over the window, breakers and fallbacks."""
    end = time.time()
    start = end - days * SECONDS_PER_DAY

    def read(client: PrometheusClient) -> ReliabilityPage:
        return ReliabilityPage(
            sources=Sources(postgres=SourceState.UNCONFIGURED, prometheus=SourceState.OK),
            slos=slos(client),
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

    prom, page = prometheus(request, "reliability", read)
    if page is not None:
        return page
    return ReliabilityPage(
        sources=Sources(postgres=SourceState.UNCONFIGURED, prometheus=prom),
        slos=(),
        latency_p95=(),
        error_rate=(),
        circuits=(),
        fallbacks=(),
    )


@router.get(CONSOLE_INCIDENTS)
def incidents(request: Request) -> IncidentsPage:
    """The newest `ROW_LIMIT` incidents, each marked `open` while unresolved."""
    pg, read = postgres(request, "incidents", lambda reader: reader.incidents())
    return IncidentsPage(
        sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
        incidents=() if read is None else read,
    )


@router.get(CONSOLE_RELEASES)
def releases(request: Request) -> ReleasesPage:
    """The newest `ROW_LIMIT` deploys, each with its linked eval run's result or `null`."""
    pg, read = postgres(request, "releases", lambda reader: reader.releases())
    return ReleasesPage(
        sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
        releases=() if read is None else read,
    )
