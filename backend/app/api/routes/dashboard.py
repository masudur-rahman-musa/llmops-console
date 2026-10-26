"""The console's read surface: one route per page, every number a page shows.

Four routes, one per console tab, all behind `app.api.deps.admin_key`:

    GET /console/home          today's traffic, spend, latency, the last eval, the live routing
    GET /console/cost          the daily rollup, estimated cache savings, budgets
    GET /console/quality       the eval gate, runs, categories, taxonomy, blocked releases
    GET /console/requests      redactions, refusals, blocks, tokens, hit rate, newest requests

The console reads through here and never touches Postgres or Prometheus itself.
A working tenant key is 401 on all four, as on `/admin/*`. The sums and source
helpers are `app.services.dashboard`; the page shapes are `app.schemas.dashboard`.

**Inputs.** `days` is an integer from `MIN_DAYS` to `MAX_DAYS`, refused as 422
`VALIDATION` naming `days` by the request validation handler before the route body
runs. `tenant` is pre-checked by
`app.api.deps.checked_tenant` before any query, because it is interpolated into a
PromQL label matcher.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Final

from fastapi import APIRouter, Query, Request

from app.api.deps import PolicyDep, PricesDep, SettingsDep, checked_tenant
from app.model_routing.policy import PolicyStore
from app.schemas.dashboard import (
    BlockedItem,
    CostPage,
    HomePage,
    ModelAlias,
    ModelCandidate,
    ModelTokens,
    QualityPage,
    RedactionCount,
    RefusalCount,
    RequestsPage,
    Routing,
    Sources,
)
from app.schemas.dashboard import SourceState as SourceState
from app.services.dashboard import budgets, first, gate_scores, postgres, prometheus
from app.services.prometheus_reads import (
    ERROR_RATE_QUERY,
    P95_QUERY,
    PrometheusClient,
    cache_hit_rate_query,
    guardrail_blocks_query,
    redactions_query,
    refusals_query,
    tokens_query,
)
from app.storage.cost_ledger import TRACE_ID_PATTERN
from app.storage.dashboard_reads import regressed_metrics

CONSOLE_HOME: Final[str] = "/console/home"
CONSOLE_COST: Final[str] = "/console/cost"
CONSOLE_QUALITY: Final[str] = "/console/quality"
CONSOLE_REQUESTS: Final[str] = "/console/requests"

CONSOLE_PATHS: Final[tuple[str, ...]] = (
    CONSOLE_HOME,
    CONSOLE_COST,
    CONSOLE_QUALITY,
    CONSOLE_REQUESTS,
)
"""All four, spelled as literals and kept off every public router."""

DAYS_PARAM: Final[str] = "days"
MIN_DAYS: Final[int] = 1
MAX_DAYS: Final[int] = 90
DEFAULT_DAYS: Final[int] = 30

Days = Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)]


router = APIRouter()


def routing(store: PolicyStore) -> Routing:
    """Each alias in file order with its providers in fallback order, from the live policy."""
    return Routing(
        loaded_at=store.loaded_at,
        aliases=tuple(
            ModelAlias(
                name=name,
                purpose=alias.purpose,
                candidates=tuple(
                    ModelCandidate(provider=c.provider, model=c.model) for c in alias.candidates
                ),
            )
            for name, alias in store.policy.aliases.items()
        ),
    )


@router.get(CONSOLE_HOME)
def home(request: Request, store: PolicyDep, settings: SettingsDep) -> HomePage:
    """Today from the live ledger, p95 and error rate from Prometheus, routing from the policy."""
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
        last_eval=None if read is None else read.last_eval,
        routing=routing(store),
        grafana_url=settings.grafana_url,
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
def quality(request: Request, settings: SettingsDep, days: Days = DEFAULT_DAYS) -> QualityPage:
    """The eval gate, eval runs over the window, and what regressed behind each blocked release."""
    now = datetime.now(UTC)
    pg, read = postgres(request, "quality", lambda reader: reader.quality(days=days, now=now))
    baseline_path = settings.eval_baseline_path
    if read is None:
        return QualityPage(
            sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
            gate=gate_scores(baseline_path, ()),
            runs=(),
            release_markers=(),
            categories=(),
            taxonomy=(),
            blocked=(),
        )
    return QualityPage(
        sources=Sources(postgres=pg, prometheus=SourceState.UNCONFIGURED),
        gate=gate_scores(baseline_path, read.runs),
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


RequestCounts = tuple[
    tuple[RedactionCount, ...],
    tuple[RefusalCount, ...],
    float | None,
    tuple[ModelTokens, ...],
    float | None,
]
"""The Prometheus half of the requests page: redactions, refusals, blocks, tokens, hit rate."""


def trace_url(project_url: str | None, trace_id: str) -> str | None:
    """`<project_url>/traces/<trace_id>`, or `None` without a project URL or a 32-hex id.

    The id becomes a URL segment, so it is matched with `re.fullmatch` against the
    ledger's own pattern: a trailing newline or a `../` gets no link. Pinned by
    tests/api/test_console_requests.py."""
    if not project_url or TRACE_ID_PATTERN.fullmatch(trace_id) is None:
        return None
    return f"{project_url.rstrip('/')}/traces/{trace_id}"


def _tokens(samples: list[tuple[dict[str, str], float]]) -> tuple[ModelTokens, ...]:
    """One row per model, `in` and `out` folded together; a missing direction is 0."""
    by_model: dict[str, dict[str, float]] = {}
    for labels, value in samples:
        directions = by_model.setdefault(labels.get("model", ""), {"in": 0.0, "out": 0.0})
        direction = labels.get("direction", "")
        if direction in directions:
            directions[direction] += value
    return tuple(
        ModelTokens(model=model, input=d["in"], output=d["out"])
        for model, d in sorted(by_model.items())
    )


@router.get(CONSOLE_REQUESTS)
def requests(request: Request, settings: SettingsDep, days: Days = DEFAULT_DAYS) -> RequestsPage:
    """Counts over the window from Prometheus, the newest ledger rows from Postgres."""

    def read(client: PrometheusClient) -> RequestCounts:
        return (
            tuple(
                RedactionCount(kind=labels.get("kind", ""), count=value)
                for labels, value in client.instant_query(redactions_query(days))
            ),
            tuple(
                RefusalCount(reason=labels.get("reason", ""), count=value)
                for labels, value in client.instant_query(refusals_query(days))
            ),
            first(client.instant_query(guardrail_blocks_query(days))),
            _tokens(client.instant_query(tokens_query(days))),
            first(client.instant_query(cache_hit_rate_query(days))),
        )

    prom, counts = prometheus(request, "requests", read)
    empty: RequestCounts = ((), (), None, (), None)
    redactions, refusals, blocks, tokens, hit_rate = counts if counts is not None else empty
    pg, rows = postgres(request, "requests", lambda reader: reader.recent_requests())
    project_url = settings.langfuse_project_url
    return RequestsPage(
        sources=Sources(postgres=pg, prometheus=prom),
        redactions=redactions,
        refusals=refusals,
        guardrail_blocks=blocks,
        tokens=tokens,
        cache_hit_rate=hit_rate,
        requests=tuple(
            row.model_copy(update={"trace_url": trace_url(project_url, row.trace_id)})
            for row in rows or ()
        ),
    )
