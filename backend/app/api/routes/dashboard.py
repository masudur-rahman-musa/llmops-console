"""The console's read surface: one route per page, every number a page shows.

Four routes, one per console tab, all behind `app.api.deps.admin_key`:

    GET /console/home          today's traffic, spend, latency, the last eval, the live routing
    GET /console/cost          the daily rollup, estimated cache savings, budgets
    GET /console/quality       the eval gate, runs, categories, taxonomy, blocked releases
    GET /console/requests      redactions, refusals, blocks, tokens, hit rate, per day, newest

The console reads through here and never touches Postgres or Prometheus itself.
A working tenant key is 401 on all four, as on `/admin/*`. The sums and source
helpers are `app.services.dashboard`; the page shapes are `app.schemas.dashboard`.

**Inputs.** `days` is an integer from `MIN_DAYS` to `MAX_DAYS`, and the requests
route's `page` one from `MIN_PAGE` to `MAX_PAGE`; either is refused as 422
`VALIDATION` naming itself by the request validation handler before the route body
runs. `tenant` is pre-checked by
`app.api.deps.checked_tenant` before any query, because it is interpolated into a
PromQL label matcher.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Final

from fastapi import APIRouter, Query, Request

from app.api.deps import PolicyDep, PricesDep, SettingsDep, checked_tenant
from app.core.validation import IdentifierInvalid
from app.model_routing.policy import PolicyStore
from app.schemas.dashboard import (
    BlockedItem,
    CostPage,
    DayRequests,
    HomePage,
    ModelAlias,
    ModelCandidate,
    QualityPage,
    RedactionCount,
    RefusalCount,
    RequestsPage,
    Routing,
    Sources,
)
from app.schemas.dashboard import SourceState as SourceState
from app.services.dashboard import (
    budgets,
    first,
    gate_scores,
    gate_verdict,
    langfuse_traces_url,
    postgres,
    prometheus,
)
from app.services.prometheus_reads import (
    DAY_STEP_SECONDS,
    ERROR_RATE_QUERY,
    P95_QUERY,
    PrometheusClient,
    errors_per_day_query,
    guardrail_blocks_query,
    redactions_query,
    refusals_query,
)
from app.storage.cost_ledger import MAX_MODEL_LENGTH, MODEL_PATTERN, TRACE_ID_PATTERN
from app.storage.dashboard_reads import RequestFilters, regressed_metrics

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

MIN_PAGE: Final[int] = 1
MAX_PAGE: Final[int] = 10000
DEFAULT_PAGE: Final[int] = 1

Page = Annotated[int, Query(ge=MIN_PAGE, le=MAX_PAGE)]


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
        last_eval=None
        if read is None or read.last_eval is None
        else read.last_eval.model_copy(
            update={"passed": gate_verdict(settings.eval_baseline_path, read.last_eval_metrics)}
        ),
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
    pg, both = postgres(
        request,
        "cost",
        lambda reader: (
            reader.cost(days=days, tenant=tenant, now=now, prices=prices.table),
            reader.month_spend(tenant=tenant, now=now),
        ),
    )
    read, spend = (None, None) if both is None else both
    return CostPage(
        sources=Sources(postgres=pg, prometheus=None),
        rollup_populated=None if read is None else read.rollup_populated,
        days=() if read is None else read.days,
        savings=() if read is None else read.savings,
        savings_usd_total=None if read is None else read.savings_usd_total,
        estimated=True,
        budgets=budgets(request, tenant, spend),
        month_spend_usd=None if spend is None else sum(spend.values()),
        cost_per_request=() if read is None else read.cost_per_request,
        top_users=() if read is None else read.top_users,
    )


@router.get(CONSOLE_QUALITY)
def quality(request: Request, settings: SettingsDep, days: Days = DEFAULT_DAYS) -> QualityPage:
    """The eval gate, eval runs over the window, and what regressed behind each blocked release."""
    now = datetime.now(UTC)
    pg, read = postgres(request, "quality", lambda reader: reader.quality(days=days, now=now))
    baseline_path = settings.eval_baseline_path
    if read is None:
        return QualityPage(
            sources=Sources(postgres=pg, prometheus=None),
            gate=gate_scores(baseline_path, ()),
            runs=(),
            release_markers=(),
            categories=(),
            taxonomy=(),
            blocked=(),
        )
    return QualityPage(
        sources=Sources(postgres=pg, prometheus=None),
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
    dict[date, float],
]
"""The Prometheus half of the requests page: redactions, refusals, blocks, errors per day."""


def trace_url(project_url: str | None, trace_id: str) -> str | None:
    """`<project_url>/traces/<trace_id>`, or `None` without a project URL or a 32-hex id.

    The id becomes a URL segment, so it is matched with `re.fullmatch` against the
    ledger's own pattern: a trailing newline or a `../` gets no link. Pinned by
    tests/api/test_console_requests.py."""
    if not project_url or TRACE_ID_PATTERN.fullmatch(trace_id) is None:
        return None
    return f"{project_url.rstrip('/')}/traces/{trace_id}"


def checked_model(value: str) -> str:
    """The `?model=` filter, against the ledger's own model rule (128 characters, no `;` or `,`)."""
    if len(value) > MAX_MODEL_LENGTH or MODEL_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("model: not a provider model id", param="model")
    return value


def errors_by_day(client: PrometheusClient, days: int, now: datetime) -> dict[date, float]:
    """5xx answers per UTC day for the last `days` days ending today.

    One range query at a one-day step from the midnight after the first day to tomorrow's
    midnight; the point at midnight D+1 counts day D. A day with no point had no errors.
    Pinned by tests/api/test_console_layout.py::test_errors_per_day_query_is_pinned."""
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    points = client.range_query(
        errors_per_day_query(),
        start=(today - timedelta(days=days - 2)).timestamp(),
        end=(today + timedelta(days=1)).timestamp(),
        step=DAY_STEP_SECONDS,
    )
    return {
        datetime.fromtimestamp(ts, UTC).date() - timedelta(days=1): value for ts, value in points
    }


@router.get(CONSOLE_REQUESTS)
def requests(
    request: Request,
    settings: SettingsDep,
    days: Days = DEFAULT_DAYS,
    page: Page = DEFAULT_PAGE,
    day: date | None = None,
    tenant: str | None = None,
    model: str | None = None,
    errors_only: Annotated[bool, Query(alias="errors")] = False,
    slow: bool = False,
    cached: bool = False,
) -> RequestsPage:
    """Counts and errors per day from Prometheus; a ledger page, the hit rate, requests per
    day and tokens per model from Postgres.

    A dead or missing ledger is page 1 of 1 with no rows, no days and no tokens. A dead
    Prometheus leaves each day's `errors` null.

    `day`, `tenant`, `model`, `errors`, `slow` and `cached` narrow the request list and its
    page count only; a bad `tenant` or `model` is 422 naming itself."""
    if tenant is not None:
        checked_tenant(tenant)
    if model is not None:
        checked_model(model)
    filters = RequestFilters(
        day=day, tenant=tenant, model=model, errors=errors_only, slow=slow, cached=cached
    )
    now = datetime.now(UTC)

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
            errors_by_day(client, days, now),
        )

    prom, counts = prometheus(request, "requests", read)
    redactions, refusals, blocks, errors_by = counts if counts is not None else ((), (), None, None)
    pg, ledger_read = postgres(
        request,
        "requests",
        lambda reader: (
            reader.recent_requests(page=page, filters=filters, days=days, now=now),
            reader.cache_hit_rate(days, now),
            reader.daily_requests(days, now),
            reader.model_tokens(days, now),
        ),
    )
    ledger, hit_rate, daily, tokens = (
        ledger_read if ledger_read is not None else (None, None, (), ())
    )
    project_url = settings.langfuse_project_url
    return RequestsPage(
        sources=Sources(postgres=pg, prometheus=prom),
        redactions=redactions,
        refusals=refusals,
        guardrail_blocks=blocks,
        tokens=tokens,
        cache_hit_rate=hit_rate,
        per_day=tuple(
            DayRequests(
                day=day,
                requests=count,
                errors=None if errors_by is None else errors_by.get(day, 0.0),
            )
            for day, count in daily
        ),
        requests=tuple(
            row.model_copy(update={"trace_url": trace_url(project_url, row.trace_id)})
            for row in (() if ledger is None else ledger.rows)
        ),
        requests_page=1 if ledger is None else ledger.page,
        requests_pages=1 if ledger is None else ledger.pages,
        slow_ms=None if ledger is None else ledger.slow_ms,
        tenants=() if ledger is None else ledger.tenants,
        models=() if ledger is None else ledger.models,
        langfuse_url=langfuse_traces_url(
            project_url, filters, None if ledger is None else ledger.slow_ms
        ),
    )
