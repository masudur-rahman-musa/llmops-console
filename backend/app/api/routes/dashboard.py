"""The console's read surface: one route per page, every number a page shows.

Four routes, one per console tab, all behind `app.api.deps.admin_key`:

    GET /console/home          today's traffic, spend, latency, the last eval, the live routing
    GET /console/cost          cost per day from the ledger, estimated cache savings, budgets
    GET /console/quality       the eval gate, runs, categories, taxonomy, blocked releases,
                               the plays that need a look
    GET /console/requests      redactions, refusals, blocks, tokens, hit rate, per day, newest

The console reads through here and never touches Postgres or Prometheus itself.
A working tenant key is 401 on all four, as on `/admin/*`. The sums and source
helpers are `app.services.dashboard`; the page shapes are `app.schemas.dashboard`.

**Inputs.** `days` is an integer from `MIN_DAYS` to `MAX_DAYS`, and the requests
route's `page` one from `MIN_PAGE` to `MAX_PAGE`; either is refused as 422
`VALIDATION` naming itself by the request validation handler before the route body
runs. `tenant` is pre-checked by
`app.api.deps.checked_tenant` before any query, because it is interpolated into a
PromQL label matcher. The cost route's `from` and `to` are dates checked by
`checked_cost_range`: `from` after `to` is 422 naming `from`, a span over `MAX_DAYS`
days (both ends kept) is 422 naming `to`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
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
    ModelTokens,
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
    ERROR_RATE_QUERY,
    P95_QUERY,
    PrometheusClient,
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
FROM_PARAM: Final[str] = "from"
TO_PARAM: Final[str] = "to"
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


def prompt_cache_share(tokens: tuple[ModelTokens, ...]) -> float | None:
    """Cache reads over all input (plain, cache reads and cache writes); null with no input,
    since no traffic is not a 0% share. A test pins this."""
    reads = sum(row.cache_read for row in tokens)
    total = sum(row.input + row.cache_read + row.cache_write for row in tokens)
    return None if total == 0 else reads / total


def checked_cost_range(start: date | None, end: date | None) -> None:
    """Refuse an upside-down range (naming `from`) or one over `MAX_DAYS` days (naming `to`).

    Both ends count, so 2026-07-06 to 2026-10-03 is exactly 90 days and passes. A test pins this."""
    if start is None or end is None:
        return
    if start > end:
        raise IdentifierInvalid("from: after to", param=FROM_PARAM)
    if (end - start).days + 1 > MAX_DAYS:
        raise IdentifierInvalid(f"to: more than {MAX_DAYS} days after from", param=TO_PARAM)


@router.get(CONSOLE_COST)
def cost(
    request: Request,
    prices: PricesDep,
    days: Days = DEFAULT_DAYS,
    tenant: str | None = None,
    start: Annotated[date | None, Query(alias=FROM_PARAM)] = None,
    end: Annotated[date | None, Query(alias=TO_PARAM)] = None,
) -> CostPage:
    """Cost per day from the ledger, estimated cache savings, budgets and cost per request.

    `from` and `to` pick the days of the Daily cost table (UTC, both kept); without them it
    is the last `days` days through today. The rest of the body keeps the `days` window."""
    if tenant is not None:
        checked_tenant(tenant)
    checked_cost_range(start, end)
    now = datetime.now(UTC)
    pg, both = postgres(
        request,
        "cost",
        lambda reader: (
            reader.cost(
                days=days, tenant=tenant, now=now, prices=prices.table, start=start, end=end
            ),
            reader.month_spend(tenant=tenant, now=now),
        ),
    )
    read, spend = (None, None) if both is None else both
    return CostPage(
        sources=Sources(postgres=pg, prometheus=None),
        days=() if read is None else read.days,
        savings=() if read is None else read.savings,
        savings_usd_total=None if read is None else read.savings_usd_total,
        estimated=True,
        budgets=budgets(request, tenant, spend),
        month_spend_usd=None if spend is None else sum(spend.values()),
        prompt_cache_saved_usd=None if read is None else read.prompt_cache_saved_usd,
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
        needs_a_look=read.needs_a_look,
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
]
"""The Prometheus half of the requests page: redactions, refusals, blocks."""


def trace_url(project_url: str | None, trace_id: str) -> str | None:
    """`<project_url>/traces/<trace_id>`, or `None` without a project URL or a 32-hex id.

    The id becomes a URL segment, so it is matched with `re.fullmatch` against the
    ledger's own pattern: a trailing newline or a `../` gets no link. Tests pin this."""
    if not project_url or TRACE_ID_PATTERN.fullmatch(trace_id) is None:
        return None
    return f"{project_url.rstrip('/')}/traces/{trace_id}"


def checked_model(value: str) -> str:
    """The `?model=` filter, against the ledger's own model rule (128 characters, no `;` or `,`)."""
    if len(value) > MAX_MODEL_LENGTH or MODEL_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("model: not a provider model id", param="model")
    return value


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
    """Redactions, refusals and blocks from Prometheus; a ledger page, the hit rate, answered
    and failed requests per day and tokens per model from Postgres.

    A dead or missing ledger is page 1 of 1 with no rows, no days and no tokens. A dead
    Prometheus leaves the per-day counts alone: they come from the ledger.

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
            # A reason with no refusals in the window is left out, so the table can say
            # "None in this window.".
            tuple(
                RefusalCount(reason=labels.get("reason", ""), count=value)
                for labels, value in client.instant_query(refusals_query(days))
                if value != 0
            ),
            first(client.instant_query(guardrail_blocks_query(days))),
        )

    prom, counts = prometheus(request, "requests", read)
    redactions, refusals, blocks = counts if counts is not None else ((), (), None)
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
        prompt_cache_share=None if ledger_read is None else prompt_cache_share(tokens),
        per_day=tuple(
            DayRequests(
                day=day,
                requests=answered,
                errors=failed,
            )
            for day, answered, failed in daily
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
