"""One registry per process: what this gateway may measure, and how a request is counted.

This module mirrors `core/tracing.py` deliberately, so that someone who has read
one file can read the other. There is a module-owned `CollectorRegistry` the way
there is a module-owned `TracerProvider`; there is one table of everything that
may be emitted the way `SpanAttributes` is the table of everything a span may
say; and one request's labels ride in the ASGI scope the way `RequestTrace` does.
The middleware that observes a finished request lives in the app module,
because a request boundary is a gateway concern and `core/` does not import from
the app layer.

**Why the registry is ours.** prometheus_client's global `REGISTRY` is write-once
per family name, so a second `create_app` in one process raises `Duplicated
timeseries` - which turns a whole test session red at once and makes a second app
impossible in a worker that hosts one. It also arrives carrying process, GC and
platform collectors nobody chose.

**Why every declared family now carries data.** Each was declared as a name
before its subsystem existed, so the work that built the subsystem
added a VALUE rather than a family - the same move made by declaring
`cache`, `cost_usd` and `fallback_count` on `SpanAttributes` without setting
them. Each was registered and empty in the meantime, never registered and zeroed:
a zero booked on every request would have drawn a confident flat line on a
dashboard for a subsystem that did not exist. Each is now filled:
`gateway_fallbacks_total`, `gateway_circuit_state`,
`gateway_cache_hits_total`, `gateway_cost_usd_total` and
`gateway_budget_ratio`. A family
declared ahead of its subsystem again belongs back in that case's unfed list.

`gateway_eval_*` is deliberately not here. The nightly job PUSHes
those gauges to a Pushgateway, so they are not this process's to declare.

**Why no client string becomes a label without passing a check first.** A
registry has no cap of its own: one distinct label tuple is one time series, in
process memory and in Prometheus's index, forever. So `alias`, `provider`,
`region` and `model` are only ever the RESOLVER's answer, `tenant` is only ever
the identifier the request context already checked against `IDENTIFIER_PATTERN`
and `MAX_IDENTIFIER_LENGTH`, `kind` is only ever a `Kind` member and `code` only
ever an `ErrorCode` member. Everything else is the empty string. That is standing
rule applied to a store with no backstop.

The session is deliberately not a label anywhere below, though it is on the
trace. A session is a conversation, so one tenant's traffic would mint a time
series per conversation and grow the index without limit - which is the same
reason it is useful on a span, where a trace is discarded on a retention policy
and a series is not.

**Why every emission is guarded.** An emission that raises must change no status,
no body and no header: a counter is how an outage is reported, never how one is
caused. Every function here that touches a family does its work inside
`_guarded`; the three that only read or build a record - `registry`,
`request_metrics` and `start_request` - touch none.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.types import Scope

from app.core.errors import ErrorCode, GatewayError

logger = logging.getLogger(__name__)

METRICS_SCOPE_KEY = "app.metrics"
"""The ASGI scope key holding this request's `RequestMetrics`.

A vendor-prefixed key of our own rather than `scope["state"]`, which is shared
with whatever else a deployment mounts and would make a name collision someone
else's bug. The same reasoning, and the same shape, as `TRACE_SCOPE_KEY`."""

COUNTED_PATH_PREFIX = "/v1/"
"""The only traffic this gateway counts.

A scrape every fifteen seconds and a kubelet probe are not traffic: counted, they
put a constant floor under every RPS panel and an unmovable denominator under
every error ratio in the first dashboard, so a real outage barely moves the
line."""

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 6.0, 10.0, 20.0, 30.0, 60.0, 120.0)
"""The histogram bounds, in seconds. prometheus_client's default set
tops out at 10s, which puts every ordinary streamed completion in `+Inf` and
makes a p95 panel unreadable exactly where it matters. `6.0` is the latency SLO's
threshold, so its error budget counts requests under exactly 6 seconds
(`prometheus.LATENCY_BUDGET_QUERY`). """

DIRECTION_IN = "in"
DIRECTION_OUT = "out"
"""The two values of the `direction` label: prompt tokens and completion tokens.
An embedding has no completion, so it records `in` alone rather than an invented
zero."""

UNSET_LABEL = ""
"""A label this request never got.

Not a `none` sentinel and not a dropped request: every `/v1/*` request is counted
exactly once so the counter reconciles with the access log, and the empty string
is what Prometheus itself means by an unset label."""

PROVIDER_FAILURE_CODES = frozenset({ErrorCode.PROVIDER_ERROR})
"""The typed codes that name a provider, and therefore move
`gateway_provider_errors_total`.

One member, and it stays one. `ALL_PROVIDERS_FAILED` is raised by the resolver
BEFORE any provider is chosen - the breaker's contribution is to make the seam
refuse more often, not to name a provider - so it has no provider to label with
and would need the empty one, which is a series nobody can act on. A residency refusal is
deliberately not here either: a 451 is a
residency decision, not a provider outage, and counting it as one would fire
the provider panel on a request no provider ever saw. It moves no failure family at all and is
counted only by
`gateway_requests_total`'s `status` label."""

_registry = CollectorRegistry()

REQUESTS = Counter(
    "gateway_requests_total",
    "Requests served, by tenant, alias, resolved provider and region, status and cache outcome.",
    ("tenant", "alias", "provider", "region", "status", "cache"),
    registry=_registry,
)

LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end request duration, measured until the last byte of the response is sent.",
    ("alias", "provider"),
    buckets=LATENCY_BUCKETS,
    registry=_registry,
)

TOKENS = Counter(
    "gateway_tokens_total",
    "Tokens billed, by tenant, provider, resolved provider model and direction.",
    ("tenant", "provider", "model", "direction"),
    registry=_registry,
)

REDACTIONS = Counter(
    "gateway_redactions_total",
    "Placeholders minted, by tenant and detector kind. One placeholder is one redacted value.",
    ("tenant", "kind"),
    registry=_registry,
)

PROVIDER_ERRORS = Counter(
    "gateway_provider_errors_total",
    "Provider call failures, by provider and typed error code.",
    ("provider", "code"),
    registry=_registry,
)

RESIDENCY_VIOLATIONS = Counter(
    "gateway_residency_violations_total",
    "Requests served from a region the tenant's policy does not allow. Zero by construction.",
    ("tenant",),
    registry=_registry,
)

CACHE_HITS = Counter(
    "gateway_cache_hits_total",
    "Responses served from a cache instead of a provider, by tenant and cache tier.",
    ("tenant", "kind"),
    registry=_registry,
)

COST_USD = Counter(
    "gateway_cost_usd_total",
    "Spend booked to the ledger, by tenant, provider and model.",
    ("tenant", "provider", "model"),
    registry=_registry,
)

FALLBACKS = Counter(
    "gateway_fallbacks_total",
    "Requests served by a candidate after an earlier one failed, by tenant, alias and "
    "the provider that actually served.",
    ("tenant", "alias", "provider"),
    registry=_registry,
)

CIRCUIT_STATE = Gauge(
    "gateway_circuit_state",
    "Breaker state per provider: 0 closed, 1 half-open, 2 open.",
    ("provider",),
    registry=_registry,
)

BUDGET_RATIO = Gauge(
    "gateway_budget_ratio",
    "Fraction of a tenant's period budget consumed.",
    ("tenant",),
    registry=_registry,
)

LEDGER_WRITES = Counter(
    "gateway_ledger_writes_total",
    "Cost-ledger rows handled by the background writer, by result.",
    ("result",),
    registry=_registry,
)
"""Three `result` values: `ok` (inserted), `error` (insert failed), `dropped`
(queue was full). The counter stays flat when the database is healthy, which
is the easy case to miss; `ok` is what a dashboard uses to confirm the ledger
is keeping up."""


def registry() -> CollectorRegistry:
    """The one registry this process exposes, holding this gateway's families only."""
    return _registry


@contextmanager
def _guarded(what: str) -> Iterator[None]:
    """Run one emission, and let nothing it raises reach the request.

    The traceback is logged rather than swallowed silently, because an emission
    that cannot render its own labels is a bug someone has to see; the request it
    belongs to has already been answered correctly by then.
    """
    try:
        yield
    except Exception:
        logger.exception("metrics emission failed: %s", what)


@dataclass
class RequestMetrics:
    """The labels one request's counter needs, filled as they become known.

    Mutable and scope-carried, which is the point. The six labels
    `gateway_requests_total` takes are known at four different points in a
    request, and none of them is the point where the request ENDS - so they are
    collected here, at exactly the call sites that already call `annotate`, and
    read once by `TracingMiddleware` after the last byte. That is also the only
    place a streamed request's real duration exists.

    Every field defaults to `UNSET_LABEL`, so a request that failed before
    resolution - or before authentication - is still counted, with the labels it
    never got left empty rather than invented.
    """

    tenant: str = UNSET_LABEL
    alias: str = UNSET_LABEL
    provider: str = UNSET_LABEL
    region: str = UNSET_LABEL
    cache: str = UNSET_LABEL


def request_metrics(scope: Scope) -> RequestMetrics | None:
    """This request's label record, or `None` when nothing started one.

    `None` is the ordinary answer on a path this gateway does not count and on an
    app built without the middleware, which is what keeps every call site a plain
    call rather than a conditional."""
    record = scope.get(METRICS_SCOPE_KEY)
    return record if isinstance(record, RequestMetrics) else None


def start_request(scope: Scope) -> RequestMetrics | None:
    """Open a label record for a counted path, and answer `None` for every other.

    This is the single place decision G is enforced: a path outside
    `COUNTED_PATH_PREFIX` gets no record, so every later emission finds nothing
    and no probe or scrape can reach a counter.
    """
    if not str(scope.get("path", "")).startswith(COUNTED_PATH_PREFIX):
        return None
    record = RequestMetrics()
    scope[METRICS_SCOPE_KEY] = record
    return record


def label_tenant(scope: Scope, tenant: str) -> None:
    """Note the authenticated tenant, once the key has resolved to one.

    Called after authentication succeeds and never before, so an unauthenticated
    caller's own `X-Tenant` header cannot mint a time series.
    """
    with _guarded("tenant label"):
        record = request_metrics(scope)
        if record is not None:
            record.tenant = tenant


def label_resolution(scope: Scope, *, alias: str, provider: str, region: str, cache: str) -> None:
    """Note what the resolver decided, and the cache outcome this request had.

    Called only AFTER `resolve()` returns, which is what bounds the cardinality of
    `alias`: an unknown alias never reaches here, so fifty distinct unknown ones
    are fifty requests against ONE time series rather than fifty permanent rows in
    Prometheus's index.
    """
    with _guarded("resolution labels"):
        record = request_metrics(scope)
        if record is not None:
            record.alias = alias
            record.provider = provider
            record.region = region
            record.cache = cache


def observe_request(scope: Scope, *, status: int, seconds: float) -> None:
    """Count one finished request and observe how long it took.

    Called once, by the middleware, after the response is fully sent. Both halves
    hold: exactly one count per request including a streamed one, and a duration
    that covers the frames rather than ending where the route returned.
    """
    with _guarded("request observation"):
        record = request_metrics(scope)
        if record is None:
            return
        REQUESTS.labels(
            tenant=record.tenant,
            alias=record.alias,
            provider=record.provider,
            region=record.region,
            status=str(status),
            cache=record.cache,
        ).inc()
        LATENCY.labels(alias=record.alias, provider=record.provider).observe(seconds)


def record_tokens(
    *,
    tenant: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int | None = None,
) -> None:
    """Book one call's tokens, `in` and - where the call has an output - `out`.

    `model` is the RESOLVED provider model, never the alias and never the name the
    provider answered with: an alias covers several provider models, so cost per
    model would be unattributable at the grain the ledger prices at, and the
    two systems could never be reconciled.

    `completion_tokens` is omitted rather than passed as zero by a caller with no
    output direction. An embedding has no completion, and a zero `out` sample is an
    invented number a dashboard would sum.
    """
    with _guarded("token counts"):
        TOKENS.labels(tenant=tenant, provider=provider, model=model, direction=DIRECTION_IN).inc(
            prompt_tokens
        )
        if completion_tokens is not None:
            TOKENS.labels(
                tenant=tenant, provider=provider, model=model, direction=DIRECTION_OUT
            ).inc(completion_tokens)


def record_cost_usd(*, tenant: str, provider: str, model: str, usd: Decimal) -> None:
    """Book one request's spend, exactly as the ledger row records it (#29).

    The number comes from the row `app.storage.record_cost` ANSWERS, never
    from a second pricing: the counter, the `X-Gateway-Cost-USD` header and the
    ledger row are one figure computed once, so a dashboard and an invoice cannot
    disagree by a rounding.

    Moved on a CACHE HIT too, by zero. A hit is a served request with a real row,
    and a counter that skipped it would still read the same total - `inc(0)`
    creates the tenant's series and adds nothing to it.

    `model` is the RESOLVED provider model for the reason `record_tokens` gives:
    the two families are reconciled against each other and against the
    ledger, and an alias covers several models.

    `Decimal` in, float out, because a Prometheus counter is a float and there is
    no way to give it anything else. The ledger keeps the exact decimal; this is
    the dashboard's copy of it, and six decimal places is far inside a float's
    exact range for any total a gateway will reach.
    """
    with _guarded("cost"):
        COST_USD.labels(tenant=tenant, provider=provider, model=model).inc(float(usd))


def record_budget_ratio(*, tenant: str, ratio: float) -> None:
    """Publish how much of one tenant's monthly budget its spend has now consumed (#27).

    Set after every SERVED request, from the total the budget counter answered
    for that request's own debit, so the gauge includes the request in hand
    rather than lagging it by one.

    A fraction rather than the raw spend, so one alert threshold and one panel
    work for every tenant whatever its cap - and NOT clamped at 1.0, because the
    gauge is what an operator reads to see how far past its cap a tenant with a
    standing override has gone.

    Nothing is published for a request whose counter could not answer. That is
    `limits/budget.py`'s fail-open decision reaching the dashboard: an unreachable
    Redis leaves the last known level standing rather than overwriting every
    tenant's series with a healthy-looking 0.0 for the length of the incident. The route decides
    that by not calling this at all, which is why there is no sentinel here.
    """
    with _guarded("budget ratio"):
        BUDGET_RATIO.labels(tenant=tenant).set(ratio)


def record_fallback(*, tenant: str, alias: str, provider: str) -> None:
    """Count one request that an earlier candidate failed to serve.

    Once per REQUEST, not once per attempt: a request that walked three
    candidates is one fallback, so the counter answers "how many requests did not
    get their first choice" rather than "how many provider calls were made". A request
    that needed none moves nothing at all, which is what keeps the panel from
    showing a permanent 100%.

    `provider` is the one that SERVED, not the one that failed, which is a
    deliberate deviation from a `(from, to)` pair of labels. What this family is for is attributing
    the traffic that actually
    left, per tenant, which is what the console and `limits/budget.py` read.

    Where the FAILED provider shows up, precisely, because the obvious answer is
    wrong: on the trace, as its own `provider:{name}` stage span on a request
    whose `provider` attribute names a different one, and on
    `gateway_circuit_state`, which opens for it after five failures in the window.
    NOT on `gateway_provider_errors_total` - that is moved by `record_failure`
    from the error handler, and a fallback that succeeds returns 200 and reaches
    no handler at all. So a provider shedding 100% of its traffic behind a working
    fallback moves this counter and the gauge, and leaves the provider-error
    counter flat.
    """
    with _guarded("fallback counts"):
        FALLBACKS.labels(tenant=tenant, alias=alias, provider=provider).inc()


def record_circuit_state(*, provider: str, state: int) -> None:
    """Publish one provider's breaker state, 0 closed, 1 half-open, 2 open.

    Called for every provider a route CONSULTS, not only for one that opens. A
    gauge that appeared when a breaker opened would leave an alert on
    `gateway_circuit_state != 0` blind to every provider until the moment it
    broke, and a dashboard with no series to draw.

    The values are `CircuitState`'s own, passed as the int the enum already is:
    the names live in `providers/breaker.py`, and importing them here would point
    `core/` at a provider module it otherwise knows nothing about.
    """
    with _guarded("circuit state"):
        CIRCUIT_STATE.labels(provider=provider).set(state)


def record_redactions(*, tenant: str, counts: Mapping[str, int]) -> None:
    """Count the placeholders one redaction pass minted, per detector kind.

    Placeholders, not detections: the same address in three turns is one redacted
    value under one placeholder (the redaction numbering), so counting detections
    would make a tenant's redaction rate a function of how often someone repeats
    themselves.
    """
    with _guarded("redaction counts"):
        for kind, count in counts.items():
            REDACTIONS.labels(tenant=tenant, kind=kind).inc(count)


EXACT_CACHE_KIND: Final[str] = "exact"
SEMANTIC_CACHE_KIND: Final[str] = "semantic"
"""The two `kind` values a cache hit is counted under, one per tier.

Both tiers are wired into the chat route, and they are a second VALUE on one
family rather than a second family, which is the whole reason this label exists:
a dashboard sums them for the hit rate and splits them for the two consistency
regimes.

The semantic value is the same word the header and the `cache` label answer
(`CACHE_STATUS_SEMANTIC` in `app.services.chat_pipeline`), so a hit rate per
tier is one query rather than a join."""


def record_cache_hit(*, tenant: str, kind: str) -> None:
    """Count one answer served from a cache instead of from a provider (#24).

    Only a HIT moves this counter. A miss is already visible as
    `gateway_requests_total`'s `cache="miss"` label, and a second family counting
    both would report a hit rate of one on every dashboard that divided them.

    Nothing here books tokens or cost: `gateway_tokens_total`
    is what the providers billed, and a hit bought nothing.
    """
    with _guarded("cache hit"):
        CACHE_HITS.labels(tenant=tenant, kind=kind).inc()


def record_failure(scope: Scope, error: GatewayError) -> None:
    """Move whichever failure counter this typed error belongs on, if any.

    One failure has a panel of its own, because one failure has an operator
    response of its own: a provider outage is answered by looking at the provider,
    so it is labelled with the one that failed and with the typed code. Every other
    typed failure is already counted by `gateway_requests_total`'s `status` label
    and gets no second family.

    A 451 is NOT among them, and used to be. A residency
    refusal is the resolver working: it is the guarantee being kept, not a
    violation of it, and the alert on `gateway_residency_violations_total` fires at
    greater than zero - so counting 451s there paged someone every time an EU
    tenant asked for `agent-strong`. That counter is now moved only by an
    `allowed=false` row in the residency log, from `record_residency_violation`
    below, which is what makes it zero by construction and alertable at all.
    """
    with _guarded("failure counters"):
        record = request_metrics(scope)
        if record is None or error.code is None:
            return
        if error.code in PROVIDER_FAILURE_CODES and record.provider != UNSET_LABEL:
            PROVIDER_ERRORS.labels(provider=record.provider, code=error.code.value).inc()


def record_ledger_write(*, result: str) -> None:
    """Count one ledger write by its outcome: `ok`, `error` or `dropped`.

    Called by the background worker in `storage/cost_ledger.py` after each
    attempt, so the counter moves on the worker thread, not on the route.
    Guarded the same way every other emission is, so a labelling bug cannot
    fail a background thread.
    """
    with _guarded("ledger write"):
        LEDGER_WRITES.labels(result=result).inc()


def record_residency_violation(*, tenant: str) -> None:
    """Book one residency violation: a request SERVED from a region policy forbids.

    The resolver cannot produce one - it filters the candidate list by region
    before anything is dialled, and the residency property test
    is the proof - so this counter is expected to stay at zero forever, which is
    exactly what makes "alert above zero" a usable alert rather than
    a threshold someone has to tune. It moves only from
    `storage/residency_log.py::record_residency`, on a row whose `allowed` came back
    false; if it ever moves, the log row beside it names the tenant, the alias,
    the provider and the region.

    Moved BEFORE the write rather than after it, so a violation is still visible
    when the database is the thing that has broken.
    """
    with _guarded("residency violations"):
        RESIDENCY_VIOLATIONS.labels(tenant=tenant).inc()
