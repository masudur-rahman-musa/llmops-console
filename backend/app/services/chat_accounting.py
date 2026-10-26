"""Clients dataclass, ChatResult, and all accounting functions for the chat pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from app.cache import ResponseCache, SemanticCache
from app.core.config import Settings
from app.core.context import RequestContext
from app.core.errors import BudgetExceeded, RateLimited
from app.core.metrics import (
    record_budget_ratio,
    record_cost_usd,
)
from app.limits.budget import (
    BudgetDecision,
    BudgetStore,
    budget_ratio,
    current_month,
    decide,
    to_micro_usd,
    verify_override,
    warn_budget,
)
from app.limits.cost import COST_QUANTUM
from app.limits.prices import PriceStore
from app.limits.ratelimit import SECONDS_PER_MINUTE, RateDimension, RateLimitStore
from app.model_routing.policy import Policy
from app.model_routing.resolver import Resolved
from app.providers.base import Adapter
from app.providers.breaker import CircuitBreaker
from app.redaction import DirectoryStore, RedactionMapStore
from app.schemas.chat import ChatCompletion
from app.schemas.provider import Usage
from app.services.ledger import record_cost
from app.storage import (
    CostLedgerRow,
    CostLedgerWriter,
)


@dataclass(frozen=True)
class Clients:
    """All the stores and clients the chat and embeddings flows read.

    Passed in place of Starlette `State` so the service functions take plain
    typed inputs and import nothing from the API layer.  `deps.get_clients`
    builds one per request from `request.app.state`.
    """

    policy: Policy
    rate_limit_store: RateLimitStore
    budget_store: BudgetStore
    settings: Settings
    cost_ledger: CostLedgerWriter
    prices: PriceStore
    adapters: Mapping[str, Adapter]
    breaker: CircuitBreaker
    response_cache: ResponseCache
    semantic_cache: SemanticCache
    redaction_map_store: RedactionMapStore
    directory_store: DirectoryStore
    clock: Callable[[], float] = time.monotonic
    """Seconds, monotonic. Read when a chat request starts and when its row is booked, so
    `latency_ms` is the difference; tests/api/test_request_status_rows.py hands in a fake."""


@dataclass
class ChatResult:
    """What `complete_chat` hands back for a non-streaming request.

    `body` is the response the route returns; `headers` are all seven
    `X-Gateway-*` headers including `X-Gateway-Cost-USD`.  The route does
    `response.headers.update(result.headers)` and returns `result.body`.
    """

    body: ChatCompletion
    headers: dict[str, str]


CHECK_COST = 1
"""What one request costs each bucket at the check.

One from the REQUEST bucket on purpose: unlike its tokens, a request is
countable before it is served, so the check and the debit are one atomic take and
a tenant's published `rpm` is exactly the number of requests it gets. One from the
TOKEN bucket is a probe, and it is what makes "this tenant has spent its minute's
tokens" expressible at all - the real usage is not known until the provider has
answered, so the check asks whether a single token is left and the
debit follows from `book_cost`. One token in a `tpm` of 120,000 is what that probe
costs.
"""


def check_rate_limit(clients: Clients, context: RequestContext) -> None:
    """Refuse this request if the tenant has spent its minute's requests or tokens (#28).

    Called at the top of BOTH routes and immediately BEFORE `check_budget`, after
    `Context` has authenticated the tenant. The position carries two criteria at
    once. A refusal must cost no provider call, no ledger row and no cache write,
    and a streamed request must be refused before the 200 and the first frames
    are on the wire, or the client gets an error inside a stream instead of a
    typed 429 with a `Retry-After` no started response could carry. And
    being before the budget check is criterion 6: both refusals are 429s, so a
    tenant over both is told to slow down rather than to top up.

    A tenant the POLICY does not carry has no ceiling and is not this function's
    to refuse. Its limit is read with `.get` and not `policy.tenants[tenant]`,
    which would raise `KeyError` and turn the resolver's typed 403 into an
    untyped 500; inventing a ceiling of zero for it would be worse, because the
    429 would then hide the tenant refusal that is the correct answer.

    A store that cannot answer serves the request (see
    `limits/ratelimit.py`): a `None` from either take refuses nobody, and the
    store has already logged at ERROR by the time it is read here.

    BOTH buckets are taken from, and the request bucket first, so a tenant over
    both spends one token of each rather than being refused on a bucket it never
    reached - which keeps the two ceilings independent of the order they are
    checked in.
    """
    policy: Policy = clients.policy
    tenant_policy = policy.tenants.get(context.tenant)
    if tenant_policy is None:
        return
    limits: RateLimitStore = clients.rate_limit_store
    limit = tenant_policy.rate_limit
    requests = limits.take(
        context.tenant,
        RateDimension.REQUESTS,
        cost=CHECK_COST,
        capacity=limit.rpm,
        rate=limit.rpm / SECONDS_PER_MINUTE,
    )
    tokens = limits.take(
        context.tenant,
        RateDimension.TOKENS,
        cost=CHECK_COST,
        capacity=limit.tpm,
        rate=limit.tpm / SECONDS_PER_MINUTE,
    )
    for taken, ceiling, unit in (
        (requests, limit.rpm, "requests"),
        (tokens, limit.tpm, "tokens"),
    ):
        if taken is None or taken.allowed:
            continue
        raise RateLimited(
            f"tenant {context.tenant!r} has reached its rate limit of {ceiling} {unit} per minute",
            retry_after_seconds=taken.retry_after_seconds,
        )


def debit_tokens(clients: Clients, context: RequestContext, *, row: CostLedgerRow) -> None:
    """Take one served request's tokens out of its tenant's minute.

    Called from `book_cost` and from nowhere else, beside `debit_budget`, which
    is what makes criterion 2 structural rather than a list of call sites to keep
    in step: every path that books a ledger row debits the bucket, by the tokens
    of the row it just booked, because there is one place that does both.

    A CACHED ROW DEBITS ZERO, and this is the one branch here. A cached row keeps
    the replayed answer's token counts on purpose, so reading `input_tokens + output_tokens`
    off it would charge a tenant's `tpm` again for every hit and make the cache
    actively expensive in the one currency it saves. Zero is still a DEBIT: the
    call happens, which is what separates a path that cost nothing from a path
    the wiring missed, and no bucket level can tell those apart.

    The answer is ignored on purpose: the request that crosses `tpm`
    completes and the NEXT one is refused, because a token count is not known
    until the provider has answered and reserving an estimate up front would
    refuse requests on a guess. Like everything else `book_cost` calls, this
    cannot change the answer - the store swallows every `redis.RedisError`, and
    the tenant was validated by `Context` before the key was built.
    """
    limits: RateLimitStore = clients.rate_limit_store
    policy: Policy = clients.policy
    tenant_policy = policy.tenants.get(context.tenant)
    if tenant_policy is None:
        return
    limit = tenant_policy.rate_limit
    spent = 0 if row.cached else row.input_tokens + row.output_tokens
    limits.take(
        context.tenant,
        RateDimension.TOKENS,
        cost=spent,
        capacity=limit.tpm,
        rate=limit.tpm / SECONDS_PER_MINUTE,
    )


def check_budget(clients: Clients, context: RequestContext, *, override: str | None) -> None:
    """Refuse this request if the tenant has already spent its month, and warn it near.

    Called at the top of BOTH routes, after `Context` has authenticated the
    tenant and before anything is resolved, redacted, embedded or dialled. The position is the
    whole point: a refusal must cost no
    provider call, no ledger row and no cache write, and a streamed
    request must be refused before the 200 and the first frames are on the wire,
    or the client gets an error inside a stream instead of a typed 429. A
    middleware could not do this job at all: it runs before the `Context`
    dependency, so it has no authenticated tenant to look a cap up for.

    A tenant the POLICY does not carry has no cap and is not this function's to
    refuse. Its cap is read with `.get` and not `policy.tenants[tenant]`, which
    would raise `KeyError` and turn the resolver's typed 403 into an untyped 500;
    inventing a cap of zero for it would be worse still, because the 429 would
    then hide the tenant refusal that is the correct answer.

    A counter that cannot answer serves the request (see
    `limits/budget.py`): `spend` answering `None` returns here having warned
    nobody and refused nobody.

    The warning covers the whole band at or above 80%, an overridden request
    included, because that is what criterion 2 says - it is emitted before the
    override is even considered, so a tenant running on a standing override is
    still visible in the log every request.
    """
    policy: Policy = clients.policy
    tenant_policy = policy.tenants.get(context.tenant)
    if tenant_policy is None:
        return
    budget_store: BudgetStore = clients.budget_store
    month = current_month()
    spent = budget_store.spend(context.tenant, month)
    if spent is None:
        return
    cap = to_micro_usd(tenant_policy.budget_usd_month)
    decision = decide(spend_micro_usd=spent, cap_micro_usd=cap)
    if decision is BudgetDecision.SERVE:
        return
    warn_budget(tenant=context.tenant, spend_micro_usd=spent, cap_micro_usd=cap)
    if decision is not BudgetDecision.REJECT:
        return
    settings: Settings = clients.settings
    if verify_override(
        override, tenant=context.tenant, month=month, secret=settings.budget_override_secret
    ):
        return
    raise BudgetExceeded(
        f"tenant {context.tenant!r} has reached its monthly budget of "
        f"{tenant_policy.budget_usd_month} USD"
    )


def debit_budget(clients: Clients, context: RequestContext, *, usd: Decimal) -> None:
    """Add one served request's cost to its tenant's month, and republish the gauge.

    Called from `book_cost` and from nowhere else, which is what makes criterion
    5 structural rather than a list to keep in step: every path that books a
    ledger row debits the counter, by the cost of the row it just booked, because
    there is one place that does both.

    ZERO IS A DEBIT. A cache hit books a row of `0.000000` and debits it, which
    both keeps the gauge fresh on a hit and makes a missing call site visible: a
    total cannot tell a path that added nothing from one that was never wired,
    and the two cache tiers and `/v1/embeddings` against `local` all cost zero.

    The gauge is set from what `debit` ANSWERED, not from a second read, so it
    includes the request in hand rather than lagging it by one. Nothing is
    published when the counter could not answer or when the tenant has no cap:
    the last known level stands rather than being overwritten with a confident
    0.0 for the length of a Redis incident.

    Like everything else `book_cost` calls, this cannot change the answer. The
    store swallows every `redis.RedisError`, `record_budget_ratio` is guarded like
    every other emission, and the tenant was validated by `Context` before the
    key was built.
    """
    policy: Policy = clients.policy
    budget_store: BudgetStore = clients.budget_store
    total = budget_store.debit(context.tenant, current_month(), to_micro_usd(usd))
    tenant_policy = policy.tenants.get(context.tenant)
    if total is None or tenant_policy is None:
        return
    record_budget_ratio(
        tenant=context.tenant,
        ratio=budget_ratio(
            spend_micro_usd=total, cap_micro_usd=to_micro_usd(tenant_policy.budget_usd_month)
        ),
    )


def book_cost(
    clients: Clients,
    context: RequestContext,
    *,
    alias: str,
    chosen: Resolved,
    usage: Usage | None,
    cached: bool,
    prompt_chars: int = 0,
    completion_chars: int = 0,
    latency_ms: int | None = None,
) -> str:
    """Price one served request, file its row, count it, debit its budget, answer the header.

    The four numbers a request reports its cost through - the row in
    `cost_ledger`, the `gateway_cost_usd_total` counter, the tenant's monthly
    budget counter and `X-Gateway-Cost-USD` - come from ONE call to
    `app.storage.record_cost` here. Four call sites pricing separately is how a dashboard comes to
    disagree with an invoice by a rounding, and how a cap comes to refuse traffic
    on a total nobody can reconcile with the ledger; only one of the four would
    ever be audited.

    Called once per REQUEST and never once per attempt, from the same closing
    stretch as `record_served`: `chosen` is the candidate that SERVED, so a
    request that fell back books one row against the provider that charged for it
    and nothing at all against the one that failed.

    `price_key` comes from the SERVING adapter rather than being rebuilt from the
    provider name, so the key the price was read under and the key the adapter
    publishes cannot drift. The lookup is the same
    `state.adapters[chosen.provider]` the route dialled: the resolver's health seam is what
    guarantees the entry exists, and a
    cache hit reads it without dialling anything.

    `cached` is keyword-only and has NO default, exactly as `gateway_headers`'
    `redacted` has none and for the same reason: a forgotten argument would book a
    replayed answer as money that was spent. What the flag then costs is decided
    inside `record_cost`, not here.

    `usage` is what the provider reported, and `None` is the streamed request that
    ended without a usage event - the two character counts are what its row is
    estimated from then.

    Nothing here can change the answer. `record_cost` swallows a failing write,
    `record_cost_usd` and `debit_budget` are guarded like every other emission,
    and the header is formatted from a `Decimal` this function already holds. The budget the
    debit moves is read at the TOP of the request by `check_budget`, never here,
    so a request is refused on what was already spent when it arrived and not on
    a total this call has just changed.
    """
    ledger: CostLedgerWriter = clients.cost_ledger
    prices: PriceStore = clients.prices
    adapters: Mapping[str, Adapter] = clients.adapters
    row = record_cost(
        ledger,
        table=prices.table,
        tenant=context.tenant,
        alias=alias,
        provider=chosen.provider,
        model=chosen.model,
        usage=usage,
        prompt_chars=prompt_chars,
        completion_chars=completion_chars,
        cached=cached,
        session=context.session,
        trace_id=context.trace_id,
        prompt_version=context.prompt_version,
        price_key=adapters[chosen.provider].price_key(chosen.model),
        latency_ms=latency_ms,
    )
    record_cost_usd(
        tenant=context.tenant, provider=chosen.provider, model=chosen.model, usd=row.cost_usd
    )
    debit_budget(clients, context, usd=row.cost_usd)
    debit_tokens(clients, context, row=row)
    return cost_header_value(row.cost_usd)


def book_failure(
    clients: Clients,
    context: RequestContext,
    *,
    alias: str,
    chosen: Resolved,
    status: int,
    latency_ms: int | None,
) -> CostLedgerRow:
    """File the row of a request that failed with a 5xx after its alias resolved.

    Cost 0 and tokens 0, against the first candidate. It moves no budget, no token bucket and
    no cost counter, because nothing was served or spent: only `record_cost` runs. Every
    served number on the console reads `status < 500`, so this row changes none of them.
    """
    adapters: Mapping[str, Adapter] = clients.adapters
    return record_cost(
        clients.cost_ledger,
        table=clients.prices.table,
        tenant=context.tenant,
        alias=alias,
        provider=chosen.provider,
        model=chosen.model,
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        cached=False,
        session=context.session,
        trace_id=context.trace_id,
        prompt_version=context.prompt_version,
        price_key=adapters[chosen.provider].price_key(chosen.model),
        status=status,
        latency_ms=latency_ms,
    )


def cost_header_value(usd: Decimal) -> str:
    """One cost as `X-Gateway-Cost-USD` carries it: six decimal places, always.

    `Decimal` formatting rather than a float or a rounded string: every cost this
    gateway books is already quantized to `COST_QUANTUM`, and the quantize here is
    what makes the SIX places a property of the header rather than of whatever
    reached it. A
    `float` would render `0.000105` as an approximation and drop the last place on
    some values, which is exactly the disagreement between a client's total and
    the ledger this header exists to prevent.
    """
    return f"{usd.quantize(COST_QUANTUM):f}"
