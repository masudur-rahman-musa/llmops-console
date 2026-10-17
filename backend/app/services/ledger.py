"""Cost and residency recording: two functions that price a request and file its row.

These live here rather than in storage because they make decisions: `record_cost`
prices the request through the price table, and `record_residency` works out whether
the served region was allowed. Storage writers only file the row they are handed.

`region_allowed` is its own copy rather than the resolver's, and that is on purpose
(see `storage.residency_log` for the reasoning). The property-tested resolver is not
touched by anything here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.limits.cost import cost_for, estimate_tokens
from app.limits.prices import PriceTable
from app.model_routing.policy import DataClass, Policy, Residency
from app.model_routing.resolver import Resolved
from app.schemas.provider import Usage
from app.storage.cost_ledger import CACHED_COST_USD, CostLedgerRow, CostLedgerWriter
from app.storage.residency_log import ResidencyLogRow, ResidencyLogWriter

__all__ = ["record_cost", "record_residency"]


def _utcnow_cost() -> datetime:
    return datetime.now(tz=UTC)


def _utcnow_residency() -> datetime:
    return datetime.now(tz=UTC)


def region_allowed(policy: Policy, tenant: str, data_class: DataClass, region: str) -> bool:
    r"""Does this tenant's policy permit this data class in this region.

    The same narrowing `routing/resolver.py` performs when it builds the
    candidate list: the data class rule's `allow_regions`, intersected with
    `policy.eu_regions` when the tenant's residency is EU. Deny by default at
    both levels — an unknown tenant and a data class with no rule each allow
    nothing, never the union of anything else.

    Duplicated from the resolver rather than shared with it on purpose. Extracting
    a helper would mean editing `routing/resolver.py`, and that module is the one
    the residency property test proves; a log that could change the resolver in
    order to describe it is not evidence. What keeps the two honest is that this
    function is only ever asked about a region the resolver ALREADY returned, so
    a disagreement between them shows up as an `allowed=false` row and an alert
    rather than as a silent pass. The rows are
    `test_region_allowed_over_the_shipped_policy` and
    `test_a_data_class_with_no_rule_allows_nothing`.
    """
    tenant_config = policy.tenants.get(tenant)
    if tenant_config is None:
        return False
    rule = tenant_config.data_class_rules.get(data_class)
    allowed = set(rule.allow_regions) if rule is not None else set()
    if tenant_config.residency is Residency.EU:
        allowed &= set(policy.eu_regions)
    return region in allowed


def record_cost(
    ledger: CostLedgerWriter,
    *,
    table: PriceTable,
    tenant: str,
    alias: str,
    provider: str,
    model: str,
    region: str,
    usage: Usage | None,
    prompt_chars: int = 0,
    completion_chars: int = 0,
    cached: bool,
    session: str | None,
    trace_id: str,
    price_key: str,
    now: Callable[[], datetime] = _utcnow_cost,
) -> CostLedgerRow:
    """Price one request, write its row, and answer the row. Never raises.

    Moved from `storage.cost_ledger` so storage only files rows and never prices
    them. The semantics are identical; see `storage.cost_ledger` for the full
    decision record.
    """
    import logging

    logger = logging.getLogger(__name__)

    at = now()
    cost = cost_for(
        table,
        price_key,
        at,
        usage=usage,
        prompt_chars=prompt_chars,
        completion_chars=completion_chars,
    )
    if usage is None:
        input_tokens = estimate_tokens(prompt_chars)
        output_tokens = estimate_tokens(completion_chars)
    else:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

    row = CostLedgerRow(
        ts=at,
        tenant=tenant,
        alias=alias,
        provider=provider,
        model=model,
        region=region,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached=cached,
        cost_usd=CACHED_COST_USD if cached else cost.usd,
        price_version=cost.price_version,
        estimated=cost.estimated,
        session=session,
        trace_id=trace_id,
    )
    try:
        ledger.write(row)
    except Exception:
        logger.exception("cost ledger write failed for tenant %s, model %s", tenant, model)
    return row


def record_residency(
    log: ResidencyLogWriter,
    *,
    policy: Policy,
    tenant: str,
    data_class: DataClass,
    alias: str,
    chosen: Resolved,
    now: Callable[[], datetime] = _utcnow_residency,
) -> None:
    """Write the row for the candidate that served. Never raises.

    Moved from `storage.residency_log` so storage only files rows and never
    decides whether a region was allowed. The semantics are identical; see
    `storage.residency_log` for the full decision record.
    """
    import logging

    from app.core.metrics import record_residency_violation

    logger = logging.getLogger(__name__)

    row = ResidencyLogRow(
        ts=now(),
        tenant=tenant,
        data_class=data_class,
        alias=alias,
        provider=chosen.provider,
        region=chosen.region,
        allowed=region_allowed(policy, tenant, data_class, chosen.region),
    )
    if not row.allowed:
        record_residency_violation(tenant=tenant)
    try:
        log.write(row)
    except Exception:
        logger.exception(
            "residency log write failed for tenant %s, provider %s", tenant, chosen.provider
        )
