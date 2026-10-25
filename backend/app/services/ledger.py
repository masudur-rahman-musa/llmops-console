"""Cost recording: one function that prices a request and files its row.

It lives here rather than in storage because it makes a decision: `record_cost`
prices the request through the price table. Storage writers only file the row
they are handed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.core.metrics import record_ledger_write
from app.core.validation import IdentifierInvalid
from app.limits.cost import cost_for, estimate_tokens
from app.limits.prices import PriceTable
from app.schemas.provider import Usage
from app.storage.cost_ledger import CACHED_COST_USD, CostLedgerRow, CostLedgerWriter

__all__ = ["record_cost"]


def _utcnow_cost() -> datetime:
    return datetime.now(tz=UTC)


def record_cost(
    ledger: CostLedgerWriter,
    *,
    table: PriceTable,
    tenant: str,
    alias: str,
    provider: str,
    model: str,
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
    except IdentifierInvalid:
        # A row failing a pre-check never reaches the writer, so it is counted
        # here (`test_a_row_failing_its_pre_check_is_counted_dropped_invalid`).
        logger.exception("cost ledger row refused for tenant %s, model %s", tenant, model)
        record_ledger_write(result="dropped_invalid")
    except Exception:
        logger.exception("cost ledger write failed for tenant %s, model %s", tenant, model)
    return row
