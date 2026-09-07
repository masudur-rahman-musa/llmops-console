"""What one request cost, and how honest that number is.

Two paths, one result shape. When the provider reported usage, the tokens are
counted and the answer carries `estimated=False` and the version string of the
price entry that produced it. When it did not, the tokens are ESTIMATED from
character counts and the answer carries `estimated=True`. The flag is the part
that matters: an estimate is a number nobody measured, and the ledger keeps it
separable from the ones that were.

The estimator is characters divided by four, applied to both directions. A real
tokeniser would be exactly right for one vendor and confidently wrong for the
others, and it would be a dependency; a stated approximation carrying a flag is
the more honest of the two.

Arithmetic is `Decimal`, quantized to six decimal places with `ROUND_HALF_UP`.
A month of rows has to sum to the same number twice, which floats do not
promise, and six places is a tenth of a microdollar - below the cost of any
single token any provider here sells.

Nothing in this module raises. A key the table does not price, and a key whose
entries are all still in the future, both answer zero flagged `estimated` with
the version string `unpriced`, because this runs on every served request and a
missing price must never be able to change a status code. Those rows stay
findable in the ledger by that exact string, which is the point of not raising.
It is also WARNED, once per key per process, so the gap is noticed by whoever
runs the gateway and not only by whoever reads the ledger a month later.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict

from opsconsole.limits.prices import PriceTable, price_for, price_version
from opsconsole.providers.base import Usage

logger = logging.getLogger(__name__)

_WARNED_KEYS: set[str] = set()
"""Which keys have already been warned about, so a permanently unpriced model
costs one log line rather than one per request. Process-local and never read
back: the ledger's `unpriced` rows are the durable record, and this only exists
so the gap is noticed the first time. Bounded by the number of `provider/model`
pairs `routing.yaml` can resolve to, which is why it does not need eviction."""

UNPRICED_VERSION = "unpriced"
"""The price version a request gets when the table does not price its key.

A literal string rather than an empty version or a `None`, so the rows are
findable with one equality in SQL. Pinned exactly by
`test_an_unpriced_key_costs_zero_and_does_not_raise`."""

PRICE_UNIT_TOKENS = Decimal(1_000_000)
"""Prices are published per 1,000,000 tokens, so that is what the table holds and
this is the divisor. Pinned exactly by `test_input_and_output_are_priced_per_1m`."""

COST_QUANTUM = Decimal("0.000001")
"""Six decimal places, `ROUND_HALF_UP`. Pinned exactly, as a string, by
`test_the_cost_is_quantized_to_exactly_six_decimal_places`."""

CHARS_PER_ESTIMATED_TOKEN = 4
"""The estimator's whole model: four characters to a token, both directions.
Pinned exactly by `test_absent_usage_is_estimated_from_characters`."""


class Cost(BaseModel):
    """What a request cost, and which price said so.

    Frozen: a cost is what it was at the moment it was computed, and the ledger
    row that carries it is written once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    usd: Decimal
    estimated: bool
    price_version: str


def estimate_tokens(chars: int) -> int:
    """Characters to tokens, the whole heuristic.

    Floor division, so an empty string estimates zero tokens rather than one and
    a request with no text is not booked as having consumed something.
    """
    return chars // CHARS_PER_ESTIMATED_TOKEN


def cost_for(
    table: PriceTable,
    key: str,
    at: date | datetime,
    usage: Usage | None = None,
    prompt_chars: int = 0,
    completion_chars: int = 0,
) -> Cost:
    """Price one request.

    `usage` is what the provider reported; `None` is the absent case, and the
    two character counts are what the estimate is built from then. An unpriced
    key short-circuits both, because there is nothing to multiply by.
    """
    entry = price_for(table, key, at)
    if entry is None:
        if key not in _WARNED_KEYS:
            _WARNED_KEYS.add(key)
            logger.warning(
                "no price in force for %s; booking the request unpriced", key, extra={"key": key}
            )
        return Cost(
            usd=Decimal(0).quantize(COST_QUANTUM, rounding=ROUND_HALF_UP),
            estimated=True,
            price_version=UNPRICED_VERSION,
        )

    if usage is None:
        prompt_tokens = estimate_tokens(prompt_chars)
        completion_tokens = estimate_tokens(completion_chars)
        estimated = True
    else:
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        estimated = False

    total = (
        Decimal(prompt_tokens) * entry.input_usd_per_1m
        + Decimal(completion_tokens) * entry.output_usd_per_1m
    ) / PRICE_UNIT_TOKENS

    return Cost(
        usd=total.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP),
        estimated=estimated,
        price_version=price_version(key, entry),
    )
