"""Prices and the cost of a request.

Two files, library only: nothing here reads a header, resolves an alias, writes
a row or touches a route. `prices.py` is the versioned price table - the schema
for `prices.yaml`, its strict loader, the effective-from selection and the
snapshot the gateway holds. `cost.py` is the arithmetic over it, and the one
decision that matters at runtime: a price the table does not carry costs zero
and says `estimated` with the version `unpriced`, rather than raising on a
request a provider already served.

`budget.py` is the other half of that arithmetic: the running total of those
same rows for the month in progress, and the two thresholds the design puts on it
(#27). It shares the ledger's quantum exactly - the counter is integer micro-USD
- so a month of rows adds up here to what the ledger holds. Per-tenant rate
limits are the one piece of this package still to come.
"""

from __future__ import annotations

from opsconsole.limits.cost import (
    CHARS_PER_ESTIMATED_TOKEN,
    COST_QUANTUM,
    PRICE_UNIT_TOKENS,
    UNPRICED_VERSION,
    Cost,
    cost_for,
    estimate_tokens,
)
from opsconsole.limits.errors import PriceTableInvalid
from opsconsole.limits.prices import (
    PriceEntry,
    PriceStore,
    PriceTable,
    load_prices,
    price_for,
    price_version,
)

__all__ = [
    "CHARS_PER_ESTIMATED_TOKEN",
    "COST_QUANTUM",
    "PRICE_UNIT_TOKENS",
    "UNPRICED_VERSION",
    "Cost",
    "PriceEntry",
    "PriceStore",
    "PriceTable",
    "PriceTableInvalid",
    "cost_for",
    "estimate_tokens",
    "load_prices",
    "price_for",
    "price_version",
]
