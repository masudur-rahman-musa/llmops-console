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
- so a month of rows adds up here to what the ledger holds.

`ratelimit.py` is the per-minute half of the same job (#28): two token buckets
per tenant, one for requests and one for tokens, refilling continuously at
`rpm/60` and `tpm/60`. It shares the budget counter's two shapes deliberately -
a Redis implementation and an in-process twin on one switch, and `None` for a
store that cannot answer, which the route reads as serve-and-log - so both
ceilings fail the same way and neither can turn a Redis blip into an outage.
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
from opsconsole.limits.ratelimit import (
    MIN_RETRY_AFTER_SECONDS,
    RATE_LIMIT_KEY_PREFIX,
    RATE_LIMIT_KEY_TTL_SECONDS,
    SECONDS_PER_MINUTE,
    InProcessRateLimitStore,
    RateDimension,
    RateLimitStore,
    RedisRateLimitStore,
    Taken,
    bucket_key,
    refill,
    retry_after,
)

__all__ = [
    "CHARS_PER_ESTIMATED_TOKEN",
    "COST_QUANTUM",
    "MIN_RETRY_AFTER_SECONDS",
    "PRICE_UNIT_TOKENS",
    "RATE_LIMIT_KEY_PREFIX",
    "RATE_LIMIT_KEY_TTL_SECONDS",
    "SECONDS_PER_MINUTE",
    "UNPRICED_VERSION",
    "Cost",
    "InProcessRateLimitStore",
    "PriceEntry",
    "PriceStore",
    "PriceTable",
    "PriceTableInvalid",
    "RateDimension",
    "RateLimitStore",
    "RedisRateLimitStore",
    "Taken",
    "bucket_key",
    "cost_for",
    "estimate_tokens",
    "load_prices",
    "price_for",
    "price_version",
    "refill",
    "retry_after",
]
