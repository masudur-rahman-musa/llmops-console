"""Typed failures raised by the limits package.

`PriceTableInvalid` is the shape `opsconsole.routing.errors.PolicyInvalid` set:
a rejected `prices.yaml` is an operator error reported to whoever loaded or
reloaded it, never an answer to a caller, so it is a plain exception rather than
a `GatewayError` carrying a wire code the handler could use.

Nothing in this package raises on a request. A price key the table does not
cover is a sentinel result, not an exception - `opsconsole.limits.cost` is where
that is decided, and `test_an_unpriced_key_costs_zero_and_does_not_raise` is
what holds it.
"""

from __future__ import annotations


class PriceTableInvalid(Exception):
    """The price table could not be read, parsed or validated.

    Raised only at LOAD, and its message names the offending field path so an
    operator sees the field rather than a traceback in three exception types.
    Pinned by `test_invalid_price_table_names_the_field`.
    """
