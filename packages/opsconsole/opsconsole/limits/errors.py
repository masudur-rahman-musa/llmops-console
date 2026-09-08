"""Typed failures raised by the limits package.

`PriceTableInvalid` is the shape `opsconsole.routing.errors.PolicyInvalid` set:
a rejected `prices.yaml` is an operator error reported to whoever loaded or
reloaded it, never an answer to a caller, so it is a plain exception rather than
a `GatewayError` carrying a wire code the handler could use.

`PriceTableInvalid` is still the only class HERE, and it is still load-only, but
this package does now raise on a request: `opsconsole.limits.budget.month_key`
raises `routing.errors.IdentifierInvalid` for a tenant or a month that may not
shape a key, before any command is sent
(`test_a_malformed_tenant_is_refused_before_any_command`). It raises the routing
package's class rather than one of its own on purpose - it is checking the
gateway's own identifier against the gateway's own `IDENTIFIER_PATTERN`, exactly
as `redaction/store.py` does, and a second class saying the same thing would give
the handler two codes for one fault.

A price key the table does not cover is still not an exception at all: it is a
sentinel result, decided in `opsconsole.limits.cost`, and
`test_an_unpriced_key_costs_zero_and_does_not_raise` is what holds it. A budget
that is reached is not one either - the refusal is `core.errors.BudgetExceeded`,
raised from `gateway/routes/chat.py::check_budget`, because it is an answer to a
caller with a wire code and a 429, which is the whole distinction this module's
first paragraph draws.
"""

from __future__ import annotations


class PriceTableInvalid(Exception):
    """The price table could not be read, parsed or validated.

    Raised only at LOAD, and its message names the offending field path so an
    operator sees the field rather than a traceback in three exception types.
    Pinned by `test_invalid_price_table_names_the_field`.
    """
