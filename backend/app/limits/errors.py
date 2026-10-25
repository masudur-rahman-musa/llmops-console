"""Typed failures raised by the limits package.

`PriceTableInvalid` is the shape `app.model_routing.errors.PolicyInvalid` set:
a rejected price table is an operator error reported to whoever loaded or
reloaded it, never an answer to a caller, so it is NOT a `GatewayError`
carrying a wire code the handler could use. It is a `LibraryInputInvalid` - a
bad input, like every load-time file failure in the evals package - but it is
raised only at load and never converted to HTTP, so that mapping stays unused.

`PriceTableInvalid` is still the only class HERE, and it is still load-only, but
this package now raises on a request from TWO seams, both of them key builders
and both raising `routing.errors.IdentifierInvalid` before any command is sent -
`app.limits.budget.month_key` for a tenant or a month and
`app.limits.ratelimit.bucket_key` for a tenant. The list is
r`grep -rn "raise Identifier[I]nvalid" backend/app/limits`,
the bracket keeping this quotation out of its own results, which
answers those two files and no others. They raise the routing package's class
rather than one of their own on purpose - each is checking the gateway's own
identifier against the gateway's own `IDENTIFIER_PATTERN`, exactly as
`redaction/store.py` does, and a second class saying the same thing would give
the handler two codes for one fault.

A price key the table does not cover is still not an exception at all: it is a
sentinel result, decided in `app.limits.cost`. A budget
that is reached is not one either, and neither is a rate limit - those refusals
are `core.errors.BudgetExceeded` and `core.errors.RateLimited`, raised from
`services/chat_pipeline.py::check_budget` and `::check_rate_limit`, because they are
answers to a caller with a wire code and a 429, which is the whole distinction
this module's first paragraph draws.
"""

from __future__ import annotations

from app.core.errors import LibraryInputInvalid


class PriceTableInvalid(LibraryInputInvalid):
    """The price table could not be read, parsed or validated.

    Raised only at LOAD, and its message names the offending field path so an
    operator sees the field rather than a traceback in three exception types.
    """
