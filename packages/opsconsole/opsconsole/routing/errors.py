"""Typed routing failures.

Each client-facing error carries the wire `code` from the API error contract and
the HTTP status it maps to. `opsconsole.gateway.errors` builds the OpenAI-shaped
body from those two attributes, so they are pinned exactly by
`test_empty_candidates_raises_residency_unavailable`, `test_unknown_alias_raises`,
`test_is_healthy_filtering_everything_raises_all_providers_unavailable`,
`test_identifier_rejection_table` and, as a set,
`test_error_code_set_is_exhaustive`.

`PolicyInvalid` is the exception to that: a rejected policy file is an operator
error reported to whoever reloaded it, never an answer to a caller, so it is a
plain exception rather than a `GatewayError` with no code the handler could use.
"""

from __future__ import annotations

from typing import ClassVar

from opsconsole.core.errors import ErrorCode, GatewayError


class RoutingError(GatewayError):
    """Base for every client-facing failure raised by the routing package."""


class PolicyInvalid(Exception):
    """The policy file could not be read, parsed or validated."""


class AliasUnknown(RoutingError):
    """The request named an alias the policy does not define.

    The error contract fixes the code but not the status. 404 is the choice
    here, matching what an OpenAI-compatible client already expects from a model
    name it cannot use; pinned by `test_unknown_alias_raises`.
    """

    code: ClassVar[ErrorCode] = ErrorCode.ALIAS_UNKNOWN
    http_status: ClassVar[int] = 404


class ResidencyUnavailable(RoutingError):
    """No candidate is allowed to serve this request.

    Deny by default: this is what an empty candidate list becomes. The resolver
    never widens the region set to avoid raising it.
    """

    code: ClassVar[ErrorCode] = ErrorCode.RESIDENCY_UNAVAILABLE
    http_status: ClassVar[int] = 451


class AllProvidersUnavailable(RoutingError):
    """Candidates were allowed by residency, but none of them is healthy."""

    code: ClassVar[ErrorCode] = ErrorCode.ALL_PROVIDERS_FAILED
    http_status: ClassVar[int] = 503


class IdentifierInvalid(RoutingError):
    """A client-supplied identifier failed its pre-check.

    r`grep -rlE "raise Identifier[I]nvalid" packages/` returns EIGHTEEN modules
    as of an earlier change - the bracket is what keeps this quotation out of its own
    results, so the command can be re-run here and compared. The list is
    re-derived whole from that command rather than appended to.

        cache/store.py           a tenant and a cache key, before any command
        cache/vectors.py         a tenant on its way into a Qdrant collection
                                 name, which is a URL segment with no backstop
        evals/cases.py           a golden line's case id, tenant and principal,
                                 which become a report path and eval_cases columns
        gateway/context.py       the X-Tenant, X-Session and X-Data-Class headers
        gateway/routes/admin.py  the admin tenant parameter, and a reload whose
                                 routing.yaml will not parse
        gateway/routes/chat.py   the X-Gateway-Rehydrate map id, on a returned
                                 answer and on a streamed one alike
        gateway/routes/directory.py  the pushed tenant, and a directory cap
                                 breach converted out of DirectoryNameInvalid
        gateway/routes/redact.py the redact endpoint's own identifiers
        ledger/cost_ledger.py    every identifier a cost row is keyed by, plus
                                 the model, the trace id and the price version
        ledger/residency.py      the identifiers a residency row is keyed by
        limits/budget.py         a tenant and a YYYY-MM month, before any command
        limits/ratelimit.py      a tenant on its way into a rate-limit bucket key
        providers/anthropic.py   tool call arguments that will not parse as JSON
        providers/azure_openai.py  a deployment name on its way into a URL path
        providers/base.py        checked_tool_name, for any identifier on its way
                                 to a provider
        redaction/directory.py   a tenant on its way into a directory key
        redaction/store.py       a tenant and a redaction_map_id
        routing/resolver.py      a tenant, an alias, a data class

    A nineteenth module, `gateway.errors`, CONSTRUCTS one without raising it, to render
    FastAPI's own `RequestValidationError` into the wire contract, so it answers a
    grep for the constructor and not the one above. A twentieth, `limits/errors.py`,
    only quotes the command in prose.

    One code covers every rejected input. It lives in this package rather than in
    `core` because the resolver was the first raiser; a later plan may well move
    it up, and the list above is what would have to move with it.
    """

    code: ClassVar[ErrorCode] = ErrorCode.VALIDATION
    http_status: ClassVar[int] = 422
