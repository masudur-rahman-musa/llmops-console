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

    Raised by the resolver for a tenant or alias, and by
    `opsconsole.gateway.context` for the `X-Tenant`, `X-Session` and
    `X-Data-Class` headers, so one code covers every rejected input.
    """

    code: ClassVar[ErrorCode] = ErrorCode.VALIDATION
    http_status: ClassVar[int] = 422
