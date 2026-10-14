"""Typed routing failures.

Each client-facing error carries the wire `code` from the API error contract and
the HTTP status it maps to. `app.core.http_errors` builds the OpenAI-shaped
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

from app.core.errors import ErrorCode, GatewayError

# `IdentifierInvalid` moved up to `core/validation.py`, beside the one
# shared identifier check. Re-exported here with the explicit `as` form so every
# raiser that imports it from the routing package keeps resolving.
from app.core.validation import IdentifierInvalid as IdentifierInvalid


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
    never widens the region set to avoid raising it - `ResidencyMode.RELAXED`
    widens by the configured mode alone, before any result exists to react to,
    and never for `pii`.
    """

    code: ClassVar[ErrorCode] = ErrorCode.RESIDENCY_UNAVAILABLE
    http_status: ClassVar[int] = 451


class AllProvidersUnavailable(RoutingError):
    """Candidates were allowed by residency, but none of them is healthy."""

    code: ClassVar[ErrorCode] = ErrorCode.ALL_PROVIDERS_FAILED
    http_status: ClassVar[int] = 503
