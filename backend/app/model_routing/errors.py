"""Typed routing failures.

Each client-facing error carries the wire `code` from the API error contract and
the HTTP status it maps to. `app.core.http_errors` builds the OpenAI-shaped
body from those two attributes, so they are fixed exactly.

`PolicyInvalid` is the exception to that: a rejected policy file is an operator
error reported to whoever reloaded it, never an answer to a caller, so it is a
plain exception rather than a `GatewayError` with no code the handler could use.
"""

from __future__ import annotations

from typing import ClassVar

from app.core.errors import ErrorCode, GatewayError


class RoutingError(GatewayError):
    """Base for every client-facing failure raised by the routing package."""


class PolicyInvalid(Exception):
    """The policy file could not be read, parsed or validated."""


class AliasUnknown(RoutingError):
    """The request named an alias the policy does not define.

    The error contract fixes the code but not the status. 404 is the choice
    here, matching what an OpenAI-compatible client already expects from a model
    name it cannot use.
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
