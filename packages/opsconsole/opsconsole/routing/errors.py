"""Typed routing failures.

Each client-facing error carries the wire `code` from the API error contract and
the HTTP status it maps to. An earlier change builds the OpenAI-shaped body from these two
attributes, so they are pinned exactly by
`test_empty_candidates_raises_residency_unavailable`, `test_unknown_alias_raises`,
`test_is_healthy_filtering_everything_raises_all_providers_unavailable` and
`test_identifier_rejection_table` rather than left to a later reader.

`PolicyInvalid` is the exception to that: a rejected policy file is an operator
error reported to whoever reloaded it, so it has no client code.
"""

from __future__ import annotations

from typing import ClassVar


class RoutingError(Exception):
    """Base for every failure raised by the routing package."""


class PolicyInvalid(RoutingError):
    """The policy file could not be read, parsed or validated."""


class AliasUnknown(RoutingError):
    """The request named an alias the policy does not define.

    The error contract fixes the code but not the status. 404 is the choice
    here, matching what an OpenAI-compatible client already expects from a model
    name it cannot use; pinned by `test_unknown_alias_raises`.
    """

    code: ClassVar[str] = "ALIAS_UNKNOWN"
    http_status: ClassVar[int] = 404


class ResidencyUnavailable(RoutingError):
    """No candidate is allowed to serve this request.

    Deny by default: this is what an empty candidate list becomes. The resolver
    never widens the region set to avoid raising it.
    """

    code: ClassVar[str] = "RESIDENCY_UNAVAILABLE"
    http_status: ClassVar[int] = 451


class AllProvidersUnavailable(RoutingError):
    """Candidates were allowed by residency, but none of them is healthy."""

    code: ClassVar[str] = "ALL_PROVIDERS_FAILED"
    http_status: ClassVar[int] = 503


class IdentifierInvalid(RoutingError):
    """A client-supplied identifier failed its pre-check."""

    code: ClassVar[str] = "VALIDATION"
    http_status: ClassVar[int] = 422
