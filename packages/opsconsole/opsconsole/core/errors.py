"""The typed error contract every client-facing failure is drawn from.

`ErrorCode` is closed on purpose: a caller writes code against these nine
strings, so adding, renaming or dropping one is a change to the published API
rather than an implementation detail. `test_error_code_set_is_exhaustive` holds
the whole table - each code's string, its HTTP status, and the requirement that
exactly one concrete `GatewayError` subclass carries it - against a literal
written out in the test, so any of those three drifting fails there.

Five of the classes below are raised by later plans (budgets, rate limits,
provider calls). They ship here because a code with no class is a code nothing
can raise, and because the exhaustive test measures the pairing rather than the
usage.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class ErrorCode(StrEnum):
    """The wire `code` on every error body. A closed set (the design)."""

    UNAUTHORIZED = "UNAUTHORIZED"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    ALIAS_UNKNOWN = "ALIAS_UNKNOWN"
    RESIDENCY_UNAVAILABLE = "RESIDENCY_UNAVAILABLE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    ALL_PROVIDERS_FAILED = "ALL_PROVIDERS_FAILED"
    VALIDATION = "VALIDATION"


class GatewayError(Exception):
    """Base for every failure the gateway reports to a client.

    A subclass that a client can be shown sets both class attributes; the two
    intermediate bases that do not (`GatewayError` itself and `RoutingError`)
    are how `concrete_error_classes` in tests/gateway/test_error_contract.py
    tells a real code from a grouping. `param` names the input at fault where
    the raiser knows it.
    """

    code: ClassVar[ErrorCode]
    http_status: ClassVar[int]

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


class Unauthorized(GatewayError):
    """No usable API key: absent, unreadable, over the length cap, or unknown.

    One answer for all four. Which of them it was is not disclosed, so a caller
    cannot use the status to probe the key store (an earlier change decisions F and G;
    pinned by `test_missing_key_is_unauthorized` and
    `test_unknown_key_is_unauthorized`).
    """

    code: ClassVar[ErrorCode] = ErrorCode.UNAUTHORIZED
    http_status: ClassVar[int] = 401


class TenantMismatch(GatewayError):
    """The key is valid, but it does not belong to the tenant the caller named.

    403 rather than 401: the caller is authenticated and is asking for something
    that is not theirs. Pinned by `test_key_tenant_mismatch_is_403`.
    """

    code: ClassVar[ErrorCode] = ErrorCode.TENANT_MISMATCH
    http_status: ClassVar[int] = 403


class BudgetExceeded(GatewayError):
    """The tenant's spend cap for the period is reached. Raised from a later plan."""

    code: ClassVar[ErrorCode] = ErrorCode.BUDGET_EXCEEDED
    http_status: ClassVar[int] = 429


class RateLimited(GatewayError):
    """The tenant's request or token rate is exceeded. Raised from a later plan."""

    code: ClassVar[ErrorCode] = ErrorCode.RATE_LIMITED
    http_status: ClassVar[int] = 429


class ProviderError(GatewayError):
    """A provider answered, and what it answered was a failure. Raised from a later plan."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_ERROR
    http_status: ClassVar[int] = 502
