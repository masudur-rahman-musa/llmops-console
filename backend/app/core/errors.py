"""The typed error contract every client-facing failure is drawn from.

`ErrorCode` is closed on purpose: a caller writes code against these nine
strings, so adding, renaming or dropping one is a change to the published API
rather than an implementation detail. A test holds the whole table - each code's
string, its HTTP status, and that exactly one concrete
`GatewayError` subclass carries it - against a literal written out in the test,
so any of those three drifting fails there.

Every class below is raised by something in the code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class ErrorCode(StrEnum):
    """The wire `code` on every error body. A closed set."""

    UNAUTHORIZED = "UNAUTHORIZED"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    ALIAS_UNKNOWN = "ALIAS_UNKNOWN"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    BUDGET_UNAVAILABLE = "BUDGET_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    ALL_PROVIDERS_FAILED = "ALL_PROVIDERS_FAILED"
    VALIDATION = "VALIDATION"


class GatewayError(Exception):
    """Base for every failure the gateway reports to a client.

    A subclass that a client can be shown sets both class attributes; the two
    intermediate bases that do not (`GatewayError` itself and `RoutingError`)
    are how the error tests tell a real code from a grouping. `param` names the input at fault where
    the raiser knows it.
    """

    code: ClassVar[ErrorCode]
    http_status: ClassVar[int]

    retry_after_seconds: int | None = None
    """Whole seconds a client should wait before retrying, or `None` for "do not".

    A class attribute defaulting to `None`, so every existing failure carries no
    wait and the error handlers render `Retry-After` for the ONE that does. Deliberately not
    derived from the
    status: a budget refusal is also a 429, and telling a tenant that has spent its month to come
    back in a second would be a header that is never true.
    """

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


class Unauthorized(GatewayError):
    """No usable API key: absent, unreadable, over the length cap, or unknown.

    One answer for all four. Which of them it was is not disclosed, so a caller
    cannot use the status to probe the key store.
    """

    code: ClassVar[ErrorCode] = ErrorCode.UNAUTHORIZED
    http_status: ClassVar[int] = 401


class TenantMismatch(GatewayError):
    """The key is valid, but it does not belong to the tenant the caller named.

    403 rather than 401: the caller is authenticated and is asking for something
    that is not theirs.
    """

    code: ClassVar[ErrorCode] = ErrorCode.TENANT_MISMATCH
    http_status: ClassVar[int] = 403


class BudgetExceeded(GatewayError):
    """The tenant's spend cap for the period is reached.

    Raised by `services/chat_pipeline.py::check_budget`, at the top of both the chat
    and the embeddings route and before anything is resolved or dialled, so a
    refusal costs no provider call and books no row. A request
    carrying a valid `X-Gateway-Budget-Override` for its own tenant and the
    current month is served instead.
    """

    code: ClassVar[ErrorCode] = ErrorCode.BUDGET_EXCEEDED
    http_status: ClassVar[int] = 429


class BudgetUnavailable(GatewayError):
    """The tenant's month spend could not be learnt from the ledger, so it is not served.

    Raised by `services/chat_accounting.py::check_budget` only while the month's counter is
    not loaded yet; a Redis that cannot answer still serves. 503, because trying again
    shortly can succeed. A test pins this.
    """

    code: ClassVar[ErrorCode] = ErrorCode.BUDGET_UNAVAILABLE
    http_status: ClassVar[int] = 503


class RateLimited(GatewayError):
    """The tenant has spent its minute's requests or its minute's tokens.

    Raised by `services/chat_pipeline.py::check_rate_limit`, at the top of both the
    chat and the embeddings route and BEFORE `check_budget`, so a tenant over
    both is told to slow down rather than to top up.

    The only failure that carries a `retry_after_seconds`, and it is required
    here rather than optional: a limiter that will not say when to come back
    makes every client guess, and the bucket that refused already knows when its
    next whole token lands.
    """

    code: ClassVar[ErrorCode] = ErrorCode.RATE_LIMITED
    http_status: ClassVar[int] = 429

    def __init__(self, message: str, *, retry_after_seconds: int, param: str | None = None) -> None:
        super().__init__(message, param=param)
        self.retry_after_seconds = retry_after_seconds


class ProviderError(GatewayError):
    """A provider call failed: it answered with a failure, timed out, or answered
    something we could not read.

    Two attributes beyond the base. `retryable` is the whole point of the class:
    the routes' fallback loop has to decide whether to try the next candidate
    without re-parsing a status it can no longer see, so the adapter that knew the
    status records the verdict here. It is true for 429, any 5xx and a timeout, false for
    every other failure.

    `refused` marks a provider refusing the request for content policy. A refused
    error is never retryable: another provider saying yes would undo the refusal,
    so it is not tried again, not passed to the next candidate, and not counted
    against the breaker.

    `provider_message` is where the provider's own words go, and they go nowhere
    else: `message` is client-facing and carries the provider name and the status
    only, so an upstream body cannot put request ids or quota wording in front of a
    tenant. Tests hold both halves against a marker planted in the fixtures.
    """

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_ERROR
    http_status: ClassVar[int] = 502

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        retryable: bool = False,
        refused: bool = False,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message, param=param)
        self.retryable = retryable and not refused
        self.refused = refused
        self.provider_message = provider_message


class LibraryError(Exception):
    """Base for a failure raised by a pure library - redaction, pricing, evals.

    A SECOND hierarchy on purpose, deliberately NOT a `GatewayError`. The eight
    codes above are a closed set and the exhaustive-code test allows
    exactly one concrete `GatewayError` per code; `VALIDATION` and
    `PROVIDER_ERROR` already have theirs (`app.core.validation.IdentifierInvalid`
    and `ProviderError`), so a library failure cannot be a `GatewayError`
    carrying either. The two
    hierarchies stay apart and `app.core.http_errors.as_gateway_error` joins
    them.

    `code` and `http_status` are set by the two middle bases below that have an
    HTTP answer; a direct subclass would set neither, for a failure that is
    neither a bad input nor a store outage.
    """

    code: ClassVar[ErrorCode]
    http_status: ClassVar[int]


class LibraryInputInvalid(LibraryError):
    """A library rejected its input. The 422 `VALIDATION` answer, named once.

    Where a route converts one, it becomes `IdentifierInvalid` - code
    `VALIDATION`, 422 - so the status lives here rather than at each call site.
    """

    code: ClassVar[ErrorCode] = ErrorCode.VALIDATION
    http_status: ClassVar[int] = 422


class LibraryDependencyUnavailable(LibraryError):
    """A library's backing store could not be reached. The 502, named once.

    Where a route converts one, it becomes a retryable `ProviderError` - code
    `PROVIDER_ERROR`, 502 - the same code an unreachable provider lands on.
    """

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_ERROR
    http_status: ClassVar[int] = 502
