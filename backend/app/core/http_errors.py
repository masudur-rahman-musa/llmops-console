"""The OpenAI-shaped error body, and the handlers that make it the only shape.

A failure leaves the gateway as `{"error": {message, type, code, param}}`, which
is what an OpenAI-compatible client already knows how to read. Four handlers
cover the four ways a request can fail, and between them they leave FastAPI and
Starlette no path to answer in their own `{"detail": ...}` shape or in the
`text/plain` body their unhandled-exception middleware writes:

* `GatewayError` for the typed failures the gateway raises itself
  (`test_residency_unavailable_is_451_openai_shaped`);
* `RequestValidationError` for the ones FastAPI raises before a route runs
  (`test_request_validation_uses_the_error_contract`);
* `StarletteHTTPException` for the ones raised before any of our code is reached
  at all - an unrouted path is a 404 and a wrong method a 405
  (`test_starlette_failures_use_the_error_contract`);
* `Exception` for a bug: anything a route raises that is not typed becomes a 500
  in the same shape, with the traceback logged and never sent
  (`test_unhandled_exception_uses_the_error_contract`). It is also the only
  handler a MIDDLEWARE's raise can reach, so it recognises a `GatewayError` and
  renders it properly rather than as a 500 (an earlier change decision F,
  `test_a_gateway_error_raised_in_middleware_keeps_its_status_and_code`).

The last two carry `code: null`. `ErrorCode` is the closed set of failures the
gateway classifies; an unrouted URL and a crash are not among them, and inventing
a code for either, or bending `VALIDATION` onto a 404, would break the
one-status-per-code table `test_error_code_set_is_exhaustive` holds (an earlier change
decision J).

`type` is the coarse OpenAI-style family, derived from the status rather than
stored per class so a new code cannot forget one. The table is pinned by
`test_error_type_matches_the_status_family`.
"""

from __future__ import annotations

import logging
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import ErrorCode, GatewayError
from app.core.metrics import record_failure
from app.core.tracing import record_error_code
from app.core.validation import IdentifierInvalid

logger = logging.getLogger(__name__)

ERROR_TYPE_BY_STATUS: dict[int, str] = {
    401: "authentication_error",
    403: "permission_error",
    429: "rate_limit_error",
}
"""Statuses whose family is not implied by 4xx-versus-5xx. Pinned by
`test_error_type_matches_the_status_family`."""

CLIENT_ERROR_TYPE = "invalid_request_error"
SERVER_ERROR_TYPE = "api_error"

INTERNAL_ERROR_STATUS = 500
"""What an unhandled exception becomes. Pinned by
`test_unhandled_exception_uses_the_error_contract`."""

GENERIC_FAILURE_MESSAGE = "the request could not be completed"
"""The message for a failure with nothing safe to say about it: a crash, or an
`HTTPException` whose detail is not a string."""


def error_type(http_status: int) -> str:
    """The OpenAI-style error family for one status."""
    if http_status in ERROR_TYPE_BY_STATUS:
        return ERROR_TYPE_BY_STATUS[http_status]
    return SERVER_ERROR_TYPE if http_status >= 500 else CLIENT_ERROR_TYPE


def error_body(
    *, message: str, code: ErrorCode | None, http_status: int, param: str | None
) -> dict[str, dict[str, str | None]]:
    """Build the one body shape every failure uses."""
    return {
        "error": {
            "message": message,
            "type": error_type(http_status),
            "code": code.value if code is not None else None,
            "param": param,
        }
    }


RETRY_AFTER_HEADER = "Retry-After"
"""The one header a failure can add to its own response.

Emitted for a typed error that carries a `retry_after_seconds` and for no other
response, which today is `RateLimited` alone (an earlier change decision D,
`test_the_429_carries_retry_after_in_whole_seconds`,
`test_a_served_response_carries_no_retry_after`)."""


def retry_after_headers(error: GatewayError) -> dict[str, str] | None:
    """`Retry-After` when this error knows when to come back, otherwise nothing.

    The condition is the ERROR carrying a wait, never its status. `if
    http_status == 429` is the obvious implementation and is wrong: a budget
    refusal is also a 429, and it would then tell a tenant that has spent its
    month to retry in a second, for ever
    (`test_a_budget_refusal_carries_no_retry_after`).

    Whole seconds, because that is what the field is defined in - the rounding
    up happens in `limits/ratelimit.retry_after`, where the bucket's own rate is
    known (`test_the_retry_after_is_the_seconds_to_the_next_whole_token`).
    """
    wait = error.retry_after_seconds
    return None if wait is None else {RETRY_AFTER_HEADER: str(wait)}


def error_response(error: GatewayError) -> JSONResponse:
    """Render a typed gateway failure as its status, its body and any wait it carries."""
    return JSONResponse(
        status_code=error.http_status,
        content=error_body(
            message=error.message,
            code=error.code,
            http_status=error.http_status,
            param=error.param,
        ),
        headers=retry_after_headers(error),
    )


async def handle_gateway_error(request: Request, exc: Exception) -> JSONResponse:
    """Turn a raised `GatewayError` into the wire contract, and onto the trace.

    Typed as `Exception` because that is the signature Starlette's handler
    registry declares; dispatch has already matched the class, so the cast
    narrows without adding a branch that no request can take.

    This is also where the root span learns WHY a request failed. The tracing
    middleware sees only a status, and a status is not a reason: 502 is every
    provider outage and 451 is every residency refusal, so the typed code is
    recorded here, where it exists, and read there. Only the code travels - never
    the message, which for a `ProviderError` carries the provider's own words
    (`test_a_provider_failure_records_the_code_on_the_span`).

    `record_failure` is the metrics half of the same fact, and it is here for the
    same reason: this is where a raised `GatewayError` arrives with its typed code
    still attached. It moves the ONE failure family that has an operator response
    of its own and nothing else - which family, and on what, is
    `core/metrics.py`'s decision rather than this file's
    (`test_a_provider_failure_increments_the_provider_error_counter`). A 451 used
    to move a second one and no longer does: `gateway_residency_violations_total`
    now counts only a request actually SERVED from a forbidden region, which no
    handler ever sees (an earlier change decision E,
    `test_a_residency_refusal_leaves_the_violation_counter_flat`).
    """
    error = cast(GatewayError, exc)
    record_error_code(request.scope, error.code.value if error.code is not None else None)
    record_failure(request.scope, error)
    return error_response(error)


async def handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Replace FastAPI's `{"detail": [...]}` with the same contract.

    The first error is the one reported: `param` is the last segment of its
    location (the query parameter or header name), which is what a client points
    at. Pinned by `test_validation_body_names_the_offending_field`.
    """
    first = cast(RequestValidationError, exc).errors()[0]
    location = first.get("loc") or ()
    param = str(location[-1]) if location else None
    message = (
        f"{param}: {first.get('msg', 'is not valid')}" if param else "the request is not valid"
    )
    return error_response(IdentifierInvalid(message, param=param))


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Render Starlette's own 404 and 405 in the same shape, with no typed code.

    Reached before routing has found anything of ours to run, so there is no
    `GatewayError` to carry a code and none is invented. The exception's headers
    are carried through, because a 405 without `Allow` is a malformed answer
    (RFC 9110); pinned by `test_starlette_failures_use_the_error_contract`.

    Only Starlette's own raises reach here in this plan, and those carry a string
    `detail`. A non-string one is replaced rather than rendered, so a later
    endpoint raising `HTTPException(409, detail={...})` cannot put a Python repr
    on the wire (`test_non_string_http_detail_is_not_rendered`).
    """
    http_exception = cast(StarletteHTTPException, exc)
    status = http_exception.status_code
    detail = http_exception.detail
    return JSONResponse(
        status_code=status,
        content=error_body(
            message=detail if isinstance(detail, str) else GENERIC_FAILURE_MESSAGE,
            code=None,
            http_status=status,
            param=None,
        ),
        headers=http_exception.headers,
    )


async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """The last resort: a bug in a route is still an answer a client can parse.

    Without this, Starlette's `ServerErrorMiddleware` answers `text/plain
    Internal Server Error`, and a client whose error path reads
    `response.json()["error"]["message"]` fails on top of the outage. The
    traceback is logged and never put in the body, because the body is the one
    place an internal detail must not reach
    (`test_unhandled_exception_uses_the_error_contract`).

    A typed failure is the exception, and it is why this handler branches. Starlette
    lifts the `Exception` entry out of `ExceptionMiddleware` into
    `ServerErrorMiddleware` at the very outside of the stack, so a `GatewayError`
    raised by a MIDDLEWARE - which is anything installed above the router,
    `TracingMiddleware` included - never reaches `handle_gateway_error` at all.
    Without the branch it would answer 500 with `code: null`, and a client's retry
    logic would read a gateway bug where the gateway meant 451. An earlier change installs
    the first middleware, so an earlier change owes the fix
    (`test_a_gateway_error_raised_in_middleware_keeps_its_status_and_code`).

    Two boundaries worth knowing before this is read as a total guarantee.
    `ServerErrorMiddleware` re-raises after the response is sent, so a real
    deployment logs each crash twice: once here and once from the ASGI server -
    and a typed failure rendered through this branch is re-raised too, which is a
    traceback in the log for a request that was answered correctly.
    And it only sends a body when the response has not started, so a crash part
    way through a streamed response still reaches the
    client as a truncated stream with no JSON at all.
    """
    if isinstance(exc, GatewayError):
        return await handle_gateway_error(request, exc)

    logger.exception("unhandled exception serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=INTERNAL_ERROR_STATUS,
        content=error_body(
            message=GENERIC_FAILURE_MESSAGE,
            code=None,
            http_status=INTERNAL_ERROR_STATUS,
            param=None,
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Install all four handlers, so no failure answers outside the contract.

    Registration order does not matter, but the four are not peers. Starlette
    puts the three specific ones in `ExceptionMiddleware`, which walks the raised
    class's MRO, and lifts the `Exception` entry out into `ServerErrorMiddleware`
    at the very outside of the stack. So the last handler catches what the other
    three do not match, and also anything raised outside `ExceptionMiddleware` -
    which includes every middleware. That is why `handle_unhandled_exception`
    checks for a `GatewayError` before treating a failure as a bug: it is the only
    handler positioned to render a middleware's typed raise, and an earlier change's
    `TracingMiddleware` made that position occupied.
    """
    app.add_exception_handler(GatewayError, handle_gateway_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unhandled_exception)
