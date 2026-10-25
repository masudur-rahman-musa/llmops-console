"""The typed failures the redaction engine, its map store and its directory raise.

None is a `GatewayError`. The nine
codes in `app.core.errors.ErrorCode` are a closed set and
a check asserts exactly one concrete `GatewayError`
subclass per code; `ErrorCode.VALIDATION` already belongs to
`app.core.validation.IdentifierInvalid` and `ErrorCode.PROVIDER_ERROR` to
`app.core.errors.ProviderError`, so a second class carrying either would
fail that test.
`PolicyInvalid` in the routing package is the same shape for the same reason.

The four here are `LibraryError` subclasses so each maps to its HTTP answer in
one place: a cap failure is a `LibraryInputInvalid` (422, `VALIDATION`), an
unreachable store a `LibraryDependencyUnavailable` (502, `PROVIDER_ERROR`), and
`app.core.http_errors.as_gateway_error` converts either.

The engine is a pure library with no HTTP boundary of its own. The request model turns an
oversized body into a 422 and imports
`MAX_REDACTION_INPUT_CHARS` rather than restating the number, and the
`/v1/redact` route turns an unreachable map store into a 502. The two
directory failures land the same way: a cap becomes a 422 on
`POST /admin/directory`, an unreachable directory store a 502 on the inline path.
"""

from __future__ import annotations

from app.core.errors import LibraryDependencyUnavailable, LibraryInputInvalid


class RedactionInputTooLong(LibraryInputInvalid):
    """Text longer than `MAX_REDACTION_INPUT_CHARS` was handed to `redact`.

    Raised before any detector runs, so an oversized input costs one length
    comparison rather than a full scan. Both numbers are
    attributes because the caller reporting the failure knows neither.
    """

    def __init__(self, length: int, limit: int) -> None:
        super().__init__(f"redaction input is {length} characters, over the {limit} limit")
        self.length = length
        self.limit = limit


class RedactionMapStoreUnavailable(LibraryDependencyUnavailable):
    """The map store could not be reached, on the read path or on the write path.

    Raised by `RedisRedactionMapStore` in place of every `redis.RedisError`, and
    always chained from the client's own error (`raise ... from exc`): that cause
    is the only description of WHY Redis was unreachable and it is what an
    operator's log line is built from.

    It stops at the route. `/v1/redact` converts it into
    `ProviderError` - 502, code `PROVIDER_ERROR` - with a message of the route's
    own words, so the client's text, which names an internal host and port, never
    reaches a caller.

    `get` raises this rather than answering `None`, because "no such map" and
    "the store is down" are different answers and rehydration would
    otherwise report a live map as missing.
    """


class DirectoryNameInvalid(LibraryInputInvalid):
    """A pushed directory name, or the size of the push, broke one of the three caps.

    One class for all three caps rather than three, because the caller acts on
    every one of them the same way: fix the push and send it again. `limit` says
    which cap was breached, and is an attribute because the route rendering the
    422 knows none of the numbers.

    The message names the field and the cap and NEVER the value. A directory name
    is a person's name, so a message built as `f"{name} is too short"` puts one
    into the caller's error body, a proxy log and whatever CI transcript captured
    the response - which is the precise thing this whole subsystem removes from
    text.

    Not a `GatewayError`, for `RedactionInputTooLong`'s reason: `ErrorCode.VALIDATION`
    belongs to `app.core.validation.IdentifierInvalid` and
    the error code set allows exactly one concrete class per code. The directory route
    converts it, carrying this message through unchanged.
    """

    def __init__(self, message: str, *, limit: int) -> None:
        super().__init__(message)
        self.limit = limit
        self.param = "names"


class DirectoryStoreUnavailable(LibraryDependencyUnavailable):
    """The directory store could not be reached, on the read path or on the write path.

    Raised by `RedisDirectoryStore` in place of every `redis.RedisError`, and
    always chained from the client's own error (`raise ... from exc`): that cause
    is the only description of WHY Redis was unreachable and it is what an
    operator's log line is built from.

    `get` raises this rather than answering an empty tuple, and that is a firm
    choice rather than a stylistic choice. Redaction is deny-by-default: a request served
    with an empty directory carries the tenant's names out to a provider while the
    response still reports `X-Gateway-Redacted: true`.

    It stops at the route. The chat and embeddings paths convert it into
    `ProviderError` - 502, code `PROVIDER_ERROR` - with words of their own, so the
    client's text, which names an internal host and port, never reaches a caller.
    """
