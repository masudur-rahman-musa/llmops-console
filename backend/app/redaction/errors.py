"""The typed failures the redaction engine, its map store and its directory raise.

Neither is a `GatewayError`. The nine
codes in `app.core.errors.ErrorCode` are a closed set and
`test_error_code_set_is_exhaustive` asserts exactly one concrete `GatewayError`
subclass per code; `ErrorCode.VALIDATION` already belongs to
`app.routing.errors.IdentifierInvalid` and `ErrorCode.PROVIDER_ERROR` to
`app.core.errors.ProviderError`, so a second class carrying either would
fail that test (`test_the_unavailable_failure_is_not_a_gateway_error`).
`PolicyInvalid` in the routing package is the same shape for the same reason.

The engine is a pure library with no HTTP boundary of its own. An earlier change turns an
oversized body into a 422 in its request model, which imports
`MAX_REDACTION_INPUT_CHARS` rather than restating the number, and an earlier change turns
an unreachable map store into a 502 in the `/v1/redact` route. An earlier change's two
directory failures land the same way: a cap becomes a 422 on
`POST /admin/directory`, an unreachable directory store a 502 on the inline path.
"""

from __future__ import annotations


class RedactionInputTooLong(Exception):
    """Text longer than `MAX_REDACTION_INPUT_CHARS` was handed to `redact`.

    Raised before any detector runs, so an oversized input costs one length
    comparison rather than a full scan; pinned by
    `test_the_cap_is_checked_before_any_detector_runs`. Both numbers are
    attributes because the caller reporting the failure knows neither.
    """

    def __init__(self, length: int, limit: int) -> None:
        super().__init__(f"redaction input is {length} characters, over the {limit} limit")
        self.length = length
        self.limit = limit


class RedactionMapStoreUnavailable(Exception):
    """The map store could not be reached, on the read path or on the write path.

    Raised by `RedisRedactionMapStore` in place of every `redis.RedisError`, and
    always chained from the client's own error (`raise ... from exc`): that cause
    is the only description of WHY Redis was unreachable and it is what an
    operator's log line is built from
    (`test_the_typed_failure_does_not_swallow_the_cause`).

    It stops at the route. `/v1/redact` converts it into
    `ProviderError` - 502, code `PROVIDER_ERROR` - with a message of the route's
    own words, so the client's text, which names an internal host and port, never
    reaches a caller
    (`test_a_store_that_cannot_be_reached_is_502_with_a_typed_code`,
    `test_a_store_failure_does_not_put_the_clients_words_in_front_of_a_caller`).

    `get` raises this rather than answering `None`, because "no such map" and
    "the store is down" are different answers and an earlier change's rehydration would
    otherwise report a live map as missing
    (`test_a_get_against_an_unreachable_redis_raises_the_typed_failure`).
    """


class DirectoryNameInvalid(Exception):
    """A pushed directory name, or the size of the push, broke one of the three caps.

    One class for all three caps rather than three, because the caller acts on
    every one of them the same way: fix the push and send it again. `limit` says
    which cap was breached, and is an attribute because the route rendering the
    422 knows none of the numbers.

    The message names the field and the cap and NEVER the value. A directory name
    is a person's name, so a message built as `f"{name} is too short"` puts one
    into the caller's error body, a proxy log and whatever CI transcript captured
    the response - which is the precise thing this whole subsystem removes from
    text (`test_a_rejection_names_the_field_and_the_cap_and_never_the_value`,
    `test_a_rejected_name_never_appears_in_the_error_body`).

    Not a `GatewayError`, for `RedactionInputTooLong`'s reason: `ErrorCode.VALIDATION`
    belongs to `app.routing.errors.IdentifierInvalid` and
    `test_error_code_set_is_exhaustive` allows exactly one concrete class per code
    (`test_the_cap_failure_is_not_a_gateway_error`). `gateway.routes.directory`
    converts it, carrying this message through unchanged.
    """

    def __init__(self, message: str, *, limit: int) -> None:
        super().__init__(message)
        self.limit = limit
        self.param = "names"


class DirectoryStoreUnavailable(Exception):
    """The directory store could not be reached, on the read path or on the write path.

    Raised by `RedisDirectoryStore` in place of every `redis.RedisError`, and
    always chained from the client's own error (`raise ... from exc`): that cause
    is the only description of WHY Redis was unreachable and it is what an
    operator's log line is built from
    (`test_the_typed_failure_does_not_swallow_the_cause`).

    `get` raises this rather than answering an empty tuple, and that is an earlier change
    decision H rather than a stylistic choice. Redaction is deny-by-default like
    residency: a request served with an empty directory carries the tenant's names
    out to a provider while the response still reports `X-Gateway-Redacted: true`
    (`test_an_unreachable_directory_store_is_502_with_no_outbound_call`).

    It stops at the route. The chat and embeddings paths convert it into
    `ProviderError` - 502, code `PROVIDER_ERROR` - with words of their own, so the
    client's text, which names an internal host and port, never reaches a caller
    (`test_a_directory_store_failure_does_not_put_the_clients_words_in_front_of_a_caller`).
    """
