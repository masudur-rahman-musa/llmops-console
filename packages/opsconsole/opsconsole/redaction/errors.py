"""The typed failures the redaction engine and its map store raise.

Neither is a `GatewayError`. The nine
codes in `opsconsole.core.errors.ErrorCode` are a closed set and
`test_error_code_set_is_exhaustive` asserts exactly one concrete `GatewayError`
subclass per code; `ErrorCode.VALIDATION` already belongs to
`opsconsole.routing.errors.IdentifierInvalid` and `ErrorCode.PROVIDER_ERROR` to
`opsconsole.core.errors.ProviderError`, so a second class carrying either would
fail that test (`test_the_unavailable_failure_is_not_a_gateway_error`).
`PolicyInvalid` in the routing package is the same shape for the same reason.

The engine is a pure library with no HTTP boundary of its own. An earlier change turns an
oversized body into a 422 in its request model, which imports
`MAX_REDACTION_INPUT_CHARS` rather than restating the number, and an earlier change turns
an unreachable map store into a 502 in the `/v1/redact` route.
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
