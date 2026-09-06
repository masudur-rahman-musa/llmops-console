"""The typed failure the redaction engine raises.

`RedactionInputTooLong` is a plain `Exception`, not a `GatewayError`. The nine
codes in `opsconsole.core.errors.ErrorCode` are a closed set and
`test_error_code_set_is_exhaustive` asserts exactly one concrete `GatewayError`
subclass per code; `ErrorCode.VALIDATION` already belongs to
`opsconsole.routing.errors.IdentifierInvalid`, so a second class carrying it
would fail that test. `PolicyInvalid` in the routing package is the same shape
for the same reason.

The engine is a pure library with no HTTP boundary of its own. An earlier change turns an
oversized body into a 422 in its request model, which imports
`MAX_REDACTION_INPUT_CHARS` rather than restating the number.
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
