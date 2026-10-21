"""The second error hierarchy and the one converter that turns it into a wire error.

`LibraryError` is a base SEPARATE from `GatewayError`: a library raises it with no
HTTP status on the wire, and only a route that catches it decides the status by
calling `as_gateway_error`. These tests pin the two middle bases' codes and
statuses, that the base is not a `GatewayError` (so the one-class-per-code wire
contract in `test_error_contract.py` stays exhaustive), that every re-parented
library failure sits under the new base, and that the converter maps each kind to
the right wire error without carrying the library's own words through.
"""

from __future__ import annotations

from app.core.errors import (
    ErrorCode,
    GatewayError,
    LibraryDependencyUnavailable,
    LibraryError,
    LibraryInputInvalid,
    ProviderError,
)
from app.core.http_errors import as_gateway_error
from app.core.validation import IdentifierInvalid
from app.evals.calibration import CalibrationRefused, CalibrationSetInvalid
from app.evals.conversations import ConversationSetInvalid
from app.evals.gate import GateInputInvalid
from app.limits.errors import PriceTableInvalid
from app.redaction.errors import (
    DirectoryNameInvalid,
    DirectoryStoreUnavailable,
    RedactionInputTooLong,
    RedactionMapStoreUnavailable,
)

# Built explicitly, not by scanning the filesystem: this is the exact set of ten
# library failures the plan re-parents, and a drift in either direction is a
# finding this list must be edited to show.
LIBRARY_FAILURES: list[type[LibraryError]] = [
    RedactionInputTooLong,
    RedactionMapStoreUnavailable,
    DirectoryNameInvalid,
    DirectoryStoreUnavailable,
    PriceTableInvalid,
    ConversationSetInvalid,
    GateInputInvalid,
    CalibrationSetInvalid,
    CalibrationRefused,
]


def test_input_invalid_names_validation_422() -> None:
    """Catches an input-kind base left without a code or set to a non-422 status."""
    assert LibraryInputInvalid.code is ErrorCode.VALIDATION
    assert LibraryInputInvalid.http_status == 422


def test_dependency_unavailable_names_provider_error_502() -> None:
    """Catches a store-kind base left without a code or set to a non-502 status."""
    assert LibraryDependencyUnavailable.code is ErrorCode.PROVIDER_ERROR
    assert LibraryDependencyUnavailable.http_status == 502


def test_the_library_base_is_not_a_gateway_error() -> None:
    """Catches `LibraryError` made a `GatewayError`, which would add codes to the wire set."""
    assert not issubclass(LibraryError, GatewayError)


def test_every_library_failure_shares_the_base() -> None:
    """Catches a library failure left a `GatewayError` instead of re-parented to `LibraryError`."""
    for cls in LIBRARY_FAILURES:
        assert issubclass(cls, LibraryError), cls
        assert not issubclass(cls, GatewayError), cls


def test_the_converter_maps_an_input_invalid_to_validation() -> None:
    """Catches a converter that drops `param` or returns the wrong wire error for input-kind."""
    result = as_gateway_error(RedactionInputTooLong(5, 4), "field: too long", param="field")

    assert isinstance(result, IdentifierInvalid)
    assert result.code is ErrorCode.VALIDATION
    assert result.http_status == 422
    assert result.param == "field"


def test_the_converter_maps_a_dependency_outage_to_a_retryable_502() -> None:
    """Catches a converter that carries the library's own words through, leaking an internal host.

    The result's message must be EXACTLY the caller's string, not the library
    exception's text, so an internal host or port never reaches the wire.
    """
    result = as_gateway_error(RedactionMapStoreUnavailable(), "the store is unavailable")

    assert isinstance(result, ProviderError)
    assert result.code is ErrorCode.PROVIDER_ERROR
    assert result.http_status == 502
    assert result.retryable is True
    assert result.message == "the store is unavailable"
