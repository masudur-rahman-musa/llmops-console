"""The typed error set and the OpenAI-shaped body every failure returns.

The table below is the contract. It is written out literally rather than derived
from `ErrorCode`, so adding, renaming or dropping a member fails this file
instead of silently redefining what the gateway promises.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Imported for their side effect on `GatewayError.__subclasses__()`: a code whose
# class lives in a module nobody imported would otherwise look missing.
import app.api.deps
import app.core.http_errors
import app.model_routing.errors  # noqa: F401
from app.core.errors import ErrorCode, GatewayError
from app.core.http_errors import GENERIC_FAILURE_MESSAGE, error_type
from app.core.validation import IdentifierInvalid
from app.model_routing.errors import AliasUnknown, AllProvidersUnavailable

from .conftest import TEST_KEYS, assert_error_body, auth_headers

NORDLICHT_KEY = TEST_KEYS["nordlicht"]

# code string -> HTTP status. the design, an earlier change acceptance criterion 4.
ERROR_CONTRACT: dict[str, int] = {
    "UNAUTHORIZED": 401,
    "TENANT_MISMATCH": 403,
    "ALIAS_UNKNOWN": 404,
    "BUDGET_EXCEEDED": 429,
    "RATE_LIMITED": 429,
    "PROVIDER_ERROR": 502,
    "ALL_PROVIDERS_FAILED": 503,
    "VALIDATION": 422,
    "BUDGET_UNAVAILABLE": 503,
}


def concrete_error_classes() -> list[type[GatewayError]]:
    """Every shipped `GatewayError` subclass that carries a code, found depth-first.

    `__subclasses__()` is process-wide, so a throwaway subclass defined in
    another test module would otherwise count against the one-class-per-code
    assertion and fail this file for a change it has nothing to do with. Only
    classes from the `opsconsole` package are the contract.
    """
    found: list[type[GatewayError]] = []

    def walk(cls: type[GatewayError]) -> None:
        for subclass in cls.__subclasses__():
            if getattr(subclass, "code", None) is not None and subclass.__module__.startswith(
                "app."
            ):
                found.append(subclass)
            walk(subclass)

    walk(GatewayError)
    return found


def test_error_code_set_is_exhaustive() -> None:
    """The whole set, exactly: nine codes, nine strings, nine statuses.

    Both directions are asserted. A member added to `ErrorCode` fails the first
    assertion; a member renamed or dropped fails it too, because the comparison
    is over the value set and not over a subset check.
    """
    assert {member.value for member in ErrorCode} == set(ERROR_CONTRACT)
    assert len(list(ErrorCode)) == len(ERROR_CONTRACT)

    for name, status in ERROR_CONTRACT.items():
        member = ErrorCode(name)
        # A StrEnum: the member IS its wire string, so no `.value` at the boundary.
        assert member == name
        assert member.value == name
        assert isinstance(member, str)
        classes = [cls for cls in concrete_error_classes() if cls.code == member]
        assert len(classes) == 1, f"{name}: expected one error class, got {classes}"
        assert classes[0].http_status == status


@pytest.mark.parametrize(
    ("error_class", "code", "status"),
    [
        (AliasUnknown, "ALIAS_UNKNOWN", 404),
        (AllProvidersUnavailable, "ALL_PROVIDERS_FAILED", 503),
        (IdentifierInvalid, "VALIDATION", 422),
    ],
    ids=["alias-unknown", "all-providers", "validation"],
)
def test_routing_errors_are_gateway_errors(
    error_class: type[GatewayError], code: str, status: int
) -> None:
    """Re-parenting `RoutingError` keeps the codes an earlier change pinned (decision E)."""
    assert issubclass(error_class, GatewayError)
    assert error_class.code == code
    assert error_class.http_status == status


# status -> the OpenAI-style family the body reports. Written out literally for
# the same reason as ERROR_CONTRACT: it is what a client switches on.
ERROR_TYPE_CONTRACT: dict[int, str] = {
    401: "authentication_error",
    403: "permission_error",
    404: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    502: "api_error",
    503: "api_error",
}


def test_error_type_matches_the_status_family() -> None:
    """Every status the contract can return has exactly one family, pinned here."""
    assert set(ERROR_TYPE_CONTRACT) == set(ERROR_CONTRACT.values())

    for status, expected in ERROR_TYPE_CONTRACT.items():
        assert error_type(status) == expected


@pytest.mark.parametrize(
    ("url", "status", "expected_type"),
    [
        ("/v1/models", 401, "authentication_error"),
        ("/_harness/refused", 403, "permission_error"),
        ("/_harness/validated", 422, "invalid_request_error"),
    ],
    ids=["unauthorized", "refused", "validation"],
)
def test_error_type_reaches_the_wire(
    url: str, status: int, expected_type: str, harness_client: TestClient
) -> None:
    """The table is not just a function: it is what the body actually carries."""
    headers = auth_headers(key=NORDLICHT_KEY, tenant="nordlicht") if status != 401 else {}

    response = harness_client.get(url, headers=headers)

    assert response.status_code == status
    assert response.json()["error"]["type"] == expected_type


def test_a_refusal_is_403_openai_shaped(harness_client: TestClient) -> None:
    """A `GatewayError` raised in a route becomes its status and its code on the wire."""
    response = harness_client.get(
        "/_harness/refused", headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
    )

    assert_error_body(response, status=403, code="TENANT_MISMATCH")


# FastAPI's own validation failure must not escape as `{"detail": [...]}`.
VALIDATION_CASES: list[tuple[str, str, dict[str, str]]] = [
    (
        "a required query parameter is missing",
        "/_harness/validated",
        auth_headers(key=NORDLICHT_KEY, tenant="nordlicht"),
    ),
    (
        "a query parameter has the wrong type",
        "/_harness/validated?count=not-a-number",
        auth_headers(key=NORDLICHT_KEY, tenant="nordlicht"),
    ),
    (
        # `X-Tenant` is required by the dependency rather than by FastAPI (see
        # `context.request_context`), so this row proves the manual check answers
        # in the same shape as FastAPI's own. Delete that check and the row goes
        # red rather than the tenant quietly becoming `None`.
        "a required header is missing",
        "/_harness/context",
        auth_headers(key=NORDLICHT_KEY, tenant=None),
    ),
]


@pytest.mark.parametrize(
    ("case", "url", "headers"), VALIDATION_CASES, ids=[c[0] for c in VALIDATION_CASES]
)
def test_request_validation_uses_the_error_contract(
    case: str, url: str, headers: dict[str, str], harness_client: TestClient
) -> None:
    response = harness_client.get(url, headers=headers)

    assert_error_body(response, status=422, code="VALIDATION")
    assert "detail" not in response.json()


def test_validation_body_names_the_offending_field(harness_client: TestClient) -> None:
    """`param` is what an OpenAI-compatible client reads to point at the input."""
    response = harness_client.get(
        "/_harness/validated?count=not-a-number",
        headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht"),
    )

    assert response.status_code == 422
    assert response.json()["error"]["param"] == "count"


@pytest.mark.parametrize(
    ("case", "method", "url", "status"),
    [
        ("an unrouted path", "GET", "/v1/nope", 404),
        ("a method the route does not have", "POST", "/v1/models", 405),
    ],
    ids=["unrouted-path", "wrong-method"],
)
def test_starlette_failures_use_the_error_contract(
    case: str, method: str, url: str, status: int, harness_client: TestClient
) -> None:
    """Starlette raises these before any of our code runs, and they still keep the shape.

    `code` is `null`, not one of the nine: an unrouted URL is not a failure the
    gateway classifies, and bending `VALIDATION` onto a 404 would break the
    one-status-per-code table above.
    """
    response = harness_client.request(
        method, url, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
    )

    assert_error_body(response, status=status, code=None)
    assert "detail" not in response.json()
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_wrong_method_keeps_the_allow_header(harness_client: TestClient) -> None:
    """RFC 9110 makes `Allow` a MUST on a 405, and the handler rebuilds the response.

    Without the exception's headers being carried through, the status survives
    and the header does not, which no other assertion here would notice.
    """
    response = harness_client.post(
        "/v1/models", headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
    )

    assert response.status_code == 405
    assert response.headers["allow"] == "GET"


def test_non_string_http_detail_is_not_rendered(harness_client: TestClient) -> None:
    """A dict detail becomes the generic message, never a Python repr on the wire."""
    response = harness_client.get(
        "/_harness/http-detail", headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
    )

    assert_error_body(response, status=409, code=None)
    assert response.json()["error"]["message"] == GENERIC_FAILURE_MESSAGE
    assert "reason" not in response.text


def test_unhandled_exception_uses_the_error_contract(
    crashing_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A bug in a route is a 500 in the same shape, with the traceback only in the log.

    Starlette's own middleware would answer `text/plain Internal Server Error`,
    which a client reading `error.message` cannot parse at the moment it most
    needs to.
    """
    with caplog.at_level(logging.ERROR):
        response = crashing_client.get(
            "/_harness/boom", headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
        )

    assert_error_body(response, status=500, code=None)
    assert response.json()["error"]["type"] == "api_error"
    assert response.json()["error"]["message"] == GENERIC_FAILURE_MESSAGE
    assert "harness: a bug in a route" not in response.text
    logged = [record for record in caplog.records if record.exc_info is not None]
    assert len(logged) == 1
    assert "/_harness/boom" in logged[0].getMessage()


def test_error_body_is_json_and_carries_no_traceback(harness_client: TestClient) -> None:
    """The message is a sentence for a client, not an internal detail."""
    response = harness_client.get(
        "/_harness/refused", headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
    )
    body: dict[str, Any] = response.json()

    assert response.headers["content-type"].startswith("application/json")
    assert "Traceback" not in body["error"]["message"]
