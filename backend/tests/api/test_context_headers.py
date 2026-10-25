"""The per-request headers: data class, session and traceparent.

Observed through `/_harness/context`, which returns the `RequestContext` the
dependency built (see the conftest docstring for why that route exists). The
caps are asserted against the imported `MAX_IDENTIFIER_LENGTH`, never a retyped
64, so a change to the constant moves the boundary rows with it.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.model_routing.policy import DataClass

from .conftest import MAX_TENANT, TEST_KEYS, assert_error_body, auth_headers

NORDLICHT_KEY = TEST_KEYS["nordlicht"]
MAX_TENANT_KEY = TEST_KEYS[MAX_TENANT]
CONTEXT = "/_harness/context"

TRACE_ID = re.compile(r"[0-9a-f]{32}")
ALL_ZERO_TRACE_ID = "0" * 32


def context_of(client: TestClient, **kwargs: str | None) -> dict[str, object]:
    """Fetch the request context the gateway built for one set of headers."""
    response = client.get(
        CONTEXT, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht", **kwargs)
    )
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


# --- X-Data-Class ------------------------------------------------------------


def test_data_class_defaults_to_confidential(harness_client: TestClient) -> None:
    """Absent means the most restrictive routing rule, not the most permissive."""
    context = context_of(harness_client)

    assert context["data_class"] == DataClass.CONFIDENTIAL.value
    assert context["data_class"] == "confidential"


@pytest.mark.parametrize(
    "data_class", [member.value for member in DataClass], ids=[m.value for m in DataClass]
)
def test_every_known_data_class_is_carried(data_class: str, harness_client: TestClient) -> None:
    context = context_of(harness_client, data_class=data_class)

    assert context["data_class"] == data_class


# An unknown value is a rejection, never a silent downgrade to the default. The
# whitespace rows are a project rule: `confidential\n` is not `confidential`.
UNKNOWN_DATA_CLASS_CASES: list[tuple[str, str]] = [
    ("a class that does not exist", "secret"),
    ("empty", ""),
    ("wrong case", "CONFIDENTIAL"),
    ("trailing newline", "confidential\n"),
    ("trailing CRLF", "confidential\r\n"),
    ("trailing space", "confidential "),
]


@pytest.mark.parametrize(
    ("case", "data_class"),
    UNKNOWN_DATA_CLASS_CASES,
    ids=[c[0] for c in UNKNOWN_DATA_CLASS_CASES],
)
def test_unknown_data_class_is_422(case: str, data_class: str, harness_client: TestClient) -> None:
    response = harness_client.get(
        CONTEXT,
        headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht", data_class=data_class),
    )

    assert_error_body(response, status=422, code="VALIDATION")


# --- X-Session ---------------------------------------------------------------


def test_session_is_optional_and_absent_means_none(harness_client: TestClient) -> None:
    context = context_of(harness_client)

    assert context["session"] is None


def test_session_header_boundary(harness_client: TestClient) -> None:
    """Inclusive at `MAX_IDENTIFIER_LENGTH`; one character over is rejected."""
    at_cap = "s" * MAX_IDENTIFIER_LENGTH
    over_cap = "s" * (MAX_IDENTIFIER_LENGTH + 1)

    context = context_of(harness_client, session=at_cap)
    rejected = harness_client.get(
        CONTEXT, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht", session=over_cap)
    )

    assert context["session"] == at_cap
    assert len(at_cap) == MAX_IDENTIFIER_LENGTH
    assert_error_body(rejected, status=422, code="VALIDATION")


SESSION_REJECTION_CASES: list[tuple[str, str]] = [
    ("one over the cap", "s" * (MAX_IDENTIFIER_LENGTH + 1)),
    ("empty", ""),
    ("trailing newline", "session-1\n"),
    ("trailing CRLF", "session-1\r\n"),
    ("trailing space", "session-1 "),
    ("leading space", " session-1"),
    ("uppercase", "Session-1"),
    ("a slash", "session/1"),
    ("a null byte escape", "session-1%00"),
]


@pytest.mark.parametrize(
    ("case", "session"), SESSION_REJECTION_CASES, ids=[c[0] for c in SESSION_REJECTION_CASES]
)
def test_session_rejection_table(case: str, session: str, harness_client: TestClient) -> None:
    """`X-Session` becomes a key and a metric label, so it is slug-checked."""
    response = harness_client.get(
        CONTEXT, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht", session=session)
    )

    assert_error_body(response, status=422, code="VALIDATION")


# --- X-Tenant ----------------------------------------------------------------


def test_tenant_header_boundary(harness_client: TestClient) -> None:
    """A tenant of exactly the cap is accepted; one over is rejected.

    The at-cap tenant has a key of its own, so the accepted row proves the
    pre-check passed rather than that auth stopped first.
    """
    assert len(MAX_TENANT) == MAX_IDENTIFIER_LENGTH
    over_cap = "a" * (MAX_IDENTIFIER_LENGTH + 1)

    accepted = harness_client.get(
        CONTEXT, headers=auth_headers(key=MAX_TENANT_KEY, tenant=MAX_TENANT)
    )
    rejected = harness_client.get(
        CONTEXT, headers=auth_headers(key=MAX_TENANT_KEY, tenant=over_cap)
    )

    assert accepted.status_code == 200
    assert accepted.json()["tenant"] == MAX_TENANT
    assert_error_body(rejected, status=422, code="VALIDATION")


TENANT_REJECTION_CASES: list[tuple[str, str]] = [
    ("one over the cap", "a" * (MAX_IDENTIFIER_LENGTH + 1)),
    ("empty", ""),
    ("trailing newline", "nordlicht\n"),
    ("trailing CRLF", "nordlicht\r\n"),
    ("trailing space", "nordlicht "),
    ("uppercase", "Nordlicht"),
    ("a path segment", "nordlicht/admin"),
]


@pytest.mark.parametrize(
    ("case", "tenant"), TENANT_REJECTION_CASES, ids=[c[0] for c in TENANT_REJECTION_CASES]
)
def test_tenant_rejection_table(case: str, tenant: str, harness_client: TestClient) -> None:
    """A valid key does not save a malformed tenant: the header is checked too."""
    response = harness_client.get(CONTEXT, headers=auth_headers(key=NORDLICHT_KEY, tenant=tenant))

    assert_error_body(response, status=422, code="VALIDATION")


# --- traceparent -------------------------------------------------------------

VALID_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
VALID_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"


def test_valid_traceparent_is_carried(harness_client: TestClient) -> None:
    """A W3C header from upstream keeps its trace id, so one trace spans both."""
    context = context_of(harness_client, traceparent=VALID_TRACEPARENT)

    assert context["trace_id"] == VALID_TRACE_ID


def test_absent_traceparent_generates_one(harness_client: TestClient) -> None:
    """No header means a new, non-zero, 32-hex trace id, and a fresh one each time."""
    first = context_of(harness_client)
    second = context_of(harness_client)

    trace_id = first["trace_id"]
    assert isinstance(trace_id, str)
    assert TRACE_ID.fullmatch(trace_id) is not None
    assert trace_id != ALL_ZERO_TRACE_ID
    assert first["trace_id"] != second["trace_id"]


# Decision C: a malformed traceparent is ignored and a new trace started, never
# a 422. The rows cover each part of the W3C format independently.
MALFORMED_TRACEPARENT_CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("not a traceparent at all", "garbage"),
    ("too few fields", "00-4bf92f3577b34da6a3ce929d0e0e4736"),
    ("too many fields", VALID_TRACEPARENT + "-extra"),
    ("unsupported version ff", "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
    ("trace id too short", "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01"),
    ("trace id not hex", "00-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-00f067aa0ba902b7-01"),
    ("all-zero trace id", f"00-{ALL_ZERO_TRACE_ID}-00f067aa0ba902b7-01"),
    ("all-zero span id", f"00-{VALID_TRACE_ID}-{'0' * 16}-01"),
    ("uppercase trace id", "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01"),
    ("trailing newline", VALID_TRACEPARENT + "\n"),
    ("trailing CRLF", VALID_TRACEPARENT + "\r\n"),
    ("trailing space", VALID_TRACEPARENT + " "),
]


@pytest.mark.parametrize(
    ("case", "traceparent"),
    MALFORMED_TRACEPARENT_CASES,
    ids=[c[0] for c in MALFORMED_TRACEPARENT_CASES],
)
def test_malformed_traceparent_generates_one(
    case: str, traceparent: str, harness_client: TestClient
) -> None:
    """A broken upstream header never fails an otherwise valid request."""
    context = context_of(harness_client, traceparent=traceparent)

    trace_id = context["trace_id"]
    assert isinstance(trace_id, str)
    assert TRACE_ID.fullmatch(trace_id) is not None
    assert trace_id != ALL_ZERO_TRACE_ID
    assert trace_id != VALID_TRACE_ID


def test_context_carries_the_authenticated_tenant(harness_client: TestClient) -> None:
    """The context's tenant is the one the key proved, and the shape is complete."""
    context = context_of(harness_client, session="session-1", data_class="internal")

    assert set(context) == {"tenant", "data_class", "session", "trace_id", "prompt_version"}
    assert context["tenant"] == "nordlicht"
    assert context["session"] == "session-1"
    assert context["data_class"] == "internal"
