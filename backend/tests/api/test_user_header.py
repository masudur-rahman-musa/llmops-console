"""`X-User`: the random user id that rides on each request.

The id is `u_` and exactly 32 lowercase hex characters, never a name or an email. It is
checked up front, after the key, and a request without it is still served and counted
under no user. Everything is settled through `TestClient` against apps `create_app`
built: the request context is read through `/_harness/context`, the cost row from the
in-memory ledger, the trace from the in-memory span exporter and the metrics from the
`/metrics` exposition.

The contract these cases are written against, before it exists:

* `RequestContext.user_id: str | None` (default `None`);
* `CostLedgerRow.user_id: str | None` (default `None`);
* `SpanAttributes.user_id`, sent on the wire as `langfuse.user.id`;
* the header is `X-User`, a 422 `VALIDATION` with `param` `X-User` on any other shape.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.storage import InMemoryCostLedgerWriter

from .conftest import (
    CHAT,
    TEST_KEYS,
    StubAdapter,
    app_of,
    assert_error_body,
    auth_headers,
    chat_body,
)

HEADER = "X-User"
USER_ID = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"
USER_WIRE_KEY = "langfuse.user.id"
CONTEXT = "/_harness/context"
METRICS = "/metrics"
TENANT = "evals"
KEY = TEST_KEYS[TENANT]


@pytest.fixture
def ledger(chat_client: TestClient) -> Iterator[InMemoryCostLedgerWriter]:
    """A fresh in-memory ledger on the shared chat app for one case, then the old one back."""
    app = app_of(chat_client)
    previous = app.state.cost_ledger
    writer = InMemoryCostLedgerWriter()
    app.state.cost_ledger = writer
    try:
        yield writer
    finally:
        app.state.cost_ledger = previous


def headers_with(user: str | None, *, key: str = KEY) -> dict[str, str]:
    """The `evals`/`internal` headers, with `X-User` only when a value is given."""
    headers = auth_headers(key=key, tenant=TENANT, data_class="internal")
    if user is not None:
        headers[HEADER] = user
    return headers


def post(client: TestClient, user: str | None) -> Response:
    """One chat completion as `evals`, with the user header when given."""
    response: Response = client.post(CHAT, json=chat_body(), headers=headers_with(user))
    return response


def test_a_good_user_id_is_carried(
    harness_client: TestClient, chat_client: TestClient, ledger: InMemoryCostLedgerWriter
) -> None:
    """Catches an implementation that reads the header and drops it before the context
    or before `record_cost` (both stay `None`), or one that rewrites it. The id is exactly
    32 hex, the width that must pass; one under and one over are refused below."""
    seen = harness_client.get(CONTEXT, headers=headers_with(USER_ID))
    served = post(chat_client, USER_ID)

    assert seen.status_code == 200, seen.text
    assert seen.json()["user_id"] == USER_ID
    assert served.status_code == 200, served.text
    assert [row.user_id for row in ledger.rows] == [USER_ID]


def test_no_user_header_means_no_user(
    harness_client: TestClient, chat_client: TestClient, ledger: InMemoryCostLedgerWriter
) -> None:
    """Catches an implementation that makes the header required (a 422 for every old
    client), or books an empty string or a placeholder id for an anonymous request, which
    the console would then rank as a user."""
    seen = harness_client.get(CONTEXT, headers=headers_with(None))
    served = post(chat_client, None)

    assert seen.status_code == 200, seen.text
    assert seen.json()["user_id"] is None
    assert served.status_code == 200, served.text
    assert [row.user_id for row in ledger.rows] == [None]


USER_ID_REJECTIONS: list[tuple[str, str]] = [
    ("31 hex", USER_ID[:-1]),
    ("33 hex", USER_ID + "a"),
    ("upper-case hex", "u_" + USER_ID[2:].upper()),
    ("an email", "alice@example.com"),
    ("a name", "Alice"),
    ("no prefix", USER_ID[2:]),
    ("upper-case prefix", "U_" + USER_ID[2:]),
    ("a different prefix", "x_" + USER_ID[2:]),
    ("non-hex letters", "u_" + "g" * 32),
    ("a trailing newline", USER_ID + "\n"),
    ("a trailing CRLF", USER_ID + "\r\n"),
    ("a trailing space", USER_ID + " "),
    ("a leading space", " " + USER_ID),
    ("empty", ""),
]


@pytest.mark.parametrize(
    ("case", "value"), USER_ID_REJECTIONS, ids=[row[0] for row in USER_ID_REJECTIONS]
)
def test_user_id_rejection_table(
    case: str,
    value: str,
    chat_client: TestClient,
    chat_stub: StubAdapter,
    ledger: InMemoryCostLedgerWriter,
) -> None:
    """Catches a check looser than the shape: `.match` with `$` (the newline row), a
    pattern with no width (31 and 33 hex), one that lowercases instead of refusing, and an
    echo of the rejected value in the error body, which would copy a name or an email into
    a response. Each is a 422 `VALIDATION` naming `X-User`, with no provider call and no
    cost row."""
    calls_before = len(chat_stub.requests)

    response = post(chat_client, value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == HEADER
    assert len(chat_stub.requests) == calls_before
    assert ledger.rows == ()
    echoed = value.strip()
    if echoed:
        assert echoed not in response.text


def test_the_user_id_check_runs_after_the_key(chat_client: TestClient) -> None:
    """Catches a check that runs before authentication: a caller with no usable key and a
    bad `X-User` must be told 401 and nothing about its other headers, exactly as for the
    other optional headers. A bad key is the only difference from the rejection table."""
    response = chat_client.post(
        CHAT, json=chat_body(), headers=headers_with("alice@example.com", key="not-a-key")
    )

    assert_error_body(response, status=401, code="UNAUTHORIZED")


def test_the_user_id_goes_out_as_langfuse_user_id(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a user id sent under the bare `user_id` key (Langfuse's Users page stays
    empty), one sent under both keys, and one put on a child span only. The root span
    carries `langfuse.user.id` equal to the header."""
    response = post(chat_client, USER_ID)
    assert response.status_code == 200, response.text

    recorded = tuple(spans.get_finished_spans())
    roots = [s for s in recorded if s.parent is None or s.parent.is_remote]
    assert len(roots) == 1
    assert dict(roots[0].attributes or {})[USER_WIRE_KEY] == USER_ID
    assert "user_id" not in {key for span in recorded for key in (span.attributes or {})}


def test_no_user_header_puts_no_user_on_the_trace(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an anonymous request traced under an empty or placeholder user id, which
    Langfuse would list as one real user holding all the anonymous traffic."""
    response = post(chat_client, None)
    assert response.status_code == 200, response.text

    recorded = tuple(spans.get_finished_spans())
    assert recorded, "the request emitted no span at all"
    assert USER_WIRE_KEY not in {key for span in recorded for key in (span.attributes or {})}


def test_the_user_id_is_on_no_metric_label(chat_client: TestClient) -> None:
    """Catches a user id used as a Prometheus label, which would mint one time series per
    user. The control is that the served request did reach the exposition."""
    response = post(chat_client, USER_ID)
    assert response.status_code == 200, response.text

    exposition = chat_client.get(METRICS)

    assert exposition.status_code == 200
    assert "gateway_requests_total" in exposition.text
    assert USER_ID not in exposition.text
