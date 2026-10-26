"""`GET /console/requests`: what the gateway hides, tokens, hit rate and the newest requests.

No backing service. Each case builds the admin app from `create_app` and then replaces the
two sources on `app.state`, as tests/api/test_console_routes.py does: `console_reader`
becomes `StubReader` below and `prometheus` becomes a real `PrometheusClient` over an
`httpx.MockTransport` that records every query. The route, its dependencies and the error
handlers stay real.

The contract these cases are written against, before it exists:

* `app.api.routes.dashboard.CONSOLE_REQUESTS == "/console/requests"`, admin key, `days`
  1 to 90 with a default of 30;
* `app.services.prometheus_reads`: `redactions_query(days)`, `tokens_query(days)`,
  `cache_hit_rate_query(days)`, `refusals_query(days)`, `guardrail_blocks_query(days)`;
* `ConsoleReader.recent_requests() -> tuple[RequestRow, ...]`, newest first, with
  `trace_url` None: the route fills `trace_url` from `Settings.langfuse_project_url` and the
  row's `trace_id`, and only when the setting is set and the id is 32 lowercase hex.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.routes import dashboard as console
from app.core.config import Settings
from app.main import create_app
from app.schemas.dashboard import RequestRow
from app.services.prometheus_reads import (
    PrometheusClient,
    cache_hit_rate_query,
    guardrail_blocks_query,
    redactions_query,
    refusals_query,
    tokens_query,
)
from app.storage.dashboard_reads import RequestsRead

from .conftest import ADMIN_KEY, NO_PROVIDER_KEYS, TEST_KEYS, assert_error_body

REQUESTS = "/console/requests"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
PROJECT_URL = "https://cloud.langfuse.com/project/demo"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
OTHER_TRACE_ID = "b7ad6b7169203331b7ad6b7169203331"
SAMPLE_TS = 1757980800

# The five queries for a 7 day window, written out so no case reads its expectation from the
# module under test.
REDACTIONS_7D = "sum by (kind) (increase(gateway_redactions_total[7d]))"
TOKENS_7D = "sum by (model, direction) (increase(gateway_tokens_total[7d]))"
HIT_RATE_7D = (
    'sum(increase(gateway_requests_total{cache=~"hit|semantic"}[7d]))'
    " / sum(increase(gateway_requests_total[7d]))"
)
REFUSALS_7D = "sum by (reason) (increase(helpdesk_refusals_total[7d]))"
BLOCKS_7D = "sum(increase(helpdesk_sod_blocks_total[7d]))"
ALL_7D = {REDACTIONS_7D, TOKENS_7D, HIT_RATE_7D, REFUSALS_7D, BLOCKS_7D}


def a_row(trace_id: str, **overrides: Any) -> RequestRow:  # noqa: ANN401
    """One ledger read model as the reader answers it: `trace_url` is the route's to fill."""
    fields: dict[str, Any] = {
        "ts": datetime(2026, 10, 26, 8, 30, tzinfo=UTC),
        "tenant": "nordlicht",
        "alias": "agent-default",
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "input_tokens": 137,
        "output_tokens": 29,
        "cached": False,
        "cost_usd": 0.000028,
        "prompt_version": "9f2c4be07a1d",
        "trace_id": trace_id,
        "trace_url": None,
    }
    fields.update(overrides)
    return RequestRow(**fields)


ROWS = (
    a_row(TRACE_ID),
    a_row(
        OTHER_TRACE_ID,
        ts=datetime(2026, 10, 26, 8, 12, tzinfo=UTC),
        tenant="cuan",
        cached=True,
        cost_usd=0.0,
        prompt_version=None,
    ),
)


class StubReader:
    """A configured `ConsoleReader` that records calls and answers the rows above.

    Only `recent_requests` is read by this route. `error`, when set, is raised instead,
    which is how the Postgres failure case arranges a failing read without a database.
    """

    def __init__(self, rows: tuple[RequestRow, ...] = ROWS) -> None:
        self.configured = True
        self.rows = rows
        self.calls: list[str] = []
        self.pages_asked: list[int] = []
        self.error: Exception | None = None
        self.answer_page = 1
        self.answer_pages = 1

    def recent_requests(self, *, page: int) -> RequestsRead:
        self.calls.append("recent_requests")
        self.pages_asked.append(page)
        if self.error is not None:
            raise self.error
        return RequestsRead(rows=self.rows, page=self.answer_page, pages=self.answer_pages)


def series(labels: dict[str, str], value: str) -> dict[str, Any]:
    return {"metric": labels, "value": [SAMPLE_TS, value]}


ANSWERS: dict[str, list[dict[str, Any]]] = {
    REDACTIONS_7D: [series({"kind": "email"}, "41"), series({"kind": "iban"}, "7")],
    TOKENS_7D: [
        series({"model": "claude-haiku-5-5", "direction": "in"}, "120450"),
        series({"model": "claude-haiku-5-5", "direction": "out"}, "30210"),
        series({"model": "gpt-5-mini", "direction": "in"}, "8800"),
        series({"model": "gpt-5-mini", "direction": "out"}, "2100"),
    ],
    HIT_RATE_7D: [series({}, "0.3125")],
    REFUSALS_7D: [
        series({"reason": "out_of_scope"}, "12"),
        series({"reason": "needs_approval"}, "3"),
    ],
    BLOCKS_7D: [series({}, "5")],
}


class PromStub:
    """A mock Prometheus that answers `ANSWERS` by query text and records every query."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.queries: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("query", "")
        self.queries.append(query)
        if self.mode == "500":
            return httpx.Response(500, text="internal error")
        result = [] if self.mode == "empty" else ANSWERS.get(query, [])
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "vector", "result": result}}
        )

    def client(self) -> PrometheusClient:
        return PrometheusClient(
            httpx.Client(
                base_url="http://prometheus.test", transport=httpx.MockTransport(self.handle)
            )
        )


class Wired:
    """One admin app with both sources replaced by the recording stubs."""

    def __init__(self, client: TestClient, reader: StubReader, prom: PromStub) -> None:
        self.client = client
        self.reader = reader
        self.prom = prom

    def get(self, **params: str | int) -> httpx.Response:
        response: httpx.Response = self.client.get(REQUESTS, params=params, headers=ADMIN_HEADERS)
        return response

    def body(self, **params: str | int) -> dict[str, Any]:
        response = self.get(**params)
        assert response.status_code == 200, response.text
        parsed: dict[str, Any] = response.json()
        return parsed


@pytest.fixture
def make_wired(admin_settings: Settings) -> Iterator[Callable[[str | None], Wired]]:
    """A builder for the wired app with a chosen Langfuse project URL; closes every app."""
    entered: list[TestClient] = []

    def build(project_url: str | None) -> Wired:
        settings = admin_settings.model_copy(update={"langfuse_project_url": project_url})
        app = create_app(settings, NO_PROVIDER_KEYS)
        reader = StubReader()
        prom = PromStub()
        client = TestClient(app)
        client.__enter__()
        entered.append(client)
        # After the lifespan is entered, where the real reader and client are built.
        app.state.console_reader = reader
        app.state.prometheus = prom.client()
        return Wired(client, reader, prom)

    yield build
    for client in entered:
        client.__exit__(None, None, None)


@pytest.fixture
def wired(make_wired: Callable[[str | None], Wired]) -> Wired:
    """The wired app with the project URL set."""
    return make_wired(PROJECT_URL)


def test_requests_queries_are_pinned(wired: Wired) -> None:
    """Catches a changed metric name, a missing `sum by`, a hit rate that counts only `hit`
    (or any cache label) instead of `hit|semantic`, a window that ignores `days`, and a route
    that asks for more or fewer queries than the five: the sent set must equal the literals."""
    assert console.CONSOLE_REQUESTS == REQUESTS
    assert redactions_query(7) == REDACTIONS_7D
    assert tokens_query(7) == TOKENS_7D
    assert cache_hit_rate_query(7) == HIT_RATE_7D
    assert refusals_query(7) == REFUSALS_7D
    assert guardrail_blocks_query(7) == BLOCKS_7D
    for query in ALL_7D:
        assert "[7d]" in query
    assert 'cache=~"hit|semantic"' in HIT_RATE_7D

    wired.body(days=7)

    assert set(wired.prom.queries) == ALL_7D
    assert len(wired.prom.queries) == 5


def test_requests_body_maps_counts(wired: Wired) -> None:
    """Catches a body that swaps in and out, drops a model, a kind or a reason, reads the
    help desk's two counters from each other's series, or turns the hit rate into a percent
    (the page does that, the body carries the ratio)."""
    body = wired.body(days=7)

    assert set(body) == {
        "sources",
        "redactions",
        "refusals",
        "guardrail_blocks",
        "tokens",
        "cache_hit_rate",
        "requests",
        "requests_page",
        "requests_pages",
    }
    assert body["sources"] == {"postgres": "ok", "prometheus": "ok"}
    assert sorted(body["redactions"], key=lambda r: r["kind"]) == [
        {"kind": "email", "count": 41},
        {"kind": "iban", "count": 7},
    ]
    assert sorted(body["refusals"], key=lambda r: r["reason"]) == [
        {"reason": "needs_approval", "count": 3},
        {"reason": "out_of_scope", "count": 12},
    ]
    assert body["guardrail_blocks"] == 5
    assert sorted(body["tokens"], key=lambda t: t["model"]) == [
        {"model": "claude-haiku-5-5", "input": 120450, "output": 30210},
        {"model": "gpt-5-mini", "input": 8800, "output": 2100},
    ]
    assert body["cache_hit_rate"] == 0.3125
    assert [(row["tenant"], row["prompt_version"]) for row in body["requests"]] == [
        ("nordlicht", "9f2c4be07a1d"),
        ("cuan", None),
    ]
    assert wired.reader.calls == ["recent_requests"]
    assert wired.reader.pages_asked == [1]


def test_no_traffic_gives_null_rate_and_blocks(wired: Wired) -> None:
    """Catches a body that answers 0 for a hit rate or a block count when Prometheus has no
    series at all (no traffic is not a 0% hit rate), or fails on the empty result."""
    wired.prom.mode = "empty"

    body = wired.body()

    assert body["sources"] == {"postgres": "ok", "prometheus": "ok"}
    assert body["cache_hit_rate"] is None
    assert body["guardrail_blocks"] is None
    assert body["redactions"] == []
    assert body["refusals"] == []
    assert body["tokens"] == []


def test_the_default_window_is_30_days(wired: Wired) -> None:
    """Catches a default other than 30 days: the queries sent with no `days` must be the 30
    day forms of the five builders."""
    wired.body()

    assert set(wired.prom.queries) == {
        redactions_query(30),
        tokens_query(30),
        cache_hit_rate_query(30),
        refusals_query(30),
        guardrail_blocks_query(30),
    }
    assert all("[30d]" in query for query in wired.prom.queries)


def test_trace_url_is_built_from_the_project_url(wired: Wired) -> None:
    """Catches a link built from the wrong part (the tenant, the session), a missing
    `/traces/` segment, a doubled slash, or one row's id used for every row."""
    rows = wired.body()["requests"]

    assert [row["trace_url"] for row in rows] == [
        f"https://cloud.langfuse.com/project/demo/traces/{TRACE_ID}",
        f"https://cloud.langfuse.com/project/demo/traces/{OTHER_TRACE_ID}",
    ]
    assert [row["trace_id"] for row in rows] == [TRACE_ID, OTHER_TRACE_ID]


BAD_TRACE_IDS: list[tuple[str, str]] = [
    ("31 characters", TRACE_ID[:-1]),
    ("33 characters", TRACE_ID + "a"),
    ("uppercase", TRACE_ID.upper()),
    ("trailing newline", TRACE_ID + "\n"),
    ("trailing CRLF", TRACE_ID + "\r\n"),
    ("trailing space", TRACE_ID + " "),
    ("a path break", "../" + TRACE_ID[3:]),
]


@pytest.mark.parametrize(("case", "trace_id"), BAD_TRACE_IDS, ids=[r[0] for r in BAD_TRACE_IDS])
def test_bad_trace_id_gets_no_link(case: str, trace_id: str, wired: Wired) -> None:
    """Catches a link built from any id the reader hands over (a trace id is joined into a
    URL segment), a length check without the hex check, and a `$`-anchored pattern that lets
    a trailing newline through. The project URL is set and the row is otherwise valid, so
    only the id check can null the link; the good row beside it still links."""
    wired.reader.rows = (a_row(trace_id), a_row(TRACE_ID))

    rows = wired.body()["requests"]

    assert rows[0]["trace_url"] is None
    assert rows[1]["trace_url"] == f"{PROJECT_URL}/traces/{TRACE_ID}"


def test_no_project_url_gets_no_link(make_wired: Callable[[str | None], Wired]) -> None:
    """Catches a link built as `None/traces/...` or from a hard-coded Langfuse address when
    the setting is unset: every id is valid here, so only the missing setting nulls them."""
    app = make_wired(None)

    rows = app.body()["requests"]

    assert len(rows) == 2
    assert [row["trace_url"] for row in rows] == [None, None]
    assert [row["trace_id"] for row in rows] == [TRACE_ID, OTHER_TRACE_ID]


def test_prometheus_down_keeps_the_request_list(wired: Wired) -> None:
    """Catches a Prometheus failure that blanks the request list, fails the whole page with
    a 500, reports `unconfigured` or `ok`, or leaves stale counts in the body."""
    wired.prom.mode = "500"

    body = wired.body()

    assert body["sources"] == {"postgres": "ok", "prometheus": "error"}
    assert body["redactions"] == []
    assert body["refusals"] == []
    assert body["tokens"] == []
    assert body["guardrail_blocks"] is None
    assert body["cache_hit_rate"] is None
    assert len(body["requests"]) == 2


def test_postgres_down_keeps_the_counts(wired: Wired) -> None:
    """Catches the reverse failure: a dead ledger that blanks the Prometheus counts, 500s
    the page, or reports the postgres source as `ok`."""
    wired.reader.error = psycopg.OperationalError("connection refused")

    body = wired.body(days=7)

    assert body["sources"] == {"postgres": "error", "prometheus": "ok"}
    assert body["requests"] == []
    assert body["guardrail_blocks"] == 5
    assert body["cache_hit_rate"] == 0.3125
    assert len(body["redactions"]) == 2


def test_nothing_configured_is_200_with_both_sources_unconfigured(admin_settings: Settings) -> None:
    """Catches a route that reads an unconfigured source anyway, or 500s without a database
    or a Prometheus: the flags say `unconfigured` and every field is empty or null."""
    with TestClient(create_app(admin_settings, NO_PROVIDER_KEYS)) as bare:
        response = bare.get(REQUESTS, headers=ADMIN_HEADERS)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "sources": {"postgres": "unconfigured", "prometheus": "unconfigured"},
        "redactions": [],
        "refusals": [],
        "guardrail_blocks": None,
        "tokens": [],
        "cache_hit_rate": None,
        "requests": [],
        "requests_page": 1,
        "requests_pages": 1,
    }


def test_requests_is_401_without_admin_key(wired: Wired, no_admin_key_client: TestClient) -> None:
    """Catches a route left off the `admin_key` dependency, or checked against the tenant key
    store: no key, a working tenant key and an app with no admin key set all get 401, and
    nothing is read first."""
    assert_error_body(wired.client.get(REQUESTS), status=401, code="UNAUTHORIZED")
    tenant = wired.client.get(
        REQUESTS, headers={"Authorization": f"Bearer {TEST_KEYS['nordlicht']}"}
    )
    assert_error_body(tenant, status=401, code="UNAUTHORIZED")
    unset = no_admin_key_client.get(REQUESTS, headers=ADMIN_HEADERS)
    assert_error_body(unset, status=401, code="UNAUTHORIZED")

    assert wired.reader.calls == []
    assert wired.prom.queries == []


def test_days_91_is_refused(wired: Wired) -> None:
    """Catches a missing upper bound (91 would reach `[91d]` in PromQL) and an off-by-one at
    the inclusive edge: 90 is 200, 91 is a 422 naming `days`, and a refusal reads nothing."""
    over = wired.get(days=91)

    assert_error_body(over, status=422, code="VALIDATION")
    assert over.json()["error"]["param"] == "days"
    assert wired.reader.calls == []
    assert wired.prom.queries == []

    assert wired.get(days=90).status_code == 200
    assert all("[90d]" in query for query in wired.prom.queries)


def test_days_0_is_refused(wired: Wired) -> None:
    """Catches a missing lower bound (`[0d]` is an empty window): 1 is 200, 0 is a 422
    naming `days`."""
    under = wired.get(days=0)

    assert_error_body(under, status=422, code="VALIDATION")
    assert under.json()["error"]["param"] == "days"
    assert wired.prom.queries == []

    assert wired.get(days=1).status_code == 200


def test_the_page_reaches_the_reader(wired: Wired) -> None:
    """Catches a route that drops `page` (the reader always gets 1), passes it positionally
    or as text, or fills the body's page numbers from the query instead of from the read."""
    wired.reader.answer_page = 2
    wired.reader.answer_pages = 7

    body = wired.body(page=3)

    assert wired.reader.pages_asked == [3]
    assert (body["requests_page"], body["requests_pages"]) == (2, 7)


def test_the_default_page_is_1(wired: Wired) -> None:
    """Catches a missing default (422 with no `page`) or a default other than 1."""
    wired.body()

    assert wired.reader.pages_asked == [1]


def test_page_10000_is_accepted_and_10001_refused(wired: Wired) -> None:
    """Catches an upper bound of 9999 or 10001 or none at all: 10000 is served, 10001 is a
    422 VALIDATION naming `page`, and the reader is never called for it."""
    wired.body(page=10000)
    assert wired.reader.pages_asked == [10000]

    response = wired.get(page=10001)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "page"
    assert wired.reader.pages_asked == [10000]


@pytest.mark.parametrize("value", ["0", "-1", "abc"], ids=["zero", "negative", "text"])
def test_page_0_and_abc_are_refused(value: str, wired: Wired) -> None:
    """Catches a lower bound of 0 (or none) and a page read as text: each is a 422
    VALIDATION naming `page`, and nothing is read."""
    response = wired.get(page=value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "page"
    assert wired.reader.calls == []


def test_postgres_down_is_one_empty_page(wired: Wired) -> None:
    """Catches a dead ledger that leaves the page numbers at 0 or at the asked page, or
    drops the two fields from the body: it must read page 1 of 1 with no rows."""
    wired.reader.error = psycopg.OperationalError("connection refused")

    body = wired.body(page=4)

    assert body["sources"]["postgres"] == "error"
    assert (body["requests_page"], body["requests_pages"]) == (1, 1)
    assert body["requests"] == []
