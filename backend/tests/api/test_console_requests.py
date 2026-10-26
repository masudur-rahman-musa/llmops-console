"""`GET /console/requests`: what the gateway hides, tokens, hit rate, per day, newest rows.

No backing service. Each case builds the admin app from `create_app` and then replaces the
two sources on `app.state`, as tests/api/test_console_routes.py does: `console_reader`
becomes `StubReader` below and `prometheus` becomes a real `PrometheusClient` over an
`httpx.MockTransport` that records every query. The route, its dependencies and the error
handlers stay real.

The contract these cases are written against, before it exists:

* `app.api.routes.dashboard.CONSOLE_REQUESTS == "/console/requests"`, admin key, `days`
  1 to 90 with a default of 30;
* `app.services.prometheus_reads`: `redactions_query(days)`, `refusals_query(days)`,
  `guardrail_blocks_query(days)` and `errors_per_day_query()` (an earlier change; `tokens_query` is
  gone, tokens come from the ledger);
* `ConsoleReader.daily_requests(days, now)` and `ConsoleReader.model_tokens(days, now)`: the per-day request counts and the per-model token sums of the ledger;
* `ConsoleReader.cache_hit_rate(days, now) -> float | None`: cached ledger rows over all rows
  in the window, None when the window has no rows;
* `ConsoleReader.recent_requests() -> tuple[RequestRow, ...]`, newest first, with
  `trace_url` None: the route fills `trace_url` from `Settings.langfuse_project_url` and the
  row's `trace_id`, and only when the setting is set and the id is 32 lowercase hex.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.routes import dashboard as console
from app.core.config import Settings
from app.main import create_app
from app.schemas.dashboard import ModelTokens, RequestRow
from app.services.prometheus_reads import (
    QUERY_RANGE_PATH,
    PrometheusClient,
    errors_per_day_query,
    guardrail_blocks_query,
    redactions_query,
    refusals_query,
)
from app.storage.dashboard_reads import RequestsRead

from .conftest import ADMIN_KEY, NO_PROVIDER_KEYS, TEST_KEYS, assert_error_body

REQUESTS = "/console/requests"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
PROJECT_URL = "https://cloud.langfuse.com/project/demo"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
OTHER_TRACE_ID = "b7ad6b7169203331b7ad6b7169203331"
SAMPLE_TS = 1757980800

# The four queries for a 7 day window, written out so no case reads its expectation from the
# module under test. The errors query has a fixed one-day window whatever `days` is.
REDACTIONS_7D = "sum by (kind) (increase(gateway_redactions_total[7d]))"
TOKENS_7D = "sum by (model, direction) (increase(gateway_tokens_total[7d]))"
REFUSALS_7D = "sum by (reason) (increase(helpdesk_refusals_total[7d]))"
BLOCKS_7D = "sum(increase(helpdesk_sod_blocks_total[7d]))"
ERRORS_PER_DAY = 'sum(increase(gateway_requests_total{status=~"5.."}[1d]))'
ALL_7D = {REDACTIONS_7D, REFUSALS_7D, BLOCKS_7D, ERRORS_PER_DAY}
DAY_SECONDS = 86400

LEDGER_TOKENS = (
    ModelTokens(model="claude-haiku-5-5", input=120450.0, output=30210.0),
    ModelTokens(model="gpt-5-mini", input=8800.0, output=2100.0),
)


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

    `recent_requests`, `cache_hit_rate`, `daily_requests` and `model_tokens` are read by this
    route. `daily` is the request count of each window day, oldest first (a short list is
    zero filled); the dates come from the `now` the route passes, so no case depends on the
    clock. `tokens` is the per-model read. `error`, when set, is raised instead, which is how
    the Postgres failure case arranges a failing read without a database.
    """

    def __init__(self, rows: tuple[RequestRow, ...] = ROWS) -> None:
        self.configured = True
        self.rows = rows
        self.calls: list[str] = []
        self.pages_asked: list[int] = []
        self.error: Exception | None = None
        self.answer_page = 1
        self.answer_pages = 1
        self.hit_rate: float | None = 0.3125
        self.hit_rate_asked: list[tuple[int, datetime]] = []
        self.daily: list[int] = [10, 0, 7]
        self.daily_asked: list[tuple[int, datetime]] = []
        self.tokens: tuple[ModelTokens, ...] = LEDGER_TOKENS
        self.tokens_asked: list[tuple[int, datetime]] = []

    def daily_requests(self, days: int, now: datetime) -> tuple[tuple[date, int], ...]:
        self.calls.append("daily_requests")
        self.daily_asked.append((days, now))
        if self.error is not None:
            raise self.error
        counts = [self.daily[i] if i < len(self.daily) else 0 for i in range(days)]
        first = now.date() - timedelta(days=days - 1)
        return tuple((first + timedelta(days=i), n) for i, n in enumerate(counts))

    def model_tokens(self, days: int, now: datetime) -> tuple[ModelTokens, ...]:
        self.calls.append("model_tokens")
        self.tokens_asked.append((days, now))
        if self.error is not None:
            raise self.error
        return self.tokens

    def cache_hit_rate(self, days: int, now: datetime) -> float | None:
        self.calls.append("cache_hit_rate")
        self.hit_rate_asked.append((days, now))
        if self.error is not None:
            raise self.error
        return self.hit_rate

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
    REFUSALS_7D: [
        series({"reason": "out_of_scope"}, "12"),
        series({"reason": "needs_approval"}, "3"),
    ],
    BLOCKS_7D: [series({}, "5")],
}


class PromStub:
    """A mock Prometheus that answers `ANSWERS` by query text and records every query.

    A range query (the errors per day) is answered from `errors`: day index (0 is the first
    window day) to its 5xx count, as one point per day at `start + index * 86400`, the way
    Prometheus lays out a one-day step. A day missing from `errors` has no point. `calls`
    keeps the path and parameters of every request.
    """

    def __init__(self) -> None:
        self.mode = "ok"
        self.queries: list[str] = []
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.errors: dict[int, float] = {0: 2.0, 2: 5.0}

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        query = params.get("query", "")
        self.queries.append(query)
        self.calls.append((request.url.path, params))
        if self.mode == "500":
            return httpx.Response(500, text="internal error")
        if request.url.path == QUERY_RANGE_PATH:
            start = float(params["start"])
            values = [[start + i * DAY_SECONDS, str(v)] for i, v in sorted(self.errors.items())]
            matrix = [] if self.mode == "empty" else [{"metric": {}, "values": values}]
            return httpx.Response(
                200, json={"status": "success", "data": {"resultType": "matrix", "result": matrix}}
            )
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
    """Catches a changed metric name, a missing `sum by`, a window that ignores `days`, and a
    route that asks for more or fewer queries than the four: the sent set must equal the
    literals (a hit-rate or tokens query still sent to Prometheus would be a fifth)."""
    assert console.CONSOLE_REQUESTS == REQUESTS
    assert redactions_query(7) == REDACTIONS_7D
    assert refusals_query(7) == REFUSALS_7D
    assert guardrail_blocks_query(7) == BLOCKS_7D
    assert errors_per_day_query() == ERRORS_PER_DAY
    for query in (REDACTIONS_7D, REFUSALS_7D, BLOCKS_7D):
        assert "[7d]" in query

    wired.body(days=7)

    assert set(wired.prom.queries) == ALL_7D
    assert len(wired.prom.queries) == 4


def test_requests_body_maps_counts(wired: Wired) -> None:
    """Catches a body that swaps in and out, drops a model, a kind or a reason, reads the
    help desk's two counters from each other's series, drops `per_day`, or turns the hit rate
    (read from the reader) into a percent (the page does that, the body carries the ratio)."""
    body = wired.body(days=7)

    assert set(body) == {
        "sources",
        "redactions",
        "refusals",
        "guardrail_blocks",
        "tokens",
        "cache_hit_rate",
        "per_day",
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
    assert sorted(wired.reader.calls) == [
        "cache_hit_rate",
        "daily_requests",
        "model_tokens",
        "recent_requests",
    ]
    assert wired.reader.pages_asked == [1]


def test_no_traffic_gives_null_rate_and_blocks(wired: Wired) -> None:
    """Catches a body that answers 0 for a hit rate (ledger with no rows) or a block count
    (Prometheus with no series) - no traffic is not a 0% hit rate - or fails on the empty
    results."""
    wired.prom.mode = "empty"
    wired.reader.hit_rate = None
    wired.reader.tokens = ()
    wired.reader.daily = []

    body = wired.body(days=3)

    assert body["sources"] == {"postgres": "ok", "prometheus": "ok"}
    assert body["cache_hit_rate"] is None
    assert body["guardrail_blocks"] is None
    assert body["redactions"] == []
    assert body["refusals"] == []
    assert body["tokens"] == []
    assert [(row["requests"], row["errors"]) for row in body["per_day"]] == [(0, 0.0)] * 3


def test_the_default_window_is_30_days(wired: Wired) -> None:
    """Catches a default other than 30 days: the queries sent with no `days` must be the 30
    day forms of the builders (the errors query is one day whatever the window), and the
    ledger reads (hit rate, per day, tokens) must be asked for 30 days."""
    body = wired.body()

    assert set(wired.prom.queries) == {
        redactions_query(30),
        refusals_query(30),
        guardrail_blocks_query(30),
        ERRORS_PER_DAY,
    }
    assert all("[30d]" in query for query in wired.prom.queries if query != ERRORS_PER_DAY)
    assert [days for days, _ in wired.reader.hit_rate_asked] == [30]
    assert [days for days, _ in wired.reader.daily_asked] == [30]
    assert [days for days, _ in wired.reader.tokens_asked] == [30]
    assert len(body["per_day"]) == 30


def test_hit_rate_comes_from_the_ledger_not_prometheus(wired: Wired) -> None:
    """Catches a route that still asks Prometheus for the hit rate (a counter series born at
    1 reads as 0), or that ignores the reader: the reader says 1 cached of 4 (0.25), which
    no Prometheus answer here carries, and no query may mention the cache label."""
    wired.reader.hit_rate = 0.25

    body = wired.body(days=7)

    assert body["cache_hit_rate"] == 0.25
    assert [days for days, _ in wired.reader.hit_rate_asked] == [7]
    now = wired.reader.hit_rate_asked[0][1]
    assert now.tzinfo is not None
    assert not any("cache=" in query for query in wired.prom.queries)
    # The errors per day query is the one that may name the requests counter, and it has no
    # cache label; any other query naming it would be a hit-rate query sent to Prometheus.
    assert not any(
        "gateway_requests_total" in query for query in wired.prom.queries if query != ERRORS_PER_DAY
    )


def test_an_empty_window_gives_a_null_hit_rate(wired: Wired) -> None:
    """Catches a route that turns a ledger with no rows (reader answers None) into 0, which
    reads as a real 0% hit rate."""
    wired.reader.hit_rate = None

    body = wired.body(days=7)

    assert body["cache_hit_rate"] is None


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
    """Catches a Prometheus failure that blanks the request list or the ledger hit rate (the
    rate and the tokens no longer come from Prometheus), fails the whole page with a 500,
    reports `unconfigured` or `ok`, or leaves stale counts in the body."""
    wired.prom.mode = "500"

    body = wired.body()

    assert body["sources"] == {"postgres": "ok", "prometheus": "error"}
    assert body["redactions"] == []
    assert body["refusals"] == []
    assert body["tokens"] == [t.model_dump() for t in LEDGER_TOKENS]
    assert len(body["per_day"]) == 30
    assert all(row["errors"] is None for row in body["per_day"])
    assert body["guardrail_blocks"] is None
    assert body["cache_hit_rate"] == 0.3125
    assert len(body["requests"]) == 2


def test_postgres_down_keeps_the_counts(wired: Wired) -> None:
    """Catches the reverse failure: a dead ledger that blanks the Prometheus counts, 500s
    the page, reports the postgres source as `ok`, or keeps a hit rate the dead ledger
    could not have answered."""
    wired.reader.error = psycopg.OperationalError("connection refused")

    body = wired.body(days=7)

    assert body["sources"] == {"postgres": "error", "prometheus": "ok"}
    assert body["requests"] == []
    assert body["per_day"] == []
    assert body["tokens"] == []
    assert body["guardrail_blocks"] == 5
    assert body["cache_hit_rate"] is None
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
        "per_day": [],
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
    assert all("[90d]" in query for query in wired.prom.queries if query != ERRORS_PER_DAY)


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
