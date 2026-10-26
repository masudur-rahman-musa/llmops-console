"""`GET /console/requests` filters and the Langfuse link.

No backing service. The admin app is built by `create_app` and `console_reader` is replaced
by a stub that records what the route asks for, as tests/api/test_console_requests.py does.
The route, its checks and the error handlers stay real.

The contract these cases are written against, before it exists:

* `app.storage.dashboard_reads.RequestFilters` (frozen): `day: date | None`, `tenant`,
  `model`, `errors`, `slow`, `cached` (booleans, default False);
* `ConsoleReader.recent_requests(*, page, filters, days, now) -> RequestsRead` with
  `slow_ms`, `tenants`, `models` on the read;
* query params `day`, `tenant` (the existing tenant check), `model` (at most 128 characters,
  `MODEL_PATTERN` as a full match), `errors`, `slow`, `cached`; a bad value is 422
  `VALIDATION` with `param` naming it, and nothing is read;
* `RequestRow.status` and `.latency_ms`; the body gains `slow_ms`, `tenants`, `models` and
  `langfuse_url`;
* `app.services.dashboard.langfuse_traces_url(project_url, filters, slow_ms)`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.dashboard import RequestRow
from app.services.dashboard import langfuse_traces_url
from app.storage.dashboard_reads import RequestFilters, RequestsRead

from .conftest import ADMIN_KEY, NO_PROVIDER_KEYS, assert_error_body

REQUESTS = "/console/requests"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
PROJECT_URL = "https://cloud.langfuse.com/project/demo"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"

DAY_FROM = "Timestamp;datetime;;%3E%3D;2026-10-27T00:00:00.000Z"
DAY_TO = "Timestamp;datetime;;%3C;2026-10-28T00:00:00.000Z"
TENANT = "Metadata;stringObject;tenant;%3D;acme"
MODEL = "Metadata;stringObject;model;%3D;gpt-4o-mini"
CACHED = "Metadata;stringObject;cache;%3D;hit"
ERRORS = "Level;string;;%3D;ERROR"
SLOW = "Latency%20%28s%29;number;;%3E;1.905"
"""The seven filters as Langfuse's address carries them, written out as literals so no case
reads its expectation from the module under test. Each value is percent-encoded with only
`,;:.` and the letters, digits and `_.-~` left bare."""

ALL_SEVEN = (
    f"{PROJECT_URL}/traces?filter={DAY_FROM},{DAY_TO},{TENANT},{MODEL},{CACHED},{ERRORS},{SLOW}"
)

EVERY_FILTER = RequestFilters(
    day=date(2026, 10, 27),
    tenant="acme",
    model="gpt-4o-mini",
    errors=True,
    slow=True,
    cached=True,
)


def a_row(**overrides: Any) -> RequestRow:  # noqa: ANN401
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
        "prompt_version": None,
        "trace_id": TRACE_ID,
        "trace_url": None,
        "status": 200,
        "latency_ms": 412,
    }
    fields.update(overrides)
    return RequestRow(**fields)


class StubReader:
    """A configured reader that records the filters, days and time of every list read."""

    configured = True

    def __init__(self) -> None:
        self.rows: tuple[RequestRow, ...] = (a_row(),)
        self.calls: list[str] = []
        self.filters_asked: list[RequestFilters] = []
        self.days_asked: list[int] = []
        self.now_asked: list[datetime] = []
        self.slow_ms: float | None = 1905.0

    def recent_requests(
        self, *, page: int, filters: RequestFilters, days: int, now: datetime
    ) -> RequestsRead:
        self.calls.append("recent_requests")
        self.filters_asked.append(filters)
        self.days_asked.append(days)
        self.now_asked.append(now)
        return RequestsRead(
            rows=self.rows,
            page=1,
            pages=1,
            slow_ms=self.slow_ms,
            tenants=("acme", "nordlicht"),
            models=("claude-haiku-5-5", "gpt-4o-mini"),
        )

    def cache_hit_rate(self, days: int, now: datetime) -> float | None:
        self.calls.append("cache_hit_rate")
        return None

    def daily_requests(self, days: int, now: datetime) -> tuple[tuple[date, int], ...]:
        self.calls.append("daily_requests")
        return ()

    def model_tokens(self, days: int, now: datetime) -> tuple[()]:
        self.calls.append("model_tokens")
        return ()


class Wired:
    def __init__(self, client: TestClient, reader: StubReader) -> None:
        self.client = client
        self.reader = reader

    def get(self, **params: str | int) -> httpx.Response:
        response: httpx.Response = self.client.get(REQUESTS, params=params, headers=ADMIN_HEADERS)
        return response


@pytest.fixture
def make_wired(admin_settings: Settings) -> Iterator[Callable[[str | None], Wired]]:
    entered: list[TestClient] = []

    def build(project_url: str | None) -> Wired:
        settings = admin_settings.model_copy(update={"langfuse_project_url": project_url})
        app = create_app(settings, NO_PROVIDER_KEYS)
        reader = StubReader()
        client = TestClient(app)
        client.__enter__()
        entered.append(client)
        app.state.console_reader = reader
        return Wired(client, reader)

    yield build
    for client in entered:
        client.__exit__(None, None, None)


@pytest.fixture
def wired(make_wired: Callable[[str | None], Wired]) -> Wired:
    return make_wired(PROJECT_URL)


# --- the params reach the reader -------------------------------------------------------


def test_filters_reach_the_reader(wired: Wired) -> None:
    """Catches a route that drops a param, passes one under another's name (errors read as
    slow), reads a boolean as text, or turns the date into a string: all six arrive as one
    `RequestFilters`. With none given the reader gets the empty filter, so a default of True
    on any of the three booleans fails too."""
    none = wired.get()
    assert none.status_code == 200, none.text
    assert wired.reader.filters_asked == [RequestFilters()]
    assert RequestFilters() == RequestFilters(
        day=None, tenant=None, model=None, errors=False, slow=False, cached=False
    )

    every = wired.get(
        day="2026-10-27",
        tenant="acme",
        model="gpt-4o-mini",
        errors="true",
        slow="true",
        cached="true",
    )

    assert every.status_code == 200, every.text
    assert wired.reader.filters_asked[1] == EVERY_FILTER
    assert wired.reader.days_asked == [30, 30]
    assert all(now.tzinfo is not None for now in wired.reader.now_asked)


SINGLE: list[tuple[str, dict[str, str], RequestFilters]] = [
    ("day", {"day": "2026-10-27"}, RequestFilters(day=date(2026, 10, 27))),
    ("tenant", {"tenant": "acme"}, RequestFilters(tenant="acme")),
    ("model", {"model": "gpt-4o-mini"}, RequestFilters(model="gpt-4o-mini")),
    ("errors", {"errors": "true"}, RequestFilters(errors=True)),
    ("slow", {"slow": "true"}, RequestFilters(slow=True)),
    ("cached", {"cached": "true"}, RequestFilters(cached=True)),
]


@pytest.mark.parametrize(("case", "params", "expected"), SINGLE, ids=[r[0] for r in SINGLE])
def test_each_param_sets_only_its_own_filter(
    case: str, params: dict[str, str], expected: RequestFilters, wired: Wired
) -> None:
    """Catches two params wired to each other's field: one param at a time, the reader gets
    that filter set and every other one left off."""
    assert wired.get(**params).status_code == 200

    assert wired.reader.filters_asked == [expected]


# --- the checks ------------------------------------------------------------------------


def test_model_at_128_passes_and_129_is_refused(wired: Wired) -> None:
    """Catches a model cap of 127 or 129 or none: 128 characters (the ledger's cap) is
    served and reaches the reader, 129 is a 422 VALIDATION naming `model`, and the refusal
    reads nothing."""
    at_cap = "m" * 128
    assert wired.get(model=at_cap).status_code == 200
    assert wired.reader.filters_asked == [RequestFilters(model=at_cap)]
    wired.reader.calls.clear()

    over = wired.get(model="m" * 129)

    assert_error_body(over, status=422, code="VALIDATION")
    assert over.json()["error"]["param"] == "model"
    assert wired.reader.calls == []


MODEL_REFUSED: list[tuple[str, str]] = [
    ("semicolon", "gpt;4o"),
    ("comma", "gpt,4o"),
    ("trailing newline", "gpt-4o\n"),
    ("trailing CRLF", "gpt-4o\r\n"),
    ("trailing space", "gpt-4o "),
    ("leading dash", "-gpt"),
]


@pytest.mark.parametrize(("case", "value"), MODEL_REFUSED, ids=[r[0] for r in MODEL_REFUSED])
def test_model_with_a_semicolon_or_comma_is_refused(case: str, value: str, wired: Wired) -> None:
    """Unhappy path. Catches a model check that is looser than the ledger's `MODEL_PATTERN`:
    `;` and `,` would break the Langfuse filter list apart, and a `$`-anchored pattern lets a
    trailing newline through. Each is a 422 naming `model` and reads nothing."""
    response = wired.get(model=value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "model"
    assert wired.reader.calls == []


def test_model_ids_the_providers_use_are_accepted(wired: Wired) -> None:
    """Catches a model check copied from the tenant rule: shipped ids carry dots, colons and
    slashes (`qwen2.5-3b-instruct`, `org/model:tag`) and must still filter."""
    for model in ("qwen2.5-3b-instruct", "org/model:tag", "Claude_3"):
        assert wired.get(model=model).status_code == 200, model


def test_tenant_at_64_passes_and_65_is_refused(wired: Wired) -> None:
    """Catches a tenant cap of 63 or 65 or none: 64 is served, 65 is a 422 VALIDATION naming
    `tenant`, and the refusal reads nothing."""
    assert wired.get(tenant="t" * 64).status_code == 200
    wired.reader.calls.clear()

    over = wired.get(tenant="t" * 65)

    assert_error_body(over, status=422, code="VALIDATION")
    assert over.json()["error"]["param"] == "tenant"
    assert wired.reader.calls == []


TENANT_REFUSED: list[tuple[str, str]] = [
    ("trailing newline", "acme\n"),
    ("trailing CRLF", "acme\r\n"),
    ("trailing space", "acme "),
    ("uppercase", "Acme"),
    ("semicolon", "ac;me"),
]


@pytest.mark.parametrize(("case", "value"), TENANT_REFUSED, ids=[r[0] for r in TENANT_REFUSED])
def test_a_bad_tenant_is_refused(case: str, value: str, wired: Wired) -> None:
    """Unhappy path. Catches a tenant check that lets a trailing newline, CRLF or space
    through, or one skipped on this route: each is a 422 naming `tenant`."""
    response = wired.get(tenant=value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "tenant"
    assert wired.reader.calls == []


@pytest.mark.parametrize("value", ["2026-13-01", "2026-02-30", "yesterday"])
def test_a_bad_day_is_refused(value: str, wired: Wired) -> None:
    """Unhappy path. Catches a day read as free text or a month 13 that rolls over: each is a
    422 VALIDATION naming `day` and reads nothing."""
    response = wired.get(day=value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "day"
    assert wired.reader.calls == []


# --- the rows and the body -------------------------------------------------------------


def test_rows_carry_status_and_latency(wired: Wired) -> None:
    """Catches a route that rebuilds rows and drops the two new fields, turns a null latency
    into 0, or fills the status with 200: a 502 row with latency 412 and a served row with a
    null latency come through as stored, and the options and the line ride along."""
    wired.reader.rows = (a_row(status=502, latency_ms=412), a_row(status=200, latency_ms=None))

    response = wired.get()

    body = response.json()
    assert [(row["status"], row["latency_ms"]) for row in body["requests"]] == [
        (502, 412),
        (200, None),
    ]
    assert body["slow_ms"] == 1905.0
    assert body["tenants"] == ["acme", "nordlicht"]
    assert body["models"] == ["claude-haiku-5-5", "gpt-4o-mini"]


# --- the Langfuse link -----------------------------------------------------------------


def test_langfuse_link_carries_the_same_filters(wired: Wired) -> None:
    """Catches a link that drops a filter, orders them differently, uses the wrong column,
    type, key or operator, takes the p95 in ms instead of seconds, or leaves the spaces and
    brackets of `Latency (s)` bare. The exact URL is pinned with seven filters."""
    response = wired.get(
        day="2026-10-27",
        tenant="acme",
        model="gpt-4o-mini",
        errors="true",
        slow="true",
        cached="true",
    )

    assert response.json()["langfuse_url"] == ALL_SEVEN


def test_the_link_function_pins_the_exact_url() -> None:
    """Same claim as above, one level down, so a failure names the function and not the
    route."""
    assert langfuse_traces_url(PROJECT_URL, EVERY_FILTER, 1905.0) == ALL_SEVEN


LINK_ROWS: list[tuple[str, RequestFilters, str]] = [
    ("day", RequestFilters(day=date(2026, 10, 27)), f"{DAY_FROM},{DAY_TO}"),
    (
        "day at year end",
        RequestFilters(day=date(2026, 12, 31)),
        "Timestamp;datetime;;%3E%3D;2026-12-31T00:00:00.000Z,"
        "Timestamp;datetime;;%3C;2027-01-01T00:00:00.000Z",
    ),
    ("tenant", RequestFilters(tenant="acme"), TENANT),
    ("model", RequestFilters(model="gpt-4o-mini"), MODEL),
    ("cached", RequestFilters(cached=True), CACHED),
    ("errors", RequestFilters(errors=True), ERRORS),
    ("slow", RequestFilters(slow=True), SLOW),
    ("tenant and errors", RequestFilters(tenant="acme", errors=True), f"{TENANT},{ERRORS}"),
]


@pytest.mark.parametrize(("case", "filters", "expected"), LINK_ROWS, ids=[r[0] for r in LINK_ROWS])
def test_each_filter_has_its_own_fragment(
    case: str, filters: RequestFilters, expected: str
) -> None:
    """Catches one filter's fragment written wrong or sent for a different filter: each set
    alone (and two together, in order) gives exactly its own fragments and no other."""
    assert langfuse_traces_url(PROJECT_URL, filters, 1905.0) == (
        f"{PROJECT_URL}/traces?filter={expected}"
    )


def test_the_slow_line_is_in_seconds() -> None:
    """Catches a threshold sent in milliseconds: 250 ms is 0.25 seconds."""
    url = langfuse_traces_url(PROJECT_URL, RequestFilters(slow=True), 250.0)

    assert url == f"{PROJECT_URL}/traces?filter=Latency%20%28s%29;number;;%3E;0.25"


def test_no_filters_links_to_all_traces_and_no_project_url_gives_none(
    make_wired: Callable[[str | None], Wired],
) -> None:
    """Catches a link with a dangling `?filter=` when nothing is set, a doubled slash from a
    project URL that ends in `/`, and a made-up address when no project URL is set (the
    setting being unset must give null, never `None/traces`)."""
    assert langfuse_traces_url(PROJECT_URL, RequestFilters(), 1905.0) == f"{PROJECT_URL}/traces"
    assert langfuse_traces_url(PROJECT_URL + "/", RequestFilters(), None) == (
        f"{PROJECT_URL}/traces"
    )
    assert langfuse_traces_url(None, EVERY_FILTER, 1905.0) is None
    assert langfuse_traces_url(None, RequestFilters(), None) is None

    assert make_wired(PROJECT_URL).get().json()["langfuse_url"] == f"{PROJECT_URL}/traces"
    assert make_wired(None).get(tenant="acme").json()["langfuse_url"] is None
