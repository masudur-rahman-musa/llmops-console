"""The six `/console/*` read routes: the key space, the pre-checks and the source flags.

No backing service. Each case builds the admin app from `create_app` (cheap and
in-process) and then replaces the two sources on `app.state`, exactly as the
provider stubs in tests/gateway/conftest.py are swapped in after the factory has
run: `console_reader` becomes `StubReader` below, which records every call and
answers fixed reads, and `prometheus` becomes a real `PrometheusClient` over an
`httpx.MockTransport` that records every query. The route, the dependency, the
pre-checks and the error handlers stay real.

What is pinned, each case naming the wrong implementation it catches:

* The six routes are in the ADMIN key space: no key, an unknown key, and a
  working tenant key are all 401, before anything is read.
* `days` and `tenant` are refused before any query runs, at the inclusive
  boundaries, because `tenant` is interpolated into PromQL.
* A source that is unconfigured or failing is a FLAG beside a 200, never an error: its own fields go empty, the other source's survive, and a
  failure logs at ERROR.
* The bodies carry what the reader and the named queries answered, and the
  derived parts - budgets, circuit state names, regressions - are computed right.

The contract these cases are written against, before it exists:

* `app.api.routes.dashboard`: the `CONSOLE_*` path constants and
  `CONSOLE_PATHS`, `MIN_DAYS`/`MAX_DAYS`/`DEFAULT_DAYS`, `SourceState`;
* `app.storage.dashboard_reads`: the read models, `ConsoleReader`,
  `UnconfiguredConsoleReader`, `HIGHER_IS_BETTER_METRICS` and `regressed_metrics`;
* `app.services.prometheus_reads`: `PrometheusClient` and the named queries;
* `app.state.console_reader` and `app.state.prometheus`, built by `create_app`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import dashboard as console
from app.core.config import Settings
from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.limits.budget import current_month
from app.limits.prices import PriceTable
from app.main import create_app
from app.schemas.dashboard import (
    AliasCost,
    CategoryCount,
    CostDay,
    EvalRunSummary,
    Incident,
    LastEval,
    Regression,
    Release,
    ReleaseMarker,
    Saving,
    TagCount,
)
from app.services.dashboard import JUDGE_NOTE
from app.services.prometheus_reads import (
    AVAILABILITY_BUDGET_QUERY,
    CIRCUITS_QUERY,
    ERROR_RATE_QUERY,
    LATENCY_BUDGET_QUERY,
    P95_QUERY,
    QUERY_PATH,
    QUERY_RANGE_PATH,
    PrometheusClient,
)
from app.storage.dashboard_reads import (
    HIGHER_IS_BETTER_METRICS,
    BlockedRelease,
    CostRead,
    HomeRead,
    QualityRead,
    UnconfiguredConsoleReader,
    regressed_metrics,
)

from .conftest import (
    ADMIN_KEY,
    MAX_TENANT,
    NO_PROVIDER_KEYS,
    TEST_KEYS,
    UNKNOWN_KEY,
    assert_error_body,
)

# The six paths, written out so no request below reads its target from the module
# under test. `test_the_six_console_paths_are_pinned` compares them with the module's
# constants and with what the factory mounted.
HOME = "/console/home"
COST = "/console/cost"
QUALITY = "/console/quality"
RELIABILITY = "/console/reliability"
INCIDENTS = "/console/incidents"
RELEASES = "/console/releases"

ALL_PATHS = [HOME, COST, QUALITY, RELIABILITY, INCIDENTS, RELEASES]
DAYS_PATHS = [COST, QUALITY, RELIABILITY]
TENANT_PATHS = [COST]
POSTGRES_PATHS = [HOME, COST, QUALITY, INCIDENTS, RELEASES]
PROMETHEUS_PATHS = [HOME, RELIABILITY]

LOGGER = "app.services.dashboard"

SHIPPED_BASELINE = Path(__file__).resolve().parents[2] / "evals" / "conversation-baseline.json"

ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
TENANT_KEY = TEST_KEYS["nordlicht"]

UNCONFIGURED = "unconfigured"
OK = "ok"
ERROR = "error"


# --- the reads the stub reader answers ---------------------------------------------

FINISHED = datetime(2026, 9, 15, 3, 0, tzinfo=UTC)
STARTED = datetime(2026, 9, 15, 2, 40, tzinfo=UTC)
DEPLOYED = datetime(2026, 9, 15, 9, 15, tzinfo=UTC)

HOME_READ = HomeRead(
    requests_today=42,
    cost_usd_today=1.25,
    last_eval=LastEval(id="run-a", finished_at=FINISHED, passed=True, check_score=0.91),
    open_incidents=2,
)

COST_READ = CostRead(
    rollup_populated=True,
    days=(
        CostDay(
            day=date(2026, 9, 14),
            tenant="nordlicht",
            provider="anthropic",
            requests=10,
            cached_requests=3,
            cost_usd=0.75,
        ),
    ),
    savings=(Saving(day=date(2026, 9, 14), tenant="nordlicht", saved_usd=0.5),),
    savings_usd_total=0.5,
    cost_per_request=(AliasCost(alias="agent-default", cost_usd=0.0125),),
)

QUALITY_READ = QualityRead(
    runs=(
        EvalRunSummary(
            id="run-a",
            started_at=STARTED,
            run_kind="ci",
            git_sha="abc1234",
            passed=True,
            metrics={"check_score": 0.91},
        ),
    ),
    release_markers=(ReleaseMarker(git_sha="abc1234", deployed_at=DEPLOYED, status="deployed"),),
    categories=(CategoryCount(category="routing", passed=4, total=5),),
    taxonomy=(TagCount(tag="wrong_tool", count=2),),
    blocked=(),
)

INCIDENT_ROWS = (
    Incident(
        id="0af7651916cd43dd8448eb211c80319c",
        started_at=STARTED,
        resolved_at=None,
        severity="sev2",
        title="gateway 5xx on the eu pool",
        summary="the eu pool answered 503 for eleven minutes",
        postmortem_url=None,
        open=True,
    ),
)

RELEASE_ROWS = (
    Release(
        git_sha="abc1234",
        deployed_at=DEPLOYED,
        status="deployed",
        pr_url=None,
        eval_run_id="run-a",
        eval_passed=True,
        check_score=0.91,
    ),
)


class StubReader:
    """A configured `ConsoleReader` recording every call, answering the reads above.

    `error`, when set, is raised by every method instead, which is how the Postgres
    failure cases arrange a read that fails without a database.
    """

    def __init__(self) -> None:
        self.configured = True
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None
        self.quality_read = QUALITY_READ

    def _record(self, name: str, **kwargs: Any) -> None:  # noqa: ANN401
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error

    def home(self, *, now: datetime) -> HomeRead:
        self._record("home", now=now)
        return HOME_READ

    def cost(self, *, days: int, tenant: str | None, now: datetime, prices: PriceTable) -> CostRead:
        self._record("cost", days=days, tenant=tenant, now=now, prices=prices)
        return COST_READ

    def quality(self, *, days: int, now: datetime) -> QualityRead:
        self._record("quality", days=days, now=now)
        return self.quality_read

    def incidents(self) -> tuple[Incident, ...]:
        self._record("incidents")
        return INCIDENT_ROWS

    def releases(self) -> tuple[Release, ...]:
        self._record("releases")
        return RELEASE_ROWS


# --- the mock Prometheus -----------------------------------------------------------

SAMPLE_TS = 1757980800

FALLBACKS_PREFIX = "sum by (tenant, alias, provider) (increase(gateway_fallbacks_total"

# An earlier change's budget queries, answered by default so every reliability case sees a full
# `slos`: a 5xx share of 0.001 and a share under 6 seconds of 0.98.
BAD_RATIO_30D = "0.001"
UNDER_THRESHOLD_SHARE_30D = "0.98"


def instant_result(query: str) -> list[dict[str, Any]]:
    """What the mock answers to one instant query."""
    if query == P95_QUERY:
        return [{"metric": {}, "value": [SAMPLE_TS, "0.734"]}]
    if query == ERROR_RATE_QUERY:
        return [{"metric": {}, "value": [SAMPLE_TS, "0.012"]}]
    if query == AVAILABILITY_BUDGET_QUERY:
        return [{"metric": {}, "value": [SAMPLE_TS, BAD_RATIO_30D]}]
    if query == LATENCY_BUDGET_QUERY:
        return [{"metric": {}, "value": [SAMPLE_TS, UNDER_THRESHOLD_SHARE_30D]}]
    if query == CIRCUITS_QUERY:
        return [
            {"metric": {"provider": "anthropic"}, "value": [SAMPLE_TS, "0"]},
            {"metric": {"provider": "local"}, "value": [SAMPLE_TS, "1"]},
            {"metric": {"provider": "openai"}, "value": [SAMPLE_TS, "2"]},
            {"metric": {"provider": "vllm"}, "value": [SAMPLE_TS, "3"]},
        ]
    if query.startswith(FALLBACKS_PREFIX):
        return [
            {
                "metric": {"tenant": "evals", "alias": "agent-default", "provider": "local"},
                "value": [SAMPLE_TS, "4"],
            }
        ]
    return []


def range_result(query: str) -> list[dict[str, Any]]:
    """What the mock answers to one range query."""
    if query == P95_QUERY:
        return [{"metric": {}, "values": [[SAMPLE_TS, "0.42"], [SAMPLE_TS + 3600, "0.51"]]}]
    if query == ERROR_RATE_QUERY:
        return [{"metric": {}, "values": [[SAMPLE_TS, "0.01"]]}]
    return []


class PromStub:
    """A mock Prometheus recording every request; `mode` says how it answers.

    `ok` answers the fixed samples above, `empty` answers well-formed empty results,
    and `timeout`, `500` and `error-status` are the three failures the plan names.
    """

    def __init__(self) -> None:
        self.mode = "ok"
        self.calls: list[tuple[str, dict[str, str]]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.calls.append((request.url.path, params))
        if self.mode == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if self.mode == "500":
            return httpx.Response(500, text="internal error")
        is_range = request.url.path == QUERY_RANGE_PATH
        query = params.get("query", "")
        if self.mode == "empty":
            result: list[dict[str, Any]] = []
        else:
            result = range_result(query) if is_range else instant_result(query)
        status = "error" if self.mode == "error-status" else "success"
        body = {
            "status": status,
            "data": {"resultType": "matrix" if is_range else "vector", "result": result},
        }
        return httpx.Response(200, json=body)

    def client(self) -> PrometheusClient:
        return PrometheusClient(
            httpx.Client(
                base_url="http://prometheus.test", transport=httpx.MockTransport(self.handle)
            )
        )

    def queries(self, path: str) -> list[str]:
        """Every PromQL string sent to one API path."""
        return [params.get("query", "") for sent, params in self.calls if sent == path]


class SpendStore:
    """A `BudgetStore` answering a fixed spend per tenant and recording what was asked."""

    def __init__(self, spend: Mapping[str, int | None]) -> None:
        self._spend = dict(spend)
        self.asked: list[tuple[str, str]] = []

    def spend(self, tenant: str, month: str) -> int | None:
        self.asked.append((tenant, month))
        return self._spend.get(tenant, 0)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        raise AssertionError(f"a read route debited {tenant} {month} {micro_usd}")


# --- the app ---------------------------------------------------------------------


@dataclass
class Wired:
    """One admin app with both sources replaced by the recording stubs."""

    client: TestClient
    app: FastAPI
    reader: StubReader
    prom: PromStub

    def get(self, path: str, **params: str | int) -> httpx.Response:
        """One authorised GET."""
        response: httpx.Response = self.client.get(path, params=params, headers=ADMIN_HEADERS)
        return response


@pytest.fixture
def wired(admin_settings: Settings) -> Iterator[Wired]:
    """The admin app, both sources configured with stubs, per case (in-process, cheap)."""
    app = create_app(admin_settings, NO_PROVIDER_KEYS)
    reader = StubReader()
    prom = PromStub()
    with TestClient(app) as client:
        # After the lifespan is entered, which is where the real reader and
        # Prometheus client are now built; swapping before would be overwritten
        # on startup.
        app.state.console_reader = reader
        app.state.prometheus = prom.client()
        yield Wired(client=client, app=app, reader=reader, prom=prom)


@pytest.fixture
def bare(admin_settings: Settings) -> Iterator[TestClient]:
    """The admin app exactly as the factory built it: no database, no Prometheus."""
    with TestClient(create_app(admin_settings, NO_PROVIDER_KEYS)) as client:
        yield client


def error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The ERROR records the console router logged."""
    return [
        record
        for record in caplog.records
        if record.name == LOGGER and record.levelno == logging.ERROR
    ]


# The fields each source owns, as they read when that source gave nothing.
POSTGRES_EMPTY: dict[str, dict[str, Any]] = {
    HOME: {
        "requests_today": None,
        "cost_usd_today": None,
        "last_eval": None,
        "open_incidents": None,
    },
    COST: {
        "rollup_populated": None,
        "days": [],
        "savings": [],
        "savings_usd_total": None,
        "cost_per_request": [],
    },
    QUALITY: {"runs": [], "release_markers": [], "categories": [], "taxonomy": [], "blocked": []},
    RELIABILITY: {},
    INCIDENTS: {"incidents": []},
    RELEASES: {"releases": []},
}

PROMETHEUS_EMPTY: dict[str, dict[str, Any]] = {
    HOME: {"p95_seconds": None, "error_rate": None},
    COST: {},
    QUALITY: {},
    RELIABILITY: {
        "latency_p95": [],
        "error_rate": [],
        "circuits": [],
        "fallbacks": [],
        "slos": [],
    },
    INCIDENTS: {},
    RELEASES: {},
}


def subset(body: Mapping[str, Any], keys: Mapping[str, Any]) -> dict[str, Any]:
    """`body` narrowed to the keys of `keys`, for comparing one source's fields."""
    return {key: body.get(key, "<absent>") for key in keys}


# --- the paths and the key space ---------------------------------------------------


def test_the_six_console_paths_are_pinned(bare: TestClient) -> None:
    """Catches a route mounted under another path or method (`/console/costs`, a POST), a
    constant renamed away from the literal, and a constant that exists while the factory
    never included the router - the mounted set is read from the app's own OpenAPI
    document."""
    assert console.CONSOLE_HOME == HOME
    assert console.CONSOLE_COST == COST
    assert console.CONSOLE_QUALITY == QUALITY
    assert console.CONSOLE_RELIABILITY == RELIABILITY
    assert console.CONSOLE_INCIDENTS == INCIDENTS
    assert console.CONSOLE_RELEASES == RELEASES
    assert tuple(console.CONSOLE_PATHS) == (
        "/console/home",
        "/console/cost",
        "/console/quality",
        "/console/reliability",
        "/console/incidents",
        "/console/releases",
    )

    app = bare.app
    assert isinstance(app, FastAPI)
    paths = app.openapi()["paths"]
    for path in ALL_PATHS:
        assert path in paths, f"{path} is not mounted"
        assert set(paths[path]) == {"get"}, f"{path} answers {set(paths[path])}"


@pytest.mark.parametrize("path", ALL_PATHS)
def test_every_console_route_is_401_without_an_admin_key(
    path: str, wired: Wired, no_admin_key_client: TestClient
) -> None:
    """Catches a console route registered without the `admin_key` dependency - every
    source here is a working stub, so such a route answers 200 - and one whose refusal
    comes after a read: the stubs record nothing on a correct refusal. The no-admin-key app
    is the other direction: an unset admin key must authorise nobody."""
    no_header = wired.client.get(path)
    assert_error_body(no_header, status=401, code="UNAUTHORIZED")

    unknown = wired.client.get(path, headers={"Authorization": f"Bearer {UNKNOWN_KEY}"})
    assert_error_body(unknown, status=401, code="UNAUTHORIZED")
    assert UNKNOWN_KEY not in unknown.text

    unset = no_admin_key_client.get(path, headers=ADMIN_HEADERS)
    assert_error_body(unset, status=401, code="UNAUTHORIZED")
    assert ADMIN_KEY not in unset.text

    assert wired.reader.calls == []
    assert wired.prom.calls == []


@pytest.mark.parametrize("path", ALL_PATHS)
def test_every_console_route_is_401_with_a_tenant_key(
    path: str, wired: Wired, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a console route authenticated against the TENANT key store (or against
    either store), which lets any tenant read every tenant's spend: the key
    presented is a working tenant key on this same app. It also catches the refusal
    echoing the key into the body or any log record, at DEBUG, in `msg` or `args`."""
    with caplog.at_level(logging.DEBUG):
        response = wired.client.get(path, headers={"Authorization": f"Bearer {TENANT_KEY}"})

    assert_error_body(response, status=401, code="UNAUTHORIZED")
    assert TENANT_KEY not in response.text
    for record in caplog.records:
        assert TENANT_KEY not in record.getMessage()
        assert TENANT_KEY not in str(record.msg)
        assert TENANT_KEY not in str(record.args)
    assert wired.reader.calls == []


# --- days ----------------------------------------------------------------------------


def days_sent(wired: Wired, path: str) -> list[int]:
    """The `days` each source was asked for on one route, read off the stubs."""
    if path == RELIABILITY:
        sent: list[int] = []
        for query in wired.prom.queries(QUERY_PATH):
            if query.startswith(FALLBACKS_PREFIX):
                window = query.rsplit("[", 1)[1]
                sent.append(int(window.split("d]", 1)[0]))
        return sent
    return [int(kwargs["days"]) for _, kwargs in wired.reader.calls]


DAYS_ROWS: list[tuple[str, int | None, int]] = [
    ("the lower bound", 1, 200),
    ("the upper bound", 90, 200),
    ("one below the lower bound", 0, 422),
    ("one above the upper bound", 91, 422),
    ("negative", -1, 422),
    ("absent, the default", None, 200),
]


@pytest.mark.parametrize("path", DAYS_PATHS)
@pytest.mark.parametrize(("case", "days", "status"), DAYS_ROWS, ids=[row[0] for row in DAYS_ROWS])
def test_the_days_boundary_is_inclusive(
    path: str, case: str, days: int | None, status: int, wired: Wired
) -> None:
    """Catches an off-by-one on either edge (`gt=1`, `lt=90`), a missing bound (91 would
    answer 200, the stubs accept anything), a default other than 30, and an accepted value
    silently clamped before it reaches the query; a refusal must name `days` and come
    before any source is asked."""
    assert (console.MIN_DAYS, console.MAX_DAYS, console.DEFAULT_DAYS) == (1, 90, 30)
    assert console.DAYS_PARAM == "days"

    response = wired.get(path) if days is None else wired.get(path, days=days)

    if status == 200:
        assert response.status_code == 200, response.text
        expected = 30 if days is None else days
        sent = days_sent(wired, path)
        assert sent != []
        assert set(sent) == {expected}
    else:
        assert_error_body(response, status=422, code="VALIDATION")
        assert response.json()["error"]["param"] == "days"
        assert wired.reader.calls == []
        assert wired.prom.calls == []


@pytest.mark.parametrize("path", DAYS_PATHS)
@pytest.mark.parametrize("days", ["abc", "1.5", "", "30d"])
def test_a_non_integer_days_is_422_naming_days(path: str, days: str, wired: Wired) -> None:
    """Catches `days` typed as a string or a float - a string reaches PromQL as a window
    like `[30dd]`, a float as `[1.5d]` - and a parser that falls back to the default on a
    value it cannot read instead of refusing it."""
    response = wired.get(path, days=days)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "days"
    assert wired.reader.calls == []
    assert wired.prom.calls == []


# --- tenant --------------------------------------------------------------------------


@pytest.mark.parametrize("path", TENANT_PATHS)
def test_the_tenant_length_boundary_is_inclusive(path: str, wired: Wired) -> None:
    """Catches an off-by-one in the tenant length pre-check (`>=` refuses a name of exactly
    `MAX_IDENTIFIER_LENGTH`) and a missing one: neither name is a policy tenant and the
    stubs accept anything, so without the check the over-cap row answers 200."""
    assert len(MAX_TENANT) == MAX_IDENTIFIER_LENGTH

    at_cap = wired.get(path, tenant=MAX_TENANT)
    assert at_cap.status_code == 200, at_cap.text
    assert [kwargs["tenant"] for _, kwargs in wired.reader.calls] == [MAX_TENANT]

    wired.reader.calls.clear()
    wired.prom.calls.clear()

    over = wired.get(path, tenant=MAX_TENANT + "a")
    assert_error_body(over, status=422, code="VALIDATION")
    assert over.json()["error"]["param"] == "tenant"
    assert wired.reader.calls == []
    assert wired.prom.calls == []


TENANT_REJECTION_ROWS: list[tuple[str, str]] = [
    ("double quote", 'nordlicht"'),
    ("closing brace", "nordlicht}"),
    ("trailing newline", "nordlicht\n"),
    ("trailing CRLF", "nordlicht\r\n"),
    ("trailing space", "nordlicht "),
    ("a matcher breakout", 'nordlicht"} or vector(1) #'),
]


@pytest.mark.parametrize("path", TENANT_PATHS)
@pytest.mark.parametrize(
    ("case", "tenant"), TENANT_REJECTION_ROWS, ids=[row[0] for row in TENANT_REJECTION_ROWS]
)
def test_the_tenant_rejection_table(path: str, case: str, tenant: str, wired: Wired) -> None:
    """Catches a route that skips admin's `_checked_tenant`, and a check written as `.match`
    with `$` (which lets a trailing newline through): every value here would otherwise reach
    the stub reader with a 200."""
    response = wired.get(path, tenant=tenant)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "tenant"
    assert wired.reader.calls == []
    assert wired.prom.calls == []


def test_the_tenant_filter_reaches_the_reader(wired: Wired) -> None:
    """Catches `?tenant=` accepted and dropped before the reader, which would show one
    tenant's cost page every tenant's spend."""
    cost = wired.get(COST, tenant="cuan")
    assert cost.status_code == 200, cost.text
    assert [kwargs["tenant"] for name, kwargs in wired.reader.calls if name == "cost"] == ["cuan"]


# --- the source flags -------------------------------------------------------------


def unconfigured_body(path: str) -> dict[str, Any]:
    """What a route answers with neither source configured and a fresh budget store."""
    body: dict[str, Any] = {"sources": {"postgres": UNCONFIGURED, "prometheus": UNCONFIGURED}}
    body.update(POSTGRES_EMPTY[path])
    body.update(PROMETHEUS_EMPTY[path])
    if path == QUALITY:
        # The gate reads the committed baseline, not a source, so its thresholds stay set.
        baseline = json.loads(SHIPPED_BASELINE.read_text(encoding="utf-8"))
        body["gate"] = [
            {
                "name": name,
                "label": label,
                "value": None,
                "threshold": baseline[name],
                "passed": None,
                "note": note,
            }
            for name, label, note in (
                ("check_score", "Check score", None),
                ("judge_score", "Judge score", JUDGE_NOTE),
            )
        ]
    if path == COST:
        body["estimated"] = True
        body["budgets"] = [
            {"tenant": "nordlicht", "cap_usd": 15.0, "spent_usd": 0.0, "ratio": 0.0},
            {"tenant": "cuan", "cap_usd": 60.0, "spent_usd": 0.0, "ratio": 0.0},
            {"tenant": "evals", "cap_usd": 10.0, "spent_usd": 0.0, "ratio": 0.0},
        ]
    return body


@pytest.mark.parametrize("path", ALL_PATHS)
def test_with_nothing_configured_every_route_is_200_and_both_sources_unconfigured(
    path: str, bare: TestClient
) -> None:
    """Catches a route that reads an unconfigured source anyway and reports `ok` with zeros
    (the unconfigured reader answers empty reads, so only the flag and the nulls tell them
    apart), a route that 500s without a database or a Prometheus, a pool opened for the
    reader when no `database_url` was given, and a body that omits a field instead of
    answering it empty."""
    app = bare.app
    assert isinstance(app, FastAPI)
    assert app.state.db_pool is None
    assert isinstance(app.state.console_reader, UnconfiguredConsoleReader)
    assert app.state.console_reader.configured is False
    assert app.state.prometheus is None

    response = bare.get(path, headers=ADMIN_HEADERS)

    assert response.status_code == 200, response.text
    assert response.json() == unconfigured_body(path)
    assert set(console.SourceState) == {"ok", "unconfigured", "error"}


@pytest.mark.parametrize("path", PROMETHEUS_PATHS)
@pytest.mark.parametrize("mode", ["timeout", "500", "error-status"])
def test_a_failing_prometheus_is_200_with_prometheus_error_and_an_error_log(
    path: str, mode: str, wired: Wired, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a `PrometheusError` escaping as a 500, one swallowed without an ERROR log (a
    dead Prometheus nobody hears about), one reported as `unconfigured` or `ok`, stale or
    partial Prometheus fields left in the body, and a failure that blanks the POSTGRES
    fields too - they must survive exactly as the reader answered them."""
    wired.prom.mode = mode

    with caplog.at_level(logging.DEBUG):
        response = wired.get(path)

    assert response.status_code == 200, response.text
    body = response.json()
    postgres_state = UNCONFIGURED if path == RELIABILITY else OK
    assert body["sources"] == {"postgres": postgres_state, "prometheus": ERROR}
    assert subset(body, PROMETHEUS_EMPTY[path]) == PROMETHEUS_EMPTY[path]
    assert error_records(caplog) != []

    if path == HOME:
        assert body["requests_today"] == 42
        assert body["open_incidents"] == 2


@pytest.mark.parametrize("path", POSTGRES_PATHS)
def test_a_failing_reader_is_200_with_postgres_error_and_an_error_log(
    path: str, wired: Wired, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a `psycopg.Error` escaping as a 500 (the whole page lost to one dead
    database), one swallowed without an ERROR log, one reported as `ok` with the fields
    left empty, and a failure that blanks the PROMETHEUS fields or the budgets, which read
    neither the reader nor its flag."""
    wired.reader.error = psycopg.OperationalError("connection refused")

    with caplog.at_level(logging.DEBUG):
        response = wired.get(path)

    assert response.status_code == 200, response.text
    body = response.json()
    prometheus_state = OK if path in PROMETHEUS_PATHS else UNCONFIGURED
    assert body["sources"] == {"postgres": ERROR, "prometheus": prometheus_state}
    assert subset(body, POSTGRES_EMPTY[path]) == POSTGRES_EMPTY[path]
    assert error_records(caplog) != []

    if path == HOME:
        assert body["p95_seconds"] == 0.734
        assert body["error_rate"] == 0.012
    if path == COST:
        assert body["estimated"] is True
        assert [row["tenant"] for row in body["budgets"]] == ["nordlicht", "cuan", "evals"]


# --- the bodies ------------------------------------------------------------------


def test_the_home_body_reads_both_sources(wired: Wired) -> None:
    """Catches a home route that swaps `p95_seconds` and `error_rate`, reads either from
    the range API instead of an instant query, drops a reader field, or reports a source
    that answered as anything but `ok`."""
    response = wired.get(HOME)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "sources": {"postgres": OK, "prometheus": OK},
        "requests_today": 42,
        "cost_usd_today": 1.25,
        "p95_seconds": 0.734,
        "error_rate": 0.012,
        "last_eval": {
            "id": "run-a",
            "finished_at": HOME_READ.model_dump(mode="json")["last_eval"]["finished_at"],
            "passed": True,
            "check_score": 0.91,
        },
        "open_incidents": 2,
    }
    assert sorted(wired.prom.queries(QUERY_PATH)) == sorted([P95_QUERY, ERROR_RATE_QUERY])
    assert wired.prom.queries(QUERY_RANGE_PATH) == []


def test_an_empty_home_sample_is_null_and_still_ok(wired: Wired) -> None:
    """Catches a home route that indexes the first sample unguarded - an IndexError, so a
    500 on a Prometheus with no traffic yet - or that turns "no sample" into `0.0`, which
    reads as a perfect error rate, or into `prometheus: error`."""
    wired.prom.mode = "empty"

    response = wired.get(HOME)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": OK, "prometheus": OK}
    assert body["p95_seconds"] is None
    assert body["error_rate"] is None


def test_the_cost_body_carries_the_reader_answer_and_is_estimated(wired: Wired) -> None:
    """Catches a cost route that drops or renames a reader field, labels savings as measured
    (`estimated` is always true, decision D), reports Prometheus as anything but
    `unconfigured`, or asks Prometheus anything at all."""
    response = wired.get(COST, days=14)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": OK, "prometheus": UNCONFIGURED}
    assert body["rollup_populated"] is True
    assert body["days"] == [row.model_dump(mode="json") for row in COST_READ.days]
    assert body["savings"] == [row.model_dump(mode="json") for row in COST_READ.savings]
    assert body["savings_usd_total"] == 0.5
    assert body["estimated"] is True
    assert body["cost_per_request"] == [
        row.model_dump(mode="json") for row in COST_READ.cost_per_request
    ]
    assert wired.prom.calls == []


def test_the_quality_body_carries_the_reader_answer(wired: Wired) -> None:
    """Catches a quality route that drops or renames one of its five lists or the gate, reorders
    the reader's rows, or reports Prometheus as anything but `unconfigured`."""
    response = wired.get(QUALITY)

    assert response.status_code == 200, response.text
    body = response.json()
    # The note is a fixed sentence the gate tests pin; here only that the judge row has one.
    notes = [row.pop("note") for row in body["gate"]]
    assert notes[0] is None
    assert "kappa 0.7" in notes[1]
    baseline = json.loads(SHIPPED_BASELINE.read_text(encoding="utf-8"))
    assert body == {
        "sources": {"postgres": OK, "prometheus": UNCONFIGURED},
        "runs": [row.model_dump(mode="json") for row in QUALITY_READ.runs],
        "release_markers": [row.model_dump(mode="json") for row in QUALITY_READ.release_markers],
        "categories": [row.model_dump(mode="json") for row in QUALITY_READ.categories],
        "taxonomy": [row.model_dump(mode="json") for row in QUALITY_READ.taxonomy],
        "blocked": [],
        "gate": [
            {
                "name": "check_score",
                "label": "Check score",
                "value": 0.91,
                "threshold": baseline["check_score"],
                "passed": False,
            },
            {
                "name": "judge_score",
                "label": "Judge score",
                "value": None,
                "threshold": baseline["judge_score"],
                "passed": None,
            },
        ],
    }
    assert wired.prom.calls == []


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows in a fixed order, for lists whose order the plan does not state."""
    return sorted(rows, key=lambda row: sorted((key, str(value)) for key, value in row.items()))


def test_the_reliability_body_reads_the_named_series(wired: Wired) -> None:
    """Catches circuit states rendered as numbers or with a wrong name mapping (0 closed, 1
    half_open, 2 open), an unknown state value passed through instead of skipped, series
    swapped between latency and error rate, a range query not over `now - days` to `now`
    at step 3600, and a fallbacks window that ignores `days`."""
    before = time.time()
    response = wired.get(RELIABILITY, days=7)
    after = time.time()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": UNCONFIGURED, "prometheus": OK}
    assert body["latency_p95"] == [[SAMPLE_TS, 0.42], [SAMPLE_TS + 3600, 0.51]]
    assert body["error_rate"] == [[SAMPLE_TS, 0.01]]
    assert sort_rows(body["circuits"]) == sort_rows(
        [
            {"provider": "anthropic", "state": "closed"},
            {"provider": "local", "state": "half_open"},
            {"provider": "openai", "state": "open"},
        ]
    )
    assert body["fallbacks"] == [
        {"tenant": "evals", "alias": "agent-default", "provider": "local", "count": 4.0}
    ]
    assert set(body) == {"sources", "latency_p95", "error_rate", "circuits", "fallbacks", "slos"}

    # An earlier change (a recorded deviation from its section 8): the budget queries join the
    # instant queries, so the exact list gains them; their values are pinned in
    # tests/gateway/test_console_slo.py.
    assert sorted(wired.prom.queries(QUERY_PATH)) == sorted(
        [
            CIRCUITS_QUERY,
            "sum by (tenant, alias, provider) (increase(gateway_fallbacks_total[7d]))",
            AVAILABILITY_BUDGET_QUERY,
            LATENCY_BUDGET_QUERY,
        ]
    )
    ranged = [params for sent, params in wired.prom.calls if sent == QUERY_RANGE_PATH]
    assert sorted(params["query"] for params in ranged) == sorted([P95_QUERY, ERROR_RATE_QUERY])
    for params in ranged:
        start, end = float(params["start"]), float(params["end"])
        assert before - 1 <= end <= after + 1
        assert abs((end - start) - 7 * 86400) < 1
        assert params["step"] == "3600"
    assert wired.reader.calls == []


ROW_ROUTES: list[tuple[str, str, Callable[[], list[dict[str, Any]]]]] = [
    (INCIDENTS, "incidents", lambda: [row.model_dump(mode="json") for row in INCIDENT_ROWS]),
    (RELEASES, "releases", lambda: [row.model_dump(mode="json") for row in RELEASE_ROWS]),
]


@pytest.mark.parametrize(("path", "key", "expected"), ROW_ROUTES, ids=["incidents", "releases"])
def test_the_incidents_and_releases_bodies_carry_the_reader_rows(
    path: str, key: str, expected: Callable[[], list[dict[str, Any]]], wired: Wired
) -> None:
    """Catches a timeline or deploys route that drops a column (`open`, `eval_passed`),
    renames its list, or reports Prometheus as anything but `unconfigured`."""
    response = wired.get(path)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "sources": {"postgres": OK, "prometheus": UNCONFIGURED},
        key: expected(),
    }
    assert wired.prom.calls == []


# --- budgets ---------------------------------------------------------------------


def test_budgets_read_the_policy_cap_and_the_store_spend(wired: Wired) -> None:
    """Catches a cap read from anywhere but the tenant's `budget_usd_month`, a spend treated
    as dollars rather than micro-USD (12 000 000 would read as twelve million), an unknown
    spend (`None`, Redis down) rendered as 0 instead of null, a month key other than the
    current UTC month, a ratio not spend over cap, a `?tenant=` the budgets ignore, and an
    unknown tenant answering someone else's budget."""
    store = SpendStore({"nordlicht": 12_000_000, "cuan": None, "evals": 2_500_000})
    wired.app.state.budget_store = store
    months = {current_month()}

    response = wired.get(COST)
    months.add(current_month())

    assert response.status_code == 200, response.text
    assert response.json()["budgets"] == [
        {"tenant": "nordlicht", "cap_usd": 15.0, "spent_usd": 12.0, "ratio": 0.8},
        {"tenant": "cuan", "cap_usd": 60.0, "spent_usd": None, "ratio": None},
        {"tenant": "evals", "cap_usd": 10.0, "spent_usd": 2.5, "ratio": 0.25},
    ]
    assert {tenant for tenant, _ in store.asked} == {"nordlicht", "cuan", "evals"}
    assert {month for _, month in store.asked} <= months

    one = wired.get(COST, tenant="evals")
    assert one.status_code == 200, one.text
    assert one.json()["budgets"] == [
        {"tenant": "evals", "cap_usd": 10.0, "spent_usd": 2.5, "ratio": 0.25}
    ]

    unknown = wired.get(COST, tenant="nobody")
    assert unknown.status_code == 200, unknown.text
    assert unknown.json()["budgets"] == []

    wired.reader.error = psycopg.OperationalError("connection refused")
    down = wired.get(COST, tenant="evals")
    assert down.json()["budgets"] == one.json()["budgets"]


# --- regressions -------------------------------------------------------------------


def test_the_higher_is_better_metrics_are_pinned() -> None:
    """Catches a lower-is-better metric (a latency, a cost, a count) added to the list, which
    would report every speed-up as a regression, and a listed accuracy dropped from it, which
    would hide a real drop on a blocked release."""
    assert HIGHER_IS_BETTER_METRICS == (
        "check_score",
        "judge_score",
    )


REGRESSION_ROWS: list[
    tuple[str, Mapping[str, object] | None, Mapping[str, object] | None, tuple[Regression, ...]]
] = [
    (
        "a listed metric lower than before",
        {"check_score": 0.8},
        {"check_score": 0.85},
        (Regression(metric="check_score", value=0.8, previous=0.85),),
    ),
    ("equal is not a regression", {"check_score": 0.85}, {"check_score": 0.85}, ()),
    ("higher is not a regression", {"check_score": 0.9}, {"check_score": 0.85}, ()),
    ("an unlisted metric lower than before", {"cost_usd": 0.5}, {"cost_usd": 0.9}, ()),
    ("a bool current value", {"check_score": False}, {"check_score": True}, ()),
    ("a bool previous value", {"judge_score": 0}, {"judge_score": True}, ()),
    ("a string current value", {"judge_score": "0.5"}, {"judge_score": 0.9}, ()),
    ("a null previous value", {"judge_score": 0.5}, {"judge_score": None}, ()),
    (
        "integers count as numbers",
        {"judge_score": 0},
        {"judge_score": 1},
        (Regression(metric="judge_score", value=0.0, previous=1.0),),
    ),
    ("absent from the current run", {}, {"check_score": 0.9}, ()),
    ("absent from the previous run", {"check_score": 0.1}, {}, ()),
    ("no current metrics", None, {"check_score": 0.9}, ()),
    ("no previous metrics", {"check_score": 0.1}, None, ()),
    (
        "ordered by the list, not the mapping",
        {"judge_score": 0.7, "check_score": 0.8},
        {"judge_score": 0.9, "check_score": 0.85},
        (
            Regression(metric="check_score", value=0.8, previous=0.85),
            Regression(metric="judge_score", value=0.7, previous=0.9),
        ),
    ),
]


@pytest.mark.parametrize(
    ("case", "current", "previous", "expected"),
    REGRESSION_ROWS,
    ids=[row[0] for row in REGRESSION_ROWS],
)
def test_regressed_metrics_decision_table(
    case: str,
    current: Mapping[str, object] | None,
    previous: Mapping[str, object] | None,
    expected: tuple[Regression, ...],
) -> None:
    """Catches each way the delta can be computed wrongly, one row per control: `<=` for
    `<`, no list filter (a cheaper run reads as a regression), `bool` accepted as a number
    (it is an `int` subclass), a string compared lexically, a missing side or a `None`
    mapping raising, and iteration in the mapping's order rather than the list's."""
    assert regressed_metrics(current, previous) == expected


def test_a_regressed_metric_is_one_lower_than_the_previous_passed_run(wired: Wired) -> None:
    """Catches a quality route that passes the blocked release through without computing
    `regressed`, computes it against the wrong side (current and previous swapped reads the
    rise in `check_score` as the regression), leaks the raw `metrics` maps into the
    body, or fails on a release with no linked run or no earlier passed run."""
    linked = BlockedRelease(
        git_sha="bad0001",
        deployed_at=DEPLOYED,
        pr_url="https://github.com/example/repo/pull/7",
        eval_run_id="run-blocked",
        metrics={
            "judge_score": 0.7,
            "check_score": 0.99,
            "p95_latency_ms": 400,
        },
        previous_metrics={
            "judge_score": 0.9,
            "check_score": 0.9,
            "p95_latency_ms": 900,
        },
    )
    first_ever = BlockedRelease(
        git_sha="bad0002",
        deployed_at=DEPLOYED,
        pr_url=None,
        eval_run_id="run-first",
        metrics={"check_score": 0.1},
        previous_metrics=None,
    )
    unlinked = BlockedRelease(
        git_sha="bad0003",
        deployed_at=DEPLOYED,
        pr_url=None,
        eval_run_id=None,
        metrics=None,
        previous_metrics=None,
    )
    wired.reader.quality_read = QUALITY_READ.model_copy(
        update={"blocked": (linked, first_ever, unlinked)}
    )

    response = wired.get(QUALITY)

    assert response.status_code == 200, response.text
    deployed_at = linked.model_dump(mode="json")["deployed_at"]
    assert response.json()["blocked"] == [
        {
            "git_sha": "bad0001",
            "deployed_at": deployed_at,
            "pr_url": "https://github.com/example/repo/pull/7",
            "eval_run_id": "run-blocked",
            "regressed": [
                {"metric": "judge_score", "value": 0.7, "previous": 0.9},
            ],
        },
        {
            "git_sha": "bad0002",
            "deployed_at": deployed_at,
            "pr_url": None,
            "eval_run_id": "run-first",
            "regressed": [],
        },
        {
            "git_sha": "bad0003",
            "deployed_at": deployed_at,
            "pr_url": None,
            "eval_run_id": None,
            "regressed": [],
        },
    ]
