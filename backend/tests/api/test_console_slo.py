"""An earlier change: the SLO error budgets `GET /console/reliability` serves as `slos`.

No backing service. The harness is `tests/gateway/test_console_routes.py`'s: the admin app
from `create_app`, the stub reader, and a real `PrometheusClient` over an
`httpx.MockTransport`. `SloPromStub` below is that module's `PromStub` with two per-query
knobs added on top, so a case can answer one budget query with a chosen value or fail it
alone, while every other query keeps the shared answers. The shared stub already answers the
three budget queries with a 5xx share of `0.001`, a share under 6 seconds of `0.98` and no
residency violation.

The contract these cases are written against, before it exists:

* `app.gateway.prometheus`: `AVAILABILITY_BUDGET_QUERY`, `LATENCY_BUDGET_QUERY`,
  `RESIDENCY_VIOLATIONS_30D_QUERY`, `AVAILABILITY_OBJECTIVE`, `AVAILABILITY_ALLOWED_BAD`,
  `LATENCY_THRESHOLD_SECONDS`, `LATENCY_ALLOWED_SLOW`;
* `ReliabilityPage.slos`: rows `{name, objective, window, observed, budget_remaining}`,
  in the order availability, latency, residency.

Float budgets are compared with `pytest.approx`: `1 - (1 - 0.98) / 0.05` is
`0.5999999999999996` in binary floating point.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.test_console_routes import (
    ERROR,
    FALLBACKS_PREFIX,
    OK,
    RELIABILITY,
    SAMPLE_TS,
    UNCONFIGURED,
    PromStub,
    StubReader,
    Wired,
    error_records,
)
from app.core.settings import Settings
from app.gateway import prometheus
from app.gateway.app import create_app
from app.gateway.prometheus import (
    AVAILABILITY_BUDGET_QUERY,
    LATENCY_BUDGET_QUERY,
    QUERY_PATH,
    RESIDENCY_VIOLATIONS_30D_QUERY,
)

from .conftest import NO_PROVIDER_KEYS

BUDGET_QUERIES = [AVAILABILITY_BUDGET_QUERY, LATENCY_BUDGET_QUERY, RESIDENCY_VIOLATIONS_30D_QUERY]

AVAILABILITY = "availability"
LATENCY = "latency"
RESIDENCY = "residency"


class SloPromStub(PromStub):
    """`PromStub`, plus `answers` (a query's sample values, replacing the shared answer) and
    `failing` (queries answered with a 500 while every other query succeeds)."""

    def __init__(self) -> None:
        super().__init__()
        self.answers: dict[str, list[str]] = {}
        self.failing: set[str] = set()

    def handle(self, request: httpx.Request) -> httpx.Response:
        query = dict(request.url.params).get("query", "")
        if request.url.path != QUERY_PATH or (
            query not in self.answers and query not in self.failing
        ):
            return super().handle(request)
        self.calls.append((request.url.path, dict(request.url.params)))
        if query in self.failing:
            return httpx.Response(500, text="internal error")
        result = [{"metric": {}, "value": [SAMPLE_TS, value]} for value in self.answers[query]]
        body = {"status": "success", "data": {"resultType": "vector", "result": result}}
        return httpx.Response(200, json=body)


@dataclass
class SloWired(Wired):
    """`Wired`, holding the stub under its own type so a case can set its knobs."""

    slo_prom: SloPromStub = field(default_factory=SloPromStub)


@pytest.fixture
def slo_wired(admin_settings: Settings) -> Iterator[SloWired]:
    """The admin app with the stub reader and `SloPromStub`, per case (in-process, cheap);
    the same arrangement as `test_console_routes.py`'s `wired`."""
    app = create_app(admin_settings, NO_PROVIDER_KEYS)
    reader = StubReader()
    prom = SloPromStub()
    app.state.console_reader = reader
    app.state.prometheus = prom.client()
    with TestClient(app) as client:
        yield SloWired(client=client, app=app, reader=reader, prom=prom, slo_prom=prom)


def slos_of(wired: SloWired, **params: str | int) -> list[dict[str, Any]]:
    """The `slos` of one successful reliability read, with Prometheus reported `ok`."""
    response = wired.get(RELIABILITY, **params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": UNCONFIGURED, "prometheus": OK}
    slos = body["slos"]
    assert isinstance(slos, list)
    return slos


def row(slos: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """The one row named `name`."""
    matched = [entry for entry in slos if entry["name"] == name]
    assert len(matched) == 1, f"{name} appears {len(matched)} times in {slos}"
    return matched[0]


def test_the_budget_queries_are_the_pinned_literals() -> None:
    """Catches a budget query that reads the GOOD share where the bad one is meant (or the
    reverse), a latency bucket label written `le="6"` (Prometheus renders `6.0`, so the ratio
    reads an empty series), a provider filter that lets the local tier in, a window other than
    `30d`, and an objective or allowed share drifting from the plan's numbers."""
    assert AVAILABILITY_BUDGET_QUERY == (
        'sum(increase(gateway_requests_total{status=~"5.."}[30d]))'
        " / sum(increase(gateway_requests_total[30d]))"
    )
    assert LATENCY_BUDGET_QUERY == (
        'sum(increase(gateway_request_latency_seconds_bucket{le="6.0",'
        'provider!~"local|local_embed|vllm"}[30d]))'
        " / sum(increase(gateway_request_latency_seconds_count"
        '{provider!~"local|local_embed|vllm"}[30d]))'
    )
    assert RESIDENCY_VIOLATIONS_30D_QUERY == (
        "sum(increase(gateway_residency_violations_total[30d]))"
    )
    assert prometheus.AVAILABILITY_OBJECTIVE == 0.995
    assert prometheus.AVAILABILITY_ALLOWED_BAD == 0.005
    assert prometheus.LATENCY_THRESHOLD_SECONDS == 6.0
    assert prometheus.LATENCY_ALLOWED_SLOW == 0.05


def test_slos_rows_are_availability_latency_residency_in_order(slo_wired: SloWired) -> None:
    """Catches a missing or extra row, rows in another order, a row renamed (`latency_p95`),
    an objective served as a percentage (`99.5`) or as the allowed share (`0.005`), a latency
    objective that is not the 6-second threshold, and a window other than `30d`."""
    slos = slos_of(slo_wired)

    assert [entry["name"] for entry in slos] == [AVAILABILITY, LATENCY, RESIDENCY]
    assert [(entry["objective"], entry["window"]) for entry in slos] == [
        (0.995, "30d"),
        (6.0, "30d"),
        (0.0, "30d"),
    ]
    for entry in slos:
        assert set(entry) == {"name", "objective", "window", "observed", "budget_remaining"}


def test_availability_budget_is_one_minus_bad_ratio_over_0_005(slo_wired: SloWired) -> None:
    """Catches `observed` served as the bad ratio itself (0.001) instead of `1 - bad`, and a
    budget computed as `observed / objective` (1.004) or as `1 - bad` (0.999) instead of
    `1 - bad / 0.005`: with a bad share of 0.001 only the right formula gives 0.8."""
    slo_wired.slo_prom.answers[AVAILABILITY_BUDGET_QUERY] = ["0.001"]

    availability = row(slos_of(slo_wired), AVAILABILITY)

    assert availability["observed"] == pytest.approx(0.999)
    assert availability["budget_remaining"] == pytest.approx(0.8)


def test_latency_budget_is_one_minus_slow_share_over_0_05(slo_wired: SloWired) -> None:
    """Catches the share under 6 seconds read as the SLOW share (observed 0.02), and a budget
    computed as `share / 0.95` (1.03) or against the availability allowance 0.005 (-3.0)
    instead of `1 - (1 - share) / 0.05`: with a share of 0.98 only that gives 0.6."""
    slo_wired.slo_prom.answers[LATENCY_BUDGET_QUERY] = ["0.98"]

    latency = row(slos_of(slo_wired), LATENCY)

    assert latency["observed"] == pytest.approx(0.98)
    assert latency["budget_remaining"] == pytest.approx(0.6)


def test_residency_budget_is_always_null(slo_wired: SloWired) -> None:
    """Catches a residency budget computed like the others (`1 - count`, or a ratio over an
    allowance), which would show a deny-by-default control as having budget to spend, and a
    violation count dropped or served as a ratio: two violations must read as `2.0` beside a
    null budget."""
    slo_wired.slo_prom.answers[RESIDENCY_VIOLATIONS_30D_QUERY] = ["2"]

    residency = row(slos_of(slo_wired), RESIDENCY)

    assert residency["observed"] == 2.0
    assert residency["budget_remaining"] is None


def test_an_overspent_budget_is_negative_and_unclamped(slo_wired: SloWired) -> None:
    """Catches a budget clamped at zero (`max(0, ...)`), which shows an SLO spent twice over
    exactly like one spent to the last request: a bad share of 0.01 against an allowance of
    0.005 must read -1.0."""
    slo_wired.slo_prom.answers[AVAILABILITY_BUDGET_QUERY] = ["0.01"]

    availability = row(slos_of(slo_wired), AVAILABILITY)

    assert availability["observed"] == pytest.approx(0.99)
    assert availability["budget_remaining"] == pytest.approx(-1.0)


def test_nan_gives_null_observed_and_budget(slo_wired: SloWired) -> None:
    """Catches "no traffic" turned into a number: a missing sample defaulted to 0 reads as a
    perfect availability with its whole budget left (observed 1.0, budget 1.0), and a latency
    share defaulted to 0 reads as every request slow. Availability answers `NaN` (0/0, which the
    client drops) and latency answers no series at all; residency still answers, so a route
    that nulls every row, or reports Prometheus as `error`, on one empty result fails too."""
    slo_wired.slo_prom.answers[AVAILABILITY_BUDGET_QUERY] = ["NaN"]
    slo_wired.slo_prom.answers[LATENCY_BUDGET_QUERY] = []
    slo_wired.slo_prom.answers[RESIDENCY_VIOLATIONS_30D_QUERY] = ["1"]

    slos = slos_of(slo_wired)

    for name in (AVAILABILITY, LATENCY):
        entry = row(slos, name)
        assert entry["observed"] is None, entry
        assert entry["budget_remaining"] is None, entry
    assert row(slos, RESIDENCY)["observed"] == 1.0


@pytest.mark.parametrize("failing", BUDGET_QUERIES, ids=[AVAILABILITY, LATENCY, RESIDENCY])
def test_a_failing_budget_query_is_200_with_empty_slos_prometheus_error_and_an_error_log(
    failing: str, slo_wired: SloWired, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a budget query run OUTSIDE the page's one Prometheus read: only that query fails
    here and every other query succeeds, so a route catching its failure separately answers
    null rows with Prometheus `ok` and no ERROR log, and a route not catching it at all answers
    500. It also catches partial rows left in `slos` beside the error flag."""
    slo_wired.slo_prom.failing.add(failing)

    with caplog.at_level(logging.DEBUG):
        response = slo_wired.get(RELIABILITY)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": UNCONFIGURED, "prometheus": ERROR}
    assert body["slos"] == []
    assert len(error_records(caplog)) == 1


def test_slos_ignore_days(slo_wired: SloWired) -> None:
    """Catches a budget query whose window follows `days` (`[1d]`, `[90d]`) instead of the
    fixed `30d`: the fallbacks query does follow `days`, so the two reads must differ there
    and send byte-identical budget queries, and serve the same `slos`."""
    short = slos_of(slo_wired, days=1)
    short_sent = slo_wired.slo_prom.queries(QUERY_PATH)
    slo_wired.slo_prom.calls.clear()

    long = slos_of(slo_wired, days=90)
    long_sent = slo_wired.slo_prom.queries(QUERY_PATH)

    def budget_sent(sent: list[str]) -> list[str]:
        return sorted(query for query in sent if not query.startswith(FALLBACKS_PREFIX))

    assert [query for query in short_sent if query.startswith(FALLBACKS_PREFIX)] != [
        query for query in long_sent if query.startswith(FALLBACKS_PREFIX)
    ]
    for query in BUDGET_QUERIES:
        assert short_sent.count(query) == 1, short_sent
        assert long_sent.count(query) == 1, long_sent
    assert budget_sent(short_sent) == budget_sent(long_sent)
    assert short == long
