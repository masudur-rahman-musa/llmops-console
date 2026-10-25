"""`app.services.prometheus_reads`: the named queries and the two parsers behind them.

No backing service. Every case talks to an `httpx.MockTransport`, so the request
the client sends and the body it parses are both written out here, and the only
code between them is the module under test.

Two claims are pinned, each case naming the wrong implementation it catches:

* The PromQL is NAMED and fixed. Every query string is
  written out as a literal, with and without a tenant matcher, because a
  caller-supplied fragment or a matcher dropped from one builder is invisible to
  every other test.
* A failure is ONE exception type. A timeout, a transport error, a non-200, a
  body whose `status` is not `success` and a body of the wrong shape all raise
  `PrometheusError`, which is what lets a route turn any of them into the
  `prometheus: error` flag without a 500.

The contract these cases are written against, before it exists:

* `PROMETHEUS_TIMEOUT_SECONDS = 2.0`, `RANGE_STEP_SECONDS = 3600`,
  `QUERY_PATH = "/api/v1/query"`, `QUERY_RANGE_PATH = "/api/v1/query_range"`;
* `build_prometheus_http(base_url) -> httpx.Client`, `PrometheusError`;
* `PrometheusClient(http)` with `instant_query(query)`,
  `range_query(query, *, start, end)` and `close()`;
* `P95_QUERY`, `ERROR_RATE_QUERY`, `CIRCUITS_QUERY`
  and `fallbacks_query(days)`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.services.prometheus_reads import (
    CIRCUITS_QUERY,
    ERROR_RATE_QUERY,
    P95_QUERY,
    PROMETHEUS_TIMEOUT_SECONDS,
    QUERY_PATH,
    QUERY_RANGE_PATH,
    RANGE_STEP_SECONDS,
    PrometheusClient,
    PrometheusError,
    build_prometheus_http,
    fallbacks_query,
)

BASE_URL = "http://prometheus.test"

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """A mock Prometheus answering one fixed response and recording every request."""

    def __init__(self, respond: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)


def client_for(handler: Handler) -> PrometheusClient:
    """A `PrometheusClient` whose every request goes to `handler`."""
    return PrometheusClient(httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler)))


def success(result_type: str, result: list[dict[str, Any]]) -> dict[str, Any]:
    """A well-formed Prometheus API body."""
    return {"status": "success", "data": {"resultType": result_type, "result": result}}


def json_response(body: dict[str, Any], status: int = 200) -> Handler:
    """A handler answering `body` with `status`, whatever was asked."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return respond


VECTOR_BODY = success(
    "vector",
    [
        {"metric": {"provider": "anthropic"}, "value": [1757980800.0, "1.5"]},
        {"metric": {"provider": "local"}, "value": [1757980800.0, "0"]},
    ],
)

MATRIX_BODY = success(
    "matrix",
    [
        {"metric": {}, "values": [[1757980800, "0.25"], [1757984400, "0.5"]]},
        {"metric": {"le": "second-series"}, "values": [[1757980800, "99"]]},
    ],
)

START = 1757894400.0
END = 1757984400.0


def run_query(client: PrometheusClient, method: str) -> object:
    """Call one of the two query methods with fixed arguments."""
    if method == "instant":
        return client.instant_query(P95_QUERY)
    return client.range_query(P95_QUERY, start=START, end=END)


METHODS = ["instant", "range"]


# --- the client ---------------------------------------------------------------


def test_the_timeout_is_two_seconds() -> None:
    """Catches `build_prometheus_http` left on httpx's default five-second timeout (or none),
    which lets one dead Prometheus hold every console page for longer than the page's own
    budget; it also catches a builder that ignores `base_url`, since every relative query
    path would then go nowhere."""
    assert PROMETHEUS_TIMEOUT_SECONDS == 2.0

    http = build_prometheus_http(BASE_URL)
    try:
        assert http.timeout == httpx.Timeout(2.0)
        assert str(http.base_url).rstrip("/") == BASE_URL
    finally:
        http.close()


def test_an_instant_vector_is_parsed_to_label_value_pairs() -> None:
    """Catches an instant parser that keeps the value as the string Prometheus sends, drops
    the label set, or queries some path other than `/api/v1/query` with the PromQL in a
    parameter other than `query`."""
    recorder = Recorder(json_response(VECTOR_BODY))
    client = client_for(recorder)

    assert client.instant_query(CIRCUITS_QUERY) == [
        ({"provider": "anthropic"}, 1.5),
        ({"provider": "local"}, 0.0),
    ]

    assert QUERY_PATH == "/api/v1/query"
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "GET"
    assert request.url.path == QUERY_PATH
    assert request.url.params["query"] == CIRCUITS_QUERY
    client.close()


def test_a_matrix_is_parsed_to_unix_ts_value_pairs() -> None:
    """Catches a range parser that merges every series into one list (the second series
    carries a value no first-series point has), that keeps timestamps or values as strings,
    or that sends `start` and `end` under other names or to a path other than
    `/api/v1/query_range`; and catches an empty matrix raising instead of answering `[]`."""
    recorder = Recorder(json_response(MATRIX_BODY))
    client = client_for(recorder)

    assert client.range_query(ERROR_RATE_QUERY, start=START, end=END) == [
        (1757980800.0, 0.25),
        (1757984400.0, 0.5),
    ]

    assert QUERY_RANGE_PATH == "/api/v1/query_range"
    request = recorder.requests[0]
    assert request.method == "GET"
    assert request.url.path == QUERY_RANGE_PATH
    assert request.url.params["query"] == ERROR_RATE_QUERY
    assert float(request.url.params["start"]) == START
    assert float(request.url.params["end"]) == END

    empty = client_for(json_response(success("matrix", [])))
    assert empty.range_query(ERROR_RATE_QUERY, start=START, end=END) == []
    client.close()
    empty.close()


def test_the_range_step_is_3600_seconds() -> None:
    """Catches a range query sent without `step` (Prometheus refuses it) or at another
    resolution, which changes how many points a 90-day reliability chart carries."""
    assert RANGE_STEP_SECONDS == 3600

    recorder = Recorder(json_response(MATRIX_BODY))
    client = client_for(recorder)
    client.range_query(P95_QUERY, start=START, end=END)

    assert recorder.requests[0].url.params["step"] == "3600"
    client.close()


@pytest.mark.parametrize("non_finite", ["NaN", "+Inf", "-Inf"])
def test_a_non_finite_sample_is_dropped(non_finite: str) -> None:
    """Catches a parser that turns `"NaN"` or `"+Inf"` into a float and passes it on, which
    either fails JSON rendering in the route (NaN is not JSON) or plots a spike that never
    happened; the finite neighbours in the same body must survive, so a parser that drops
    the whole body on the first bad sample is caught too."""
    instant = client_for(
        json_response(
            success(
                "vector",
                [
                    {"metric": {"provider": "anthropic"}, "value": [1757980800, non_finite]},
                    {"metric": {"provider": "local"}, "value": [1757980800, "2"]},
                ],
            )
        )
    )
    assert instant.instant_query(CIRCUITS_QUERY) == [({"provider": "local"}, 2.0)]

    ranged = client_for(
        json_response(
            success(
                "matrix",
                [
                    {
                        "metric": {},
                        "values": [
                            [1757980800, "0.25"],
                            [1757984400, non_finite],
                            [1757988000, "0.75"],
                        ],
                    }
                ],
            )
        )
    )
    assert ranged.range_query(P95_QUERY, start=START, end=END) == [
        (1757980800.0, 0.25),
        (1757988000.0, 0.75),
    ]
    instant.close()
    ranged.close()


# --- failures -----------------------------------------------------------------


def raising(error: Exception) -> Handler:
    """A handler that raises what the transport would raise."""

    def respond(request: httpx.Request) -> httpx.Response:
        raise error

    return respond


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("timed out"), httpx.ConnectTimeout("timed out"), httpx.ConnectError("no")],
    ids=["read timeout", "connect timeout", "connect error"],
)
def test_a_timeout_is_error(method: str, error: Exception) -> None:
    """Catches an httpx timeout or transport error escaping the client as itself, which a
    route catching only `PrometheusError` would turn into a 500 for the whole page."""
    client = client_for(raising(error))
    with pytest.raises(PrometheusError):
        run_query(client, method)
    client.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("status", [400, 500, 503])
def test_a_non_200_is_error(method: str, status: int) -> None:
    """Catches a client that parses the body without looking at the status: the body sent
    with the error status here is a WELL-FORMED success body of the right result type, so
    the status check is the only thing that can refuse it."""
    body = VECTOR_BODY if method == "instant" else MATRIX_BODY
    client = client_for(json_response(body, status))
    with pytest.raises(PrometheusError):
        run_query(client, method)
    client.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("status_value", ["error", "warning", "", None])
def test_a_non_success_status_body_is_error(method: str, status_value: str | None) -> None:
    """Catches a client that reads `data.result` whenever it is present: every body here
    carries a valid result of the right type beside a `status` that is not `success`, so
    only the `status` check can refuse it."""
    body = dict(VECTOR_BODY if method == "instant" else MATRIX_BODY)
    body["status"] = status_value
    client = client_for(json_response(body))
    with pytest.raises(PrometheusError):
        run_query(client, method)
    client.close()


MALFORMED_ROWS: list[tuple[str, str, Handler]] = [
    ("instant given a matrix", "instant", json_response(MATRIX_BODY)),
    ("range given a vector", "range", json_response(VECTOR_BODY)),
    ("instant given a scalar", "instant", json_response(success("scalar", []))),
    ("instant with no data", "instant", json_response({"status": "success"})),
    ("range with no data", "range", json_response({"status": "success"})),
    (
        "instant value not a number",
        "instant",
        json_response(success("vector", [{"metric": {}, "value": [1757980800, "abc"]}])),
    ),
    (
        "range value not a pair",
        "range",
        json_response(success("matrix", [{"metric": {}, "values": [[1757980800]]}])),
    ),
    ("instant not json", "instant", lambda request: httpx.Response(200, text="<html>")),
    ("range not json", "range", lambda request: httpx.Response(200, text="<html>")),
]


@pytest.mark.parametrize(
    ("case", "method", "handler"), MALFORMED_ROWS, ids=[row[0] for row in MALFORMED_ROWS]
)
def test_an_unexpected_result_type_or_malformed_body_is_error(
    case: str, method: str, handler: Handler
) -> None:
    """Catches a parser that trusts the body's shape: a KeyError, IndexError, ValueError or
    JSON decode error escaping as itself is a 500 on the console page, and a matrix read as a
    vector (or the reverse) silently answers the wrong numbers."""
    client = client_for(handler)
    with pytest.raises(PrometheusError):
        run_query(client, method)
    client.close()


# --- the named queries ----------------------------------------------------------


def test_the_named_queries_are_pinned() -> None:
    """Catches any drift in the PromQL the console's numbers come from - a `[5m]` rate where
    the page says a day, a `by` clause missing a label the page groups on - and, for the
    builder, a `days` window that ignores its argument."""
    assert P95_QUERY == (
        "histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket[1d])))"
    )
    assert ERROR_RATE_QUERY == (
        'sum(rate(gateway_requests_total{status=~"5.."}[1d]))'
        " / sum(rate(gateway_requests_total[1d]))"
    )
    assert CIRCUITS_QUERY == "max by (provider) (gateway_circuit_state)"

    assert fallbacks_query(7) == (
        "sum by (tenant, alias, provider) (increase(gateway_fallbacks_total[7d]))"
    )
    assert fallbacks_query(90) == (
        "sum by (tenant, alias, provider) (increase(gateway_fallbacks_total[90d]))"
    )
