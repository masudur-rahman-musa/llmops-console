"""The console's Prometheus client: named queries only, and one failure type.

No caller supplies PromQL. Every query the console runs is
a constant or a builder in this module, and every one is pinned as a literal by
`test_the_named_queries_are_pinned`, so a changed metric name or window shows up
as a failing test rather than as an empty panel.

The only client-supplied text that reaches a query is `tenant`, inside
`redactions_query`'s label matcher. It arrives here already pre-checked by
`routes.admin._checked_tenant`, a lowercase slug under `re.fullmatch`, which is
what keeps a `"` or a `}` from closing the matcher
(`test_the_tenant_rejection_table`). This module does not re-check it; the route
is the one boundary a client string crosses.

Every failure is `PrometheusError`: a timeout, a transport error, a non-200, a
body whose `status` is not `success`, and a body of the wrong shape
(`test_a_timeout_is_error`, `test_a_non_200_is_error`,
`test_a_non_success_status_body_is_error`,
`test_an_unexpected_result_type_or_malformed_body_is_error`). The routes turn that
one type into `prometheus: error` beside the data.

A non-finite sample is dropped, not returned. `histogram_quantile` over no traffic
answers `NaN` and a ratio of two zero rates does too; JSON has no spelling for
either, and a panel is better served by a missing point than by a 500
(`test_a_non_finite_sample_is_dropped`).
"""

from __future__ import annotations

import math
from typing import Final

import httpx

PROMETHEUS_TIMEOUT_SECONDS: Final[float] = 2.0
"""Every phase of every call. The console waits on this per query, so a slow
Prometheus costs a page load seconds rather than a hung worker
(`test_the_timeout_is_two_seconds`)."""

RANGE_STEP_SECONDS: Final[int] = 3600
"""One point per hour on the reliability series (`test_the_range_step_is_3600_seconds`)."""

QUERY_PATH: Final[str] = "/api/v1/query"
QUERY_RANGE_PATH: Final[str] = "/api/v1/query_range"

SUCCESS: Final[str] = "success"
VECTOR: Final[str] = "vector"
MATRIX: Final[str] = "matrix"

P95_QUERY: Final[str] = (
    "histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket[1d])))"
)
"""Global p95 over the last day. `gateway_request_latency_seconds` carries no
`tenant` label, so no tenant filter can narrow it. Reused as the reliability
range series, where each hourly point is then a rolling day."""

ERROR_RATE_QUERY: Final[str] = (
    'sum(rate(gateway_requests_total{status=~"5.."}[1d])) / sum(rate(gateway_requests_total[1d]))'
)
"""The share of requests answered 5xx over the last day, reused the same way."""

CIRCUITS_QUERY: Final[str] = "max by (provider) (gateway_circuit_state)"
"""The worst breaker state per provider across replicas. The breaker is in-process, so two pods can disagree and the page shows the worse of them."""


AVAILABILITY_OBJECTIVE: Final[float] = 0.995
AVAILABILITY_ALLOWED_BAD: Final[float] = 0.005
LATENCY_THRESHOLD_SECONDS: Final[float] = 6.0
LATENCY_ALLOWED_SLOW: Final[float] = 0.05
"""The SLO numbers `project-obs/slo.md` documents, pinned by
`test_the_budget_queries_are_the_pinned_literals`. The allowed shares are what an
error budget spends: a p95 under 6 s allows 5% of requests at 6 s or over."""

AVAILABILITY_BUDGET_QUERY: Final[str] = (
    'sum(increase(gateway_requests_total{status=~"5.."}[30d]))'
    " / sum(increase(gateway_requests_total[30d]))"
)
"""The 5xx share over a fixed 30 days: the BAD ratio, which the route turns into
`observed = 1 - bad` and `budget_remaining = 1 - bad / 0.005`.

The three budget queries are literals rather than builders because Grafana's
`gateway-overview.json` carries each one byte for byte, and
`test_each_slo_panel_expr_equals_its_prometheus_constant` fails on drift. The window
never follows a page's `days` (`test_slos_ignore_days`)."""

LATENCY_BUDGET_QUERY: Final[str] = (
    "sum(increase(gateway_request_latency_seconds_bucket"
    '{le="6.0",provider!~"local|local_embed|vllm"}[30d]))'
    " / sum(increase(gateway_request_latency_seconds_count"
    '{provider!~"local|local_embed|vllm"}[30d]))'
)
"""The API tier's share of requests under 6 s over 30 days. Prometheus renders the
bucket label `6.0`, never `6`. Streamed requests are counted too: the histogram has
no streaming label, a gap `project-obs/slo.md` records."""

RESIDENCY_VIOLATIONS_30D_QUERY: Final[str] = (
    "sum(increase(gateway_residency_violations_total[30d]))"
)
"""Violations over 30 days. The target is zero, so there is no budget to spend."""


def redactions_query(days: int, tenant: str | None) -> str:
    """Placeholders minted per tenant and kind over the window, optionally for one tenant."""
    matcher = "" if tenant is None else f'{{tenant="{tenant}"}}'
    return f"sum by (tenant, kind) (increase(gateway_redactions_total{matcher}[{days}d]))"


def fallbacks_query(days: int) -> str:
    """Requests served by a later candidate over the window."""
    return f"sum by (tenant, alias, provider) (increase(gateway_fallbacks_total[{days}d]))"


class PrometheusError(Exception):
    """Prometheus could not answer this query. The message never reaches a response body."""


def build_prometheus_http(base_url: str) -> httpx.Client:
    """The `httpx.Client` a `PrometheusClient` sends over, bounded on every phase."""
    return httpx.Client(base_url=base_url, timeout=httpx.Timeout(PROMETHEUS_TIMEOUT_SECONDS))


def _finite(raw: object) -> float | None:
    """A sample value as a float, or `None` when it is NaN or infinite.

    Prometheus sends sample values as strings. Anything `float()` refuses is a
    malformed body and raises; `NaN` and `+Inf` parse and are dropped.
    """
    if not isinstance(raw, str | int | float) or isinstance(raw, bool):
        raise PrometheusError("a sample value is not a number")
    value = float(raw)
    return value if math.isfinite(value) else None


class PrometheusClient:
    """Instant and range queries over one bounded `httpx.Client`."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def _result(self, path: str, params: dict[str, str], result_type: str) -> list[object]:
        """The `data.result` list of one successful call, or `PrometheusError`."""
        try:
            response = self._http.get(path, params=params)
        except httpx.HTTPError as exc:
            raise PrometheusError(f"{path}: {type(exc).__name__}") from exc
        if response.status_code != httpx.codes.OK:
            raise PrometheusError(f"{path}: status {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise PrometheusError(f"{path}: the body is not JSON") from exc
        if not isinstance(body, dict) or body.get("status") != SUCCESS:
            raise PrometheusError(f"{path}: the body's status is not {SUCCESS}")
        data = body.get("data")
        if not isinstance(data, dict) or data.get("resultType") != result_type:
            raise PrometheusError(f"{path}: the result is not a {result_type}")
        result = data.get("result")
        if not isinstance(result, list):
            raise PrometheusError(f"{path}: the result is not a list")
        return result

    def instant_query(self, query: str) -> list[tuple[dict[str, str], float]]:
        """One `(labels, value)` pair per series, non-finite series dropped."""
        samples: list[tuple[dict[str, str], float]] = []
        try:
            for series in self._result(QUERY_PATH, {"query": query}, VECTOR):
                assert isinstance(series, dict)
                labels = series["metric"]
                assert isinstance(labels, dict)
                value = _finite(series["value"][1])
                if value is not None:
                    samples.append(({str(k): str(v) for k, v in labels.items()}, value))
        except (AssertionError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise PrometheusError(f"{QUERY_PATH}: a series is malformed") from exc
        return samples

    def range_query(self, query: str, *, start: float, end: float) -> list[tuple[float, float]]:
        """The first series' `(unix_ts, value)` points at `RANGE_STEP_SECONDS`.

        The console's range queries are all aggregated to one series, so the first
        is the only one; no series at all is an empty list.
        """
        params = {
            "query": query,
            "start": str(start),
            "end": str(end),
            "step": str(RANGE_STEP_SECONDS),
        }
        result = self._result(QUERY_RANGE_PATH, params, MATRIX)
        if not result:
            return []
        points: list[tuple[float, float]] = []
        try:
            series = result[0]
            assert isinstance(series, dict)
            for ts, raw in series["values"]:
                value = _finite(raw)
                if value is not None:
                    points.append((float(ts), value))
        except (AssertionError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise PrometheusError(f"{QUERY_RANGE_PATH}: a series is malformed") from exc
        return points

    def close(self) -> None:
        """Release the connection pool. Called by `app.lifespan` on shutdown."""
        self._http.close()
