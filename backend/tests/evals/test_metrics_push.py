r"""`evals/push.py`: pushing one run's `gateway_eval_*` gauges to a Pushgateway.

The contract these cases are written against:

* `PUSHGATEWAY_JOB: Final[str] = "eval_nightly"` - the fixed job label, pinned by value.
* `push_metrics(metrics, pushgateway_url, *, client) -> str | None`, taking a
  `Mapping[str, object]` shaped like `results.compute_scores`'s output and an injected
  `httpx.Client`. It answers `None` on a successful push or a short string on any way the
  push failed, and never raises for a transport failure.
* Exactly three gauges can exist, one per `scores.METRIC_NAMES` key (`safety`,
  `outcome_every_play`, `outcome_at_least_once`), named `gateway_eval_safety` and so on. A key
  whose value is `None` contributes no series at all: a gauge left at its zero default
  would read as a real score of zero to `NightlyEvalRegression`. Any other key, old or
  new, is never pushed.
* A single PUT to `f"{pushgateway_url.rstrip('/')}/metrics/job/{PUSHGATEWAY_JOB}"`.

The exposition text is parsed and compared by NAME rather than checked with `in`, and every
transport-failure case sends a complete, well-formed request, so only the response or
transport behaviour named in the row can flip the result.
"""

from __future__ import annotations

import httpx
import prometheus_client
import pytest

from app.evals.push import PUSHGATEWAY_JOB, push_metrics

PUSHGATEWAY_URL = "http://pushgateway.example:9091"

#: A small metrics dict: the two published scores (one of them null) plus a count and one
#: key of the old format, which must never become a gauge. Names are read BY NAME, not by
#: iterating a fixed dict, so the exposition text is parsed and compared per name.
METRICS: dict[str, object] = {
    "outcome_every_play": 87.88,
    "outcome_at_least_once": None,
    "cases": 33,
    "decision_accuracy": 88.0,
}


def parsed_exposition(body: bytes) -> dict[str, float]:
    """The exposition text as `{family_name: value}`, ignoring comment (`#`) lines.

    Parsed rather than scanned with `in`: a metric name that is a prefix of another
    (`cost_usd_total` / `cost_usd_per_turn`) would let a substring check pass on the wrong
    line entirely.
    """
    values: dict[str, float] = {}
    for line in body.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        assert name, f"a non-comment exposition line has no name: {line!r}"
        values[name] = float(value)
    return values


class Transport:
    """A recording `httpx.MockTransport` handler, one scripted response or exception."""

    def __init__(self, response_or_exception: httpx.Response | Exception) -> None:
        self._answer = response_or_exception
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


# --------------------------------------------------------------------------
# the fixed job label and the URL it is pushed to
# --------------------------------------------------------------------------


def test_the_job_label_is_the_published_constant() -> None:
    """Catches `PUSHGATEWAY_JOB` drifting off the value the CronJob manifest
    and `NightlyEvalRegression`'s alert both assume a nightly push carries - a renamed
    constant here silently orphans every future rule written against `job="eval_nightly"`."""
    assert PUSHGATEWAY_JOB == "eval_nightly"


def test_the_push_targets_the_fixed_job_path_and_nothing_from_the_caller() -> None:
    """Catches a URL built from anything other than the fixed constant - a job label derived
    from a tenant or a run-kind argument would be a new externally-shaped identifier this
    suite's acceptance criteria forbid, and the
    signature carries no such argument at all so the only way to fail this is to read the
    label off `pushgateway_url` or invent a second one. One PUT, to exactly the path built
    from the fixed job constant."""
    transport = Transport(httpx.Response(200))

    result = push_metrics(METRICS, PUSHGATEWAY_URL, client=transport.client())

    assert result is None
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == "PUT"
    assert str(request.url) == f"{PUSHGATEWAY_URL}/metrics/job/{PUSHGATEWAY_JOB}"


def test_a_pushgateway_url_with_a_trailing_slash_is_not_doubled() -> None:
    """Catches a naive f-string join producing `.../9091//metrics/job/...`, which several
    Pushgateway/Prometheus HTTP servers 404 on: the trailing slash has to be stripped before
    the fixed path is appended."""
    transport = Transport(httpx.Response(200))

    push_metrics(METRICS, f"{PUSHGATEWAY_URL}/", client=transport.client())

    assert str(transport.requests[0].url) == f"{PUSHGATEWAY_URL}/metrics/job/{PUSHGATEWAY_JOB}"


# --------------------------------------------------------------------------
# what is actually pushed
# --------------------------------------------------------------------------


def test_push_sends_the_scores_that_exist() -> None:
    """Catches two wrong implementations at once, on the same small dict, by NAME rather than
    by substring: one that emits a `gateway_eval_outcome_at_least_once 0.0` line for the `None`
    key - a Gauge left at its unset default reads as a genuine score of zero to the alert
    this run feeds - and one that drops a present key entirely, e.g. by iterating
    `METRIC_NAMES` and calling `metrics[name]` in a way that only reads a subset. Values are
    compared as floats."""
    transport = Transport(httpx.Response(200))

    push_metrics(METRICS, PUSHGATEWAY_URL, client=transport.client())

    body = transport.requests[0].content
    values = parsed_exposition(body)
    assert values.get("gateway_eval_outcome_every_play") == pytest.approx(87.88)
    assert "gateway_eval_outcome_at_least_once" not in values


def test_the_content_type_is_prometheus_clients_own_exposition_type() -> None:
    """Catches a hand-written `Content-Type` (e.g. plain `text/plain`) that a real Pushgateway
    accepts leniently but that silently drops the metric TYPE information a stricter server
    would reject on - the module contract names `prometheus_client.CONTENT_TYPE_LATEST`
    exactly, so this pins the value against the installed library rather than a guess."""
    transport = Transport(httpx.Response(200))

    push_metrics(METRICS, PUSHGATEWAY_URL, client=transport.client())

    sent = transport.requests[0].headers.get("content-type", "")
    assert sent.split(";")[0] == prometheus_client.CONTENT_TYPE_LATEST.split(";")[0]


# --------------------------------------------------------------------------
# the unhappy path: a push failure is reported, never raised, never fatal
# --------------------------------------------------------------------------


def test_a_transport_failure_answers_a_string_and_does_not_raise() -> None:
    """The unhappy path, half one: a Pushgateway that is unreachable. Catches
    `httpx.ConnectError` escaping `push_metrics` and aborting the whole CLI run over a
    metrics side-channel that was never load-bearing for the run's own exit code."""
    transport = Transport(httpx.ConnectError("connection refused"))

    result = push_metrics(METRICS, PUSHGATEWAY_URL, client=transport.client())

    assert result is not None
    assert isinstance(result, str)


def test_a_non_2xx_response_answers_a_string_and_does_not_raise() -> None:
    """The unhappy path, half two: a Pushgateway that answers but refuses the push (a
    500, or a 400 on a malformed job label). Catches an implementation that only guards the
    transport call and lets `response.raise_for_status()` propagate on a non-2xx reply."""
    transport = Transport(httpx.Response(500, text="internal error"))

    result = push_metrics(METRICS, PUSHGATEWAY_URL, client=transport.client())

    assert result is not None
    assert isinstance(result, str)


def test_a_successful_push_answers_none() -> None:
    """The success half of the same contract, without which the two failure cases above would
    pass against an implementation that always returns a string - a constant "ok" sentinel
    would also read as falsy-string-truthy and could slip past a check that only asserts
    `is not None` on failure."""
    transport = Transport(httpx.Response(200))

    result = push_metrics(METRICS, PUSHGATEWAY_URL, client=transport.client())

    assert result is None


def test_all_scores_are_pushed_when_they_exist_and_nothing_else_is() -> None:
    """Catches a push of the old metric names (`gateway_eval_decision_accuracy` would keep the
    nightly alert reading a series nobody writes) and one that skips a score. Only the
    three published names appear, so a count or an old key never becomes a gauge."""
    transport = Transport(httpx.Response(200))

    push_metrics(
        {
            "safety": 100.0,
            "outcome_every_play": 90.0,
            "outcome_at_least_once": 95.5,
            "cases": 33,
            "check_score": 88.0,
        },
        PUSHGATEWAY_URL,
        client=transport.client(),
    )

    values = parsed_exposition(transport.requests[0].content)
    assert values == {
        "gateway_eval_safety": pytest.approx(100.0),
        "gateway_eval_outcome_every_play": pytest.approx(90.0),
        "gateway_eval_outcome_at_least_once": pytest.approx(95.5),
    }
