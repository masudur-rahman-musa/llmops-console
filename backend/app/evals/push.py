"""Pushing one run's `gateway_eval_*` gauges to a Pushgateway.

`core/metrics.py` deliberately never registers `gateway_eval_*`: the
nightly job pushes those gauges rather than have the gateway process serve them, so
this module owns a registry of its own, built fresh on every call rather than
shared across runs the way `core/metrics.py`'s process-lifetime one is - there is
no long-running process here to hold one open.

The `httpx.Client` is injected for the reason `judge.JudgeClient`'s and
`target.TargetClient`'s are: the whole request is pinned against
`httpx.MockTransport` with nothing real listening. A push failure - the
Pushgateway unreachable, or answering a non-2xx - is reported as a string and
never raised: this is a side channel the CLI's own exit code does not depend on
(`app.cli.evals`'s own `_store_rows` answers a database failure the same way, though
unlike that one a pushgateway failure never changes the caller's exit code).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

from app.evals.scores import METRIC_NAMES

PUSHGATEWAY_JOB: Final[str] = "eval_nightly"
"""The fixed job label every push carries, never derived from a tenant, a run kind or any
other caller-supplied value, so a push cannot introduce a new identifier."""


def push_metrics(
    metrics: Mapping[str, object], pushgateway_url: str, *, client: httpx.Client
) -> str | None:
    """Push one run's non-null `gateway_eval_*` gauges, or answer what went wrong.

    A key whose value is `None` (a `judge_score` when nothing was judged) contributes no
    series at all: a `Gauge` left at its unset zero would read as a genuine score of zero
    to `NightlyEvalRegression`. Only the two published scores can become gauges
    (`scores.METRIC_NAMES`); any other key, old or new, is never pushed. Never raises for a
    transport failure or a non-2xx response; both answer a short string.
    """
    registry = CollectorRegistry()
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if value is None:
            continue
        Gauge(f"gateway_eval_{name}", f"Nightly eval metric: {name}.", registry=registry).set(
            float(value)  # type: ignore[arg-type]
        )

    url = f"{pushgateway_url.rstrip('/')}/metrics/job/{PUSHGATEWAY_JOB}"
    try:
        response = client.put(
            url,
            content=generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )
    except httpx.HTTPError as exc:
        return f"the nightly eval's metrics were not pushed to {pushgateway_url}: {exc}"
    if not response.is_success:
        return (
            f"the nightly eval's metrics were not pushed to {pushgateway_url}: "
            f"{response.status_code}"
        )
    return None
