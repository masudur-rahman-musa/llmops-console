"""`GET /metrics`: the Prometheus scrape.

Unauthenticated, like `/healthz` and `/readyz`. It is what
every Prometheus scrape config assumes, and once an earlier change writes the
NetworkPolicies the endpoint is reachable from inside the cluster only. The cost
is honest and it is on the known-issues list until then: tenant names are label
values, so a `/metrics` exposed to the internet discloses the tenant list.

The path is deliberately outside `/v1/`, which is what keeps a scrape every
fifteen seconds out of `gateway_requests_total` without a rule of its own
(`test_the_probe_and_scrape_paths_are_not_counted`).
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from opsconsole.core.metrics import registry

METRICS = "/metrics"
"""The published scrape path. Pinned against the mounted route by
`test_metrics_answers_the_exposition_content_type`."""

router = APIRouter()


@router.get(METRICS)
def scrape() -> Response:
    """Render the module registry in the text exposition format, with its content type.

    The content type carries the format version, and a Prometheus scrape reads it
    before the body: a target answering a bare `text/plain` is dropped, so every
    panel in the design would be empty while the endpoint answered 200 to a
    browser (`test_metrics_answers_the_exposition_content_type`).

    Declared `def` rather than `async def`: rendering walks every sample in the
    registry, and FastAPI runs a `def` route in a worker thread rather than on the
    event loop, the same reason the model routes give.
    """
    return Response(content=generate_latest(registry()), media_type=CONTENT_TYPE_LATEST)
