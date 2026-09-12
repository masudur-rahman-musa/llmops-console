"""`GET /metrics`: the Prometheus scrape.

Unauthenticated, like `/healthz` and `/readyz`. It is what
every Prometheus scrape config assumes, and it is served in-cluster only: the
`gateway` Ingress routes `/v1` and nothing wider, so no public router matches
this path and a request for it from the internet is Traefik's own 404
(`tests/infra/test_k8s_apps.py::test_no_public_router_matches_an_admin_path_or_the_scrape`).
That matters because tenant names are label values, so a `/metrics` reachable
from the internet would disclose the tenant list. A path with no router is not a
control this endpoint carries - it is the absence of a route to it, which is why
the in-cluster listener is unchanged and every pod allowed to open a connection
still reads the exposition here.

Two network facts sit under that and neither is a path rule. `apps`'s
`default-deny` closes ingress and `gateway-ingress` admits `kube-system` and pods
in `apps`, so Traefik itself can still open a connection to this port - a policy
decides which pods may connect and never which path they may ask for. And
`apps`'s `gateway-ingress` admits the `monitoring` namespace on this same port, which is the one thing the scrape needed here: no port, no Service
and no probe changed for it.

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
