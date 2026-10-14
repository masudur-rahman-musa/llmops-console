"""Liveness, readiness, the readiness probes, and the Prometheus scrape.

`/healthz` says the process is up and checks nothing else, so a dependency
outage never takes the container down (pinned by `test_registries_are_per_app`,
which asserts 200 from `/healthz` while `/readyz` is 503). `/readyz` runs the
registered probes and returns 503 the moment one of them fails, which is what
takes the pod out of the load balancer.

The body names every probe that RAN, which is the half of the contract an empty
registry makes visible: `{"checks": {}}` is a readiness that checked nothing, and
an operator can see that rather than having to infer it from a green 200
(`test_a_refused_redis_makes_readyz_503_naming_every_probe`). Beside `checks` it
carries `residency_mode`, so a process running the opt-in relaxed mode cannot be
mistaken for a strict one by anything that reads the endpoint.
A top-level key rather than another entry in `checks`, because `checks` means "a
probe ran and this is its verdict" and readiness is `all(checks.values())` - a
string filed in there would read as a passing probe
(`test_readyz_reports_the_residency_mode`).

Four probes live here and every one of them obeys the same three rules.

It answers a bool and never raises. `_run` catches too, but a probe that relied
on that would put the reason for a failure in a log line that says only
"readiness probe raised": each factory catches its own client's error and logs it
with the service name (`test_the_redis_probe_answers_a_bool_for_every_client`,
`test_the_postgres_probe_answers_a_bool_for_every_connection`,
`test_the_qdrant_probe_answers_false_when_the_call_raises`,
`test_the_provider_probe_needs_one_adapter_that_answers_healthy`).

It is bounded. The clients the probes run over come from `core/clients.py`, each
carrying `PROBE_TIMEOUT_SECONDS` on connect and read.

It asks the server something. A probe that only opens a connection passes against
a Postgres still replaying WAL and a Qdrant still loading its collections, so the
Postgres probe runs `SELECT 1` and the Qdrant probe reads the status of a real
request (`test_the_postgres_probe_runs_a_statement_rather_than_only_connecting`,
`test_the_qdrant_probe_reads_the_status`).

The provider probe is the exception to that last rule, and it is honest about it
here rather than in a plan file: `Adapter.healthcheck()` currently answers True
without calling out, so today this probe proves only that at least one adapter
was BUILT - which is what tells an operator that no provider key reached the
process. The circuit breaker did NOT change that: it learns from the outcomes of
real requests through the routes' `is_healthy` seam and calls `healthcheck()`
nowhere, so a provider it has taken out of rotation still reads green here. Read
a green `provider` as "the gateway has somewhere to send a request", not as "a
provider answered".

Registration is configured-only: a store probe exists when
its URL is set and does not exist otherwise, while the provider probe is always
registered. An unconfigured store that registered a failing probe would make
`/readyz` red forever on a machine that never wanted that store, and one that
registered a passing probe would be worse - readiness would be green having
checked nothing
(`test_register_probes_registers_the_configured_stores_and_always_the_provider`).

`GET /metrics` is the Prometheus scrape, unauthenticated like `/healthz` and
`/readyz`. It is served in-cluster only: the `gateway`
Ingress routes `/v1` and nothing wider, so no public router matches this path and
a request for it from the internet is Traefik's own 404
(`tests/infra/test_k8s_apps.py::test_no_public_router_matches_an_admin_path_or_the_scrape`).
That matters because tenant names are label values. The path is deliberately
outside `/v1/`, which keeps a scrape every fifteen seconds out of
`gateway_requests_total` without a rule of its own
(`test_the_probe_and_scrape_paths_are_not_counted`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Final

import httpx
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.clients import (
    SupportsExecute,
    SupportsPing,
    build_postgres_connect,
    build_qdrant_client,
    build_redis_client,
)
from app.core.metrics import registry
from app.model_routing.policy import ResidencyMode
from app.providers.base import Adapter

logger = logging.getLogger(__name__)

Probe = Callable[[], bool]

READY_STATUS = 200
NOT_READY_STATUS = 503

REDIS_PROBE: Final[str] = "redis"
POSTGRES_PROBE: Final[str] = "postgres"
QDRANT_PROBE: Final[str] = "qdrant"
PROVIDER_PROBE: Final[str] = "provider"
"""The four keys the `/readyz` body publishes. They are read by an operator and
by whatever alerts on readiness, so they are API rather than internal labels;
pinned by `test_the_probe_names_are_the_ones_the_readyz_body_publishes`."""

QDRANT_READY_PATH: Final[str] = "/readyz"
"""Qdrant's own readiness path. Its `/healthz` answers 200 while collections are
still loading, which is exactly the state this probe has to refuse; pinned by
`test_the_qdrant_readiness_path_is_the_services_own`."""

PROBE_STATUS_FLOOR: Final[int] = 400
"""The first HTTP status a probe reads as not ready. Pinned by the 399/400 pair in
`test_the_qdrant_probe_reads_the_status`."""

POSTGRES_PROBE_STATEMENT: Final[str] = "SELECT 1"

METRICS = "/metrics"
"""The published scrape path. Pinned against the mounted route by
`test_metrics_answers_the_exposition_content_type`."""


class ProbeRegistry:
    """The readiness probes an app checks.

    Iteration follows registration order, and a second registration under an
    existing name replaces the first. Both are pinned by
    `test_registering_the_same_name_replaces_the_probe`.
    """

    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}

    def register(self, name: str, fn: Probe) -> None:
        """Register `fn` under `name`, replacing any probe already at that name."""
        self._probes[name] = fn

    def __iter__(self) -> Iterator[tuple[str, Probe]]:
        return iter(self._probes.items())


def _run(probe: Probe) -> bool:
    """Run one probe. A probe that raises is a probe that failed."""
    try:
        return bool(probe())
    except Exception:
        # `/readyz` only ever reports a bool, so the cause has to be logged here
        # or it is lost to whoever is looking at a pod that will not go ready.
        logger.exception("readiness probe raised")
        return False


def redis_probe(client: SupportsPing) -> Probe:
    """A probe answering whether `client` still responds to `PING`.

    Asked on every call rather than once: readiness is polled precisely so a
    Redis that went away since the last check is noticed
    (`test_the_redis_probe_asks_the_server_every_time_it_is_called`).
    """

    def probe() -> bool:
        try:
            return bool(client.ping())
        except Exception:
            logger.warning("readiness: redis probe failed", exc_info=True)
            return False

    return probe


def postgres_probe(connect: Callable[[], SupportsExecute]) -> Probe:
    """A probe that opens a connection, runs `SELECT 1`, and releases it either way.

    The connection is opened per call rather than held, because a probe holding
    an idle connection through an outage is a probe that reports on a socket
    rather than on a server. `with` is what releases it on the failing path too
    (`test_the_postgres_probe_releases_the_connection_even_when_the_statement_fails`).
    """

    def probe() -> bool:
        try:
            with connect() as connection:
                connection.execute(POSTGRES_PROBE_STATEMENT)
        except Exception:
            logger.warning("readiness: postgres probe failed", exc_info=True)
            return False
        return True

    return probe


def qdrant_probe(client: httpx.Client, base_url: str) -> Probe:
    """A probe reading the status of `GET {base_url}{QDRANT_READY_PATH}`.

    Ready is any status below `PROBE_STATUS_FLOOR`, not `== 200`: Qdrant answers
    a 2xx that is not 200 on some builds, and a 503 while it loads is the case
    this must refuse (`test_the_qdrant_probe_reads_the_status`).
    """

    def probe() -> bool:
        try:
            response = client.get(base_url + QDRANT_READY_PATH)
        except Exception:
            logger.warning("readiness: qdrant probe failed", exc_info=True)
            return False
        return response.status_code < PROBE_STATUS_FLOOR

    return probe


def provider_probe(adapters: Mapping[str, Adapter]) -> Probe:
    """A probe answering whether at least ONE built adapter calls itself healthy.

    `any`, not `all`: a gateway with a dead US provider and a live EU one can
    still serve every request the residency rules let it serve, so failing
    readiness there would take out a pod that works. An empty mapping is False -
    a process that built no adapter has nowhere to send anything - and an adapter
    whose `healthcheck()` raises counts as unhealthy rather than stopping the
    walk (`test_the_provider_probe_needs_one_adapter_that_answers_healthy`).

    What this proves today is narrower than its name: see the module docstring.
    """

    def probe() -> bool:
        for name, adapter in adapters.items():
            try:
                healthy = bool(adapter.healthcheck())
            except Exception:
                logger.warning("readiness: adapter %s raised on healthcheck", name, exc_info=True)
                continue
            if healthy:
                return True
        return False

    return probe


def register_probes(app: FastAPI) -> None:
    """Register the probes this app's settings configure, plus the provider probe.

    Reads `app.state.settings`, `app.state.adapters` and `app.state.probes`, so it
    is callable against any app carrying those three and is exercised against a
    hand-built one rather than only through `create_app`
    (`test_register_probes_registers_the_configured_stores_and_always_the_provider`).

    The provider probe closes over the adapter MAPPING rather than over a copy of
    its values, so a policy reload that rebuilds the adapters in place
    is seen by a probe registered before it.
    """
    settings = app.state.settings
    registry_: ProbeRegistry = app.state.probes

    if settings.redis_url is not None:
        registry_.register(REDIS_PROBE, redis_probe(build_redis_client(settings.redis_url)))
    if settings.database_url is not None:
        registry_.register(
            POSTGRES_PROBE, postgres_probe(build_postgres_connect(settings.database_url))
        )
    if settings.qdrant_url is not None:
        registry_.register(QDRANT_PROBE, qdrant_probe(build_qdrant_client(), settings.qdrant_url))
    registry_.register(PROVIDER_PROBE, provider_probe(app.state.adapters))


router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: 200 as long as the process is serving. Pinned by `test_healthz_ok`."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> Response:
    """Readiness: 200 when every registered probe passes, 503 naming the failures.

    An empty registry is ready. Pinned by `test_readyz_ok_when_all_probes_pass`
    and `test_readyz_503_when_a_probe_fails`. The body also states this process's
    `residency_mode`, which is settings and not a probe, so it never affects the
    status (`test_readyz_reports_the_residency_mode`).
    """
    probes: ProbeRegistry = request.app.state.probes
    mode: ResidencyMode = request.app.state.settings.residency_mode
    checks = {name: _run(probe) for name, probe in probes}
    ready = all(checks.values())
    return JSONResponse(
        status_code=READY_STATUS if ready else NOT_READY_STATUS,
        # The mode is reported at both statuses: a relaxed process that is also
        # unready is exactly the one nobody should mistake for strict.
        content={"checks": checks, "residency_mode": mode.value},
    )


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
