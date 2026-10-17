"""Liveness, readiness, the readiness probes, and the Prometheus scrape.

`/healthz` says the process is up and checks nothing else, so a dependency
outage never takes the container down: `/healthz` stays 200 while `/readyz` is 503. `/readyz` runs
the
registered probes and returns 503 the moment one of them fails, which is what
takes the pod out of the load balancer.

The body names every probe that RAN, which is the half of the contract an empty
registry makes visible: `{"checks": {}}` is a readiness that checked nothing, and
an operator can see that rather than having to infer it from a green 200. Beside `checks` it
carries `residency_mode`, so a process running the opt-in relaxed mode cannot be
mistaken for a strict one by anything that reads the endpoint.
A top-level key rather than another entry in `checks`, because `checks` means "a
probe ran and this is its verdict" and readiness is `all(checks.values())` - a
string filed in there would read as a passing probe.

Four probes live here and every one of them obeys the same three rules.

It answers a bool and never raises. `_run` catches too, but a probe that relied
on that would put the reason for a failure in a log line that says only
"readiness probe raised": each factory catches its own client's error and logs it
with the service name.

It is bounded. The clients the probes run over come from `core/clients.py`, each
carrying `PROBE_TIMEOUT_SECONDS` on connect and read.

It asks the server something. A probe that only opens a connection passes against
a Postgres still replaying WAL and a Qdrant still loading its collections, so the
Postgres probe runs `SELECT 1` and the Qdrant probe reads the status of a real
request.

The provider probe is the exception to that last rule, and it is honest about it
here rather than in a plan file: it proves only that the adapter mapping is
non-empty, which tells an operator that at least one provider key reached the
process. The circuit breaker (`providers/breaker.py`) learns from the outcomes
of real requests through the routes' `is_healthy` seam; a provider it has taken
out of rotation still reads green on the readiness probe. Read a green
`provider` as "the gateway has somewhere to send a request", not as "a provider
answered".

Registration is configured-only: a store probe exists when
its URL is set and does not exist otherwise, while the provider probe is always
registered. An unconfigured store that registered a failing probe would make
`/readyz` red forever on a machine that never wanted that store, and one that
registered a passing probe would be worse - readiness would be green having
checked nothing.

`GET /metrics` is the Prometheus scrape, unauthenticated like `/healthz` and
`/readyz`. It is served in-cluster only: the `gateway`
Ingress routes `/v1` and nothing wider, so no public router matches this path and
a request for it from the internet is Traefik's own 404.
That matters because tenant names are label values. The path is deliberately
outside `/v1/`, which keeps a scrape every fifteen seconds out of
`gateway_requests_total` without a rule of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from typing import Final, Protocol

import httpx
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.clients import SupportsExecute, SupportsPing
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
by whatever alerts on readiness, so they are API rather than internal labels."""

QDRANT_READY_PATH: Final[str] = "/readyz"
"""Qdrant's own readiness path. Its `/healthz` answers 200 while collections are
still loading, which is exactly the state this probe has to refuse."""

PROBE_STATUS_FLOOR: Final[int] = 400
"""The first HTTP status a probe reads as not ready."""

POSTGRES_PROBE_STATEMENT: Final[str] = "SELECT 1"

METRICS = "/metrics"
"""The published scrape path."""


class ProbeRegistry:
    """The readiness probes an app checks.

    Iteration follows registration order, and a second registration under an
    existing name replaces the first.
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
    Redis that went away since the last check is noticed.
    """

    def probe() -> bool:
        try:
            return bool(client.ping())
        except Exception:
            logger.warning("readiness: redis probe failed", exc_info=True)
            return False

    return probe


class SupportsConnection(Protocol):
    """A connection pool the Postgres probe borrows one connection from.

    Structural, so the shared `psycopg_pool.ConnectionPool` satisfies it without
    this module depending on that class, and a recording double satisfies it
    without a live server. `connection()` is the pool's own context manager: it
    checks a connection out, yields it, and returns it to the pool on exit, which
    is what keeps the probe from leaking a connection on the failing path too.
    """

    def connection(self) -> AbstractContextManager[SupportsExecute]: ...


def postgres_probe(pool: SupportsConnection) -> Probe:
    """A probe that borrows a pooled connection, runs `SELECT 1`, and returns it either way.

    The connection is borrowed from the SHARED pool per call rather than held,
    because a probe holding an idle connection through an outage is a probe that
    reports on a socket rather than on a server. The pool's `connection()` context
    manager is what returns it on the failing path too.
    One pool serves the writers and this probe, so a down database is what makes
    this probe fail.
    """

    def probe() -> bool:
        try:
            with pool.connection() as connection:
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
    this must refuse.
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
    """A probe answering whether at least one adapter was built.

    True when the mapping is non-empty; False when empty. No adapter method is
    called: the gateway is ready to serve as soon as it has somewhere to send
    a request, which is exactly what a non-empty mapping means. Circuit breaker
    state (`providers/breaker.py`) tracks whether each provider is healthy at
    request time.


    """

    def probe() -> bool:
        return bool(adapters)

    return probe


def register_probes(app: FastAPI) -> None:
    """Register a probe over each shared client the lifespan built, plus the provider probe.

    Run by `lifespan` once the shared clients are on `app.state`, so each store
    probe runs over the very `redis.Redis`, Qdrant `httpx.Client` or Postgres pool
    the stores, caches and writers use - not a second one minted here. Registration is still
    configured-only: a client exists only when its URL is
    set, so a store probe exists exactly when its service is configured and the
    provider probe is always registered.

    Reads `app.state.settings`, `app.state.adapters`, `app.state.probes` and the
    shared clients, so it is callable against any app carrying those and is
    exercised against a hand-built one rather than only through the lifespan.

    The provider probe closes over the adapter MAPPING rather than over a copy of
    its values, so a policy reload that rebuilds the adapters in place
    is seen by a probe registered before it.
    """
    settings = app.state.settings
    registry_: ProbeRegistry = app.state.probes

    if settings.redis_url is not None:
        registry_.register(REDIS_PROBE, redis_probe(app.state.redis_client))
    if settings.database_url is not None:
        registry_.register(POSTGRES_PROBE, postgres_probe(app.state.db_pool))
    if settings.qdrant_url is not None:
        registry_.register(QDRANT_PROBE, qdrant_probe(app.state.qdrant_client, settings.qdrant_url))
    registry_.register(PROVIDER_PROBE, provider_probe(app.state.adapters))


router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: 200 as long as the process is serving."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> Response:
    """Readiness: 200 when every registered probe passes, 503 naming the failures.

    An empty registry is ready. The body also states this process's
    `residency_mode`, which is settings and not a probe, so it never affects the
    status.
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
    dashboard panel would be empty while the endpoint answered 200 to a
    browser.

    Declared `def` rather than `async def`: rendering walks every sample in the
    registry, and FastAPI runs a `def` route in a worker thread rather than on the
    event loop, the same reason the model routes give.
    """
    return Response(content=generate_latest(registry()), media_type=CONTENT_TYPE_LATEST)
