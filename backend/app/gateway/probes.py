"""The readiness probes: one per configured backing service, plus the providers.

`/readyz` is what takes a pod out of the load balancer, so what it checks is a
decision rather than a detail. Four probes live here and every one of them obeys
the same three rules.

It answers a bool and never raises. `gateway/health.py` catches too, but a probe
that relied on that would put the reason for a failure in a log line that says
only "readiness probe raised": each factory catches its own client's error and
logs it with the service name (`test_the_redis_probe_answers_a_bool_for_every_client`,
`test_the_postgres_probe_answers_a_bool_for_every_connection`,
`test_the_qdrant_probe_answers_false_when_the_call_raises`,
`test_the_provider_probe_needs_one_adapter_that_answers_healthy`).

It is bounded. Every client this module builds carries `PROBE_TIMEOUT_SECONDS`
on both the connect and the read, because a backing service that accepts a
connection and then never answers is the shape that hangs readiness itself and
takes a healthy pod out of rotation for the wrong reason
(`test_the_redis_client_bounds_both_the_connect_and_the_read`,
`test_the_qdrant_client_bounds_every_phase_of_the_call`).

It asks the server something. A probe that only opens a connection passes
against a Postgres still replaying WAL and a Qdrant still loading its
collections, so the Postgres probe runs `SELECT 1` and the Qdrant probe reads
the status of a real request
(`test_the_postgres_probe_runs_a_statement_rather_than_only_connecting`,
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
checked nothing. The `/readyz` body names every probe that ran, which is what
makes the difference visible from outside
(`test_register_probes_registers_the_configured_stores_and_always_the_provider`,
`test_a_refused_redis_makes_readyz_503_naming_every_probe`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Final, Protocol

import httpx
import psycopg
import redis
from fastapi import FastAPI

from app.gateway.health import Probe, ProbeRegistry
from app.providers.base import Adapter

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
"""How long any one readiness check may take, connecting and reading, in seconds.

Pinned exactly by `test_the_probe_timeout_is_exactly_two_seconds` and asserted on
the built clients by `test_the_redis_client_bounds_both_the_connect_and_the_read`
and `test_the_qdrant_client_bounds_every_phase_of_the_call`. Kubernetes gives a
readiness probe its own timeout as well; this one exists so the process cannot be
the thing that hangs."""

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


class SupportsPing(Protocol):
    """The one method the Redis probe is allowed to use.

    Narrow on purpose: the probe takes a client rather than a URL so a test can
    hand it a double, and a protocol with one method is what stops the probe
    growing a second call that a double would then have to reimplement.
    """

    def ping(self) -> object: ...


class SupportsExecute(Protocol):
    """A database connection that runs a statement and releases itself.

    Structural, so `psycopg.Connection` satisfies it without this module
    depending on that class in its signatures, and a recording double satisfies
    it without a live server.
    """

    def execute(self, statement: str) -> object: ...

    def __enter__(self) -> SupportsExecute: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


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


def build_redis_client(url: str) -> redis.Redis:
    """A Redis client for `url`, bounded on both the connect and the read.

    `socket_timeout` is the half that matters and the half a library default
    leaves unset: a server that accepts the connection and never answers is what
    hangs a probe (`test_the_redis_client_bounds_both_the_connect_and_the_read`).

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, and a parametrised annotation fails `mypy --strict` with `type-arg`.
    """
    return redis.Redis.from_url(
        url,
        socket_connect_timeout=PROBE_TIMEOUT_SECONDS,
        socket_timeout=PROBE_TIMEOUT_SECONDS,
    )


def build_qdrant_client() -> httpx.Client:
    """An `httpx.Client` whose four timeout phases all carry `PROBE_TIMEOUT_SECONDS`.

    `httpx.Timeout(x)` sets connect, read, write and pool together; the library
    default bounds the connect at 5s and leaves the read unbounded, which is the
    hang this exists to prevent
    (`test_the_qdrant_client_bounds_every_phase_of_the_call`).
    """
    return httpx.Client(timeout=httpx.Timeout(PROBE_TIMEOUT_SECONDS))


def build_postgres_connect(url: str) -> Callable[[], SupportsExecute]:
    """A callable opening one bounded connection to `url`.

    Two bounds, because libpq has two. `connect_timeout` covers the handshake and
    is whole seconds, so the float is floored to an int here rather than silently
    truncated by the driver. `statement_timeout` covers the query, in
    milliseconds, and is what stops `SELECT 1` waiting forever on a server that
    connected and then stalled.
    """
    connect_seconds = max(1, int(PROBE_TIMEOUT_SECONDS))
    statement_ms = int(PROBE_TIMEOUT_SECONDS * 1000)

    def connect() -> SupportsExecute:
        return psycopg.connect(
            url,
            connect_timeout=connect_seconds,
            options=f"-c statement_timeout={statement_ms}",
        )

    return connect


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
    registry: ProbeRegistry = app.state.probes

    if settings.redis_url is not None:
        registry.register(REDIS_PROBE, redis_probe(build_redis_client(settings.redis_url)))
    if settings.database_url is not None:
        registry.register(
            POSTGRES_PROBE, postgres_probe(build_postgres_connect(settings.database_url))
        )
    if settings.qdrant_url is not None:
        registry.register(QDRANT_PROBE, qdrant_probe(build_qdrant_client(), settings.qdrant_url))
    registry.register(PROVIDER_PROBE, provider_probe(app.state.adapters))
