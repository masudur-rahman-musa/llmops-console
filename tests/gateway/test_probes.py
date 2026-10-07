"""The readiness probes: one per backing service, over an injected client.

Every case here is arranged so that only the control it names can decide the
outcome, because `gateway/health.py`'s `_run` is a NEIGHBOURING guard that also
turns a raising probe into `False`. The probe factories are therefore called
DIRECTLY in every "raises" row - no `/readyz`, no registry - so a probe that
propagated its client's error would raise out of the test rather than quietly be
caught by the endpoint's net.

Four groups:

* the four probe factories over a fake client each: an answer that passes, an
  answer that fails, a client that raises, and a client that raises a TIMEOUT,
  which is the plan's hung-service risk and the row `_run` would otherwise be
  credited for;
* the two client builders, whose timeouts are pinned to `PROBE_TIMEOUT_SECONDS`
  exactly rather than to "some timeout is set";
* `register_probes`, asserted through the names the registry actually carries,
  so "registered only what is configured" (decision C) is measured rather than
  inferred from the settings that went in;
* one end-to-end unhappy path through `TestClient`: a Redis that refuses
  connections, `/readyz` 503 naming every probe, `/healthz` still 200 and a chat
  completion unaffected (a project rule - the status comes from a request, not
  from reading the handler).

No backing service is started. The refused-connection case points at a port
nothing listens on, which fails immediately rather than after the timeout.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import httpx
import pytest
import redis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsconsole.core.settings import Settings
from opsconsole.gateway.app import create_app
from opsconsole.gateway.health import ProbeRegistry
from opsconsole.gateway.probes import (
    POSTGRES_PROBE,
    PROBE_TIMEOUT_SECONDS,
    PROVIDER_PROBE,
    QDRANT_PROBE,
    QDRANT_READY_PATH,
    REDIS_PROBE,
    build_qdrant_client,
    build_redis_client,
    postgres_probe,
    provider_probe,
    qdrant_probe,
    redis_probe,
    register_probes,
)
from opsconsole.providers.base import Adapter

from .conftest import (
    API_KEYS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    StubAdapter,
    chat_body,
    post_chat,
)

# A port nothing listens on: the connection is refused at once, so the
# refused-Redis case costs no timeout and cannot be confused with a hang.
UNREACHABLE_REDIS_URL = "redis://127.0.0.1:1/0"

QDRANT_BASE_URL = "http://qdrant.invalid:6333"
"""Never resolved: every qdrant case goes through `httpx.MockTransport`."""


# --------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------


def test_the_probe_timeout_is_exactly_two_seconds() -> None:
    """Catches a timeout widened to a value that no longer bounds readiness (30s, or
    `None`): one hung backing service would then hang `/readyz` itself and take a
    healthy pod out of rotation, and a band would admit every wrong value."""
    assert PROBE_TIMEOUT_SECONDS == 2.0


def test_the_probe_names_are_the_ones_the_readyz_body_publishes() -> None:
    """Catches a rename of a probe key (`postgres` -> `pg`, `provider` -> `providers`):
    the `/readyz` body is what an operator and a Kubernetes probe read, so the four
    strings are published API rather than internal labels."""
    assert REDIS_PROBE == "redis"
    assert POSTGRES_PROBE == "postgres"
    assert QDRANT_PROBE == "qdrant"
    assert PROVIDER_PROBE == "provider"


def test_the_qdrant_readiness_path_is_the_services_own() -> None:
    """Catches a probe pointed at `/healthz` or `/` instead: Qdrant answers 200 on paths
    that say nothing about whether it has finished loading its collections, so the wrong
    path is a readiness check that passes while the service cannot serve."""
    assert QDRANT_READY_PATH == "/readyz"


# --------------------------------------------------------------------------
# redis
# --------------------------------------------------------------------------


class FakePing:
    """A client with just the `ping` the redis probe is allowed to use."""

    def __init__(self, answer: object = True, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.calls = 0

    def ping(self) -> object:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._answer


REDIS_CASES: list[tuple[str, object, Exception | None, bool]] = [
    ("a reachable redis answers true", True, None, True),
    ("a ping that answers false", False, None, False),
    ("a refused connection", None, redis.ConnectionError("connection refused"), False),
    ("a hung server hitting the read timeout", None, redis.TimeoutError("timed out"), False),
    ("an error the client does not model", None, RuntimeError("something else"), False),
]


@pytest.mark.parametrize(
    ("case", "answer", "error", "expected"), REDIS_CASES, ids=[c[0] for c in REDIS_CASES]
)
def test_the_redis_probe_answers_a_bool_for_every_client(
    case: str, answer: object, error: Exception | None, expected: bool
) -> None:
    """Catches a probe that lets its client's error escape - the factory is called here
    directly, so `/readyz`'s own `except` cannot be what turns the raising rows into
    `False`, and the timeout row is the plan's hung-service risk rather than a repeat of
    the refused one."""
    client = FakePing(answer=answer, error=error)

    probe = redis_probe(client)

    assert probe() is expected
    assert client.calls == 1


def test_the_redis_probe_asks_the_server_every_time_it_is_called() -> None:
    """Catches a probe that answers from a cached first result: readiness is polled every
    few seconds precisely so a Redis that went away since the last call is noticed, and a
    memoised probe reports the pod ready forever."""
    client = FakePing(answer=True)
    probe = redis_probe(client)

    assert probe() is True
    assert probe() is True
    assert client.calls == 2


# --------------------------------------------------------------------------
# postgres
# --------------------------------------------------------------------------


class FakeConnection:
    """A connection that records its statements and whether it was released."""

    def __init__(self, error: Exception | None = None) -> None:
        self.statements: list[str] = []
        self.error = error
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> FakeConnection:
        self.entered += 1
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exited += 1

    def execute(self, statement: str) -> object:
        self.statements.append(statement)
        if self.error is not None:
            raise self.error
        return self


POSTGRES_CASES: list[tuple[str, Exception | None, Exception | None, bool]] = [
    ("a reachable database", None, None, True),
    ("a refused connection", RuntimeError("connection refused"), None, False),
    ("a connect that hits the timeout", TimeoutError("timed out"), None, False),
    ("a server that refuses the statement", None, RuntimeError("terminating connection"), False),
    ("a statement that hits the read timeout", None, TimeoutError("timed out"), False),
]


@pytest.mark.parametrize(
    ("case", "connect_error", "execute_error", "expected"),
    POSTGRES_CASES,
    ids=[c[0] for c in POSTGRES_CASES],
)
def test_the_postgres_probe_answers_a_bool_for_every_connection(
    case: str, connect_error: Exception | None, execute_error: Exception | None, expected: bool
) -> None:
    """Catches a probe that propagates a connection failure instead of answering `False`:
    the factory is called directly, so `/readyz`'s net cannot be what makes the failing
    rows pass, and the two error positions are separate rows because a `try` around only
    the connect leaves a dead server answering ready."""
    connection = FakeConnection(error=execute_error)

    def connect() -> FakeConnection:
        if connect_error is not None:
            raise connect_error
        return connection

    probe = postgres_probe(connect)

    assert probe() is expected


def test_the_postgres_probe_runs_a_statement_rather_than_only_connecting() -> None:
    """Catches a probe that calls `connect()` and answers `True`: a Postgres accepting TCP
    while it replays WAL is exactly the state readiness has to refuse, and only a
    statement that comes back distinguishes it."""
    connection = FakeConnection()

    probe = postgres_probe(lambda: connection)

    assert probe() is True
    assert connection.statements == ["SELECT 1"]


def test_the_postgres_probe_releases_the_connection_even_when_the_statement_fails() -> None:
    """Catches a probe that leaks a connection per failed check: readiness runs every few
    seconds, so a probe that never exits its connection exhausts the server's slots while
    reporting the outage it caused."""
    connection = FakeConnection(error=RuntimeError("terminating connection"))

    probe = postgres_probe(lambda: connection)

    assert probe() is False
    assert connection.exited == 1


# --------------------------------------------------------------------------
# qdrant
# --------------------------------------------------------------------------


def responding_client(status: int, seen: list[httpx.Request]) -> httpx.Client:
    """An `httpx.Client` answering `status` to everything, recording what it was asked."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    return httpx.Client(transport=httpx.MockTransport(handle))


def raising_client(error: Exception, seen: list[httpx.Request]) -> httpx.Client:
    """An `httpx.Client` whose transport raises, for the unreachable and hung rows."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise error

    return httpx.Client(transport=httpx.MockTransport(handle))


# The boundary pair is 399/400: `< 400`, so a probe written `== 200` fails the
# 204 row and one written `< 500` fails the 404 row.
QDRANT_STATUS_CASES: list[tuple[str, int, bool]] = [
    ("200 is ready", 200, True),
    ("204 is ready", 204, True),
    ("399 is the last ready status", 399, True),
    ("400 is the first unready status", 400, False),
    ("404 is not ready", 404, False),
    ("503 is not ready", 503, False),
]


@pytest.mark.parametrize(
    ("case", "status", "expected"),
    QDRANT_STATUS_CASES,
    ids=[c[0] for c in QDRANT_STATUS_CASES],
)
def test_the_qdrant_probe_reads_the_status(case: str, status: int, expected: bool) -> None:
    """Catches a probe that answers `True` for any response that came back at all: a
    Qdrant returning 503 while it loads is reachable and not ready, and the 399/400 pair
    is what pins the comparison rather than a status set someone listed."""
    seen: list[httpx.Request] = []
    with responding_client(status, seen) as client:
        probe = qdrant_probe(client, QDRANT_BASE_URL)

        assert probe() is expected


QDRANT_ERROR_CASES: list[tuple[str, Exception]] = [
    ("an unreachable service", httpx.ConnectError("connection refused")),
    ("a hung service hitting the read timeout", httpx.ReadTimeout("timed out")),
    ("an error httpx does not model", RuntimeError("something else")),
]


@pytest.mark.parametrize(
    ("case", "error"), QDRANT_ERROR_CASES, ids=[c[0] for c in QDRANT_ERROR_CASES]
)
def test_the_qdrant_probe_answers_false_when_the_call_raises(case: str, error: Exception) -> None:
    """Catches a probe that lets a transport error escape: called directly here, so
    `/readyz`'s own `except` cannot be credited, and the timeout row is the hung-service
    risk the plan names rather than a second unreachable row."""
    seen: list[httpx.Request] = []
    with raising_client(error, seen) as client:
        probe = qdrant_probe(client, QDRANT_BASE_URL)

        assert probe() is False


def test_the_qdrant_probe_asks_for_the_readiness_path_on_the_configured_url() -> None:
    """Catches a probe that hits the base URL, or one that ignores the configured host and
    goes to a default: the request is read off the transport, so the URL is measured
    rather than assumed from the arguments that went in."""
    seen: list[httpx.Request] = []
    with responding_client(200, seen) as client:
        probe = qdrant_probe(client, QDRANT_BASE_URL)

        assert probe() is True

    assert [str(request.url) for request in seen] == [QDRANT_BASE_URL + QDRANT_READY_PATH]
    assert seen[0].method == "GET"


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------


class ProbeAdapter(StubAdapter):
    """A stub adapter whose `healthcheck` can answer false or raise."""

    def __init__(self, name: str, *, healthy: bool = True, error: Exception | None = None) -> None:
        super().__init__(name=name)
        self._healthy = healthy
        self._healthcheck_error = error
        self.healthchecks = 0

    def healthcheck(self) -> bool:
        self.healthchecks += 1
        if self._healthcheck_error is not None:
            raise self._healthcheck_error
        return self._healthy


def adapters(*built: ProbeAdapter) -> Mapping[str, Adapter]:
    """The adapter mapping in the shape `app.state.adapters` carries it."""
    return {adapter.name: adapter for adapter in built}


PROVIDER_CASES: list[tuple[str, Mapping[str, Adapter], bool]] = [
    ("no adapter was built at all", adapters(), False),
    ("one healthy adapter", adapters(ProbeAdapter("anthropic")), True),
    ("one unhealthy adapter", adapters(ProbeAdapter("anthropic", healthy=False)), False),
    (
        "one adapter whose healthcheck raises",
        adapters(ProbeAdapter("anthropic", error=RuntimeError("adapter blew up"))),
        False,
    ),
    (
        "a raising adapter before a healthy one",
        adapters(
            ProbeAdapter("anthropic", error=RuntimeError("adapter blew up")),
            ProbeAdapter("local"),
        ),
        True,
    ),
    (
        "an unhealthy adapter before a healthy one",
        adapters(ProbeAdapter("anthropic", healthy=False), ProbeAdapter("local")),
        True,
    ),
    (
        "every adapter unhealthy",
        adapters(
            ProbeAdapter("anthropic", healthy=False),
            ProbeAdapter("local", healthy=False),
        ),
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "built", "expected"), PROVIDER_CASES, ids=[c[0] for c in PROVIDER_CASES]
)
def test_the_provider_probe_needs_one_adapter_that_answers_healthy(
    case: str, built: Mapping[str, Adapter], expected: bool
) -> None:
    """Catches a probe written `all(...)` instead of `any(...)`, and one that reports
    `True` for an empty mapping because `any` of nothing is false but `all` of nothing is
    true: the empty row and the mixed rows disagree under those two readings, and the
    raising adapter is placed FIRST so a probe that propagates its error cannot reach the
    healthy one."""
    probe = provider_probe(built)

    assert probe() is expected


# --------------------------------------------------------------------------
# the built clients
# --------------------------------------------------------------------------


def test_the_redis_client_bounds_both_the_connect_and_the_read() -> None:
    """Catches a client built with the library defaults (no timeout at all) or with only
    the connect bounded: a Redis that accepts the connection and never answers is the
    shape that hangs readiness, and only `socket_timeout` bounds that half."""
    client = build_redis_client("redis://127.0.0.1:6379/0")

    kwargs = client.connection_pool.connection_kwargs

    assert kwargs["socket_connect_timeout"] == PROBE_TIMEOUT_SECONDS
    assert kwargs["socket_timeout"] == PROBE_TIMEOUT_SECONDS


def test_the_redis_client_points_at_the_configured_url() -> None:
    """Catches a builder that ignores its argument and connects to localhost defaults: the
    cluster's Redis is a service name, and a client silently pointed elsewhere would make
    readiness pass against a Redis nobody uses."""
    client = build_redis_client("redis://127.0.0.1:6390/3")

    kwargs = client.connection_pool.connection_kwargs

    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 6390
    assert kwargs["db"] == 3


def test_the_qdrant_client_bounds_every_phase_of_the_call() -> None:
    """Catches an `httpx.Client()` built with the library default (5s connect and no read
    bound at all): the timeout object is read off the client, so "a timeout was passed"
    is measured rather than assumed, and all four phases carry the probe's number."""
    with build_qdrant_client() as client:
        assert client.timeout == httpx.Timeout(PROBE_TIMEOUT_SECONDS)
        assert client.timeout.connect == PROBE_TIMEOUT_SECONDS
        assert client.timeout.read == PROBE_TIMEOUT_SECONDS
        assert client.timeout.write == PROBE_TIMEOUT_SECONDS
        assert client.timeout.pool == PROBE_TIMEOUT_SECONDS


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def probe_names(registry: ProbeRegistry) -> set[str]:
    """The names a registry carries, which is what the `/readyz` body is built from."""
    return {name for name, _ in registry}


def probe_settings(
    *,
    redis_url: str | None = None,
    database_url: str | None = None,
    qdrant_url: str | None = None,
) -> Settings:
    """Settings with all three backing-service URLs stated, absent ones included.

    Stated rather than defaulted on purpose: `Settings` reads `OPSCONSOLE_REDIS_URL`
    and friends from the process environment, and the Redis suite in
    `tests/redaction/test_redis_store.py` asks a developer to export exactly
    those. An "unconfigured" row that inherited one would then assert the
    opposite of what it says.
    """
    return Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        redis_url=redis_url,
        database_url=database_url,
        qdrant_url=qdrant_url,
    )


def bare_app(
    *,
    redis_url: str | None = None,
    database_url: str | None = None,
    qdrant_url: str | None = None,
) -> FastAPI:
    """The three pieces of state `register_probes` reads, and nothing else.

    Built by hand rather than by `create_app`, so this group measures
    `register_probes` alone: a factory that registered the probes itself could
    otherwise make these cases pass with the function under test doing nothing.
    """
    app = FastAPI()
    app.state.settings = probe_settings(
        redis_url=redis_url, database_url=database_url, qdrant_url=qdrant_url
    )
    app.state.adapters = adapters(ProbeAdapter("anthropic"))
    app.state.probes = ProbeRegistry()
    return app


REGISTRATION_CASES: list[tuple[str, dict[str, str], set[str]]] = [
    ("nothing configured", {}, {PROVIDER_PROBE}),
    ("redis alone", {"redis_url": UNREACHABLE_REDIS_URL}, {REDIS_PROBE, PROVIDER_PROBE}),
    (
        "postgres alone",
        {"database_url": "postgresql://opsconsole@127.0.0.1:5432/opsconsole"},
        {POSTGRES_PROBE, PROVIDER_PROBE},
    ),
    ("qdrant alone", {"qdrant_url": QDRANT_BASE_URL}, {QDRANT_PROBE, PROVIDER_PROBE}),
    (
        "every store configured",
        {
            "redis_url": UNREACHABLE_REDIS_URL,
            "database_url": "postgresql://opsconsole@127.0.0.1:5432/opsconsole",
            "qdrant_url": QDRANT_BASE_URL,
        },
        {REDIS_PROBE, POSTGRES_PROBE, QDRANT_PROBE, PROVIDER_PROBE},
    ),
]


@pytest.mark.parametrize(
    ("case", "urls", "expected"), REGISTRATION_CASES, ids=[c[0] for c in REGISTRATION_CASES]
)
def test_register_probes_registers_the_configured_stores_and_always_the_provider(
    case: str, urls: dict[str, str], expected: set[str]
) -> None:
    """Catches a registration that always registers all four (decision C: an unconfigured
    store would then fail readiness forever) and one that registers only the stores and
    forgets the provider - the names are read off the registry, so what was registered is
    measured rather than inferred from the settings that went in."""
    app = bare_app(**urls)

    register_probes(app)

    assert probe_names(app.state.probes) == expected


def test_the_registered_redis_probe_is_wired_to_the_configured_redis() -> None:
    """Catches a registration that files a placeholder answering `True` under the name
    `redis`: the URL points at a port nothing listens on, so the only registration that
    can answer `False` here is one holding a client that really tried to reach it."""
    app = bare_app(redis_url=UNREACHABLE_REDIS_URL)

    register_probes(app)

    checks = {name: probe() for name, probe in app.state.probes}
    assert checks[REDIS_PROBE] is False
    assert checks[PROVIDER_PROBE] is True


def test_create_app_registers_the_probes_its_settings_configure() -> None:
    """Catches a factory that builds `probes.py` and never calls it: `/readyz` would then
    be 200 while checking nothing at all, which is the failure decision C's named probe
    set exists to make visible."""
    app = create_app(probe_settings(redis_url=UNREACHABLE_REDIS_URL), NO_PROVIDER_KEYS)

    assert probe_names(app.state.probes) == {REDIS_PROBE, PROVIDER_PROBE}


# --------------------------------------------------------------------------
# the unhappy path, end to end
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def refused_redis_client() -> Iterator[TestClient]:
    """The shipped app configured against a Redis that refuses connections.

    The adapter set is MUTATED rather than replaced: the provider probe holds the
    mapping `create_app` registered it with, so rebinding `app.state.adapters`
    would leave the probe reading a dict nothing else uses.
    """
    app = create_app(probe_settings(redis_url=UNREACHABLE_REDIS_URL), NO_PROVIDER_KEYS)
    stub: Any = StubAdapter(name="anthropic")
    app.state.adapters[stub.name] = stub
    with TestClient(app) as client:
        yield client


def test_a_refused_redis_makes_readyz_503_naming_every_probe(
    refused_redis_client: TestClient,
) -> None:
    """Catches a readiness that reports 503 without saying which dependency failed, and
    one that reports only the failures: the body is compared as a WHOLE, so a probe that
    ran and was left out of the answer - the green-because-nothing-was-configured case
    the criteria call out - fails here."""
    response = refused_redis_client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "checks": {REDIS_PROBE: False, PROVIDER_PROBE: True},
        # An earlier change: readiness now reports the residency mode beside the probes, and
        # this assertion compares the body WHOLE, so the key belongs here too.
        "residency_mode": "strict",
    }


def test_a_refused_redis_leaves_liveness_and_completions_alone(
    refused_redis_client: TestClient,
) -> None:
    """Catches a probe wired into `/healthz` or into the request path: a dependency outage
    must take the pod out of the load balancer, never restart the container and never fail
    a completion that needs no Redis at all."""
    live = refused_redis_client.get("/healthz")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}

    completion = post_chat(refused_redis_client, chat_body())

    assert completion.status_code == 200, completion.text
