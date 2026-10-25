"""Where the backing-service clients and their stores are built, and when.

An earlier change moves the four backing-service clients - the one `redis.Redis`, the one
qdrant `httpx.Client`, the one Postgres pool and the Prometheus http client - and
every store that wraps them out of `create_app` and into the app's `lifespan`.
`create_app` is left building only the things a misconfiguration should refuse at
once (the key store, the policy, the adapters, the breaker, the prices) plus an
empty probe registry; everything that opens a socket waits for startup and is
closed on shutdown.

These are the tests that make that a property rather than an intention:

* the factory opens no client - the four builders are spied and must be called
  zero times, and the four client attributes must be absent on `app.state`
  before the lifespan is entered;
* entering the lifespan builds all four on `app.state`, and closing it closes
  all four exactly once;
* one `redis.Redis`, one qdrant client and one pool are shared by every store
  that reads them, read off the stores' own held clients so a per-store client
  would be visible;
* a route reads its store through a typed dependency getter, provable by
  overriding the getter and seeing the route call the override rather than
  `app.state`;
* no route reads `app.state` by hand - the nine route source files are scanned
  for the string;
* the readiness probes registered at startup hold the SAME shared client, shown
  by toggling `app.state.redis_client` to fail and watching `/readyz` turn 503.

Every builder is monkeypatched on `app.main`, because the lifespan reaches them
through that module. No backing service is started: the shared-client test uses
the real lazy clients against ports nothing listens on, and nothing dials them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core.config import Settings
from app.main import create_app
from app.services.prometheus_reads import PrometheusClient

from .conftest import (
    ADMIN_KEY,
    API_KEYS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    sha256_hex,
)

# Addresses nothing listens on. Every real client here is lazy: building it opens
# no connection, so these URLs are never dialled by the tests that use them.
UNCONNECTED_REDIS_URL = "redis://127.0.0.1:1/0"
UNCONNECTED_QDRANT_URL = "http://127.0.0.1:1"
UNCONNECTED_POSTGRES_URL = "postgresql://opsconsole@127.0.0.1:1/opsconsole"
UNCONNECTED_PROMETHEUS_URL = "http://127.0.0.1:1"


def all_urls_settings() -> Settings:
    """A `Settings` with all four backing-service URLs pointed at dead ports.

    Every one is stated so the lifespan builds every client and registers every
    probe; the ports are unconnected because no case here needs a live service.
    """
    return Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        redis_url=UNCONNECTED_REDIS_URL,
        qdrant_url=UNCONNECTED_QDRANT_URL,
        database_url=UNCONNECTED_POSTGRES_URL,
        prometheus_url=UNCONNECTED_PROMETHEUS_URL,
    )


# --------------------------------------------------------------------------
# the doubles the builders can be made to return
# --------------------------------------------------------------------------


class FakeRedis:
    """Stands in for `redis.Redis`: records `close`, answers a toggleable `ping`.

    The Redis-backed store constructors only hold the client or call
    `register_script` on it, so nothing else needs modelling for construction.
    """

    def __init__(self) -> None:
        self.ping_answer: object = True
        self.ping_error: Exception | None = None
        self.closed = 0

    def ping(self) -> object:
        if self.ping_error is not None:
            raise self.ping_error
        return self.ping_answer

    def register_script(self, script: str) -> Callable[..., None]:
        """`RedisRateLimitStore.__init__` calls this; it wants a callable back."""
        return lambda *args, **kwargs: None

    def close(self) -> None:
        self.closed += 1


class FakeResponse:
    """What `FakeHTTPClient.get` answers: just the status the qdrant probe reads."""

    status_code = 200


class FakeHTTPClient:
    """Stands in for an `httpx.Client`: records `close`, answers 200 to `get`.

    Used for the qdrant data client and for the http client a `PrometheusClient`
    wraps. The qdrant readiness probe may call `get`, so it answers something with
    a status; nothing at construction needs more.
    """

    def __init__(self) -> None:
        self.closed = 0
        self.is_closed = False

    def get(self, url: str, *args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    def close(self) -> None:
        self.closed += 1
        self.is_closed = True


class FakePoolConnection:
    """A connection a `FakePool` lends out: answers `execute`, enters and exits."""

    def execute(self, statement: str) -> object:
        return self


class FakePool:
    """Stands in for the Postgres pool: records `close`, lends a connection.

    The Postgres writer constructors only hold the pool; the Postgres readiness
    probe borrows a connection and runs a statement, which is why `connection`
    yields one.
    """

    def __init__(self) -> None:
        self.closed = 0

    @contextmanager
    def connection(self) -> Iterator[FakePoolConnection]:
        yield FakePoolConnection()

    def close(self) -> None:
        self.closed += 1


class Builders:
    """The four builder spies, each recording its calls and returning a double."""

    def __init__(self) -> None:
        self.redis_calls: list[str] = []
        self.qdrant_calls: list[str] = []
        self.pool_calls: list[str] = []
        self.prometheus_calls: list[str] = []
        self.redis = FakeRedis()
        self.qdrant = FakeHTTPClient()
        self.pool = FakePool()
        self.prometheus_http = FakeHTTPClient()

    def build_redis_client(self, url: str) -> FakeRedis:
        self.redis_calls.append(url)
        return self.redis

    def build_qdrant_data_client(self, url: str) -> FakeHTTPClient:
        self.qdrant_calls.append(url)
        return self.qdrant

    def build_pool(self, url: str) -> FakePool:
        self.pool_calls.append(url)
        return self.pool

    def build_prometheus_http(self, url: str) -> FakeHTTPClient:
        self.prometheus_calls.append(url)
        return self.prometheus_http


def patch_builders(monkeypatch: pytest.MonkeyPatch, builders: Builders) -> None:
    """Point the four `app.main` builder names at the spy's recording versions."""
    monkeypatch.setattr(main, "build_redis_client", builders.build_redis_client)
    monkeypatch.setattr(main, "build_qdrant_data_client", builders.build_qdrant_data_client)
    monkeypatch.setattr(main, "build_pool", builders.build_pool)
    monkeypatch.setattr(main, "build_prometheus_http", builders.build_prometheus_http)


# --------------------------------------------------------------------------
# the factory opens nothing
# --------------------------------------------------------------------------


def test_building_the_app_opens_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a `create_app` that still builds the backing-service clients itself: the
    four builders are spied and must be called zero times, and the four client slots must
    be absent, so a factory that opens a connection pool at import time - the thing plan
    61 moves into startup - turns one of these red."""
    builders = Builders()
    patch_builders(monkeypatch, builders)

    app = create_app(all_urls_settings(), {})

    assert builders.redis_calls == []
    assert builders.qdrant_calls == []
    assert builders.pool_calls == []
    assert builders.prometheus_calls == []
    assert not hasattr(app.state, "redis_client")
    assert not hasattr(app.state, "qdrant_client")
    assert not hasattr(app.state, "db_pool")
    assert not hasattr(app.state, "prometheus")


# --------------------------------------------------------------------------
# startup builds, shutdown closes
# --------------------------------------------------------------------------


def test_clients_are_built_on_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a lifespan that never builds the clients: a factory that left the slots
    unset would leave every store with nothing to read, and the doubles returned by the
    spied builders are exactly what must land on `app.state`."""
    builders = Builders()
    patch_builders(monkeypatch, builders)

    app = create_app(all_urls_settings(), {})
    with TestClient(app):
        assert app.state.redis_client is builders.redis
        assert app.state.qdrant_client is builders.qdrant
        assert app.state.db_pool is builders.pool
        assert isinstance(app.state.prometheus, PrometheusClient)


def test_clients_are_closed_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a lifespan that builds the clients and never closes them: a dropped pool
    or `httpx` client leaks connections and file descriptors, and nothing else in the
    suite would go red, so each double must record exactly one `close` after shutdown."""
    builders = Builders()
    patch_builders(monkeypatch, builders)

    app = create_app(all_urls_settings(), {})
    with TestClient(app):
        pass

    assert builders.pool.closed == 1
    assert builders.redis.closed == 1
    assert builders.qdrant.closed == 1
    assert builders.prometheus_http.closed == 1


# --------------------------------------------------------------------------
# one client per service, shared by every store that reads it
# --------------------------------------------------------------------------


def test_one_redis_one_qdrant_one_pool_are_shared() -> None:
    """Catches a startup that mints a fresh client per store: two gateway replicas would
    then open twice the connection pools for one dependency, and the readiness probe would
    check a client no store uses - the stores' own held clients are compared by identity,
    so a per-store client is visible here."""
    app = create_app(all_urls_settings(), NO_PROVIDER_KEYS)
    with TestClient(app):
        redis_client = app.state.redis_client
        assert app.state.redaction_map_store._client is redis_client
        assert app.state.response_cache._client is redis_client
        assert app.state.budget_store._client is redis_client
        assert app.state.directory_store._client is redis_client

        assert app.state.semantic_cache._client is app.state.qdrant_client

        pool = app.state.db_pool
        assert app.state.cost_ledger._pool is pool
        assert app.state.console_events._pool is pool
        assert app.state.console_reader._pool is pool


# --------------------------------------------------------------------------
# a route reads its store through a typed dependency
# --------------------------------------------------------------------------


class RecordingDirectoryStore:
    """A `DirectoryStore` that records every `get` and `put`, holding no names."""

    def __init__(self) -> None:
        self.gets: list[str] = []
        self.puts: list[tuple[str, tuple[str, ...]]] = []

    def get(self, tenant: str) -> tuple[str, ...]:
        self.gets.append(tenant)
        return ()

    def put(self, tenant: str, names: Sequence[str]) -> None:
        self.puts.append((tenant, tuple(names)))


def test_a_route_reads_its_client_through_a_dependency() -> None:
    """Catches a route that still reads `request.app.state.directory_store` by hand: the
    dependency `get_directory_store` is overridden with a recording store, so only a route
    that resolves the dependency reaches the override - one still reading `app.state` would
    talk to the real in-process store and leave the override's `put`/`get` untouched."""
    from app.api.deps import get_directory_store

    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        admin_key_sha256=sha256_hex(ADMIN_KEY),
    )
    app = create_app(settings, {})
    store = RecordingDirectoryStore()
    app.dependency_overrides[get_directory_store] = lambda: store

    with TestClient(app) as client:
        response = client.post(
            "/admin/directory",
            json={"tenant": "acme", "names": ["Ada"]},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )

    assert response.status_code == 200, response.text
    assert store.puts == [("acme", ("Ada",))]
    assert store.gets != []


# --------------------------------------------------------------------------
# no route reads app.state by hand
# --------------------------------------------------------------------------


ROUTE_FILES = (
    "chat.py",
    "embeddings.py",
    "redact.py",
    "model_list.py",
    "dashboard.py",
    "admin/routing.py",
    "admin/directory.py",
    "admin/releases.py",
    "admin/incidents.py",
)


def test_routes_do_not_read_app_state_by_hand() -> None:
    """Catches a route reaching into `request.app.state` instead of taking a typed
    dependency: that is the pattern an earlier change removes so startup can own where state is
    built, and the source text of all nine route files is scanned so a single leftover
    `app.state` fails here."""
    routes_dir = Path(main.__file__).resolve().parent / "api" / "routes"
    offenders = [
        name
        for name in ROUTE_FILES
        if "app.state" in (routes_dir / name).read_text(encoding="utf-8")
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# the probes run over the shared clients
# --------------------------------------------------------------------------


def test_readyz_probes_run_over_the_shared_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a readiness probe registered over a client other than the one on
    `app.state`: the redis builder returns a fake whose ping is then toggled to raise, and
    `/readyz` must turn 503 - a probe holding a different client would stay 200 and prove
    the registration was not wired to the shared client startup built."""
    fake = FakeRedis()
    monkeypatch.setattr(main, "build_redis_client", lambda url: fake)

    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        redis_url=UNCONNECTED_REDIS_URL,
    )
    app = create_app(settings, NO_PROVIDER_KEYS)
    with TestClient(app) as client:
        assert app.state.redis_client is fake

        healthy = client.get("/readyz")
        assert healthy.status_code == 200, healthy.text

        fake.ping_error = RuntimeError("redis went away")
        failed = client.get("/readyz")

        assert failed.status_code == 503
        assert failed.json()["checks"]["redis"] is False


# --------------------------------------------------------------------------
# migrating at start
# --------------------------------------------------------------------------


class MigrateSpy:
    """Stands in for `main.migrate_at_start`: records the pool it got and when."""

    def __init__(self, builders: Builders, error: Exception | None = None) -> None:
        self.builders = builders
        self.error = error
        self.pools: list[object] = []
        self.pools_built_before: list[int] = []

    def __call__(self, pool: object, schema: str = "opsconsole") -> None:
        self.pools.append(pool)
        self.pools_built_before.append(len(self.builders.pool_calls))
        if self.error is not None:
            raise self.error


def switch_on_settings() -> Settings:
    """`all_urls_settings` with the start-up migrate switch on."""
    return all_urls_settings().model_copy(update={"migrate_on_start": True})


def test_the_switch_on_migrates_once_at_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a lifespan that never migrates, migrates more than once, or migrates before
    the pool exists: the seam must be called once, with the built pool, after `build_pool`."""
    builders = Builders()
    patch_builders(monkeypatch, builders)
    spy = MigrateSpy(builders)
    monkeypatch.setattr(main, "migrate_at_start", spy)

    app = create_app(switch_on_settings(), {})
    with TestClient(app):
        assert spy.pools == [builders.pool]
        assert spy.pools_built_before == [1]

    assert len(spy.pools) == 1


def test_the_switch_off_never_migrates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a lifespan that migrates whatever the switch says: with default settings
    (switch off) the seam must never be called, even with a pool present."""
    builders = Builders()
    patch_builders(monkeypatch, builders)
    spy = MigrateSpy(builders)
    monkeypatch.setattr(main, "migrate_at_start", spy)

    app = create_app(all_urls_settings(), {})
    with TestClient(app):
        pass

    assert builders.pool_calls != []
    assert spy.pools == []


def test_the_switch_on_without_a_database_never_migrates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a lifespan that calls the seam with no pool (switch on, no database URL),
    which would crash a gateway that is meant to run without Postgres."""
    builders = Builders()
    patch_builders(monkeypatch, builders)
    spy = MigrateSpy(builders)
    monkeypatch.setattr(main, "migrate_at_start", spy)
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        migrate_on_start=True,
    )

    with TestClient(create_app(settings, {})):
        pass

    assert spy.pools == []


def test_a_failed_migration_stops_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a lifespan that swallows a migration error and serves on a half-migrated
    schema: the lifespan must raise, and the pool it opened must still be closed."""
    builders = Builders()
    patch_builders(monkeypatch, builders)
    monkeypatch.setattr(main, "migrate_at_start", MigrateSpy(builders, RuntimeError("bad sql")))

    app = create_app(switch_on_settings(), {})
    with pytest.raises(RuntimeError, match="bad sql"), TestClient(app):
        pytest.fail("the app served after a failed migration")

    assert builders.pool.closed == 1
