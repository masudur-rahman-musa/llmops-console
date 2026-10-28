"""ASGI entrypoint and the gateway app factory.

`uvicorn app.main:app` for a supervisor that owns the port; `serve()` when the
port should come from settings. `create_app` builds all the per-app state; the
router include wiring lives in `app.api.router`.

`close_adapters` is imported here and used by `lifespan`; it is defined in
`providers.factory`, beside the `build_adapters` it undoes. Importing it here
re-exports it - in the `import X as X` form `mypy --strict` requires for an
explicit re-export - so `from app.main import close_adapters`, which the wiring
tests use, keeps working while `services.routing_reload` reaches the same
function directly.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from opentelemetry.context import Context
from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.deps import SettingsKeyStore, parsed_traceparent
from app.api.router import include_routers
from app.api.routes.health import ProbeRegistry, register_probes
from app.cache import (
    InProcessResponseCache,
    InProcessSemanticCache,
    QdrantSemanticCache,
    RedisResponseCache,
)
from app.core.clients import PROBE_TIMEOUT_SECONDS, build_redis_client
from app.core.concurrency import build_provider_limiter
from app.core.config import Settings
from app.core.http_errors import register_error_handlers
from app.core.metrics import observe_request, record_closed_circuits, start_request
from app.core.tracing import (
    ERROR_STATUS_FLOOR,
    TRACE_SCOPE_KEY,
    UNTRACED_PATHS,
    end_root_span,
    flush_tracing,
    install_exporter,
    parent_context,
    root_span_name,
    start_root_span,
)
from app.limits.budget import InProcessBudgetStore, RedisBudgetStore
from app.limits.prices import PriceStore
from app.limits.ratelimit import (
    InProcessRateLimitStore,
    RedisRateLimitStore,
)
from app.model_routing.policy import PolicyStore, check_policy_models_priced
from app.providers.breaker import CircuitBreaker
from app.providers.factory import build_adapters
from app.providers.factory import close_adapters as close_adapters
from app.redaction import (
    InProcessDirectoryStore,
    InProcessRedactionMapStore,
    RedisDirectoryStore,
    RedisRedactionMapStore,
)
from app.services.prometheus_reads import PrometheusClient, build_prometheus_http
from app.services.routing_reload import RUNTIME_SCOPE_KEY, RoutingRuntimeStore
from app.storage import (
    InMemoryConsoleEventWriter,
    InMemoryCostLedgerWriter,
    NoDatabaseMonthSpendReader,
    PostgresConsoleEventWriter,
    PostgresCostLedgerWriter,
    PostgresMonthSpendReader,
    build_pool,
)
from app.storage.dashboard_reads import (
    PostgresConsoleReader,
    UnconfiguredConsoleReader,
)
from app.storage.db import DEFAULT_SCHEMA, apply_migrations_locked


class RoutingLeaseMiddleware:
    """Take one `RoutingRuntime` per HTTP request; release it after the last byte.

    Pure ASGI, for the same reason `TracingMiddleware` is: the `finally` fires
    after the full response body, including every frame of a streamed reply, so
    the lease covers the whole request and the adapter set is never closed while
    a request still holds it.
    """

    def __init__(self, app: ASGIApp, store: RoutingRuntimeStore) -> None:
        self.app = app
        self._store = store

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Lease one runtime for the request; release it unconditionally after."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        runtime = self._store.acquire()
        scope[RUNTIME_SCOPE_KEY] = runtime
        try:
            await self.app(scope, receive, send)
        finally:
            self._store.release(runtime)


def build_qdrant_data_client(base_url: str) -> httpx.Client:
    """The `httpx.Client` the semantic cache sends its searches and writes over.

    Carries the base URL, because `QdrantSemanticCache` sends relative paths and
    holds no endpoint of its own; that is why this is not
    `core.clients.build_qdrant_client`, which is handed a whole URL per call.

    Bounded on every phase by `PROBE_TIMEOUT_SECONDS`, the same two seconds the
    readiness probe uses, and reused rather than given a number of its own: this
    client sits in front of a chat request, so an unbounded read here is the
    whole request hanging on a cache that has nothing to say. Two seconds is a
    generous bound for a vector search and a miss is free.
    """
    return httpx.Client(base_url=base_url, timeout=httpx.Timeout(PROBE_TIMEOUT_SECONDS))


def migrate_at_start(
    pool: ConnectionPool[Connection[TupleRow]], schema: str = DEFAULT_SCHEMA
) -> None:
    """Apply the pending ledger migrations before the gateway serves.

    The seam the lifespan calls, so a test can stand in for Postgres. Prints what
    it applied, the same words `ops-migrate` uses, so a start-up log shows it.
    """
    with pool.connection() as conn:
        applied = apply_migrations_locked(conn, schema=schema)
    if applied:
        print(f"applied {len(applied)} migration(s): {', '.join(applied)}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the backing-service clients and the stores over them on startup, close on shutdown.

    Everything here runs when the process starts serving, not when `create_app`
    builds the app, so importing `app.main` and building the app opens no socket.
    The key store, the policy, the prices, the breaker and the adapters are built
    in `create_app` instead: they open no connection, and a malformed config file
    should stop the process before it serves rather than on a live request.

    One Redis client, one Qdrant client and one Postgres pool are built here and
    shared by the stores, caches, writers and the readiness probes, so a
    deployment opens one connection set per service and the probe that checks a
    service runs over the very client the routes use. Each is settings-driven on
    its own switch: `OPSCONSOLE_REDIS_URL` gives the five Redis-backed
    stores a shared client and an in-process twin otherwise `OPSCONSOLE_QDRANT_URL` the semantic
    cache,
    `OPSCONSOLE_DATABASE_URL` the one pool both Postgres writers, the console event
    writer and the console reader share and
    `OPSCONSOLE_PROMETHEUS_URL` the console's Prometheus client. An
    absent switch keeps the twin, which is what lets the suite and a bare local run
    need no backing service at all. Only a local run reaches here without the Redis
    and Qdrant URLs: `create_app` refuses to build a non-local process when either
    is missing, so the Redis and Qdrant twins below serve local runs only.

    The three Redis readers that never raise (the response cache, the budget store
    and the rate-limit store) and the semantic cache keep that contract: a Redis or
    Qdrant outage is a miss, not a failed request. With `migrate_on_start` on,
    the pending migrations are applied right after the pool is built, through
    `migrate_at_start` under an advisory lock so replicas coming up together do
    not race the same DDL. A failed migration stops the start.

    `register_probes` runs last, once the shared clients are on `app.state`, and
    the four clients all close on shutdown.
    """
    resolved: Settings = app.state.settings

    redis_client = (
        build_redis_client(resolved.redis_url) if resolved.redis_url is not None else None
    )
    app.state.redis_client = redis_client
    app.state.redaction_map_store = (
        RedisRedactionMapStore(redis_client)
        if redis_client is not None
        else InProcessRedactionMapStore()
    )
    app.state.directory_store = (
        RedisDirectoryStore(redis_client) if redis_client is not None else InProcessDirectoryStore()
    )
    app.state.response_cache = (
        RedisResponseCache(redis_client) if redis_client is not None else InProcessResponseCache()
    )
    app.state.budget_store = (
        RedisBudgetStore(redis_client) if redis_client is not None else InProcessBudgetStore()
    )
    app.state.rate_limit_store = (
        RedisRateLimitStore(redis_client) if redis_client is not None else InProcessRateLimitStore()
    )

    qdrant_client = (
        build_qdrant_data_client(resolved.qdrant_url) if resolved.qdrant_url is not None else None
    )
    app.state.qdrant_client = qdrant_client
    app.state.semantic_cache = (
        QdrantSemanticCache(qdrant_client)
        if qdrant_client is not None
        else InProcessSemanticCache()
    )

    pool = build_pool(resolved.database_url) if resolved.database_url is not None else None
    if pool is not None and resolved.migrate_on_start:
        try:
            migrate_at_start(pool)
        except BaseException:
            # A gateway serving on a half-migrated schema is the failure this
            # switch exists to stop, so the start stops here instead.
            pool.close()
            raise
    app.state.db_pool = pool
    app.state.cost_ledger = (
        PostgresCostLedgerWriter(pool) if pool is not None else InMemoryCostLedgerWriter()
    )
    app.state.console_events = (
        PostgresConsoleEventWriter(pool) if pool is not None else InMemoryConsoleEventWriter()
    )
    app.state.console_reader = (
        PostgresConsoleReader(pool) if pool is not None else UnconfiguredConsoleReader()
    )
    app.state.month_spend = (
        PostgresMonthSpendReader(pool) if pool is not None else NoDatabaseMonthSpendReader()
    )

    prometheus = (
        PrometheusClient(build_prometheus_http(resolved.prometheus_url))
        if resolved.prometheus_url is not None
        else None
    )
    app.state.prometheus = prometheus

    register_probes(app)

    try:
        yield
    finally:
        close_adapters(app.state.adapters)
        if prometheus is not None:
            prometheus.close()
        if pool is not None:
            app.state.cost_ledger.close()
            pool.close()
        if redis_client is not None:
            redis_client.close()
        if qdrant_client is not None:
            qdrant_client.close()
        flush_tracing()


def incoming_parent(scope: Scope) -> Context | None:
    """The remote parent this request's `traceparent` names, when it names a usable one.

    Parsed with `app.api.deps.parsed_traceparent` rather than with OpenTelemetry's
    own propagator, because the propagator accepts headers this gateway has
    already decided to reject. One parser means the trace a request JOINS and the
    header contract `X-Gateway-Trace-Id` answers under can never disagree about
    what a valid upstream trace is; a malformed header starts a new trace.
    """
    for name, value in scope.get("headers", ()):
        if name.lower() == b"traceparent":
            parsed = parsed_traceparent(value.decode("latin-1"))
            return parent_context(*parsed) if parsed is not None else None
    return None


class TracingMiddleware:
    """The boundary of every request: opened before routing, closed after the last byte.

    It owns two records, and it owns them for the same reason. The root span, and
    the `RequestMetrics` labels a request is counted with: both describe a
    whole request, and both can only be closed at a point no route can see. The
    name is the tracing one it was born with, because the seam is the same seam.

    Pure ASGI rather than `BaseHTTPMiddleware`, for two reasons that both matter. It runs OUTSIDE
    `ExceptionMiddleware`, so a request
    that fails before any route - an unauthenticated one, which never builds a
    `RequestContext` at all - still has a span to record the failure on and is
    still counted. And it ends the
    span, and takes the latency observation, only once the response body is fully
    sent - which for a streamed completion is after the last frame rather than
    when the route returned.

    The outcome is read off the wire rather than guessed: `send` is wrapped so the
    response's own status decides whether the root span failed and what the
    counter's `status` label says, and the typed code comes from whatever the
    handlers in `core/http_errors.py` recorded on the scope. A crash that never
    produces a response start is a failure too, which is why the initial status is
    500 rather than 200.

    Not every request gets a metrics record: `start_request` opens one for the
    counted paths only. A probe or a scrape (`UNTRACED_PATHS`) is passed straight
    through, neither traced nor counted.

    The clock is `perf_counter`, which is monotonic: a wall clock stepped by NTP
    mid-request would otherwise book a negative or a wildly long duration into a
    latency histogram that has no way to take it back. That one is a choice
    recorded rather than pinned - no test can tell the two clocks apart without
    stepping the system clock underneath a running suite.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Trace and count one HTTP request; pass lifespan and websocket scopes through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path") in UNTRACED_PATHS:
            await self.app(scope, receive, send)
            return

        record = start_root_span(incoming_parent(scope))
        scope[TRACE_SCOPE_KEY] = record
        start_request(scope)
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException:
            end_root_span(record, failed=True, name=root_span_name(scope))
            observe_request(scope, status=status, seconds=time.perf_counter() - started)
            raise
        end_root_span(record, failed=status >= ERROR_STATUS_FLOOR, name=root_span_name(scope))
        observe_request(scope, status=status, seconds=time.perf_counter() - started)


def create_app(
    settings: Settings | None = None, environ: Mapping[str, str] | None = None
) -> FastAPI:
    """Build the gateway app.

    Each call gets its own `ProbeRegistry` on `app.state`, so probes registered
    against one app never leak into another.

    The key store and the policy snapshot are built here too, so a malformed
    `OPSCONSOLE_API_KEYS` entry or an invalid `routing.yaml` stops the process
    from starting instead of turning into a puzzling failure on a live request. A route reads one
    snapshot
    per request from `app.state.policy`.

    The adapter set is built once, here, and closed on shutdown by `lifespan`:
    building one per request would mint an `httpx.Client`
    and a connection pool per call. It is keyed by the policy provider name, so
    the chat route can use "an adapter exists for this provider" as the
    resolver's health seam.

    The circuit breaker is built here for the same reason and is the other half of
    that seam: one per app, so state accumulated against one app cannot decide
    another app's routing, and per process rather than shared through Redis.
    `POST /admin/routing/reload` rebuilds it alongside the adapter set
    it rebuilds, because the state it holds is keyed by provider names out of the
    policy that has just been replaced - a reload that renamed or repointed a
    provider would otherwise leave a breaker open against a name that no longer
    means what it meant.

    `environ` is where the provider keys are read from, injectable for the same
    reason `build_adapters` takes it: a test passes `{}` and no adapter is
    built from a key that happens to be exported in the developer's shell, so a
    suite that would make a real billed call cannot pass locally and fail in CI.

    The clients that open a connection - the Redis client, the Qdrant client, the
    Postgres pool and the Prometheus client - and the stores, caches, writers and
    readers built over them are NOT built here: `lifespan` builds them on startup
    and closes them on shutdown, so building the app opens no socket. `register_probes` runs there
    too,
    over those shared clients. A route reaches each through a typed dependency in
    `app.api.deps` rather than off `app.state`.

    The price table is built here for the reason the key store and the policy are:
    a bad provider file stops the process from starting rather than
    surfacing on a live request that then cannot be priced. It opens no socket,
    which is why it stays here while the connection-backed stores move.

    """
    resolved = settings or Settings()
    if resolved.env != "local":
        # Deny by default: only a local run keeps the in-process stand-ins, which
        # give each copy its own budgets and rate limits. Any other value, the
        # empty string included, must reach shared Redis and Qdrant or the process
        # stops here, at build time, naming the missing URL rather than serving one
        # request with a limit that silently means nothing.
        missing = [
            name
            for name, value in (
                ("OPSCONSOLE_REDIS_URL", resolved.redis_url),
                ("OPSCONSOLE_QDRANT_URL", resolved.qdrant_url),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"a non-local process (env={resolved.env!r}) needs {' and '.join(missing)}"
            )
    app = FastAPI(title="AI Operations Console gateway", lifespan=lifespan)
    app.state.settings = resolved
    app.state.probes = ProbeRegistry()
    app.state.key_store = SettingsKeyStore(resolved.api_keys)
    app.state.policy = PolicyStore(resolved.routing_policy_path, resolved.providers_dir)
    app.state.provider_environ = os.environ if environ is None else environ
    app.state.adapters = build_adapters(app.state.policy.policy, app.state.provider_environ)
    app.state.breaker = CircuitBreaker()
    record_closed_circuits(app.state.policy.policy.providers)
    app.state.prices = PriceStore(resolved.providers_dir)
    check_policy_models_priced(app.state.policy.policy, app.state.prices.table)
    app.state.provider_limiter = build_provider_limiter()
    routing_store = RoutingRuntimeStore(app.state)
    app.state.routing_store = routing_store
    register_error_handlers(app)
    install_exporter(resolved)
    app.add_middleware(RoutingLeaseMiddleware, store=routing_store)
    app.add_middleware(TracingMiddleware)
    include_routers(app)
    return app


app: FastAPI = create_app()


def serve(reload: bool = False) -> None:
    """Run the gateway on the configured port.

    The listening port comes from `Settings`, so `OPSCONSOLE_PORT` reaches the
    socket. Clients that dial
    the gateway (`make eval`, the compose healthcheck) carry their own address
    and are not covered by that test.
    """
    uvicorn.run("app.main:app", port=Settings().port, reload=reload)
