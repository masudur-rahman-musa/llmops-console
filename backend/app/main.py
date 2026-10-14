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

import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from opentelemetry.context import Context
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.deps import SettingsKeyStore, parsed_traceparent
from app.api.router import include_routers
from app.api.routes.health import ProbeRegistry, register_probes
from app.cache import (
    InProcessResponseCache,
    InProcessSemanticCache,
    QdrantSemanticCache,
    RedisResponseCache,
    ResponseCache,
    SemanticCache,
)
from app.core.clients import PROBE_TIMEOUT_SECONDS, build_redis_client
from app.core.config import Settings
from app.core.http_errors import register_error_handlers
from app.core.metrics import observe_request, start_request
from app.core.tracing import (
    ERROR_STATUS_FLOOR,
    TRACE_SCOPE_KEY,
    end_root_span,
    flush_tracing,
    install_exporter,
    parent_context,
    start_root_span,
)
from app.gateway.console_reads import (
    ConsoleReader,
    PostgresConsoleReader,
    UnconfiguredConsoleReader,
)
from app.ledger import (
    ConsoleEventWriter,
    CostLedgerWriter,
    InMemoryConsoleEventWriter,
    InMemoryCostLedgerWriter,
    InMemoryResidencyLogWriter,
    PostgresConsoleEventWriter,
    PostgresCostLedgerWriter,
    PostgresResidencyLogWriter,
    ResidencyLogWriter,
    build_pool,
)
from app.limits.budget import BudgetStore, InProcessBudgetStore, RedisBudgetStore
from app.limits.prices import PriceStore
from app.limits.ratelimit import (
    InProcessRateLimitStore,
    RateLimitStore,
    RedisRateLimitStore,
)
from app.model_routing.policy import PolicyStore, ResidencyMode
from app.model_routing.resolver import RELAXED_REGIONS
from app.providers.breaker import CircuitBreaker
from app.providers.factory import build_adapters
from app.providers.factory import close_adapters as close_adapters
from app.redaction import (
    DirectoryStore,
    InProcessDirectoryStore,
    InProcessRedactionMapStore,
    RedactionMapStore,
    RedisDirectoryStore,
    RedisRedactionMapStore,
)
from app.services.prometheus_reads import PrometheusClient, build_prometheus_http


def build_qdrant_data_client(base_url: str) -> httpx.Client:
    """The `httpx.Client` the semantic cache sends its searches and writes over.

    Carries the base URL, because `QdrantSemanticCache` sends relative paths and
    holds no endpoint of its own; that is why this is not
    `core.clients.build_qdrant_client`, which is handed a whole URL per call.

    Bounded on every phase by `PROBE_TIMEOUT_SECONDS`, the same two seconds the
    readiness probe uses, and reused rather than given a number of its own: this
    client sits in front of a chat request, so an unbounded read here is the
    whole request hanging on a cache that has nothing to say. Two seconds is a
    generous bound for a vector search and a miss is free
    (`test_the_semantic_cache_client_bounds_every_phase_of_the_call`).
    """
    return httpx.Client(base_url=base_url, timeout=httpx.Timeout(PROBE_TIMEOUT_SECONDS))


logger = logging.getLogger(__name__)

RELAXED_RESIDENCY_WARNING = (
    "residency_mode is relaxed: this process allows %s for every data class "
    "except pii. Never run a tenant's real traffic through it."
)
"""The one line that stops a relaxed process passing for a strict one.

Logged exactly once, by `create_app`, and never per request: the mode is read at
startup and cannot change, so a per-resolution warning would be five records per
request and would bury itself (`test_create_app_warns_once_under_relaxed_residency`).
`/readyz` carries the same fact for anything that reads machines rather than logs
(`test_readyz_reports_a_relaxed_residency_mode`).
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the adapter set, and the console's Prometheus client, on shutdown.

    Nothing is built here: the key store, the policy and the adapters are built
    in `create_app`, so a misconfiguration stops the process from starting rather
    than surfacing on the first request. This is only the other end of that.
    """
    try:
        yield
    finally:
        close_adapters(app.state.adapters)
        prometheus: PrometheusClient | None = app.state.prometheus
        if prometheus is not None:
            prometheus.close()
        flush_tracing()


def incoming_parent(scope: Scope) -> Context | None:
    """The remote parent this request's `traceparent` names, when it names a usable one.

    Parsed with `app.api.deps.parsed_traceparent` rather than with OpenTelemetry's
    own propagator, because the propagator accepts headers this gateway has
    already decided to reject. One parser means the trace a request JOINS and the
    header contract `X-Gateway-Trace-Id` answers under can never disagree about
    what a valid upstream trace is; the rejection table is
    `test_a_malformed_traceparent_starts_a_new_trace`.
    """
    for name, value in scope.get("headers", ()):
        if name.lower() == b"traceparent":
            parsed = parsed_traceparent(value.decode("latin-1"))
            return parent_context(*parsed) if parsed is not None else None
    return None


class TracingMiddleware:
    """The boundary of every request: opened before routing, closed after the last byte.

    It owns two records, and it owns them for the same reason. The root span, and
    the `RequestMetrics` labels an earlier change counts a request with: both describe a
    whole request, and both can only be closed at a point no route can see. The
    name is the tracing one it was born with, because the seam is the same seam.

    Pure ASGI rather than `BaseHTTPMiddleware`, for two reasons that both decide
    an acceptance criterion. It runs OUTSIDE `ExceptionMiddleware`, so a request
    that fails before any route - an unauthenticated one, which never builds a
    `RequestContext` at all - still has a span to record the failure on and is
    still counted (`test_an_unauthorized_request_still_emits_a_trace`,
    `test_an_unauthorized_request_is_counted_with_empty_labels`). And it ends the
    span, and takes the latency observation, only once the response body is fully
    sent - which for a streamed completion is after the last frame rather than
    when the route returned (`test_a_streamed_requests_root_span_covers_the_frames`,
    `test_a_streamed_requests_latency_covers_the_frames`).

    The outcome is read off the wire rather than guessed: `send` is wrapped so the
    response's own status decides whether the root span failed and what the
    counter's `status` label says, and the typed code comes from whatever the
    handlers in `core/http_errors.py` recorded on the scope. A crash that never
    produces a response start is a failure too, which is why the initial status is
    500 rather than 200.

    Not every request gets a metrics record: `start_request` opens one for the
    counted paths only, so a probe and a scrape pass through traced and uncounted
    (`test_the_probe_and_scrape_paths_are_not_counted`).

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
            end_root_span(record, failed=True)
            observe_request(scope, status=status, seconds=time.perf_counter() - started)
            raise
        end_root_span(record, failed=status >= ERROR_STATUS_FLOOR)
        observe_request(scope, status=status, seconds=time.perf_counter() - started)


def create_app(
    settings: Settings | None = None, environ: Mapping[str, str] | None = None
) -> FastAPI:
    """Build the gateway app.

    Each call gets its own `ProbeRegistry` on `app.state`, so probes registered
    against one app never leak into another. Pinned by
    `test_registries_are_per_app`.

    The key store and the policy snapshot are built here too, so a malformed
    `OPSCONSOLE_API_KEYS` entry or an invalid `routing.yaml` stops the process
    from starting instead of turning into a puzzling failure on a live request;
    pinned by `test_create_app_refuses_a_malformed_key_setting` and
    `test_create_app_refuses_an_invalid_policy_file`. A route reads one snapshot
    per request from `app.state.policy`.

    The adapter set is built once, here, and closed on shutdown by `lifespan`: building one per request would mint an `httpx.Client`
    and a connection pool per call. It is keyed by the policy provider name, so
    the chat route can use "an adapter exists for this provider" as the
    resolver's health seam. Pinned by `test_create_app_builds_the_adapter_set`
    and `test_adapters_are_closed_on_shutdown`.

    The circuit breaker is built here for the same reason and is the other half of
    that seam: one per app, so state accumulated against one app cannot decide
    another app's routing, and per process rather than shared through Redis. `POST /admin/routing/reload` rebuilds it alongside the adapter set
    it rebuilds, because the state it holds is keyed by provider names out of the
    policy that has just been replaced - a reload that renamed or repointed a
    provider would otherwise leave a breaker open against a name that no longer
    means what it meant (`test_a_policy_reload_rebuilds_the_breaker`).

    `environ` is where the provider keys are read from, injectable for the same
    reason `build_adapters` takes it (#66): a test passes `{}` and no adapter is
    built from a key that happens to be exported in the developer's shell, so a
    suite that would make a real billed call cannot pass locally and fail in CI.
    Pinned by `test_create_app_reads_keys_only_from_the_given_environ`.

    The redaction map store is settings-driven:
    `OPSCONSOLE_REDIS_URL` set gives every replica the same maps through
    `RedisRedactionMapStore`, and absent keeps `InProcessRedactionMapStore`, which
    files them in this process only and is what lets the
    suite and a bare local run need no backing service. Pinned by
    `test_create_app_keeps_the_in_process_store_without_a_redis_url` and
    `test_create_app_uses_the_redis_store_when_a_redis_url_is_configured`.

    The tenant directory store is built from the SAME client and on the same
    switch: one `redis.Redis` per app, so a deployment with Redis
    configured opens one connection pool rather than two for what is one
    dependency, and the readiness probe that already checks Redis covers both
    stores. Redis decodes nothing for either - the map store stores JSON and the
    directory store decodes its own hash values.

    The exact-match response cache is built from that SAME client and on the same
    switch: `RedisResponseCache` when Redis is configured,
    `InProcessResponseCache` otherwise, so a bare local run and the suite need no
    backing service and still take the cache path
    (`test_create_app_keeps_the_in_process_cache_without_a_redis_url`,
    `test_create_app_uses_the_redis_cache_when_a_redis_url_is_configured`). It is
    the third reader of one connection pool, and unlike the two stores above it
    never raises: a Redis outage is a miss on this seam, not a failed request.

    The monthly budget counter is built from that same client and on the same
    switch: `RedisBudgetStore` when Redis is configured, so every
    replica adds to one total per tenant, and `InProcessBudgetStore` otherwise,
    which counts in this process only and is what lets the suite enforce a cap
    with no backing service
    (`test_create_app_keeps_the_in_process_budget_store_without_a_redis_url`,
    `test_create_app_uses_the_redis_budget_store_when_a_redis_url_is_configured`).
    Like the response cache and unlike the two stores above it, it never raises:
    a Redis outage leaves the counter unknown, and an unknown counter serves the
    request.

    The two rate-limit buckets are built from that same client and on the same
    switch: `RedisRateLimitStore` when Redis is configured, so two
    replicas cannot each grant a tenant the full `rpm`, and
    `InProcessRateLimitStore` otherwise, which is what lets the suite and a bare
    local run enforce a ceiling with no backing service
    (`test_create_app_keeps_the_in_process_rate_limit_store_without_a_redis_url`,
    `test_create_app_uses_the_redis_rate_limit_store_when_a_redis_url_is_configured`).
    It never raises either: a limiter that cannot answer serves the request.

    The semantic cache is settings-driven on its OWN switch, `OPSCONSOLE_QDRANT_URL`: configured builds `QdrantSemanticCache` over a bounded `httpx`
    client, absent builds `InProcessSemanticCache`, so the suite and a bare local
    run take the semantic path with no Qdrant anywhere
    (`test_create_app_keeps_the_in_process_semantic_cache_without_a_qdrant_url`,
    `test_create_app_uses_the_qdrant_semantic_cache_when_a_qdrant_url_is_configured`).
    A different switch from the three Redis readers above because it is a
    different service; like the response cache and unlike the two stores, it never
    raises, so an unreachable Qdrant is a miss and not a failed request.

    The price table is built here for the reason the key store and the policy are:
    a malformed `prices.yaml` stops the process from starting rather than
    surfacing on a live request that then cannot be priced
    (`test_create_app_refuses_a_malformed_prices_file`,
    `test_the_price_store_is_built_over_the_configured_path`). A route reads one
    snapshot per request from `app.state.prices`, exactly as it reads the policy.

    The residency log writer is settings-driven for the same reason and by the
    same rule: `OPSCONSOLE_DATABASE_URL` set builds `PostgresResidencyLogWriter`
    over a pool, absent builds `InMemoryResidencyLogWriter`, which files rows in
    this process and is what lets the suite and a bare local run open no
    connection at all
    (`test_the_app_builds_the_in_process_writer_without_a_database_url`). The
    migration is NOT applied here: `make migrate` is a deploy step, so that
    replicas coming up together do not race each other over the same DDL.

    The cost ledger writer takes that same switch and that same pool (an earlier change
    decision D): ONE `app.state.db_pool` serves both Postgres writers, so a
    deployment opens one set of connections for what is one dependency and the
    `postgres` readiness probe that already exists covers both
    (`test_both_postgres_writers_share_one_pool`,
    `test_the_app_builds_the_in_memory_cost_ledger_without_a_database_url`). The
    argument is the one that gives the three Redis readers one client above.

    The console event writer is the third on that same pool and the same switch. It is what `/admin/incidents` and `/admin/releases`
    write through, so the in-memory branch is the one that keeps those routes
    answering on a laptop with no Postgres at all - at the cost that an incident
    recorded there is lost at exit, which is what the writer's own docstring says
    it is (`test_the_routes_answer_with_no_database_url_configured`).

    The console reader takes that same pool and the same switch: a
    `PostgresConsoleReader` over it, or `UnconfiguredConsoleReader`, which the
    `/console/*` routes read as `postgres: unconfigured` and never call. The
    Prometheus client is built only from `OPSCONSOLE_PROMETHEUS_URL`; absent leaves
    `app.state.prometheus` as `None` and every metric field unconfigured
    (`test_with_nothing_configured_every_route_is_200_and_both_sources_unconfigured`).
    Neither registers a readiness probe.

    `register_probes` runs last of the state-building steps, because the probes it
    registers close over the adapter mapping and the settings built above; what it
    registers is decided by which backing services are configured
    (`test_create_app_registers_the_probes_its_settings_configure`).

    `OPSCONSOLE_RESIDENCY_MODE=relaxed` logs `RELAXED_RESIDENCY_WARNING` here, once
    and before any state is built. An unknown value never gets
    this far: `Settings()` above refuses it, so the process exits instead of
    serving one request under a mode nobody chose
    (`test_create_app_warns_once_under_relaxed_residency`,
    `test_create_app_logs_no_residency_warning_under_strict`,
    `test_create_app_refuses_an_unknown_residency_mode`).
    """
    resolved = settings or Settings()
    if resolved.residency_mode is ResidencyMode.RELAXED:
        # Before anything is built, so the line precedes any startup noise.
        logger.warning(RELAXED_RESIDENCY_WARNING, ", ".join(RELAXED_REGIONS))
    app = FastAPI(title="AI Operations Console gateway", lifespan=lifespan)
    app.state.settings = resolved
    app.state.probes = ProbeRegistry()
    app.state.key_store = SettingsKeyStore(resolved.api_keys)
    app.state.policy = PolicyStore(resolved.routing_policy_path)
    app.state.provider_environ = os.environ if environ is None else environ
    app.state.adapters = build_adapters(app.state.policy.policy, app.state.provider_environ)
    app.state.breaker = CircuitBreaker()
    redis_client = (
        build_redis_client(resolved.redis_url) if resolved.redis_url is not None else None
    )
    store: RedactionMapStore = (
        RedisRedactionMapStore(redis_client)
        if redis_client is not None
        else InProcessRedactionMapStore()
    )
    app.state.redaction_map_store = store
    directory: DirectoryStore = (
        RedisDirectoryStore(redis_client) if redis_client is not None else InProcessDirectoryStore()
    )
    app.state.directory_store = directory
    response_cache: ResponseCache = (
        RedisResponseCache(redis_client) if redis_client is not None else InProcessResponseCache()
    )
    app.state.response_cache = response_cache
    budget_store: BudgetStore = (
        RedisBudgetStore(redis_client) if redis_client is not None else InProcessBudgetStore()
    )
    app.state.budget_store = budget_store
    rate_limit_store: RateLimitStore = (
        RedisRateLimitStore(redis_client) if redis_client is not None else InProcessRateLimitStore()
    )
    app.state.rate_limit_store = rate_limit_store
    semantic_cache: SemanticCache = (
        QdrantSemanticCache(build_qdrant_data_client(resolved.qdrant_url))
        if resolved.qdrant_url is not None
        else InProcessSemanticCache()
    )
    app.state.semantic_cache = semantic_cache
    app.state.prices = PriceStore(resolved.prices_path)
    pool = build_pool(resolved.database_url) if resolved.database_url is not None else None
    app.state.db_pool = pool
    residency_log: ResidencyLogWriter = (
        PostgresResidencyLogWriter(pool) if pool is not None else InMemoryResidencyLogWriter()
    )
    app.state.residency_log = residency_log
    cost_ledger: CostLedgerWriter = (
        PostgresCostLedgerWriter(pool) if pool is not None else InMemoryCostLedgerWriter()
    )
    app.state.cost_ledger = cost_ledger
    console_events: ConsoleEventWriter = (
        PostgresConsoleEventWriter(pool) if pool is not None else InMemoryConsoleEventWriter()
    )
    app.state.console_events = console_events
    console_reader: ConsoleReader = (
        PostgresConsoleReader(pool) if pool is not None else UnconfiguredConsoleReader()
    )
    app.state.console_reader = console_reader
    app.state.prometheus = (
        PrometheusClient(build_prometheus_http(resolved.prometheus_url))
        if resolved.prometheus_url is not None
        else None
    )
    register_probes(app)
    register_error_handlers(app)
    install_exporter(resolved)
    app.add_middleware(TracingMiddleware)
    include_routers(app)
    return app


app: FastAPI = create_app()


def serve(reload: bool = False) -> None:
    """Run the gateway on the configured port.

    The listening port comes from `Settings`, so `OPSCONSOLE_PORT` reaches the
    socket. Pinned by `test_serve_uses_the_configured_port`. Clients that dial
    the gateway (`make eval`, the compose healthcheck) carry their own address
    and are not covered by that test.
    """
    uvicorn.run("app.main:app", port=Settings().port, reload=reload)
