"""The gateway app factory.

`close_adapters` is imported here and used by `lifespan`; it is defined in
`providers.factory`, beside the `build_adapters` it undoes. Importing it here
re-exports it - in the `import X as X` form `mypy --strict` requires for an
explicit re-export - so `from opsconsole.gateway.app import close_adapters`,
which the wiring tests use, keeps working while `gateway.routes.admin` reaches
the same function without importing this module back (this module imports every
route module, so that would be a cycle).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.context import Context
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from opsconsole.cache import InProcessResponseCache, RedisResponseCache, ResponseCache
from opsconsole.core.metrics import observe_request, start_request
from opsconsole.core.settings import Settings
from opsconsole.core.tracing import (
    ERROR_STATUS_FLOOR,
    TRACE_SCOPE_KEY,
    end_root_span,
    flush_tracing,
    install_exporter,
    parent_context,
    start_root_span,
)
from opsconsole.gateway.auth import SettingsKeyStore
from opsconsole.gateway.context import parsed_traceparent
from opsconsole.gateway.errors import register_error_handlers
from opsconsole.gateway.health import ProbeRegistry
from opsconsole.gateway.health import router as health_router
from opsconsole.gateway.probes import build_redis_client, register_probes
from opsconsole.gateway.routes.admin import router as admin_router
from opsconsole.gateway.routes.chat import router as chat_router
from opsconsole.gateway.routes.directory import router as directory_router
from opsconsole.gateway.routes.embeddings import router as embeddings_router
from opsconsole.gateway.routes.metrics import router as metrics_router
from opsconsole.gateway.routes.models import router as models_router
from opsconsole.gateway.routes.redact import router as redact_router
from opsconsole.ledger import (
    InMemoryResidencyLogWriter,
    PostgresResidencyLogWriter,
    ResidencyLogWriter,
    build_pool,
)
from opsconsole.providers.breaker import CircuitBreaker
from opsconsole.providers.factory import build_adapters
from opsconsole.providers.factory import close_adapters as close_adapters
from opsconsole.redaction import (
    DirectoryStore,
    InProcessDirectoryStore,
    InProcessRedactionMapStore,
    RedactionMapStore,
    RedisDirectoryStore,
    RedisRedactionMapStore,
)
from opsconsole.routing.policy import PolicyStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the adapter set on shutdown.

    Nothing is built here: the key store, the policy and the adapters are built
    in `create_app`, so a misconfiguration stops the process from starting rather
    than surfacing on the first request. This is only the other end of that.
    """
    try:
        yield
    finally:
        close_adapters(app.state.adapters)
        flush_tracing()


def incoming_parent(scope: Scope) -> Context | None:
    """The remote parent this request's `traceparent` names, when it names a usable one.

    Parsed with `gateway.context.parsed_traceparent` rather than with
    OpenTelemetry's own propagator, because the propagator accepts headers this
    gateway has already decided to reject. One parser means the trace a request
    JOINS and the header contract `X-Gateway-Trace-Id` answers under can never
    disagree about what a valid upstream trace is; the rejection table is
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
    handlers in `gateway/errors.py` recorded on the scope. A crash that never
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

    The residency log writer is settings-driven for the same reason and by the
    same rule: `OPSCONSOLE_DATABASE_URL` set builds `PostgresResidencyLogWriter`
    over a pool, absent builds `InMemoryResidencyLogWriter`, which files rows in
    this process and is what lets the suite and a bare local run open no
    connection at all
    (`test_the_app_builds_the_in_process_writer_without_a_database_url`). The
    migration is NOT applied here: `make migrate` is a deploy step, so that
    replicas coming up together do not race each other over the same DDL.

    `register_probes` runs last of the state-building steps, because the probes it
    registers close over the adapter mapping and the settings built above; what it
    registers is decided by which backing services are configured
    (`test_create_app_registers_the_probes_its_settings_configure`).
    """
    resolved = settings or Settings()
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
    residency_log: ResidencyLogWriter = (
        PostgresResidencyLogWriter(build_pool(resolved.database_url))
        if resolved.database_url is not None
        else InMemoryResidencyLogWriter()
    )
    app.state.residency_log = residency_log
    register_probes(app)
    register_error_handlers(app)
    install_exporter(resolved)
    app.add_middleware(TracingMiddleware)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(redact_router)
    app.include_router(admin_router)
    app.include_router(directory_router)
    return app
