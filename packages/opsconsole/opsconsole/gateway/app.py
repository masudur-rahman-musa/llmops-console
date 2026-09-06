"""The gateway app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.context import Context
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
from opsconsole.gateway.routes.chat import router as chat_router
from opsconsole.gateway.routes.embeddings import router as embeddings_router
from opsconsole.gateway.routes.models import router as models_router
from opsconsole.gateway.routes.redact import router as redact_router
from opsconsole.providers.base import Adapter
from opsconsole.providers.factory import build_adapters
from opsconsole.redaction import InProcessRedactionMapStore
from opsconsole.routing.policy import PolicyStore


def close_adapters(adapters: Mapping[str, Adapter]) -> None:
    """Release whatever an adapter holds open, for every adapter that holds anything.

    `close()` is not on the `Adapter` protocol: an adapter with no connection
    pool has nothing to close, and requiring the method would make every future
    adapter write an empty one. `AnthropicAdapter` has it, and its `httpx.Client`
    owns a pool that outlives the process otherwise.

    An earlier change's policy reload calls this on the adapters it drops, which is why it
    is a module function rather than inline in the lifespan.
    """
    for adapter in adapters.values():
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


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
    """The root span of every request: opened before routing, closed after the last byte.

    Pure ASGI rather than `BaseHTTPMiddleware`, for two reasons that both decide
    an acceptance criterion. It runs OUTSIDE `ExceptionMiddleware`, so a request
    that fails before any route - an unauthenticated one, which never builds a
    `RequestContext` at all - still has a span to record the failure on
    (`test_an_unauthorized_request_still_emits_a_trace`). And it ends the span
    only once the response body is fully sent, which for a streamed completion is
    after the last frame rather than when the route returned
    (`test_a_streamed_requests_root_span_covers_the_frames`).

    The outcome is read off the wire rather than guessed: `send` is wrapped so the
    response's own status decides whether the root span failed, and the typed code
    comes from whatever the handlers in `gateway/errors.py` recorded on the scope.
    A crash that never produces a response start is a failure too, which is why
    the initial status is 500 rather than 200.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Trace one HTTP request; pass lifespan and websocket scopes straight through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        record = start_root_span(incoming_parent(scope))
        scope[TRACE_SCOPE_KEY] = record
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
            raise
        end_root_span(record, failed=status >= ERROR_STATUS_FLOOR)


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

    `environ` is where the provider keys are read from, injectable for the same
    reason `build_adapters` takes it (#66): a test passes `{}` and no adapter is
    built from a key that happens to be exported in the developer's shell, so a
    suite that would make a real billed call cannot pass locally and fail in CI.
    Pinned by `test_create_app_reads_keys_only_from_the_given_environ`.

    The redaction map store is built here too, and it is the one piece of
    per-process STATE the app owns: `InProcessRedactionMapStore` keeps the
    placeholder maps `/v1/redact` files, so two replicas do not share them. Nothing reads a map before an earlier change, and an earlier change
    replaces this with a Redis implementation of the same protocol.
    """
    resolved = settings or Settings()
    app = FastAPI(title="AI Operations Console gateway", lifespan=lifespan)
    app.state.settings = resolved
    app.state.probes = ProbeRegistry()
    app.state.key_store = SettingsKeyStore(resolved.api_keys)
    app.state.policy = PolicyStore(resolved.routing_policy_path)
    app.state.adapters = build_adapters(app.state.policy.policy, environ)
    app.state.redaction_map_store = InProcessRedactionMapStore()
    register_error_handlers(app)
    install_exporter(resolved)
    app.add_middleware(TracingMiddleware)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(redact_router)
    return app
