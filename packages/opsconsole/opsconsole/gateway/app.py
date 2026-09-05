"""The gateway app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI

from opsconsole.core.settings import Settings
from opsconsole.gateway.auth import SettingsKeyStore
from opsconsole.gateway.errors import register_error_handlers
from opsconsole.gateway.health import ProbeRegistry
from opsconsole.gateway.health import router as health_router
from opsconsole.gateway.routes.chat import router as chat_router
from opsconsole.gateway.routes.embeddings import router as embeddings_router
from opsconsole.gateway.routes.models import router as models_router
from opsconsole.providers.base import Adapter
from opsconsole.providers.factory import build_adapters
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
    """
    resolved = settings or Settings()
    app = FastAPI(title="AI Operations Console gateway", lifespan=lifespan)
    app.state.settings = resolved
    app.state.probes = ProbeRegistry()
    app.state.key_store = SettingsKeyStore(resolved.api_keys)
    app.state.policy = PolicyStore(resolved.routing_policy_path)
    app.state.adapters = build_adapters(app.state.policy.policy, environ)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    return app
