"""The gateway app factory."""

from __future__ import annotations

from fastapi import FastAPI

from opsconsole.core.settings import Settings
from opsconsole.gateway.auth import SettingsKeyStore
from opsconsole.gateway.errors import register_error_handlers
from opsconsole.gateway.health import ProbeRegistry
from opsconsole.gateway.health import router as health_router
from opsconsole.gateway.routes.models import router as models_router
from opsconsole.routing.policy import PolicyStore


def create_app(settings: Settings | None = None) -> FastAPI:
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
    """
    resolved = settings or Settings()
    app = FastAPI(title="AI Operations Console gateway")
    app.state.settings = resolved
    app.state.probes = ProbeRegistry()
    app.state.key_store = SettingsKeyStore(resolved.api_keys)
    app.state.policy = PolicyStore(resolved.routing_policy_path)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(models_router)
    return app
