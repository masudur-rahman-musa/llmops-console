"""The gateway app factory."""

from __future__ import annotations

from fastapi import FastAPI

from opsconsole.core.settings import Settings
from opsconsole.gateway.health import ProbeRegistry
from opsconsole.gateway.health import router as health_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the gateway app.

    Each call gets its own `ProbeRegistry` on `app.state`, so probes registered
    against one app never leak into another. Pinned by
    `test_registries_are_per_app`.
    """
    resolved = settings or Settings()
    app = FastAPI(title="AI Operations Console gateway")
    app.state.settings = resolved
    app.state.probes = ProbeRegistry()
    app.include_router(health_router)
    return app
