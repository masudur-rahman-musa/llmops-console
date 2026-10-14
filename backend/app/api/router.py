"""Mount every route onto the app, in one place.

`create_app` in `app.main` builds the state; this wires the routers onto it, so
the include order lives beside the routes rather than inside the factory. The
order is the one `create_app` used before the split: health and the scrape, the
model list, the three `/v1` pipelines, then the operator surface and the console.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.admin.directory import router as directory_router
from app.api.routes.admin.incidents import router as incidents_router
from app.api.routes.admin.releases import router as releases_router
from app.api.routes.admin.routing import router as routing_router
from app.api.routes.chat import router as chat_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.health import router as health_router
from app.api.routes.model_list import router as model_list_router
from app.api.routes.redact import router as redact_router


def include_routers(app: FastAPI) -> None:
    """Include every route module's router on `app`."""
    app.include_router(health_router)
    app.include_router(model_list_router)
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(redact_router)
    app.include_router(routing_router)
    app.include_router(incidents_router)
    app.include_router(releases_router)
    app.include_router(dashboard_router)
    app.include_router(directory_router)
