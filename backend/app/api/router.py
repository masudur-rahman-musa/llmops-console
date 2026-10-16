"""Mount every route onto the app, in one place.

`create_app` in `app.main` builds the state; this wires the routers onto it, so
the include order lives beside the routes rather than inside the factory.

Auth is enforced at the parent-router level, not by each route individually.
Every `/v1` route sits under `caller_router` (checks the bearer API key via
`request_context`); every `/admin` and `/console` route sits under
`operator_router` (checks the admin key). Two named paths skip both:
`OPEN_PATHS` holds exactly `/healthz` and `/readyz`. `/metrics` is outside both
parents and `OPEN_PATHS` — it is unauthenticated in the app and kept off the
public port by the gateway Ingress, not by a key.

A new route added under a parent is checked without declaring anything. A route
added outside both parents and outside `OPEN_PATHS` fails
`test_every_route_is_covered_by_the_one_auth_layer`, so the build goes red
rather than shipping an open route.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from app.api.deps import admin_key, request_context
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

OPEN_PATHS: tuple[str, ...] = ("/healthz", "/readyz")
"""Paths that skip both auth checks. Pinned by
`test_the_open_list_is_exactly_healthz_and_readyz`."""

CLUSTER_ONLY_PATHS: tuple[str, ...] = ("/metrics",)
"""Paths that are unauthenticated in the app but kept off the public port by
the gateway Ingress. Not in `OPEN_PATHS` because that list is for the walk
test's "these need no key" category; `/metrics` needs no key AND must not
be reachable from the internet."""

caller_router: APIRouter = APIRouter(dependencies=[Depends(request_context)])
"""The parent that every `/v1` route sits under. Exported so tests can copy
its dependencies when proving that a bare child route gets 401 without a key.
No routes live here; `include_routers` builds a fresh intermediary each call."""

operator_router: APIRouter = APIRouter(dependencies=[Depends(admin_key)])
"""The parent that every `/admin` and `/console` route sits under. Exported
for the same reason as `caller_router`. No routes live here."""


def include_routers(app: FastAPI) -> None:
    """Include every route module's router on `app`.

    Builds fresh intermediary routers each call so that creating multiple apps
    (e.g. in tests) never accumulates duplicate routes on the module-level
    `caller_router` or `operator_router`.
    """
    # /healthz, /readyz, /metrics — no auth.
    app.include_router(health_router)

    # Every /v1 route behind the caller check.
    _caller = APIRouter(dependencies=list(caller_router.dependencies))
    _caller.include_router(model_list_router)
    _caller.include_router(chat_router)
    _caller.include_router(embeddings_router)
    _caller.include_router(redact_router)
    app.include_router(_caller)

    # Every /admin and /console route behind the admin key.
    _operator = APIRouter(dependencies=list(operator_router.dependencies))
    _operator.include_router(routing_router)
    _operator.include_router(incidents_router)
    _operator.include_router(releases_router)
    _operator.include_router(dashboard_router)
    _operator.include_router(directory_router)
    app.include_router(_operator)
