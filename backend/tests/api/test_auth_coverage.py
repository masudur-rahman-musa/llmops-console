"""Auth coverage tests for an earlier change: one auth layer that blocks by default.

Every route mounted on the app must either:
- be in OPEN_PATHS (exactly /healthz and /readyz), or
- be /metrics (unauthenticated but cluster-only), or
- carry request_context (caller check) or admin_key in its dependency chain.

A new route that fits none of these must fail the walk test so the build goes
red rather than shipping an open route.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute, _IncludedRouter
from fastapi.testclient import TestClient

from app.api.deps import admin_key, request_context
from app.api.router import OPEN_PATHS
from app.main import create_app

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _route_dep_calls(route: APIRoute) -> set[object]:
    """Every callable in the route's own dependency chain.

    FastAPI stores the resolved dependency graph on ``route.dependant``.
    We walk the top-level list of ``Dependant`` objects it holds and collect
    every ``call`` we find.
    """
    calls: set[object] = set()

    def _walk(dependant: object) -> None:
        deps = getattr(dependant, "dependencies", [])
        for dep in deps:
            call = getattr(dep, "call", None)
            if call is not None:
                calls.add(call)
            _walk(dep)

    _walk(route.dependant)
    return calls


def _router_dep_calls(router: object) -> set[object]:
    """Callables from an APIRouter's own ``dependencies`` list.

    In FastAPI's newer ``_IncludedRouter`` model, parent-router dependencies
    are NOT merged into each child route's dependant — they stay on the router
    and are applied at runtime. This helper reads them from
    ``_IncludedRouter.include_context.included_router.dependencies``.
    """
    calls: set[object] = set()
    for dep in getattr(router, "dependencies", []):
        fn = getattr(dep, "dependency", None)
        if fn is not None:
            calls.add(fn)
    return calls


def _all_routes_with_router_deps(
    routes: Sequence[object],
    parent_calls: frozenset[object] = frozenset(),
) -> list[tuple[APIRoute, frozenset[object]]]:
    """Every (route, effective_dep_calls) pair, walking nested _IncludedRouters.

    ``effective_dep_calls`` is the union of the route's own dependant chain
    and every ancestor router's dependencies, so a route covered only by the
    parent router (like admin/console routes) still shows the parent's check.
    """
    result: list[tuple[APIRoute, frozenset[object]]] = []
    for r in routes:
        if isinstance(r, APIRoute):
            route_calls = _route_dep_calls(r)
            result.append((r, parent_calls | route_calls))
        elif isinstance(r, _IncludedRouter):
            # The router's own deps live on include_context.included_router.
            incl = r.include_context.included_router
            router_calls = _router_dep_calls(incl)
            result.extend(
                _all_routes_with_router_deps(
                    r.original_router.routes,
                    parent_calls | router_calls,
                )
            )
        elif hasattr(r, "routes"):
            result.extend(_all_routes_with_router_deps(r.routes, parent_calls))
    return result


def _all_routes(app: FastAPI) -> list[APIRoute]:
    """Every APIRoute mounted on the app, walking nested _IncludedRouters."""
    return [r for r, _ in _all_routes_with_router_deps(app.routes)]


# ---------------------------------------------------------------------------
# test_the_open_list_is_exactly_healthz_and_readyz
# ---------------------------------------------------------------------------


def test_the_open_list_is_exactly_healthz_and_readyz() -> None:
    """Catches an implementation that adds a path to OPEN_PATHS that should not
    be there, or forgets one of the two required paths, or puts /metrics in."""
    assert OPEN_PATHS == ("/healthz", "/readyz")
    assert "/metrics" not in OPEN_PATHS


# ---------------------------------------------------------------------------
# test_every_route_is_covered_by_the_one_auth_layer
# ---------------------------------------------------------------------------


def test_every_route_is_covered_by_the_one_auth_layer() -> None:
    """Catches an implementation that mounts a route without putting it under a
    parent router that enforces the caller check or admin key, so a new route
    could be open to the internet with no auth."""
    app = create_app(environ={})
    # Use the router-aware walk so parent-router deps (e.g. admin_key on the
    # operator parent) count as covering the route even when the handler itself
    # does not declare the dep.
    route_pairs = _all_routes_with_router_deps(app.routes)

    uncovered: list[str] = []
    v1_without_caller: list[str] = []
    admin_without_key: list[str] = []

    for route, calls in route_pairs:
        path = route.path
        has_caller = request_context in calls
        has_admin = admin_key in calls

        if path in OPEN_PATHS or path == "/metrics":
            # open or cluster-only — no auth required
            continue

        if not has_caller and not has_admin:
            uncovered.append(path)
            continue

        if path.startswith("/v1") and not has_caller:
            v1_without_caller.append(path)

        if (path.startswith("/admin") or path.startswith("/console")) and not has_admin:
            admin_without_key.append(path)

    assert not uncovered, f"routes with no auth check: {uncovered}"
    assert not v1_without_caller, f"/v1 routes missing caller check: {v1_without_caller}"
    assert not admin_without_key, (
        f"/admin or /console routes missing admin key: {admin_without_key}"
    )


# ---------------------------------------------------------------------------
# test_a_new_v1_route_with_no_caller_check_is_blocked
# ---------------------------------------------------------------------------


def test_a_new_v1_route_with_no_caller_check_is_blocked() -> None:
    """Catches an implementation where the caller parent does not enforce the
    caller check on a child route that declares no auth of its own."""
    from app.api.router import caller_router

    # Build a fresh parent with the same dependencies as caller_router — do NOT
    # mutate the module-level router, which would accumulate routes across tests.
    _parent = APIRouter(dependencies=list(caller_router.dependencies))

    @_parent.get("/v1/test-bare-no-auth")
    def bare_handler() -> dict[str, str]:
        return {"ok": "yes"}

    # create_app gives us a real app.state.key_store that request_context needs.
    app = create_app(environ={})
    app.include_router(_parent)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/v1/test-bare-no-auth")

    assert response.status_code == 401, (
        f"expected 401 for a no-key request to a bare /v1 route, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# test_a_new_admin_route_with_no_key_check_is_blocked
# ---------------------------------------------------------------------------


def test_a_new_admin_route_with_no_key_check_is_blocked() -> None:
    """Catches an implementation where the operator parent does not enforce the
    admin key on a child route that declares no auth of its own."""
    from app.api.router import operator_router

    # Same approach as the v1 case: copy deps, don't mutate the module object.
    _parent = APIRouter(dependencies=list(operator_router.dependencies))

    @_parent.get("/admin/test-bare-no-auth")
    def bare_admin_handler() -> dict[str, str]:
        return {"ok": "yes"}

    app = create_app(environ={})
    app.include_router(_parent)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/admin/test-bare-no-auth")

    assert response.status_code == 401, (
        f"expected 401 for a no-key request to a bare /admin route, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# test_metrics_is_cluster_only_and_unauthenticated
# ---------------------------------------------------------------------------


def test_metrics_is_cluster_only_and_unauthenticated() -> None:
    """Catches an implementation that accidentally puts /metrics behind an auth
    check, which would break the in-cluster Prometheus scrape."""
    app = create_app(environ={})

    # /metrics must not be in OPEN_PATHS (already pinned separately) and must
    # not carry either auth check in its dependency chain.
    metrics_routes = [
        r for r in _all_routes(app) if isinstance(r, APIRoute) and r.path == "/metrics"
    ]
    assert metrics_routes, "/metrics route not found on the app"

    for route in metrics_routes:
        calls = _route_dep_calls(route)
        assert request_context not in calls, "/metrics must not require the caller check"
        assert admin_key not in calls, "/metrics must not require the admin key"
        assert route.path not in OPEN_PATHS, "/metrics must not be in OPEN_PATHS"

    # A no-key request must still answer 200.
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200, (
        f"expected 200 for an unauthenticated /metrics scrape, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# test_the_caller_check_runs_once_when_a_route_also_declares_context
# ---------------------------------------------------------------------------


def test_the_caller_check_runs_once_when_a_route_also_declares_context() -> None:
    """Catches an implementation where a /v1 route that sits under the caller
    parent AND declares Context in its handler runs the auth check twice, producing
    two auth spans instead of one."""
    app = create_app(environ={})

    # Find /v1/chat/completions — it declares Context in its handler and will
    # also sit under the caller parent after an earlier change.
    chat_routes = [
        r for r in _all_routes(app) if isinstance(r, APIRoute) and r.path == "/v1/chat/completions"
    ]
    assert chat_routes, "/v1/chat/completions not found — is the app wired?"

    for route in chat_routes:
        # Collect every occurrence of request_context across the full tree,
        # counting duplicates: FastAPI deduplicates with skip_repeats=True in
        # get_flat_dependant, but we want to see it listed MORE than once only
        # if the tree really has it twice (once from the parent, once from the handler).
        # We use the raw tree traversal so we can count.
        count = _count_dep_call(route, request_context)
        assert count <= 1, (
            f"request_context appears {count} times in the dependency tree of "
            f"{route.path}; the caller check must run exactly once"
        )


def _count_dep_call(route: APIRoute, target: object) -> int:
    """How many times `target` appears as a dependency call in the route's tree.

    This counts raw nodes (before FastAPI's skip_repeats deduplication), so it
    measures whether the same function was registered twice.
    """
    count = 0

    def _walk(dependant: object) -> None:
        nonlocal count
        deps = getattr(dependant, "dependencies", [])
        for dep in deps:
            call = getattr(dep, "call", None)
            if call is target:
                count += 1
            _walk(dep)

    _walk(route.dependant)
    return count
