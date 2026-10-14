"""`GET /admin/routing` and `POST /admin/routing/reload`: the resolved view and the policy swap.

Both behind the admin key space (`admin_key`), both cluster-internal: the
`gateway` Ingress routes `/v1` and nothing wider, so no public router matches
`/admin/*` and a request for one from the internet is Traefik's own 404
(`infra/tests/infra/test_k8s_apps.py::test_no_public_router_matches_an_admin_path_or_the_scrape`,
whose paths are imported from here and from `directory.py`).

The view is the resolver's own answer rather than a re-reading of `routing.yaml`.
For every tenant, every alias and every data class it calls `resolve()` and
records what came back, so what an operator debugging a 451 sees is the function
the request path runs, not a second implementation of it that can drift. Only
`ResidencyUnavailable` and `AliasUnknown` are swallowed - the two that mean "not
available to this tenant" - and deliberately not the `RoutingError` base.

`?tenant=` is the one client-supplied input on this surface and is pre-checked
against the resolver's own `MAX_IDENTIFIER_LENGTH` and `IDENTIFIER_PATTERN` by
`app.api.deps._checked_tenant` before any lookup happens
(`test_the_tenant_filter_length_boundary_is_inclusive`,
`test_the_tenant_filter_rejection_table`). A valid but unknown tenant is an empty
view rather than an error (`test_an_unknown_but_valid_tenant_is_an_empty_view`).

Reload parses before it rebinds, which `PolicyStore.reload` already guarantees,
then rebuilds the adapter set and breaker (`services.routing_reload`). A rejected
file is 422 `VALIDATION` carrying the field path `PolicyInvalid` already names,
and the previous snapshot goes on serving
(`test_an_invalid_policy_on_reload_is_422_naming_the_field`,
`test_a_failed_reload_keeps_the_previous_policy_serving`).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import _checked_tenant, admin_key
from app.model_routing.errors import (
    AliasUnknown,
    IdentifierInvalid,
    PolicyInvalid,
    ResidencyUnavailable,
)
from app.model_routing.policy import DataClass, Policy, PolicyStore, ResidencyMode
from app.model_routing.resolver import resolve
from app.schemas.admin import AliasView, ReloadResult, ResolvedEntry, RoutingView, TenantView
from app.services.routing_reload import rebuild_adapter_set

ADMIN_ROUTING = "/admin/routing"
"""The resolved-routing view. Pinned against the mounted route by
`test_the_admin_paths_and_statuses_are_pinned`."""

ADMIN_ROUTING_RELOAD = "/admin/routing/reload"
"""The policy reload. Pinned by `test_the_admin_paths_and_statuses_are_pinned`."""


def _tenant_view(policy: Policy, tenant: str, mode: ResidencyMode) -> TenantView:
    """Resolve every alias and data class for one tenant, in policy and enum order.

    `mode` is the process's own `residency_mode`, so this view shows what the
    gateway WOULD answer rather than what strict would: an operator reading it on
    a relaxed process has to see the relaxed resolution, or the one surface that
    explains routing disagrees with every route.
    """
    entries: list[AliasView] = []
    for alias in policy.aliases:
        for data_class in DataClass:
            try:
                resolved = resolve(policy, tenant, alias, data_class.value, mode=mode)
            except (ResidencyUnavailable, AliasUnknown):
                continue
            entries.append(
                AliasView(
                    alias=alias,
                    data_class=data_class.value,
                    resolved=tuple(
                        ResolvedEntry(provider=item.provider, model=item.model, region=item.region)
                        for item in resolved
                    ),
                )
            )
    return TenantView(tenant=tenant, entries=tuple(entries))


router = APIRouter(dependencies=[Depends(admin_key)])


@router.get(ADMIN_ROUTING)
def routing_view(request: Request, tenant: Annotated[str | None, Query()] = None) -> RoutingView:
    """Every tenant's resolved routing, or one tenant's when `?tenant=` names one.

    Declared `def` rather than `async def`: the matrix is tenants x aliases x data
    classes calls into `resolve()`, which is CPU work over an in-memory snapshot,
    and FastAPI runs a `def` route in a worker thread rather than on the event
    loop - the same reason the model routes give. On the shipped policy that is
    tens of calls; it is not a hot path and is not cached.
    """
    if tenant is not None:
        _checked_tenant(tenant)

    store: PolicyStore = request.app.state.policy
    policy = store.policy
    mode: ResidencyMode = request.app.state.settings.residency_mode
    names = [tenant] if tenant is not None else list(policy.tenants)
    return RoutingView(
        tenants=tuple(_tenant_view(policy, name, mode) for name in names if name in policy.tenants)
    )


@router.post(ADMIN_ROUTING_RELOAD)
def reload_routing(request: Request) -> ReloadResult:
    """Re-read `routing.yaml`, swap the snapshot, and rebuild the adapter set.

    The order is the whole safety property. `PolicyStore.reload` parses into a new
    `Policy` before it rebinds, so a file that does not validate leaves the
    previous snapshot serving and nothing here has touched the adapters yet
    (`test_a_failed_reload_keeps_the_previous_policy_serving`). Only once the new
    snapshot is in place does `rebuild_adapter_set` rebuild the adapters from it,
    swap them in, and close the PREVIOUS set.

    `PolicyInvalid` becomes a 422 `VALIDATION` carrying the message the loader
    built, which names the field path. The exception is deliberately not allowed
    to escape as a 500: the file is this endpoint's input, so a bad one is a
    request-level validation failure rather than a gateway fault (an earlier change
    decision D, `test_an_invalid_policy_on_reload_is_422_naming_the_field`).
    """
    store: PolicyStore = request.app.state.policy
    try:
        store.reload()
    except PolicyInvalid as exc:
        raise IdentifierInvalid(str(exc), param="routing.yaml") from exc

    return ReloadResult(reloaded=True, providers=rebuild_adapter_set(request.app))
