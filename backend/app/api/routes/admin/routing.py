"""`GET /admin/routing` and `POST /admin/routing/reload`: the resolved view and the policy swap.

Both behind the admin key space (`admin_key`), both cluster-internal: the
`gateway` Ingress routes `/v1` and nothing wider, so no public router matches
`/admin/*` and a request for one from the internet is Traefik's own 404
(the infra tests import the paths from here and from `directory.py` to check that).

The view is the resolver's own answer rather than a re-reading of `routing.yaml`.
For every tenant, every alias and every data class it calls `resolve()` and
records what came back, so what an operator debugging a 503 sees is the function
the request path runs, not a second implementation of it that can drift. Only
`AllProvidersUnavailable` and `AliasUnknown` are swallowed - the two that mean "not
available to this tenant" - and deliberately not the `RoutingError` base.

`?tenant=` is the one client-supplied input on this surface and is pre-checked
against the resolver's own `MAX_IDENTIFIER_LENGTH` and `IDENTIFIER_PATTERN` by
`app.api.deps.checked_tenant` before any lookup happens. A valid but unknown tenant is an empty
view rather than an error.

Reload delegates entirely to `RoutingRuntimeStore.reload`, which parses the new
policy, builds new adapters and a new breaker, and swaps all three atomically
under one lock. A file that does not parse raises `PolicyInvalid` before any
state is touched, so the previous set goes on serving.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import (
    PolicyDep,
    ProviderEnvironDep,
    RoutingStoreDep,
    checked_tenant,
)
from app.core.validation import IdentifierInvalid
from app.model_routing.errors import (
    AliasUnknown,
    AllProvidersUnavailable,
    PolicyInvalid,
)
from app.model_routing.policy import DataClass, Policy
from app.model_routing.resolver import resolve
from app.schemas.admin import AliasView, ReloadResult, ResolvedEntry, RoutingView, TenantView

ADMIN_ROUTING = "/admin/routing"
"""The resolved-routing view."""

ADMIN_ROUTING_RELOAD = "/admin/routing/reload"
"""The policy reload."""


def _tenant_view(policy: Policy, tenant: str) -> TenantView:
    """Resolve every alias and data class for one tenant, in policy and enum order."""
    entries: list[AliasView] = []
    for alias in policy.aliases:
        for data_class in DataClass:
            try:
                resolved = resolve(policy, tenant, alias, data_class.value)
            except (AllProvidersUnavailable, AliasUnknown):
                continue
            entries.append(
                AliasView(
                    alias=alias,
                    data_class=data_class.value,
                    resolved=tuple(
                        ResolvedEntry(provider=item.provider, model=item.model) for item in resolved
                    ),
                )
            )
    return TenantView(tenant=tenant, entries=tuple(entries))


router = APIRouter()


@router.get(ADMIN_ROUTING)
def routing_view(store: PolicyDep, tenant: Annotated[str | None, Query()] = None) -> RoutingView:
    """Every tenant's resolved routing, or one tenant's when `?tenant=` names one.

    Declared `def` rather than `async def`: the matrix is tenants x aliases x data
    classes calls into `resolve()`, which is CPU work over an in-memory snapshot,
    and FastAPI runs a `def` route in a worker thread rather than on the event
    loop - the same reason the model routes give. On the shipped policy that is
    tens of calls; it is not a hot path and is not cached.
    """
    if tenant is not None:
        checked_tenant(tenant)

    policy = store.policy
    names = [tenant] if tenant is not None else list(policy.tenants)
    return RoutingView(
        tenants=tuple(_tenant_view(policy, name) for name in names if name in policy.tenants)
    )


@router.post(ADMIN_ROUTING_RELOAD)
def reload_routing(
    store: PolicyDep, routing_store: RoutingStoreDep, environ: ProviderEnvironDep
) -> ReloadResult:
    """Re-read `routing.yaml`, swap the policy, adapters and breaker as one, close the old set.

    `RoutingRuntimeStore.reload` parses the new policy outside the lock, so a
    file that does not parse raises `PolicyInvalid` before any state is touched.
    Under the lock it builds new adapters and a new breaker, then swaps all three
    atomically - so `acquire()` always sees either the fully old or the fully new
    set, never a mix.

    `PolicyInvalid` becomes a 422 `VALIDATION` carrying the message the loader
    built, which names the field path. The exception is deliberately not allowed
    to escape as a 500: the file is this endpoint's input, so a bad one is a
    request-level validation failure rather than a gateway fault.
    """
    try:
        providers = routing_store.reload(environ)
    except PolicyInvalid as exc:
        raise IdentifierInvalid(str(exc), param="routing.yaml") from exc
    return ReloadResult(reloaded=True, providers=providers)
