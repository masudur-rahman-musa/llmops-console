"""`/admin/routing` and `/admin/routing/reload`: the resolved view, and the swap.

Two routes for an operator, behind a key space of their own. `admin_key_sha256`
is not `api_keys`: a working tenant key is 401 here, and an unset admin key
authorises nobody rather than everybody, which is the failure direction that
matters for a route that can rebuild the gateway's provider set
(`test_admin_routes_reject_a_tenant_api_key`,
`test_admin_routes_reject_every_caller_when_no_admin_key_is_configured`).

The view is the resolver's own answer rather than a re-reading of `routing.yaml`.
For every tenant, every alias and every data class it calls `resolve()` and
records what came back, so what an operator debugging a 451 sees is the function
the request path runs, not a second implementation of it that can drift. Only
`ResidencyUnavailable` and `AliasUnknown` are swallowed - the two that mean "not
available to this tenant" - and deliberately not the `RoutingError` base, for the
reason `/v1/models` gives: once the breaker lands, `AllProvidersUnavailable`
means a provider incident, and a view that quietly shrinks during one reads as a
deleted alias. Until then the view resolves under the default always-healthy
seam, so it shows the POLICY's answer.

`?tenant=` is the one client-supplied input on this surface and is pre-checked
against the resolver's own `MAX_IDENTIFIER_LENGTH` and `IDENTIFIER_PATTERN`
before any lookup happens (`test_the_tenant_filter_length_boundary_is_inclusive`,
`test_the_tenant_filter_rejection_table`). A valid but unknown tenant is an empty
view rather than an error (`test_an_unknown_but_valid_tenant_is_an_empty_view`).

Reload parses before it rebinds, which `PolicyStore.reload` already guarantees,
then rebuilds the adapter set from the new policy and closes the previous one. Not closing it leaks an `httpx` connection pool per
reload; the cost is that a call already in flight on a dropped client fails, and
the chat route renders that as 502 `PROVIDER_ERROR`. A rejected file is 422
`VALIDATION` carrying the field path `PolicyInvalid` already names, and the
previous snapshot goes on serving (`test_an_invalid_policy_on_reload_is_422_naming_the_field`,
`test_a_failed_reload_keeps_the_previous_policy_serving`).

Neither route is counted by the request metrics: `COUNTED_PATH_PREFIX` is `/v1/`. Both are traced, because `TracingMiddleware` spans every
HTTP scope. And the admin key is the only control in front of them until row
21b takes `/admin/*` off the public router: row 21a's NetworkPolicies admit
Traefik to the gateway on its serving port and a policy cannot see a path, so
the network half of #70 closes who may open a connection and the router half is
what decides which paths that connection may ask for.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict

from opsconsole.core.errors import Unauthorized
from opsconsole.core.settings import Settings
from opsconsole.gateway.auth import bearer_key
from opsconsole.providers.base import Adapter
from opsconsole.providers.breaker import CircuitBreaker
from opsconsole.providers.factory import build_adapters, close_adapters
from opsconsole.routing.errors import (
    AliasUnknown,
    IdentifierInvalid,
    PolicyInvalid,
    ResidencyUnavailable,
)
from opsconsole.routing.policy import DataClass, Policy, PolicyStore
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH, resolve

ADMIN_ROUTING = "/admin/routing"
"""The resolved-routing view. Pinned against the mounted route by
`test_the_admin_paths_and_statuses_are_pinned`."""

ADMIN_ROUTING_RELOAD = "/admin/routing/reload"
"""The policy reload. Pinned by `test_the_admin_paths_and_statuses_are_pinned`."""

TENANT_PARAM = "tenant"
"""The query parameter the view filters on, and the `param` every rejection of it
names (`test_the_tenant_filter_rejection_table`)."""


def _checked_tenant(value: str) -> str:
    """Validate the `?tenant=` filter before it is used to look anything up.

    a project rule applied to the one client-supplied string on this surface.
    The cap and the pattern are the resolver's own rather than numbers copied
    here, so the filter and the request path can never disagree about what a
    tenant name is; `IDENTIFIER_PATTERN` is a `fullmatch`, which is what keeps a
    trailing newline out. The inclusive boundary is
    `test_the_tenant_filter_length_boundary_is_inclusive`.
    """
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"{TENANT_PARAM}: longer than {MAX_IDENTIFIER_LENGTH} characters",
            param=TENANT_PARAM,
        )
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{TENANT_PARAM}: not a lowercase slug", param=TENANT_PARAM)
    return value


def admin_key(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    """Authorise an `/admin/*` call, or raise `Unauthorized`.

    The header shape is `gateway.auth.bearer_key`'s, so the scheme handling and
    the length cap are the ones the tenant path already uses; only the store the
    key is checked against differs. The digest of the presented key is compared
    with the configured one under `hmac.compare_digest`, which does not leak the
    position of the first differing byte.

    An unset `admin_key_sha256` authorises nobody. It is compared rather than
    short-circuited, so there is no "not configured" branch a deployment could
    fall into open; the comparison simply never succeeds, because no key hashes
    to the empty string. Pinned by
    `test_admin_routes_reject_every_caller_when_no_admin_key_is_configured`.

    Nothing here logs or echoes the presented key: the refusal message says only
    that no usable admin key was presented, exactly as the tenant path's does
    (`test_admin_key_never_appears_in_body_or_log`).
    """
    settings: Settings = request.app.state.settings
    key = bearer_key(authorization)
    presented = hashlib.sha256(key.encode("utf-8")).hexdigest() if key is not None else ""
    if not key or not hmac.compare_digest(settings.admin_key_sha256, presented):
        raise Unauthorized("no usable admin key was presented")


class ResolvedEntry(BaseModel):
    """One place a triple resolves to: the provider, its model id and its region."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    region: str


class AliasView(BaseModel):
    """What one (alias, data class) resolves to for one tenant, in candidate order."""

    model_config = ConfigDict(frozen=True)

    alias: str
    data_class: str
    resolved: tuple[ResolvedEntry, ...]


class TenantView(BaseModel):
    """One tenant's whole resolved matrix. A triple that resolves to nothing is absent."""

    model_config = ConfigDict(frozen=True)

    tenant: str
    entries: tuple[AliasView, ...]


class RoutingView(BaseModel):
    """The view's envelope: the tenants asked for, in policy order."""

    model_config = ConfigDict(frozen=True)

    tenants: tuple[TenantView, ...]


class ReloadResult(BaseModel):
    """What a successful reload did: it happened, and this is the adapter set now.

    `providers` is the sorted key set of `app.state.adapters` after the rebuild,
    which is what makes "the reload dropped a provider" observable from outside
    the process (`test_a_reload_rebuilds_the_adapter_set_for_the_new_policy`). A
    failed reload is a 422 and produces no body of this shape at all.
    """

    model_config = ConfigDict(frozen=True)

    reloaded: bool
    providers: tuple[str, ...]


def _tenant_view(policy: Policy, tenant: str) -> TenantView:
    """Resolve every alias and data class for one tenant, in policy and enum order."""
    entries: list[AliasView] = []
    for alias in policy.aliases:
        for data_class in DataClass:
            try:
                resolved = resolve(policy, tenant, alias, data_class.value)
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
    names = [tenant] if tenant is not None else list(policy.tenants)
    return RoutingView(
        tenants=tuple(_tenant_view(policy, name) for name in names if name in policy.tenants)
    )


@router.post(ADMIN_ROUTING_RELOAD)
def reload_routing(request: Request) -> ReloadResult:
    """Re-read `routing.yaml`, swap the snapshot, and rebuild the adapter set.

    The order is the whole safety property. `PolicyStore.reload` parses into a new
    `Policy` before it rebinds, so a file that does not validate leaves the
    previous snapshot serving and nothing here has touched the adapters yet
    (`test_a_failed_reload_keeps_the_previous_policy_serving`). Only once the new
    snapshot is in place is the adapter set rebuilt from it, swapped in, and the
    PREVIOUS set closed - previous, not dropped: a rebuild mints a fresh client
    for every surviving provider too, so the old objects are all garbage, and
    closing them is what keeps a reload from leaking a connection pool
    (`test_a_reload_leaves_the_new_adapter_set_callable`).

    `PolicyInvalid` becomes a 422 `VALIDATION` carrying the message the loader
    built, which names the field path. The exception is deliberately not allowed
    to escape as a 500: the file is this endpoint's input, so a bad one is a
    request-level validation failure rather than a gateway fault (an earlier change
    decision D, `test_an_invalid_policy_on_reload_is_422_naming_the_field`).

    The circuit breaker is rebuilt with the adapter set, and for the same reason:
    both are keyed by the provider names of the policy that has just been
    replaced, so a reload that renamed or repointed a provider would otherwise
    leave a breaker open against a name that no longer means what it meant
    (`test_a_policy_reload_rebuilds_the_breaker`). The cost is that a reload
    forgets a genuinely failing provider and the seam re-learns it over the next
    five requests, which is the cheaper of the two mistakes.
    """
    store: PolicyStore = request.app.state.policy
    try:
        store.reload()
    except PolicyInvalid as exc:
        raise IdentifierInvalid(str(exc), param="routing.yaml") from exc

    previous: dict[str, Adapter] = request.app.state.adapters
    request.app.state.adapters = build_adapters(store.policy, request.app.state.provider_environ)
    request.app.state.breaker = CircuitBreaker()
    close_adapters(previous)
    return ReloadResult(reloaded=True, providers=tuple(sorted(request.app.state.adapters)))
