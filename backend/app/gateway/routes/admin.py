"""The operator surface: the resolved view, the policy swap, and the two hand-written records.

Five routes, all behind a key space of their own:

    GET   /admin/routing            the resolver's own answer, per tenant
    POST  /admin/routing/reload     re-read routing.yaml and rebuild the adapters
    POST  /admin/incidents          record an incident
    PATCH /admin/incidents/{incident_id}   resolve one
    POST  /admin/releases           record a deploy

The last three are the two tables nothing else in the system produces: the
console's timeline and its deploys page are written by a human and by CI rather
than by a served request, which is why they arrive here rather than as a code
path. What they store, and why each field is checked before any statement runs,
is `app.ledger.console_events`.

`admin_key_sha256` is not `api_keys`: a working tenant key is 401 on every one of
them, and an unset admin key authorises nobody rather than everybody, which is
the failure direction that matters for routes that can rebuild the gateway's
provider set and write the records an operator reads
(`test_admin_routes_reject_a_tenant_api_key`,
`test_every_console_event_route_is_401_without_an_admin_key`,
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
HTTP scope. And both are now cluster-internal: the `gateway` Ingress routes `/v1`
and nothing wider, so no public router matches `/admin/*` and a request for one
from the internet is Traefik's own 404
(`tests/infra/test_k8s_apps.py::test_no_public_router_matches_an_admin_path_or_the_scrape`,
whose four paths are imported from here and from `directory.py` rather than
retyped). The admin key is unchanged and is still the control: the router rule
removed a way in, it did not replace a check, and anything reaching the pod
another way - an in-cluster caller through `Service/gateway`, a port-forward -
meets the key exactly as before. The network half of #70 is a third layer and a
different question: row 21a's NetworkPolicies decide which pods may open a
connection, never which path that connection may ask for.
"""

from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, ConfigDict

from app.core.errors import Unauthorized
from app.core.settings import Settings
from app.gateway.auth import bearer_key
from app.ledger.console_events import (
    EVAL_RUN_ID_FIELD,
    ConsoleEventWriter,
    IncidentRow,
    ReleaseRow,
    checked_incident_id,
)
from app.providers.base import Adapter
from app.providers.breaker import CircuitBreaker
from app.providers.factory import build_adapters, close_adapters
from app.routing.errors import (
    AliasUnknown,
    IdentifierInvalid,
    PolicyInvalid,
    ResidencyUnavailable,
)
from app.routing.policy import DataClass, Policy, PolicyStore, ResidencyMode
from app.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH, resolve

ADMIN_ROUTING = "/admin/routing"
"""The resolved-routing view. Pinned against the mounted route by
`test_the_admin_paths_and_statuses_are_pinned`."""

ADMIN_ROUTING_RELOAD = "/admin/routing/reload"
"""The policy reload. Pinned by `test_the_admin_paths_and_statuses_are_pinned`."""

ADMIN_INCIDENTS = "/admin/incidents"
"""Where an incident is recorded. Pinned against the mounted route by
`test_the_admin_console_event_paths_and_statuses_are_pinned`."""

ADMIN_INCIDENT = "/admin/incidents/{incident_id}"
"""Where one is resolved. Pinned by the same case."""

ADMIN_RELEASES = "/admin/releases"
"""Where a deploy is recorded. Pinned by the same case."""

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


class IncidentCreate(BaseModel):
    """What `POST /admin/incidents` accepts, and what it deliberately does NOT.

    There is no `resolved_at` field, and that absence is the criterion: an
    incident is born OPEN, a NULL `resolved_at` is what "open" means on the
    timeline, and a body offering one would let an operator file an outage that
    was never open. A caller sending the key anyway has it ignored by Pydantic's
    default extra handling rather than stored
    (`test_the_incident_post_returns_a_thirty_two_hex_id_and_never_sets_resolved_at`).

    `id` is absent for the other half of decision A: the server generates it, so
    no client string shapes a primary key.
    """

    model_config = ConfigDict(frozen=True)

    started_at: datetime
    severity: str
    title: str
    summary: str
    postmortem_url: str | None = None


class IncidentCreated(BaseModel):
    """The 201 body: the id the caller needs in order to resolve this incident later.

    The whole row is not echoed. The caller sent every other field, and the id is
    the one thing it does not already know (decision A).
    """

    model_config = ConfigDict(frozen=True)

    id: str


class IncidentResolve(BaseModel):
    """What `PATCH /admin/incidents/{incident_id}` accepts: the resolution time only.

    Nothing else on an incident is editable here. A wrong severity or a missing
    postmortem link is a correction rather than a state change, and this plan's
    surface is the state change; widening it later is a plan of its own.
    """

    model_config = ConfigDict(frozen=True)

    resolved_at: datetime


class ReleaseCreate(BaseModel):
    """What `POST /admin/releases` accepts: one deploy of one sha.

    `eval_run_id` and `pr_url` are optional because a sha can be deployed before
    any eval ran and from a branch that never had a pull request; both are NULL on
    the row rather than empty strings, which is the difference between "not
    measured" and "a run id that does not exist".
    """

    model_config = ConfigDict(frozen=True)

    git_sha: str
    deployed_at: datetime
    eval_run_id: str | None = None
    pr_url: str | None = None
    status: str


def _console_events(request: Request) -> ConsoleEventWriter:
    """The writer `create_app` built: Postgres when a `database_url` is configured,
    in-memory otherwise."""
    writer: ConsoleEventWriter = request.app.state.console_events
    return writer


@router.post(ADMIN_INCIDENTS, status_code=201)
def record_incident(request: Request, body: IncidentCreate) -> IncidentCreated:
    """Record one incident and answer the id it was filed under.

    201 rather than 200: this creates a record the caller can address afterwards,
    and the id in the body is that address. `uuid4().hex` rather than a slug the
    caller supplies (decision A), so two incidents filed in the same minute cannot
    collide on a name someone typed twice and no client string reaches a primary
    key.

    `resolved_at` is `None` and cannot be anything else - `IncidentCreate` has no
    such field. Every value the body did carry is pre-checked inside
    `write_incident` before any statement runs, and each refusal is an
    `IdentifierInvalid` naming its own field, which the error handler renders as
    422 `VALIDATION` with that field as `param`.

    Declared `def` rather than `async def`, like the two routes above: the write
    is blocking work over a pooled connection, so FastAPI running it in a worker
    thread is what keeps it off the event loop.
    """
    row = IncidentRow(
        id=uuid.uuid4().hex,
        started_at=body.started_at,
        resolved_at=None,
        severity=body.severity,
        title=body.title,
        summary=body.summary,
        postmortem_url=body.postmortem_url,
    )
    _console_events(request).write_incident(row)
    return IncidentCreated(id=row.id)


@router.patch(ADMIN_INCIDENT)
def resolve_incident(request: Request, incident_id: str, body: IncidentResolve) -> IncidentCreated:
    """Close one open incident, or refuse with the reason it could not be closed.

    The path parameter is the one place a client string takes the shape of an
    incident id, so it is pre-checked here as well as in the writer: a malformed
    id is refused before a connection is borrowed, exactly as `?tenant=` is on the
    view above.

    Both failures are 422 `VALIDATION` naming `id` and are
    told apart by their messages, which `ledger.console_events` owns:
    `ErrorCode.VALIDATION` is fixed at 422, and a 404 or a 409 carrying that code
    would break the one-status-per-code table
    `tests/gateway/test_error_contract.py` holds. Resolving twice is a refusal
    rather than a no-op (decision E), because the second call would otherwise
    overwrite the timestamp a postmortem's duration is computed from.

    Answers the id it resolved, so the response has the same shape as the create's
    and a caller scripting both reads one field from both.
    """
    checked_incident_id(incident_id)
    _console_events(request).resolve_incident(incident_id, body.resolved_at)
    return IncidentCreated(id=incident_id)


@router.post(ADMIN_RELEASES, status_code=201)
def record_release(request: Request, body: ReleaseCreate) -> ReleaseCreate:
    """Record one deploy of one sha.

    A named `eval_run_id` is checked for EXISTENCE before the insert. The column is
    a foreign key, so without this check a CI job posting a run whose rows never
    stored would reach the constraint and the integrity error would leave the route
    as a 500 - a deploy step failing with a stack trace instead of a typed message
    naming the input at fault (`test_a_release_naming_an_unstored_run_is_422_
    validation_naming_eval_run_id`). The check is the writer's, so the in-memory
    writer answers True for every well-formed id and a local run keeps working.

    Two deploys of one sha are two rows (decision B): nothing here looks for an
    existing row, because a rollback re-deploys an earlier sha and an upsert would
    erase the original deploy.

    Echoes the body rather than an id: the key is `(git_sha, deployed_at)`, both of
    which the caller sent, so there is nothing here it does not already know.
    """
    writer = _console_events(request)
    if body.eval_run_id is not None and not writer.eval_run_exists(body.eval_run_id):
        raise IdentifierInvalid(
            f"{EVAL_RUN_ID_FIELD}: no eval run carries this id", param=EVAL_RUN_ID_FIELD
        )
    writer.write_release(
        ReleaseRow(
            git_sha=body.git_sha,
            deployed_at=body.deployed_at,
            eval_run_id=body.eval_run_id,
            pr_url=body.pr_url,
            status=body.status,
        )
    )
    return body
