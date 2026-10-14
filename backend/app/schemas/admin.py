"""The request and response models for the operator surface.

The resolved-routing view's shapes, the reload result, and the bodies the two
hand-written records accept. What each field is checked against before a
statement runs is `app.storage.console_events` (today `app.ledger.console_events`).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
