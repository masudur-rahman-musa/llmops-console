"""The request and response models for the operator surface.

The resolved-routing view's shapes, the reload result, and the body the
hand-written release record accepts. What each field is checked against before a
statement runs is `app.storage.console_events`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ResolvedEntry(BaseModel):
    """One place a triple resolves to: the provider and its model id."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str


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
    the process. A
    failed reload is a 422 and produces no body of this shape at all.
    """

    model_config = ConfigDict(frozen=True)

    reloaded: bool
    providers: tuple[str, ...]


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
