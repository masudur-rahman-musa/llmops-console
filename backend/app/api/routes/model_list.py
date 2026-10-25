"""`GET /v1/models`: the aliases the caller may actually use.

The list is the aliases that resolve for the calling tenant
under the request's data class, not every alias the policy defines. Listing an
alias that answers 503 on first use advertises a model the caller cannot have,
which is the opposite of what the endpoint is for. The cost is one `resolve()`
per alias per call, over a snapshot already in memory.

The list is the caller's own and never a probe into another tenant: `X-Tenant`
has to match the key before this route runs, so naming someone else's tenant is
a 403. A tenant the policy does not
define resolves nothing and gets an empty list rather than an error.

Only the errors that mean "this alias is not for you" are swallowed, not the
`RoutingError` base. This route resolves without the health seam, so
`AllProvidersUnavailable` here means every candidate is turned off. A breaker
must not make a catalogue flicker.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from app.api.deps import Context, PolicyDep
from app.core.errors import TenantMismatch
from app.model_routing.errors import AliasUnknown, AllProvidersUnavailable
from app.model_routing.resolver import resolve

OWNER = "opsconsole"
"""`owned_by` on every entry. The aliases are ours; the provider behind one is
not something a client is told."""


class Model(BaseModel):
    """One alias, in the shape an OpenAI-compatible client expects."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: str = "model"
    owned_by: str = OWNER


class ModelList(BaseModel):
    """The `{"object": "list", "data": [...]}` envelope."""

    model_config = ConfigDict(frozen=True)

    object: str = "list"
    data: tuple[Model, ...]


router = APIRouter()


@router.get("/v1/models")
def list_models(context: Context, store: PolicyDep) -> ModelList:
    """List the aliases that resolve for this tenant and data class, in policy order."""
    policy = store.policy
    listed: list[Model] = []
    for alias in policy.aliases:
        try:
            resolve(policy, context.tenant, alias, context.data_class.value)
        except (AllProvidersUnavailable, TenantMismatch, AliasUnknown):
            continue
        listed.append(Model(id=alias))
    return ModelList(data=tuple(listed))
