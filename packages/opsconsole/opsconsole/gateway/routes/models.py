"""`GET /v1/models`: the aliases the caller may actually use.

An earlier change decision D: the list is the aliases that resolve for the calling tenant
under the request's data class, not every alias the policy defines. Listing an
alias that answers 451 on first use advertises a model the caller cannot have,
which is the opposite of what the endpoint is for. The cost is one `resolve()`
per alias per call, over a snapshot already in memory.

The list is the caller's own and never a probe into another tenant: `X-Tenant`
has to match the key before this route runs, so naming someone else's tenant is
a 403 (`test_models_ignores_another_tenants_rules`). A tenant the policy does not
define resolves nothing and gets an empty list, which is deny-by-default rather
than an error (`test_models_for_a_tenant_the_policy_does_not_define_is_empty`).

Only the two errors that mean "this alias is not for you" are swallowed, not the
`RoutingError` base. `AllProvidersUnavailable` now means a provider incident -
every candidate's adapter missing, or every candidate's breaker open - and a
catalogue that quietly shrank during one would read to an operator as an alias
that had been deleted. This route resolves without the health seam for the same
reason: a breaker must not make a catalogue flicker.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from opsconsole.gateway.context import Context
from opsconsole.routing.errors import AliasUnknown, ResidencyUnavailable
from opsconsole.routing.policy import PolicyStore, ResidencyMode
from opsconsole.routing.resolver import resolve

OWNER = "opsconsole"
"""`owned_by` on every entry. The aliases are ours; the provider behind one is
not something a client is told. Pinned by `test_models_lists_only_resolvable_aliases`."""


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
def list_models(request: Request, context: Context) -> ModelList:
    """List the aliases that resolve for this tenant and data class, in policy order."""
    store: PolicyStore = request.app.state.policy
    policy = store.policy
    mode: ResidencyMode = request.app.state.settings.residency_mode
    listed: list[Model] = []
    for alias in policy.aliases:
        try:
            resolve(policy, context.tenant, alias, context.data_class.value, mode=mode)
        except (ResidencyUnavailable, AliasUnknown):
            continue
        listed.append(Model(id=alias))
    return ModelList(data=tuple(listed))
