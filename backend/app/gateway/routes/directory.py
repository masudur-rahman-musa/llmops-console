"""`POST /admin/directory`: the operator endpoint that seeds a tenant's display names.

The third route behind the admin key, and it is behind that key rather than a
tenant's own for the reason an earlier change decision A gives: a directory is
operator-configured state like the routing policy, and it decides what the
gateway redacts out of that tenant's traffic. A tenant able to rewrite its own
would be able to switch the control off; a tenant able to rewrite another's would
be able to name that tenant's staff. So the key space is `admin_key_sha256`, the
dependency is `admin.admin_key` imported rather than reimplemented, and the
tenant is a field of the BODY - an operator seeds a tenant it is not
authenticating as (`test_the_directory_route_rejects_a_tenant_api_key`,
`test_a_push_names_the_tenant_from_the_body`).

A push MERGES (decision C) and there is no delete in this plan.

Two client-supplied things reach a store here and both are pre-checked in the
route, before `put` is called at all.

* `tenant` shapes a Redis key, so it goes through the resolver's own
  `IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH` - `fullmatch`, never `match`,
  or a trailing newline would put a caller-controlled line break in a key
  (`test_a_push_for_a_malformed_tenant_is_422_before_any_store_write`).
* `names` goes through `checked_entries` against what the tenant already holds,
  which is the same function the store applies. Checking here as well is not
  duplication: `RecordingDirectoryStore` in the endpoint's tests is the shape of
  every wrapper, and "the refusal happened before anything was written" is only
  observable if the route never called `put`
  (`test_a_push_outside_a_cap_is_422_before_any_store_write`).

Nothing here logs or echoes a name. The response is a count, the refusal names
the field and the cap, and both are asserted for the absence of the value
(`test_the_response_carries_no_name_back`,
`test_a_pushed_name_never_reaches_a_log_line`). That is the criterion this route
exists under: a directory that leaks through an admin endpoint's error body is
still a directory in someone's proxy log.

Like `/admin/routing`, this path is not counted by the request metrics
(`COUNTED_PATH_PREFIX` is `/v1/`), is traced because `TracingMiddleware` spans
every scope, and is cluster-internal: the `gateway` Ingress routes `/v1` and
nothing wider, so no public router matches `/admin/directory`
(`tests/infra/test_k8s_apps.py::test_no_public_router_matches_an_admin_path_or_the_scrape`,
which imports `ADMIN_DIRECTORY` from this module). The admin key above is
unchanged and is still the control an in-cluster caller meets. Row 21a's
NetworkPolicies are a separate layer again: they decide which pods may open a
connection to the gateway, and Traefik is one of them, so what closed the path
was the router rule and not the policy.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from app.core.errors import ProviderError
from app.core.validation import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH
from app.gateway.routes.admin import admin_key
from app.model_routing.errors import IdentifierInvalid
from app.redaction import (
    DirectoryNameInvalid,
    DirectoryStore,
    DirectoryStoreUnavailable,
    checked_entries,
)

ADMIN_DIRECTORY = "/admin/directory"
"""The published route path. Pinned against the mounted route by
`test_the_directory_path_is_the_published_one`."""

TENANT_FIELD = "tenant"
"""The body field the directory is filed under, and the `param` every rejection of
it names (`test_a_push_for_a_malformed_tenant_is_422_before_any_store_write`)."""


class DirectoryPush(BaseModel):
    """One push: the tenant to file under, and the names to merge into its directory.

    Frozen and `extra="forbid"` like every other model on this API boundary. A
    caller sending `replace: true` would otherwise get a 200 for a request whose
    meaning the gateway silently dropped, and a misspelled `names` would answer
    200 having stored nothing
    (`test_a_body_of_the_wrong_shape_is_422_before_any_store_write`).

    Neither field carries a pydantic length constraint. The caps belong to the
    directory and their message has to name the field and the cap without the
    value; a `max_length` on the list item would render pydantic's own error,
    which reports the offending input and files it under the item's index rather
    than under `names`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str
    names: tuple[str, ...]


class DirectoryPushResult(BaseModel):
    """What a push did, said without saying any of it.

    `count` is the number of names the tenant's directory holds AFTER the merge,
    which is what an operator paging a large directory needs in order to know it
    is not silently at the cap. It is deliberately not the names themselves: an
    echo is how a directory ends up in a proxy log or a CI transcript that nobody
    classified as personal data (`test_the_response_carries_no_name_back`).
    """

    model_config = ConfigDict(frozen=True)

    stored: bool
    count: int


def checked_tenant(value: str) -> str:
    """Validate the tenant a push files under, before it reaches the store.

    The cap and the pattern are the resolver's own rather than numbers copied
    here, exactly as `/admin/routing`'s filter does it, so the name a directory
    accepts and the name the request path authenticates cannot disagree. The
    inclusive boundary is `test_a_tenant_at_the_identifier_cap_is_accepted`.
    """
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"{TENANT_FIELD}: longer than {MAX_IDENTIFIER_LENGTH} characters",
            param=TENANT_FIELD,
        )
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{TENANT_FIELD}: not a lowercase slug", param=TENANT_FIELD)
    return value


def held_names(store: DirectoryStore, tenant: str) -> tuple[str, ...]:
    """This tenant's current directory, with an unreachable store rendered as a 502.

    `DirectoryStoreUnavailable` is a plain exception, not a `GatewayError`
    (`ErrorCode.PROVIDER_ERROR` already belongs to `ProviderError` and
    `test_error_code_set_is_exhaustive` allows one concrete class per code), so
    letting it escape would answer a 500 with no code. The directory store is an
    outbound dependency like a provider and like the map store, which is why it
    lands on `PROVIDER_ERROR` and why the message is this route's own words: the
    client's name an internal host and port.
    """
    try:
        return store.get(tenant)
    except DirectoryStoreUnavailable as exc:
        raise ProviderError("the directory store is unavailable", retryable=True) from exc


router = APIRouter(dependencies=[Depends(admin_key)])


@router.post(ADMIN_DIRECTORY)
def push_directory(request: Request, body: DirectoryPush) -> DirectoryPushResult:
    """Merge the pushed names into that tenant's directory and answer how many it holds.

    The order is the safety property. The tenant is validated, the tenant's
    current directory is read, the whole push is validated against it, and only
    then is anything written - so a refusal leaves the directory exactly as it
    was and the store was never asked to write
    (`test_a_push_outside_a_cap_is_422_before_any_store_write`).

    `DirectoryNameInvalid` becomes a 422 `VALIDATION` carrying its own message,
    which names the field and the cap and never the value. It is deliberately not
    allowed to escape as a 500: the push is this endpoint's input, so a cap
    breach is a request-level validation failure.

    Declared `def` rather than `async def` like every other route here: hashing
    up to 2000 names is CPU work, and FastAPI runs a `def` route in a worker
    thread rather than on the event loop.
    """
    store: DirectoryStore = request.app.state.directory_store
    tenant = checked_tenant(body.tenant)
    held = held_names(store, tenant)

    try:
        checked_entries(body.names, held)
    except DirectoryNameInvalid as exc:
        raise IdentifierInvalid(str(exc), param=exc.param) from exc

    if body.names:
        try:
            store.put(tenant, body.names)
        except DirectoryStoreUnavailable as exc:
            raise ProviderError("the directory store is unavailable", retryable=True) from exc

    return DirectoryPushResult(stored=True, count=len(held_names(store, tenant)))
