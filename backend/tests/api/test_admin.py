"""`/admin/routing` and `/admin/routing/reload`: the second key space, the view and the swap.

Three claims are pinned here, and each case says which wrong implementation it
would catch rather than merely which behaviour it likes.

* The admin key space is separate from the tenant one. A tenant key that works
  on `/v1/models` is a 401 on both admin routes, and an unset admin key
  authorises nobody, including a caller presenting nothing at all.
* The view is the resolver's own answer, computed here by calling `resolve()`
  the way `resolvable_aliases` does rather than by copying `routing.yaml`, so a
  policy edit moves the expectation and the test with it. Every case that
  compares a computed structure also asserts one literal membership, so an empty
  view cannot pass.
* A reload swaps the snapshot the next `/v1` request routes under AND the
  adapter set built from it; a reload that does not validate keeps the previous
  policy serving.

The tenant-filter rejection rows are all strings the policy does not name as a
tenant, which is what makes them a test of the pre-check rather than of a
neighbouring guard: without the pre-check the view would simply find no matching
tenant and answer 200 with an empty list, never reaching `resolve()` and so never
raising `IdentifierInvalid` by accident.

The reload cases own a `tmp_path` app each: rewriting the
policy file is the thing under test, so they cannot share the session-scoped
`admin_client`. Everything else uses it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.settings import Settings
from app.gateway.app import create_app
from app.gateway.routes import admin
from app.providers.base import Adapter
from app.providers.breaker import MAX_FAILURES, CircuitState
from app.routing.errors import AliasUnknown, ResidencyUnavailable
from app.routing.policy import DataClass, Policy
from app.routing.resolver import MAX_IDENTIFIER_LENGTH, resolve

from .conftest import (
    ADMIN_KEY,
    ADMIN_ROUTING,
    ADMIN_ROUTING_RELOAD,
    API_KEYS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    assert_error_body,
    auth_headers,
    sha256_hex,
)

MODELS = "/v1/models"
NORDLICHT_KEY = TEST_KEYS["nordlicht"]

# A provider key that is not one: the adapter it builds is never called, and no
# request in this file leaves the process.
FAKE_ANTHROPIC_KEY = "test-key-not-a-real-one"
WITH_ANTHROPIC_KEY = {"ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY}

ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}


def admin_get(client: TestClient, **params: str) -> httpx.Response:
    """`GET /admin/routing` with the admin key, and whatever query the case sends."""
    response: httpx.Response = client.get(ADMIN_ROUTING, headers=ADMIN_HEADERS, params=params)
    return response


def expected_view(policy: Policy) -> list[dict[str, Any]]:
    """The view computed from `resolve()` itself: tenants x aliases x data classes.

    In policy order for the first two and enum order for the third, swallowing
    only the two errors that mean "not available to this tenant" - the same pair
    `/v1/models` swallows, and deliberately not the `RoutingError` base.
    """
    tenants: list[dict[str, Any]] = []
    for tenant in policy.tenants:
        entries: list[dict[str, Any]] = []
        for alias in policy.aliases:
            for data_class in DataClass:
                try:
                    resolved = resolve(policy, tenant, alias, data_class.value)
                except (ResidencyUnavailable, AliasUnknown):
                    continue
                entries.append(
                    {
                        "alias": alias,
                        "data_class": data_class.value,
                        "resolved": [
                            {"provider": item.provider, "model": item.model, "region": item.region}
                            for item in resolved
                        ],
                    }
                )
        tenants.append({"tenant": tenant, "entries": entries})
    return tenants


def regions_in(tenant_view: dict[str, Any]) -> set[str]:
    """Every region named anywhere under one tenant's entries."""
    return {
        entry_region["region"]
        for entry in tenant_view["entries"]
        for entry_region in entry["resolved"]
    }


def tenant_slice(body: dict[str, Any], name: str) -> dict[str, Any]:
    """The one tenant view named `name`, asserting there is exactly one."""
    found = [view for view in body["tenants"] if view["tenant"] == name]
    assert len(found) == 1, body
    view: dict[str, Any] = found[0]
    return view


# --- the tmp_path app the reload cases own -----------------------------------


@dataclass(frozen=True)
class AdminApp:
    """One app built over a policy file the case may rewrite."""

    client: TestClient
    app: FastAPI
    policy_path: Path


@contextmanager
def admin_app_on(tmp_path: Path, environ: dict[str, str] | None = None) -> Iterator[AdminApp]:
    """An app over a private copy of the shipped `routing.yaml`, closed at exit.

    The one documented exception to a project rule: these cases rewrite the file
    the app was built from, so they cannot share a session-scoped client. It costs
    a YAML parse and no backing service.
    """
    policy_path = tmp_path / "routing.yaml"
    policy_path.write_text(SHIPPED_POLICY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=policy_path,
        admin_key_sha256=sha256_hex(ADMIN_KEY),
    )
    app = create_app(settings, NO_PROVIDER_KEYS if environ is None else environ)
    with TestClient(app) as client:
        yield AdminApp(client=client, app=app, policy_path=policy_path)


def rewrite_policy(path: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    """Parse the policy file, hand the document to `edit`, and write it back."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def listed_aliases(client: TestClient) -> list[str]:
    """The aliases `/v1/models` lists for `nordlicht` under `internal`."""
    response = client.get(
        MODELS, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht", data_class="internal")
    )
    assert response.status_code == 200, response.text
    return [entry["id"] for entry in response.json()["data"]]


def http_client_of(adapter: Adapter) -> httpx.Client:
    """The `httpx.Client` an adapter's transport holds, for reading whether it was closed.

    `close` is not on the `Adapter` protocol and neither is the transport, so both
    hops are read with `getattr` and asserted: that is what makes "the pool this
    adapter owns is open" an observation rather than a hope.
    """
    transport = getattr(adapter, "http", None)
    held = getattr(transport, "client", None)
    assert isinstance(held, httpx.Client), transport
    return held


# --- the published surface ---------------------------------------------------


def test_the_admin_paths_and_statuses_are_pinned(admin_client: TestClient) -> None:
    """Catches an implementation that mounts the routes under some other path, or
    renders a refused caller as 403/404 and a rejected filter as 400/500 rather
    than the exact 401 UNAUTHORIZED and 422 VALIDATION the contract fixes.

    a project rule: both paths and both statuses are values this plan
    introduces, so they are asserted exactly rather than by shape.
    """
    assert admin.ADMIN_ROUTING == ADMIN_ROUTING == "/admin/routing"
    assert admin.ADMIN_ROUTING_RELOAD == ADMIN_ROUTING_RELOAD == "/admin/routing/reload"

    # Both paths exist on the built app and answer these two exact statuses.
    assert admin_client.get(ADMIN_ROUTING, headers=ADMIN_HEADERS).status_code == 200
    assert admin_client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS).status_code == 200

    assert_error_body(admin_client.get(ADMIN_ROUTING), status=401, code="UNAUTHORIZED")
    assert_error_body(admin_client.post(ADMIN_ROUTING_RELOAD), status=401, code="UNAUTHORIZED")
    assert_error_body(admin_get(admin_client, tenant="Nordlicht"), status=422, code="VALIDATION")


# --- the admin key space -----------------------------------------------------

TENANT_KEY_CASES: list[tuple[str, str]] = [
    ("view", ADMIN_ROUTING),
    ("reload", ADMIN_ROUTING_RELOAD),
]


@pytest.mark.parametrize(
    ("case", "path"), TENANT_KEY_CASES, ids=[row[0] for row in TENANT_KEY_CASES]
)
def test_admin_routes_reject_a_tenant_api_key(
    case: str, path: str, admin_client: TestClient
) -> None:
    """Catches an implementation that authenticates `/admin/*` against the tenant
    key store (or against any key store at all) instead of comparing the digest
    with `admin_key_sha256`: the key sent here is a WORKING tenant key on the same
    app, so a 200 here means the two key spaces were merged.
    """
    response = admin_client.request(
        "POST" if path == ADMIN_ROUTING_RELOAD else "GET",
        path,
        headers={"Authorization": f"Bearer {NORDLICHT_KEY}"},
    )
    assert_error_body(response, status=401, code="UNAUTHORIZED")

    # The same key on the tenant route still works, so the 401 above is the admin
    # comparison refusing it and not a broken key or a broken app.
    assert (
        admin_client.get(
            MODELS, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
        ).status_code
        == 200
    )


NO_ADMIN_KEY_CASES: list[tuple[str, dict[str, str]]] = [
    ("the key a configured app would accept", {"Authorization": f"Bearer {ADMIN_KEY}"}),
    ("no Authorization header at all", {}),
    ("an empty Authorization header", {"Authorization": ""}),
    ("a Bearer scheme with an empty value", {"Authorization": "Bearer "}),
    ("an empty key hashed to the empty setting", {"Authorization": "Bearer "}),
]


@pytest.mark.parametrize(
    ("case", "headers"), NO_ADMIN_KEY_CASES, ids=[row[0] for row in NO_ADMIN_KEY_CASES]
)
@pytest.mark.parametrize("path", [ADMIN_ROUTING, ADMIN_ROUTING_RELOAD])
def test_admin_routes_reject_every_caller_when_no_admin_key_is_configured(
    case: str, headers: dict[str, str], path: str, no_admin_key_client: TestClient
) -> None:
    """Catches an implementation that reads an unset `admin_key_sha256` as "no
    admin key means no check" (`if not configured: return`) rather than as
    "authorises nobody" - the first row presents the very key the configured app
    accepts, so a permissive-when-unset dependency answers 200 to it.
    """
    response = no_admin_key_client.request(
        "POST" if path == ADMIN_ROUTING_RELOAD else "GET", path, headers=headers
    )
    assert_error_body(response, status=401, code="UNAUTHORIZED")


@pytest.mark.parametrize("path", [ADMIN_ROUTING, ADMIN_ROUTING_RELOAD])
def test_admin_key_never_appears_in_body_or_log(
    path: str, admin_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches an implementation that echoes the presented admin key in a response
    (a "key X is not authorised" message) or logs it while authenticating a
    SUCCESSFUL call, which is the case a 401-only leak test would never reach.

    DEBUG for the whole request, and both the rendered message and the raw `args`
    are checked: `%s` formatting hides a value from `record.message`.
    """
    with caplog.at_level(logging.DEBUG):
        response = admin_client.request(
            "POST" if path == ADMIN_ROUTING_RELOAD else "GET", path, headers=ADMIN_HEADERS
        )

    assert response.status_code == 200, response.text
    assert ADMIN_KEY not in response.text
    for record in caplog.records:
        assert ADMIN_KEY not in record.getMessage()
        assert ADMIN_KEY not in str(record.msg)
        assert ADMIN_KEY not in str(record.args)


# --- the view ----------------------------------------------------------------


def test_the_view_resolves_every_tenant_alias_and_data_class(
    admin_client: TestClient, policy: Policy
) -> None:
    """Catches an implementation that drops a dimension of the matrix - one entry
    per alias with no data class, or the tenant's default candidates instead of
    its override - and one that reports a US provider for an EU tenant.

    The expectation is computed by calling `resolve()`, so it cannot drift from
    the policy file; the literal membership below means an empty view fails, and
    the `evals` row means the "no `us`" assertion is not passing merely because
    the body carries no regions at all.
    """
    response = admin_get(admin_client)
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body) == {"tenants"}
    assert body["tenants"] == expected_view(policy)

    # Residency, read off the wire: neither EU tenant names a US region anywhere.
    assert "us" not in regions_in(tenant_slice(body, "nordlicht"))
    assert "us" not in regions_in(tenant_slice(body, "cuan"))
    # And the view does carry US regions where the policy allows them, so the two
    # assertions above are about residency and not about an empty body.
    assert "us" in regions_in(tenant_slice(body, "evals"))

    # One triple asserted literally: nordlicht's override, not the alias default.
    nordlicht = tenant_slice(body, "nordlicht")
    entry = [
        item
        for item in nordlicht["entries"]
        if item["alias"] == "agent-default" and item["data_class"] == "internal"
    ]
    assert entry == [
        {
            "alias": "agent-default",
            "data_class": "internal",
            "resolved": [{"provider": "local", "model": "qwen2.5-3b-instruct", "region": "self"}],
        }
    ]


def test_the_view_filters_to_one_tenant(admin_client: TestClient, policy: Policy) -> None:
    """Catches an implementation that accepts `?tenant=` and ignores it, returning
    the whole matrix - the unfiltered body has three tenants, so a filtered one
    with three entries fails here rather than being masked by a coincidence.
    """
    response = admin_get(admin_client, tenant="cuan")
    assert response.status_code == 200, response.text
    body = response.json()

    expected = [view for view in expected_view(policy) if view["tenant"] == "cuan"]
    assert body["tenants"] == expected
    assert [view["tenant"] for view in body["tenants"]] == ["cuan"]
    # The unfiltered view carries more than this one, so the filter did something.
    assert len(expected_view(policy)) > 1
    assert any(entry["alias"] == "agent-default" for entry in body["tenants"][0]["entries"])


def test_an_unknown_but_valid_tenant_is_an_empty_view(admin_client: TestClient) -> None:
    """Catches an implementation that treats an unknown tenant as an error (404, or
    a `KeyError` rendered as 500) instead of as an empty view: the name here is a
    well-formed identifier, so the pre-check has nothing to say about it and only
    the lookup can decide the answer.
    """
    response = admin_get(admin_client, tenant="tenant-the-policy-does-not-define")
    assert response.status_code == 200, response.text
    assert response.json() == {"tenants": []}


BOUNDARY_CASES: list[tuple[str, str, int]] = [
    ("exactly the cap", "a" * MAX_IDENTIFIER_LENGTH, 200),
    ("one character over the cap", "a" * (MAX_IDENTIFIER_LENGTH + 1), 422),
]


@pytest.mark.parametrize(
    ("case", "tenant", "status"), BOUNDARY_CASES, ids=[row[0] for row in BOUNDARY_CASES]
)
def test_the_tenant_filter_length_boundary_is_inclusive(
    case: str, tenant: str, status: int, admin_client: TestClient
) -> None:
    """Catches an off-by-one in the `?tenant=` length pre-check: `>=` in place of
    `>` rejects a name of exactly `MAX_IDENTIFIER_LENGTH`, and a missing length
    check answers 200 to the over-cap row (neither name is a policy tenant, so
    without the check both would simply be an empty view).

    a project rule: the cap is the store's own `MAX_IDENTIFIER_LENGTH`, not a
    number copied here.
    """
    response = admin_get(admin_client, tenant=tenant)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json() == {"tenants": []}
    else:
        assert_error_body(response, status=422, code="VALIDATION")


REJECTION_CASES: list[tuple[str, str]] = [
    ("trailing newline", "nordlicht\n"),
    ("trailing CRLF", "nordlicht\r\n"),
    ("trailing space", "nordlicht "),
    ("leading space", " nordlicht"),
    ("uppercase", "Nordlicht"),
    ("empty", ""),
    ("path traversal", "../nordlicht"),
]


@pytest.mark.parametrize(
    ("case", "tenant"), REJECTION_CASES, ids=[row[0] for row in REJECTION_CASES]
)
def test_the_tenant_filter_rejection_table(
    case: str, tenant: str, admin_client: TestClient
) -> None:
    """Catches a pre-check written as `re.match` with `^...$`, which accepts a
    trailing newline, and one that skips the check for a falsy value (`if tenant:`)
    - and, more generally, its absence: none of these strings names a policy
    tenant, so without the pre-check every row would answer 200 with an empty view
    or, for the empty row, the whole unfiltered matrix, never reaching `resolve()`
    and never raising `IdentifierInvalid` from the neighbouring guard inside it.

    a project rule: the table's first three rows are the newline, the CRLF and
    the space.
    """
    assert_error_body(admin_get(admin_client, tenant=tenant), status=422, code="VALIDATION")


# --- reload ------------------------------------------------------------------


def test_reload_swaps_the_policy_the_next_request_routes_under(tmp_path: Path) -> None:
    """Catches a reload that parses the file and drops the result - a fresh
    `PolicyStore` built into a local, or a `load_policy` whose value is never
    rebound - by reading the answer through `/v1/models`, which holds no policy of
    its own and can only reflect the snapshot on `app.state`.
    """
    with admin_app_on(tmp_path) as built:
        before = listed_aliases(built.client)
        assert "embed" in before

        rewrite_policy(built.policy_path, lambda document: document["aliases"].pop("embed"))

        response = built.client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS)
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body) == {"reloaded", "providers"}
        assert body["reloaded"] is True

        after = listed_aliases(built.client)
        assert "embed" not in after
        assert after == [alias for alias in before if alias != "embed"]


def test_an_invalid_policy_on_reload_is_422_naming_the_field(tmp_path: Path) -> None:
    """Catches a reload that lets `PolicyInvalid` escape as an untyped 500, and one
    that answers 422 with a generic message: the field path is the whole point of
    the answer, so the message is asserted to carry the one the policy loader
    produced.
    """
    with admin_app_on(tmp_path) as built:
        rewrite_policy(
            built.policy_path,
            lambda document: document["aliases"]["judge"]["candidates"][0].update(
                {"provider": "provider-that-does-not-exist"}
            ),
        )

        response = built.client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS)
        assert_error_body(response, status=422, code="VALIDATION")
        assert "aliases.judge.candidates[0].provider" in response.json()["error"]["message"]


def test_a_failed_reload_keeps_the_previous_policy_serving(tmp_path: Path) -> None:
    """Catches a reload that clears state before it validates - swapping in a new
    store, or emptying `app.state.adapters` on the way to rebuilding it - which
    would leave the gateway serving nothing after an operator's typo.
    """
    with admin_app_on(tmp_path, WITH_ANTHROPIC_KEY) as built:
        before = listed_aliases(built.client)
        adapters_before = dict(built.app.state.adapters)
        assert adapters_before, "the fixture must build at least one adapter to be a test"

        built.policy_path.write_text("aliases: not-a-mapping-of-aliases\n", encoding="utf-8")
        assert_error_body(
            built.client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS),
            status=422,
            code="VALIDATION",
        )

        assert listed_aliases(built.client) == before
        assert built.app.state.adapters == adapters_before
        assert not http_client_of(built.app.state.adapters["local"]).is_closed


def test_a_reload_rebuilds_the_adapter_set_for_the_new_policy(tmp_path: Path) -> None:
    """Catches a reload that swaps the policy snapshot and leaves `app.state.adapters`
    as it was: the dropped provider would keep an adapter the new policy does not
    name, and the resolver's health seam would go on reporting it as usable.
    """
    with admin_app_on(tmp_path, WITH_ANTHROPIC_KEY) as built:
        assert set(built.app.state.adapters) == {"anthropic", "local", "local_embed"}

        rewrite_policy(
            built.policy_path,
            lambda document: document["providers"]["anthropic"].update({"enabled": False}),
        )

        response = built.client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS)
        assert response.status_code == 200, response.text

        assert response.json()["providers"] == ["local", "local_embed"]
        assert set(built.app.state.adapters) == {"local", "local_embed"}
        assert "anthropic" not in built.app.state.adapters


def test_a_policy_reload_rebuilds_the_breaker(tmp_path: Path) -> None:
    """Catches a reload that rebuilds the adapter set and leaves the breaker behind.
    Both are keyed by the provider names of the policy that was just replaced, so a
    reload that renamed or repointed a provider would leave a breaker open against a
    name that no longer means what it meant - a provider the operator has just fixed
    by editing the file, refused for the next thirty seconds by state about the old
    one. The breaker is opened here first, so a kept breaker reads OPEN afterwards
    and only a rebuilt one reads CLOSED."""
    with admin_app_on(tmp_path, WITH_ANTHROPIC_KEY) as built:
        breaker = built.app.state.breaker
        for _ in range(MAX_FAILURES):
            breaker.record_failure("anthropic")
        assert breaker.state("anthropic") is CircuitState.OPEN

        response = built.client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS)
        assert response.status_code == 200, response.text

        rebuilt = built.app.state.breaker
        assert rebuilt is not breaker
        assert rebuilt.state("anthropic") is CircuitState.CLOSED


def test_a_reload_leaves_the_new_adapter_set_callable(tmp_path: Path) -> None:
    """Catches decision C applied to the wrong set: closing the adapters that
    SURVIVE the reload rather than the ones it dropped, which passes a
    set-comparison test and leaves every later request holding a closed connection
    pool.
    """
    with admin_app_on(tmp_path, WITH_ANTHROPIC_KEY) as built:
        dropped = built.app.state.adapters["anthropic"]

        rewrite_policy(
            built.policy_path,
            lambda document: document["providers"]["anthropic"].update({"enabled": False}),
        )
        assert built.client.post(ADMIN_ROUTING_RELOAD, headers=ADMIN_HEADERS).status_code == 200

        survivor = built.app.state.adapters["local"]
        assert not http_client_of(survivor).is_closed
        # The set the reload dropped is closed, which is the other half of C.
        assert http_client_of(dropped).is_closed
        # And the app still serves under the new snapshot.
        assert listed_aliases(built.client) == ["agent-default", "embed"]
