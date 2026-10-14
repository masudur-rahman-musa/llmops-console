"""`GET /v1/models`: the aliases a caller may actually use, and nothing else.

Decision D: the list is the aliases `resolve()` does not raise for, under the
caller's own tenant and the request's data class. Expectations are computed by
calling `resolve()` directly, and every case also asserts one membership
literally, so a route that returned an empty list could not pass.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.model_routing.errors import AllProvidersUnavailable
from app.model_routing.policy import DataClass, Policy
from app.model_routing.resolver import Resolved

from .conftest import TEST_KEYS, assert_error_body, auth_headers, resolvable_aliases

NORDLICHT_KEY = TEST_KEYS["nordlicht"]
CUAN_KEY = TEST_KEYS["cuan"]
EVALS_KEY = TEST_KEYS["evals"]
MODELS = "/v1/models"


def listed_ids(client: TestClient, *, key: str, tenant: str, data_class: str | None) -> list[str]:
    """Call the endpoint and return the alias ids, asserting the envelope shape."""
    response = client.get(
        MODELS, headers=auth_headers(key=key, tenant=tenant, data_class=data_class)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"object", "data"}
    assert body["object"] == "list"
    for entry in body["data"]:
        assert set(entry) == {"id", "object", "owned_by"}
        assert entry["object"] == "model"
        assert entry["owned_by"] == "opsconsole"
    ids: list[str] = [entry["id"] for entry in body["data"]]
    return ids


LISTING_CASES: list[tuple[str, str, str, DataClass]] = [
    ("nordlicht, default class", NORDLICHT_KEY, "nordlicht", DataClass.CONFIDENTIAL),
    ("nordlicht, public", NORDLICHT_KEY, "nordlicht", DataClass.PUBLIC),
    ("nordlicht, internal", NORDLICHT_KEY, "nordlicht", DataClass.INTERNAL),
    ("nordlicht, pii", NORDLICHT_KEY, "nordlicht", DataClass.PII),
    ("cuan, internal", CUAN_KEY, "cuan", DataClass.INTERNAL),
    ("cuan, pii", CUAN_KEY, "cuan", DataClass.PII),
    ("evals, internal", EVALS_KEY, "evals", DataClass.INTERNAL),
    ("evals, confidential", EVALS_KEY, "evals", DataClass.CONFIDENTIAL),
    ("evals, pii", EVALS_KEY, "evals", DataClass.PII),
]


@pytest.mark.parametrize(
    ("case", "key", "tenant", "data_class"), LISTING_CASES, ids=[c[0] for c in LISTING_CASES]
)
def test_models_lists_only_resolvable_aliases(
    case: str,
    key: str,
    tenant: str,
    data_class: DataClass,
    gateway_client: TestClient,
    policy: Policy,
) -> None:
    """Every listed alias resolves, and every resolvable alias is listed.

    The literal anchor: `embed` runs on the box we host, so it resolves for every
    tenant under every class in the shipped policy and must always be listed.
    """
    expected = resolvable_aliases(policy, tenant, data_class)

    ids = listed_ids(gateway_client, key=key, tenant=tenant, data_class=data_class.value)

    assert ids == expected
    assert "embed" in ids
    assert set(ids) <= set(policy.aliases)


def test_models_omits_an_alias_that_would_be_451(gateway_client: TestClient) -> None:
    """`agent-strong` is US-only while `anthropic_eu` is disabled.

    An EU tenant gets 451 for it, so listing it would advertise an alias that
    fails on first use. It is absent for both EU tenants and present for `evals`,
    whose residency is `any`.
    """
    nordlicht = listed_ids(
        gateway_client, key=NORDLICHT_KEY, tenant="nordlicht", data_class="internal"
    )
    cuan = listed_ids(gateway_client, key=CUAN_KEY, tenant="cuan", data_class="internal")
    evals = listed_ids(gateway_client, key=EVALS_KEY, tenant="evals", data_class="internal")

    assert "agent-strong" not in nordlicht
    assert "agent-strong" not in cuan
    assert "agent-strong" in evals


def test_models_differs_per_tenant(gateway_client: TestClient, policy: Policy) -> None:
    """Two tenants with different rules get different lists.

    Literal anchor, so the test is not a tautology over `resolve()`: under
    `internal`, `judge` resolves for `evals` (US providers are allowed) and does
    not for `nordlicht` (an EU tenant, and both judge candidates sit in `us`).
    """
    nordlicht = listed_ids(
        gateway_client, key=NORDLICHT_KEY, tenant="nordlicht", data_class="internal"
    )
    evals = listed_ids(gateway_client, key=EVALS_KEY, tenant="evals", data_class="internal")

    assert "judge" in evals
    assert "judge" not in nordlicht
    assert nordlicht != evals
    assert set(nordlicht) < set(evals)
    assert nordlicht == resolvable_aliases(policy, "nordlicht", DataClass.INTERNAL)
    assert evals == resolvable_aliases(policy, "evals", DataClass.INTERNAL)


def test_models_reflects_the_requested_data_class(gateway_client: TestClient) -> None:
    """The list narrows as the class gets more sensitive, for the same tenant.

    `evals` may reach US providers for `internal` but not for `confidential`,
    where its rule allows only `eu` and `self`.
    """
    internal = listed_ids(gateway_client, key=EVALS_KEY, tenant="evals", data_class="internal")
    confidential = listed_ids(
        gateway_client, key=EVALS_KEY, tenant="evals", data_class="confidential"
    )

    assert "judge" in internal
    assert "judge" not in confidential
    assert set(confidential) < set(internal)


def test_models_ignores_another_tenants_rules(gateway_client: TestClient) -> None:
    """The list is the caller's own, never a probe into a tenant it names.

    `nordlicht` cannot ask about `evals` even with its own valid key: the header
    must match the key, so the answer is 403 rather than `evals`' list.
    """
    response = gateway_client.get(MODELS, headers=auth_headers(key=NORDLICHT_KEY, tenant="evals"))

    assert_error_body(response, status=403, code="TENANT_MISMATCH")


def test_models_propagates_a_health_failure(
    gateway_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outage is a 503, not an alias quietly missing from the catalogue.

    The route swallows only the two errors that mean "this alias is not for you".
    `AllProvidersUnavailable` now means a provider incident - every candidate's
    adapter missing, or every candidate's breaker open - and a catalogue that
    shrank during one would read to an operator as an alias that had been
    deleted. Widening the catch back to `RoutingError`
    fails here.
    """

    def unavailable(*args: object, **kwargs: object) -> tuple[Resolved, ...]:
        raise AllProvidersUnavailable("every provider is unhealthy")

    monkeypatch.setattr("app.gateway.routes.models.resolve", unavailable)

    response = gateway_client.get(MODELS, headers=auth_headers(key=EVALS_KEY, tenant="evals"))

    assert_error_body(response, status=503, code="ALL_PROVIDERS_FAILED")


def test_models_requires_auth(gateway_client: TestClient) -> None:
    """An unauthenticated call is 401 and leaks no alias name."""
    response = gateway_client.get(MODELS)

    assert_error_body(response, status=401, code="UNAUTHORIZED")
    assert "agent-default" not in response.text


def test_models_for_a_tenant_the_policy_does_not_define_is_empty(
    gateway_client: TestClient,
) -> None:
    """Deny by default: an authenticated tenant with no policy rules lists nothing.

    `cap-key` holds a key but no entry in `routing.yaml`, so every alias raises
    `ResidencyUnavailable` and the list is empty rather than the alias defaults.
    """
    ids = listed_ids(gateway_client, key=TEST_KEYS["cap-key"], tenant="cap-key", data_class=None)

    assert ids == []
