"""Resolving (tenant, alias, data class) to an ordered candidate list.

Residency is deny by default. Every unhappy path here asserts the exact typed
code and HTTP status, because an earlier change maps those values into
the wire body and a drift would be invisible until then.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from opsconsole.routing.errors import (
    AliasUnknown,
    AllProvidersUnavailable,
    IdentifierInvalid,
    ResidencyUnavailable,
)
from opsconsole.routing.policy import DataClass, Policy, Residency, load_policy
from opsconsole.routing.resolver import MAX_IDENTIFIER_LENGTH, Resolved, resolve

from .conftest import SHIPPED_POLICY_PATH


def as_tuples(resolved: tuple[Resolved, ...]) -> list[tuple[str, str, str]]:
    return [(r.provider, r.model, r.region) for r in resolved]


def test_override_beats_alias_default(policy: Policy) -> None:
    """`acme` overrides `chat`, so the alias default never appears."""
    result = resolve(policy, "acme", "chat", "public")

    assert as_tuples(result) == [("eu-b", "m-override", "eu")]


def test_disabled_provider_filtered(policy: Policy) -> None:
    result = resolve(policy, "acme", "disabled-first", "public")

    assert as_tuples(result) == [("eu-a", "m-eu-a", "eu")]


def test_order_preserved(policy: Policy) -> None:
    """Three candidates, returned in the policy's order, not sorted."""
    result = resolve(policy, "acme", "ordered", "public")

    assert as_tuples(result) == [
        ("eu-a", "m-eu-a", "eu"),
        ("eu-b", "m-eu-b", "eu"),
        ("self-a", "m-self-a", "self"),
    ]


def test_eu_residency_clamps_a_data_class_that_allows_us(policy: Policy) -> None:
    """Decision B: residency is an outer clamp applied after the data class.

    `acme` is an eu tenant whose `public` rule allows `us`. The us candidate is
    first in the alias, so a resolver that only applied the data class rule
    would return it at the head of the list.
    """
    result = resolve(policy, "acme", "mixed", "public")

    assert as_tuples(result) == [("eu-a", "m-eu-a", "eu")]


def test_non_eu_tenant_is_not_clamped(policy: Policy) -> None:
    """The contrast case: without `residency: eu` the us candidate survives."""
    result = resolve(policy, "globex", "mixed", "public")

    assert as_tuples(result) == [("us-a", "m-us-a", "us"), ("eu-a", "m-eu-a", "eu")]


# Every route to an empty candidate list, including the missing-rule row where
# the tenant has no entry for the data class at all.
EMPTY_CASES: list[tuple[str, str, str, str]] = [
    ("data class excludes the only region", "acme", "us-only", "internal"),
    ("eu clamp empties an us-only allowance", "narrow", "us-only", "public"),
    ("tenant has no rule for the data class", "acme", "ordered", "pii"),
    ("tenant has no rule and no candidates survive", "globex", "ordered", "confidential"),
]


@pytest.mark.parametrize(
    ("case", "tenant", "alias", "data_class"), EMPTY_CASES, ids=[c[0] for c in EMPTY_CASES]
)
def test_empty_candidates_raises_residency_unavailable(
    case: str, tenant: str, alias: str, data_class: str, policy: Policy
) -> None:
    with pytest.raises(ResidencyUnavailable) as excinfo:
        resolve(policy, tenant, alias, data_class)

    assert excinfo.value.code == "RESIDENCY_UNAVAILABLE"
    assert excinfo.value.http_status == 451


def test_unknown_tenant_raises_residency_unavailable(policy: Policy) -> None:
    """Deny by default: an unknown tenant has no rules, so nothing is allowed."""
    with pytest.raises(ResidencyUnavailable) as excinfo:
        resolve(policy, "not-a-tenant", "ordered", "public")

    assert excinfo.value.code == "RESIDENCY_UNAVAILABLE"
    assert excinfo.value.http_status == 451


def test_unknown_alias_raises(policy: Policy) -> None:
    with pytest.raises(AliasUnknown) as excinfo:
        resolve(policy, "acme", "not-an-alias", "public")

    assert excinfo.value.code == "ALIAS_UNKNOWN"
    assert excinfo.value.http_status == 404


def always_unhealthy(provider: str) -> bool:
    return False


def only_eu_b_healthy(provider: str) -> bool:
    return provider == "eu-b"


def always_healthy(provider: str) -> bool:
    return True


def test_is_healthy_filtering_everything_raises_all_providers_unavailable(
    policy: Policy,
) -> None:
    """A health outage is 503, never 451: the region set was fine."""
    with pytest.raises(AllProvidersUnavailable) as excinfo:
        resolve(policy, "acme", "ordered", "public", is_healthy=always_unhealthy)

    assert excinfo.value.code == "ALL_PROVIDERS_FAILED"
    assert excinfo.value.http_status == 503


def test_is_healthy_filters_a_subset_and_keeps_order(policy: Policy) -> None:
    result = resolve(policy, "acme", "ordered", "public", is_healthy=only_eu_b_healthy)

    assert as_tuples(result) == [("eu-b", "m-eu-b", "eu")]


def test_default_is_healthy_keeps_every_candidate(policy: Policy) -> None:
    """The seam an earlier change fills defaults to always-true."""
    healthy: Callable[[str], bool] = always_healthy

    assert resolve(policy, "acme", "ordered", "public") == resolve(
        policy, "acme", "ordered", "public", is_healthy=healthy
    )


def test_max_identifier_length_is_pinned() -> None:
    """The cap is a bound the pre-check enforces, so it is pinned exactly."""
    assert MAX_IDENTIFIER_LENGTH == 64


# Rejection table. The trailing-newline row is the one a
# `^...$` pattern with `.match` would wrongly accept.
BAD_IDENTIFIERS: list[tuple[str, str]] = [
    ("trailing newline", "acme\n"),
    ("trailing crlf", "acme\r\n"),
    ("trailing space", "acme "),
    ("leading space", " acme"),
    ("uppercase", "ACME"),
    ("empty string", ""),
    ("one over the maximum length", "a" * (MAX_IDENTIFIER_LENGTH + 1)),
    ("path separator", "acme/../root"),
    ("null byte", "acme\x00"),
]

FIELDS: list[str] = ["tenant", "alias"]


@pytest.mark.parametrize("field", FIELDS, ids=FIELDS)
@pytest.mark.parametrize(("case", "value"), BAD_IDENTIFIERS, ids=[c[0] for c in BAD_IDENTIFIERS])
def test_identifier_rejection_table(case: str, value: str, field: str, policy: Policy) -> None:
    tenant = value if field == "tenant" else "acme"
    alias = value if field == "alias" else "ordered"

    with pytest.raises(IdentifierInvalid) as excinfo:
        resolve(policy, tenant, alias, "public")

    assert excinfo.value.code == "VALIDATION"
    assert excinfo.value.http_status == 422
    # The message names which argument was rejected, so a caller can fix it.
    assert field in str(excinfo.value)


def test_identifier_at_the_maximum_length_is_accepted(boundary_policy: Policy) -> None:
    """The inclusive boundary: exactly `MAX_IDENTIFIER_LENGTH` resolves."""
    name = "a" * MAX_IDENTIFIER_LENGTH

    result = resolve(boundary_policy, name, name, "public")

    assert as_tuples(result) == [("eu-a", "m-eu-a", "eu")]


BAD_DATA_CLASSES: list[tuple[str, str]] = [
    ("not a member", "secret"),
    ("uppercase member", "PUBLIC"),
    ("trailing newline", "public\n"),
    ("trailing space", "public "),
    ("empty string", ""),
]


@pytest.mark.parametrize(
    ("case", "data_class"), BAD_DATA_CLASSES, ids=[c[0] for c in BAD_DATA_CLASSES]
)
def test_unknown_data_class_raises(case: str, data_class: str, policy: Policy) -> None:
    with pytest.raises(IdentifierInvalid) as excinfo:
        resolve(policy, "acme", "ordered", data_class)

    assert excinfo.value.code == "VALIDATION"
    assert "data_class" in str(excinfo.value)


def test_identifier_check_runs_before_any_lookup(policy: Policy) -> None:
    """A bad identifier is `IdentifierInvalid`, not `AliasUnknown`.

    Order matters: the pre-check must reject before the string is used as a key.
    """
    with pytest.raises(IdentifierInvalid):
        resolve(policy, "acme", "no such alias", "public")


def allowed_regions(policy: Policy, tenant_name: str, data_class: DataClass) -> set[str]:
    tenant = policy.tenants[tenant_name]
    rule = tenant.data_class_rules.get(data_class)
    allowed = set(rule.allow_regions) if rule is not None else set()
    if tenant.residency is Residency.EU:
        allowed &= set(policy.eu_regions)
    return allowed


def test_shipped_policy_exhaustive() -> None:
    """Every triple in the shipped file resolves inside the allowed set or 451.

    There is no third outcome: no widening, no empty success, no candidate in a
    region the tenant's rules (clamped by residency) do not permit.
    """
    policy = load_policy(SHIPPED_POLICY_PATH)
    checked = 0

    for tenant_name in policy.tenants:
        for alias_name in policy.aliases:
            for data_class in DataClass:
                checked += 1
                allowed = allowed_regions(policy, tenant_name, data_class)
                try:
                    result = resolve(policy, tenant_name, alias_name, data_class.value)
                except ResidencyUnavailable:
                    continue
                assert result, (tenant_name, alias_name, data_class)
                for candidate in result:
                    assert candidate.region in allowed, (
                        tenant_name,
                        alias_name,
                        data_class,
                        candidate,
                    )
                    assert policy.providers[candidate.provider].enabled

    assert checked == len(policy.tenants) * len(policy.aliases) * len(DataClass)
    assert checked > 0
