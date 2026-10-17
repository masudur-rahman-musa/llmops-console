"""Tests for `Tenant.allowed_regions`, the single place both the resolver and
the eval check read a tenant's data class rule.

All three tests use plain in-memory Pydantic objects. No running service is
needed, so there is no `db` marker.
"""

from __future__ import annotations

import pytest

from app.evals.region_check import ResidencyVerdict, check_residency
from app.model_routing.errors import ResidencyUnavailable
from app.model_routing.policy import (
    Alias,
    Candidate,
    DataClass,
    Provider,
    ProviderKind,
    Residency,
)
from app.model_routing.resolver import resolve

from .conftest import make_policy, make_tenant


def test_allowed_regions_returns_the_rules_regions() -> None:
    """Catches an implementation that returns the wrong regions or the wrong type
    for a data class the tenant has an explicit rule for."""
    tenant = make_tenant(
        residency=Residency.EU,
        rules={DataClass.INTERNAL: ("eu", "self")},
    )
    result = tenant.allowed_regions(DataClass.INTERNAL)
    assert result == ("eu", "self")


def test_allowed_regions_denies_a_class_with_no_rule() -> None:
    """Catches an implementation that widens or falls back to a non-empty set
    when the tenant has no rule for the requested data class."""
    tenant = make_tenant(
        residency=Residency.EU,
        rules={DataClass.PUBLIC: ("us", "eu", "self")},
    )
    result = tenant.allowed_regions(DataClass.INTERNAL)
    assert result == ()


def test_the_gateway_and_the_eval_check_read_the_same_rule() -> None:
    """Catches an implementation that adds `allowed_regions` but leaves the
    resolver or the eval check using its own separate inline lookup, so that
    one caller could return a different allowed set than the method itself."""
    eu_provider = Provider(kind=ProviderKind.ANTHROPIC, region="eu")
    tenant_obj = make_tenant(
        residency=Residency.EU,
        rules={DataClass.PUBLIC: ("eu", "self")},
        # No INTERNAL rule: every caller must agree on () for this class.
    )
    policy = make_policy(
        providers={"eu-a": eu_provider},
        eu_regions=("eu", "self"),
        aliases={"chat": Alias(candidates=(Candidate(provider="eu-a", model="m-eu"),))},
        tenants={"acme-eu": tenant_obj},
    )

    # The method itself must say () — no rule means no allowed regions.
    assert tenant_obj.allowed_regions(DataClass.INTERNAL) == ()

    # The resolver must agree: no candidates survive, so it raises.
    with pytest.raises(ResidencyUnavailable):
        resolve(policy, "acme-eu", "chat", "internal")

    # The eval check must agree: violation=True and allowed=().
    verdict: ResidencyVerdict = check_residency(
        policy,
        case_id="c1",
        tenant="acme-eu",
        data_class=DataClass.INTERNAL,
        observed="eu",
    )
    assert verdict.violation is True
    assert verdict.allowed == ()
