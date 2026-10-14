"""The residency property (REQUIREMENTS #65): the untouchable test.

For any policy, any tenant with `residency: eu` and any data class, every
candidate the resolver returns sits in an EU region. Raising
`ResidencyUnavailable` is the only other permitted outcome; widening the region
set is not. The strategy deliberately generates data class rules that allow
non-EU regions, so the outer clamp is what makes the property hold, and the
test's own counter proves the generator actually produced those cases rather
than passing vacuously.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from app.model_routing.errors import ResidencyUnavailable
from app.model_routing.policy import (
    Alias,
    Candidate,
    DataClass,
    DataClassRule,
    Policy,
    Provider,
    ProviderKind,
    RateLimit,
    Residency,
    Tenant,
)
from app.model_routing.resolver import resolve

# The pool the policy's `eu_regions` is drawn from, and the wider pool provider
# regions and `allow_regions` are drawn from. The two non-EU entries are what
# the clamp exists to remove.
EU_REGION_POOL = ("eu", "eu-central", "self")
REGION_POOL = ("eu", "eu-central", "self", "us", "ap")

TENANT_NAME = "eu-tenant"


@dataclass(frozen=True)
class Case:
    policy: Policy
    alias: str
    data_class: DataClass


@st.composite
def cases(draw: DrawFn) -> Case:
    eu_regions = tuple(
        draw(st.lists(st.sampled_from(EU_REGION_POOL), min_size=1, max_size=3, unique=True))
    )
    provider_count = draw(st.integers(min_value=1, max_value=5))
    providers = {
        f"p{i}": Provider(
            kind=ProviderKind.ANTHROPIC,
            region=draw(st.sampled_from(REGION_POOL)),
            enabled=draw(st.booleans()),
        )
        for i in range(provider_count)
    }
    provider_names = sorted(providers)

    alias_count = draw(st.integers(min_value=1, max_value=3))
    aliases = {
        f"a{j}": Alias(
            candidates=tuple(
                Candidate(provider=draw(st.sampled_from(provider_names)), model=f"m{k}")
                for k in range(draw(st.integers(min_value=1, max_value=4)))
            )
        )
        for j in range(alias_count)
    }
    alias_names = sorted(aliases)

    rules: dict[DataClass, DataClassRule] = {}
    for data_class in DataClass:
        if draw(st.booleans()):
            rules[data_class] = DataClassRule(
                allow_regions=tuple(
                    draw(
                        st.lists(
                            st.sampled_from(REGION_POOL),
                            min_size=1,
                            max_size=len(REGION_POOL),
                            unique=True,
                        )
                    )
                )
            )

    overrides: dict[str, tuple[Candidate, ...]] = {}
    if draw(st.booleans()):
        overridden = draw(st.sampled_from(alias_names))
        overrides[overridden] = tuple(
            Candidate(provider=draw(st.sampled_from(provider_names)), model=f"o{k}")
            for k in range(draw(st.integers(min_value=1, max_value=3)))
        )

    tenant = Tenant(
        residency=Residency.EU,
        data_class_rules=rules,
        budget_usd_month=Decimal("100.00"),
        rate_limit=RateLimit(rpm=60, tpm=100_000),
        overrides=overrides,
    )
    policy = Policy(
        providers=providers,
        eu_regions=eu_regions,
        aliases=aliases,
        tenants={TENANT_NAME: tenant},
    )
    return Case(
        policy=policy,
        alias=draw(st.sampled_from(alias_names)),
        data_class=draw(st.sampled_from(list(DataClass))),
    )


def clamp_had_work_to_do(case: Case) -> bool:
    """True when a non-EU candidate was allowed by the data class rule alone."""
    tenant = case.policy.tenants[TENANT_NAME]
    rule = tenant.data_class_rules.get(case.data_class)
    if rule is None:
        return False
    allowed = set(rule.allow_regions)
    eu_regions = set(case.policy.eu_regions)
    candidates = tenant.overrides.get(case.alias, case.policy.aliases[case.alias].candidates)
    return any(
        case.policy.providers[c.provider].enabled
        and case.policy.providers[c.provider].region in allowed
        and case.policy.providers[c.provider].region not in eu_regions
        for c in candidates
    )


def test_eu_tenant_never_gets_non_eu_provider() -> None:
    """The property, plus the guard that proves it was not checked vacuously.

    The counter lives in this function rather than at module scope so the guard
    reads state only its own `@given` run wrote: a split across pytest-xdist
    workers, or running either half alone, cannot make it pass or fail spuriously.
    """
    clamped = 0

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(case=cases())
    def check(case: Case) -> None:
        nonlocal clamped
        if clamp_had_work_to_do(case):
            clamped += 1

        try:
            result = resolve(case.policy, TENANT_NAME, case.alias, case.data_class.value)
        except ResidencyUnavailable:
            # Deny by default is always a permitted outcome; widening is not.
            return

        assert result
        for candidate in result:
            assert candidate.region in case.policy.eu_regions
            assert case.policy.providers[candidate.provider].enabled

    check()

    # Without this the property above could hold without the clamp ever running.
    assert clamped > 0
