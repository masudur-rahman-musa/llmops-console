"""Relaxed residency mode: what the opt-in widening adds, and what it never adds.

`ResidencyMode.RELAXED` unions `RELAXED_REGIONS` into the allowed set AFTER the
EU clamp, so a local demo can reach a US provider. Three things must survive
that: the `pii` class, which widens for nobody; a data class with no rule, which
stays denied because deny-by-default is structural rather than a region list; and
an unknown tenant, which is refused before any widening is considered.

Every case here states its mode explicitly, including the ones that want strict,
because the default is the behaviour under test in `test_strict_is_the_default_mode`
and nowhere else. The three nordlicht cases read the SHIPPED `routing.yaml`: they
are the ones the request asked for, and a demo that cannot reach OpenAI through
the shipped file is not fixed by a hand-built fixture.

`test_residency_property.py` is imported read-only, for its `cases()` strategy
alone. Nothing here may change that file or its guarantee.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from opsconsole.routing.errors import ResidencyUnavailable
from opsconsole.routing.policy import (
    Alias,
    Candidate,
    DataClass,
    Policy,
    Provider,
    ProviderKind,
    Residency,
    ResidencyMode,
    load_policy,
)
from opsconsole.routing.resolver import RELAXED_REGIONS, Resolved, resolve

from .conftest import SHIPPED_POLICY_PATH, make_policy, make_tenant
from .test_residency_property import TENANT_NAME, Case, cases

# --------------------------------------------------------------------------
# the hand-built arrangement
# --------------------------------------------------------------------------
#
# One provider per region, and two aliases: `us-only` holds a single US
# candidate, so a resolution that succeeds for it did so because `us` was
# allowed and for no other reason, and `self-then-us` holds a permitted
# candidate BEFORE the US one, so a widening shows up as an extra entry rather
# than as the difference between denied and served.

PROVIDERS = {
    "eu-a": Provider(kind=ProviderKind.ANTHROPIC, region="eu"),
    "us-a": Provider(kind=ProviderKind.OPENAI, region="us"),
    "self-a": Provider(kind=ProviderKind.OPENAI_COMPATIBLE, region="self"),
}

ALIASES = {
    "us-only": Alias(candidates=(Candidate(provider="us-a", model="m-us-a"),)),
    "self-then-us": Alias(
        candidates=(
            Candidate(provider="self-a", model="m-self-a"),
            Candidate(provider="us-a", model="m-us-a"),
        )
    ),
}

# An EU tenant with a rule for every class, `pii` narrowed to `self` the way the
# shipped file narrows it.
RULED_TENANT = "acme"
# An EU tenant with a rule for `public` ONLY, so `internal`, `confidential` and
# `pii` are the no-rule case rather than a narrow one.
RULELESS_TENANT = "sparse"
UNKNOWN_TENANT = "nobody"

POLICY: Policy = make_policy(
    providers=PROVIDERS,
    aliases=ALIASES,
    tenants={
        RULED_TENANT: make_tenant(
            residency=Residency.EU,
            rules={
                DataClass.PUBLIC: ("us", "eu", "self"),
                DataClass.INTERNAL: ("eu", "self"),
                DataClass.CONFIDENTIAL: ("eu", "self"),
                DataClass.PII: ("self",),
            },
        ),
        RULELESS_TENANT: make_tenant(
            residency=Residency.EU,
            rules={DataClass.PUBLIC: ("eu", "self")},
        ),
    },
)

DENIED = "ResidencyUnavailable"
"""The outcome of a denied resolution, so two modes can be compared as values."""

Outcome = list[tuple[str, str, str]] | str


def as_tuples(resolved: tuple[Resolved, ...]) -> list[tuple[str, str, str]]:
    return [(r.provider, r.model, r.region) for r in resolved]


def outcome(
    policy: Policy,
    tenant: str,
    alias: str,
    data_class: DataClass,
    mode: ResidencyMode,
) -> Outcome:
    """The resolution as a comparable value: the candidate list, or `DENIED`.

    Only `ResidencyUnavailable` is folded into a value. Any other routing error is
    a different claim and is left to raise out of the case that provoked it.
    """
    try:
        return as_tuples(resolve(policy, tenant, alias, data_class.value, mode=mode))
    except ResidencyUnavailable:
        return DENIED


# --------------------------------------------------------------------------
# the widening (criteria 4, 5, 6, 7)
# --------------------------------------------------------------------------

# Decision table over the data classes a relaxed run may widen. Every row's
# allowed set excludes `us` under strict - `public` because the EU clamp removes
# it, the other two because the rule never named it - so the row's two outcomes
# differ by the mode and by nothing else.
WIDENED_CASES: list[tuple[str, DataClass]] = [
    ("public, where the clamp removed us", DataClass.PUBLIC),
    ("internal", DataClass.INTERNAL),
    ("confidential, the class the Helpdesk sends chat as", DataClass.CONFIDENTIAL),
]


@pytest.mark.parametrize(("case", "data_class"), WIDENED_CASES, ids=[c[0] for c in WIDENED_CASES])
def test_relaxed_widens_each_class_for_an_eu_tenant(case: str, data_class: DataClass) -> None:
    """Catches a resolver that ignores the mode, one that applies the union BEFORE the EU
    clamp (which then removes `us` again, leaving the `public` row denied), and one that
    widens only the two classes the request named by hand.

    The alias holds a single US candidate, so the relaxed row can only succeed through the
    union, and the strict assertion in the same row proves `us` was never allowed to begin
    with - without it the case would pass against a resolver that allows `us` always."""
    assert outcome(POLICY, RULED_TENANT, "us-only", data_class, ResidencyMode.STRICT) == DENIED

    relaxed = outcome(POLICY, RULED_TENANT, "us-only", data_class, ResidencyMode.RELAXED)

    assert relaxed == [("us-a", "m-us-a", "us")]


def test_relaxed_never_widens_pii() -> None:
    """Catches a union applied to every class with a rule: `pii` is the one class personal
    data travels under, and widening it sends it out of the regions the tenant allows.

    The alias holds a permitted `self` candidate FOLLOWED by a US one, so a widened `pii`
    shows as a second entry rather than as a denial turning into a success - a case built
    on a US-only alias would pass against a resolver that widens `pii` and then finds the
    tenant's rule empty for another reason."""
    strict = outcome(POLICY, RULED_TENANT, "self-then-us", DataClass.PII, ResidencyMode.STRICT)

    relaxed = outcome(POLICY, RULED_TENANT, "self-then-us", DataClass.PII, ResidencyMode.RELAXED)

    assert strict == [("self-a", "m-self-a", "self")]
    assert relaxed == strict


def test_relaxed_still_denies_a_class_with_no_rule() -> None:
    """Catches a union written as "relaxed means `us` is allowed" rather than "a rule that
    exists also allows `us`" (decision B): a tenant with no rule for the class would then
    be served from a US provider on the strength of the mode alone, which is deny-by-default
    inverted.

    The `public` row of the same tenant IS widened, which is what stops this case passing
    against a resolver that widens nothing at all for this policy."""
    ruled = outcome(POLICY, RULELESS_TENANT, "us-only", DataClass.PUBLIC, ResidencyMode.RELAXED)

    unruled = outcome(
        POLICY, RULELESS_TENANT, "us-only", DataClass.CONFIDENTIAL, ResidencyMode.RELAXED
    )

    assert ruled == [("us-a", "m-us-a", "us")]
    assert unruled == DENIED


def test_relaxed_still_denies_an_unknown_tenant() -> None:
    """Catches a relaxed path that resolves the alias default for a tenant the policy does
    not define: the unknown-tenant refusal is reached before any mode is consulted, so a
    union moved above it would serve a caller with no rules at all."""
    with pytest.raises(ResidencyUnavailable):
        resolve(
            POLICY,
            UNKNOWN_TENANT,
            "us-only",
            DataClass.PUBLIC.value,
            mode=ResidencyMode.RELAXED,
        )


def test_relaxed_regions_is_pinned() -> None:
    """Catches a widening that grows a region without a plan: the exact tuple is the whole
    content of the loosening, and an extra entry (`ap`, every region) would pass every
    other case in this file."""
    assert RELAXED_REGIONS == ("us",)


def test_strict_is_the_default_mode() -> None:
    """Catches a `mode` parameter defaulting to relaxed, and one whose default is read from
    the environment: every existing call site and `test_residency_property.py` call
    `resolve()` with no mode, so a relaxed default would widen the whole product silently.

    The third assertion is what stops the case passing against a resolver that ignores
    `mode` entirely, which would make the first two trivially equal."""
    stated = outcome(POLICY, RULED_TENANT, "us-only", DataClass.PUBLIC, ResidencyMode.STRICT)
    try:
        implicit: Outcome = as_tuples(
            resolve(POLICY, RULED_TENANT, "us-only", DataClass.PUBLIC.value)
        )
    except ResidencyUnavailable:
        implicit = DENIED
    relaxed = outcome(POLICY, RULED_TENANT, "us-only", DataClass.PUBLIC, ResidencyMode.RELAXED)

    assert implicit == stated
    assert implicit == DENIED
    assert relaxed != DENIED


# --------------------------------------------------------------------------
# the shipped file (criteria 13, 14, 15)
# --------------------------------------------------------------------------
#
# These read `routing.yaml` as it ships, because the request is about the
# Helpdesk's own traffic: nordlicht, data class `confidential`, the chat alias.

NORDLICHT = "nordlicht"
CHAT_ALIAS = "agent-default"
STRONG_ALIAS = "agent-strong"
LOCAL_MODEL = "qwen2.5-3b-instruct"
OPENAI_MODEL = "gpt-4o"


def shipped() -> Policy:
    return load_policy(SHIPPED_POLICY_PATH)


def test_nordlicht_confidential_resolves_to_openai_under_relaxed() -> None:
    """Catches the two halves of the demo being broken separately: a widening that reaches
    no US candidate because `routing.yaml` never got one, and an override still pinned to
    `local` alone, which would resolve the local model however wide the regions are.

    First rather than present, because the first entry is the one a completion actually
    uses: an OpenAI candidate appended behind the mis-extracting local model would leave
    the demo exactly as it was."""
    result = resolve(
        shipped(), NORDLICHT, CHAT_ALIAS, DataClass.CONFIDENTIAL.value, mode=ResidencyMode.RELAXED
    )

    assert result
    assert (result[0].provider, result[0].model, result[0].region) == ("openai", OPENAI_MODEL, "us")


def test_nordlicht_confidential_stays_local_under_strict() -> None:
    """Catches a widening that leaks into strict: the same shipped override now names a US
    candidate first, so a resolver that unions `RELAXED_REGIONS` unconditionally - or reads
    the mode from somewhere other than its argument - answers `openai` here.

    The region assertion is separate from the provider one because a later plan may add
    another self-hosted candidate: a US region in this tuple is the violation, whatever its
    provider is called."""
    result = resolve(
        shipped(), NORDLICHT, CHAT_ALIAS, DataClass.CONFIDENTIAL.value, mode=ResidencyMode.STRICT
    )

    assert as_tuples(result) == [("local", LOCAL_MODEL, "self")]
    assert all(candidate.region != "us" for candidate in result)


def test_nordlicht_agent_strong_is_denied_under_strict_and_served_under_relaxed() -> None:
    """The 451 pair the request asked for, on the alias that genuinely denies it (decision I):
    `agent-strong` has no self-hosted candidate, so strict leaves nothing allowed.

    Catches a mode that widens nothing (the relaxed half stays a 451) and a strict path
    that widens anyway (the strict half is served). Both halves are in one case because
    either alone passes against an implementation that answers the same way in both
    modes."""
    policy = shipped()

    with pytest.raises(ResidencyUnavailable):
        resolve(
            policy, NORDLICHT, STRONG_ALIAS, DataClass.CONFIDENTIAL.value, mode=ResidencyMode.STRICT
        )

    relaxed = resolve(
        policy, NORDLICHT, STRONG_ALIAS, DataClass.CONFIDENTIAL.value, mode=ResidencyMode.RELAXED
    )

    assert relaxed
    assert relaxed[0].provider == "anthropic"


# --------------------------------------------------------------------------
# the property mirror (criterion 5, over the whole generator)
# --------------------------------------------------------------------------


def relaxing_pii_would_change_the_outcome(case: Case) -> bool:
    """True when a `pii` union WOULD have added a candidate, so the guard is not vacuous.

    The tenant has a `pii` rule (without one the no-rule branch denies in both modes) and
    the chosen alias offers an enabled candidate sitting in a relaxed region. These are the
    examples where a resolver that widened `pii` answers differently from one that does not.
    """
    tenant = case.policy.tenants[TENANT_NAME]
    if tenant.data_class_rules.get(DataClass.PII) is None:
        return False
    candidates = tenant.overrides.get(case.alias, case.policy.aliases[case.alias].candidates)
    return any(
        case.policy.providers[candidate.provider].enabled
        and case.policy.providers[candidate.provider].region in RELAXED_REGIONS
        for candidate in candidates
    )


def test_pii_resolves_identically_in_both_modes() -> None:
    """The `pii` exception over the property test's own generator, not one fixture.

    Catches a union that excludes `pii` for the shipped `confidential`-shaped tenant and
    forgets it under some other shape - an override, a rule naming `us` outright, a
    disabled candidate - which `test_relaxed_never_widens_pii` cannot reach. The data class
    is pinned to `pii` rather than drawn, so all 200 examples exercise the exception.

    The counter is the guard: without it the property would hold over examples where no
    relaxed region was reachable at all, and would pass against a resolver that widens
    `pii` freely.
    """
    relaxable = 0

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(case=cases())
    def check(case: Case) -> None:
        nonlocal relaxable
        if relaxing_pii_would_change_the_outcome(case):
            relaxable += 1

        strict = outcome(case.policy, TENANT_NAME, case.alias, DataClass.PII, ResidencyMode.STRICT)
        relaxed = outcome(
            case.policy, TENANT_NAME, case.alias, DataClass.PII, ResidencyMode.RELAXED
        )

        assert relaxed == strict

    check()

    assert relaxable > 0
