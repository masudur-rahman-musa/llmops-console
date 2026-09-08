"""`evals/residency.py`: the per-turn residency verdict (#38), deny by default.

Pure and in-memory: a `Policy` built straight from the models, a region string,
a verdict back. No YAML, no network, no database - `16c` brings Postgres, and
an earlier change decision B reads the allow list from the shipped policy rather than
from a second copy of the rule.

The contract these cases are written against, before it exists:

* `ResidencyVerdict(case_id=..., tenant=..., data_class=..., observed=...,
  allowed=(...), violation=...)`, frozen and `extra="forbid"`.
* `check_residency(policy, *, case_id, tenant, data_class, observed)` reading
  `policy.tenants[tenant].data_class_rules[data_class].allow_regions`.
* `checked_data_class(value) -> DataClass`, the caller's coercion from a string,
  raising `IdentifierInvalid` on anything the closed enum does not name.
* `EVAL_DATA_CLASS` - the data class an eval turn runs under, pinned exactly by a
  test.

Deny by default is the whole module, and the three shapes of it get
a case each: a region outside the list, NO region at all, and a tenant with no
rule for that data class. Each is arranged so only the check under test can
decide it - the tenant exists in every case, and the "outside the list" case has
a rule with regions in it, so a passing implementation cannot be one that merely
notices a missing key.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

import pytest

from opsconsole.evals.residency import (
    EVAL_DATA_CLASS,
    ResidencyVerdict,
    check_residency,
    checked_data_class,
)
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.policy import (
    DataClass,
    DataClassRule,
    Policy,
    RateLimit,
    Residency,
    Tenant,
)

INTERNAL = DataClass.INTERNAL


def make_tenant(rules: Mapping[DataClass, tuple[str, ...]]) -> Tenant:
    """One tenant carrying exactly the data class rules a case names."""
    return Tenant(
        residency=Residency.EU,
        data_class_rules={
            data_class: DataClassRule(allow_regions=regions)
            for data_class, regions in rules.items()
        },
        budget_usd_month=Decimal("10.00"),
        rate_limit=RateLimit(rpm=60, tpm=60_000),
    )


def make_policy(tenants: Mapping[str, Tenant]) -> Policy:
    """A policy with nothing in it but tenants: nothing here resolves an alias."""
    return Policy(providers={}, eu_regions=("eu", "self"), aliases={}, tenants=tenants)


POLICY = make_policy(
    {
        "acme": make_tenant({INTERNAL: ("eu", "self"), DataClass.PUBLIC: ("us", "eu", "self")}),
        # Every class BUT the one an eval turn runs under, so a missing rule is
        # what the "no rule" case tests and not a tenant with no rules at all.
        "sparse": make_tenant({DataClass.PUBLIC: ("us", "eu", "self")}),
    }
)
"""The two tenants every case below resolves against."""


def test_the_eval_data_class_is_the_published_one() -> None:
    """Catches an eval turn silently reclassified: run as `public`, `acme`'s rule allows
    `us` and a US-served turn stops being a violation at all, so #38's gate would go quiet
    without a single test changing. Pinned as an exact value."""
    assert EVAL_DATA_CLASS is DataClass.INTERNAL


def test_an_allowed_region_is_not_a_violation() -> None:
    """Catches a check that reports every turn as a violation, which would make the run
    fail closed on a correct target and #38's gate unusable. `eu` is in the tenant's
    `internal` list and nowhere near its edge, so only the membership test can decide it."""
    verdict = check_residency(
        POLICY, case_id="case-1", tenant="acme", data_class=INTERNAL, observed="eu"
    )

    assert verdict.violation is False


def test_a_region_outside_the_allow_list_is_a_violation() -> None:
    """Catches a check that only asks whether a region was reported. `us` is a real region
    the policy names elsewhere (it is in this tenant's `public` list), so a check reading
    the WRONG data class's rule passes it - that is the neighbouring guard this row is
    arranged against, and only the `internal` allow list can refuse it."""
    verdict = check_residency(
        POLICY, case_id="case-1", tenant="acme", data_class=INTERNAL, observed="us"
    )

    assert verdict.violation is True


def test_a_missing_region_is_a_violation_and_not_a_skip() -> None:
    """Catches the shape a half-implemented target actually produces: an envelope with no
    `region` at all. A check that skips when there is nothing to compare reports a clean
    run for a target that never proved where the data went, which is a project rule's
    deny-by-default read backwards."""
    verdict = check_residency(
        POLICY, case_id="case-1", tenant="acme", data_class=INTERNAL, observed=None
    )

    assert verdict.violation is True
    assert verdict.observed is None


def test_a_tenant_with_no_rule_for_the_class_denies_rather_than_permits() -> None:
    """Catches `data_class_rules.get(data_class)` treated as "no restriction". The tenant
    exists and has a `public` rule, so the only thing deciding this row is what a MISSING
    rule means; `allowed` is asserted empty as well, so an implementation that widened to
    every region the policy knows cannot pass by accident."""
    verdict = check_residency(
        POLICY, case_id="case-1", tenant="sparse", data_class=INTERNAL, observed="eu"
    )

    assert verdict.violation is True
    assert verdict.allowed == ()


def test_the_verdict_names_the_case_the_tenant_the_observation_and_the_allowed_set() -> None:
    """Catches a verdict that carries only a boolean: `16c` writes these fields into a
    report and an operator reading "residency violation" needs to know which case, which
    tenant, where it went and where it was allowed to go. Every field is read back, so a
    verdict that drops one fails here rather than in an empty report column."""
    verdict = check_residency(
        POLICY, case_id="access-auto-001", tenant="acme", data_class=INTERNAL, observed="us"
    )

    assert verdict == ResidencyVerdict(
        case_id="access-auto-001",
        tenant="acme",
        data_class="internal",
        observed="us",
        allowed=("eu", "self"),
        violation=True,
    )


def test_an_unknown_tenant_raises_rather_than_passing() -> None:
    """Catches `policy.tenants.get(tenant)` falling through to "no rule, no problem" or to
    a `KeyError` the runner never catches. A tenant the policy does not define is an
    operator mistake, and the only safe answers are a refusal or a violation - never a
    clean verdict for traffic no policy ever governed."""
    with pytest.raises(IdentifierInvalid):
        check_residency(
            POLICY, case_id="case-1", tenant="globex", data_class=INTERNAL, observed="eu"
        )


ACCEPTED_DATA_CLASSES: list[tuple[str, DataClass]] = [
    ("public", DataClass.PUBLIC),
    ("internal", DataClass.INTERNAL),
    ("confidential", DataClass.CONFIDENTIAL),
    ("pii", DataClass.PII),
]


@pytest.mark.parametrize(
    ("value", "expected"), ACCEPTED_DATA_CLASSES, ids=[row[0] for row in ACCEPTED_DATA_CLASSES]
)
def test_every_published_data_class_coerces(value: str, expected: DataClass) -> None:
    """Catches a coercion that only knows the class an eval turn happens to use today:
    `16c` and the console pass this string through from a report, so a helper that refuses
    `confidential` would make the harness unusable for the tenants that need it most."""
    assert checked_data_class(value) is expected


REJECTED_DATA_CLASSES: list[tuple[str, str]] = [
    ("empty", ""),
    ("a class nothing defines", "secret"),
    ("uppercase", "Internal"),
    ("a trailing newline", "internal\n"),
    ("a trailing CRLF", "internal\r\n"),
    ("a trailing space", "internal "),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_DATA_CLASSES, ids=[row[0] for row in REJECTED_DATA_CLASSES]
)
def test_an_unknown_data_class_is_refused(label: str, value: str) -> None:
    """Catches a bare `ValueError` from the enum reaching the operator as a traceback, and
    catches a coercion that strips or lowercases first: `internal\\n` is not a fifth class
    and a helper that quietly repaired it would make a report say a class the caller never
    named. The trailing-newline and CRLF rows are a project rule's."""
    with pytest.raises(IdentifierInvalid):
        checked_data_class(value)
