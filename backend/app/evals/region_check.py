"""Every turn's observed region against the tenant's allow list (#38).

Deny by default, in three shapes and with no fourth: a region
outside the list is a violation, NO region at all is a violation rather than a
skip, and a tenant with no rule for the data class denies rather than permits.
Each has its own case in `tests/evals/test_residency.py`, arranged so only the
check under test can decide it.

The allow list is read from the shipped `routing.yaml` `Policy` (an earlier change
decision B) rather than from the gateway's `residency_log` in Postgres: no
database is in this plan, and reusing the policy means there is no second copy of
the rule to drift from the one the gateway enforces. Nothing here widens anything
- a verdict is a record, and the runner turns it into a failed run.

`16c` writes these fields into a report, which is why the verdict names the case,
the tenant, the observation and the allowed set rather than carrying a bare
boolean (`test_the_verdict_names_the_case_the_tenant_the_observation_and_the_allowed_set`).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from app.core.validation import IdentifierInvalid
from app.model_routing.policy import DataClass, Policy

EVAL_DATA_CLASS: Final[DataClass] = DataClass.INTERNAL
"""The data class an eval turn runs under.

`internal` rather than `public`, because the golden set carries a seed's ticket
text and its principals. The choice is load-bearing rather than cosmetic: under
`public` most tenants allow `us` as well, and a US-served turn would stop being a
violation at all - #38's gate would go quiet without a test changing. Pinned by
`test_the_eval_data_class_is_the_published_one`."""


class ResidencyVerdict(BaseModel):
    """Where one turn was served, and whether the tenant's policy allowed it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    tenant: str
    data_class: str
    observed: str | None
    allowed: tuple[str, ...]
    violation: bool


def checked_data_class(value: str) -> DataClass:
    """Coerce a data class name into the closed enum, or refuse it.

    A thin wrapper over the enum, and deliberately its own: `routing/resolver.py`
    and `gateway/context.py` each carry a private `_checked_data_class` of the
    same shape for their own callers, and importing a private name across
    packages is a worse dependency than three lines that name the same enum. The
    enum is the single authority either way.

    Nothing is stripped or lowercased first: `internal\\n` is not a fifth class,
    and a helper that repaired it would put a class the caller never named into a
    report (`test_an_unknown_data_class_is_refused`).
    """
    try:
        return DataClass(value)
    except ValueError as exc:
        raise IdentifierInvalid(f"data_class: {value!r} is not a known data class") from exc


def check_residency(
    policy: Policy,
    *,
    case_id: str,
    tenant: str,
    data_class: DataClass,
    observed: str | None,
) -> ResidencyVerdict:
    """Judge one turn's observed region against what the tenant's policy allows.

    An unknown tenant raises rather than answering a clean verdict: traffic no
    policy governs is an operator's mistake, and the only safe answers are a
    refusal or a violation (`test_an_unknown_tenant_raises_rather_than_passing`).

    A data class the tenant has no rule for is an EMPTY allow list, matching the
    resolver's own reading of a missing rule, so nothing is widened on the way to
    a verdict (`test_a_tenant_with_no_rule_for_the_class_denies_rather_than_permits`).
    """
    regime = policy.tenants.get(tenant)
    if regime is None:
        raise IdentifierInvalid(f"tenant: {tenant!r} is not defined by this policy")

    allowed = regime.allowed_regions(data_class)
    return ResidencyVerdict(
        case_id=case_id,
        tenant=tenant,
        data_class=data_class.value,
        observed=observed,
        allowed=allowed,
        violation=observed is None or observed not in allowed,
    )
