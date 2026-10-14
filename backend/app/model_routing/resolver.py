"""Turning (tenant, alias, data class) into an ordered candidate list.

A pure function over a policy snapshot: no I/O, no provider calls, no state. The
one seam is `is_healthy`, which the routes fill with "an adapter exists AND
`providers/breaker.py` allows it". The breaker stays out of this module on
purpose: mutable state inside this function is what would stop the residency
property test being a proof.

Residency is deny by default. Under the default `ResidencyMode.STRICT` the filter
only ever narrows and no branch adds a region back, which is what makes
`test_eu_tenant_never_gets_non_eu_provider` provable rather than merely
unfalsified, and `test_shipped_policy_exhaustive` checks the same claim over
every triple the shipped file can produce. Both call `resolve()` with no `mode`,
so both are statements about strict.

`ResidencyMode.RELAXED` is the one exception, opt-in per process and off by
default: it unions `RELAXED_REGIONS` into the allowed set after the
clamp, for a data class that HAS a rule and is not `pii`. It widens nothing on
failure - the union is decided by the mode alone and never by an empty result -
so a project rule still holds, and `pii` is identical in both modes
(`test_relaxed_never_widens_pii`, `test_pii_resolves_identically_in_both_modes`).
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from app.core.validation import IdentifierInvalid, checked_identifier
from app.model_routing.errors import (
    AliasUnknown,
    AllProvidersUnavailable,
    ResidencyUnavailable,
)
from app.model_routing.policy import (
    Candidate,
    DataClass,
    Policy,
    Residency,
    ResidencyMode,
)

RELAXED_REGIONS = ("us",)
"""The regions `ResidencyMode.RELAXED` adds, and the whole of what relaxed means.

Exactly one region, pinned by `test_relaxed_regions_is_pinned`: the mode exists so
a local demo can reach a US chat provider, and every region not named
here is as denied under relaxed as it is under strict. `pii` never gains these,
whatever the mode (`test_relaxed_never_widens_pii`).
"""


class Resolved(BaseModel):
    """One place to send the request: a provider, its model id, and its region."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    region: str


def _always_healthy(provider: str) -> bool:
    """The default health seam, for a caller that has no health to report.

    The routes replace it with `gateway.routes.chat.health_seam`; the admin view
    and `/v1/models` deliberately keep this one, so an operator's catalogue does
    not flicker with a breaker.
    """
    return True


def _checked_data_class(value: str) -> DataClass:
    """Coerce the request's data class into the closed enum, or reject it."""
    try:
        return DataClass(value)
    except ValueError as exc:
        raise IdentifierInvalid(f"data_class: {value!r} is not a known data class") from exc


def resolve(
    policy: Policy,
    tenant: str,
    alias: str,
    data_class: str,
    *,
    is_healthy: Callable[[str], bool] = _always_healthy,
    mode: ResidencyMode = ResidencyMode.STRICT,
) -> tuple[Resolved, ...]:
    """Resolve a request to the providers allowed to serve it, best first.

    Raises `IdentifierInvalid` before any lookup happens, `AliasUnknown` for an
    alias the policy does not define, `ResidencyUnavailable` when nothing is
    allowed to serve the request, and `AllProvidersUnavailable` when candidates
    were allowed but none is healthy.

    `mode` defaults to `STRICT`, so a caller that says nothing gets today's
    behaviour and a call site missed in a later change fails safe
    (`test_strict_is_the_default_mode`). `RELAXED` is the one widening this module
    has, it is applied below rather than here, and `pii` is outside it.
    """
    checked_identifier(tenant, "tenant")
    checked_identifier(alias, "alias")
    requested_class = _checked_data_class(data_class)

    if alias not in policy.aliases:
        raise AliasUnknown(f"alias: {alias!r} is not defined by the policy")

    tenant_config = policy.tenants.get(tenant)
    if tenant_config is None:
        # Deny by default. An unknown tenant has no rules, so no region is
        # allowed; widening to the alias default would be exactly the silent
        # widening the residency guarantee forbids.
        raise ResidencyUnavailable(f"tenant: {tenant!r} is not defined by the policy")

    candidates: tuple[Candidate, ...] = tenant_config.overrides.get(
        alias, policy.aliases[alias].candidates
    )

    rule = tenant_config.data_class_rules.get(requested_class)
    allowed = set(rule.allow_regions) if rule is not None else set()
    if tenant_config.residency is Residency.EU:
        # The outer clamp. A data class rule may name a non-EU region, and for an
        # EU tenant that region still does not survive.
        allowed &= set(policy.eu_regions)
    if mode is ResidencyMode.RELAXED and rule is not None and requested_class is not DataClass.PII:
        # The one branch in this module that ADDS a region, and the reason the
        # residency property test calls `resolve()` without a mode: under the
        # default it cannot run at all. Deliberately AFTER the clamp, because a
        # union before it is removed again and relaxed would mean nothing; and
        # gated on `rule is not None`, so a data class the tenant never ruled on
        # stays denied and deny-by-default remains structural (decision B).
        # `pii` is excluded here and nowhere else, which is what keeps personal
        # data on the box under every mode.
        allowed |= set(RELAXED_REGIONS)

    permitted = tuple(
        Resolved(
            provider=candidate.provider,
            model=candidate.model,
            region=policy.providers[candidate.provider].region,
        )
        for candidate in candidates
        if policy.providers[candidate.provider].enabled
        and policy.providers[candidate.provider].region in allowed
    )
    if not permitted:
        raise ResidencyUnavailable(
            f"no candidate for tenant {tenant!r}, alias {alias!r}, "
            f"data class {requested_class.value!r} is allowed in {sorted(allowed)}"
        )

    healthy = tuple(item for item in permitted if is_healthy(item.provider))
    if not healthy:
        raise AllProvidersUnavailable(
            f"every provider allowed for tenant {tenant!r}, alias {alias!r} is unhealthy"
        )
    return healthy
