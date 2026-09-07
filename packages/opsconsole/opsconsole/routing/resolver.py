"""Turning (tenant, alias, data class) into an ordered candidate list.

A pure function over a policy snapshot: no I/O, no provider calls, no state. The
one seam is `is_healthy`, which the routes fill with "an adapter exists AND
`providers/breaker.py` allows it". The breaker stays out of this module on
purpose: mutable state inside this function is what would stop the residency
property test being a proof.

Residency is deny by default and the filter only ever narrows. There is no
branch that adds a region back, which is what makes
`test_eu_tenant_never_gets_non_eu_provider` provable rather than merely
unfalsified, and `test_shipped_policy_exhaustive` checks the same claim over
every triple the shipped file can produce.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from opsconsole.routing.errors import (
    AliasUnknown,
    AllProvidersUnavailable,
    IdentifierInvalid,
    ResidencyUnavailable,
)
from opsconsole.routing.policy import Candidate, DataClass, Policy, Residency

MAX_IDENTIFIER_LENGTH = 64
"""Longest tenant or alias a request may name. Pinned by `test_max_identifier_length_is_pinned`."""

IDENTIFIER_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
"""Lowercase slug, matched with `re.fullmatch`.

`.match` with `^...$` would accept a trailing newline, and these strings become
dictionary keys, metric labels and cache key parts. The rejection table in
`test_identifier_rejection_table` covers a trailing newline, a trailing CRLF and
a trailing space.
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


def _checked_identifier(value: str, field: str) -> str:
    """Validate one client-supplied identifier before it is used as a key."""
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(f"{field}: longer than {MAX_IDENTIFIER_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(f"{field}: not a lowercase slug")
    return value


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
) -> tuple[Resolved, ...]:
    """Resolve a request to the providers allowed to serve it, best first.

    Raises `IdentifierInvalid` before any lookup happens, `AliasUnknown` for an
    alias the policy does not define, `ResidencyUnavailable` when nothing is
    allowed to serve the request, and `AllProvidersUnavailable` when candidates
    were allowed but none is healthy.
    """
    _checked_identifier(tenant, "tenant")
    _checked_identifier(alias, "alias")
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
