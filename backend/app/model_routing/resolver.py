"""Turning (tenant, alias, data class) into an ordered candidate list.

A pure function over a policy snapshot: no I/O, no provider calls, no state. The
one seam is `is_healthy`, which the routes fill with "an adapter exists AND
`providers/breaker.py` allows it". The breaker stays out of this module on
purpose: mutable state inside this function would make it hard to test.

The data class does not pick providers. It decides redaction, and it is part of
the cache key, so it is checked here and nowhere narrower.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from app.core.errors import TenantMismatch
from app.core.validation import IdentifierInvalid, checked_identifier
from app.model_routing.errors import AliasUnknown, AllProvidersUnavailable
from app.model_routing.policy import Candidate, DataClass, Policy


class Resolved(BaseModel):
    """One place to send the request: a provider and its model id."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str


def _always_healthy(provider: str) -> bool:
    """The default health seam, for a caller that has no health to report.

    The routes replace it with `services.chat_pipeline.health_seam`; the admin view
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
) -> tuple[Resolved, ...]:
    """Resolve a request to the providers that can serve it, best first.

    Raises `IdentifierInvalid` before any lookup happens, `AliasUnknown` for an
    alias the policy does not define, `TenantMismatch` for a tenant the policy
    does not carry, and `AllProvidersUnavailable` when no candidate is turned on
    and healthy.
    """
    checked_identifier(tenant, "tenant")
    checked_identifier(alias, "alias")
    _checked_data_class(data_class)

    if alias not in policy.aliases:
        raise AliasUnknown(f"alias: {alias!r} is not defined by the policy")

    tenant_config = policy.tenants.get(tenant)
    if tenant_config is None:
        raise TenantMismatch(f"tenant: {tenant!r} is not defined by the policy", param="tenant")

    candidates: tuple[Candidate, ...] = tenant_config.overrides.get(
        alias, policy.aliases[alias].candidates
    )

    enabled = tuple(
        Resolved(provider=candidate.provider, model=candidate.model)
        for candidate in candidates
        if policy.providers[candidate.provider].enabled
    )
    healthy = tuple(item for item in enabled if is_healthy(item.provider))
    if not healthy:
        raise AllProvidersUnavailable(
            f"no enabled, healthy provider for tenant {tenant!r}, alias {alias!r}"
        )
    return healthy
