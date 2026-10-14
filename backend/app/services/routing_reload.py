"""Rebuild the adapter set and the circuit breaker after a policy reload.

The route re-reads `routing.yaml` and swaps the snapshot; this is the other
half, kept out of the route so the rebuild is one named step the route calls
rather than inline wiring. The order is the whole safety property: only once the
new snapshot is in place is the adapter set rebuilt from it, swapped in, and the
PREVIOUS set closed - previous, not dropped, because a rebuild mints a fresh
client for every surviving provider too, so the old objects are all garbage and
closing them is what keeps a reload from leaking a connection pool
(`test_a_reload_leaves_the_new_adapter_set_callable`).

The circuit breaker is rebuilt with the adapter set, and for the same reason:
both are keyed by the provider names of the policy that has just been replaced,
so a reload that renamed or repointed a provider would otherwise leave a breaker
open against a name that no longer means what it meant
(`test_a_policy_reload_rebuilds_the_breaker`).
"""

from __future__ import annotations

from fastapi import FastAPI

from app.model_routing.policy import PolicyStore
from app.providers.base import Adapter
from app.providers.breaker import CircuitBreaker
from app.providers.factory import build_adapters, close_adapters


def rebuild_adapter_set(app: FastAPI) -> tuple[str, ...]:
    """Rebuild the adapters and breaker from the current policy snapshot, closing the old set.

    Returns the sorted provider key set of the new adapter mapping, which is what
    `ReloadResult.providers` publishes.
    """
    store: PolicyStore = app.state.policy
    previous: dict[str, Adapter] = app.state.adapters
    app.state.adapters = build_adapters(store.policy, app.state.provider_environ)
    app.state.breaker = CircuitBreaker()
    close_adapters(previous)
    return tuple(sorted(app.state.adapters))
