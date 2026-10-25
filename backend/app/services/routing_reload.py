"""Atomic swap of the routing runtime: policy, prices, adapters and breaker as one.

Every in-flight request holds one `RoutingRuntime` for its whole life, so it
never sees a new policy beside old adapters. The old adapter set closes only
once the last request holding it releases — so no request loses its client
mid-call. A reload that fails to parse keeps the old set untouched.

`RoutingRuntimeStore` watches `app.state`: the adapters, the breaker and the
policy store keep their current home there, so no fixture that sets a stub
adapter has to change.

The lock is a `threading.Lock` rather than an asyncio lock: reload runs in a
FastAPI worker thread while leases are touched on the event loop, so a plain
threading lock — which is safe across both — is the right choice.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from starlette.datastructures import State

from app.limits.prices import PriceStore
from app.model_routing.policy import (
    Policy,
    PolicyStore,
    check_policy_models_priced,
    load_routing,
)
from app.providers.base import Adapter
from app.providers.breaker import CircuitBreaker
from app.providers.factory import build_adapters, close_adapters

RUNTIME_SCOPE_KEY = "routing_runtime"
"""The ASGI scope key that holds the `RoutingRuntime` for the current request.

`RoutingLeaseMiddleware` writes this before calling the inner app and reads it
back after, in a `finally`, to release the lease even when the response is
streamed. `get_clients` in `app.api.deps` reads it to build `Clients` with a
consistent snapshot of the policy, adapters and breaker.
"""


@dataclass(frozen=True)
class RoutingRuntime:
    """One consistent snapshot of the three things a request routes with.

    Passed to the request by `RoutingLeaseMiddleware` and read by `get_clients`
    from the ASGI scope, so the policy, adapters and breaker are always from
    the same reload generation.
    """

    policy: Policy
    adapters: Mapping[str, Adapter]
    breaker: CircuitBreaker


class RoutingRuntimeStore:
    """Manages the routing runtime and reference-counts it per in-flight request.

    `acquire()` takes a consistent snapshot of the current policy, adapters and
    breaker and increments the refcount for that adapter set. `release()` drops
    the refcount and closes the adapter set if it was dropped by a reload and
    no other request still holds it. `reload()` parses the new policy, builds
    new adapters and a new breaker, then swaps all three into `app.state` under
    the lock — so `acquire()` always sees either the fully old or the fully new
    set, never a mix.

    The store holds a reference to the Starlette `State` object rather than the
    three pieces directly, because test fixtures set stubs on `app.state` and
    those stubs must be what `acquire()` reads.
    """

    def __init__(self, state: State) -> None:
        self._state = state
        self._lock = threading.Lock()
        # id(adapters_dict) -> (adapters_dict, refcount, pending_close)
        self._leases: dict[int, tuple[Mapping[str, Adapter], int, bool]] = {}

    def acquire(self) -> RoutingRuntime:
        """Take a consistent snapshot of the current runtime and open a lease.

        Must be balanced by a `release()` call once the request is done —
        `RoutingLeaseMiddleware` does this in a `finally`, so streaming replies
        release only after the last frame.
        """
        with self._lock:
            adapters: Mapping[str, Adapter] = self._state.adapters
            runtime = RoutingRuntime(
                policy=self._state.policy.policy,
                adapters=adapters,
                breaker=self._state.breaker,
            )
            key = id(adapters)
            if key in self._leases:
                prev, count, pending = self._leases[key]
                self._leases[key] = (prev, count + 1, pending)
            else:
                self._leases[key] = (adapters, 1, False)
            return runtime

    def release(self, runtime: RoutingRuntime) -> None:
        """Drop the lease for one request.  Closes the adapter set if it was
        marked for close during a reload and this was the last holder.
        """
        to_close: Mapping[str, Adapter] | None = None
        with self._lock:
            key = id(runtime.adapters)
            if key not in self._leases:
                return
            prev, count, pending = self._leases[key]
            count -= 1
            if count <= 0:
                del self._leases[key]
                if pending:
                    to_close = prev
            else:
                self._leases[key] = (prev, count, pending)
        if to_close is not None:
            close_adapters(to_close)

    def reload(self, environ: Mapping[str, str]) -> tuple[str, ...]:
        """Parse the new policy and prices, build new adapters and breaker, swap all four.

        `routing.yaml` and the provider folder are read together, once, so the
        new policy and the new prices come from the same files. Parsing happens
        outside the lock — it is the slow part and raises `PolicyInvalid`
        without touching any shared state if a file is bad. Under the lock the
        four pieces are swapped atomically, so `acquire()`
        always sees a consistent set. The old adapter set is closed once all
        leases on it are released; if none are open it is closed right away.

        Returns the sorted provider key tuple of the new adapter mapping, which
        is what `ReloadResult.providers` publishes.
        """
        policy_store: PolicyStore = self._state.policy
        price_store: PriceStore = self._state.prices
        # Parse OUTSIDE the lock: raises PolicyInvalid without changing state.
        # The price-coverage check runs here too, for the same reason.
        new_policy, new_table = load_routing(policy_store._path, policy_store._providers_dir)
        check_policy_models_priced(new_policy, new_table)

        to_close: Mapping[str, Adapter] | None = None
        with self._lock:
            new_adapters = build_adapters(new_policy, environ)
            new_breaker = CircuitBreaker()
            old_adapters: Mapping[str, Adapter] = self._state.adapters
            old_key = id(old_adapters)

            # Swap all four atomically.
            policy_store._policy = new_policy
            price_store._table = new_table
            self._state.adapters = new_adapters
            self._state.breaker = new_breaker

            # Deferred or immediate close for the old adapter set.
            if old_key in self._leases:
                prev, count, _ = self._leases[old_key]
                if count <= 0:
                    del self._leases[old_key]
                    to_close = old_adapters
                else:
                    self._leases[old_key] = (prev, count, True)
            else:
                to_close = old_adapters

        if to_close is not None:
            close_adapters(to_close)
        return tuple(sorted(new_adapters))

    @property
    def lease_count(self) -> int:
        """The total number of open leases across all adapter sets.

        Zero means no request is holding any runtime. Tests use it to prove the
        middleware releases every lease it takes.
        """
        with self._lock:
            return sum(count for _, count, _ in self._leases.values())
