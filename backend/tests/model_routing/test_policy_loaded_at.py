"""When the routing policy was loaded: at start, on `replace`, and on a runtime reload.

The clock is a plain object the test moves by hand, so there is no real waiting.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from starlette.datastructures import State

from app.limits.prices import PriceStore
from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import PolicyStore, load_policy
from app.providers.breaker import CircuitBreaker
from app.providers.factory import close_adapters
from app.services.routing_reload import RoutingRuntimeStore

from .conftest import (
    SHIPPED_POLICY_PATH,
    SHIPPED_PROVIDERS_DIR,
    DocFactory,
    PolicyWriter,
    providers_dir_of,
)

NINE = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
NINE_FIVE = datetime(2026, 10, 26, 9, 5, tzinfo=UTC)


class FakeClock:
    """A clock that reads whatever `now` is set to."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_store_records_load_time(policy_file: PolicyWriter, valid_doc: DocFactory) -> None:
    """Catches a store that reads the real clock (or records nothing) instead of the clock
    it was given, so `loaded_at` could never be pinned."""
    path = policy_file(valid_doc())

    store = PolicyStore(path, providers_dir_of(path), clock=FakeClock(NINE))

    assert store.loaded_at == NINE


def test_replace_moves_load_time(policy_file: PolicyWriter, valid_doc: DocFactory) -> None:
    """Catches a `replace` that swaps the policy but leaves `loaded_at` at the old time, or
    sets the time without swapping."""
    path = policy_file(valid_doc(rpm=60))
    clock = FakeClock(NINE)
    store = PolicyStore(path, providers_dir_of(path), clock=clock)
    other_path = policy_file(valid_doc(rpm=120), name="other.yaml")
    other = load_policy(other_path, providers_dir_of(other_path))
    clock.now = NINE_FIVE

    store.replace(other)

    assert store.policy is other
    assert store.loaded_at == NINE_FIVE


def test_failed_reload_keeps_old_time(policy_file: PolicyWriter, valid_doc: DocFactory) -> None:
    """Catches a reload that stamps the new time before it parses, so a refused file moves
    `loaded_at` while the old policy is still the live one."""
    path = policy_file(valid_doc())
    clock = FakeClock(NINE)
    store = PolicyStore(path, providers_dir_of(path), clock=clock)
    before = store.policy
    path.write_text("providers: [oops\n  - :::\n", encoding="utf-8")
    clock.now = NINE_FIVE

    with pytest.raises(PolicyInvalid):
        store.reload()

    assert store.policy is before
    assert store.loaded_at == NINE


def test_runtime_reload_moves_load_time() -> None:
    """Catches a runtime reload that still writes the private policy field directly, which
    skips `replace` and leaves the page showing the start-up time after a live reload."""
    clock = FakeClock(NINE)
    state = State()
    state.policy = PolicyStore(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR, clock=clock)
    state.prices = PriceStore(SHIPPED_PROVIDERS_DIR)
    state.adapters = {}
    state.breaker = CircuitBreaker()
    store = RoutingRuntimeStore(state)
    assert state.policy.loaded_at == NINE
    clock.now = NINE_FIVE

    store.reload({})

    try:
        assert state.policy.loaded_at == NINE_FIVE
    finally:
        close_adapters(state.adapters)
