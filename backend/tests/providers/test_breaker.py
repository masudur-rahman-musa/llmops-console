"""The per-provider circuit breaker: its three numbers, its states, its transitions.

Everything here is pure. The breaker takes its clock as an argument, so every case
drives time by assignment rather than by sleeping, and the whole file runs in
microseconds. No app, no adapter, no I/O.

The contract these tests are written against, before it exists:

* `app.providers.breaker.CircuitState` - an `IntEnum` whose members are
  exactly `CLOSED = 0`, `HALF_OPEN = 1`, `OPEN = 2`, because those are the values
  `gateway_circuit_state` publishes.
* `app.providers.breaker.MAX_FAILURES`, `FAILURE_WINDOW_SECONDS`,
  `HALF_OPEN_AFTER_SECONDS` - the three numbers of SPEC section 6.
* `app.providers.breaker.CircuitBreaker(clock=...)` - `allows(provider)`,
  `record_success(provider)`, `record_failure(provider)` and `state(provider)`,
  the last being what the gauge reads. `clock` is keyword-only and returns
  monotonic seconds.

One piece of the contract these cases pin that the plan states only in prose: an
OPEN breaker is resolved by a TRIAL, never by the failure window emptying
underneath it. `test_the_open_state_survives_the_failure_window_emptying` is the
case, and `test_one_trial_failure_reopens_it_and_restarts_the_thirty_seconds`
depends on it - it is what makes that case about the single trial failure rather
than about the fifth failure in a window.
"""

from __future__ import annotations

import pytest

from app.providers.breaker import (
    FAILURE_WINDOW_SECONDS,
    HALF_OPEN_AFTER_SECONDS,
    MAX_FAILURES,
    CircuitBreaker,
    CircuitState,
)

PROVIDER = "anthropic-eu"
"""One provider name, taken from no policy file: the breaker keys on whatever
string it is handed and knows nothing about routing."""

OTHER_PROVIDER = "local"


class ManualClock:
    """A clock that only moves when a test moves it.

    The injected seam itself: a breaker built on `time.monotonic` could only be
    tested by sleeping 30 seconds, so the half-open boundary would be either slow
    or untested.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def opened_breaker(clock: ManualClock) -> CircuitBreaker:
    """A breaker taken to OPEN by exactly `MAX_FAILURES` failures at the clock's now."""
    breaker = CircuitBreaker(clock=clock)
    for _ in range(MAX_FAILURES):
        breaker.record_failure(PROVIDER)
    assert breaker.state(PROVIDER) is CircuitState.OPEN, "the arrangement did not open it"
    return breaker


# --------------------------------------------------------------------------
# the numbers, exactly
# --------------------------------------------------------------------------


def test_the_three_numbers_are_the_spec_ones_exactly() -> None:
    """Catches a breaker tuned to a value near SPEC section 6's rather than at it - three
    failures instead of five, a 10s window, a 5s recovery - each of which changes how much
    provider trouble the gateway absorbs before it stops sending traffic, and none of
    which any behavioural case below could tell from the intended one on its own."""
    assert MAX_FAILURES == 5
    assert FAILURE_WINDOW_SECONDS == 60.0
    assert HALF_OPEN_AFTER_SECONDS == 30.0


def test_the_state_values_are_the_ones_the_gauge_publishes() -> None:
    """Catches a state enum whose members are auto-numbered from 1, or a `StrEnum`:
    `gateway_circuit_state` is documented as 0 closed, 1 half-open, 2 open, so an operator's
    alert on `== 2` would fire on a healthy provider (or never fire at all) while every
    behavioural test in this file still passed."""
    assert int(CircuitState.CLOSED) == 0
    assert int(CircuitState.HALF_OPEN) == 1
    assert int(CircuitState.OPEN) == 2
    assert [int(state) for state in CircuitState] == [0, 1, 2]


# --------------------------------------------------------------------------
# opening: the failure count and its window
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failures", "expected_state", "expected_allows"),
    [
        (MAX_FAILURES - 1, CircuitState.CLOSED, True),
        (MAX_FAILURES, CircuitState.OPEN, False),
    ],
    ids=["one under the cap", "at the cap"],
)
def test_the_failure_count_boundary(
    failures: int, expected_state: CircuitState, expected_allows: bool
) -> None:
    """Both sides of the exact boundary, which is the only thing that separates `>=
    MAX_FAILURES` from `> MAX_FAILURES`: an off-by-one opens the breaker on the sixth
    failure, so one extra request reaches a provider that is already down on every
    outage, and it is invisible to any case that only counts to ten."""
    clock = ManualClock()
    breaker = CircuitBreaker(clock=clock)

    for _ in range(failures):
        breaker.record_failure(PROVIDER)

    assert breaker.state(PROVIDER) is expected_state
    assert breaker.allows(PROVIDER) is expected_allows


def test_failures_older_than_the_window_do_not_count() -> None:
    """Catches a breaker that counts failures for all time instead of inside
    `FAILURE_WINDOW_SECONDS`: five unrelated failures spread over a week would then open
    it on a provider that is working, and the deque would grow without bound. The old
    failure is the fifth one, so a breaker that kept it would be OPEN here."""
    clock = ManualClock()
    breaker = CircuitBreaker(clock=clock)
    breaker.record_failure(PROVIDER)

    clock.now = FAILURE_WINDOW_SECONDS + 1.0
    for _ in range(MAX_FAILURES - 1):
        breaker.record_failure(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.CLOSED
    assert breaker.allows(PROVIDER) is True


def test_an_unknown_provider_is_allowed() -> None:
    """Catches a breaker that denies what it has no record of - an allow list rather than
    a breaker - which would mean a fresh process, or a provider newly added to the policy,
    serving nothing until something failed."""
    breaker = CircuitBreaker(clock=ManualClock())

    assert breaker.allows(OTHER_PROVIDER) is True
    assert breaker.state(OTHER_PROVIDER) is CircuitState.CLOSED


# --------------------------------------------------------------------------
# half-opening: the recovery boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("elapsed", "expected_state", "expected_allows"),
    [
        (HALF_OPEN_AFTER_SECONDS - 0.1, CircuitState.OPEN, False),
        (HALF_OPEN_AFTER_SECONDS, CircuitState.HALF_OPEN, True),
    ],
    ids=["just under 30s", "exactly 30s"],
)
def test_the_half_open_boundary(
    elapsed: float, expected_state: CircuitState, expected_allows: bool
) -> None:
    """Both sides of the recovery boundary. Catches a breaker that half-opens strictly
    after 30s and one that half-opens early: the first strands a recovered provider for a
    request or forever, the second keeps hammering a provider that asked for time, and a
    single-sided case cannot tell either from the intended `>=`."""
    clock = ManualClock()
    breaker = opened_breaker(clock)

    clock.now = elapsed

    assert breaker.state(PROVIDER) is expected_state
    assert breaker.allows(PROVIDER) is expected_allows


def test_the_open_state_survives_the_failure_window_emptying() -> None:
    """Catches a breaker whose state is only ever derived from the failures still inside
    the window: 61 seconds after it opened, every failure has aged out, so such an
    implementation reads CLOSED and sends full traffic to a provider that has passed no
    trial at all. Only a trial resolves an open breaker, and the state here is the
    HALF_OPEN one that offers exactly one request."""
    clock = ManualClock()
    breaker = opened_breaker(clock)

    clock.now = FAILURE_WINDOW_SECONDS + 1.0

    assert breaker.state(PROVIDER) is CircuitState.HALF_OPEN


# --------------------------------------------------------------------------
# resolving the trial
# --------------------------------------------------------------------------


def test_one_trial_success_closes_it_and_clears_the_count() -> None:
    """Catches two failures a single "state is CLOSED" assertion would miss. A breaker
    that needs several successes to close leaves a recovered provider at one request per
    30 seconds; one that closes but keeps the old failure timestamps re-opens on the very
    next failure inside the window, which is why four more failures follow here - a
    breaker that had not cleared its count would be back OPEN at the end."""
    clock = ManualClock()
    breaker = opened_breaker(clock)

    clock.now = HALF_OPEN_AFTER_SECONDS
    assert breaker.allows(PROVIDER) is True, "the trial request was never offered"
    breaker.record_success(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.CLOSED

    for _ in range(MAX_FAILURES - 1):
        breaker.record_failure(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.CLOSED, "the old failures still counted"
    assert breaker.allows(PROVIDER) is True


def test_a_failure_while_open_does_not_push_the_recovery_out() -> None:
    """Catches a breaker that treats every failure arriving after it opened as a failed
    trial. Such failures are ordinary under concurrency - several worker threads pass the
    health seam together, and the ones behind the fifth land after the breaker opened - so
    a burst would restart the thirty seconds once per thread and recovery would take
    arbitrarily longer than the boundary promises. A trial exists only in HALF_OPEN."""
    clock = ManualClock()
    breaker = opened_breaker(clock)

    clock.now = HALF_OPEN_AFTER_SECONDS - 0.1
    breaker.record_failure(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.OPEN

    clock.now = HALF_OPEN_AFTER_SECONDS

    assert breaker.state(PROVIDER) is CircuitState.HALF_OPEN, "the 30 seconds was pushed out"


def test_a_success_does_not_clear_a_closed_breakers_window() -> None:
    """The difference between "five failures in sixty seconds" and "five CONSECUTIVE
    failures", which every other case in this file is blind to because none of them
    interleaves a success. A breaker that cleared its count on any success would never
    open on the most ordinary outage there is - a provider degraded to a partial error
    rate, which serves just often enough to reset the counter - and every request would
    go on paying that provider's retries and read timeout before falling back. Four
    failures, a success, then a fifth failure, all inside the window."""
    clock = ManualClock()
    breaker = CircuitBreaker(clock=clock)

    for _ in range(MAX_FAILURES - 1):
        breaker.record_failure(PROVIDER)
    breaker.record_success(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.CLOSED, "four failures must not open it"

    breaker.record_failure(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.OPEN
    assert breaker.allows(PROVIDER) is False


def test_one_trial_failure_reopens_it_and_restarts_the_thirty_seconds() -> None:
    """Catches a breaker that needs `MAX_FAILURES` again to re-open after a failed trial,
    which would send five requests into a provider that just refused its one trial, and a
    breaker that keeps the ORIGINAL opening time - it would half-open again immediately
    instead of 30 seconds after the failed trial, turning the backoff into no backoff. The
    trial happens after the window has emptied, so the ONE failure is the only thing that
    can open it here and the count cannot do it for the code under test."""
    clock = ManualClock()
    breaker = opened_breaker(clock)

    clock.now = FAILURE_WINDOW_SECONDS + 1.0
    assert breaker.state(PROVIDER) is CircuitState.HALF_OPEN
    breaker.record_failure(PROVIDER)

    assert breaker.state(PROVIDER) is CircuitState.OPEN
    assert breaker.allows(PROVIDER) is False

    clock.now = FAILURE_WINDOW_SECONDS + 1.0 + HALF_OPEN_AFTER_SECONDS - 0.1
    assert breaker.state(PROVIDER) is CircuitState.OPEN, "the 30 seconds did not restart"

    clock.now = FAILURE_WINDOW_SECONDS + 1.0 + HALF_OPEN_AFTER_SECONDS
    assert breaker.state(PROVIDER) is CircuitState.HALF_OPEN
