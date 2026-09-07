"""One per-provider circuit breaker: when to stop dialling a provider that is failing.

It lives beside the adapters rather than in `routing/` because what it observes is
the outcome of a provider CALL. `routing/resolver.py` stays a pure function over a
policy snapshot, which is what makes `test_eu_tenant_never_gets_non_eu_provider` a
proof rather than a sample; a breaker holding mutable state inside that function
would end that. The routes join the two through the seam the resolver already
takes: `is_healthy` becomes "an adapter exists AND this breaker allows it", so an
open provider is invisible to a resolution in exactly the way a missing adapter
already is, and residency is still decided before health.

**The three numbers are SPEC section 6's**, and each is pinned exactly by
`test_the_three_numbers_are_the_spec_ones_exactly`. Five failures inside a rolling
sixty seconds open it; thirty seconds later it offers one trial request; the trial
closes it or re-opens it.

**Why an open breaker is resolved only by a trial.** State is not derived from the
failures still inside the window. If it were, a breaker that opened at t=0 would
read closed again at t=61 with nothing having recovered, and full traffic would go
back to a provider that has passed nothing. So opening records a time and clears
the count, and only `record_success` closes it
(`test_the_open_state_survives_the_failure_window_emptying`,
`test_one_trial_failure_reopens_it_and_restarts_the_thirty_seconds`).

**Why the clock is an argument.** The half-open boundary is thirty seconds, so a
breaker built on `time.monotonic` could only be tested by sleeping through it. The
default is still `time.monotonic`, which is what a wall clock's jumps make
necessary: an NTP correction must not half-open every breaker in the process.

**Why the state is per process.** a design note. Sharing it through Redis would give
the breaker a dependency of its own, and a Redis outage would then either fail
open (no breaker at all) or fail closed (no traffic at all), both of which are
worse than each replica learning within five requests of its own that a provider
is down. The gateway runs one replica today; the ADR says what changes when it
does not.

**Half-open is not serialised.** Two threads that both read HALF_OPEN are both
let through as the trial, so a recovering provider can get two requests instead of
one. Accepted: the alternative is holding a lock across a sixty-second network
call. The lock below guards the RECORD only, never a provider call.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum

MAX_FAILURES = 5
"""Failures inside `FAILURE_WINDOW_SECONDS` that open a provider's breaker."""

FAILURE_WINDOW_SECONDS = 60.0
"""How far back a failure still counts. Older ones are pruned on every read, so
the record cannot grow without bound and no background task is needed."""

HALF_OPEN_AFTER_SECONDS = 30.0
"""How long an open breaker refuses before it offers one trial request."""


class CircuitState(IntEnum):
    """What `gateway_circuit_state` publishes for one provider.

    The values are the gauge's, not an implementation detail: an operator's alert
    is written against `== 2`, so an auto-numbered enum would make it fire on a
    healthy provider (`test_the_state_values_are_the_ones_the_gauge_publishes`).
    """

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


@dataclass
class _Record:
    """One provider's failure history, or the time its breaker opened.

    The two are exclusive by construction: opening clears the failures, and the
    count is only ever consulted while `opened_at` is `None`.
    """

    failures: deque[float] = field(default_factory=deque)
    opened_at: float | None = None


class CircuitBreaker:
    """Per-provider open/closed state, keyed by the provider names in the policy.

    Every key here comes from a policy file that was schema-validated on load, so
    no client-supplied string reaches this map.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._records: dict[str, _Record] = {}

    def state(self, provider: str) -> CircuitState:
        """This provider's state right now, which is what the gauge reads.

        A provider nothing is known about is CLOSED: a breaker that denied what it
        had no record of would be an allow list, and a fresh process would serve
        nothing until something failed (`test_an_unknown_provider_is_allowed`).
        """
        with self._lock:
            record = self._records.get(provider)
            if record is None:
                return CircuitState.CLOSED
            return self._state(record, self._clock())

    def allows(self, provider: str) -> bool:
        """Whether a route may dial this provider. False only while OPEN."""
        return self.state(provider) is not CircuitState.OPEN

    def record_success(self, provider: str) -> None:
        """Resolve a TRIAL. A success on a closed breaker changes nothing at all.

        One success closes an open breaker and clears the count behind it, so a
        recovered provider is not left at one request per thirty seconds and the
        next failure inside the old window cannot re-open it immediately
        (`test_one_trial_success_closes_it_and_clears_the_count`).

        A success while the breaker is CLOSED deliberately does not touch the
        window, and that is the difference between "five failures in sixty
        seconds" and "five CONSECUTIVE failures". A provider degraded to an 80%
        error rate serves one request in five, so a breaker that cleared its count
        on every success would never open on the most ordinary kind of outage, and
        every request would keep paying that provider's retries and read timeout
        before falling back (`test_a_success_does_not_clear_a_closed_breakers_window`).
        """
        with self._lock:
            record = self._records.get(provider)
            if record is not None and record.opened_at is not None:
                del self._records[provider]

    def record_failure(self, provider: str) -> None:
        """Book one failure, opening the breaker if it is the fifth in the window.

        A failure while the breaker is HALF_OPEN is the trial's: it re-opens from
        NOW rather than keeping the original opening time, which is what makes the
        backoff a backoff
        (`test_one_trial_failure_reopens_it_and_restarts_the_thirty_seconds`).

        A failure while the breaker is still OPEN is neither, and is deliberately
        ignored. Such a failure exists - several worker threads can pass the health
        seam together, and the ones behind the fifth land after it opened - and
        letting each of them re-open would push the half-open moment further out
        every time, so a burst of concurrent failures would make recovery take
        longer than the thirty seconds the boundary promises
        (`test_a_failure_while_open_does_not_push_the_recovery_out`).
        """
        now = self._clock()
        with self._lock:
            record = self._records.setdefault(provider, _Record())
            if record.opened_at is not None:
                if self._state(record, now) is CircuitState.HALF_OPEN:
                    self._open(record, now)
                return
            self._prune(record, now)
            record.failures.append(now)
            if len(record.failures) >= MAX_FAILURES:
                self._open(record, now)

    @staticmethod
    def _open(record: _Record, now: float) -> None:
        """Take one record to OPEN at `now`, discarding the failures behind it."""
        record.opened_at = now
        record.failures.clear()

    @staticmethod
    def _prune(record: _Record, now: float) -> None:
        """Drop the failures that have aged out of the rolling window."""
        while record.failures and now - record.failures[0] >= FAILURE_WINDOW_SECONDS:
            record.failures.popleft()

    @staticmethod
    def _state(record: _Record, now: float) -> CircuitState:
        """Read one record's state without touching it, the window applied."""
        if record.opened_at is not None:
            if now - record.opened_at >= HALF_OPEN_AFTER_SECONDS:
                return CircuitState.HALF_OPEN
            return CircuitState.OPEN
        return CircuitState.CLOSED
