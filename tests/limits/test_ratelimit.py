r"""`limits/ratelimit.py`: the bucket key, the refill arithmetic, the take and the retry-after.

Library only. Nothing here builds an app, sends a request or touches Redis - the
route wiring is tests/gateway/test_rate_limit_enforcement.py and the Lua script's
Redis semantics are tests/limits/test_redis_ratelimit.py. What this file pins is
the arithmetic and the validation that decide, for one bucket and one clock,
whether a request is served and when a refused one may come back.

NO TEST HERE SLEEPS. The clock is a parameter of `refill` and a constructor
argument of both stores, so "half a token has refilled" is arranged by handing
the store a different number, never by waiting for one. A rate-limit suite that
sleeps is a tax on every future session (CLAUDE.md's conventions), and a sleeping
test also cannot express "exactly one whole token", only "at least".

The contract these cases are written against, before it exists:

* `RateDimension` - a closed `StrEnum` with exactly `REQUESTS = "req"` and
  `TOKENS = "tok"`. It is an enum rather than a string because the dimension is
  the second half of a Redis key and must never be reachable from client input.
* `bucket_key(tenant, dimension, *, prefix=RATE_LIMIT_KEY_PREFIX)` -
  `ratelimit:{tenant}:{req|tok}`, raising `IdentifierInvalid` for a tenant that
  may not shape a key. The check lives in the key function rather than in each
  store method, so "before any command is sent" is a property of the only thing
  a command needs.
* `refill(*, tokens, last_seen, rate_per_second, capacity, now) -> float` - pure,
  clamped at `capacity`, and monotone: a clock that went backwards adds nothing.
* `retry_after(*, tokens, cost, rate_per_second) -> int` - whole seconds, rounded
  UP, never below `MIN_RETRY_AFTER_SECONDS`.
* `Taken` - what a take answers: `allowed: bool` and `retry_after_seconds: int`,
  the latter `0` on an allowed take.
* `RateLimitStore` - `take(tenant, dimension, *, cost, capacity, rate) -> Taken |
  None`, `None` meaning "cannot answer" exactly as `BudgetStore`'s `None` does.
  The three numbers are keyword-only: `take(t, d, 1, 120, 2.0)` and
  `take(t, d, 120, 1, 2.0)` are both well-typed and only one of them is a rate
  limit.
* `InProcessRateLimitStore(clock=time.time)` and
  `RedisRateLimitStore(client, *, key_prefix=RATE_LIMIT_KEY_PREFIX,
  clock=time.time)`.
* `RATE_LIMIT_KEY_PREFIX`, `RATE_LIMIT_KEY_TTL_SECONDS`,
  `MIN_RETRY_AFTER_SECONDS`, `SECONDS_PER_MINUTE` - the constants a project rule
  requires a test to pin exactly.

THE ONE SEMANTIC WORTH READING BEFORE THE ASSERTIONS, because two plausible
token buckets differ on it and the criteria pick one:

    refill first; `allowed = tokens >= cost`; then `tokens = max(tokens - cost, 0)`.

Consumption is floored at zero and happens whether or not the take was allowed.
That floor is not decoration. A bucket that subtracts on a REFUSED take sinks
further with every retry, so a client that retries at ten times its limit is
locked out for ever rather than for the fraction of a second criterion 3
promises; and a bucket that consumes NOTHING on a refused take can never be
drained by an over-large token debit, so `tpm` would leak a residue that keeps
the next check passing for ever. `debit_tokens` books a real usage through the
same `take` and ignores the answer, which is decision B: the request that
crosses `tpm` completes and the next one is refused.

The tenant is checked against the gateway's OWN `IDENTIFIER_PATTERN` and
`MAX_IDENTIFIER_LENGTH`, imported rather than restated, so the name a bucket
accepts and the name `gateway/context.py` authenticated can never disagree.
a project rule is why the rejection table below carries a trailing newline, a
trailing CRLF and a trailing space: a check written `^...$` with `.match` accepts
the first of those and puts a caller-controlled line break into a Redis key.
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from opsconsole.limits.ratelimit import (
    MIN_RETRY_AFTER_SECONDS,
    RATE_LIMIT_KEY_PREFIX,
    RATE_LIMIT_KEY_TTL_SECONDS,
    SECONDS_PER_MINUTE,
    InProcessRateLimitStore,
    RateDimension,
    Taken,
    bucket_key,
    refill,
    retry_after,
)
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.resolver import MAX_IDENTIFIER_LENGTH

TENANT = "evals"
OTHER_TENANT = "cuan"

MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)

RPM = 120
TPM = 120_000
"""`evals`'s shipped ceilings. Written out here so the arithmetic below is over
numbers a deployment actually runs, and pinned against `routing.yaml` itself by
tests/gateway/test_rate_limit_enforcement.py rather than here."""

RATE = 2.0
"""`RPM / SECONDS_PER_MINUTE`, the request bucket's refill rate."""

FROZEN = 1_800_000_000.0
"""An arbitrary wall-clock instant. Every case that needs time to pass states the
LATER instant explicitly, so no case depends on how long it took to run."""


def frozen_store(now: list[float]) -> InProcessRateLimitStore:
    """A store whose clock reads `now[0]`, so a case advances time by assigning to it.

    A one-element list rather than a `float` because the clock is a callable the
    store keeps: rebinding a local would leave the store reading the old value.
    """
    return InProcessRateLimitStore(clock=lambda: now[0])


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------


def test_the_bucket_key_is_the_specced_shape() -> None:
    """Catches a key built in any other shape - the two segments swapped, a key with no
    dimension in it at all (which would make one bucket serve both `rpm` and `tpm`, so a
    tenant's requests and its tokens would spend the same allowance), or a prefix that
    collides with the budget counter's `budget:` or the response cache's. The prefix and
    the dimension are asserted as their own segments, so a rename that did not reach the
    key is caught too."""
    assert bucket_key(TENANT, RateDimension.REQUESTS) == f"{RATE_LIMIT_KEY_PREFIX}:{TENANT}:req"
    assert bucket_key(TENANT, RateDimension.TOKENS) == f"{RATE_LIMIT_KEY_PREFIX}:{TENANT}:tok"
    assert bucket_key(TENANT, RateDimension.REQUESTS).split(":") == [
        RATE_LIMIT_KEY_PREFIX,
        TENANT,
        "req",
    ]
    assert bucket_key(TENANT, RateDimension.REQUESTS) != bucket_key(TENANT, RateDimension.TOKENS)
    assert bucket_key(OTHER_TENANT, RateDimension.REQUESTS) != bucket_key(
        TENANT, RateDimension.REQUESTS
    )


def test_the_key_prefix_is_the_published_one() -> None:
    """Pins the prefix exactly. Catches a rename that orphans every
    bucket a running gateway is holding, which hands every tenant in flight a full
    allowance at the moment of a deploy, and catches a prefix equal to the budget
    counter's, which would have `INCRBY` and a token bucket writing the same key."""
    assert RATE_LIMIT_KEY_PREFIX == "ratelimit"
    assert RATE_LIMIT_KEY_PREFIX != "budget"


REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_TENANT),
    ("uppercase", "Evals"),
    ("a space", "ev als"),
    ("a colon, which would forge a second key segment", "ev:als"),
    ("a path separator", "ev/als"),
    ("a glob character, which would widen a later SCAN", "evals*"),
    ("a trailing newline", "evals\n"),
    ("a trailing CRLF", "evals\r\n"),
    ("a trailing space", "evals "),
    ("a leading space", " evals"),
    ("a leading separator", "-evals"),
]


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_a_malformed_tenant_is_refused_by_the_key_function(label: str, tenant: str) -> None:
    """Catches a key function that interpolates a client-supplied tenant straight into the
    key: `ev:als` addresses the `als` tenant's `req` bucket under a dimension named `ev`,
    and `evals\\n` puts a caller-controlled line break into a Redis key. The refusal is in
    the key function, which is what every command needs first, so there is no order in
    which a command could be sent with one of these. "No command was
    sent" is asserted against a recording client in
    tests/limits/test_redis_ratelimit.py, which is where a client exists at all. The
    trailing-newline and trailing-CRLF rows are a project rule's: a check written
    `^...$` with `.match` accepts the first."""
    with pytest.raises(IdentifierInvalid):
        bucket_key(tenant, RateDimension.REQUESTS)


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_the_token_dimension_refuses_the_same_tenants(label: str, tenant: str) -> None:
    """The other dimension of the same check. Catches a pre-check written into one branch
    of the key function - a `if dimension is RateDimension.REQUESTS` above the validation
    - which would leave `tpm` keys forgeable while `rpm` keys were safe, and there is no
    reading of the rejection table that applies to one dimension and not the other."""
    with pytest.raises(IdentifierInvalid):
        bucket_key(tenant, RateDimension.TOKENS)


def test_the_tenant_length_boundary_is_inclusive() -> None:
    """Catches an off-by-one at the cap in either direction - a check written `>=` that
    refuses a tenant of exactly the permitted length, which `gateway/context.py` already
    authenticated so the request would 500 rather than being rate limited, and a check
    written against a constant of this module's own rather than the gateway's, which
    would let a tenant one character over the gateway's cap mint a bucket."""
    assert (
        bucket_key(MAX_TENANT, RateDimension.REQUESTS)
        == f"{RATE_LIMIT_KEY_PREFIX}:{MAX_TENANT}:req"
    )
    with pytest.raises(IdentifierInvalid):
        bucket_key(OVER_CAP_TENANT, RateDimension.REQUESTS)


# --------------------------------------------------------------------------
# the dimension is not a string
# --------------------------------------------------------------------------


def test_the_dimension_is_a_closed_enum_and_not_a_string() -> None:
    """The approach's claim that the dimension "is not client input at all", asserted as
    the three things that make it true.

    Catches a `RateDimension` declared as a plain `Enum` rather than a `StrEnum`: the key
    function interpolates the member, so a plain enum renders
    `ratelimit:evals:RateDimension.REQUESTS` - a key that works, is stable, and is not
    the specced one, which no assertion about the enum's VALUES would notice. Catches
    members spelled `requests`/`tokens`, which changes every key in a running Redis.
    Catches a third member arriving later with a colon or a space in its value, which
    would forge a key segment from a name the key function never validates because it
    trusts the enum. And catches the set being opened up: `RateDimension("rpm")` must
    raise rather than mint a bucket nothing reads."""
    assert issubclass(RateDimension, StrEnum)
    assert {member.value for member in RateDimension} == {"req", "tok"}
    assert RateDimension.REQUESTS.value == "req"
    assert RateDimension.TOKENS.value == "tok"
    assert f"{RateDimension.REQUESTS}" == "req"
    assert f"{RateDimension.TOKENS}" == "tok"
    for member in RateDimension:
        assert member.value == member.value.strip()
        assert ":" not in member.value
        assert "*" not in member.value
    with pytest.raises(ValueError, match="rpm"):
        RateDimension("rpm")


# --------------------------------------------------------------------------
# the refill
# --------------------------------------------------------------------------

REFILL_ROWS: list[tuple[str, float, float, float, float]] = [
    # (label, tokens, now, capacity, expected)
    ("no time passed", 10.0, FROZEN, 120.0, 10.0),
    ("half a token", 10.0, FROZEN + 0.25, 120.0, 10.5),
    ("exactly one token", 0.0, FROZEN + 0.5, 120.0, 1.0),
    ("a long idle clamped to capacity", 0.0, FROZEN + 10_000.0, 120.0, 120.0),
    ("already full stays full", 120.0, FROZEN + 10_000.0, 120.0, 120.0),
    ("a clock that went backwards", 10.0, FROZEN - 10.0, 120.0, 10.0),
    ("a clock that went backwards on a full bucket", 120.0, FROZEN - 10.0, 120.0, 120.0),
]
"""The whole refill table at `RATE` = 2 tokens per second, `last_seen` = `FROZEN`.
Every row states the LATER instant rather than a duration, because that is what
the store hands `refill` and a duration would let a sign error pass."""


@pytest.mark.parametrize(
    ("label", "tokens", "now", "capacity", "expected"),
    REFILL_ROWS,
    ids=[row[0] for row in REFILL_ROWS],
)
def test_the_refill_table(
    label: str, tokens: float, now: float, capacity: float, expected: float
) -> None:
    """Criterion 3's continuous refill, one row each, with the clock injected.

    Catches a fixed-window counter wearing a bucket's name: "half a token" and "exactly
    one token" both answer the bucket's starting level under a window that only resets on
    the minute, so a tenant refused at t would wait up to a whole minute instead of half
    a second. Catches a refill that is not clamped, which lets an idle tenant accumulate
    ten thousand tokens and then spend its whole day's traffic in one burst - the clamp
    is the only thing that makes `capacity` mean anything. Catches a refill computed with
    `time.monotonic()` semantics, which the "clock that went backwards" row stands in
    for: two replicas share one bucket and their monotonic origins differ, so a timestamp
    written by one replica reads as the future to the next and a signed elapsed time
    would then SUBTRACT tokens from a tenant that spent nothing."""
    assert refill(
        tokens=tokens,
        last_seen=FROZEN,
        rate_per_second=RATE,
        capacity=capacity,
        now=now,
    ) == pytest.approx(expected)


def test_the_refill_rate_is_the_ceiling_over_one_minute() -> None:
    """Criterion 3's "rpm/60 and tpm/60", pinned on the constant the rates are derived
    from. Catches a rate computed per SECOND from the ceiling itself
    (120 tokens a second, sixty times the limit, so nothing is ever refused) and one
    computed per hour."""
    assert SECONDS_PER_MINUTE == 60
    assert RPM / SECONDS_PER_MINUTE == RATE
    assert TPM / SECONDS_PER_MINUTE == 2000.0


# --------------------------------------------------------------------------
# the take
# --------------------------------------------------------------------------


def test_an_empty_bucket_refuses_at_exactly_zero_tokens() -> None:
    """The inclusive boundary of the whole feature, in both directions.

    A bucket holding exactly one token serves the request that costs one; the same bucket
    one take later holds exactly zero and refuses. Catches a comparison written `>`,
    which refuses the last request a tenant paid for - the tenant is capped at 119
    requests a minute while `routing.yaml` says 120 - and catches `>= 0` or a truthiness
    test on the level, which serves for ever out of an empty bucket. The clock does not
    move between the two takes, so the refill is not what decides either answer."""
    now = [FROZEN]
    store = frozen_store(now)

    first = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)
    second = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)

    assert first is not None and first.allowed is True
    assert second is not None and second.allowed is False


def test_a_full_bucket_serves_exactly_its_capacity_and_no_more() -> None:
    """The ceiling itself, at a capacity small enough to exhaust in a test. Catches an
    off-by-one that serves `capacity + 1` (the bucket starts at `capacity` and the first
    take is free) and one that serves `capacity - 1`. The clock is frozen, so nothing
    refills mid-run and the count is exactly the capacity."""
    now = [FROZEN]
    store = frozen_store(now)

    served = 0
    for _ in range(6):
        taken = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=4, rate=RATE)
        assert taken is not None
        served += int(taken.allowed)

    assert served == 4


def test_a_refused_take_never_drives_the_bucket_below_zero() -> None:
    """The floor, which is what keeps a refusal from compounding.

    A client that retries hard is refused many times over; if each refused take still
    subtracted its cost, the bucket would be at -20 after twenty retries and the tenant
    would wait ten seconds for the next token instead of the half second criterion 3
    promises - a limiter that punishes retries rather than shaping them. Catches exactly
    that: after an empty bucket is hammered, half a second of refill must be enough to
    serve again. Nothing else in this file distinguishes a floored bucket from a sinking
    one, because a sinking bucket refuses every row above just as correctly."""
    now = [FROZEN]
    store = frozen_store(now)
    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)
    for _ in range(20):
        hammered = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)
        assert hammered is not None and hammered.allowed is False

    now[0] = FROZEN + 0.5
    after = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)

    assert after is not None and after.allowed is True


def test_an_over_large_debit_empties_the_bucket_rather_than_bouncing_off_it() -> None:
    """Decision B for the token dimension: the request that CROSSES `tpm` completes, and
    the next one is refused.

    `debit_tokens` books a real usage through `take` and ignores the answer, so a take
    whose cost exceeds the level has to leave the bucket empty. Catches the natural
    reading of "a refused take consumes nothing", under which a 5000-token debit against
    a bucket holding 100 leaves those 100 in place, the next check for one token passes,
    and `tpm` is never enforced again for as long as the tenant keeps sending large
    requests. The check-sized take afterwards is what makes the emptiness observable."""
    now = [FROZEN]
    store = frozen_store(now)

    store.take(TENANT, RateDimension.TOKENS, cost=900, capacity=1000, rate=RATE)
    overdraft = store.take(TENANT, RateDimension.TOKENS, cost=5000, capacity=1000, rate=RATE)
    probe = store.take(TENANT, RateDimension.TOKENS, cost=1, capacity=1000, rate=RATE)

    assert overdraft is not None and overdraft.allowed is False
    assert probe is not None and probe.allowed is False


def test_a_take_answers_no_retry_after_when_it_is_allowed() -> None:
    """Catches a `Taken` that carries a retry-after on every answer, which the route would
    put on a 200 response's `Retry-After` header - a header telling a client to back off
    from a request that was served. Decision D says the header is on this error only, and
    this is the store-side half of that."""
    now = [FROZEN]
    store = frozen_store(now)

    taken = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=4, rate=RATE)

    assert taken == Taken(allowed=True, retry_after_seconds=0)


def test_a_refused_take_carries_the_wait_for_its_own_bucket() -> None:
    """Catches a refusal that answers a fixed retry-after regardless of the bucket, and one
    that answers the TTL. Two buckets refilling at different rates are refused in the
    same arrangement, and the slower one has to say the longer wait, which is the only
    reading of "computed from the bucket's own refill rate" (decision D) that a constant
    cannot satisfy."""
    now = [FROZEN]
    store = frozen_store(now)
    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=1.0)
    store.take(OTHER_TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=0.2)

    fast = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=1.0)
    slow = store.take(OTHER_TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=0.2)

    assert fast is not None and fast.allowed is False
    assert slow is not None and slow.allowed is False
    assert fast.retry_after_seconds == 1
    assert slow.retry_after_seconds == 5


# --------------------------------------------------------------------------
# the retry-after
# --------------------------------------------------------------------------

RETRY_ROWS: list[tuple[str, float, int, float, int]] = [
    # (label, tokens, cost, rate_per_second, expected)
    ("an empty bucket at one token a second", 0.0, 1, 1.0, 1),
    ("an empty bucket at two tokens a second, rounded up from half", 0.0, 1, 2.0, 1),
    ("an empty bucket at one token every five seconds", 0.0, 1, 0.2, 5),
    ("exactly three seconds, which must not round to four", 0.0, 1, 1.0 / 3.0, 3),
    ("a hair over three seconds, which must round to four", 0.0, 1, 0.3, 4),
    ("a partly refilled bucket owes only the remainder", 0.5, 1, 1.0, 1),
    ("a bucket 1.5 tokens short at one a second", 0.5, 2, 1.0, 2),
    ("a token bucket short by a whole request", 0.0, 500, 2000.0, 1),
    ("a bucket that already holds the cost still says come back", 5.0, 1, 1.0, 1),
]
"""The rounding table. The `1.0 / 3.0` and `0.3` rows are the boundary standing
rule 2 asks for on a rounded value: one is exactly three seconds and the other is
3.33, and `round()` answers 3 for both while `ceil` answers 3 and 4."""


@pytest.mark.parametrize(
    ("label", "tokens", "cost", "rate_per_second", "expected"),
    RETRY_ROWS,
    ids=[row[0] for row in RETRY_ROWS],
)
def test_the_retry_after_is_the_seconds_to_the_next_whole_token(
    label: str, tokens: float, cost: int, rate_per_second: float, expected: int
) -> None:
    """Criterion 7's arithmetic, in whole seconds, rounded UP.

    Catches a retry-after computed with `round()` or `int()`: both answer 3 for a wait of
    3.33 seconds, so a client that obeys the header comes back a third of a second early
    and is refused again, which is a retry storm the gateway itself scheduled. Catches a
    value that is not floored: at two tokens a second the true wait is half a second, and
    `Retry-After: 0` tells a client to retry immediately, which is the same storm without
    the arithmetic. And catches a wait computed from the CAPACITY rather than from the
    cost, which would tell a tenant to wait a whole minute for one request."""
    assert retry_after(tokens=tokens, cost=cost, rate_per_second=rate_per_second) == expected


def test_the_retry_after_floor_is_a_whole_second() -> None:
    """Pins the floor exactly, on the constant rather than through a row.
    Catches a floor of zero, which is a `Retry-After` that means "immediately" on a
    response whose whole purpose is to slow a client down; the field is defined in whole
    seconds, so any real wait under a second has to be reported as one."""
    assert MIN_RETRY_AFTER_SECONDS == 1
    assert retry_after(tokens=0.0, cost=1, rate_per_second=1000.0) == MIN_RETRY_AFTER_SECONDS


def test_the_retry_after_is_a_whole_number_of_seconds() -> None:
    """The type, not the value. `Retry-After: 0.5` is not a legal header value and a client
    parsing it as an integer fails on the response that was trying to help it. Catches a
    function annotated `-> int` that answers a float anyway, which `float.__eq__` makes
    invisible to every row of the table above (`1.0 == 1`)."""
    answer = retry_after(tokens=0.0, cost=1, rate_per_second=2.0)

    assert isinstance(answer, int)
    assert not isinstance(answer, float)


# --------------------------------------------------------------------------
# the key's expiry
# --------------------------------------------------------------------------


def test_the_bucket_key_ttl_is_pinned() -> None:
    """Pins the exact constant, because both wrong values are wrong
    quietly. Too short and a bucket expires while a tenant is still being limited by it,
    which hands that tenant a full allowance mid-minute and makes the ceiling
    unenforceable in exactly the burst it exists for; absent, and one key per tenant per
    dimension accumulates in Redis for ever. An hour is sixty times the sixty seconds an
    empty bucket needs to refill to capacity, so a key can only ever expire once the
    bucket it held was full anyway - which is what makes expiry equivalent to keeping
    it."""
    assert RATE_LIMIT_KEY_TTL_SECONDS == 3600
    assert RATE_LIMIT_KEY_TTL_SECONDS == 60 * SECONDS_PER_MINUTE
    assert RATE_LIMIT_KEY_TTL_SECONDS > SECONDS_PER_MINUTE


# --------------------------------------------------------------------------
# the window's other traffic
# --------------------------------------------------------------------------


def test_a_token_debit_between_two_request_takes_refuses_neither() -> None:
    """a project rule: the window's other traffic, interleaved.

    Every other case here spends one bucket at a time, and a store that keyed on the
    tenant alone - one bucket for `rpm` and `tpm` together - passes all of them. Here a
    five-hundred-token debit lands between two request takes against a request bucket
    holding two: with one shared bucket the second request take is refused, and with the
    specced two keys both are served. Catches exactly that key, and catches a store whose
    `dimension` argument is accepted and ignored."""
    now = [FROZEN]
    store = frozen_store(now)

    first = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=2, rate=RATE)
    store.take(TENANT, RateDimension.TOKENS, cost=500, capacity=TPM, rate=2000.0)
    second = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=2, rate=RATE)

    assert first is not None and first.allowed is True
    assert second is not None and second.allowed is True


def test_a_refill_landing_mid_sequence_serves_the_request_that_follows_it() -> None:
    """a project rule again, on the reset rather than on the other dimension: what RESETS
    this counter is the passage of time, so the case where the reset lands mid-sequence
    is the one that separates a token bucket from a fixed window.

    Two requests fill a bucket of two, the third is refused, one second refills exactly
    one token, the fourth is served and the fifth is refused again. A fixed-window
    counter passes the first three rows and then serves BOTH of the last two, because a
    window that has rolled over grants the whole allowance at once - which is the 2x
    burst across the boundary decision A rejects. A bucket that refills only when it is
    empty, or only on the first take after an idle period, fails the fourth row."""
    now = [FROZEN]
    store = frozen_store(now)
    for _ in range(2):
        served = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=2, rate=1.0)
        assert served is not None and served.allowed is True
    refused = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=2, rate=1.0)

    now[0] = FROZEN + 1.0
    after_refill = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=2, rate=1.0)
    following = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=2, rate=1.0)

    assert refused is not None and refused.allowed is False
    assert after_refill is not None and after_refill.allowed is True
    assert following is not None and following.allowed is False


def test_one_tenants_bucket_is_not_anothers() -> None:
    """Catches a key that drops the tenant segment: with one bucket for everyone, a quiet
    tenant is refused the moment a busy neighbour reaches its own ceiling - and the
    neighbour's traffic is invisible in the quiet tenant's own numbers, so nobody can
    explain the 429s. Read with a tenant that genuinely has a live bucket, so "no bucket
    yet" cannot be what makes it pass."""
    now = [FROZEN]
    store = frozen_store(now)

    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)
    other = store.take(OTHER_TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)
    same = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)

    assert other is not None and other.allowed is True
    assert same is not None and same.allowed is False


# --------------------------------------------------------------------------
# the in-process twin
# --------------------------------------------------------------------------


def test_the_in_process_twin_never_answers_unknown() -> None:
    """Catches a twin that copied the Redis one's `None` returns. It cannot fail, so a
    `None` from it would only ever be a bug that silently disables both ceilings on the
    local and suite configuration - which is every run of this suite and every bare
    `uvicorn` run. `None` means "serve, the limiter could not answer" (decision E), so a
    twin that answered it would be a limiter that is wired and enforces nothing."""
    now = [FROZEN]
    store = frozen_store(now)

    assert store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE) is not None
    assert store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE) is not None
    assert store.take(TENANT, RateDimension.TOKENS, cost=0, capacity=TPM, rate=2000.0) is not None


def test_the_twin_is_constructible_with_no_clock() -> None:
    """Catches a store whose `clock` is a required argument: `create_app` builds it with no
    arguments, so a required clock would leave no caller able to construct one and the
    shipped gateway would fail at startup rather than in a test.

    WHICH clock the default is - `time.time()` rather than `time.monotonic()`, which the
    approach chose because replicas share one bucket and their monotonic origins differ -
    is deliberately NOT asserted here, and no assertion in this repository can reach it:
    the store's timestamp is internal, and both clocks produce a bucket that behaves
    identically inside one process. It is untestable by design; what
    IS testable is that a timestamp written by one store is read by another, which is
    what the live differential in tests/limits/test_redis_ratelimit.py exercises."""
    store = InProcessRateLimitStore()

    taken = store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=1, rate=RATE)

    assert taken is not None and taken.allowed is True
