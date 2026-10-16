r"""One tenant's requests and tokens per minute: two buckets, and what they decide (#28).

the design asks for a token bucket per tenant for `rpm` and `tpm`, and a bucket is
not a window. A fixed-window counter lets a tenant send twice its ceiling across
the boundary - the whole of one window at its end and the whole of the next at
its start - and it makes a refused tenant wait up to a whole minute for an
allowance that a bucket hands back continuously. So the state here is a level and
a timestamp, the level refills at `ceiling / SECONDS_PER_MINUTE` tokens a second,
and a tenant refused at t is served again once one whole token has landed
(`test_a_refill_landing_mid_sequence_serves_the_request_that_follows_it` in
tests/limits/test_ratelimit.py).

Four things are decisions rather than mechanics.

THE CLOCK IS THE WALL CLOCK, never `time.monotonic()`.
Replicas share one bucket in Redis, and their monotonic origins differ, so a
timestamp written by one replica is meaningless to the next - it would read as
the future or as hours ago depending on which process booted first. `refill`
therefore ignores a clock that went backwards rather than subtracting tokens for
it (the two backwards rows of `test_the_refill_table`).

CONSUMPTION IS FLOORED AT ZERO AND HAPPENS ON A REFUSED TAKE TOO. Both halves are
forced. A bucket that consumed nothing on a refusal could never be drained by a
token debit larger than its level, so `tpm` would keep a residue that passes the
next check for ever
(`test_an_over_large_debit_empties_the_bucket_rather_than_bouncing_off_it`); a
bucket with no floor sinks further with every retry, so a client retrying at ten
times its limit is locked out for minutes rather than for the fraction of a
second criterion 3 promises
(`test_a_refused_take_never_drives_the_bucket_below_zero`).

A STORE THAT CANNOT ANSWER SERVES THE REQUEST. `take`
answers `None` for "unknown" rather than raising or refusing, and the route reads
unknown as serve-and-log. Failing closed would turn a Redis blip into a total
outage on traffic every provider would have served, which is the one failure a
rate limit must not cause; refusing silently would look identical to a limit
being enforced. This is `limits/budget.py`'s decision B applied to a bucket, and
it is written here so nobody later "fixes" it
(`test_an_unreachable_redis_reads_unknown_rather_than_raising` in
tests/limits/test_redis_ratelimit.py,
`test_an_unreachable_limiter_serves_and_refuses_nobody` in
tests/gateway/test_rate_limit_enforcement.py). Serving is not saying nothing:
every swallowed failure is logged at ERROR with the tenant and the dimension,
because decision F adds no metric family and the log is then the ONLY place the
window in which a ceiling stopped being enforced is visible
(`test_every_swallowed_failure_is_logged_with_the_tenant_and_the_dimension`).

THE REDIS BUCKET IS ONE LUA SCRIPT. Refill, compare and
consume in three commands lets two replicas both spend the last token, and a
`INCR`/`EXPIRE` pair is a window rather than a bucket. One registered script does
all three in one round trip, and it adds no dependency: `register_script` is
`redis-py`'s own
(`test_the_script_is_registered_once_and_run_once_per_take`).

The tenant segment is pre-checked before any command is sent,
and the check lives in `bucket_key` rather than in each store method, because the
key is the one thing every command needs first: there is no order in which a
command reaches Redis with an unchecked tenant
(`test_a_take_refuses_a_malformed_tenant_before_touching_redis`). The dimension
segment is not client input at all - it is a closed `StrEnum` this module owns
(`test_the_dimension_is_a_closed_enum_and_not_a_string`).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

import redis

from app.core.validation import checked_identifier

RATE_LIMIT_KEY_PREFIX: Final[str] = "ratelimit"
"""The first segment of every key a tenant's bucket is stored under.

A running gateway's buckets are addressed by this string, so renaming it hands
every tenant in flight a full allowance at the moment of a deploy; and a prefix
equal to the budget counter's would put `INCRBY` and a token bucket on one key.
Pinned by `test_the_key_prefix_is_the_published_one`."""

RATE_LIMIT_KEY_TTL_SECONDS: Final[int] = 3600
"""How long an idle bucket is kept, in seconds: one hour.

Both wrong values are silent. Absent, and two keys per tenant accumulate for
ever; shorter than the time a bucket needs to refill, and a key expires while its
tenant is still being limited by it, which hands that tenant a full allowance
mid-minute. An empty bucket refills to capacity in exactly
`SECONDS_PER_MINUTE`, so an hour is sixty times the longest a bucket can hold
anything but a full level: a key can only ever expire once the bucket it held was
full anyway, which is what makes expiry equivalent to keeping it. Pinned by
`test_the_bucket_key_ttl_is_pinned`, and read back off a live server by
`test_the_bucket_expires_after_exactly_the_published_ttl`."""

SECONDS_PER_MINUTE: Final[int] = 60
"""What `rpm` and `tpm` are per, and therefore what the refill rate is divided by.

Criterion 3's `rpm/60` and `tpm/60`. Pinned by
`test_the_refill_rate_is_the_ceiling_over_one_minute`."""

MIN_RETRY_AFTER_SECONDS: Final[int] = 1
"""The smallest wait a refusal may report.

`Retry-After` is defined in whole seconds, so a true wait of half a second has to
be reported as one: `Retry-After: 0` tells a client to retry immediately, which
is the retry storm the header exists to prevent. Pinned by
`test_the_retry_after_floor_is_a_whole_second`."""


class RateDimension(StrEnum):
    """Which of a tenant's two ceilings a take is against.

    A `StrEnum` and not a plain `Enum`: the member is interpolated into the key,
    and a plain enum would render `ratelimit:evals:RateDimension.REQUESTS` - a
    key that works, is stable, and is not the specced one. A closed set and not a
    `str` parameter, because this is the second segment of a Redis key and must
    never be reachable from client input
    (`test_the_dimension_is_a_closed_enum_and_not_a_string`).
    """

    REQUESTS = "req"
    TOKENS = "tok"


@dataclass(frozen=True, slots=True)
class Taken:
    """What one take answers: whether it was allowed, and when to come back if not.

    Two fields rather than a bool, because a refusal that will not say when to
    retry makes every client guess (decision D). `retry_after_seconds` is `0` on
    an allowed take and is what the route puts on the 429 and on no other
    response (`test_a_take_answers_no_retry_after_when_it_is_allowed`,
    `test_a_served_response_carries_no_retry_after`).

    Frozen so a caller cannot edit an answer the store already decided, and
    comparable so a case can assert the whole answer at once.
    """

    allowed: bool
    retry_after_seconds: int


_log = logging.getLogger(__name__)


def bucket_key(
    tenant: str, dimension: RateDimension, *, prefix: str = RATE_LIMIT_KEY_PREFIX
) -> str:
    """The key one tenant's bucket for one dimension is kept under, tenant checked.

    `{prefix}:{tenant}:{req|tok}` (`test_the_bucket_key_is_the_specced_shape`).
    The tenant is a client-influenced string that shapes a key, and a colon in it
    would forge a segment and address another tenant's bucket, so it is checked
    against the gateway's OWN `IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH` -
    imported rather than restated, so the name a bucket accepts and the name
    `gateway/context.py` authenticated can never disagree. A rejected value
    raises `IdentifierInvalid` and no command is sent at all, which is what the
    tests assert rather than "no key exists afterwards"
    (`test_a_take_refuses_a_malformed_tenant_before_touching_redis`,
    `test_the_tenant_length_boundary_is_inclusive`).

    The dimension needs no check: it is a member of a closed enum this module
    owns, so it cannot carry a separator or a glob character
    (`test_the_dimension_is_a_closed_enum_and_not_a_string`).

    `prefix` exists for the suite: per-test isolation against a live Redis is a
    namespace rather than a `FLUSHDB` that would delete keys the suite does not
    own.
    """
    checked_identifier(tenant, "tenant")
    return f"{prefix}:{tenant}:{dimension.value}"


def refill(
    *, tokens: float, last_seen: float, rate_per_second: float, capacity: float, now: float
) -> float:
    """The level a bucket has climbed to by `now`, clamped at `capacity`.

    Pure, so criterion 3's continuous refill is settled by a table rather than by
    a sleeping test (`test_the_refill_table`). Two properties beyond the
    arithmetic:

    The clamp is what makes `capacity` mean anything. Without it an idle tenant
    accumulates a day of tokens and spends its whole allowance in one burst,
    which is the ceiling not being a ceiling.

    Elapsed time is floored at zero, so a clock that went BACKWARDS adds nothing
    rather than subtracting. That is not a hypothetical: replicas share one
    bucket and read their own wall clocks, so a timestamp written by a replica
    running a second ahead reads as the future to the next one, and a signed
    elapsed time would take tokens from a tenant that spent nothing.
    """
    elapsed = max(now - last_seen, 0.0)
    return min(tokens + elapsed * rate_per_second, capacity)


def retry_after(*, tokens: float, cost: int, rate_per_second: float) -> int:
    """Whole seconds until this bucket holds `cost` tokens, rounded UP (criterion 7).

    `ceil` and never `round` or `int`: a wait of 3.33 seconds rounds to 3 under
    both, so a client that obeys the header comes back a third of a second early
    and is refused again - a retry storm the gateway itself scheduled. Floored at
    `MIN_RETRY_AFTER_SECONDS` for the reason that constant carries
    (`test_the_retry_after_is_the_seconds_to_the_next_whole_token`,
    `test_the_retry_after_floor_is_a_whole_second`).

    Computed from the bucket's OWN rate, so two buckets refilling at different
    rates report different waits (decision D,
    `test_a_refused_take_carries_the_wait_for_its_own_bucket`). A rate of zero has
    no arrival to wait for and answers the floor rather than raising, because this
    runs inside a request that is already being refused.
    """
    if rate_per_second <= 0:
        return MIN_RETRY_AFTER_SECONDS
    short = max(cost - tokens, 0.0)
    return max(math.ceil(short / rate_per_second), MIN_RETRY_AFTER_SECONDS)


class RateLimitStore(Protocol):
    """Take `cost` tokens from one tenant's bucket, and say what that means.

    One method, and one rule: `None` means UNKNOWN, and it is the only thing a
    store that cannot answer may say. Not an exception, which would make a Redis
    blip a 5xx on traffic every provider would have served; and not
    `Taken(allowed=False, ...)`, which fails closed silently - every tenant
    refused for as long as the incident lasts, with nothing in the answer to say
    a limit was never consulted (decision E, and `BudgetStore`'s `None` for the
    same reason).

    The three numbers are keyword-only on purpose: `take(t, d, 1, 120, 2.0)` and
    `take(t, d, 120, 1, 2.0)` are both well typed and only one of them is a rate
    limit.

    A take CONSUMES whether or not it was allowed, floored at zero - the module
    docstring argues both halves.
    """

    def take(
        self, tenant: str, dimension: RateDimension, *, cost: int, capacity: int, rate: float
    ) -> Taken | None:
        """Spend `cost` from this bucket and answer the verdict, or `None` if unknown."""
        ...


class InProcessRateLimitStore:
    """A `RateLimitStore` keeping every bucket in one process's memory.

    What `create_app` keeps when `OPSCONSOLE_REDIS_URL` is unset, exactly as the
    response cache, the directory store and the budget counter do, so the suite
    and a local run need no backing service
    (`test_create_app_keeps_the_in_process_rate_limit_store_without_a_redis_url`).
    Two gateway replicas do not share a bucket this way, so each would grant the
    full `rpm`: acceptable for a single G0 node, and the reason
    `RedisRateLimitStore` is what ships to the cluster.

    It never answers `None` (`test_the_in_process_twin_never_answers_unknown`): a
    dictionary cannot be unreachable, so "unknown" has no meaning here, and a
    route that saw `None` from it would have a limiter that is wired and enforces
    nothing.

    `clock` is a constructor argument so the suite can freeze it - "half a token
    has refilled" is arranged by handing the store a different number, never by
    sleeping - and it defaults, because `create_app` builds this with no
    arguments (`test_the_twin_is_constructible_with_no_clock`).
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def take(
        self, tenant: str, dimension: RateDimension, *, cost: int, capacity: int, rate: float
    ) -> Taken | None:
        """Refill this bucket, decide, then consume - the order the module docstring sets.

        A bucket nobody has taken from starts FULL, at `capacity`: a tenant whose
        first request of the day was refused would be a limiter that had counted
        traffic that never happened.
        """
        key = bucket_key(tenant, dimension)
        now = self._clock()
        stored = self._buckets.get(key)
        held, last_seen = (float(capacity), now) if stored is None else stored
        level = refill(
            tokens=held,
            last_seen=last_seen,
            rate_per_second=rate,
            capacity=float(capacity),
            now=now,
        )
        allowed = level >= cost
        wait = 0 if allowed else retry_after(tokens=level, cost=cost, rate_per_second=rate)
        self._buckets[key] = (max(level - cost, 0.0), now)
        return Taken(allowed=allowed, retry_after_seconds=wait)


TAKE_SCRIPT: Final[str] = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])
local floor_wait = tonumber(ARGV[6])

local held = redis.call('HMGET', KEYS[1], 'tokens', 'seen')
local tokens = tonumber(held[1])
local seen = tonumber(held[2])
if tokens == nil or seen == nil then
  tokens = capacity
  seen = now
end

local elapsed = now - seen
if elapsed < 0 then elapsed = 0 end
tokens = tokens + elapsed * rate
if tokens > capacity then tokens = capacity end

local allowed = 0
if tokens >= cost then allowed = 1 end

local short = cost - tokens
if short < 0 then short = 0 end

tokens = tokens - cost
if tokens < 0 then tokens = 0 end

redis.call('HSET', KEYS[1], 'tokens', string.format('%.17g', tokens),
           'seen', string.format('%.17g', now))
redis.call('EXPIRE', KEYS[1], ttl)

local wait = 0
if allowed == 0 then
  if rate <= 0 then
    wait = floor_wait
  else
    wait = math.ceil(short / rate)
    if wait < floor_wait then wait = floor_wait end
  end
end
return {allowed, wait}
"""
r"""Refill, decide, consume and expire, in one round trip (decision A).

The whole bucket is here rather than in Python because three commands are not
one: two replicas reading a level, both deciding it is enough and both writing
back have each spent the last token, and no single-replica test can ever see it
(`test_the_script_is_registered_once_and_run_once_per_take` is what pins the
count that makes the race impossible).

It is a line-by-line twin of `refill`, the `>=` comparison, `retry_after` and the
floored consumption above, because the enforcement suite runs against
`InProcessRateLimitStore` and is only evidence about THIS script while the two
answer identically - which is what
`test_the_two_stores_answer_identically_over_one_interleaved_sequence` asserts
against a live server.

The level is written with `string.format('%.17g', ...)` rather than handed over
as a Lua number: Redis renders a number argument with `%.14g`, which silently
drops the last three significant digits of a double, and a bucket that loses
precision on every take diverges from the twin the differential compares it to.

A bucket that does not exist starts FULL, for the reason
`InProcessRateLimitStore.take` gives, and the expiry is re-sent on every take:
a TTL applied only when the key was created would leave every bucket immortal
whenever that one call was the command that failed
(`test_a_second_take_does_not_drop_the_expiry`)."""


class RedisRateLimitStore:
    """A `RateLimitStore` keeping every bucket in Redis, where all replicas share one.

    The implementation the gateway ships with once `OPSCONSOLE_REDIS_URL` is set
    (`test_create_app_uses_the_redis_rate_limit_store_when_a_redis_url_is_configured`).
    It differs from the in-process twin in three ways that matter.

    The arithmetic is `TAKE_SCRIPT`, registered ONCE at construction through
    `redis-py`'s own `register_script`, so no dependency is added and the script
    body is not shipped on every request
    (`test_the_script_is_registered_once_and_run_once_per_take`). What the script
    decides is what this store answers - it runs no arithmetic of its own, which
    would be a second bucket in the process disagreeing with the shared one
    (`test_the_scripts_answer_becomes_the_taken`).

    Every `redis.RedisError` is swallowed and becomes `None` - decision E, argued
    in the module docstring. The catch is on the base class and not on
    `ConnectionError`: a timeout, an authentication failure and the `NOSCRIPT` a
    flushed script cache answers are all "the limiter cannot answer", and this
    store meets the last of those in production before it meets a dropped
    connection (the four rows of
    `test_an_unreachable_redis_reads_unknown_rather_than_raising`). A reply the
    script could not have produced is unknown for the same reason
    (`test_an_answer_the_script_could_not_have_produced_reads_unknown`).

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, so a parametrised annotation fails `mypy --strict` with `type-arg`.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        key_prefix: str = RATE_LIMIT_KEY_PREFIX,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._key_prefix = key_prefix
        self._clock = clock
        self._script = client.register_script(TAKE_SCRIPT)

    def key(self, tenant: str, dimension: RateDimension) -> str:
        """The Redis key one bucket lives under, tenant checked before any command."""
        return bucket_key(tenant, dimension, prefix=self._key_prefix)

    def take(
        self, tenant: str, dimension: RateDimension, *, cost: int, capacity: int, rate: float
    ) -> Taken | None:
        """Run the script against this bucket and answer what it decided.

        The clock reading is passed IN rather than read from Redis's own `TIME`:
        a bucket timestamped by the server and refilled against it would be
        correct, but the twin could not then be the same function, and the
        differential is the only evidence this script has.
        """
        stored = self.key(tenant, dimension)
        try:
            answer = self._script(
                keys=[stored],
                args=[
                    capacity,
                    rate,
                    cost,
                    self._clock(),
                    RATE_LIMIT_KEY_TTL_SECONDS,
                    MIN_RETRY_AFTER_SECONDS,
                ],
            )
        except redis.RedisError:
            _log.exception(
                "rate limit for tenant %s could not be taken on %s, serving unlimited",
                tenant,
                dimension.value,
                extra={"tenant": tenant, "dimension": dimension.value},
            )
            return None
        return _as_taken(answer, tenant=tenant, dimension=dimension)


def _as_taken(answer: object, *, tenant: str, dimension: RateDimension) -> Taken | None:
    """Read the script's two-element reply, or report unknown.

    A malformed reply is a corrupted or replaced script rather than an outage,
    and it persists until somebody redeploys, so every request for as long as it
    lasts would otherwise be decided by whatever `bool()` made of the reply. It
    is read as UNKNOWN and logged like any other swallowed failure, never
    unpacked optimistically: `allowed, wait = answer` on a one-element list is a
    `ValueError` from the middle of a served request, with no typed code
    (`test_an_answer_the_script_could_not_have_produced_reads_unknown`).
    """
    if not isinstance(answer, Sequence) or isinstance(answer, str | bytes) or len(answer) != 2:
        _log.error(
            "rate limit script answered %r for tenant %s on %s, serving unlimited",
            answer,
            tenant,
            dimension.value,
            extra={"tenant": tenant, "dimension": dimension.value},
        )
        return None
    return Taken(allowed=bool(int(answer[0])), retry_after_seconds=int(answer[1]))
