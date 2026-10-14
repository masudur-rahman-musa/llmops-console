r"""`RedisRateLimitStore`: the one round trip, the expiry, the silence, and the differential.

Two halves, and the split is the one tests/limits/test_redis_budget.py,
tests/cache/test_redis_cache.py and tests/redaction/test_redis_store.py already
make.

The half that needs NOTHING is the validation and failure half, over a client
that records its calls and refuses to answer. "No token was taken" is asserted as
"Redis was never called at all", which is the claim a project rule actually
makes, and the fail-open rows (decision E) need a client that raises rather than
a server that is down.

The half that needs a REAL Redis is the half whose claims are the Lua script's:
that refill-and-take is atomic and answers what the pure functions say it should,
that the two dimensions are genuinely separate keys, and that the expiry is on
the key - read back with `TTL`, never slept for. Redis is required by an explicit
conditional on `OPSCONSOLE_TEST_REDIS_URL`; the fixture skips
with that variable's name in the reason, so a run that checked nothing says so in
its own output. Deliberately NOT `OPSCONSOLE_REDIS_URL`, which `Settings` itself
reads: exporting that one to run this file would hand every other test in the
suite a Redis-backed gateway it never asked for, and the root `tests/conftest.py`
clears it for exactly that reason.

THE DIFFERENTIAL IS THE POINT OF THIS FILE. Section 5's second risk is that the
Lua script cannot be exercised without a live Redis and this machine has never
started the compose stack, so the in-process twin carries every enforcement case
in tests/gateway/test_rate_limit_enforcement.py. That twin is only EVIDENCE about
the script if the two answer identically for the same clock and the same
sequence, which is what `test_the_two_stores_answer_identically_over_one_interleaved_sequence`
asserts and what nothing else in the repository does.

NO TEST HERE SLEEPS. Both stores take their clock as a constructor argument and
the differential hands them the same one, so the slowest case is a handful of
round trips.

The contract these cases are written against, before it exists:

* `RedisRateLimitStore(client, *, key_prefix=RATE_LIMIT_KEY_PREFIX,
  clock=time.time)`, which registers its Lua script through
  `client.register_script(...)` ONCE, at construction (decision A: one registered
  script, no dependency added, and no `EVAL` of a whole script body per request);
* the registered script is called ONCE per `take`, with `keys=[the bucket key]`
  and args carrying the capacity, the rate, the cost, the clock reading and the
  TTL, and answers a two-element sequence read as `Taken`;
* `store.key(tenant, dimension)` renders exactly what `bucket_key` renders under
  the store's own prefix;
* every `redis.RedisError` is SWALLOWED, `take` answers `None`, and the failure
  is logged at ERROR through `app.limits.ratelimit` naming the tenant and
  the dimension.

`None` is "cannot answer", never "refused" and never "allowed" - the route reads
it as "serve, and log" (decision E). A limiter that failed CLOSED would turn a
Redis blip into a total outage on traffic every provider would have served, which
is the one thing a rate limit must not cause.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import pytest
import redis

from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.limits.ratelimit import (
    RATE_LIMIT_KEY_PREFIX,
    RATE_LIMIT_KEY_TTL_SECONDS,
    InProcessRateLimitStore,
    RateDimension,
    RedisRateLimitStore,
    Taken,
    bucket_key,
)
from app.model_routing.errors import IdentifierInvalid

REDIS_URL_VAR = "OPSCONSOLE_TEST_REDIS_URL"
REDIS_URL = os.environ.get(REDIS_URL_VAR)

RATE_LIMIT_LOGGER = "app.limits.ratelimit"
"""Where a swallowed failure is reported. Named so the assertions below count THIS
module's records and nothing else's."""

TENANT = "evals"
OTHER_TENANT = "cuan"

MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)

FROZEN = 1_800_000_000.0

CAPACITY = 4
RATE = 2.0
"""A capacity small enough to exhaust in a handful of takes and a rate that
refills half a token every quarter second, so every instant this file states is a
different bucket level."""

TOKEN_CAPACITY = 1000
TOKEN_RATE = 50.0
"""The token dimension's numbers, deliberately different from the request
dimension's in both directions: a store that swapped the two arguments, or that
read one dimension's state under the other's key, answers differently here."""


def as_client(fake: object) -> redis.Redis:
    """Hand a recording double to a store typed for a real client.

    The store is annotated against `redis.Redis`, and the cases in this half of
    the file are about calls that never leave the process; casting keeps the
    double honest about its methods without a `# type: ignore` on every call.
    """
    return cast(redis.Redis, fake)


class RecordingRedis:
    """A client that records every script call and never talks to anything.

    `error` is what the registered script raises when it is set, which is how the
    fail-open rows are arranged; `calls` is what makes "the pre-check ran before
    any effect" observable, because an empty list is the only evidence that no
    token was taken. `register_script` is recorded separately from `calls`, since
    registration happens at construction and is not a per-take round trip.
    """

    def __init__(self, error: Exception | None = None, answer: Sequence[int] = (1, 0)) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[Any, ...]]] = []
        self.registered: list[str] = []
        self._error = error
        self._answer = answer

    def register_script(self, script: str) -> Callable[..., object]:
        """Return the callable the store keeps, exactly as `redis-py` does."""
        self.registered.append(script)

        def run(keys: Sequence[str], args: Sequence[Any]) -> object:
            self.calls.append((tuple(keys), tuple(args)))
            if self._error is not None:
                raise self._error
            return list(self._answer)

        return run


def fake_store(
    *, error: Exception | None = None, answer: Sequence[int] = (1, 0)
) -> tuple[RedisRateLimitStore, RecordingRedis]:
    """A store over a recording client, and the client, so calls can be asserted."""
    client = RecordingRedis(error=error, answer=answer)
    return RedisRateLimitStore(as_client(client), clock=lambda: FROZEN), client


@contextmanager
def caplog_at(level: int) -> Iterator[list[logging.LogRecord]]:
    """Collect this module's records at `level` and put the logger back afterwards.

    A handler on the named logger rather than pytest's `caplog` fixture, because
    every case in this file is a plain function and one of them needs the records
    from two calls in one arrangement; the restore is what keeps a failing case
    from leaking a handler into the rest of the session.
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(RATE_LIMIT_LOGGER)
    handler = _Collect(level)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# --------------------------------------------------------------------------
# the pre-checks: no Redis, no effect
# --------------------------------------------------------------------------

REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_TENANT),
    ("uppercase", "Evals"),
    ("a colon, which would forge a second key segment", "ev:als"),
    ("a glob character, which would widen a later SCAN", "evals*"),
    ("a trailing newline", "evals\n"),
    ("a trailing CRLF", "evals\r\n"),
    ("a trailing space", "evals "),
    ("a leading space", " evals"),
]


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_a_take_refuses_a_malformed_tenant_before_touching_redis(label: str, tenant: str) -> None:
    """Catches a store that builds its key by interpolation and lets Redis decide: a take
    against `ratelimit:ev:als:req` shapes a bucket nobody reads, so the tenant named
    `ev` is limited by traffic that is not its own, and a glob character widens any later
    `SCAN` an operator runs over these keys. The assertion is on the recorded calls
    rather than on "no key exists", because that is the claim a project rule makes - the
    check runs BEFORE the effect, not instead of it. The trailing-newline row is standing
    rule 4's: a check written `^...$` with `.match` accepts it, and a line break inside a
    Redis key is a key nothing else can address."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.take(tenant, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert client.calls == []


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_the_token_dimension_refuses_the_same_tenants_before_touching_redis(
    label: str, tenant: str
) -> None:
    """The other dimension of the same check. Catches a pre-check applied on the request
    path alone - the path every request takes, and therefore the one that gets the
    attention - leaving `tpm` keys forgeable by any name that reached the route."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.take(tenant, RateDimension.TOKENS, cost=1, capacity=TOKEN_CAPACITY, rate=TOKEN_RATE)

    assert client.calls == []


def test_the_tenant_length_boundary_is_inclusive() -> None:
    """The inclusive boundary a project rule asks for, on the store rather than on the key
    function. Catches a `>=` that refuses a tenant of exactly the permitted length - a
    name `gateway/context.py` already authenticated, so every request from that tenant
    would fail with an untyped error before it reached a provider."""
    store, client = fake_store()

    store.take(MAX_TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert len(client.calls) == 1


# --------------------------------------------------------------------------
# the key and the one round trip (decision A)
# --------------------------------------------------------------------------


def test_the_key_is_the_one_the_bucket_key_function_renders() -> None:
    """Catches a store with a key format of its own, which would leave the bucket an
    operator reads with `HGETALL ratelimit:evals:req` and the bucket the gateway spends
    two different keys - and would make the differential below compare two stores that
    were never looking at the same thing."""
    store, client = fake_store()

    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert store.key(TENANT, RateDimension.REQUESTS) == bucket_key(TENANT, RateDimension.REQUESTS)
    assert tuple(client.calls[0][0]) == (bucket_key(TENANT, RateDimension.REQUESTS),)


def test_the_default_prefix_is_the_one_the_key_is_written_under() -> None:
    """Catches a store built with no prefix argument writing under something other than the
    published prefix: `create_app` builds it with no prefix, so the shipped gateway is
    exactly this case, and every test that passes a prefix would still pass."""
    store, client = fake_store()

    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert str(client.calls[0][0][0]).startswith(f"{RATE_LIMIT_KEY_PREFIX}:")


def test_a_prefixed_store_writes_under_its_own_namespace() -> None:
    """Catches a `key_prefix` argument that is accepted and ignored, which would make every
    live case below share one bucket and this file's isolation a fiction (standing rule
    7): the differential would then be comparing a Redis bucket that another test had
    already drained."""
    client = RecordingRedis()
    store = RedisRateLimitStore(
        as_client(client), key_prefix="test-rl-prefix", clock=lambda: FROZEN
    )

    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert client.calls[0][0][0] == f"test-rl-prefix:{TENANT}:req"


def test_the_script_is_registered_once_and_run_once_per_take() -> None:
    """Decision A's whole justification, asserted as the two counts that carry it.

    Registered once: a store that called `register_script` per take would ship the script
    body on every request, which is the round trip `register_script` exists to avoid.
    Run once per take: a read-modify-write bucket in three commands is what lets two
    replicas both spend the last token, and no single-replica test can ever see that
    race, so "one call" here is the only place atomicity is checkable at all. Catches
    both, and catches a store that registers nothing and falls back to `EVAL` per
    request."""
    store, client = fake_store()

    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)
    store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert len(client.registered) == 1
    assert len(client.calls) == 2


def test_the_script_is_handed_the_cost_the_capacity_the_rate_and_the_ttl() -> None:
    """Catches a script call that leaves one of the four numbers behind, which is silent in
    a way nothing else here can see: without the cost the script takes a fixed one and
    `tpm` counts requests instead of tokens; without the capacity it invents a ceiling;
    without the rate the bucket never refills; without the TTL every key is kept for
    ever. The clock reading is asserted too, because a script that read Redis's own
    `TIME` would give two replicas two different notions of when the bucket was last
    seen. The order of the arguments is deliberately not asserted - the numbers are all
    distinct, so containment is enough to catch an omission without pinning a layout the
    plan never agreed."""
    store, client = fake_store()

    store.take(TENANT, RateDimension.TOKENS, cost=166, capacity=TOKEN_CAPACITY, rate=TOKEN_RATE)

    passed = {float(value) for value in client.calls[0][1]}
    assert 166.0 in passed
    assert float(TOKEN_CAPACITY) in passed
    assert TOKEN_RATE in passed
    assert float(RATE_LIMIT_KEY_TTL_SECONDS) in passed
    assert FROZEN in passed


def test_a_different_cost_reaches_the_script_as_a_different_argument() -> None:
    """The other half of the case above, and the one that stops it passing against a store
    that hardcodes the numbers this file happens to use. Catches a `cost` parameter
    accepted and ignored, which would make every token debit cost the same regardless of
    the usage it books - so a tenant sending one huge request per minute would never
    reach `tpm`."""
    store, client = fake_store()

    store.take(TENANT, RateDimension.TOKENS, cost=166, capacity=TOKEN_CAPACITY, rate=TOKEN_RATE)
    store.take(TENANT, RateDimension.TOKENS, cost=483, capacity=TOKEN_CAPACITY, rate=TOKEN_RATE)

    first = {float(value) for value in client.calls[0][1]}
    second = {float(value) for value in client.calls[1][1]}
    assert 166.0 in first and 483.0 not in first
    assert 483.0 in second and 166.0 not in second


def test_the_scripts_answer_becomes_the_taken() -> None:
    """Catches a store that discards what the script decided and answers `allowed` from its
    own arithmetic - which would be a second bucket, in the process, disagreeing with the
    shared one - and one that drops the retry-after, leaving every 429 without the header
    decision D exists for. Both fields are read from a scripted answer no local
    computation could produce."""
    refusing, _ = fake_store(answer=(0, 7))
    allowing, _ = fake_store(answer=(1, 0))

    refused = refusing.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)
    served = allowing.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert refused == Taken(allowed=False, retry_after_seconds=7)
    assert served == Taken(allowed=True, retry_after_seconds=0)


# --------------------------------------------------------------------------
# fail open (decision E)
# --------------------------------------------------------------------------

REDIS_FAILURES: list[tuple[str, Exception]] = [
    ("the connection dropped", redis.ConnectionError("connection refused")),
    ("the command timed out", redis.TimeoutError("timed out")),
    ("the server answered an error", redis.ResponseError("NOSCRIPT")),
    ("authentication failed", redis.AuthenticationError("wrong password")),
]


@pytest.mark.parametrize(("label", "error"), REDIS_FAILURES, ids=[row[0] for row in REDIS_FAILURES])
def test_an_unreachable_redis_reads_unknown_rather_than_raising(
    label: str, error: Exception
) -> None:
    """Decision E, at the seam where it is decided.

    Catches a store that lets `redis.RedisError` escape, which turns a Redis blip into a
    5xx on every request a provider would have served - the failure a rate limit must not
    cause, and the reason the plan chose fail-open over fail-closed. And catches the two
    over-corrections, both of which are worse than raising: answering
    `Taken(allowed=False, ...)` fails CLOSED silently, refusing every tenant for as long
    as Redis is down; answering `Taken(allowed=True, ...)` is indistinguishable from a
    limiter that decided to serve, so the route has nothing to log and the window in
    which no ceiling was enforced leaves no trace at all. `None` is what makes "unknown"
    expressible.

    Four failure rows because the swallow has to be on `redis.RedisError`, the base:
    catching `ConnectionError` alone passes the first row and raises on the other three.
    `NOSCRIPT` is the row that is specific to decision A - a `ResponseError` is exactly
    what a flushed script cache answers, and it is the failure this store will meet in
    production before it ever meets a dropped connection."""
    store, client = fake_store(error=error)

    assert store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE) is None
    assert len(client.calls) == 1


def test_every_swallowed_failure_is_logged_with_the_tenant_and_the_dimension() -> None:
    """The other half of decision E, and the half that is easy to leave out.

    Failing open means a ceiling silently stops being enforced: the request is served, no
    metric family exists to notice it (decision F), and the log is then the ONLY place
    that window is visible at all. Catches a bare `except redis.RedisError: return None`,
    which is the obvious implementation and leaves an operator with no way to tell a
    quiet minute from an unreachable limiter. Both dimensions are asserted, because they
    fail independently and an operator needs to know whether it was `rpm` or `tpm` that
    stopped being enforced; the values are read off the record rather than out of the
    formatted string, so the wording stays the implementation's to choose."""
    store, _ = fake_store(error=redis.ConnectionError("connection refused"))

    with caplog_at(logging.ERROR) as records:
        store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)
        store.take(TENANT, RateDimension.TOKENS, cost=166, capacity=TOKEN_CAPACITY, rate=TOKEN_RATE)

    assert len(records) == 2, [record.getMessage() for record in records]
    assert {record.__dict__["tenant"] for record in records} == {TENANT}
    assert {record.__dict__["dimension"] for record in records} == {"req", "tok"}
    for record in records:
        assert TENANT in record.getMessage()


def test_an_answer_the_script_could_not_have_produced_reads_unknown() -> None:
    """A malformed reply is a corrupted or replaced script, not an outage, and it is the one
    failure that persists until somebody re-deploys - every request for as long as it
    lasts would otherwise be decided by whatever `bool()` made of the reply.

    Catches an unpacking that raises (`allowed, wait = answer` on a one-element list is a
    `ValueError` from the middle of a served request, with no typed code) and catches a
    truthy read of a non-empty list, which would allow every request while looking like a
    working limiter."""
    store, _ = fake_store(answer=())

    assert store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE) is None


# --------------------------------------------------------------------------
# against a real Redis
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    """One client for the whole session, built once.

    The skip is an explicit conditional on `OPSCONSOLE_TEST_REDIS_URL` (standing
    rule 9), never a swallowed connection error: a suite that could not reach
    Redis has to say so by name, or the half of this file that makes the
    in-process twin evidence about the Lua script would report success having
    checked nothing.
    """
    if REDIS_URL is None:
        pytest.skip(f"{REDIS_URL_VAR} is not set: start Redis and export it to run these")
    client: redis.Redis = redis.Redis.from_url(REDIS_URL)
    yield client
    client.close()


@pytest.fixture
def key_prefix() -> str:
    """A namespace unique to one test, which is the whole of its isolation.

    Never `FLUSHDB`: this suite may share a database with anything else, and a
    flush would delete keys it does not own. Every key this
    file writes carries the shipped TTL, so nothing it leaves behind is
    permanent.
    """
    return f"test-ratelimit-{uuid.uuid4().hex}"


@pytest.fixture
def clock() -> list[float]:
    """The instant both stores read, as a one-element list a case can assign into."""
    return [FROZEN]


@pytest.fixture
def live_store(
    redis_client: redis.Redis, key_prefix: str, clock: list[float]
) -> RedisRateLimitStore:
    """A store over the session client, namespaced to this test, on the injected clock."""
    return RedisRateLimitStore(redis_client, key_prefix=key_prefix, clock=lambda: clock[0])


# (label, tenant, dimension, cost, capacity, rate, the instant this take happens)
SEQUENCE: list[tuple[str, str, RateDimension, int, int, float, float]] = [
    ("the first request", TENANT, RateDimension.REQUESTS, 1, CAPACITY, RATE, FROZEN),
    ("its tokens", TENANT, RateDimension.TOKENS, 400, TOKEN_CAPACITY, TOKEN_RATE, FROZEN),
    ("a second request", TENANT, RateDimension.REQUESTS, 1, CAPACITY, RATE, FROZEN),
    ("another tenant's request", OTHER_TENANT, RateDimension.REQUESTS, 1, CAPACITY, RATE, FROZEN),
    (
        "a token debit that overdraws",
        TENANT,
        RateDimension.TOKENS,
        900,
        TOKEN_CAPACITY,
        TOKEN_RATE,
        FROZEN,
    ),
    (
        "the token check that follows it",
        TENANT,
        RateDimension.TOKENS,
        1,
        TOKEN_CAPACITY,
        TOKEN_RATE,
        FROZEN,
    ),
    ("the third request", TENANT, RateDimension.REQUESTS, 1, CAPACITY, RATE, FROZEN),
    ("the fourth request", TENANT, RateDimension.REQUESTS, 1, CAPACITY, RATE, FROZEN),
    (
        "the fifth request, over the ceiling",
        TENANT,
        RateDimension.REQUESTS,
        1,
        CAPACITY,
        RATE,
        FROZEN,
    ),
    (
        "the same request a quarter second later, still short",
        TENANT,
        RateDimension.REQUESTS,
        1,
        CAPACITY,
        RATE,
        FROZEN + 0.25,
    ),
    (
        "half a second after the refusal, one whole token",
        TENANT,
        RateDimension.REQUESTS,
        1,
        CAPACITY,
        RATE,
        FROZEN + 0.5,
    ),
    ("and immediately again", TENANT, RateDimension.REQUESTS, 1, CAPACITY, RATE, FROZEN + 0.5),
    (
        "the token bucket a second later",
        TENANT,
        RateDimension.TOKENS,
        1,
        TOKEN_CAPACITY,
        TOKEN_RATE,
        FROZEN + 1.5,
    ),
    (
        "the other tenant, untouched by any of it",
        OTHER_TENANT,
        RateDimension.REQUESTS,
        1,
        CAPACITY,
        RATE,
        FROZEN + 1.5,
    ),
]
"""One interleaved sequence over both dimensions and two tenants, with the clock
moving inside it. It is not a list of independent cases: the
overdraw at row five is what makes row six a refusal, and the refusal at row nine
is what makes rows ten and eleven mean anything."""


def test_the_two_stores_answer_identically_over_one_interleaved_sequence(
    live_store: RedisRateLimitStore, clock: list[float]
) -> None:
    """The differential section 5's second risk asks for, and the only thing that makes the
    in-process twin evidence about the Lua script.

    Every enforcement case in tests/gateway/test_rate_limit_enforcement.py runs against
    the twin, because the suite has no Redis. If the script and the twin disagree
    anywhere - a refill computed from Redis's own clock, an off-by-one at the empty
    bucket, a floor missing on an overdraw, a retry-after rounded down in Lua's `math`
    where Python used `ceil` - then every one of those cases is a statement about code
    that is not what a deployment runs, and no single-store test can notice. Catches that
    class whole, over a sequence that interleaves both dimensions, two tenants and a
    clock that moves mid-run. The answers are compared element by element so the failure
    names the row that diverged."""
    twin = InProcessRateLimitStore(clock=lambda: clock[0])

    for label, tenant, dimension, cost, capacity, rate, at in SEQUENCE:
        clock[0] = at
        from_redis = live_store.take(tenant, dimension, cost=cost, capacity=capacity, rate=rate)
        from_twin = twin.take(tenant, dimension, cost=cost, capacity=capacity, rate=rate)

        assert from_redis is not None, label
        assert from_redis == from_twin, label


def test_the_live_bucket_refuses_at_exactly_its_ceiling(live_store: RedisRateLimitStore) -> None:
    """The inclusive boundary against the real script rather than against the twin, so the
    differential above cannot pass by having both stores wrong in the same way. Catches a
    Lua comparison written `>` instead of `>=`, which caps a tenant one request below
    what `routing.yaml` promises, and one written on the level AFTER the subtraction,
    which serves one request too many."""
    served = 0
    for _ in range(CAPACITY + 2):
        taken = live_store.take(
            TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE
        )
        assert taken is not None
        served += int(taken.allowed)

    assert served == CAPACITY


def test_the_two_dimensions_are_separate_keys_on_a_live_server(
    live_store: RedisRateLimitStore,
) -> None:
    """Catches a script that drops the dimension from its key - one bucket for `rpm` and
    `tpm` together, so a single ordinary request's token debit exhausts the request
    allowance for the whole minute and the gateway refuses everything. Draining the token
    bucket completely and then taking a request is the arrangement in which one shared
    key is the only thing that can refuse."""
    drained = live_store.take(
        TENANT, RateDimension.TOKENS, cost=TOKEN_CAPACITY, capacity=TOKEN_CAPACITY, rate=TOKEN_RATE
    )
    request = live_store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    assert drained is not None and drained.allowed is True
    assert request is not None and request.allowed is True


def test_one_tenants_live_bucket_is_not_anothers(live_store: RedisRateLimitStore) -> None:
    """Catches a key that drops the tenant segment: with one bucket for everyone, a quiet
    tenant is refused the moment a busy neighbour reaches its ceiling. Read with a tenant
    that genuinely has a live bucket, so "no key yet" cannot be what makes it pass."""
    for _ in range(CAPACITY):
        live_store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    exhausted = live_store.take(
        TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE
    )
    neighbour = live_store.take(
        OTHER_TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE
    )

    assert exhausted is not None and exhausted.allowed is False
    assert neighbour is not None and neighbour.allowed is True


def test_the_bucket_expires_after_exactly_the_published_ttl(
    live_store: RedisRateLimitStore, redis_client: redis.Redis
) -> None:
    """Read back off the server rather than slept for. Catches a bucket with no expiry at
    all (`TTL` answers -1, which is two keys per tenant kept for ever) and a key that was
    never written (-2), which is what a script that computes an answer and stores nothing
    looks like - it would hand every request a full bucket and enforce nothing. The
    expiry is set inside the same script call as the take, so there is no second for the
    clock to tick in and the value reads back exactly."""
    live_store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    ttl = redis_client.ttl(live_store.key(TENANT, RateDimension.REQUESTS))

    assert ttl != -1, "the bucket carries no expiry, so it is kept for ever"
    assert ttl != -2, "no bucket was written at all"
    assert ttl == RATE_LIMIT_KEY_TTL_SECONDS


def test_a_second_take_does_not_drop_the_expiry(
    live_store: RedisRateLimitStore, redis_client: redis.Redis
) -> None:
    """Catches a script that sets the TTL only when it thinks the key is new and gets the
    condition wrong, and one that rewrites the hash without a TTL on the second take.
    Every request of the minute goes through this path, so an expiry that survives only
    the first one survives nothing."""
    live_store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)
    live_store.take(TENANT, RateDimension.REQUESTS, cost=1, capacity=CAPACITY, rate=RATE)

    ttl = redis_client.ttl(live_store.key(TENANT, RateDimension.REQUESTS))

    assert ttl != -1, "the second take left the bucket with no expiry"
    assert ttl == RATE_LIMIT_KEY_TTL_SECONDS
