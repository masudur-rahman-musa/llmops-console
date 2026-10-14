r"""`RedisResponseCache`: the key it writes, the TTL on it, the pre-checks, and its silence.

Two halves, and the split is the same one `tests/redaction/test_redis_store.py`
makes, for the same reasons.

The half that needs a REAL Redis is the half whose claims are Redis semantics:
the JSON round trip, the TTL read back with `TTL` (never by sleeping), the tenant
half of the key, and the key rendering. Redis is required by an explicit
conditional on `OPSCONSOLE_TEST_REDIS_URL`; the fixture skips
with that variable's name in the reason. Deliberately NOT `OPSCONSOLE_REDIS_URL`,
which `Settings` itself reads: exporting that one to run this file would hand
every other test in the suite a Redis cache and a `redis` readiness probe it never
asked for, and the root `tests/conftest.py` clears it for exactly that reason.

The half that needs NOTHING is the validation half, over a client that records
its calls and refuses to answer: both pre-checks run before any command is sent,
so "no key was written" is asserted as "Redis was never called at all", which is
what a project rule actually claims. Redis is an unsized store and there is no
backstop behind these two checks.

The difference from the redaction map store is the point of the failure rows: a
map store RAISES `RedactionMapStoreUnavailable` so a rehydration cannot silently
answer "no such map", while this store SWALLOWS every `redis.RedisError` and
answers absent, because a cache that raises turns a Redis incident into 5xx on
requests the provider could have served (plan §5).

a project rule: one session-scoped client, built once. Per-test isolation is a
unique `key_prefix` handed to the store, never `FLUSHDB` (which would delete a
neighbouring suite's keys) and never a fresh client.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import cast

import pytest
import redis

from app.cache import (
    CACHE_KEY_PREFIX,
    RESPONSE_CACHE_TTL_SECONDS,
    RedisResponseCache,
    ResponseCache,
)
from app.providers.base import ChatResponse
from app.routing.errors import IdentifierInvalid
from app.routing.resolver import MAX_IDENTIFIER_LENGTH

from .conftest import OTHER_TENANT, TENANT, cached_answer

REDIS_URL_VAR = "OPSCONSOLE_TEST_REDIS_URL"
REDIS_URL = os.environ.get(REDIS_URL_VAR)

WELL_FORMED_KEY = "0123456789abcdef" * 4
"""64 lowercase hex characters: the shape `cache_key` mints and `get` accepts."""

OTHER_KEY = "fedcba9876543210" * 4


def as_client(fake: object) -> redis.Redis:
    """Hand a recording double to a store typed for a real client.

    The store is annotated against `redis.Redis`, and the cases in this half of
    the file are about calls that never leave the process; casting keeps the
    double honest about its methods without a `# type: ignore` on every call.
    """
    return cast(redis.Redis, fake)


class RecordingRedis:
    """A client that records every call and never talks to anything.

    `error` is what both commands raise when it is set, which is how the
    unreachable-Redis rows are arranged; `calls` is what makes "the pre-check ran
    before any effect" observable, because an empty list is the only evidence that
    no key was written.
    """

    def __init__(self, error: Exception | None = None, value: bytes | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self._error = error
        self._value = value

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        self.calls.append(("set", (name, value, ex)))
        if self._error is not None:
            raise self._error
        return True

    def get(self, name: str) -> bytes | None:
        self.calls.append(("get", (name,)))
        if self._error is not None:
            raise self._error
        return self._value


def fake_cache(
    *, error: Exception | None = None, value: bytes | None = None
) -> tuple[RedisResponseCache, RecordingRedis]:
    """A cache over a recording client, and the client, so calls can be asserted."""
    client = RecordingRedis(error=error, value=value)
    return RedisResponseCache(as_client(client)), client


# --------------------------------------------------------------------------
# the pre-checks: no Redis, no effect
# --------------------------------------------------------------------------

MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)

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
    ("a leading separator", "-evals"),
]


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_put_refuses_a_malformed_tenant_before_touching_redis(case: str, tenant: str) -> None:
    """Catches a cache that builds its key from an unchecked tenant: the colon row would
    otherwise write into another tenant's namespace, the newline rows are what separate
    `fullmatch` from `match`, and asserting the client saw NO call is
    what pins "writes no key" rather than "writes a key and fails afterwards". Redis is
    unsized, so these two checks are the only thing between a caller and its keyspace."""
    cache, client = fake_cache()

    with pytest.raises(IdentifierInvalid):
        cache.put(tenant, WELL_FORMED_KEY, cached_answer())

    assert client.calls == []


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_get_refuses_a_malformed_tenant_before_touching_redis(case: str, tenant: str) -> None:
    """Catches a pre-check applied to `put` alone: `get` builds the same key from the same
    string, so an unchecked tenant on the read path reads across namespaces even when the
    write path is safe."""
    cache, client = fake_cache()

    with pytest.raises(IdentifierInvalid):
        cache.get(tenant, WELL_FORMED_KEY)

    assert client.calls == []


def test_a_tenant_at_the_length_cap_is_accepted() -> None:
    """Catches an exclusive comparison (`>=` where `>` belongs) on the length cap: exactly
    `MAX_IDENTIFIER_LENGTH` characters is a legal tenant name everywhere else in the
    gateway, so refusing it here would make a tenant that authenticates uncacheable, and
    the cap is imported rather than restated so the two cannot drift."""
    cache, client = fake_cache()

    cache.put(MAX_TENANT, WELL_FORMED_KEY, cached_answer())

    assert [call[0] for call in client.calls] == ["set"]


def test_a_tenant_one_character_over_the_cap_is_rejected() -> None:
    """The other half of the boundary pair: catches a cap that was never applied, which
    the accepted row alone cannot see."""
    cache, client = fake_cache()

    with pytest.raises(IdentifierInvalid):
        cache.put(OVER_CAP_TENANT, WELL_FORMED_KEY, cached_answer())

    assert client.calls == []


REJECTED_KEYS: list[tuple[str, str]] = [
    ("empty", ""),
    ("sixty-three hex characters", WELL_FORMED_KEY[:63]),
    ("sixty-five hex characters", WELL_FORMED_KEY + "0"),
    ("uppercase hex", WELL_FORMED_KEY.upper()),
    ("sixty-four non-hex characters", "g" * 64),
    ("a trailing newline", WELL_FORMED_KEY + "\n"),
    ("a trailing CRLF", WELL_FORMED_KEY + "\r\n"),
    ("a trailing space", WELL_FORMED_KEY + " "),
    ("a colon, which would forge a fourth key segment", WELL_FORMED_KEY[:63] + ":"),
    ("a glob character", WELL_FORMED_KEY[:63] + "*"),
]


@pytest.mark.parametrize(("case", "key"), REJECTED_KEYS, ids=[c[0] for c in REJECTED_KEYS])
def test_get_refuses_a_malformed_key_before_touching_redis(case: str, key: str) -> None:
    r"""Catches a check written `re.match` rather than `re.fullmatch`: the 65-character row
    and all three trailing-whitespace rows pass `match` and would put a line break into a
    Redis key. The empty and non-hex rows keep a cache that checks only
    the length from passing the table."""
    cache, client = fake_cache()

    with pytest.raises(IdentifierInvalid):
        cache.get(TENANT, key)

    assert client.calls == []


@pytest.mark.parametrize(("case", "key"), REJECTED_KEYS, ids=[c[0] for c in REJECTED_KEYS])
def test_put_refuses_a_malformed_key_before_touching_redis(case: str, key: str) -> None:
    """Catches a key pre-check on the read path alone: the write path is where a malformed
    key actually LANDS in the keyspace, so a `put` that skips the check is the half that
    does the damage the read half only reads back."""
    cache, client = fake_cache()

    with pytest.raises(IdentifierInvalid):
        cache.put(TENANT, key, cached_answer())

    assert client.calls == []


def test_get_accepts_a_well_formed_key_and_asks_redis_for_it() -> None:
    """Catches a pre-check so strict that no key survives it - every other row in this
    group asserts a refusal, so a `get` that raised for everything would pass all of them,
    and this is the row that separates the two readings."""
    cache, client = fake_cache(value=None)

    assert cache.get(TENANT, WELL_FORMED_KEY) is None
    assert [call[0] for call in client.calls] == ["get"]


def test_put_sends_the_ttl_with_the_write() -> None:
    """Catches a `SET` with no expiry, or one with the TTL passed as milliseconds: the
    argument is read off the call the client actually received, so a store that computed
    the right number and never handed it over fails here. Without it an entry lives until
    Redis evicts it, and #22's twenty-four hours would be a comment."""
    cache, client = fake_cache()

    cache.put(TENANT, WELL_FORMED_KEY, cached_answer())

    assert [call[0] for call in client.calls] == ["set"]
    name, value, expiry = client.calls[0][1]
    assert name == f"{CACHE_KEY_PREFIX}:{TENANT}:{WELL_FORMED_KEY}"
    assert expiry == RESPONSE_CACHE_TTL_SECONDS
    assert isinstance(value, str | bytes)


# --------------------------------------------------------------------------
# an unreachable Redis: absent, never a failure
# --------------------------------------------------------------------------

REDIS_FAILURES: list[tuple[str, Exception]] = [
    ("a refused connection", redis.ConnectionError("connection refused")),
    ("a hung server hitting the read timeout", redis.TimeoutError("timed out")),
    ("a server that answered an error", redis.ResponseError("READONLY")),
    ("the base failure the client raises", redis.RedisError("something went wrong")),
]


@pytest.mark.parametrize(("case", "error"), REDIS_FAILURES, ids=[c[0] for c in REDIS_FAILURES])
def test_a_get_against_an_unreachable_redis_answers_absent(case: str, error: Exception) -> None:
    """Catches a cache that lets `redis.RedisError` escape to the route: unlike the
    redaction map store, this one is not a control, so a Redis outage has to look exactly
    like a miss - anything else turns an incident into 5xx on requests the provider would
    have served. The four rows are the failures the client actually raises, not one."""
    cache, _ = fake_cache(error=error)

    assert cache.get(TENANT, WELL_FORMED_KEY) is None


@pytest.mark.parametrize(("case", "error"), REDIS_FAILURES, ids=[c[0] for c in REDIS_FAILURES])
def test_a_put_against_an_unreachable_redis_is_swallowed(case: str, error: Exception) -> None:
    """Catches the swallow being applied to the read path alone: the write happens AFTER a
    provider has already answered, so an escaping error there would fail a request that
    was fully served and paid for - the most expensive way to lose a response there is."""
    cache, _ = fake_cache(error=error)

    cache.put(TENANT, WELL_FORMED_KEY, cached_answer())


def test_unreadable_stored_bytes_answer_absent_rather_than_raising() -> None:
    """Catches a `get` that trusts whatever is under the key: a value written by an older
    build, a half-written entry or another process's key would raise a `ValidationError`
    out of `model_validate_json` and 500 a request that only wanted a cache lookup."""
    cache, _ = fake_cache(value=b"not json at all")

    assert cache.get(TENANT, WELL_FORMED_KEY) is None


# --------------------------------------------------------------------------
# against a real Redis
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    """One client for the whole session, built once.

    The skip is an explicit conditional on `OPSCONSOLE_TEST_REDIS_URL` (standing
    rule 9), never a swallowed connection error: this store answers `None` on
    every failure, so a suite that could not reach Redis would otherwise report
    success having checked nothing at all - which is exactly the trap this
    store's own semantics set for its tests.
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
    flush would delete keys it does not own. Every key written under this prefix
    expires on its own TTL.
    """
    return f"test-{uuid.uuid4().hex}"


@pytest.fixture
def cache(redis_client: redis.Redis, key_prefix: str) -> RedisResponseCache:
    """The cache under test, namespaced to this test and sharing the session's client."""
    return RedisResponseCache(redis_client, key_prefix=key_prefix)


def test_a_stored_answer_round_trips_through_redis(cache: RedisResponseCache) -> None:
    """Catches a value written with `str(answer)` or read back as raw bytes: the entry
    travels as JSON and comes back as a `ChatResponse`, so a round trip that lost the tool
    calls or the usage would serve a hit the output path renders differently from the
    answer that was cached."""
    answer = cached_answer()

    cache.put(TENANT, WELL_FORMED_KEY, answer)
    read = cache.get(TENANT, WELL_FORMED_KEY)

    assert isinstance(read, ChatResponse)
    assert read == answer


def test_an_unknown_but_well_formed_key_answers_absent(cache: RedisResponseCache) -> None:
    """Catches a `get` that raises on a miss: a miss is the ordinary outcome for every
    ineligible request this gateway serves, so the absent answer is the hot path and not
    an edge case."""
    assert cache.get(TENANT, WELL_FORMED_KEY) is None


def test_one_tenant_cannot_read_another_tenants_live_entry(cache: RedisResponseCache) -> None:
    """Catches a key built from the cache key alone: the key handed over is a REAL, live
    entry for the other tenant, so "unknown key" cannot be what refuses it, and the two
    closing reads prove the refusals are not a cache that lost both entries."""
    mine = cached_answer(content="the answer for evals")
    theirs = cached_answer(content="the answer for cuan")

    cache.put(TENANT, WELL_FORMED_KEY, mine)
    cache.put(OTHER_TENANT, OTHER_KEY, theirs)

    assert cache.get(OTHER_TENANT, WELL_FORMED_KEY) is None
    assert cache.get(TENANT, OTHER_KEY) is None
    assert cache.get(TENANT, WELL_FORMED_KEY) == mine
    assert cache.get(OTHER_TENANT, OTHER_KEY) == theirs


def test_the_same_key_under_two_tenants_holds_two_answers(cache: RedisResponseCache) -> None:
    """Catches a Redis key missing its tenant segment: two tenants sending identical
    requests would then write the same Redis key, and the second put would overwrite the
    first - a cross-tenant leak an "unknown key" arrangement can never see."""
    mine = cached_answer(content="the answer for evals")
    theirs = cached_answer(content="the answer for cuan")

    cache.put(TENANT, WELL_FORMED_KEY, mine)
    cache.put(OTHER_TENANT, WELL_FORMED_KEY, theirs)

    assert cache.get(TENANT, WELL_FORMED_KEY) == mine
    assert cache.get(OTHER_TENANT, WELL_FORMED_KEY) == theirs


def test_the_key_carries_the_prefix_the_tenant_and_the_cache_key(
    cache: RedisResponseCache, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a key built in another shape (`response:{key}`, or the segments swapped):
    the key is what a second implementation, a migration and an operator's `SCAN` all
    depend on, and it is asserted against Redis rather than against the store's own
    `key()` alone."""
    cache.put(TENANT, WELL_FORMED_KEY, cached_answer())

    expected = f"{key_prefix}:{TENANT}:{WELL_FORMED_KEY}"
    assert cache.key(TENANT, WELL_FORMED_KEY) == expected
    assert redis_client.exists(expected) == 1


def test_the_default_prefix_is_the_one_the_key_is_written_under(
    redis_client: redis.Redis,
) -> None:
    """Catches a default `key_prefix` that drifts from `CACHE_KEY_PREFIX`: every other
    real case passes its own prefix, so the shipped default - the one `create_app` uses -
    is exercised nowhere else. The tenant is unique per run, so this writes into nobody's
    namespace."""
    cache = RedisResponseCache(redis_client)
    tenant = f"t{uuid.uuid4().hex}"

    cache.put(tenant, WELL_FORMED_KEY, cached_answer())

    assert cache.key(tenant, WELL_FORMED_KEY) == f"{CACHE_KEY_PREFIX}:{tenant}:{WELL_FORMED_KEY}"
    assert redis_client.exists(f"response:{tenant}:{WELL_FORMED_KEY}") == 1


def test_the_key_expires_after_exactly_the_published_ttl(
    cache: RedisResponseCache, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a key written with `SET` and no expiry at all (`TTL` answers -1, not 86400)
    and a TTL set in the wrong unit by `PEXPIRE`: the value is read back off the server
    rather than slept for, and the exact number is what separates a day from a plausible
    neighbour."""
    cache.put(TENANT, WELL_FORMED_KEY, cached_answer())

    ttl = redis_client.ttl(f"{key_prefix}:{TENANT}:{WELL_FORMED_KEY}")

    assert ttl == RESPONSE_CACHE_TTL_SECONDS


def test_the_redis_cache_satisfies_the_store_protocol(
    redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches method names or signatures that drift from `ResponseCache`: the route reads
    whichever cache `create_app` chose through this protocol, so a `put` taking its
    arguments in another order is a 500 on the first cacheable request in the one
    deployment that has Redis - the one no in-process test would cover."""
    protocol_typed: ResponseCache = RedisResponseCache(redis_client, key_prefix=key_prefix)
    answer = cached_answer()

    protocol_typed.put(TENANT, WELL_FORMED_KEY, answer)

    assert protocol_typed.get(TENANT, WELL_FORMED_KEY) == answer
