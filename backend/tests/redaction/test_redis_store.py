r"""`RedisRedactionMapStore`: the key it writes, the TTL on it, and the two pre-checks.

Two halves, and the split is deliberate.

The half that needs a REAL Redis is the half whose claims are Redis semantics:
the round trip, the TTL read back with `TTL` (never by sleeping), the tenant half
of the key, and the id rendering. A fake would prove those against a
reimplementation of Redis rather than against Redis (plan decision E). Redis is
required by an explicit conditional on `OPSCONSOLE_TEST_REDIS_URL` (standing rule
9); the fixture below skips with that variable's name in the reason, so a suite
that silently checked nothing is legible in the run's own output rather than
inferred. Deliberately NOT `OPSCONSOLE_REDIS_URL`, which `Settings` itself reads:
exporting that one to run this file would hand every other test in the suite a
Redis map store and a `redis` readiness probe it never asked for, and the root
`tests/conftest.py` clears it for exactly that reason.

The half that needs NOTHING is the validation half: both pre-checks run before
any call reaches Redis, so a client that records its calls and refuses to answer
is a sharper witness than a live server - "no key was written" is asserted as
"Redis was never called at all", which is the claim the criterion actually makes.
The unavailable-store mapping lives there too, over a client that raises
`redis.RedisError`.

a project rule: one session-scoped client, built once. Per-test isolation is a
unique `key_prefix` handed to the store, never `FLUSHDB` (which would delete a
neighbouring suite's keys) and never a fresh client.

a project rule on `MAP_ID_PATTERN`: the rejection table carries a trailing
newline, a trailing CRLF and a trailing space, because `re.match` with `[0-9a-f]{32}$`
accepts the first of those and would hand Redis a key with a line break in it.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Mapping
from typing import cast

import pytest
import redis

from app.core.errors import GatewayError
from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.model_routing.errors import IdentifierInvalid
from app.redaction import (
    REDACTION_MAP_TTL_SECONDS,
    RedactionMapStore,
)
from app.redaction.errors import RedactionMapStoreUnavailable
from app.redaction.store import (
    MAP_ID_PATTERN,
    REDACTION_MAP_KEY_PREFIX,
    RedisRedactionMapStore,
)

REDIS_URL_VAR = "OPSCONSOLE_TEST_REDIS_URL"
REDIS_URL = os.environ.get(REDIS_URL_VAR)

TENANT = "nordlicht"
OTHER_TENANT = "cuan"

MAPPING: Mapping[str, str] = {
    "<EMAIL_1>": "alice@example.com",
    "<EMAIL_2>": "bob@example.net",
    "<IPV4_1>": "192.0.2.1",
}
OTHER_MAPPING: Mapping[str, str] = {"<EMAIL_1>": "carol@example.org"}

WELL_FORMED_ID = "0123456789abcdef0123456789abcdef"
"""32 lowercase hex characters: the shape `put` mints and `get` accepts."""


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


def fake_store(
    *, error: Exception | None = None, value: bytes | None = None
) -> tuple[RedisRedactionMapStore, RecordingRedis]:
    """A store over a recording client, and the client, so calls can be asserted."""
    client = RecordingRedis(error=error, value=value)
    return RedisRedactionMapStore(as_client(client)), client


# --------------------------------------------------------------------------
# the constants and the exports
# --------------------------------------------------------------------------


def test_the_key_prefix_is_the_published_one() -> None:
    """Catches a prefix renamed after data exists under the old one: a running gateway's
    live maps are addressed by this string, so changing it silently orphans every map a
    caller is still holding an id for."""
    assert REDACTION_MAP_KEY_PREFIX == "redaction"


def test_the_redis_store_and_its_failure_are_exported_from_the_package() -> None:
    """Catches an implementation that lands the class in `store.py` and never re-exports
    it: `gateway/app.py` and the redact route both import from `app.redaction`, so
    a missing export is an ImportError at process start rather than at a test."""
    from app import redaction

    assert redaction.RedisRedactionMapStore is RedisRedactionMapStore
    assert redaction.RedactionMapStoreUnavailable is RedactionMapStoreUnavailable


def test_the_unavailable_failure_is_not_a_gateway_error() -> None:
    """Catches a failure class made a `GatewayError` subclass: `ErrorCode.PROVIDER_ERROR`
    already belongs to `ProviderError` and `test_error_code_set_is_exhaustive` allows
    exactly one concrete class per code, so a subclass here reddens a test in another
    directory with no obvious connection to this one."""
    assert issubclass(RedactionMapStoreUnavailable, Exception)
    assert not issubclass(RedactionMapStoreUnavailable, GatewayError)


# --------------------------------------------------------------------------
# the pre-checks: no Redis, no effect
# --------------------------------------------------------------------------

MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)

REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_TENANT),
    ("uppercase", "Nordlicht"),
    ("a space", "nord licht"),
    ("a colon, which would forge a second key segment", "nord:licht"),
    ("a path separator", "nord/licht"),
    ("a glob character, which would widen a later SCAN", "nord*"),
    ("a trailing newline", "nordlicht\n"),
    ("a trailing CRLF", "nordlicht\r\n"),
    ("a trailing space", "nordlicht "),
    ("a leading separator", "-nordlicht"),
]


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_put_refuses_a_malformed_tenant_before_touching_redis(case: str, tenant: str) -> None:
    """Catches a store that builds its key from an unchecked tenant: the colon row would
    otherwise write into another tenant's namespace, the newline rows are what separate
    `fullmatch` from `match`, and asserting the client saw NO call is
    what pins "writes no key" rather than "writes a key and fails afterwards"."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.put(tenant, MAPPING)

    assert client.calls == []


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_get_refuses_a_malformed_tenant_before_touching_redis(case: str, tenant: str) -> None:
    """Catches a pre-check applied to `put` alone: `get` builds the same key from the same
    client-supplied string, so an unchecked tenant on the read path reads across
    namespaces even when the write path is safe."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.get(tenant, WELL_FORMED_ID)

    assert client.calls == []


def test_a_tenant_at_the_length_cap_is_accepted() -> None:
    """Catches an exclusive comparison (`>=` where `>` belongs) on the length cap: exactly
    `MAX_IDENTIFIER_LENGTH` characters is a legal tenant name everywhere else in the
    gateway, so refusing it here would make a tenant that authenticates unable to
    redact."""
    store, client = fake_store()

    map_id = store.put(MAX_TENANT, MAPPING)

    assert MAP_ID_PATTERN.fullmatch(map_id) is not None
    assert [call[0] for call in client.calls] == ["set"]


def test_a_tenant_one_character_over_the_cap_is_rejected() -> None:
    """The other half of the boundary pair: catches a cap that was never applied, which
    the accepted row alone cannot see."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.put(OVER_CAP_TENANT, MAPPING)

    assert client.calls == []


REJECTED_MAP_IDS: list[tuple[str, str]] = [
    ("empty", ""),
    ("thirty-one hex characters", "0" * 31),
    ("thirty-three hex characters", "0" * 33),
    ("uppercase hex", "0123456789ABCDEF0123456789ABCDEF"),
    ("thirty-two non-hex characters", "g" * 32),
    ("a trailing newline", WELL_FORMED_ID + "\n"),
    ("a trailing CRLF", WELL_FORMED_ID + "\r\n"),
    ("a trailing space", WELL_FORMED_ID + " "),
    ("a colon, which would forge a third key segment", WELL_FORMED_ID[:31] + ":"),
    ("a glob character", WELL_FORMED_ID[:31] + "*"),
]


@pytest.mark.parametrize(("case", "map_id"), REJECTED_MAP_IDS, ids=[c[0] for c in REJECTED_MAP_IDS])
def test_get_refuses_a_malformed_map_id_before_touching_redis(case: str, map_id: str) -> None:
    """Catches a check written `re.match` rather than `re.fullmatch`: the 33-character row
    and all three trailing-whitespace rows pass `match` and would put a caller-controlled
    line break into a Redis key. The empty and non-hex rows keep a store
    that checks only the length from passing the table."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.get(TENANT, map_id)

    assert client.calls == []


def test_get_accepts_a_well_formed_id_and_asks_redis_for_it() -> None:
    """Catches a pre-check so strict that no id survives it - every other row in this
    group asserts a refusal, so a `get` that raised for everything would pass all of them
    and this is the row that separates the two readings."""
    store, client = fake_store(value=None)

    assert store.get(TENANT, WELL_FORMED_ID) is None
    assert [call[0] for call in client.calls] == ["get"]


REDIS_FAILURES: list[tuple[str, Exception]] = [
    ("a refused connection", redis.ConnectionError("connection refused")),
    ("a hung server hitting the read timeout", redis.TimeoutError("timed out")),
    ("a server that answered an error", redis.ResponseError("READONLY")),
    ("the base failure the client raises", redis.RedisError("something went wrong")),
]


@pytest.mark.parametrize(("case", "error"), REDIS_FAILURES, ids=[c[0] for c in REDIS_FAILURES])
def test_a_put_against_an_unreachable_redis_raises_the_typed_failure(
    case: str, error: Exception
) -> None:
    """Catches a store that lets `redis.RedisError` escape to the route: the route catches
    the typed failure to answer 502 PROVIDER_ERROR, so a raw client error there is a 500
    with no code - and the rows are the four the client actually raises, not one."""
    store, _ = fake_store(error=error)

    with pytest.raises(RedactionMapStoreUnavailable):
        store.put(TENANT, MAPPING)


@pytest.mark.parametrize(("case", "error"), REDIS_FAILURES, ids=[c[0] for c in REDIS_FAILURES])
def test_a_get_against_an_unreachable_redis_raises_the_typed_failure(
    case: str, error: Exception
) -> None:
    """Catches a conversion applied to the write path alone, and - worse - a `get` that
    answered `None` on a connection failure: an earlier change's rehydration would then report "no
    such map" for a map that is sitting in a Redis nobody could reach."""
    store, _ = fake_store(error=error)

    with pytest.raises(RedactionMapStoreUnavailable):
        store.get(TENANT, WELL_FORMED_ID)


def test_the_typed_failure_does_not_swallow_the_cause() -> None:
    """Catches a `raise ... from None`: the client's own error is the only description of
    WHY Redis was unreachable, and the log line an operator reads is built from the
    chained cause."""
    original = redis.ConnectionError("connection refused")
    store, _ = fake_store(error=original)

    with pytest.raises(RedactionMapStoreUnavailable) as raised:
        store.put(TENANT, MAPPING)

    assert raised.value.__cause__ is original


# --------------------------------------------------------------------------
# against a real Redis
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    """One client for the whole session, built once.

    The skip is an explicit conditional on `OPSCONSOLE_REDIS_URL` (standing rule
    9), never a swallowed connection error: a suite that could not reach Redis
    has to say so by name, or the half of this file that pins the TTL would
    report success having checked nothing.
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
def store(redis_client: redis.Redis, key_prefix: str) -> RedisRedactionMapStore:
    """The store under test, namespaced to this test and sharing the session's client."""
    return RedisRedactionMapStore(redis_client, key_prefix=key_prefix)


def test_a_stored_map_round_trips_every_entry(store: RedisRedactionMapStore) -> None:
    """Catches a store that writes the map and reads back something else - a JSON value
    written but decoded as bytes, or only the placeholders kept: rehydration substitutes
    every pair back, so a map that came out short leaves a placeholder in text a caller
    believes was restored."""
    map_id = store.put(TENANT, MAPPING)
    read = store.get(TENANT, map_id)

    assert read is not None
    assert dict(read) == dict(MAPPING)


def test_a_stored_map_id_is_thirty_two_lowercase_hex_characters(
    store: RedisRedactionMapStore,
) -> None:
    """Catches an id whose rendering drifts from what `get` will accept: the pre-check
    refuses anything that is not 32 lowercase hex, so an id minted as a UUID with dashes
    would be refused by the very store that issued it."""
    map_id = store.put(TENANT, MAPPING)

    assert len(map_id) == 32
    assert MAP_ID_PATTERN.fullmatch(map_id) is not None


def test_an_unknown_but_well_formed_id_answers_absent(store: RedisRedactionMapStore) -> None:
    """Catches a `get` that raises on a miss, or one that returns an empty mapping: plan
    12 asks about ids a caller made up, and `{}` would be read as "this map redacted
    nothing" rather than "there is no such map"."""
    assert store.get(TENANT, WELL_FORMED_ID) is None


def test_one_tenant_cannot_read_another_tenants_live_map(store: RedisRedactionMapStore) -> None:
    """Catches a store keyed on the map id alone: the id handed over is a REAL, live id
    for the other tenant, so "unknown id" cannot be what refuses it, and the two closing
    reads prove the refusals are not a store that simply lost both maps."""
    mine = store.put(TENANT, MAPPING)
    theirs = store.put(OTHER_TENANT, OTHER_MAPPING)

    assert store.get(OTHER_TENANT, mine) is None
    assert store.get(TENANT, theirs) is None
    assert store.get(TENANT, mine) == MAPPING
    assert store.get(OTHER_TENANT, theirs) == OTHER_MAPPING


def test_the_key_carries_the_prefix_the_tenant_and_the_id(
    store: RedisRedactionMapStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a key built in another shape (`redaction:{map_id}`, or the two segments
    swapped): the key is what a second implementation, a migration and an operator's
    `SCAN` all depend on, and it is asserted against Redis rather than against the store's
    own `key()` alone."""
    map_id = store.put(TENANT, MAPPING)

    expected = f"{key_prefix}:{TENANT}:{map_id}"
    assert store.key(TENANT, map_id) == expected
    assert redis_client.exists(expected) == 1


def test_the_default_prefix_is_the_one_the_key_is_written_under(
    redis_client: redis.Redis,
) -> None:
    """Catches a default `key_prefix` that drifts from `REDACTION_MAP_KEY_PREFIX`: every
    other real case passes its own prefix, so the shipped default is exercised nowhere
    else - and the tenant is unique per run, so this writes into nobody's namespace."""
    store = RedisRedactionMapStore(redis_client)
    tenant = f"t{uuid.uuid4().hex}"

    map_id = store.put(tenant, MAPPING)

    assert store.key(tenant, map_id) == f"{REDACTION_MAP_KEY_PREFIX}:{tenant}:{map_id}"
    assert redis_client.exists(f"redaction:{tenant}:{map_id}") == 1


def test_the_key_expires_after_exactly_the_published_ttl(
    store: RedisRedactionMapStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a key written with `SET` and no expiry at all (`TTL` answers -1, not 3600)
    and a TTL set in the wrong unit by `PEXPIRE`: the value is read back off the server
    rather than slept for, and the exact number is what separates 3600 from a plausible
    neighbour."""
    map_id = store.put(TENANT, MAPPING)

    assert redis_client.ttl(f"{key_prefix}:{TENANT}:{map_id}") == REDACTION_MAP_TTL_SECONDS


def test_two_maps_never_share_an_id(store: RedisRedactionMapStore) -> None:
    """Catches an id derived from the mapping's contents or from a counter: two puts of
    the SAME mapping for the same tenant are the arrangement that separates real entropy
    from a hash, and the second put would otherwise overwrite the first."""
    first = store.put(TENANT, MAPPING)
    second = store.put(TENANT, MAPPING)

    assert first != second
    assert store.get(TENANT, first) == MAPPING
    assert store.get(TENANT, second) == MAPPING


def test_the_redis_store_satisfies_the_store_protocol(
    redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches method names or signatures that drift from `RedactionMapStore`: the route
    reads whichever store `create_app` chose through this protocol, so a `put` taking its
    arguments in another order is a 500 on the first redaction and `mypy --strict` is what
    the annotation below asks."""
    protocol_typed: RedactionMapStore = RedisRedactionMapStore(redis_client, key_prefix=key_prefix)

    map_id = protocol_typed.put(TENANT, MAPPING)

    assert protocol_typed.get(TENANT, map_id) == MAPPING
