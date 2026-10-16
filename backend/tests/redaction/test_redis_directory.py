r"""`RedisDirectoryStore`: the key it writes, the hashed field in it, and the pre-check.

The same two halves, and the same split, as
`tests/redaction/test_redis_store.py`.

The half that needs a REAL Redis is the half whose claims are Redis semantics:
the round trip, the key shape, the hash type, and the field read back off the
server with `HGET`. A fake would prove those against a reimplementation of Redis
rather than against Redis. Redis is required by an explicit conditional on
`OPSCONSOLE_TEST_REDIS_URL`; the fixture below skips with that
variable's name in the reason, so a suite that silently checked nothing is
legible in the run's own output rather than inferred. Deliberately NOT
`OPSCONSOLE_REDIS_URL`, which `Settings` itself reads:
exporting that one to run this file would hand every other test in the suite a
Redis-backed gateway it never asked for, and the root `tests/conftest.py` clears
it for exactly that reason.

The half that needs NOTHING is the validation half: the tenant pre-check and the
three caps run before any call reaches Redis, so a client that records its calls
and refuses to answer is a sharper witness than a live server - "no name was
stored" is asserted as "Redis was never called at all", which is the claim the
criterion actually makes. The unavailable-store mapping lives there too, over a
client that raises `redis.RedisError`.

a project rule: one session-scoped client, built once. Per-test isolation is a
unique `key_prefix` handed to the store, never `FLUSHDB` (which would delete a
neighbouring suite's keys) and never a fresh client. A directory key carries no
TTL - unlike a placeholder map, it is meant to outlive the request - so the
`store` fixture deletes the keys IT wrote, by their exact names, when the test
ends. That is not a flush: no key this file did not write is named.

a project rule on the tenant: the rejection table carries a trailing newline, a
trailing CRLF and a trailing space, because a check written `re.match` with
`$` accepts the first of those and would put a caller-controlled line break into
a Redis key.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import cast

import pytest
import redis

from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.redaction.errors import DirectoryNameInvalid, DirectoryStoreUnavailable
from app.redaction.names_store import (
    DIRECTORY_KEY_PREFIX,
    MAX_DIRECTORY_NAME_CHARS,
    MAX_DIRECTORY_NAMES,
    DirectoryStore,
    RedisDirectoryStore,
    name_field,
    normalise_name,
)

TENANT = "nordlicht"
OTHER_TENANT = "cuan"
TENANTS = (TENANT, OTHER_TENANT)

NAME_A = "Anna Schmidt"
NAME_B = "Bernd Kowalski"


def as_client(fake: object) -> redis.Redis:
    """Hand a recording double to a store typed for a real client.

    The store is annotated against `redis.Redis`, and the cases in this half of
    the file are about calls that never leave the process; casting keeps the
    double honest about its methods without a `# type: ignore` on every call.
    """
    return cast(redis.Redis, fake)


class RecordingRedis:
    """A client that records every call and never talks to anything.

    `error` is what every command raises when it is set, which is how the
    unreachable-Redis rows are arranged; `calls` is what makes "the pre-check ran
    before any effect" observable, because an empty list is the only evidence that
    nothing was written.
    """

    def __init__(
        self, error: Exception | None = None, hash_: dict[bytes, bytes] | None = None
    ) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self._error = error
        self._hash = hash_ or {}

    def _record(self, command: str, *args: object) -> None:
        self.calls.append((command, args))
        if self._error is not None:
            raise self._error

    def hset(self, name: str, mapping: Mapping[str, str]) -> int:
        self._record("hset", name, dict(mapping))
        return len(mapping)

    def hgetall(self, name: str) -> dict[bytes, bytes]:
        self._record("hgetall", name)
        return dict(self._hash)

    def hlen(self, name: str) -> int:
        self._record("hlen", name)
        return len(self._hash)


def fake_store(
    *, error: Exception | None = None, hash_: dict[bytes, bytes] | None = None
) -> tuple[RedisDirectoryStore, RecordingRedis]:
    """A store over a recording client, and the client, so calls can be asserted."""
    client = RecordingRedis(error=error, hash_=hash_)
    return RedisDirectoryStore(as_client(client)), client


def commands(client: RecordingRedis) -> list[str]:
    """Just the command names, for the rows about which side effect happened."""
    return [call[0] for call in client.calls]


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
    otherwise write another tenant's directory, which is this store's one unacceptable
    failure, the newline rows are what separate `fullmatch` from `match` (standing rule
    4), and asserting the client saw NO call is what pins "writes nothing" rather than
    "writes and then fails"."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.put(tenant, [NAME_A])

    assert client.calls == []


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_get_refuses_a_malformed_tenant_before_touching_redis(case: str, tenant: str) -> None:
    """Catches a pre-check applied to `put` alone: `get` builds the same key from the same
    client-supplied string, and this is the read every redacted request makes, so an
    unchecked tenant here compiles ANOTHER tenant's names into the matcher."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.get(tenant)

    assert client.calls == []


def test_the_tenant_rejection_names_the_tenant_parameter() -> None:
    """Catches a refusal filed under the wrong `param`: the route renders this as a 422
    naming the field the caller must fix, and `redaction_map_id` or an empty `param` sends
    them looking at the wrong one."""
    store, _ = fake_store()

    with pytest.raises(IdentifierInvalid) as raised:
        store.put("Nordlicht", [NAME_A])

    assert raised.value.param == "tenant"


def test_a_tenant_at_the_length_cap_is_accepted() -> None:
    """Catches an exclusive comparison (`>=` where `>` belongs) on the length cap: exactly
    `MAX_IDENTIFIER_LENGTH` characters is a legal tenant name everywhere else in the
    gateway, so refusing it here would make a tenant that authenticates unable to have a
    directory at all."""
    store, client = fake_store()

    store.put(MAX_TENANT, [NAME_A])

    assert "hset" in commands(client)


def test_a_tenant_one_character_over_the_cap_is_rejected() -> None:
    """The other half of the boundary pair: catches a cap that was never applied, which
    the accepted row alone cannot see."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.put(OVER_CAP_TENANT, [NAME_A])

    assert client.calls == []


@pytest.mark.parametrize(
    ("case", "names"),
    [
        ("one character under the minimum", ["L"]),
        ("an empty name", [""]),
        ("one character over the maximum", ["a" * (MAX_DIRECTORY_NAME_CHARS + 1)]),
        ("one name over the count cap", [f"Persona{index:05d}" for index in range(2001)]),
        ("a bad name last in an otherwise legal push", [NAME_A, "L"]),
    ],
)
def test_put_refuses_a_name_outside_the_caps_before_touching_redis(
    case: str, names: Sequence[str]
) -> None:
    """Catches caps applied only by the in-process store: this is the implementation that
    ships, so a Redis store that writes first and validates never - or validates per name
    inside its write loop - leaves a tenant's directory half applied with no way to tell
    which half. The count row is `MAX_DIRECTORY_NAMES` plus one exactly."""
    store, client = fake_store()

    with pytest.raises(DirectoryNameInvalid):
        store.put(TENANT, names)

    assert "hset" not in commands(client)


def test_a_full_legal_push_reaches_redis() -> None:
    """Catches a pre-check so strict that no push survives it - every row in the group
    above asserts a refusal, so a `put` that raised for everything would pass all of them
    and this is the row that separates the two readings."""
    store, client = fake_store()

    store.put(TENANT, [f"Persona{index:05d}" for index in range(MAX_DIRECTORY_NAMES)])

    assert "hset" in commands(client)


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
    the typed failure to answer a 502 PROVIDER_ERROR, so a raw client error there is a 500
    with no code - and the rows are the four the client actually raises, not one."""
    store, _ = fake_store(error=error)

    with pytest.raises(DirectoryStoreUnavailable):
        store.put(TENANT, [NAME_A])


@pytest.mark.parametrize(("case", "error"), REDIS_FAILURES, ids=[c[0] for c in REDIS_FAILURES])
def test_a_get_against_an_unreachable_redis_raises_the_typed_failure(
    case: str, error: Exception
) -> None:
    """Catches a `get` that answers an EMPTY TUPLE on a connection failure, which is the
    dangerous direction decision H exists for: the request would be redacted without the
    tenant's names and served, so the very values this control removes would leave the box
    while the response still reported a redaction."""
    store, _ = fake_store(error=error)

    with pytest.raises(DirectoryStoreUnavailable):
        store.get(TENANT)


def test_the_typed_failure_does_not_swallow_the_cause() -> None:
    """Catches a `raise ... from None`: the client's own error is the only description of
    WHY Redis was unreachable, and the log line an operator reads is built from the
    chained cause."""
    original = redis.ConnectionError("connection refused")
    store, _ = fake_store(error=original)

    with pytest.raises(DirectoryStoreUnavailable) as raised:
        store.get(TENANT)

    assert raised.value.__cause__ is original


# --------------------------------------------------------------------------
# against a real Redis
# --------------------------------------------------------------------------


@pytest.fixture
def store(redis_client: redis.Redis, key_prefix: str) -> Iterator[RedisDirectoryStore]:
    """The store under test, namespaced to this test and sharing the session's client.

    `redis_client` and `key_prefix` come from the top `tests/conftest.py`, built
    once per run; this file builds no client of its own.

    A directory key has no TTL, so the teardown deletes the two keys this file can
    have written under this test's prefix, by their exact names. Naming them is
    what keeps it a cleanup rather than a flush.
    """
    built = RedisDirectoryStore(redis_client, key_prefix=key_prefix)
    yield built
    redis_client.delete(*(built.key(tenant) for tenant in TENANTS))


def test_a_pushed_name_round_trips_through_redis(store: RedisDirectoryStore) -> None:
    """Catches a store that writes the plaintext and reads back bytes it never decodes:
    the matcher compiles whatever `get` answers, and a `b'Anna Schmidt'` repr compiled
    into the alternation matches nothing a caller ever sends."""
    store.put(TENANT, [NAME_A, NAME_B])

    assert sorted(store.get(TENANT)) == sorted([NAME_A, NAME_B])


def test_an_empty_directory_answers_an_empty_tuple(store: RedisDirectoryStore) -> None:
    """Catches a `get` that answers `None` for a tenant with no key at all: every redacted
    request makes this read, so `None` is a 500 on the first request of every tenant that
    has not pushed anything."""
    assert store.get(TENANT) == ()


def test_the_key_carries_the_prefix_and_the_tenant(
    store: RedisDirectoryStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a key built in another shape, and a directory written as a STRING or a SET
    rather than the hash decision B chose: the type is read back off the server, because a
    set answers the same `get` and has no per-entry handle at all."""
    store.put(TENANT, [NAME_A])

    expected = f"{key_prefix}:{TENANT}"
    assert store.key(TENANT) == expected
    assert redis_client.exists(expected) == 1
    assert redis_client.type(expected) == b"hash"


def test_the_field_is_the_hashed_name_and_the_value_is_the_plaintext(
    store: RedisDirectoryStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a hash keyed on the plaintext name, which is the design's "hashed" half thrown
    away: the field is read straight off the server under the digest this suite computes
    itself, so a store hashing something else - the raw string, the tenant plus the name -
    misses."""
    store.put(TENANT, [NAME_A])

    key = f"{key_prefix}:{TENANT}"
    assert redis_client.hget(key, name_field(NAME_A)) == NAME_A.encode("utf-8")
    assert redis_client.hget(key, normalise_name(NAME_A)) is None


def test_re_pushing_a_name_leaves_the_directory_unchanged(
    store: RedisDirectoryStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a store appending to a list, or hashing the raw string: the second push
    spells the name differently, so a raw-keyed hash grows to two fields and the tenant
    burns a slot of the 2000 on a name it already had."""
    store.put(TENANT, [NAME_A])

    store.put(TENANT, ["  anna schmidt  "])

    assert redis_client.hlen(f"{key_prefix}:{TENANT}") == 1
    assert len(store.get(TENANT)) == 1


def test_a_second_push_merges_rather_than_replacing(store: RedisDirectoryStore) -> None:
    """Catches a `DEL` before the write, or a `HSET` of a freshly built mapping over a
    deleted key: decision C is merge, and replace-whole makes a paged push silently delete
    everything the caller already stored."""
    store.put(TENANT, [NAME_A])

    store.put(TENANT, [NAME_B])

    assert sorted(store.get(TENANT)) == sorted([NAME_A, NAME_B])


def test_one_tenant_cannot_read_another_tenants_directory(store: RedisDirectoryStore) -> None:
    """Catches a store keyed on the prefix alone: one tenant's staff names would be
    compiled into every other tenant's matcher, which leaks who works where AND wrecks
    precision for everyone. The two closing reads prove the refusals are not a store that
    lost both directories."""
    store.put(TENANT, [NAME_A])
    store.put(OTHER_TENANT, [NAME_B])

    assert store.get(TENANT) == (NAME_A,)
    assert store.get(OTHER_TENANT) == (NAME_B,)


def test_the_default_prefix_is_the_one_the_key_is_written_under(
    redis_client: redis.Redis,
) -> None:
    """Catches a default `key_prefix` that drifts from `DIRECTORY_KEY_PREFIX`: every other
    real case passes its own prefix, so the shipped default is exercised nowhere else -
    and the tenant is unique per run, so this writes into nobody's namespace."""
    built = RedisDirectoryStore(redis_client)
    tenant = f"t{uuid.uuid4().hex}"
    try:
        built.put(tenant, [NAME_A])

        assert built.key(tenant) == f"{DIRECTORY_KEY_PREFIX}:{tenant}"
        assert redis_client.exists(f"directory:{tenant}") == 1
    finally:
        redis_client.delete(built.key(tenant))


def test_the_redis_store_satisfies_the_directory_protocol(
    redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches method names or signatures that drift from `DirectoryStore`: the routes
    read whichever store `create_app` chose through this protocol, so a `put` taking a
    single name instead of a sequence is a 500 on the first push, and `mypy --strict` is
    what the annotation below asks."""
    protocol_typed: DirectoryStore = RedisDirectoryStore(redis_client, key_prefix=key_prefix)
    try:
        protocol_typed.put(TENANT, [NAME_A])

        assert protocol_typed.get(TENANT) == (NAME_A,)
    finally:
        redis_client.delete(f"{key_prefix}:{TENANT}")
