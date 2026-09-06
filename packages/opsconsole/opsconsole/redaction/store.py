"""Where a placeholder map lives between the call that made it and the one that reads it.

Redaction is only half a control. `redact` turns values into `<KIND_N>`
placeholders and hands back the map that inverts them; something has to hold that
map for the hour in which a caller may want its text back (#20, an earlier change). This
module is that seam and nothing else - it does not redact, and it never looks
inside a mapping it is given.

Two things are deliberate here rather than incidental.

The key is `(tenant, map_id)`, not the id alone. A map holds the raw personal
data a request carried, so a tenant reading another tenant's map is the one
failure this store must not have; pinned by
`test_one_tenant_cannot_read_another_tenants_map`, which reads with an id that is
genuinely valid for the other tenant so "unknown id" cannot be what makes it
pass.

The clock is injected. The TTL boundary is a criterion with an exact number on
it, and a test that pinned it by sleeping for an hour is not a test - `FakeClock`
in `tests/redaction/test_store.py` moves time by hand, and the pair at
`REDACTION_MAP_TTL_SECONDS` and one second past it is what pins `>` where `>=`
would be wrong.

`RedactionMapStore` is a Protocol because the in-process implementation was
temporary and known to be. It keeps maps in the process, so two gateway replicas
do not share them. `RedisRedactionMapStore` below is the
second implementation of the same protocol rather than a replacement:
`create_app` picks it when `OPSCONSOLE_REDIS_URL` is set and keeps the
in-process one otherwise, so a local run and the suite
still need no backing service.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable, Mapping
from typing import Final, Protocol

import redis

from opsconsole.redaction.errors import RedactionMapStoreUnavailable
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

REDACTION_MAP_TTL_SECONDS: Final[int] = 3600
"""How long a placeholder map stays readable, in seconds (the design's one hour).

Pinned exactly by `test_the_ttl_is_exactly_one_hour` and bracketed by
`test_a_map_is_readable_up_to_and_including_the_ttl` and
`test_a_map_is_gone_past_the_ttl`. The boundary is inclusive: a map is still
there at exactly this many seconds and gone one second later."""

REDACTION_MAP_ID_BYTES: Final[int] = 16
"""How much randomness a map id carries, before hex rendering doubles its length.

A map id is the only thing standing between a caller and the raw values behind
someone else's placeholders, so it is drawn from `secrets` and never from a
counter or a hash of the mapping. Pinned exactly by
`test_the_id_is_exactly_sixteen_random_bytes`, with the rendered length pinned
against both this constant and the literal 32 by
`test_an_id_is_thirty_two_lowercase_hex_characters`."""


class RedactionMapStore(Protocol):
    """Put a placeholder map away under a tenant; read it back by id until it expires.

    The contract is two methods and one rule: `get` answers `None` for an id this
    tenant did not write or whose map has expired, and raises for neither
    (`test_an_unknown_id_answers_absent_rather_than_raising`). A store that raised
    would make "expired" and "wrong tenant" distinguishable from the outside,
    which is exactly the difference a caller must not be able to measure.
    """

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        """Store one map for `tenant` and answer the fresh id it was stored under."""
        ...

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        """The map `tenant` stored under `map_id`, or `None` if there is not one."""
        ...


class InProcessRedactionMapStore:
    """A `RedactionMapStore` holding everything in one process's memory.

    Expiry is lazy: a map is dropped when it is read past its TTL rather than by
    a sweeper, so there is no background task and no lock. That leaves expired
    entries occupying memory until someone asks for them, which is the trade a
    single G0 node can afford and a reason this implementation is not the one
    that ships past an earlier change.

    `clock` defaults to `time.monotonic` rather than `time.time`: a wall clock
    stepped backwards by NTP would resurrect an expired map, and monotonic time
    cannot be stepped. The default exists so `create_app` can build one with no
    arguments (`test_the_clock_parameter_has_a_default`).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._maps: dict[tuple[str, str], tuple[float, Mapping[str, str]]] = {}

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        """Store `mapping` under a fresh id and answer it.

        The mapping is copied, so a caller that goes on to mutate what it handed
        over cannot change what a later `get` answers. The id is drawn fresh
        every call, so two identical mappings stored at the same instant get
        different ids (`test_two_maps_never_share_an_id`).
        """
        map_id = secrets.token_hex(REDACTION_MAP_ID_BYTES)
        self._maps[(tenant, map_id)] = (self._clock(), dict(mapping))
        return map_id

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        """The stored map, or `None` when this tenant has no live map under that id.

        `>` and not `>=`: the map is still readable at exactly
        `REDACTION_MAP_TTL_SECONDS` and gone the second after.
        """
        entry = self._maps.get((tenant, map_id))
        if entry is None:
            return None
        stored_at, mapping = entry
        if self._clock() - stored_at > REDACTION_MAP_TTL_SECONDS:
            del self._maps[(tenant, map_id)]
            return None
        return mapping


REDACTION_MAP_KEY_PREFIX: Final[str] = "redaction"
"""The first segment of every Redis key a map is stored under.

A live gateway's maps are addressed by this string, so renaming it orphans every
map a caller is still holding an id for; pinned by
`test_the_key_prefix_is_the_published_one`."""

MAP_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")
r"""What a map id must look like before it is allowed to shape a Redis key.

`REDACTION_MAP_ID_BYTES` rendered as hex, which is what `put` mints. Used with
`fullmatch` and never with `match`: in Python `$` also matches
before a trailing newline, so `match` would let `<32 hex>\n` through and put a
caller-controlled line break into a key. The rejection table - 31, 33,
uppercase, non-hex, a trailing newline, a trailing CRLF, a trailing space, a
colon and a glob character - is
`test_get_refuses_a_malformed_map_id_before_touching_redis`."""


class RedisRedactionMapStore:
    """A `RedactionMapStore` keeping maps in Redis, where every replica can read them.

    The implementation the gateway ships with once `OPSCONSOLE_REDIS_URL` is set. It differs from `InProcessRedactionMapStore` in three
    ways that matter.

    Expiry is the server's. The key is written by `SET ... EX`, which carries
    `REDACTION_MAP_TTL_SECONDS` in the same round trip, so a map is dropped by
    Redis whether or not anyone asks for it again and there is no window in
    which a map exists without its expiry. (`SETEX` says the same thing and is
    deprecated in the installed client.) The TTL is read back off the server
    rather than slept for
    (`test_the_key_expires_after_exactly_the_published_ttl`). There is no clock
    to inject here, which is why this class takes none.

    Both key segments are pre-checked before any command is sent (standing rule
    1). `tenant` and `map_id` are client-supplied strings that shape the key, and
    a colon in either would forge a segment and address another tenant's
    namespace: `tenant` is checked against the gateway's own
    `IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH`, `map_id` against
    `MAP_ID_PATTERN`. A rejected value raises `IdentifierInvalid` and Redis is
    never called at all, which is what the tests assert rather than "no key
    exists afterwards"
    (`test_put_refuses_a_malformed_tenant_before_touching_redis`,
    `test_get_refuses_a_malformed_tenant_before_touching_redis`,
    `test_get_refuses_a_malformed_map_id_before_touching_redis`).

    Every `redis.RedisError` becomes `RedactionMapStoreUnavailable`, chained. A
    store that is down is an outbound dependency failing, not a missing map, so
    `get` raises rather than answering `None`
    (`test_a_get_against_an_unreachable_redis_raises_the_typed_failure`).

    `key_prefix` exists for the suite: per-test isolation is a namespace rather
    than a `FLUSHDB` that would delete keys this suite does not own (standing
    rule 7). The default is the shipped one, exercised by
    `test_the_default_prefix_is_the_one_the_key_is_written_under`.

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, so a parametrised annotation fails `mypy --strict` with `type-arg`.
    """

    def __init__(self, client: redis.Redis, *, key_prefix: str = REDACTION_MAP_KEY_PREFIX) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def key(self, tenant: str, map_id: str) -> str:
        """The Redis key one tenant's map is stored under.

        Public because it is what a migration, an operator's `SCAN` and the test
        that reads the key back all depend on
        (`test_the_key_carries_the_prefix_the_tenant_and_the_id`).
        """
        return f"{self._key_prefix}:{tenant}:{map_id}"

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        """Store `mapping` for `tenant` under a fresh id, expiring on the shared TTL.

        The id is drawn from `secrets` exactly as the in-process store draws it,
        so two identical mappings stored at the same instant get different ids
        and neither overwrites the other (`test_two_maps_never_share_an_id`).
        """
        _checked_tenant(tenant)
        map_id = secrets.token_hex(REDACTION_MAP_ID_BYTES)
        try:
            self._client.set(
                self.key(tenant, map_id),
                json.dumps(dict(mapping)),
                ex=REDACTION_MAP_TTL_SECONDS,
            )
        except redis.RedisError as exc:
            raise RedactionMapStoreUnavailable("the redaction map store is unreachable") from exc
        return map_id

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        """The map `tenant` stored under `map_id`, or `None` when there is not one.

        `None` covers an id nobody wrote, an id that has expired, and an id that
        is live for a DIFFERENT tenant - the last because the tenant is part of
        the key, so the read simply misses
        (`test_one_tenant_cannot_read_another_tenants_live_map`).
        """
        _checked_tenant(tenant)
        _checked_map_id(map_id)
        try:
            stored = self._client.get(self.key(tenant, map_id))
        except redis.RedisError as exc:
            raise RedactionMapStoreUnavailable("the redaction map store is unreachable") from exc
        if stored is None:
            return None
        decoded: dict[str, str] = json.loads(stored)
        return decoded


def _checked_tenant(tenant: str) -> None:
    """Refuse a tenant that may not shape a key, before any command is sent.

    The gateway's own identifier rules, imported rather than restated, so the
    name a key store accepts and the name `gateway/context.py` authenticated can
    never disagree. Inclusive at the cap: exactly `MAX_IDENTIFIER_LENGTH`
    characters is accepted (`test_a_tenant_at_the_length_cap_is_accepted`,
    `test_a_tenant_one_character_over_the_cap_is_rejected`).
    """
    if len(tenant) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"tenant: longer than {MAX_IDENTIFIER_LENGTH} characters", param="tenant"
        )
    if IDENTIFIER_PATTERN.fullmatch(tenant) is None:
        raise IdentifierInvalid("tenant: not a valid identifier", param="tenant")


def _checked_map_id(map_id: str) -> None:
    """Refuse a map id that may not shape a key, before any command is sent."""
    if MAP_ID_PATTERN.fullmatch(map_id) is None:
        raise IdentifierInvalid("redaction_map_id: not a valid map id", param="redaction_map_id")
