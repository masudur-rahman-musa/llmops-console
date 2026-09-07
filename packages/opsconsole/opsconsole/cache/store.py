"""Where an exact-match answer lives for the day after it was bought.

The shape mirrors `redaction/store.py` on purpose - a Protocol, an in-process
implementation and a Redis one, picked by `OPSCONSOLE_REDIS_URL` in `create_app`
(decision C, the same switch an earlier change decision B chose for the map store) - so
someone who has read that file can read this one. Two things about it are
deliberately DIFFERENT, and both follow from what a cache is.

**A failure here is a miss, never an error.** `RedisRedactionMapStore.get`
raises `RedactionMapStoreUnavailable` when Redis is down, because a caller
holding a map id asked for something specific and "absent" would be a lie. This
store has no such duty: nobody asked for the entry, and a cache that cannot
answer simply has not got one. So every `redis.RedisError` is swallowed, `get`
answers `None` and `put` returns - pinned by
`test_a_get_against_an_unreachable_redis_answers_absent` and
`test_a_put_against_an_unreachable_redis_is_swallowed`, four failure rows each,
and end to end by `test_a_cache_that_raises_still_serves_the_request`. A
`ValidationError` off an unreadable stored value is swallowed the same way
(`test_unreadable_stored_bytes_answer_absent_rather_than_raising`): a value
written by an older build is a cache entry this build cannot use, which is a
miss and not a 500.

**The value is the NEUTRAL `ChatResponse`, not the wire body** (decision E).
`to_wire` then runs on a hit exactly as it runs on a miss, so a replayed answer
is rendered by the same code and gets its OWN completion id rather than
replaying another request's; and the wire shape is not frozen into a store that
outlives it by a day.

The rest is the map store's argument unchanged. The key is `(tenant, key)`, so
one tenant cannot read another's entry
(`test_one_tenant_cannot_read_another_tenants_entry`,
`test_one_tenant_cannot_read_another_tenants_live_entry`, both reading with a
key that is genuinely live for the other tenant so "unknown key" cannot be what
makes them pass). The clock is injected, so the TTL boundary is a pair of
assertions rather than a day of waiting. Both key segments are pre-checked
before any command is sent, against the gateway's own
`IDENTIFIER_PATTERN` and `CACHE_ID_PATTERN` rather than a second copy of either.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Final, Protocol

import redis
from pydantic import ValidationError

from opsconsole.cache.key import CACHE_ID_PATTERN, CACHE_KEY_PREFIX
from opsconsole.providers.base import ChatResponse
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

RESPONSE_CACHE_TTL_SECONDS: Final[int] = 86400
"""How long an exact-match answer stays readable, in seconds (#22's twenty-four hours).

Pinned exactly by `test_the_ttl_is_exactly_one_day`, which asserts it against
both the literal and `24 * 60 * 60`, and bracketed by
`test_an_entry_is_readable_up_to_and_including_the_ttl` and
`test_an_entry_is_gone_past_the_ttl`. The boundary is inclusive: an entry is
still there at exactly this many seconds and gone one second later."""


class ResponseCache(Protocol):
    """Put an answer away under a tenant and a key; read it back until it expires.

    Two methods and one rule, and the rule is the opposite of
    `RedactionMapStore`'s: NEITHER method raises, for any reason a store can
    have. `get` answers `None` for a key this tenant did not write, for one that
    has expired, for a value it cannot read and for a store that is unreachable;
    `put` returns whether or not the write landed. An implementation that raised
    would turn a Redis incident into 5xx on requests a provider would happily
    have served, which is the one failure a CACHE must not cause.

    The two segments a caller supplies still have to be well formed - a
    malformed tenant or key raises `IdentifierInvalid` BEFORE any store call. That is not a store failure: it is a programming error at
    the call site, and the route never reaches it with anything but the tenant
    `gateway/context.py` authenticated and a key `cache_key` minted.
    """

    def get(self, tenant: str, key: str) -> ChatResponse | None:
        """The answer `tenant` stored under `key`, or `None` if there is not one."""
        ...

    def put(self, tenant: str, key: str, answer: ChatResponse) -> None:
        """Store one answer for `tenant` under `key`, expiring on the shared TTL."""
        ...


class InProcessResponseCache:
    """A `ResponseCache` holding everything in one process's memory.

    What a bare local run and the suite get when `OPSCONSOLE_REDIS_URL` is unset,
    so neither needs a backing service. Two replicas do not share it, which for a
    cache costs a hit rate rather than correctness - the reason this
    implementation is acceptable where the redaction map's in-process twin was
    always temporary.

    Expiry is lazy, exactly as `InProcessRedactionMapStore`'s is: an entry is
    dropped when it is read past its TTL rather than by a sweeper, so there is no
    background task and no lock, and an entry nobody asks for again occupies
    memory until the process ends.

    `clock` defaults to `time.monotonic` rather than `time.time`: a wall clock
    stepped backwards by NTP would resurrect an expired entry, and monotonic time
    cannot be stepped. The default exists so `create_app` can build one with no
    arguments (`test_the_clock_parameter_has_a_default`).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[tuple[str, str], tuple[float, ChatResponse]] = {}

    def get(self, tenant: str, key: str) -> ChatResponse | None:
        """The stored answer, or `None` when this tenant has no live entry under that key.

        `>` and not `>=`: the entry is still readable at exactly
        `RESPONSE_CACHE_TTL_SECONDS` and gone the second after.
        """
        entry = self._entries.get((tenant, key))
        if entry is None:
            return None
        stored_at, answer = entry
        if self._clock() - stored_at > RESPONSE_CACHE_TTL_SECONDS:
            del self._entries[(tenant, key)]
            return None
        return answer

    def put(self, tenant: str, key: str, answer: ChatResponse) -> None:
        """Store `answer` under `(tenant, key)`, replacing whatever was there.

        A `ChatResponse` is frozen, so it is filed as it stands rather than
        copied: a caller cannot mutate what a later `get` answers.
        """
        self._entries[(tenant, key)] = (self._clock(), answer)


class RedisResponseCache:
    """A `ResponseCache` keeping answers in Redis, where every replica can read them.

    What the gateway ships with once `OPSCONSOLE_REDIS_URL` is set. It differs
    from `InProcessResponseCache` in the way `RedisRedactionMapStore` differs
    from its in-process twin, plus the swallow this whole module is built around.

    Expiry is the server's. The key is written by `SET ... EX`, which carries
    `RESPONSE_CACHE_TTL_SECONDS` in the same round trip, so an entry is dropped
    by Redis whether or not anyone asks for it again and there is no window in
    which one exists without its expiry. The TTL is read back off the server
    rather than slept for (`test_the_key_expires_after_exactly_the_published_ttl`)
    and off the call itself (`test_put_sends_the_ttl_with_the_write`). There is
    no clock to inject here, which is why this class takes none.

    Both key segments are pre-checked before any command is sent (standing rule
    1). `tenant` is a client-supplied string and a colon in it would forge a
    segment and address another tenant's namespace; `key` is minted by
    `cache_key`, and checking it anyway is what keeps that guarantee from
    depending on one call site
    (`test_put_refuses_a_malformed_tenant_before_touching_redis`,
    `test_get_refuses_a_malformed_tenant_before_touching_redis`,
    `test_get_refuses_a_malformed_key_before_touching_redis`,
    `test_put_refuses_a_malformed_key_before_touching_redis`, each asserting the
    client recorded no call at all).

    `key_prefix` exists for the suite: per-test isolation is a namespace rather
    than a `FLUSHDB` that would delete keys this suite does not own (standing
    rule 7). The default is the shipped one, exercised by
    `test_the_default_prefix_is_the_one_the_key_is_written_under`.

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, so a parametrised annotation fails `mypy --strict` with `type-arg`.
    """

    def __init__(self, client: redis.Redis, *, key_prefix: str = CACHE_KEY_PREFIX) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def key(self, tenant: str, key: str) -> str:
        """The Redis key one tenant's cached answer is stored under.

        Public because it is what an operator's `SCAN`, a future invalidation and
        the test that reads the key back all depend on
        (`test_the_key_carries_the_prefix_the_tenant_and_the_cache_key`).
        """
        return f"{self._key_prefix}:{tenant}:{key}"

    def get(self, tenant: str, key: str) -> ChatResponse | None:
        """The answer `tenant` stored under `key`, or `None` when there is not one.

        `None` covers a key nobody wrote, one that has expired, one that is live
        for a DIFFERENT tenant (the tenant is part of the Redis key, so the read
        simply misses), a value this build cannot parse, and a Redis that is not
        answering at all. A caller cannot tell those apart, and for a cache there
        is nothing it could do differently if it could.
        """
        _checked_tenant(tenant)
        _checked_key(key)
        try:
            stored = self._client.get(self.key(tenant, key))
        except redis.RedisError:
            return None
        if stored is None:
            return None
        try:
            return ChatResponse.model_validate_json(stored)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def put(self, tenant: str, key: str, answer: ChatResponse) -> None:
        """Store `answer` for `tenant` under `key`, expiring on the shared TTL.

        Swallows the same failures `get` does, and for a sharper reason: the
        write happens after a provider has already answered and been paid, so an
        escaping error would fail a request that was fully served.
        """
        _checked_tenant(tenant)
        _checked_key(key)
        try:
            self._client.set(
                self.key(tenant, key),
                answer.model_dump_json(),
                ex=RESPONSE_CACHE_TTL_SECONDS,
            )
        except redis.RedisError:
            return


def _checked_tenant(tenant: str) -> None:
    """Refuse a tenant that may not shape a key, before any command is sent.

    The gateway's own identifier rules, imported rather than restated, so the
    name a cache accepts and the name `gateway/context.py` authenticated can
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


def _checked_key(key: str) -> None:
    """Refuse a cache key that may not shape a Redis key, before any command is sent.

    `fullmatch` against `CACHE_ID_PATTERN`, because `$` in Python also matches
    before a trailing newline; the rejection table is
    `test_get_refuses_a_malformed_key_before_touching_redis` and its `put` twin.
    """
    if CACHE_ID_PATTERN.fullmatch(key) is None:
        raise IdentifierInvalid("cache key: not a valid cache key", param="key")
