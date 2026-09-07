"""The response cache: its TTL, its boundary, its tenant scoping, and its absent answers.

Everything here is in-process and the clock is injected, so no test sleeps and no
backing service is started; a project rule has nothing to apply to beyond that.

Three claims carry the plan's criteria and each is arranged so that only the
control it names can decide the outcome:

* the TTL is a boundary PAIR against one clock (readable at exactly
  `RESPONSE_CACHE_TTL_SECONDS`, gone one second later), so a `>=` written where a
  `>` belongs reddens on the first row and a store that never expires anything
  reddens on the second;
* the value is pinned against the constant AND against the literal 86400, so an
  implementation that changed the constant in step with the code still fails;
* tenant scoping is asserted with a key that IS live - for the other tenant - so
  "unknown key" cannot be what makes the case pass.

The absent answers are the reason this store is not `RedisRedactionMapStore`: a
cache is not a control. An unknown key, an expired entry and a broken store all
answer `None` rather than raising, because a raise here would turn a Redis
incident into a 5xx on requests the provider could have served (plan §5).
"""

from __future__ import annotations

import pytest

from opsconsole.cache import (
    RESPONSE_CACHE_TTL_SECONDS,
    InProcessResponseCache,
    ResponseCache,
)

from .conftest import OTHER_TENANT, START, TENANT, FakeClock, cached_answer

KEY = "0123456789abcdef" * 4
"""64 lowercase hex characters: the shape `cache_key` mints and the Redis store
pre-checks. The in-process store does not parse it, so it is here to keep both
implementations of the protocol exercised with the same value."""

OTHER_KEY = "fedcba9876543210" * 4

ONE_DAY = 24 * 60 * 60


def test_the_ttl_is_exactly_one_day() -> None:
    """Catches a TTL written in minutes, or set to a plausible neighbour like 3600 (the
    redaction map's) or 604800: #22 says twenty-four hours, a band would admit every one
    of those, and the third assertion is what a reader checks the arithmetic against."""
    assert RESPONSE_CACHE_TTL_SECONDS == 86400
    assert RESPONSE_CACHE_TTL_SECONDS == ONE_DAY


def test_a_stored_answer_round_trips_every_field() -> None:
    """Catches a store that keeps the content and drops the rest: the cached value is the
    NEUTRAL answer (decision E), so the tool calls, the finish reason and the usage are
    what the output path re-renders from - a hit that lost the tool calls would answer
    text where the caller expected a call."""
    store = InProcessResponseCache(clock=FakeClock())
    answer = cached_answer()

    store.put(TENANT, KEY, answer)
    read = store.get(TENANT, KEY)

    assert read == answer


@pytest.mark.parametrize("elapsed", [0.0, 1.0, 86399.0, 86400.0])
def test_an_entry_is_readable_up_to_and_including_the_ttl(elapsed: float) -> None:
    """Catches an expiry written `now - stored_at >= TTL` (or an off-by-one on the clock):
    the 86400.0 row is exactly the moment the plan says the entry is still there, and the
    earlier rows keep a store that expires everything immediately from passing the pair by
    accident."""
    clock = FakeClock()
    store = InProcessResponseCache(clock=clock)
    store.put(TENANT, KEY, cached_answer())

    clock.now = START + elapsed

    assert store.get(TENANT, KEY) == cached_answer()


@pytest.mark.parametrize("elapsed", [86401.0, 172800.0])
def test_an_entry_is_gone_past_the_ttl(elapsed: float) -> None:
    """Catches a store that never expires anything - the key is right, the tenant is
    right, and the ONLY thing that can make this answer `None` is the TTL, so a plain dict
    with no clock reading fails here and nowhere else. An entry that outlives its day is a
    day-old answer served as current, which is the failure #22's window exists to bound."""
    clock = FakeClock()
    store = InProcessResponseCache(clock=clock)
    store.put(TENANT, KEY, cached_answer())

    clock.now = START + elapsed

    assert store.get(TENANT, KEY) is None


def test_an_unknown_key_answers_absent_rather_than_raising() -> None:
    """Catches a `get` written as a bare `self._entries[key]`: a miss is the ORDINARY case
    on this route - every ineligible request looks one up - so a KeyError here is a 500
    with no typed code on traffic the provider would have served happily."""
    store = InProcessResponseCache(clock=FakeClock())

    assert store.get(TENANT, KEY) is None


def test_one_tenant_cannot_read_another_tenants_entry() -> None:
    """Catches a store keyed on the cache key alone: the key handed to `get` is a REAL,
    unexpired entry for the other tenant, so "unknown key" cannot be what refuses it, and
    the two closing reads prove the refusals are not a store that lost both entries. A
    global dict here serves one tenant another tenant's completion."""
    store = InProcessResponseCache(clock=FakeClock())
    mine = cached_answer(content="the answer for evals")
    theirs = cached_answer(content="the answer for cuan")

    store.put(TENANT, KEY, mine)
    store.put(OTHER_TENANT, OTHER_KEY, theirs)

    assert store.get(OTHER_TENANT, KEY) is None
    assert store.get(TENANT, OTHER_KEY) is None
    assert store.get(TENANT, KEY) == mine
    assert store.get(OTHER_TENANT, OTHER_KEY) == theirs


def test_the_same_key_under_two_tenants_holds_two_answers() -> None:
    """The sharper half of the case above: two tenants whose requests differ ONLY by
    tenant would key the same if `cache_key` dropped the tenant, so the store must file
    them apart as well. Catches a store namespaced by nothing, which would let the second
    put overwrite the first and hand `evals` the answer `cuan` bought."""
    store = InProcessResponseCache(clock=FakeClock())
    mine = cached_answer(content="the answer for evals")
    theirs = cached_answer(content="the answer for cuan")

    store.put(TENANT, KEY, mine)
    store.put(OTHER_TENANT, KEY, theirs)

    assert store.get(TENANT, KEY) == mine
    assert store.get(OTHER_TENANT, KEY) == theirs


def test_two_keys_for_one_tenant_hold_two_answers() -> None:
    """Catches a store that keeps ONE entry per tenant (a `dict[str, ChatResponse]` keyed
    by tenant): every case above uses one key per tenant and would pass, while a live
    gateway would serve every request of a tenant the answer to its most recent one."""
    store = InProcessResponseCache(clock=FakeClock())
    first = cached_answer(content="the first answer")
    second = cached_answer(content="the second answer")

    store.put(TENANT, KEY, first)
    store.put(TENANT, OTHER_KEY, second)

    assert store.get(TENANT, KEY) == first
    assert store.get(TENANT, OTHER_KEY) == second


def test_the_in_process_store_satisfies_the_protocol() -> None:
    """Catches method names or signatures that drift from `ResponseCache`: the route reads
    whichever store `create_app` chose through this protocol, so a `put` taking its
    arguments in another order is a 500 on the first cacheable request, and the annotation
    below is what `mypy --strict` checks."""
    store: ResponseCache = InProcessResponseCache()
    answer = cached_answer()

    store.put(TENANT, KEY, answer)

    assert store.get(TENANT, KEY) == answer


def test_the_clock_parameter_has_a_default() -> None:
    """Catches a store whose `clock` parameter is required: every other case here injects
    one, so a missing default would go unnoticed in this file and turn `app.py`'s
    `InProcessResponseCache()` into a TypeError at process start."""
    store = InProcessResponseCache()
    answer = cached_answer()

    store.put(TENANT, KEY, answer)

    assert store.get(TENANT, KEY) == answer


def test_the_package_exports_the_whole_contract() -> None:
    """Catches an implementation that lands the names in `key.py` and `store.py` and never
    re-exports them: `gateway/app.py` and the chat route both import from
    `opsconsole.cache`, so a missing export is an ImportError at process start rather than
    a failure in any one test."""
    from opsconsole import cache

    exported = (
        "CACHE_KEY_PREFIX",
        "CACHE_ID_PATTERN",
        "CACHE_HEADER",
        "CACHE_ALLOW",
        "cache_key",
        "is_cacheable",
        "RESPONSE_CACHE_TTL_SECONDS",
        "ResponseCache",
        "InProcessResponseCache",
        "RedisResponseCache",
    )
    missing = [name for name in exported if not hasattr(cache, name)]

    assert missing == []
