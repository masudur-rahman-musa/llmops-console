r"""Loading a month's counter from the ledger: `seeded_spend` on both budget stores.

An earlier change. The counter used to start at zero whenever it was missing, so a tenant
whose ledger already held its month's spend was served. These cases pin the load:
the first read of a counter asks the ledger once, marks the counter loaded, and
counts on from the ledger's number.

Two halves, as in `test_redis_budget.py`. The half with no service uses the
in-process store and a recording client that answers nothing real. The half that
needs a REAL Redis (the load script and the marker's expiry are Redis semantics)
carries the `db` marker and takes `redis_client` and `key_prefix` from the root
`tests/conftest.py`.

The contract these cases are written against, before it exists:

* `budget_seeded_key(tenant, month, *, prefix)` is `budget-seeded:{tenant}:{YYYY-MM}`
  and `user_seeded_key(tenant, user_id, month, *, prefix)` is
  `budget-seeded-user:{tenant}:{user_id}:{YYYY-MM}`, both checked before any command;
* `seeded_spend(tenant, month, loader)` and `seeded_user_spend(tenant, user_id, month,
  loader)` on both stores; a loader answers micro-USD or `None` (nothing to load from);
* the Redis store asks `exists(marker)` first, then runs ONE `eval(script, 2, marker,
  counter, ledger_micro, BUDGET_KEY_TTL_SECONDS)`.
"""

from __future__ import annotations

from typing import cast

import pytest
import redis

from app.core.validation import IdentifierInvalid
from app.limits.budget import (
    BUDGET_KEY_PREFIX,
    BUDGET_KEY_TTL_SECONDS,
    InProcessBudgetStore,
    RedisBudgetStore,
    budget_seeded_key,
    user_seeded_key,
)
from app.storage.cost_ledger import LedgerUnreadable

TENANT = "nordlicht"
MONTH = "2026-10"
USER = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"

LEDGER = 15_070_000
"""What the ledger holds for the month: $15.07, above a $15.00 cap."""

EARLY_DEBIT = 500
LATER_DEBIT = 10_000
COUNTER_ONLY = 500


class CountingLoader:
    """A loader that answers a fixed value and counts how often it was asked."""

    def __init__(self, answer: int | None) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self) -> int | None:
        self.calls += 1
        return self.answer


def as_client(fake: object) -> redis.Redis:
    """Hand a recording double to a store typed for a real client."""
    return cast(redis.Redis, fake)


class ScriptedRedis:
    """A client that records every command and answers from a small script.

    `marker_present` is what `exists` answers; `counter` is what `get` and `eval`
    answer; `error` makes every command raise it; `error_on` limits that to one
    command name.
    """

    def __init__(
        self,
        *,
        marker_present: bool = False,
        counter: int = 0,
        error: Exception | None = None,
        error_on: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.marker_present = marker_present
        self.counter = counter
        self._error = error
        self._error_on = error_on

    def _record(self, command: str, *args: object) -> None:
        self.calls.append((command, args))
        if self._error is not None and self._error_on in (None, command):
            raise self._error

    def exists(self, *names: str) -> int:
        self._record("exists", *names)
        return 1 if self.marker_present else 0

    def get(self, name: str) -> bytes | None:
        self._record("get", name)
        return str(self.counter).encode()

    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> int:
        self._record("eval", script, numkeys, *keys_and_args)
        return self.counter

    def incrby(self, name: str, amount: int) -> int:
        self._record("incrby", name, amount)
        return amount

    def expire(self, name: str, seconds: int) -> bool:
        self._record("expire", name, seconds)
        return True

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


# --------------------------------------------------------------------------
# the in-process store
# --------------------------------------------------------------------------


def test_a_missing_counter_loads_the_ledger_spend_once() -> None:
    """Catches a store that asks the ledger on every request (a read of the ledger per
    request, which the plan forbids), and one that ignores the loader's answer and keeps
    answering zero, the bug that served a tenant $0.07 over its cap."""
    store = InProcessBudgetStore()
    loader = CountingLoader(LEDGER)

    first = store.seeded_spend(TENANT, MONTH, loader)
    second = store.seeded_spend(TENANT, MONTH, loader)

    assert first == LEDGER
    assert second == LEDGER
    assert loader.calls == 1


def test_the_load_replaces_a_debit_made_before_it() -> None:
    """Catches a load that ADDS the ledger total to what the counter already holds: a debit
    that landed first is also in the ledger, so adding would count it twice."""
    store = InProcessBudgetStore()
    store.debit(TENANT, MONTH, EARLY_DEBIT)

    assert store.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER)) == LEDGER


def test_a_debit_after_the_load_counts_on_from_the_ledger_total() -> None:
    """Catches a load that is not written to the counter itself (the answer is right once,
    then the next debit counts on from zero again)."""
    store = InProcessBudgetStore()
    store.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER))

    assert store.debit(TENANT, MONTH, LATER_DEBIT) == LEDGER + LATER_DEBIT
    assert store.spend(TENANT, MONTH) == LEDGER + LATER_DEBIT


def test_a_loader_answering_none_keeps_the_counter() -> None:
    """Catches a store that treats `None` (no database) as zero and overwrites the counter,
    and one that marks the counter loaded anyway so a later real answer is never used."""
    store = InProcessBudgetStore()
    store.debit(TENANT, MONTH, COUNTER_ONLY)
    nothing = CountingLoader(None)

    assert store.seeded_spend(TENANT, MONTH, nothing) == COUNTER_ONLY
    assert store.seeded_spend(TENANT, MONTH, nothing) == COUNTER_ONLY
    assert nothing.calls == 2, "a None answer set the loaded marker"

    real = CountingLoader(LEDGER)
    assert store.seeded_spend(TENANT, MONTH, real) == LEDGER
    assert real.calls == 1


def test_the_user_counter_loads_apart_from_the_tenant_counter() -> None:
    """Catches a user load that shares the tenant's marker (the second counter is never
    loaded) or writes into the tenant's counter."""
    store = InProcessBudgetStore()
    tenant_loader = CountingLoader(LEDGER)
    user_loader = CountingLoader(3_000_000)

    assert store.seeded_spend(TENANT, MONTH, tenant_loader) == LEDGER
    assert store.seeded_user_spend(TENANT, USER, MONTH, user_loader) == 3_000_000
    assert store.seeded_user_spend(TENANT, USER, MONTH, user_loader) == 3_000_000
    assert user_loader.calls == 1
    assert store.spend(TENANT, MONTH) == LEDGER


def test_a_loader_that_cannot_read_the_ledger_lets_the_error_out() -> None:
    """Catches a store that swallows `LedgerUnreadable` and answers zero or the counter:
    the route must see it to refuse with BUDGET_UNAVAILABLE."""

    def failing() -> int | None:
        raise LedgerUnreadable("ledger down")

    store = InProcessBudgetStore()

    with pytest.raises(LedgerUnreadable):
        store.seeded_spend(TENANT, MONTH, failing)
    with pytest.raises(LedgerUnreadable):
        store.seeded_user_spend(TENANT, USER, MONTH, failing)


# --------------------------------------------------------------------------
# the marker keys
# --------------------------------------------------------------------------


def test_the_marker_key_shapes() -> None:
    """Catches a marker key that drifts from the published shapes, or one that collides
    with the counter keys (`budget:...`), which would make the marker the counter."""
    assert budget_seeded_key(TENANT, MONTH) == "budget-seeded:nordlicht:2026-10"
    assert user_seeded_key(TENANT, USER, MONTH) == f"budget-seeded-user:nordlicht:{USER}:2026-10"


def test_the_marker_prefix_replaces_the_leading_budget_word() -> None:
    """Catches a prefix glued on in front (`test-1budget-seeded`), which leaves a test
    namespace that does not match the counter keys' namespace."""
    assert budget_seeded_key(TENANT, MONTH, prefix="p1") == "p1-seeded:nordlicht:2026-10"
    assert (
        user_seeded_key(TENANT, USER, MONTH, prefix="p1")
        == f"p1-seeded-user:nordlicht:{USER}:2026-10"
    )
    assert BUDGET_KEY_PREFIX == "budget"


BAD_SEGMENTS: list[tuple[str, str, str, str]] = [
    ("a colon in the tenant", "nord:licht", MONTH, USER),
    ("a trailing newline on the tenant", "nordlicht\n", MONTH, USER),
    ("a trailing CRLF on the tenant", "nordlicht\r\n", MONTH, USER),
    ("a trailing space on the tenant", "nordlicht ", MONTH, USER),
    ("a trailing newline on the month", TENANT, "2026-10\n", USER),
    ("a trailing CRLF on the month", TENANT, "2026-10\r\n", USER),
    ("a trailing space on the month", TENANT, "2026-10 ", USER),
    ("month thirteen", TENANT, "2026-13", USER),
    ("a colon in the user id", TENANT, MONTH, "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6:"),
    ("a trailing newline on the user id", TENANT, MONTH, USER + "\n"),
    ("a trailing CRLF on the user id", TENANT, MONTH, USER + "\r\n"),
    ("a trailing space on the user id", TENANT, MONTH, USER + " "),
]


@pytest.mark.parametrize(
    ("label", "tenant", "month", "user"), BAD_SEGMENTS, ids=[row[0] for row in BAD_SEGMENTS]
)
def test_the_marker_keys_are_checked_before_any_command(
    label: str, tenant: str, month: str, user: str
) -> None:
    """Catches a marker key built by interpolation: a colon in a segment would address
    another tenant's marker, and a trailing newline would slip past a `$` pattern. The
    fake client must have seen no command at all."""
    client = ScriptedRedis()
    store = RedisBudgetStore(as_client(client))
    loader = CountingLoader(LEDGER)

    if label.endswith("user id"):
        with pytest.raises(IdentifierInvalid):
            user_seeded_key(tenant, user, month)
        with pytest.raises(IdentifierInvalid):
            store.seeded_user_spend(tenant, user, month, loader)
    else:
        with pytest.raises(IdentifierInvalid):
            budget_seeded_key(tenant, month)
        with pytest.raises(IdentifierInvalid):
            store.seeded_spend(tenant, month, loader)
        with pytest.raises(IdentifierInvalid):
            user_seeded_key(tenant, USER, month)

    assert client.calls == []
    assert loader.calls == 0


# --------------------------------------------------------------------------
# the Redis store over a recording client
# --------------------------------------------------------------------------


def test_an_unreachable_redis_answers_unknown_without_reading_the_ledger() -> None:
    """Catches a store that reads the ledger before it knows Redis is up (a Redis outage
    would then cost a ledger read per request), and one that lets the error out as a 500.
    The answer is `None`: unknown, so the route serves."""
    client = ScriptedRedis(error=redis.ConnectionError("refused"))
    store = RedisBudgetStore(as_client(client))
    loader = CountingLoader(LEDGER)

    assert store.seeded_spend(TENANT, MONTH, loader) is None
    assert store.seeded_user_spend(TENANT, USER, MONTH, loader) is None
    assert loader.calls == 0


def test_a_present_marker_reads_the_counter_and_skips_the_ledger() -> None:
    """Catches a store that asks the ledger even when the counter is already loaded: the
    plan says the ledger is not read again that month."""
    client = ScriptedRedis(marker_present=True, counter=LEDGER)
    store = RedisBudgetStore(as_client(client))
    loader = CountingLoader(LEDGER)

    assert store.seeded_spend(TENANT, MONTH, loader) == LEDGER

    assert loader.calls == 0
    assert client.names() == ["exists", "get"]
    assert client.calls[0] == ("exists", (budget_seeded_key(TENANT, MONTH),))
    assert client.calls[1] == ("get", (f"budget:{TENANT}:{MONTH}",))


def test_an_absent_marker_runs_one_script_with_the_published_arguments() -> None:
    """Catches a load done as separate SET commands (two replicas can interleave them), a
    script given the keys in the wrong order, the ledger total or the expiry swapped or
    missing, and a store that does not answer what the script answered."""
    client = ScriptedRedis(marker_present=False, counter=LEDGER)
    store = RedisBudgetStore(as_client(client))
    loader = CountingLoader(LEDGER)

    assert store.seeded_spend(TENANT, MONTH, loader) == LEDGER

    assert loader.calls == 1
    assert client.names() == ["exists", "eval"]
    script, numkeys, *rest = client.calls[1][1]
    assert isinstance(script, str)
    assert script != ""
    assert numkeys == 2
    assert rest == [
        budget_seeded_key(TENANT, MONTH),
        f"budget:{TENANT}:{MONTH}",
        LEDGER,
        BUDGET_KEY_TTL_SECONDS,
    ]


def test_the_user_load_passes_the_user_keys_to_the_script() -> None:
    """Catches a user load that runs against the tenant's marker or counter."""
    client = ScriptedRedis(marker_present=False, counter=3_000_000)
    store = RedisBudgetStore(as_client(client))

    assert store.seeded_user_spend(TENANT, USER, MONTH, CountingLoader(3_000_000)) == 3_000_000

    _script, numkeys, *rest = client.calls[1][1]
    assert numkeys == 2
    assert rest == [
        user_seeded_key(TENANT, USER, MONTH),
        f"budget-user:{TENANT}:{USER}:{MONTH}",
        3_000_000,
        BUDGET_KEY_TTL_SECONDS,
    ]


def test_a_loader_answering_none_reads_the_counter_and_sets_no_marker() -> None:
    """Catches a store that runs the load script with a made-up zero when there is no
    database, which would mark the counter loaded for a month and hide the real total."""
    client = ScriptedRedis(marker_present=False, counter=COUNTER_ONLY)
    store = RedisBudgetStore(as_client(client))

    assert store.seeded_spend(TENANT, MONTH, CountingLoader(None)) == COUNTER_ONLY

    assert client.names() == ["exists", "get"]


def test_a_redis_that_fails_during_the_load_answers_unknown() -> None:
    """Catches a store that lets a `RedisError` from the script escape: Redis that cannot
    answer still serves, as before."""
    client = ScriptedRedis(error=redis.TimeoutError("slow"), error_on="eval")
    store = RedisBudgetStore(as_client(client))

    assert store.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER)) is None


def test_a_ledger_that_cannot_be_read_lets_the_error_out_of_the_redis_store() -> None:
    """Catches a Redis store that swallows `LedgerUnreadable` (its `except` is meant for
    Redis errors only) and so serves a tenant whose spend it could not learn."""

    def failing() -> int | None:
        raise LedgerUnreadable("ledger down")

    client = ScriptedRedis(marker_present=False)
    store = RedisBudgetStore(as_client(client))

    with pytest.raises(LedgerUnreadable):
        store.seeded_spend(TENANT, MONTH, failing)
    assert "eval" not in client.names()


# --------------------------------------------------------------------------
# a real Redis
# --------------------------------------------------------------------------


@pytest.fixture
def live_store(redis_client: redis.Redis, key_prefix: str) -> RedisBudgetStore:
    """A store on the session client, in this test's own key namespace."""
    return RedisBudgetStore(redis_client, key_prefix=key_prefix)


@pytest.mark.db
def test_two_replicas_loading_at_once_count_the_spend_once(
    redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a load script that SETs the counter without checking the marker, and a load
    done as separate commands: replica A has asked the ledger and not yet written when
    replica B runs its whole load. A's script must see the marker and just read, so the
    total is 15.07 and not 30.14 (an add) and a debit B made in between survives."""
    replica_a = RedisBudgetStore(redis_client, key_prefix=key_prefix)
    replica_b = RedisBudgetStore(redis_client, key_prefix=key_prefix)
    b_loader = CountingLoader(LEDGER)

    def a_loader() -> int | None:
        assert replica_b.seeded_spend(TENANT, MONTH, b_loader) == LEDGER
        assert replica_b.debit(TENANT, MONTH, LATER_DEBIT) == LEDGER + LATER_DEBIT
        return LEDGER

    seen_by_a = replica_a.seeded_spend(TENANT, MONTH, a_loader)

    assert seen_by_a == LEDGER + LATER_DEBIT, "the second load replaced a newer total"
    assert replica_a.spend(TENANT, MONTH) == LEDGER + LATER_DEBIT


@pytest.mark.db
def test_two_replicas_both_loading_the_same_total_leave_it_unchanged(
    redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a load that adds instead of sets: both replicas load 15_070_000 and the
    counter must read 15_070_000, not 30_140_000."""
    replica_a = RedisBudgetStore(redis_client, key_prefix=key_prefix)
    replica_b = RedisBudgetStore(redis_client, key_prefix=key_prefix)

    def a_loader() -> int | None:
        replica_b.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER))
        return LEDGER

    replica_a.seeded_spend(TENANT, MONTH, a_loader)

    assert replica_a.spend(TENANT, MONTH) == LEDGER
    assert replica_b.spend(TENANT, MONTH) == LEDGER


@pytest.mark.db
def test_the_marker_expires_with_the_counter(
    live_store: RedisBudgetStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a marker with no expiry (it outlives the counter, so a counter that expired
    is never reloaded and starts at zero again) and a counter the script wrote without one.
    Both are read back off the server, exactly the published 62 days."""
    live_store.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER))

    marker_ttl = redis_client.ttl(budget_seeded_key(TENANT, MONTH, prefix=key_prefix))
    counter_ttl = redis_client.ttl(live_store.key(TENANT, MONTH))

    assert marker_ttl == BUDGET_KEY_TTL_SECONDS
    assert counter_ttl == BUDGET_KEY_TTL_SECONDS


@pytest.mark.db
def test_the_real_load_replaces_an_earlier_debit_and_counts_on(
    live_store: RedisBudgetStore,
) -> None:
    """Catches a script that sets the counter only when it is missing (a debit made before
    the load would stay and the ledger would be ignored) and one that loses the debit
    that follows the load."""
    live_store.debit(TENANT, MONTH, EARLY_DEBIT)

    assert live_store.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER)) == LEDGER
    assert live_store.debit(TENANT, MONTH, LATER_DEBIT) == LEDGER + LATER_DEBIT


@pytest.mark.db
def test_the_real_user_load_uses_its_own_marker(
    live_store: RedisBudgetStore, redis_client: redis.Redis, key_prefix: str
) -> None:
    """Catches a user load that shares the tenant's marker: loading the tenant first must
    not stop the user's counter from loading."""
    live_store.seeded_spend(TENANT, MONTH, CountingLoader(LEDGER))
    user_loader = CountingLoader(3_000_000)

    assert live_store.seeded_user_spend(TENANT, USER, MONTH, user_loader) == 3_000_000

    assert user_loader.calls == 1
    assert (
        redis_client.ttl(user_seeded_key(TENANT, USER, MONTH, prefix=key_prefix))
        == BUDGET_KEY_TTL_SECONDS
    )
