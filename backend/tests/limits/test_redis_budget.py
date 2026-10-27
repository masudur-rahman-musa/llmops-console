r"""`InProcessBudgetStore` and `RedisBudgetStore`: the counter, its expiry, and its silence.

Two halves, and the split is the one `tests/cache/test_redis_cache.py` and
`tests/redaction/test_redis_store.py` already make.

The half that needs a REAL Redis is the half whose claims are Redis semantics:
that `INCRBY` accumulates across calls, that the total reads back as the integer
that was booked, that the tenant and the month are genuinely separate keys, and
that the expiry is on the key - read back with `TTL`, never slept for. Redis is
required by an explicit conditional on `OPSCONSOLE_TEST_REDIS_URL` (standing rule
9); the fixture skips with that variable's name in the reason, so a run that
checked nothing says so in its own output. Deliberately NOT
`OPSCONSOLE_REDIS_URL`, which `Settings` itself reads: exporting that one to run
this file would hand every other test in the suite a Redis-backed gateway it
never asked for, and the root `tests/conftest.py` clears it for exactly that
reason.

The half that needs NOTHING is the validation and failure half, over a client
that records its calls and refuses to answer. "No spend was booked" is asserted
as "Redis was never called at all", which is the claim a project rule actually
makes, and the fail-open rows (decision B) need a client that raises rather than
a server that is down.

Decision B is the difference from the redaction map store, and it is stated here
because it is the behaviour someone will later mistake for a bug: a
`redis.RedisError` is SWALLOWED and both methods answer `None`. `None` is
"unknown", not "zero" - the route serves the request, logs, and moves NO gauge,
because a gauge fed with a zero it invented would read a healthy 0% for every
tenant for as long as Redis is down, which is exactly when someone is looking at
it.

a project rule: one session-scoped client, built once. Per-test isolation is a
unique `key_prefix` handed to the store, never `FLUSHDB` - this database may be
shared with anything else - and never a fresh client.

The contract these cases are written against, before it exists:

* `BudgetStore` - `spend(tenant, month) -> int | None` and
  `debit(tenant, month, micro_usd) -> int | None`, the running total in
  micro-USD after the debit;
* `InProcessBudgetStore()` - the twin `create_app` keeps when
  `OPSCONSOLE_REDIS_URL` is unset, which never answers `None`;
* `RedisBudgetStore(client, key_prefix=BUDGET_KEY_PREFIX)` - `INCRBY` plus an
  expiry of `BUDGET_KEY_TTL_SECONDS`, over the key `month_key` renders.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
import redis

from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.limits.budget import (
    BUDGET_KEY_PREFIX,
    BUDGET_KEY_TTL_SECONDS,
    BudgetStore,
    InProcessBudgetStore,
    RedisBudgetStore,
    month_key,
    user_month_key,
)

TENANT = "evals"
OTHER_TENANT = "cuan"
MONTH = "2026-09"
OTHER_MONTH = "2026-08"

CHAT_MICRO = 282
STREAM_MICRO = 691
"""Two costs in the counter's own unit, different from each other and from every
other number here, so a swapped argument or a doubled debit is visible."""

MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)


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
    fail-open rows are arranged; `calls` is what makes "the pre-check ran before
    any effect" observable, because an empty list is the only evidence that
    nothing was booked.
    """

    def __init__(self, error: Exception | None = None, value: bytes | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self._error = error
        self._value = value
        self.total = 0

    def _record(self, command: str, *args: object) -> None:
        self.calls.append((command, args))
        if self._error is not None:
            raise self._error

    def incrby(self, name: str, amount: int) -> int:
        self._record("incrby", name, amount)
        self.total += amount
        return self.total

    def expire(self, name: str, seconds: int) -> bool:
        self._record("expire", name, seconds)
        return True

    def get(self, name: str) -> bytes | None:
        self._record("get", name)
        return self._value


def fake_store(
    *, error: Exception | None = None, value: bytes | None = None
) -> tuple[RedisBudgetStore, RecordingRedis]:
    """A store over a recording client, and the client, so calls can be asserted."""
    client = RecordingRedis(error=error, value=value)
    return RedisBudgetStore(as_client(client)), client


BUDGET_LOGGER = "app.limits.budget"
"""Where a swallowed failure is reported. Named so the assertion below counts THIS
module's records and nothing else's."""


@contextmanager
def caplog_at(level: int) -> Iterator[list[logging.LogRecord]]:
    """Collect this module's records at `level` and put the logger back afterwards.

    A handler on the named logger rather than pytest's `caplog` fixture, because
    every case in this file is a plain function and two of them need the records
    from one arrangement; the restore is what keeps a failing case from leaking a
    handler into the rest of the session.
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(BUDGET_LOGGER)
    handler = _Collect(level)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def commands(client: RecordingRedis) -> list[str]:
    """Just the command names, for the rows about which side effect happened."""
    return [call[0] for call in client.calls]


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
]

REJECTED_MONTHS: list[tuple[str, str]] = [
    ("empty", ""),
    ("a one-digit month", "2026-9"),
    ("month thirteen", "2026-13"),
    ("a colon, which would forge a fourth key segment", "2026-09:x"),
    ("a trailing newline", "2026-09\n"),
    ("a trailing space", "2026-09 "),
]


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_debit_refuses_a_malformed_tenant_before_touching_redis(label: str, tenant: str) -> None:
    """Catches a store that builds its key by interpolation and lets Redis decide: `INCRBY
    budget:ev:als:2026-09 282` books one tenant's spend against a key nobody reads, and a
    glob character widens any later `SCAN` an operator runs over these counters. The
    assertion is on the recorded calls rather than on "no key exists", because that is
    the claim a project rule makes - the check runs BEFORE the effect, not instead of
    it."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.debit(tenant, MONTH, CHAT_MICRO)

    assert client.calls == []


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_spend_refuses_a_malformed_tenant_before_touching_redis(label: str, tenant: str) -> None:
    """The read side of the same check. Catches a store that validates only on the write
    path, which would let a crafted tenant address another namespace's key on the read
    that decides whether a request is refused."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.spend(tenant, MONTH)

    assert client.calls == []


@pytest.mark.parametrize(
    ("label", "month"), REJECTED_MONTHS, ids=[row[0] for row in REJECTED_MONTHS]
)
def test_a_malformed_month_reaches_no_command_on_either_method(label: str, month: str) -> None:
    """Catches a month pre-check that exists in the key function but is bypassed by a
    store method that renders its own key, which is how the check in one place becomes no
    check at all."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.debit(TENANT, month, CHAT_MICRO)
    with pytest.raises(IdentifierInvalid):
        store.spend(TENANT, month)

    assert client.calls == []


def test_the_tenant_length_boundary_is_inclusive() -> None:
    """The inclusive boundary a project rule asks for, on the store rather than on the
    key function. Catches a `>=` that refuses a tenant of exactly the permitted length -
    a name `gateway/context.py` already authenticated, so the request would fail after
    the provider had answered."""
    store, client = fake_store()

    store.debit(MAX_TENANT, MONTH, CHAT_MICRO)

    assert commands(client)[0] == "incrby"


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------


def test_the_key_is_the_one_the_month_key_function_renders() -> None:
    """Catches a store with a key format of its own, which would leave the counter an
    operator reads with `GET budget:evals:2026-09` and the counter the gateway increments
    two different keys."""
    store, client = fake_store()

    store.debit(TENANT, MONTH, CHAT_MICRO)

    assert store.key(TENANT, MONTH) == month_key(TENANT, MONTH)
    assert client.calls[0] == ("incrby", (month_key(TENANT, MONTH), CHAT_MICRO))


def test_the_default_prefix_is_the_one_the_key_is_written_under() -> None:
    """Catches a store built with no prefix argument writing under something other than
    the published prefix: `create_app` builds it with no arguments, so the shipped
    gateway is exactly this case, and every test that passes a prefix would still
    pass."""
    store, client = fake_store()

    store.debit(TENANT, MONTH, CHAT_MICRO)

    assert str(client.calls[0][1][0]).startswith(f"{BUDGET_KEY_PREFIX}:")


def test_a_prefixed_store_writes_under_its_own_namespace() -> None:
    """Catches a `key_prefix` argument that is accepted and ignored, which would make
    every test below share one live key and this file's isolation a fiction (standing
    rule 7)."""
    client = RecordingRedis()
    store = RedisBudgetStore(as_client(client), key_prefix="test-budget-prefix")

    store.debit(TENANT, MONTH, CHAT_MICRO)

    assert client.calls[0][1][0] == f"test-budget-prefix:{TENANT}:{MONTH}"


# --------------------------------------------------------------------------
# the expiry (section 5's third risk)
# --------------------------------------------------------------------------


def test_a_debit_sets_the_expiry_in_the_same_call_sequence() -> None:
    """Catches a counter written with `INCRBY` and no expiry at all, which is the risk
    section 5 names: one key per tenant per month, kept for ever. `INCRBY` does not carry
    a TTL the way `SET ... EX` does, so the expiry is a second command and its absence is
    invisible until an operator counts the keys a year later. The order is asserted too -
    an `EXPIRE` sent first would set a TTL on a key that does not exist yet and be
    dropped."""
    store, client = fake_store()

    store.debit(TENANT, MONTH, CHAT_MICRO)

    assert commands(client) == ["incrby", "expire"]
    assert client.calls[1] == ("expire", (month_key(TENANT, MONTH), BUDGET_KEY_TTL_SECONDS))


# --------------------------------------------------------------------------
# fail open (decision B)
# --------------------------------------------------------------------------

REDIS_FAILURES: list[tuple[str, Exception]] = [
    ("the connection dropped", redis.ConnectionError("connection refused")),
    ("the command timed out", redis.TimeoutError("timed out")),
    ("the server answered an error", redis.ResponseError("READONLY")),
    ("authentication failed", redis.AuthenticationError("wrong password")),
]


@pytest.mark.parametrize(("label", "error"), REDIS_FAILURES, ids=[row[0] for row in REDIS_FAILURES])
def test_an_unreachable_redis_serves_and_moves_no_gauge(label: str, error: Exception) -> None:
    """Decision B, at the seam where it is decided.

    Catches a store that lets `redis.RedisError` escape, which turns a Redis blip into
    5xx on every request a provider would have served - the failure a BUDGET must not
    cause. And catches the over-correction that is worse than raising: answering 0
    instead of `None`. Zero is a spend, so the route would feed
    `gateway_budget_ratio` a confident 0.0 for every tenant for as long as Redis is
    down, and the one dashboard an operator checks during the incident would read
    healthy. `None` is "unknown", which is what makes "move no gauge" expressible at
    all.

    Four failure rows because the swallow has to be on `redis.RedisError`, the base:
    catching `ConnectionError` alone passes the first row and raises on the other
    three."""
    store, client = fake_store(error=error)

    assert store.spend(TENANT, MONTH) is None
    assert store.debit(TENANT, MONTH, CHAT_MICRO) is None
    assert commands(client) == ["get", "incrby"]


def test_every_swallowed_failure_is_logged_with_the_tenant_and_the_month() -> None:
    """The other half of decision B, and the half that is easy to leave out.

    Failing open means a cap silently stops being enforced: the request is served, the
    gauge deliberately does not move, and a debit that was lost is money spent against a
    counter that never rose. The log is then the ONLY place that window is visible at
    all. Catches a bare `except redis.RedisError: return None`, which is the obvious
    implementation and leaves an operator with no way to tell a quiet month from an
    unreachable counter. Both methods are asserted, because a `debit` that vanished
    matters more than a `spend` that could not be read, and the tenant and the month are
    asserted in the record because a message naming neither cannot be acted on."""
    store, _ = fake_store(error=redis.ConnectionError("connection refused"))

    with caplog_at(logging.ERROR) as records:
        store.spend(TENANT, MONTH)
        store.debit(TENANT, MONTH, CHAT_MICRO)

    assert len(records) == 2, [record.getMessage() for record in records]
    for record in records:
        assert TENANT in record.getMessage()
        assert MONTH in record.getMessage()


def test_a_stored_value_that_is_not_an_integer_is_logged_before_it_reads_unknown() -> None:
    """A counter holding something that is not an integer is a corrupted key, not an
    outage, and it is the one failure that persists until somebody deletes the key -
    every request for the rest of the month reads unknown and books nothing. Catches a
    silent `return None` on that path, which is indistinguishable from a Redis that is
    down and needs a completely different response."""
    store, _ = fake_store(value=b"not-an-integer")

    with caplog_at(logging.ERROR) as records:
        assert store.spend(TENANT, MONTH) is None

    assert len(records) == 1
    assert TENANT in records[0].getMessage()


def test_a_stored_value_that_is_not_an_integer_reads_unknown_rather_than_zero() -> None:
    """Catches an `int(stored)` that raises on a key another build (or another tool) wrote
    a non-integer into, which would be a 500 on a served request. It answers unknown for
    the same reason an unreachable Redis does: the route may serve, but it may not
    publish a ratio it did not read."""
    store, _ = fake_store(value=b"not-an-integer")

    assert store.spend(TENANT, MONTH) is None


def test_a_missing_counter_reads_zero_and_not_unknown() -> None:
    """The row that keeps the one above honest. A month nobody has spent in yet is a
    KNOWN zero, not a failure: catching an implementation that answers `None` for an
    absent key would leave the first request of every month unable to feed the gauge and
    would make the empty case indistinguishable from an outage."""
    store, _ = fake_store(value=None)

    assert store.spend(TENANT, MONTH) == 0


# --------------------------------------------------------------------------
# the in-process twin
# --------------------------------------------------------------------------


def test_the_in_process_twin_accumulates_and_answers_the_running_total() -> None:
    """Catches a twin whose `debit` answers the amount it was handed rather than the new
    total - the route feeds the gauge from that answer, so every tenant's ratio would
    read the cost of its last request instead of its spend for the month - and a twin
    that replaces rather than accumulates."""
    store = InProcessBudgetStore()

    assert store.debit(TENANT, MONTH, CHAT_MICRO) == CHAT_MICRO
    assert store.debit(TENANT, MONTH, STREAM_MICRO) == CHAT_MICRO + STREAM_MICRO
    assert store.spend(TENANT, MONTH) == CHAT_MICRO + STREAM_MICRO


def test_the_in_process_twin_keeps_tenants_and_months_apart() -> None:
    """Catches a twin keyed on the tenant alone, which never rolls over at the month
    boundary and refuses every request for ever once a cap is reached; and one keyed on
    the month alone, which pools every tenant's spend into one counter and refuses the
    quiet tenants because a loud one spent."""
    store = InProcessBudgetStore()

    store.debit(TENANT, MONTH, CHAT_MICRO)

    assert store.spend(OTHER_TENANT, MONTH) == 0
    assert store.spend(TENANT, OTHER_MONTH) == 0
    assert store.spend(TENANT, MONTH) == CHAT_MICRO


def test_the_in_process_twin_never_answers_unknown() -> None:
    """Catches a twin that copied the Redis one's `None` returns. It cannot fail, so a
    `None` from it would only ever be a bug that silently disables the cap on the local
    and suite configuration - which is every run of this suite."""
    store = InProcessBudgetStore()

    assert store.spend(TENANT, MONTH) is not None
    assert store.debit(TENANT, MONTH, 0) is not None


def test_a_zero_debit_still_answers_the_running_total() -> None:
    """A cache hit debits zero (criterion 5), and the route feeds the gauge from what
    `debit` answers. Catches a store that short-circuits a zero debit and answers `None`
    or nothing, which would make the gauge go stale on every hit."""
    store = InProcessBudgetStore()
    store.debit(TENANT, MONTH, CHAT_MICRO)

    assert store.debit(TENANT, MONTH, 0) == CHAT_MICRO


@pytest.mark.parametrize("store_name", ["in-process", "redis"])
def test_both_twins_answer_the_same_running_total(store_name: str) -> None:
    """The two implementations are picked by an environment variable, so a difference
    between them is a difference between a developer's run and the cluster's.

    Catches the drift that matters: one twin answering the new TOTAL and the other the
    amount it was just handed. The route feeds `gateway_budget_ratio` from that answer,
    so the local run would show a rising ratio and the deployed one a ratio that reads
    the last request's cost - and no test that exercises a single debit can tell them
    apart, which is why this one debits twice."""
    store: BudgetStore = (
        InProcessBudgetStore()
        if store_name == "in-process"
        else RedisBudgetStore(as_client(RecordingRedis()))
    )

    assert store.debit(TENANT, MONTH, CHAT_MICRO) == CHAT_MICRO
    assert store.debit(TENANT, MONTH, STREAM_MICRO) == CHAT_MICRO + STREAM_MICRO


# --------------------------------------------------------------------------
# against a real Redis
# --------------------------------------------------------------------------


@pytest.fixture
def live_store(redis_client: redis.Redis, key_prefix: str) -> RedisBudgetStore:
    """A store over the session client, namespaced to this test.

    `redis_client` and `key_prefix` come from the top `tests/conftest.py`, built
    once per run; this file builds no client of its own.
    """
    return RedisBudgetStore(redis_client, key_prefix=key_prefix)


def test_a_debit_accumulates_and_reads_back(live_store: RedisBudgetStore) -> None:
    """The claim that is Redis's rather than ours: `INCRBY` adds to what is there. Catches
    a store that `SET`s the amount, which would leave a month's spend equal to the last
    request's cost and no cap would ever be reached; and one whose `spend` reads a
    different key than `debit` wrote, which reads zero for ever."""
    assert live_store.spend(TENANT, MONTH) == 0
    assert live_store.debit(TENANT, MONTH, CHAT_MICRO) == CHAT_MICRO
    assert live_store.debit(TENANT, MONTH, STREAM_MICRO) == CHAT_MICRO + STREAM_MICRO
    assert live_store.spend(TENANT, MONTH) == CHAT_MICRO + STREAM_MICRO


def test_one_tenants_spend_is_not_anothers(live_store: RedisBudgetStore) -> None:
    """Catches a key that drops the tenant segment: with one counter for everyone, the
    tenant that spends nothing is refused the moment a busy neighbour reaches its own
    cap. Read with a tenant that genuinely has a live counter, so "unknown key" cannot be
    what makes it pass."""
    live_store.debit(TENANT, MONTH, CHAT_MICRO)
    live_store.debit(OTHER_TENANT, MONTH, STREAM_MICRO)

    assert live_store.spend(TENANT, MONTH) == CHAT_MICRO
    assert live_store.spend(OTHER_TENANT, MONTH) == STREAM_MICRO


def test_one_months_spend_is_not_anothers(live_store: RedisBudgetStore) -> None:
    """Catches a key that drops the month segment, which is a budget that never resets:
    the tenant is refused for ever once its first month's cap is reached."""
    live_store.debit(TENANT, MONTH, CHAT_MICRO)
    live_store.debit(TENANT, OTHER_MONTH, STREAM_MICRO)

    assert live_store.spend(TENANT, MONTH) == CHAT_MICRO
    assert live_store.spend(TENANT, OTHER_MONTH) == STREAM_MICRO


def test_the_key_expires_after_exactly_the_published_ttl(
    live_store: RedisBudgetStore, redis_client: redis.Redis
) -> None:
    """Read back off the server rather than slept for. Catches a counter with no expiry at
    all (`TTL` answers -1, which is a key that lives for ever - section 5's third risk)
    and a key that never existed (-2). The value is read back exactly, as
    tests/cache/test_redis_cache.py and tests/redaction/test_redis_store.py read theirs:
    the expiry is written in the same call sequence as the increment, so there is no
    second for the clock to tick in. The constant itself is pinned as a literal in
    tests/limits/test_budget.py."""
    live_store.debit(TENANT, MONTH, CHAT_MICRO)

    ttl = redis_client.ttl(live_store.key(TENANT, MONTH))

    assert ttl != -1, "the counter carries no expiry, so it is kept for ever"
    assert ttl != -2, "no counter was written at all"
    assert ttl == BUDGET_KEY_TTL_SECONDS


def test_a_second_debit_does_not_drop_the_expiry(
    live_store: RedisBudgetStore, redis_client: redis.Redis
) -> None:
    """Catches an implementation that sets the TTL only when it thinks the key is new and
    gets the condition wrong, and one that re-creates the key without a TTL on the second
    debit. Every request of the month goes through this path, so an expiry that survives
    only the first one survives nothing."""
    live_store.debit(TENANT, MONTH, CHAT_MICRO)
    live_store.debit(TENANT, MONTH, STREAM_MICRO)

    ttl = redis_client.ttl(live_store.key(TENANT, MONTH))

    assert ttl != -1, "the second debit left the counter with no expiry"
    assert ttl == BUDGET_KEY_TTL_SECONDS


# --------------------------------------------------------------------------
# the per-user counters
# --------------------------------------------------------------------------

USER_A = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"
USER_B = "u_00000000000000000000000000000001"


@pytest.mark.db
def test_user_counters_are_apart_per_user_and_from_the_tenant(
    live_store: RedisBudgetStore, redis_client: redis.Redis
) -> None:
    """Catches a user key that drops the user segment (every user of a tenant shares one
    cap), one that reuses the tenant's key (a user's debit raises the tenant total, so the
    tenant is billed twice), and a user counter written without the 62-day expiry, which
    keeps one key per user per month for ever. TTL is read off the server, not slept for."""
    live_store.debit(TENANT, MONTH, CHAT_MICRO)

    assert live_store.user_debit(TENANT, USER_A, MONTH, STREAM_MICRO) == STREAM_MICRO
    assert live_store.user_debit(TENANT, USER_A, MONTH, STREAM_MICRO) == 2 * STREAM_MICRO
    assert live_store.user_debit(TENANT, USER_B, MONTH, CHAT_MICRO) == CHAT_MICRO

    assert live_store.user_spend(TENANT, USER_A, MONTH) == 2 * STREAM_MICRO
    assert live_store.user_spend(TENANT, USER_B, MONTH) == CHAT_MICRO
    assert live_store.spend(TENANT, MONTH) == CHAT_MICRO
    assert live_store.user_key(TENANT, USER_A, MONTH) != live_store.key(TENANT, MONTH)
    assert redis_client.ttl(live_store.user_key(TENANT, USER_A, MONTH)) == BUDGET_KEY_TTL_SECONDS


def test_the_user_key_is_the_one_the_user_month_key_function_renders() -> None:
    """Catches a store with a user key format of its own, and a `key_prefix` ignored for
    users (every test would then share one live user key): the default is the published
    `budget-user`, a prefixed store files its users under `{prefix}-user`."""
    store, client = fake_store()

    store.user_debit(TENANT, USER_A, MONTH, CHAT_MICRO)

    assert store.user_key(TENANT, USER_A, MONTH) == user_month_key(TENANT, USER_A, MONTH)
    assert client.calls[0] == ("incrby", (user_month_key(TENANT, USER_A, MONTH), CHAT_MICRO))
    prefixed = RedisBudgetStore(as_client(RecordingRedis()), key_prefix="test-x")
    assert prefixed.user_key(TENANT, USER_A, MONTH) == f"test-x-user:{TENANT}:{USER_A}:{MONTH}"


def test_a_bad_user_id_reaches_no_command_on_either_user_method() -> None:
    """Catches a user pre-check that lives only in the key function but is bypassed by a
    method that renders its own key: Redis must see no call at all."""
    store, client = fake_store()

    with pytest.raises(IdentifierInvalid):
        store.user_debit(TENANT, USER_A + "\n", MONTH, CHAT_MICRO)
    with pytest.raises(IdentifierInvalid):
        store.user_spend(TENANT, USER_A + "\n", MONTH)

    assert client.calls == []


def test_an_unreachable_redis_answers_unknown_on_the_user_methods() -> None:
    """Catches user methods that let `redis.RedisError` escape (a Redis blip becomes a 5xx)
    or answer 0 (a cap that reads as unspent): unknown is `None`, as for the tenant."""
    store, _ = fake_store(error=redis.ConnectionError("connection refused"))

    assert store.user_spend(TENANT, USER_A, MONTH) is None
    assert store.user_debit(TENANT, USER_A, MONTH, CHAT_MICRO) is None


def test_the_in_process_twin_counts_users_apart() -> None:
    """Catches a twin that keys users on the tenant alone, or lets a user's debit raise
    the tenant's own total."""
    store = InProcessBudgetStore()

    assert store.user_debit(TENANT, USER_A, MONTH, CHAT_MICRO) == CHAT_MICRO

    assert store.user_spend(TENANT, USER_B, MONTH) == 0
    assert store.user_spend(TENANT, USER_A, OTHER_MONTH) == 0
    assert store.spend(TENANT, MONTH) == 0
