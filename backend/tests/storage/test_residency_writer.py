r"""The residency writer without a database: the pre-checks, the row, the policy answer.

Nothing here opens a connection. Two pool doubles do the work, and which one a
case takes IS the claim it makes:

* `ForbiddenPool` fails the test the moment anything asks it for a connection, so
  "the pre-check ran BEFORE any SQL" is asserted as "no SQL was attempted at
  all", not as "the INSERT came back with an error".
* `RecordingPool` keeps every statement, which is the only way to see the
  per-process-per-day partition cache: whether a second write on the same day
  issued a second `CREATE` is invisible against a real PostgreSQL, where the
  second create would simply be skipped or ignored.

The cases needing a real server live in tests/storage/test_postgres_writer.py,
tests/storage/test_migrations.py and tests/storage/test_cli.py, and are skipped
without `OPSCONSOLE_TEST_DATABASE_URL`.

a project rule on the identifier pre-checks: `tenant`, `alias`, `provider` and
`region` reach an INSERT and a partition name, so the rejection table carries a
trailing newline, a trailing CRLF and a trailing space - the three rows that
separate `re.fullmatch` from `re.match` with a `$`.

The contract these cases are written against, before it exists:

* `PostgresResidencyLogWriter(pool, schema=...)`, whose `write` borrows a
  connection with `pool.connection()` and runs statements with
  `conn.execute(query, params)`;
* `record_residency(...)`, which COMPUTES `allowed` with `region_allowed` and
  never takes it as a parameter, and which never raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.model_routing.policy import DataClass, Policy, load_policy
from app.model_routing.resolver import Resolved
from app.storage import (
    RESIDENCY_LOG_RETENTION_DAYS,
    InMemoryResidencyLogWriter,
    PostgresResidencyLogWriter,
    ResidencyLogRow,
    ResidencyLogWriter,
    record_residency,
    region_allowed,
)
from app.storage.residency_log import partition_name

Pool = ConnectionPool[Connection[TupleRow]]

SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"
"""Resolved from this file, so the cases do not depend on pytest's directory."""

# One fixed instant, with a time of day far from both midnights so a writer that
# stamped the row in local time would land on a different DAY here.
WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
NEXT_DAY = WHEN + timedelta(days=1)

TENANT = "nordlicht"
ALIAS = "agent-default"
PROVIDER = "local"
REGION = "self"


def a_row(**overrides: Any) -> ResidencyLogRow:  # noqa: ANN401
    """One well-formed row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "ts": WHEN,
        "tenant": TENANT,
        "data_class": DataClass.INTERNAL,
        "alias": ALIAS,
        "provider": PROVIDER,
        "region": REGION,
        "allowed": True,
    }
    fields.update(overrides)
    return ResidencyLogRow(**fields)


# --------------------------------------------------------------------------
# the two pool doubles
# --------------------------------------------------------------------------


class ForbiddenPool:
    """A pool that fails the test if a writer asks it for a connection at all."""

    def connection(self) -> NoReturn:
        raise AssertionError("a connection was borrowed before the pre-checks refused the row")


class RecordingCursor:
    """What `execute` answers. The writer's statements return no rows."""

    def fetchall(self) -> list[tuple[object, ...]]:
        return []

    def fetchone(self) -> tuple[object, ...] | None:
        return None


class RecordingConnection:
    """A connection that records the text of every statement and runs none of them."""

    def __init__(self, pool: RecordingPool) -> None:
        self._pool = pool

    def execute(self, query: object, params: object = None) -> RecordingCursor:
        # `str()` rather than the object, because a statement may arrive as a
        # `psycopg.sql.Composed`, whose text - the partition name included - is
        # in its repr. The cases below match on substrings for that reason.
        self._pool.statements.append(str(query))
        if self._pool.refuse_create and "create" in str(query).lower():
            raise PermissionError("permission denied for schema opsconsole")
        return RecordingCursor()


class RecordingPool:
    """Every statement the writer issued, in order, and an optional refused CREATE."""

    def __init__(self, *, refuse_create: bool = False) -> None:
        self.statements: list[str] = []
        self.refuse_create = refuse_create

    @contextmanager
    def connection(self) -> Iterator[RecordingConnection]:
        yield RecordingConnection(self)

    def creates(self) -> list[str]:
        """The statements that create a partition."""
        return [text for text in self.statements if "create" in text.lower()]

    def inserts(self) -> list[str]:
        """The statements that write a row."""
        return [text for text in self.statements if "insert" in text.lower()]


def writer_over(pool: ForbiddenPool | RecordingPool) -> PostgresResidencyLogWriter:
    """The real writer over a double, with the pool typing narrowed at the seam.

    The same shape `tests/redaction/test_redis_store.py` uses for its recording
    Redis client: the double is honest about the two methods the writer calls, and
    the cast keeps that from costing a `# type: ignore` in every case.
    """
    return PostgresResidencyLogWriter(cast(Pool, pool))


class RaisingWriter:
    """A writer whose `write` always fails, and which counts the attempts.

    The count is the half that matters: "the request was unchanged" would also be
    true of a `record_residency` that skipped the write entirely.
    """

    def __init__(self) -> None:
        self.calls = 0

    def write(self, row: ResidencyLogRow) -> None:
        self.calls += 1
        raise RuntimeError("the ledger is unreachable")


@pytest.fixture(scope="module")
def policy() -> Policy:
    """The shipped routing policy, loaded once."""
    return load_policy(SHIPPED_POLICY_PATH)


# --------------------------------------------------------------------------
# the constants this plan introduces
# --------------------------------------------------------------------------


def test_the_retention_is_ninety_days() -> None:
    """Catches a retention silently widened or narrowed. The number is a data-protection
    commitment, not a tuning knob: at 900 the gateway keeps request-level residency
    evidence for ten times as long as it promised, and at 9 an auditor asking about last
    month has nothing to read. Asserted exactly, because a range would admit both."""
    assert RESIDENCY_LOG_RETENTION_DAYS == 90


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 9, 7), "residency_log_20260907"),
        (date(2026, 1, 2), "residency_log_20260102"),
        (date(2026, 12, 31), "residency_log_20261231"),
    ],
)
def test_the_partition_name_is_the_pinned_format(day: date, expected: str) -> None:
    """Catches `%Y-%m-%d`, an unpadded month or day, or a `_p` infix. The name is the
    only thing tying a write to a purge: the writer creates it and the purge decides
    what is old by parsing it back, so a format they agree on but nobody pinned can be
    changed in both places at once and silently orphan every partition already on disk.
    The January row is the one that catches unpadded formatting."""
    assert partition_name(day) == expected


def test_the_public_names_are_exported_from_the_package() -> None:
    """Catches names that land in `residency_log.py` and are never re-exported: `gateway/
    app.py` and the chat route import from `app.storage`, so a missing export is an
    ImportError at process start rather than a failure in one test."""
    from app import storage

    assert storage.ResidencyLogRow is ResidencyLogRow
    assert storage.InMemoryResidencyLogWriter is InMemoryResidencyLogWriter
    assert storage.PostgresResidencyLogWriter is PostgresResidencyLogWriter
    assert storage.record_residency is record_residency
    assert storage.region_allowed is region_allowed
    assert storage.RESIDENCY_LOG_RETENTION_DAYS == RESIDENCY_LOG_RETENTION_DAYS


# --------------------------------------------------------------------------
# the row and the in-process writer
# --------------------------------------------------------------------------


def test_the_in_memory_writer_keeps_every_row_in_the_order_written() -> None:
    """Catches a writer that keeps only the last row, or that stores rows in a set or a
    dict keyed by tenant. Every gateway case reads "exactly one row per served request"
    off this list, so a writer that dropped or deduplicated rows would make a route that
    logged nothing at all look correct."""
    writer = InMemoryResidencyLogWriter()
    first = a_row(provider="local")
    second = a_row(provider="anthropic", region="us")

    writer.write(first)
    writer.write(second)

    assert writer.rows == (first, second)


def test_a_rows_snapshot_does_not_change_when_a_later_row_is_written() -> None:
    """Catches `.rows` handing back the internal list: a caller holding what it believes
    is a snapshot would see rows appear in it afterwards, so "one row was written during
    this request" could be asserted against a list that grew after the assertion was
    set up."""
    writer = InMemoryResidencyLogWriter()
    writer.write(a_row())

    snapshot = writer.rows
    writer.write(a_row(provider="anthropic", region="us"))

    assert len(snapshot) == 1
    assert len(writer.rows) == 2


def test_a_row_is_frozen() -> None:
    """Catches a mutable row: the writer hands the same object to a store and to whatever
    logs it, and a row edited after the fact would make the in-process writer's contents
    disagree with what was actually written."""
    row = a_row()

    with pytest.raises(ValueError):
        row.allowed = False  # type: ignore[misc]


# --------------------------------------------------------------------------
# the identifier pre-checks: no SQL, no effect
# --------------------------------------------------------------------------

AT_CAP = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP = "a" * (MAX_IDENTIFIER_LENGTH + 1)

REJECTED_IDENTIFIERS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP),
    ("uppercase", "Anthropic"),
    ("an inner space", "anth ropic"),
    ("a single quote, which would close an SQL literal", "anthropic'"),
    ("a semicolon", "anthropic;"),
    ("a double dash, which would start an SQL comment", "anthropic--x"),
    ("a quoted identifier delimiter", 'anthropic"'),
    ("a trailing newline", "anthropic\n"),
    ("a trailing CRLF", "anthropic\r\n"),
    ("a trailing space", "anthropic "),
    ("a leading separator", "-anthropic"),
    ("a dot", "anthropic.eu"),
]

IDENTIFIER_FIELDS = ("tenant", "alias", "provider", "region")
"""The four client-reachable strings on the row. `data_class` is a `DataClass`
member by type and needs no check of its own."""


@pytest.mark.parametrize("field", IDENTIFIER_FIELDS)
@pytest.mark.parametrize(
    ("case", "value"), REJECTED_IDENTIFIERS, ids=[case for case, _ in REJECTED_IDENTIFIERS]
)
def test_write_refuses_a_malformed_identifier_before_any_sql(
    field: str, case: str, value: str
) -> None:
    """Catches a writer that hands an unchecked client string to PostgreSQL. The pool
    double fails the test on any connection, so what is pinned is "no SQL was attempted"
    rather than "the INSERT came back with an error" - the difference between a rejected
    row and a row half-written under a partition name built from the same string. The
    newline and CRLF rows are what separate `re.fullmatch` from `re.match` with a
    trailing `$`, and parametrising the FIELD is what catches a
    pre-check applied to `tenant` alone: `provider` and `region` arrive from policy today
    but are the two that reach a DDL-adjacent name."""
    writer = writer_over(ForbiddenPool())

    with pytest.raises(IdentifierInvalid):
        writer.write(a_row(**{field: value}))


@pytest.mark.parametrize("field", IDENTIFIER_FIELDS)
def test_an_identifier_at_exactly_the_length_cap_is_accepted(field: str) -> None:
    """Catches an exclusive comparison - `>=` where `>` belongs - on `MAX_IDENTIFIER_
    LENGTH`, which would refuse the longest legal value on all four fields. The
    INSERT having been issued is the assertion: a writer that returned quietly without
    writing anything would also raise nothing."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(**{field: AT_CAP}))

    assert len(pool.inserts()) == 1


def test_a_row_with_every_identifier_at_the_cap_is_still_written() -> None:
    """Catches a length check applied to the CONCATENATION of the identifiers, or to the
    rendered statement, rather than to each field: every field alone is legal here, so a
    writer summing them refuses a row nothing is wrong with."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(tenant=AT_CAP, alias=AT_CAP, provider=AT_CAP, region=AT_CAP))

    assert len(pool.inserts()) == 1


# --------------------------------------------------------------------------
# the per-process-per-day partition cache
# --------------------------------------------------------------------------


def test_two_rows_on_the_same_day_create_that_partition_once() -> None:
    """Catches a writer that issues its `CREATE ... PARTITION OF` on every write: the
    cache is the whole point of doing this on demand, and a DDL statement per served
    request takes an `ACCESS EXCLUSIVE` lock on the parent table on every request. Both
    INSERTs are asserted too, so a writer that cached the CREATE by skipping the write
    is not what makes this pass."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(ts=WHEN))
    writer.write(a_row(ts=WHEN.replace(hour=23, minute=59)))

    assert len(pool.creates()) == 1
    assert len(pool.inserts()) == 2


def test_a_row_on_the_next_day_creates_a_second_partition() -> None:
    """The reset happening mid-sequence: the cache is keyed by DAY, so
    the first write after midnight must create tomorrow's partition. Catches a cache
    keyed by nothing - a plain "have I ever created one" flag - which is the shape that
    passes every same-day test and then sends every row after the first midnight into the
    DEFAULT partition, silently, forever. The second create is matched against tomorrow's
    partition NAME so that a re-issued create for the same day cannot pass this."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(ts=WHEN))
    writer.write(a_row(ts=WHEN.replace(hour=23, minute=59)))
    writer.write(a_row(ts=NEXT_DAY))

    creates = pool.creates()
    assert len(creates) == 2
    assert partition_name(WHEN.date()) in creates[0]
    assert partition_name(NEXT_DAY.date()) in creates[1]
    assert len(pool.inserts()) == 3


def test_a_refused_partition_creation_still_issues_the_insert() -> None:
    """Decision B's backstop, seen from the writer's side. Catches a create whose failure
    propagates: a read-only role or a revoked CREATE privilege would then turn every
    served request's log write into an exception, and the guard around the call site
    would swallow the row entirely rather than filing it in the DEFAULT partition. The
    INSERT is what proves the write continued past the refusal."""
    pool = RecordingPool(refuse_create=True)
    writer = writer_over(pool)

    writer.write(a_row())

    assert len(pool.creates()) == 1
    assert len(pool.inserts()) == 1


# --------------------------------------------------------------------------
# `region_allowed`: the policy's answer, not the caller's
# --------------------------------------------------------------------------

REGION_ROWS: list[tuple[str, str, DataClass, str, bool]] = [
    # An EU tenant whose data class rule NAMES `us`: the residency clamp is the
    # only thing that can keep it out, so this row is the clamp and nothing else.
    ("the eu clamp beats a rule that allows us", "nordlicht", DataClass.PUBLIC, "us", False),
    ("an eu tenant in eu", "nordlicht", DataClass.PUBLIC, "eu", True),
    ("an eu tenant on our own box", "nordlicht", DataClass.PUBLIC, "self", True),
    # The rule is narrower than the clamp here, so the clamp cannot be what denies.
    ("a rule narrower than the clamp", "nordlicht", DataClass.PII, "eu", False),
    ("the one region that rule allows", "nordlicht", DataClass.PII, "self", True),
    ("an any-residency tenant in us", "evals", DataClass.PUBLIC, "us", True),
    ("an any-residency tenant its rule denies", "evals", DataClass.CONFIDENTIAL, "us", False),
    ("an any-residency tenant its rule allows", "evals", DataClass.CONFIDENTIAL, "eu", True),
    ("a tenant the policy does not define", "nobody", DataClass.PUBLIC, "eu", False),
    ("a region no provider is in", "evals", DataClass.PUBLIC, "moon", False),
]


@pytest.mark.parametrize(
    ("case", "tenant", "data_class", "region", "expected"),
    REGION_ROWS,
    ids=[row[0] for row in REGION_ROWS],
)
def test_region_allowed_over_the_shipped_policy(
    policy: Policy, case: str, tenant: str, data_class: DataClass, region: str, expected: bool
) -> None:
    """Catches the three ways this can be wrong in the direction that matters. A function
    that reads the data class rule and forgets the EU intersection passes every other row
    and fails the first, which is the one row where the rule alone would say yes. A
    function that treats an unknown tenant as permissive - the widening a project rule
    forbids - fails the ninth. A function that answers True for any region the policy
    mentions anywhere fails the last. The True rows are here so that "always False",
    which would satisfy every deny row, cannot pass."""
    assert region_allowed(policy, tenant, data_class, region) is expected


def test_a_data_class_with_no_rule_allows_nothing(policy: Policy) -> None:
    """Deny by default at the data class level, which the shipped policy cannot express
    because every tenant in it lists all four classes. Catches a missing rule read as a
    permissive default - the union of the tenant's other rules, or the alias's regions -
    which would file a `pii` request served from `us` as `allowed=true` for a tenant that
    never declared a `pii` rule at all. The `public` row is asserted as True in the same
    call, so a function that denies everything for this tenant is not what passes."""
    without_pii = Policy.model_validate(
        {
            "providers": {"local": {"kind": "openai_compatible", "region": "self"}},
            "eu_regions": ["eu", "self"],
            "aliases": {"agent-default": {"candidates": [{"provider": "local", "model": "m"}]}},
            "tenants": {
                "sparse": {
                    "residency": "any",
                    "data_class_rules": {"public": {"allow_regions": ["self", "us"]}},
                    "budget_usd_month": "1.00",
                    "rate_limit": {"rpm": 1, "tpm": 1},
                }
            },
        }
    )

    assert region_allowed(without_pii, "sparse", DataClass.PUBLIC, "self") is True
    assert region_allowed(without_pii, "sparse", DataClass.PII, "self") is False
    assert region_allowed(without_pii, "sparse", DataClass.INTERNAL, "us") is False
    # The shipped policy's own answer for the same class, so the case above is a
    # statement about the MISSING rule rather than about `DataClass.PII` itself.
    assert region_allowed(policy, "nordlicht", DataClass.PII, "self") is True


# --------------------------------------------------------------------------
# `record_residency`: `allowed` is computed, and the write never escapes
# --------------------------------------------------------------------------


def record(
    log: ResidencyLogWriter,
    policy: Policy,
    *,
    tenant: str,
    data_class: DataClass,
    provider: str,
    region: str,
    alias: str = ALIAS,
) -> None:
    """Drive one `record_residency` with the clock fixed at `WHEN`."""
    record_residency(
        log,
        policy=policy,
        tenant=tenant,
        data_class=data_class,
        alias=alias,
        chosen=Resolved(provider=provider, model="a-model-id", region=region),
        now=lambda: WHEN,
    )


def test_allowed_is_computed_from_the_policy_and_can_be_false(policy: Policy) -> None:
    """The criterion that `allowed` is never passed in as True by the caller. Catches a
    call site that fills the field with a literal - the shape the resolver makes tempting,
    since nothing it lets through can be a violation - by driving a candidate the policy
    would NOT allow: an EU tenant served from `us` under a rule that names `us`. Nothing
    but the clamp inside `region_allowed` can make this row False, so a literal `True`, a
    field read from the caller, or a computation that skipped the clamp all fail here."""
    log = InMemoryResidencyLogWriter()

    record(log, policy, tenant="nordlicht", data_class=DataClass.PUBLIC, provider="x", region="us")

    assert len(log.rows) == 1
    assert log.rows[0].allowed is False


def test_a_permitted_candidate_is_logged_as_allowed(policy: Policy) -> None:
    """The other direction, so `allowed=False` for everything - which would satisfy the
    case above and fire the violation alert on every request the gateway ever serves -
    cannot pass. Catches an inverted comparison in `region_allowed`'s caller."""
    log = InMemoryResidencyLogWriter()

    record(
        log,
        policy,
        tenant="evals",
        data_class=DataClass.INTERNAL,
        provider="anthropic",
        region="us",
    )

    assert log.rows[0].allowed is True


def test_the_row_names_the_candidate_that_served(policy: Policy) -> None:
    """Catches a row filled from the alias or from the request rather than from `chosen`:
    `provider` reading `agent-default` and `region` reading the tenant's home region is
    a log that answers "where did the data go" with the question. Every field is asserted
    at once because they are filled from four different sources."""
    log = InMemoryResidencyLogWriter()

    record(
        log, policy, tenant="evals", data_class=DataClass.INTERNAL, provider="local", region="self"
    )

    row = log.rows[0]
    assert row.tenant == "evals"
    assert row.alias == ALIAS
    assert row.data_class is DataClass.INTERNAL
    assert row.provider == "local"
    assert row.region == "self"


def test_the_row_is_stamped_from_the_injected_clock(policy: Policy) -> None:
    """Catches a timestamp taken inside the database (`default now()`) or filled with a
    naive local `datetime.now()`. The partition a row lands in is chosen from this field,
    so a naive local stamp files an evening request under the wrong day for half the
    world, and a database-side default makes the row's day unknowable to the writer that
    just created the partition for it."""
    log = InMemoryResidencyLogWriter()

    record(
        log, policy, tenant="evals", data_class=DataClass.INTERNAL, provider="local", region="self"
    )

    row = log.rows[0]
    assert row.ts == WHEN
    assert row.ts.tzinfo is not None
    assert row.ts.utcoffset() == timedelta(0)


def test_one_call_writes_exactly_one_row(policy: Policy) -> None:
    """Catches a writer called once per candidate or once per policy rule: the criterion
    is one row per SERVED request, and a duplicate makes every "where did the data go"
    count wrong in the direction that looks like more traffic than there was."""
    log = InMemoryResidencyLogWriter()

    record(
        log, policy, tenant="evals", data_class=DataClass.INTERNAL, provider="local", region="self"
    )

    assert len(log.rows) == 1


def test_a_failing_write_is_swallowed(policy: Policy) -> None:
    """Decision D and a design note, at the seam. Catches an unguarded call: a Postgres blip
    would then propagate out of `record_served` and take out a gateway whose residency
    behaviour is entirely unchanged, on the hot path of every served request. The attempt
    count is what stops a `record_residency` that never writes at all from passing - it
    would satisfy "does not raise" perfectly."""
    log = RaisingWriter()

    record(
        log, policy, tenant="evals", data_class=DataClass.INTERNAL, provider="local", region="self"
    )

    assert log.calls == 1
