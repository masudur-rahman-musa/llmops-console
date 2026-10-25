r"""The cost ledger without a database: the pre-checks, the row, the price the row carries.

Nothing here opens a connection. Two pool doubles do the work, and which one a
case takes IS the claim it makes:

* `ForbiddenPool` fails the test the moment anything asks it for a connection, so
  "the pre-check ran BEFORE any SQL" is asserted as "no SQL was attempted at
  all", not as "the INSERT came back with an error".
* `RecordingPool` keeps every statement, so a boundary case can assert the INSERT
  was actually ISSUED - a writer that returned quietly without writing anything
  would raise nothing either, and would pass an "is accepted" case that only
  checked for the absence of an exception.

The cases needing a real server live in tests/storage/test_cost_ledger_postgres.py
and are skipped without `OPSCONSOLE_TEST_DATABASE_URL`.

a project rule on the three patterns: `tenant`, `alias`, `provider`
and `session` are checked against the resolver's identifier rule, `model` against
this module's own wider one (decision D), and `trace_id` against 32 lowercase hex
characters, so each rejection table carries a trailing newline, a trailing CRLF
and a trailing space. The trace id rows are built so the WHOLE value is still
exactly `TRACE_ID_LENGTH` characters long (31 hex plus a newline, 30 hex plus a
CRLF), because a 33-character value would be refused by the length check and the
case would then pass without the pattern being involved at all.

The contract these cases are written against, before it exists:

* `CostLedgerRow`, a frozen model with the design's thirteen columns, `session`
  nullable and `cost_usd` a `Decimal`;
* `PostgresCostLedgerWriter(pool, *, schema=...)`, whose `write` runs every
  pre-check, then borrows a connection with `pool.connection()` and issues one
  `conn.execute(query, params)`;
* `record_cost(ledger, *, table, tenant, alias, provider, model, usage,
  prompt_chars=0, completion_chars=0, cached, session, trace_id, price_key,
  now=...)`, which PRICES the request itself through
  `app.limits.cost.cost_for` (decision C, so there is no `cost_usd`
  parameter to fill) and never raises (decision E).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from app.core.validation import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.limits.cost import UNPRICED_VERSION
from app.limits.prices import PriceTable
from app.model_routing.policy import Policy, load_policy
from app.schemas.provider import Usage
from app.services.ledger import record_cost
from app.storage import (
    CostLedgerRow,
    CostLedgerWriter,
    InMemoryCostLedgerWriter,
    PostgresCostLedgerWriter,
)
from app.storage.cost_ledger import (
    COST_LEDGER,
    COST_LEDGER_DAILY,
    MAX_MODEL_LENGTH,
    MAX_PRICE_VERSION_LENGTH,
    MODEL_PATTERN,
    TRACE_ID_LENGTH,
    TRACE_ID_PATTERN,
)

Pool = ConnectionPool[Connection[TupleRow]]

SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"
SHIPPED_PROVIDERS_DIR = SHIPPED_POLICY_PATH.parent / "providers"
"""Resolved from this file, so the cases do not depend on pytest's directory."""

# One fixed instant, with a time of day far from both midnights so a writer that
# stamped the row in local time would land on a different DAY here - which is the
# day `cost_ledger_daily` groups by.
WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)

TENANT = "nordlicht"
ALIAS = "agent-default"
PROVIDER = "local"
MODEL = "qwen2.5-7b-instruct"
SESSION = "sess-01j9x8"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
"""32 lowercase hex characters, the W3C trace-context shape the gateway carries."""

PRICED_KEY = f"{PROVIDER}/{MODEL}"
UNPRICED_KEY = f"{PROVIDER}/a-model-nobody-priced"

PRICE_TABLE = PriceTable.model_validate(
    {
        "prices": {
            PRICED_KEY: [
                {
                    "effective_from": "2026-01-01",
                    "input_usd_per_1m": "3.000000",
                    "output_usd_per_1m": "15.000000",
                }
            ]
        }
    }
)
"""One key, priced differently in each direction: a cost that multiplied both
token counts by the INPUT price would come out the same under equal prices."""

PRICED_VERSION = f"{PRICED_KEY}@2026-01-01"

USAGE = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
# (1000 * 3 + 500 * 15) / 1_000_000, written out so the case does not repeat the
# implementation's own arithmetic.
USAGE_COST = Decimal("0.010500")


def a_row(**overrides: Any) -> CostLedgerRow:  # noqa: ANN401
    """One well-formed row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "ts": WHEN,
        "tenant": TENANT,
        "alias": ALIAS,
        "provider": PROVIDER,
        "model": MODEL,
        "input_tokens": USAGE.prompt_tokens,
        "output_tokens": USAGE.completion_tokens,
        "cached": False,
        "cost_usd": USAGE_COST,
        "price_version": PRICED_VERSION,
        "estimated": False,
        "session": SESSION,
        "trace_id": TRACE_ID,
    }
    fields.update(overrides)
    return CostLedgerRow(**fields)


def record(ledger: CostLedgerWriter, **overrides: Any) -> CostLedgerRow:  # noqa: ANN401
    """Drive one `record_cost` with the clock fixed at `WHEN`, and answer its row.

    Every argument is keyword-only on `record_cost` itself, so this helper is the
    one place the full call is spelled out and a case names only what it changes.

    The return is decision G: `record_cost` answers the row it built,
    so the header, the metric and the row are one number computed once. The cases
    written before that decision read the row off the writer instead and ignore
    what this hands back, which is why the change is a signature and not a
    rewrite.
    """
    call: dict[str, Any] = {
        "table": PRICE_TABLE,
        "tenant": TENANT,
        "alias": ALIAS,
        "provider": PROVIDER,
        "model": MODEL,
        "usage": USAGE,
        "prompt_chars": 0,
        "completion_chars": 0,
        "cached": False,
        "session": SESSION,
        "trace_id": TRACE_ID,
        "price_key": PRICED_KEY,
        "now": lambda: WHEN,
    }
    call.update(overrides)
    return record_cost(ledger, **call)


# --------------------------------------------------------------------------
# the two pool doubles
# --------------------------------------------------------------------------


class ForbiddenPool:
    """A pool that fails the test if the writer asks it for a connection at all."""

    def connection(self) -> NoReturn:
        raise AssertionError("a connection was borrowed before the pre-checks refused the row")


class RecordingCursor:
    """What `execute` answers. The writer's one statement returns no rows."""

    def fetchall(self) -> list[tuple[object, ...]]:
        return []

    def fetchone(self) -> tuple[object, ...] | None:
        return None


class RecordingConnection:
    """A connection that records the text of every statement and runs none of them."""

    def __init__(self, pool: RecordingPool) -> None:
        self._pool = pool

    def execute(self, query: object, params: object = None) -> RecordingCursor:
        # `str()` rather than the object, because the statement arrives as a
        # `psycopg.sql.Composed` whose text is in its repr.
        self._pool.statements.append(str(query))
        return RecordingCursor()


class RecordingPool:
    """Every statement the writer issued, in order."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    @contextmanager
    def connection(self) -> Iterator[RecordingConnection]:
        yield RecordingConnection(self)

    def inserts(self) -> list[str]:
        """The statements that write a row."""
        return [text for text in self.statements if "insert" in text.lower()]


def writer_over(pool: ForbiddenPool | RecordingPool) -> PostgresCostLedgerWriter:
    """The real writer over a double, with the pool typing narrowed at one seam."""
    return PostgresCostLedgerWriter(cast(Pool, pool))


class RaisingWriter:
    """A writer whose `write` always fails, and which counts the attempts.

    The count is the half that matters: "no exception reached the caller" would
    also be true of a `record_cost` that never wrote anything at all.
    """

    def __init__(self) -> None:
        self.calls = 0

    def write(self, row: CostLedgerRow) -> None:
        self.calls += 1
        raise RuntimeError("the ledger is unreachable")


@pytest.fixture(scope="module")
def policy() -> Policy:
    """The shipped routing policy, loaded once."""
    return load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR)


# --------------------------------------------------------------------------
# the constants this plan introduces
# --------------------------------------------------------------------------


def test_the_table_names_are_the_ones_the_migration_creates() -> None:
    """Catches a name that drifts between the module and `002_cost_ledger.sql`: the
    INSERT and the refresh both build their identifiers from these, so a renamed table
    is an `UndefinedTable` on every served request and a rollup nothing refreshes.
    Asserted as literals, because comparing them to each other would agree with any
    pair of wrong names."""
    assert COST_LEDGER == "cost_ledger"
    assert COST_LEDGER_DAILY == "cost_ledger_daily"


def test_the_model_and_price_version_caps_are_pinned() -> None:
    """Catches a cap silently widened or narrowed. `MAX_MODEL_LENGTH` is decision D's
    whole point - it is deliberately NOT the resolver's 64, because the shipped policy
    already names ids the resolver's rule refuses - and a value narrowed back to the
    resolver's would drop every row it used to accept. A range would admit both."""
    assert MAX_MODEL_LENGTH == 128
    assert MAX_PRICE_VERSION_LENGTH == 256


def test_a_trace_id_is_exactly_thirty_two_characters() -> None:
    """Catches a length constant that drifts from the pattern beside it: the id is a
    W3C trace-context id and the console joins ledger rows to traces on it, so a cap of
    64 would file rows carrying an id no trace has."""
    assert TRACE_ID_LENGTH == 32
    assert TRACE_ID_PATTERN.fullmatch(TRACE_ID) is not None


# --------------------------------------------------------------------------
# the row and the in-process writer
# --------------------------------------------------------------------------


def test_a_row_is_frozen() -> None:
    """Catches a mutable row: the same object is handed to a writer and to whatever
    counts it, and a `cost_usd` edited after the fact would make the ledger disagree
    with the number the gateway already reported for that request."""
    row = a_row()

    with pytest.raises(ValueError):
        row.cost_usd = Decimal("0.000000")  # type: ignore[misc]


def test_the_in_memory_writer_keeps_every_row_in_the_order_written() -> None:
    """Catches a writer that keeps only the last row, or that stores rows in a dict
    keyed by tenant or by trace id. Spend is a SUM over these rows, so a writer that
    deduplicated them would under-report every tenant that asked the same thing twice."""
    writer = InMemoryCostLedgerWriter()
    first = a_row()
    second = a_row(model="a-second-model", cost_usd=Decimal("0.000200"))

    writer.write(first)
    writer.write(second)

    assert writer.rows == (first, second)


def test_a_rows_snapshot_does_not_change_when_a_later_row_is_written() -> None:
    """Catches `.rows` handing back the internal list: a caller holding what it believes
    is a snapshot would see rows appear in it afterwards, so "this request wrote one
    row" could be asserted against a list that grew after the assertion was set up."""
    writer = InMemoryCostLedgerWriter()
    writer.write(a_row())

    snapshot = writer.rows
    writer.write(a_row(tenant="cuan"))

    assert len(snapshot) == 1
    assert len(writer.rows) == 2


# --------------------------------------------------------------------------
# the pre-checks: no SQL, no effect
# --------------------------------------------------------------------------

AT_IDENTIFIER_CAP = "a" * MAX_IDENTIFIER_LENGTH
OVER_IDENTIFIER_CAP = "a" * (MAX_IDENTIFIER_LENGTH + 1)

REJECTED_IDENTIFIERS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_IDENTIFIER_CAP),
    ("uppercase", "Nordlicht"),
    ("an inner space", "nord licht"),
    ("a single quote, which would close an SQL literal", "nordlicht'"),
    ("a semicolon", "nordlicht;"),
    ("a double dash, which would start an SQL comment", "nordlicht--x"),
    ("a quoted identifier delimiter", 'nordlicht"'),
    ("a trailing newline", "nordlicht\n"),
    ("a trailing CRLF", "nordlicht\r\n"),
    ("a trailing space", "nordlicht "),
    ("a leading separator", "-nordlicht"),
    ("a dot", "nordlicht.eu"),
]

IDENTIFIER_FIELDS = ("tenant", "alias", "provider", "session")
"""The five slug-shaped strings on the row. `model` has its own wider rule
(decision D) and `trace_id` its own narrower one, so both are checked separately."""


@pytest.mark.parametrize("field", IDENTIFIER_FIELDS)
@pytest.mark.parametrize(
    ("case", "value"), REJECTED_IDENTIFIERS, ids=[case for case, _ in REJECTED_IDENTIFIERS]
)
def test_write_refuses_a_malformed_identifier_before_any_sql(
    field: str, case: str, value: str
) -> None:
    """Catches a writer that hands an unchecked string to PostgreSQL. Every other field
    on the row is well-formed, so the field under test is the ONLY thing that can refuse
    it; the pool double fails the test on any connection, which makes the claim "no SQL
    was attempted" rather than "the INSERT came back with an error". The newline and
    CRLF rows separate `re.fullmatch` from `re.match` with a trailing `$` (standing rule
    4), and parametrising the FIELD catches a check applied to `tenant` alone - `session`
    is the one that arrives straight from a client header."""
    writer = writer_over(ForbiddenPool())

    with pytest.raises(IdentifierInvalid):
        writer.write(a_row(**{field: value}))


REJECTED_MODELS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", "a" * (MAX_MODEL_LENGTH + 1)),
    ("an inner space", "qwen2.5 7b"),
    ("a single quote, which would close an SQL literal", "qwen2.5-7b'"),
    ("a semicolon", "qwen2.5-7b;"),
    ("a quoted identifier delimiter", 'qwen2.5-7b"'),
    ("a backslash", "qwen2.5-7b\\x"),
    ("a percent, which is psycopg's placeholder", "qwen2.5-7b%s"),
    ("a comma", "qwen2.5-7b,instruct"),
    ("a trailing newline", "qwen2.5-7b\n"),
    ("a trailing CRLF", "qwen2.5-7b\r\n"),
    ("a trailing space", "qwen2.5-7b "),
]


@pytest.mark.parametrize(
    ("case", "value"), REJECTED_MODELS, ids=[case for case, _ in REJECTED_MODELS]
)
def test_write_refuses_a_malformed_model_before_any_sql(case: str, value: str) -> None:
    r"""Catches decision D's rule being widened into "anything goes": `model` is the one
    field the resolver's slug check deliberately does not cover, so if this pre-check is
    missing nothing else refuses it and a provider id carrying a quote or a line break
    reaches the audit table. Every rejected value here is under the length cap except the
    row that names it, so the PATTERN is what has to refuse the rest."""
    writer = writer_over(ForbiddenPool())

    with pytest.raises(IdentifierInvalid):
        writer.write(a_row(model=value))


# Each of these is exactly TRACE_ID_LENGTH characters long, so the length check
# cannot be what refuses them - only the pattern can.
REJECTED_TRACE_IDS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character short", "0" * (TRACE_ID_LENGTH - 1)),
    ("one character long", "0" * (TRACE_ID_LENGTH + 1)),
    ("uppercase hex", TRACE_ID.upper()),
    ("a non-hex letter", "z" + TRACE_ID[1:]),
    ("a hyphen, as a dashed uuid would carry", "0af76519-16cd-43dd-8448-eb211c80319c"[:32]),
    ("a trailing newline, at the exact length", "0" * (TRACE_ID_LENGTH - 1) + "\n"),
    ("a trailing CRLF, at the exact length", "0" * (TRACE_ID_LENGTH - 2) + "\r\n"),
    ("a trailing space, at the exact length", "0" * (TRACE_ID_LENGTH - 1) + " "),
]


@pytest.mark.parametrize(
    ("case", "value"), REJECTED_TRACE_IDS, ids=[case for case, _ in REJECTED_TRACE_IDS]
)
def test_write_refuses_a_malformed_trace_id_before_any_sql(case: str, value: str) -> None:
    r"""Catches a `trace_id` written through unchecked. The three trailing-whitespace rows
    are the sharp ones: each is exactly 32 characters, so a length check alone accepts
    them and only `re.fullmatch` against 32 hex refuses - which is the difference between
    a ledger row that joins to a trace and one carrying an id with a line break in it.
    The 31 and 33 rows are the other side, where the length is what has to refuse."""
    writer = writer_over(ForbiddenPool())

    with pytest.raises(IdentifierInvalid):
        writer.write(a_row(trace_id=value))


def test_write_refuses_an_over_long_price_version_before_any_sql() -> None:
    """Catches the one field with no pattern and therefore no other guard at all.
    `price_version` is built from a price key and a date, so it is only as bounded as the
    key that fed it; without this cap a pathological key would land unbounded `text` in
    the audit table. The at-cap value is asserted accepted in the next case, so an
    off-by-one that refuses every legal version is not what makes this pass."""
    writer = writer_over(ForbiddenPool())

    with pytest.raises(IdentifierInvalid):
        writer.write(a_row(price_version="v" * (MAX_PRICE_VERSION_LENGTH + 1)))


# --------------------------------------------------------------------------
# the inclusive boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", IDENTIFIER_FIELDS)
def test_an_identifier_at_exactly_the_length_cap_is_accepted(field: str) -> None:
    """Catches an exclusive comparison - `>=` where `>` belongs - on `MAX_IDENTIFIER_
    LENGTH`, which would refuse the longest legal value on all five fields. The INSERT
    having been ISSUED is the assertion: a writer that returned quietly without writing
    anything would also raise nothing."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(**{field: AT_IDENTIFIER_CAP}))

    writer.flush()
    assert len(pool.inserts()) == 1


def test_a_model_at_exactly_the_length_cap_is_accepted() -> None:
    """The inclusive boundary of decision D's own cap, with the row one character over
    it asserted in the same case so an off-by-one in either direction fails here. Catches
    a cap that accepts 129 characters as readily as 128, which is how a `text` column
    ends up with no bound at all."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(model="m" * MAX_MODEL_LENGTH))

    writer.flush()
    assert len(pool.inserts()) == 1
    with pytest.raises(IdentifierInvalid):
        writer_over(ForbiddenPool()).write(a_row(model="m" * (MAX_MODEL_LENGTH + 1)))


def test_a_price_version_at_exactly_the_cap_is_accepted() -> None:
    """The inclusive boundary of `MAX_PRICE_VERSION_LENGTH`. Catches a check that refuses
    the longest legal version, which would turn every request priced by a long key into a
    lost row rather than a written one."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(price_version="v" * MAX_PRICE_VERSION_LENGTH))

    writer.flush()
    assert len(pool.inserts()) == 1


def test_a_trace_id_of_exactly_thirty_two_hex_characters_is_accepted() -> None:
    """The other direction for the trace id, so a check that refuses everything - which
    would satisfy every rejection row above and silently lose the entire ledger - cannot
    pass. The all-`f` value is here because a pattern written as `[0-9a-e]` or one that
    forgot the letters would accept the sample id and refuse this one."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(a_row(trace_id=TRACE_ID))
    writer.write(a_row(trace_id="f" * TRACE_ID_LENGTH))

    writer.flush()
    assert len(pool.inserts()) == 2


def test_a_row_with_every_field_at_its_cap_is_still_written() -> None:
    """Catches a length check applied to the CONCATENATION of the fields, or to the
    rendered statement, rather than to each one: every field here is legal on its own, so
    a writer summing them refuses a row nothing is wrong with."""
    pool = RecordingPool()
    writer = writer_over(pool)

    writer.write(
        a_row(
            tenant=AT_IDENTIFIER_CAP,
            alias=AT_IDENTIFIER_CAP,
            provider=AT_IDENTIFIER_CAP,
            session=AT_IDENTIFIER_CAP,
            model="m" * MAX_MODEL_LENGTH,
            price_version="v" * MAX_PRICE_VERSION_LENGTH,
        )
    )

    writer.flush()
    assert len(pool.inserts()) == 1


# --------------------------------------------------------------------------
# decision D: the pattern has to admit what the shipped policy resolves to
# --------------------------------------------------------------------------


def test_every_model_the_shipped_policy_can_resolve_to_passes_the_pattern(
    policy: Policy,
) -> None:
    """Decision D's trap, caught here rather than in production. `routing.yaml` names
    `qwen2.5-3b-instruct` and `bge-small-en-v1.5`, and both fail the resolver's slug rule
    because of the dots - so a `model` pre-check that reused `IDENTIFIER_PATTERN` would
    refuse, and therefore silently lose, every ledger row for the local provider while
    the gateway went on answering normally. The tenant OVERRIDES are swept too, because
    an override is a candidate the alias list does not contain. The set is asserted
    non-empty so an empty sweep cannot pass this."""
    models = {
        candidate.model for alias in policy.aliases.values() for candidate in alias.candidates
    }
    models |= {
        candidate.model
        for tenant in policy.tenants.values()
        for candidates in tenant.overrides.values()
        for candidate in candidates
    }

    assert models
    assert {model for model in models if MODEL_PATTERN.fullmatch(model) is None} == set()


def test_the_embedding_provider_id_passes_the_ledger_pre_check(policy: Policy) -> None:
    """The `provider` half of the same trap, swept off the shipped file rather than off a
    literal. Row 20e splits the local provider in two, and `provider` is a column the ledger
    pre-checks against the RESOLVER's identifier rule before any SQL runs - so a provider id
    that rule refuses raises `IdentifierInvalid` and silently loses every row booked under it,
    while the gateway goes on answering normally. Catches an implementation that names the new
    provider anything the rule refuses: `localEmbed` and `local.embed` fail the pattern,
    `local-embed-` and `local__embed` fail its separator rule, and a very long id fails the
    cap - none of which `routing.yaml`'s own schema, the resolver's happy path or any manifest
    test would notice. Swept over EVERY provider key, so the same failure introduced by a later
    row is caught here too, and asserted non-empty so an empty sweep cannot pass this."""
    providers = set(policy.providers)

    assert providers
    assert {name for name in providers if IDENTIFIER_PATTERN.fullmatch(name) is None} == set()
    assert {name for name in providers if len(name) > MAX_IDENTIFIER_LENGTH} == set()


ACCEPTED_MODELS = [
    "qwen2.5-7b-instruct",
    "bge-small-en-v1.5",
    "vendor/model-3.1-8b-instruct",
    "ft:model-4o:acme:2026",
    "MODEL-With-Caps",
]


@pytest.mark.parametrize("model", ACCEPTED_MODELS)
def test_the_model_pattern_admits_the_shapes_provider_ids_actually_take(model: str) -> None:
    r"""The other direction for decision D's pattern, one row per character class it has
    to admit: a dot, a slash, a colon and an uppercase letter. Catches a pattern narrowed
    one class at a time - dropping `/` alone would lose every hosted open-weights id, and
    dropping `:` alone every fine-tune - which no single rejection case can see."""
    assert MODEL_PATTERN.fullmatch(model) is not None


# --------------------------------------------------------------------------
# `record_cost`: the price comes from the table (decision C)
# --------------------------------------------------------------------------


def test_the_cost_is_priced_from_the_table_and_cannot_be_supplied_by_the_caller() -> None:
    """Decision C. Catches a `record_cost` that takes a finished cost from its caller -
    the shape that records the gateway's belief rather than the price table's answer, and
    that lets the number in the row disagree with the `price_version` beside it. The two
    prices in the table differ per direction, so a row that priced both token counts at
    the input price is a different number and fails here; asserting the parameter list as
    well is what pins that there is no way to pass one in at all."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger)

    parameters = set(inspect.signature(record_cost).parameters)
    assert parameters & {"cost", "cost_usd", "price_version", "estimated"} == set()
    row = ledger.rows[0]
    assert row.cost_usd == USAGE_COST
    assert row.price_version == PRICED_VERSION
    assert row.estimated is False


def test_an_unpriced_key_is_recorded_as_zero_at_six_places_and_flagged_estimated() -> None:
    """The criterion's exact words, and the case that must never raise: a model added to
    `routing.yaml` and not to its provider file is a normal Friday, and a `record_cost` that
    raised there would take out the request as well as the row. Catches a row filed with
    no cost at all, or with a NULL price version, either of which makes the unpriced rows
    unfindable - the literal `unpriced` is what an operator greps for. The exponent is
    asserted because `Decimal("0") == Decimal("0.000000")`, so equality alone would not
    see a number that lost its scale."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger, price_key=UNPRICED_KEY)

    row = ledger.rows[0]
    assert row.cost_usd == Decimal("0.000000")
    assert row.cost_usd.as_tuple().exponent == -6
    assert row.price_version == UNPRICED_VERSION
    assert row.estimated is True


def test_absent_usage_is_priced_from_the_characters_and_flagged_estimated() -> None:
    """The estimate half of the criterion. Catches a `record_cost` that books a request
    with no provider usage as free - zero tokens and zero cost is the shape that makes a
    whole provider look cheap - and one that files the estimate with `estimated=false`,
    which would make a number nobody measured indistinguishable from a metered one. The
    token counts on the row are asserted too, so a cost computed from the characters but
    stored beside zero tokens fails here."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger, usage=None, prompt_chars=40, completion_chars=20)

    row = ledger.rows[0]
    # 40 // 4 input tokens and 20 // 4 output tokens, at 3 and 15 per million.
    assert (row.input_tokens, row.output_tokens) == (10, 5)
    assert row.cost_usd == Decimal("0.000105")
    assert row.estimated is True
    assert row.price_version == PRICED_VERSION


def test_the_row_names_what_served_and_is_stamped_from_the_injected_clock() -> None:
    """Catches a row filled from the wrong argument - `model` taking the alias, `provider`
    taking the tenant - which is invisible in a green suite and turns the
    console's per-model spend into a chart of one row. Every field is asserted at once
    because they arrive from different sources, and `cached=True` is carried through here
    because the rollup counts cached requests separately. The timestamp is the clock's:
    a database-side `default now()` would file a request under the wrong day for anyone
    whose evening is another day's morning, and the day is what the rollup groups by.
    One row, because a ledger that wrote twice per request doubles every tenant's spend."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger, cached=True)

    assert len(ledger.rows) == 1
    row = ledger.rows[0]
    assert row.ts == WHEN
    assert row.tenant == TENANT
    assert row.alias == ALIAS
    assert row.provider == PROVIDER
    assert row.model == MODEL
    assert row.input_tokens == USAGE.prompt_tokens
    assert row.output_tokens == USAGE.completion_tokens
    assert row.cached is True
    assert row.session == SESSION
    assert row.trace_id == TRACE_ID


def test_a_null_session_is_accepted_and_reaches_the_writer_as_none() -> None:
    """Catches a `session` coerced to a placeholder - `""` or the string `"None"` - on the
    way to a nullable column. Telemetry carries session and tenant and never a user, so
    "no session" is a real and common state; a row carrying an empty string would join to
    a session that does not exist and make the console's session count wrong in the
    direction that looks like traffic. The identifier pre-check refusing `""` is what
    makes the placeholder shape a lost row rather than a wrong one, so this also pins
    that `None` is not run through that check."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger, session=None)

    assert ledger.rows[0].session is None


def test_a_failing_write_is_swallowed() -> None:
    """Decision E, and the same rule as an earlier change decision D. Catches an unguarded call:
    the provider has already charged by the time this runs, so a Postgres blip would turn
    a served request into a 500 and lose the answer as well as the row. The attempt count
    is what stops a `record_cost` that never writes at all from passing - it would
    satisfy "does not raise" perfectly."""
    ledger = RaisingWriter()

    record(ledger)

    assert ledger.calls == 1


class RefusingWriter:
    """A writer whose pre-check refuses every row, as the real one does for a bad value."""

    def __init__(self) -> None:
        self.calls = 0

    def write(self, row: CostLedgerRow) -> None:
        self.calls += 1
        raise IdentifierInvalid("model: not a provider model id")


def test_a_row_failing_its_pre_check_is_counted_dropped_invalid() -> None:
    """An earlier change. Catches a `record_cost` that only logs a row its writer refused: the row
    is lost and no metric, so no alert, says so. It still must not raise (the provider has
    charged), so the call count and the counter are what show it was handed over and counted."""
    from app.core.metrics import registry

    def dropped_invalid() -> float:
        for family in registry().collect():
            if family.name == "gateway_ledger_writes":
                for sample in family.samples:
                    if sample.labels.get("result") == "dropped_invalid":
                        return float(sample.value)
        return 0.0

    ledger = RefusingWriter()
    before = dropped_invalid()

    record(ledger)

    assert ledger.calls == 1
    assert dropped_invalid() - before == 1.0


# --------------------------------------------------------------------------
# a cached request costs zero, and the ledger is what says so (14c decision C)
# --------------------------------------------------------------------------

ZERO = Decimal("0.000000")

CACHED_PRICING_CASES: list[tuple[str, bool, str, Decimal, str, bool]] = [
    ("a fresh request at a priced key", False, PRICED_KEY, USAGE_COST, PRICED_VERSION, False),
    ("a cached request at a priced key", True, PRICED_KEY, ZERO, PRICED_VERSION, False),
    ("a fresh request at an unpriced key", False, UNPRICED_KEY, ZERO, UNPRICED_VERSION, True),
    ("a cached request at an unpriced key", True, UNPRICED_KEY, ZERO, UNPRICED_VERSION, True),
]
"""The whole decision table `cached` x priced, because zero has two different
meanings here and only the version string tells them apart. The two `cached=False`
rows are what stop the table passing on an implementation that returns zero for
everything, and the unpriced rows are what stop it passing on one that answers the
price version in force whether or not a price was found."""


@pytest.mark.parametrize(
    ("case", "cached", "price_key", "cost", "version", "estimated"),
    CACHED_PRICING_CASES,
    ids=[case for case, *_ in CACHED_PRICING_CASES],
)
def test_a_cached_request_is_booked_at_zero_with_the_price_version_in_force(
    case: str,
    cached: bool,
    price_key: str,
    cost: Decimal,
    version: str,
    estimated: bool,
) -> None:
    """An earlier change decision C: the zero rule lives HERE, not in the route. Catches a
    `record_cost` that prices a cache hit at the cost it AVOIDED, which puts money that
    was never spent into the spend column where no `WHERE cached = false` can find it;
    catches one that answers a NULL or `unpriced` version on a hit, which would lose the
    price a hit was measured against and with it an earlier change's cache-savings figure; and
    catches a route-side zero being the only thing that makes a hit free, since nothing
    in this file is a route. The exponent is asserted because `Decimal("0") ==
    Decimal("0.000000")`, so equality alone would not see a number that lost its scale."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger, cached=cached, price_key=price_key)

    row = ledger.rows[0]
    assert row.cached is cached
    assert row.cost_usd == cost
    assert row.cost_usd.as_tuple().exponent == -6
    assert row.price_version == version
    assert row.estimated is estimated


def test_a_cached_row_keeps_the_replayed_answers_token_counts() -> None:
    """The other half of decision C, and the half a zero cost makes easy to lose. Catches
    a `record_cost` that zeroes the TOKENS along with the cost on a hit - the shape that
    reads as "this request consumed nothing" - which would make the cached rows useless
    for the one thing they are kept for: saying how much a tenant did not spend. The
    counts differ from each other, so a swap is visible as well as a zero-fill."""
    ledger = InMemoryCostLedgerWriter()

    record(ledger, cached=True)

    row = ledger.rows[0]
    assert (row.input_tokens, row.output_tokens) == (
        USAGE.prompt_tokens,
        USAGE.completion_tokens,
    )


# --------------------------------------------------------------------------
# `record_cost` answers the row it built (14c decision G)
# --------------------------------------------------------------------------


def test_record_cost_answers_the_row_it_wrote() -> None:
    """An earlier change decision G. Catches a `record_cost` that stays `-> None` and leaves the
    route to re-price for the header and the metric, which is how the row, the counter and
    `X-Gateway-Cost-USD` become three numbers that can disagree; and catches one that
    answers a SECOND row built from the same arguments, which would carry a different
    timestamp and, on a table reloaded between the two, a different price. Identity is
    asserted, not equality: two rows built from one call compare equal on every field."""
    ledger = InMemoryCostLedgerWriter()

    row = record(ledger)

    assert row is ledger.rows[0]
    assert row.cost_usd == USAGE_COST
    assert row.price_version == PRICED_VERSION


def test_record_cost_answers_the_row_even_when_the_write_failed() -> None:
    """Decision G's whole point, and the case that separates it from a convenience. The
    header describes what the request COST, not what the database managed to keep, so a
    writer that raised must still yield the number the caller reports - 14b decision E
    already says the ledger is a lower bound during an incident. Catches a `record_cost`
    that answers `None` on a failed write, which would leave the route with nothing to put
    in the header and the counter exactly when a Postgres blip is under way, and catches
    one that answers a zero-cost row as a placeholder. The attempt count is what stops a
    `record_cost` that never wrote at all from passing."""
    ledger = RaisingWriter()

    row = record(ledger)

    assert ledger.calls == 1
    assert row.cost_usd == USAGE_COST
    assert row.price_version == PRICED_VERSION
    assert row.estimated is False
