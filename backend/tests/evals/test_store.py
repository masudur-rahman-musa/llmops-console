r"""`storage/eval_store.py`: the two rows a run stores, and the checks that run before any SQL.

Nothing here opens a connection. Two pool doubles do the work, and which one a
case takes IS the claim it makes:

* `ForbiddenPool` fails the test the moment anything asks it for a connection, so
  "the pre-check ran BEFORE the borrow" is asserted as "no connection was
  borrowed at all", never as "the INSERT came back with an error";
* `RecordingPool` keeps every statement and every `executemany`, so an accepted
  boundary can assert the write was actually ISSUED - a writer that returned
  quietly without writing anything raises nothing either, and would pass an "is
  accepted" case that only checked for the absence of an exception.

The cases needing a real server live in tests/storage/test_eval_store_postgres.py
and are skipped without `OPSCONSOLE_TEST_DATABASE_URL`.

Every pattern is checked with `fullmatch`: `tenant`, `provider`, `case_id` and
`category` against the resolver's identifier rule, `git_sha` against
`report.SET_SHA_PATTERN`, `run_kind` against the three words `ops-eval` accepts. Each
rejection table below carries a trailing newline, a trailing CRLF and a trailing space.

An earlier change rewrote this file for the conversation report. A case row's `details` holds the
three checks, the judged replies and the error. The four measured columns stay NULL
because a conversation answer measures none of them. `category` is any slug, because the
help desk names its own categories, so the old closed list of twelve is gone with the
single-reply format. The tests that only covered the removed assertions, residency
verdict, taxonomy tags and 1-to-5 judge verdict were deleted with those modules.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from psycopg import Connection
from psycopg.adapt import Transformer
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb, JsonbBinaryDumper
from psycopg_pool import ConnectionPool

from app.cli.evals import RUN_KINDS
from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.evals.checks import CaseCheck, TurnCheck
from app.evals.report import REPORT_JSON_NAME, RunCalibration, RunReport, build_report
from app.evals.runner import CaseLine
from app.storage.eval_store import (
    MAX_REPORT_PATH_LENGTH,
    MAX_SET_NAME_LENGTH,
    MAX_TARGET_LENGTH,
    EvalCaseRow,
    EvalRunRow,
    PostgresEvalStore,
    rows_from_report,
)

from .builders import case_line

Pool = ConnectionPool[Connection[TupleRow]]

STARTED = datetime(2026, 3, 4, 9, 12, 31, 204512, tzinfo=UTC)
FINISHED = datetime(2026, 3, 4, 9, 14, 2, 881044, tzinfo=UTC)

# Every identity value distinct from every other and of a different shape where
# it can be helped, so a writer that fills one column from another argument shows
# up as a wrong value rather than as a plausible one.
SET_NAME = "cases.jsonl"
SET_SHA = "e4b2c0d9a71f3856cbb4d20e9f61a7c845d3e0b2"
GIT_SHA = "b71f0c3a9d5e42188a6c0f37d21be9054c8a7f6d"
TENANT = "all"
TARGET = "https://helpdesk.example/gateway"
RUN_ID = "9f2c1d7e4b0a48f5a6d3c1e0b7248f5a"
PROVIDER = "anthropic_eu"

SEVEN_HEX = "a1b2c3d"
SIXTY_FOUR_HEX = "0123456789abcdef" * 4

AT_IDENTIFIER_CAP = "t" * MAX_IDENTIFIER_LENGTH
OVER_IDENTIFIER_CAP = "t" * (MAX_IDENTIFIER_LENGTH + 1)


def build(lines: tuple[CaseLine, ...], *, run_kind: str = "ci") -> RunReport:
    """`build_report` over saved case lines, with the identity values above."""
    return build_report(
        lines,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name=SET_NAME,
        set_sha=SET_SHA,
        set_sha_source="commit",
        run_kind=run_kind,
        git_sha=GIT_SHA,
        tenant=TENANT,
        target=TARGET,
    )


MIXED = (
    case_line("alpha", passed=True, verdicts=["pass"], category="paraphrases"),
    case_line("beta", passed=False, verdicts=["fail", "pass"], category="follow_ups"),
    case_line("gamma", passed=False, category="attacks"),
    case_line("delta", passed=False, error="turn failed: HTTP 502 for delta", category="typos"),
)
"""Four cases, one of each shape a real run produces, and no two alike in category, verdict
or judged replies - so a mapping that fills one row from another case's record cannot pass
by coincidence."""


def a_run_row(**overrides: Any) -> EvalRunRow:  # noqa: ANN401
    """One well-formed run row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "id": RUN_ID,
        "started_at": STARTED,
        "finished_at": FINISHED,
        "set_name": SET_NAME,
        "set_sha": SET_SHA,
        "set_sha_source": "commit",
        "run_kind": "ci",
        "target": TARGET,
        "tenant": TENANT,
        "git_sha": GIT_SHA,
        "metrics": {"cases": 1, "passed": 1, "failed": 0},
        "passed": True,
        "report_path": f"/srv/evals/nightly/{REPORT_JSON_NAME}",
    }
    fields.update(overrides)
    return EvalRunRow(**fields)


def a_case_row(**overrides: Any) -> EvalCaseRow:  # noqa: ANN401
    """One well-formed case row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "case_id": "alpha",
        "category": "paraphrases",
        "passed": True,
        "latency_ms": 137,
        "cost_usd": Decimal("0.001200"),
        "provider": PROVIDER,
        "details": {"check": None, "judged": [], "error": None},
    }
    fields.update(overrides)
    return EvalCaseRow(**fields)


# --------------------------------------------------------------------------
# the two pool doubles
# --------------------------------------------------------------------------


class ForbiddenPool:
    """A pool that fails the test if the store asks it for a connection at all."""

    def connection(self) -> NoReturn:
        raise AssertionError("a connection was borrowed before the pre-checks refused the row")


class RecordingCursor:
    """A cursor that records what it was asked to run and runs none of it."""

    def __init__(self, pool: RecordingPool) -> None:
        self._pool = pool

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def execute(self, query: object, params: object = None) -> RecordingCursor:
        self._pool.statements.append(str(query))
        return self

    def executemany(self, query: object, params_seq: object) -> None:
        self._pool.statements.append(str(query))
        self._pool.executemany_calls.append((str(query), list(cast(Any, params_seq))))

    def fetchall(self) -> list[tuple[object, ...]]:
        return []

    def fetchone(self) -> tuple[object, ...] | None:
        return None


class RecordingConnection:
    """A connection recording every statement, every transaction and every cursor."""

    def __init__(self, pool: RecordingPool) -> None:
        self._pool = pool

    def execute(self, query: object, params: object = None) -> RecordingCursor:
        # `str()` rather than the object, because the statement arrives as a
        # `psycopg.sql.Composed` whose text is in its repr.
        self._pool.statements.append(str(query))
        return RecordingCursor(self._pool)

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self._pool)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._pool.transactions += 1
        yield


class RecordingPool:
    """Every statement the store issued, and how it grouped them."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.executemany_calls: list[tuple[str, list[Any]]] = []
        self.transactions = 0

    @contextmanager
    def connection(self) -> Iterator[RecordingConnection]:
        yield RecordingConnection(self)

    def inserts(self) -> list[str]:
        """The statements that write a row."""
        return [text for text in self.statements if "insert" in text.lower()]


def store_over(pool: ForbiddenPool | RecordingPool) -> PostgresEvalStore:
    """The real store over a double, with the pool typing narrowed at one seam."""
    return PostgresEvalStore(cast(Pool, pool))


def write_refused(**overrides: Any) -> None:  # noqa: ANN401
    """Write one run row with the named fields replaced, expecting a refusal and no borrow."""
    with pytest.raises(IdentifierInvalid):
        store_over(ForbiddenPool()).write(a_run_row(**overrides), (a_case_row(),))


def case_write_refused(**overrides: Any) -> None:  # noqa: ANN401
    """Write one case row with the named fields replaced, expecting a refusal and no borrow."""
    with pytest.raises(IdentifierInvalid):
        store_over(ForbiddenPool()).write(a_run_row(), (a_case_row(**overrides),))


def written(**overrides: Any) -> RecordingPool:  # noqa: ANN401
    """Write one run row and one case row through a recording pool, and answer it."""
    pool = RecordingPool()
    store_over(pool).write(a_run_row(**overrides), (a_case_row(),))
    return pool


# --------------------------------------------------------------------------
# rows_from_report: the run row
# --------------------------------------------------------------------------


def test_the_run_row_carries_every_identity_field_from_the_report() -> None:
    """Catches a mapping that fills a column from the wrong field of the report - the two
    SHAs, the set name, the tenant and the target are all `text`, so `set_sha` written
    into `git_sha` is a row PostgreSQL accepts without a murmur. `id`, `passed` and
    `metrics` are asserted from the report too, because a store that generated its own run
    id would break the join every case row makes back to it, and `metrics` is what the
    gate reads. The metrics are compared against the report's JSON form, the exact bytes
    the file holds."""
    report = build(MIXED)

    run_row, _ = rows_from_report(report, report_path=Path("/srv/evals/nightly/report.json"))

    assert run_row.id == report.id
    assert run_row.started_at == STARTED
    assert run_row.finished_at == FINISHED
    assert run_row.set_name == SET_NAME
    assert run_row.set_sha == SET_SHA
    assert run_row.set_sha_source == "commit"
    assert run_row.run_kind == "ci"
    assert run_row.target == TARGET
    assert run_row.tenant == TENANT
    assert run_row.git_sha == GIT_SHA
    assert run_row.passed is False
    assert run_row.metrics == json.loads(report.model_dump_json())["metrics"]
    assert run_row.metrics["check_score"] == 25.0


def test_there_is_one_case_row_per_case_report_in_report_order() -> None:
    """Catches a store that writes only the failures, only the cases that answered, or
    de-duplicates by category: the denominator of every score is counted over these rows.
    Order is the report's own, which is the cases file's order."""
    report = build(MIXED)

    run_row, case_rows = rows_from_report(report, report_path=Path("/tmp/report.json"))

    assert [row.case_id for row in case_rows] == ["alpha", "beta", "gamma", "delta"]
    assert [row.category for row in case_rows] == ["paraphrases", "follow_ups", "attacks", "typos"]
    assert [row.passed for row in case_rows] == [True, False, False, False]
    assert {row.run_id for row in case_rows} == {run_row.id}


def test_a_conversation_case_stores_no_measurements() -> None:
    """Catches the four measured columns filled with `0` and `""`: a conversation answer
    measures no latency, cost, or provider, and a zero would be averaged in as a
    free, instant success. `is None` on each, for every case, the answered ones too."""
    _, case_rows = rows_from_report(build(MIXED), report_path=Path("/tmp/report.json"))

    for row in case_rows:
        assert row.latency_ms is None
        assert row.cost_usd is None
        assert row.provider is None
        assert row.tags == (("judge_low",) if row.case_id == "beta" else ())


def test_a_failed_case_row_carries_the_report_tags() -> None:
    """Catches a mapping that drops the tags (the row gets `()` and Taxonomy stays empty) or
    reshapes them: a case tagged with two names must reach its row with both, in order."""
    line = CaseLine(
        case_id="tagged",
        category="attacks",
        passed=False,
        error=None,
        check=CaseCheck(
            tools_in_order=False,
            turns=(TurnCheck(turn=1, no_early_change=False, no_made_up_name=True),),
        ),
        judged=(),
        answer=None,
    )
    report = build((line,))

    _, case_rows = rows_from_report(report, report_path=Path("/tmp/report.json"))

    assert report.cases[0].tags == ("wrong_tool", "early_change")
    assert case_rows[0].tags == report.cases[0].tags


def test_the_report_path_is_the_report_json_that_was_written() -> None:
    """Catches a store recording the `--out` DIRECTORY rather than the file: the gate and
    the console both open `report_path` expecting a document."""
    written_path = Path("/srv/evals/nightly") / REPORT_JSON_NAME

    run_row, _ = rows_from_report(build(MIXED), report_path=written_path)

    assert run_row.report_path == str(written_path)
    assert run_row.report_path.endswith(REPORT_JSON_NAME)


def test_a_case_that_never_answered_keeps_its_error_in_the_details() -> None:
    """Catches an error that is lost once `report.json` has been rotated away: the row is
    all that is left, and it must still say why the case failed. `check` is `null`, not
    absent."""
    _, case_rows = rows_from_report(build(MIXED), report_path=Path("/tmp/report.json"))

    broken = case_rows[3]
    assert broken.case_id == "delta"
    assert broken.details["error"] == "turn failed: HTTP 502 for delta"
    assert broken.details["check"] is None


def test_the_details_column_carries_the_judged_replies() -> None:
    """Catches a `details` that drops the judged replies, which are the only record of WHY
    a case whose three checks passed was still marked down by the judge. The keys are the
    same on a case nothing judged (`judged` is an empty list), because a key that comes and
    goes makes two rows of the same set unqueryable by one expression. The dump goes through
    psycopg's `Jsonb` for real, since a pydantic model there is a `TypeError` at INSERT time."""
    _, case_rows = rows_from_report(build(MIXED), report_path=Path("/tmp/report.json"))

    judged = case_rows[1].details
    assert set(judged) == {"check", "judged", "error"}
    assert [(r["turn"], r["verdict"]) for r in judged["judged"]] == [(1, "fail"), (2, "pass")]
    assert case_rows[2].details["judged"] == []
    dumper = JsonbBinaryDumper(Jsonb, Transformer())
    assert dumper.dump(Jsonb(dict(judged))) is not None


def test_the_columns_nothing_fills_arrive_empty_rather_than_absent() -> None:
    """Catches `judge_kappa` or `alias_resolution` filled with `0.0`, which a dashboard
    would render as a real, terrible score. This report carries no calibration, so both are
    `None`. Asserted exactly rather than as falsy, because `0.0` and `None` are both falsy."""
    run_row, _ = rows_from_report(build(MIXED), report_path=Path("/tmp/report.json"))

    assert run_row.judge_kappa is None
    assert run_row.alias_resolution is None


def test_the_stored_metrics_are_the_json_the_report_file_carries() -> None:
    """Catches a `metrics` column that differs from `report.json`'s: the two must state the
    same numbers. `check_score` and `judge_score` are JSON-native, so the dump through
    psycopg's `Jsonb` works; the old metric names are gone."""
    report = build(MIXED)

    run_row, _ = rows_from_report(report, report_path=Path("/tmp/report.json"))

    assert run_row.metrics == json.loads(report.model_dump_json())["metrics"]
    assert run_row.metrics["judge_score"] == 66.67
    assert "decision_accuracy" not in run_row.metrics
    dumper = JsonbBinaryDumper(Jsonb, Transformer())
    assert dumper.dump(Jsonb(dict(run_row.metrics))) is not None


# --------------------------------------------------------------------------
# the caps this module publishes
# --------------------------------------------------------------------------


def test_the_length_caps_are_the_published_ones() -> None:
    """Catches a cap silently widened or narrowed. These three are the only bound on
    values no other module caps at all - a set name and a target come from a command line,
    a report path from a filesystem - so a cap that drifts is the difference between a
    bounded `text` column and an unbounded one. Asserted as the exact integers a range
    would admit either side of."""
    assert MAX_SET_NAME_LENGTH == 128
    assert MAX_REPORT_PATH_LENGTH == 1024
    assert MAX_TARGET_LENGTH == 512


# --------------------------------------------------------------------------
# the inclusive boundaries
# --------------------------------------------------------------------------


CAPPED_FIELDS: list[tuple[str, str, int]] = [
    ("set_name", "s", MAX_SET_NAME_LENGTH),
    ("report_path", "p", MAX_REPORT_PATH_LENGTH),
    ("target", "t", MAX_TARGET_LENGTH),
    ("tenant", "n", MAX_IDENTIFIER_LENGTH),
]


@pytest.mark.parametrize(
    ("field", "filler", "cap"), CAPPED_FIELDS, ids=[row[0] for row in CAPPED_FIELDS]
)
def test_a_run_row_value_at_exactly_its_cap_is_accepted(field: str, filler: str, cap: int) -> None:
    """Catches an exclusive comparison - `>=` where `>` belongs - which refuses the longest
    legal value on each of the four. That is the half of the boundary a rejection table
    never sees, and here it would lose a whole run's row over a long directory name. The
    write having been ISSUED is the assertion: a store that returned quietly without
    writing anything also raises nothing."""
    pool = written(**{field: filler * cap})

    assert len(pool.inserts()) >= 1
    assert len(pool.executemany_calls) == 1


@pytest.mark.parametrize(
    ("field", "filler", "cap"), CAPPED_FIELDS, ids=[row[0] for row in CAPPED_FIELDS]
)
def test_a_run_row_value_one_character_over_its_cap_is_refused_before_the_borrow(
    field: str, filler: str, cap: int
) -> None:
    """The other half of the same boundary. Catches a cap that accepts one character over
    as readily as one under, which is how a `text` column ends up with no bound at all, and
    catches a check placed after the borrow: the pool double fails the test on the
    connection itself, so this is "nothing was borrowed" rather than "the INSERT errored"."""
    write_refused(**{field: filler * (cap + 1)})


def test_a_case_id_at_exactly_the_identifier_cap_is_accepted_and_one_over_is_refused() -> None:
    """Catches a `case_id` cap restated as this module's own literal, which would then
    drift from the one 16a's loader applies: a case id a golden set may legally carry
    would become a run whose rows are refused at the end, after every case has been paid
    for. Both directions in one case, so an off-by-one either way fails here."""
    pool = RecordingPool()

    store_over(pool).write(a_run_row(), (a_case_row(case_id=AT_IDENTIFIER_CAP),))

    assert len(pool.executemany_calls) == 1
    case_write_refused(case_id=OVER_IDENTIFIER_CAP)


# --------------------------------------------------------------------------
# the identifier vocabulary
# --------------------------------------------------------------------------


REJECTED_IDENTIFIERS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_IDENTIFIER_CAP),
    ("uppercase", "Nordlicht"),
    ("a space", "nord licht"),
    ("a single quote", "nordlicht'"),
    ("a semicolon, which would close the statement", "nordlicht; select 1"),
    ("a trailing newline", "nordlicht\n"),
    ("a trailing CRLF", "nordlicht\r\n"),
    ("a trailing space", "nordlicht "),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_IDENTIFIERS, ids=[row[0] for row in REJECTED_IDENTIFIERS]
)
def test_write_refuses_a_malformed_tenant_before_any_sql(label: str, value: str) -> None:
    r"""Catches a `tenant` taken on trust into the audit row that says which tenant's
    golden set was measured. Every value here is bound as a parameter, so the check is not
    what stands between the store and an injection - it is what keeps the stored row
    honest, and `nordlicht\n` and `Nordlicht` are two tenants a query would count as one
    and as none. The three trailing-whitespace rows are what separate `re.fullmatch` from
    an anchored `.match`, which accepts the newline."""
    write_refused(tenant=value)


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_IDENTIFIERS, ids=[row[0] for row in REJECTED_IDENTIFIERS]
)
def test_write_refuses_a_malformed_provider_before_any_sql(label: str, value: str) -> None:
    """Catches a check applied to the run row's identifiers and not to the case rows',
    which is the easy half to forget: `provider` arrives from a target's own
    response headers rather than from a command line, so it is among the least trusted
    strings in the whole record."""
    case_write_refused(provider=value)


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_IDENTIFIERS, ids=[row[0] for row in REJECTED_IDENTIFIERS]
)
def test_write_refuses_a_malformed_case_id_before_any_sql(label: str, value: str) -> None:
    """Catches a `case_id` written straight through from a report that was never loaded by
    16a's loader - a report file is a document anyone can hand this store. The case id is
    what a regression is chased by, so `alpha\\n` and `Alpha` beside `alpha` are three
    cases nobody can join, and the trailing rows are a project rule's."""
    case_write_refused(case_id=value)


def test_a_case_that_measured_nothing_is_written_rather_than_refused() -> None:
    """The other direction for the case-row provider, and the row a broken turn
    produces. Catches a pre-check that runs on `provider` unconditionally:
    `None` is not a malformed identifier, it is the absence the NULL column exists for,
    and a store refusing it would lose exactly the rows a failing target produces - the
    ones a run is stored for."""
    pool = RecordingPool()

    store_over(pool).write(
        a_run_row(),
        (
            a_case_row(),
            a_case_row(
                case_id="delta",
                passed=False,
                latency_ms=None,
                cost_usd=None,
                provider=None,
                details={"check": None, "judged": [], "error": "turn failed"},
            ),
        ),
    )

    assert len(pool.executemany_calls) == 1
    assert len(pool.executemany_calls[0][1]) == 2


# --------------------------------------------------------------------------
# git_sha, run_kind and category
# --------------------------------------------------------------------------


ACCEPTED_GIT_SHAS: list[tuple[str, str | None]] = [
    ("git could not answer", None),
    ("the shortest accepted, seven", SEVEN_HEX),
    ("a full commit", GIT_SHA),
    ("the longest accepted, sixty-four", SIXTY_FOUR_HEX),
]


@pytest.mark.parametrize(
    ("label", "value"), ACCEPTED_GIT_SHAS, ids=[row[0] for row in ACCEPTED_GIT_SHAS]
)
def test_a_git_sha_the_pattern_admits_or_none_is_written(label: str, value: str | None) -> None:
    """Catches a check that refuses everything - which would satisfy every rejection row
    below and lose every run - and, in the `None` row, one that refuses the absence 16c
    records when `git rev-parse` cannot answer. `null` is the honest value there, and a
    store that refused it would turn a run made outside a checkout into no row at all. The
    seven and sixty-four rows are `SET_SHA_PATTERN`'s inclusive boundaries."""
    pool = written(git_sha=value)

    assert len(pool.executemany_calls) == 1


REJECTED_GIT_SHAS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character under the floor, six", "a1b2c3"),
    ("one character over the cap, sixty-five", SIXTY_FOUR_HEX + "a"),
    ("uppercase hex", "A1B2C3D"),
    ("not hex at all", "zzzzzzz"),
    ("a path separator", "../a1b2c3d"),
    ("a trailing newline", SEVEN_HEX + "\n"),
    ("a trailing CRLF", SEVEN_HEX + "\r\n"),
    ("a trailing space", SEVEN_HEX + " "),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_GIT_SHAS, ids=[row[0] for row in REJECTED_GIT_SHAS]
)
def test_write_refuses_a_malformed_git_sha_before_any_sql(label: str, value: str) -> None:
    r"""Catches a `git_sha` stored as whatever a caller handed over. It is the column a
    regression is bisected on, so `b71f0c3\n` joins to no commit and `A1B2C3D` joins to
    none either while looking perfectly plausible in a table. Nothing else in this store
    looks at the value, so removing the pre-check is the only thing that flips these rows,
    and the three trailing rows are the ones an anchored `.match` accepts."""
    write_refused(git_sha=value)


@pytest.mark.parametrize("run_kind", RUN_KINDS)
def test_each_run_kind_the_command_accepts_is_written(run_kind: str) -> None:
    """Catches a vocabulary in this module that has drifted from the one `ops-eval`'s
    `--run-kind` accepts: a word the command line takes and the store refuses would fail
    every nightly run at the last step, after the target has been measured. All three are
    covered because a check built around one of them passes any single row."""
    pool = written(run_kind=run_kind)

    assert len(pool.executemany_calls) == 1


REJECTED_RUN_KINDS: list[tuple[str, str]] = [
    ("empty", ""),
    ("a fourth word", "smoke"),
    ("uppercase", "CI"),
    ("a near miss", "nightly-2"),
    ("a trailing newline", "ci\n"),
    ("a trailing CRLF", "ci\r\n"),
    ("a trailing space", "ci "),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_RUN_KINDS, ids=[row[0] for row in REJECTED_RUN_KINDS]
)
def test_write_refuses_a_run_kind_outside_the_three_before_any_sql(label: str, value: str) -> None:
    r"""Catches a `run_kind` reaching the column as free text. A gate run and a nightly run
    are told apart by it, so `CI` and `ci\n` are gate runs that no query for `ci` ever
    finds - the gate would report on a subset of its own history and say nothing. The
    CHECK constraint in
    `003_eval_runs.sql` is the second guard and it is pinned in
    tests/storage/test_eval_store_postgres.py; this is the one that refuses before a
    connection is borrowed at all."""
    write_refused(run_kind=value)


HELP_DESK_CATEGORIES = (
    "attacks",
    "changes_of_mind",
    "follow_ups",
    "for_other_people",
    "live_failures",
    "off_topic",
    "other_languages",
    "paraphrases",
    "short_answers_and_picks",
    "typos",
)


def test_every_category_the_help_desk_uses_is_written() -> None:
    """Catches a check that refuses a category the help desk names: it owns the list, and a
    refused category would lose the rows of a whole run at the last step. All ten, with
    underscores, because a slug check that forgot them refuses most of the set."""
    pool = RecordingPool()
    store_over(pool).write(
        a_run_row(),
        tuple(
            a_case_row(case_id=f"case-{index}", category=category)
            for index, category in enumerate(HELP_DESK_CATEGORIES)
        ),
    )

    assert len(pool.executemany_calls[0][1]) == len(HELP_DESK_CATEGORIES)


REJECTED_CATEGORIES: list[tuple[str, str]] = [
    ("empty", ""),
    ("uppercase", "HELPDESK"),
    ("a space", "follow ups"),
    ("a trailing newline", "typos\n"),
    ("a trailing CRLF", "typos\r\n"),
    ("a trailing space", "typos "),
]


@pytest.mark.parametrize(
    ("label", "value"), REJECTED_CATEGORIES, ids=[row[0] for row in REJECTED_CATEGORIES]
)
def test_write_refuses_a_malformed_category_before_any_sql(label: str, value: str) -> None:
    r"""Catches a category reaching the table as free text: `typos\n` is a category with one
    case in it and `typos` is one case short, and neither reads as a mistake in a grouped
    query. A report file reaches this store without passing the case loader, so this check
    is the only one on this path."""
    case_write_refused(category=value)


# --------------------------------------------------------------------------
# the shape of the write (decision G)
# --------------------------------------------------------------------------


def test_the_run_row_and_its_cases_are_written_in_one_transaction() -> None:
    """Decision G before a server can be involved. Catches a store issuing a statement per
    case with no transaction around them: a run whose cases half-landed is counted by
    every later query as a run with fewer cases, and nothing downstream can tell that from
    a shorter golden set. One `executemany` over the cases is asserted with it, because a
    per-case loop is the shape that makes the half-landing possible in the first place."""
    report = build(MIXED)
    run_row, case_rows = rows_from_report(report, report_path=Path("/tmp/report.json"))
    pool = RecordingPool()

    store_over(pool).write(run_row, case_rows)

    assert pool.transactions == 1
    assert len(pool.inserts()) >= 1
    assert len(pool.executemany_calls) == 1
    assert len(pool.executemany_calls[0][1]) == len(case_rows)


# --------------------------------------------------------------------------
# judge_kappa, filled from the run's own calibration record (17f)
# --------------------------------------------------------------------------


def calibrated_report(kappa: float | None) -> RunReport:
    """The same MIXED run, built with a calibration record carrying that kappa."""
    return build_report(
        MIXED,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name=SET_NAME,
        set_sha=SET_SHA,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=GIT_SHA,
        tenant=TENANT,
        target=TARGET,
        judge_prompt_version=SIXTY_FOUR_HEX,
        calibration=RunCalibration(
            judge_prompt_version=SIXTY_FOUR_HEX,
            labels_name="labels.jsonl",
            labels_sha=SIXTY_FOUR_HEX,
            pairs=40,
            kappa=kappa,
            trusted=kappa is not None and kappa >= 0.7,
        ),
    )


def test_the_judge_kappa_column_is_filled_from_the_reports_calibration_record() -> None:
    """Criterion 5, the column 16d declared and left NULL on every run since. Catches
    `rows_from_report` still shipping `judge_kappa=None` while the report carries a number -
    the prediction this row fills - and catches the value read out of the `metrics` mapping
    instead of off the typed field: `metrics` is `dict[str, object]`, so that route needs a
    cast `mypy --strict` cannot check and puts whatever the key holds into a `double
    precision` column. `0.699` is a value nothing else in this file carries, so a column
    filled from another number shows a wrong one."""
    run_row, _ = rows_from_report(calibrated_report(0.699), report_path=Path("/tmp/report.json"))

    assert run_row.judge_kappa == 0.699


def test_a_calibration_whose_own_kappa_is_null_stores_null_rather_than_zero() -> None:
    """17e's undefined kappa arriving on the stored row, and the state the two neighbouring
    cases cannot see: the report DOES carry a calibration, so "no calibration" is not what
    empties the column. Catches `report.calibration.kappa or 0.0` and any other falsy
    coalescing, which would publish "the judge agrees no better than chance" as a
    measurement in a `double precision` column a dashboard reads, and catches an
    `AttributeError` from a record read without checking its kappa."""
    run_row, _ = rows_from_report(calibrated_report(None), report_path=Path("/tmp/report.json"))

    assert run_row.judge_kappa is None
