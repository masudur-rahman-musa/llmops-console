r"""`evals/report.py`: the two files a run writes, and the record `evals/store.py` stores.

Pure: no network, no backing service, no clock of its own. `build_report` is
handed the two timestamps (decision F), so every case here is deterministic and
the whole module is exercised from an in-memory `RunResult`.

The contract these cases are written against, before it exists:

* `REPORT_JSON_NAME`, `REPORT_MD_NAME`, `RUN_ID_LENGTH` and `SET_SHA_PATTERN` -
  the constants a project rule requires a test to pin exactly.
* `CaseReport` and `RunReport`, both frozen and `extra="forbid"`. `RunReport`
  carries `id`, `started_at`, `finished_at`, `set_name`, `set_sha`,
  `set_sha_source`, `run_kind`, `git_sha`, `tenant`, `target`, `passed`,
  `metrics` and `cases`; `CaseReport` carries `case_id`, `category`, `passed`,
  `latency_ms`, `cost_usd`, `provider`, `region`, `assertions`, `residency` and
  `error`. `assertions` and `residency` are `None` exactly when the turn failed.
* `build_report(result, *, started_at, finished_at, set_name, set_sha,
  set_sha_source, run_kind, git_sha, tenant, target) -> RunReport`, deriving
  every count from `RunResult`'s own properties (decision E).
* `report_json(report) -> str`, `report_markdown(report) -> str` and
  `write_reports(out, report) -> None`.
* `report_markdown` renders a `## Failures` section, which is where the per-case
  rows live and where a clean run says so in one line (criterion 3). The heading
  is the one structural thing these cases name, because "one line rather than an
  empty table" cannot be asserted without knowing which part of the document is
  the failures part.
* `metrics` is a mapping (decision G) carrying `cases`, `passed`, `failed`,
  `residency_violations`, `leaked` and `categories`, the last being
  `{category: {"passed": n, "failed": n}}`.

TWO THINGS THIS MODULE NEEDS THAT `16b`'s `RunResult` DOES NOT CARRY, reported
before any of this was written: a `CaseReport` states the case's `category` and
its four measured fields (`latency_ms`, `cost_usd`, `provider`, `region`), and
`CaseRun` has neither - `run_set` discards the `CaseOutcome` it scored. Since
`build_report` takes the `RunResult` and nothing else, these cases are written
against `CaseRun` gaining `category` and `outcome: CaseOutcome | None`, `outcome`
being `None` on exactly the turns where `assertions` is.

The per-category and per-count cases are arranged so no two numbers in a run are
equal (six cases, two passed, four failed, one violation): a builder that fills
`failed` from `passed` or counts a category twice cannot pass by coincidence.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from opsconsole.evals.assertions import (
    ASSERTION_NAMES,
    AssertionResult,
    AssertionStatus,
    CaseAssertions,
)
from opsconsole.evals.cases import EvalCategory
from opsconsole.evals.outcome import CaseOutcome
from opsconsole.evals.report import (
    REPORT_JSON_NAME,
    REPORT_MD_NAME,
    RUN_ID_LENGTH,
    SET_SHA_PATTERN,
    CaseReport,
    RunReport,
    build_report,
    report_json,
    report_markdown,
    write_reports,
)
from opsconsole.evals.residency import ResidencyVerdict
from opsconsole.evals.runner import CaseRun, RunResult
from opsconsole.evals.target import TurnFailureKind

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

REPORT_FIXTURE = FIXTURE_DIR / "report.json"
"""The worked report `docs/eval-report.md` quotes byte for byte.

Four cases, one of each shape a real run produces: passed, a failed assertion, a
residency violation with every assertion passing, and a turn that never answered.
`test_the_worked_report_fixture_parses_field_by_field` is the drift check, and it
is the reason a renamed field is a red test rather than a doc that lies.
"""

FAILURES_HEADING = "## Failures"

STARTED = datetime(2026, 3, 4, 9, 12, 31, 204512, tzinfo=UTC)
FINISHED = datetime(2026, 3, 4, 9, 14, 2, 881044, tzinfo=UTC)

IDENTITY: dict[str, str] = {
    "set_name": "helpdesk-golden-v1.jsonl",
    "set_sha": "e4b2c0d9a71f3856cbb4d20e9f61a7c845d3e0b2",
    "set_sha_source": "commit",
    "run_kind": "ci",
    "git_sha": "b71f0c3a9d5e42188a6c0f37d21be9054c8a7f6d",
    "tenant": "acme",
    "target": "https://helpdesk.example",
}
"""Every identity argument, each a value no other argument could be confused with.

Deliberately no two alike and no two of the same shape where it can be helped:
the two SHAs differ, `tenant` is not the `set_name`, and neither is the `target`,
so a builder that wires one field from another's argument fails the identity case
instead of passing on a coincidence.
"""


def make_outcome(
    *,
    latency_ms: int | None = 120,
    cost_usd: Decimal | None = Decimal("0.0010"),
    provider: str | None = "anthropic_eu",
    region: str | None = "eu",
) -> CaseOutcome:
    """What the target measured on one turn, as `to_outcome` would have built it."""
    return CaseOutcome(
        reply="Done.",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        provider=provider,
        region=region,
    )


def make_run(
    case_id: str,
    *,
    verdict: str = "pass",
    category: EvalCategory = EvalCategory.HELPDESK,
    outcome: CaseOutcome | None = None,
    failure_kind: TurnFailureKind | None = None,
) -> CaseRun:
    """One per-case record of the named shape, as `run_set` would have built it.

    `verdict` is one of `pass`, `fail`, `residency`, `leak`, `broken`; `broken` is
    the turn that never answered, which carries no assertions, no verdict and no
    measurements at all. `failure_kind` only matters on a `broken` verdict.
    """
    if verdict == "broken":
        return CaseRun(
            case_id=case_id,
            category=category,
            assertions=None,
            residency=None,
            outcome=None,
            error=f"turn failed: HTTP 502 for {case_id}",
            failure_kind=failure_kind,
            passed=False,
        )

    results = [AssertionResult(name="intent", status=AssertionStatus.PASSED)]
    if verdict == "fail":
        results.append(
            AssertionResult(
                name="decision",
                status=AssertionStatus.FAILED,
                expected="auto_approved",
                actual="escalated",
            )
        )
    if verdict == "leak":
        results.append(
            AssertionResult(
                name="cross_tenant_leak",
                status=AssertionStatus.FAILED,
                expected="no marker of another tenant",
                actual="['globex']",
            )
        )
    observed = "us" if verdict == "residency" else "eu"
    return CaseRun(
        case_id=case_id,
        category=category,
        assertions=CaseAssertions(case_id=case_id, results=tuple(results)),
        residency=ResidencyVerdict(
            case_id=case_id,
            tenant="acme",
            data_class="internal",
            observed=observed,
            allowed=("eu", "self"),
            violation=verdict == "residency",
        ),
        outcome=outcome if outcome is not None else make_outcome(region=observed),
        error=None,
        passed=verdict == "pass",
    )


def build(result: RunResult) -> RunReport:
    """`build_report` over one run, with the identity table above."""
    return build_report(
        result,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name=IDENTITY["set_name"],
        set_sha=IDENTITY["set_sha"],
        set_sha_source=IDENTITY["set_sha_source"],
        run_kind=IDENTITY["run_kind"],
        git_sha=IDENTITY["git_sha"],
        tenant=IDENTITY["tenant"],
        target=IDENTITY["target"],
    )


MIXED = RunResult(
    runs=(
        make_run("alpha", verdict="pass"),
        make_run("beta", verdict="pass"),
        make_run("gamma", verdict="fail"),
        make_run("delta", verdict="fail"),
        make_run("epsilon", verdict="residency"),
        make_run("zeta", verdict="leak"),
    )
)
"""Six cases, two passed, four failed, one violation, one leak.

No two counts are equal, which is what stops a builder that fills one count from
another - or prints a constant - from passing the counts case by coincidence.
"""

CLEAN = RunResult(runs=(make_run("alpha"), make_run("beta"), make_run("gamma")))

ALL_BROKEN = RunResult(
    runs=(
        make_run("alpha", verdict="broken"),
        make_run("beta", verdict="broken"),
    )
)


def counted(markdown: str, label: str, value: int) -> bool:
    """Does the document state this count against this word?

    A count is looked for BESIDE its label rather than anywhere in the text: the
    identity header carries two SHAs full of digits, so a bare `"6" in markdown`
    would pass on a document that prints no counts at all.
    """
    return re.search(rf"{label}\D{{0,10}}{value}(?!\d)", markdown, re.IGNORECASE) is not None


def failures_section(markdown: str) -> str:
    """Everything the document says after its failures heading."""
    _, _, tail = markdown.partition(FAILURES_HEADING)
    return tail


# --------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------


def test_the_report_file_names_and_the_run_id_length_are_the_published_ones() -> None:
    """Catches a timestamped or renamed report file (decision B): row 18's gate reads a
    known path under `--out`, and a file name that moves turns that gate into a glob that
    picks "the newest", which is a rule nobody wrote down. `RUN_ID_LENGTH` is asserted as
    the integer 32 rather than as "the length of a uuid4 hex", so a builder that switches
    to a uuid1, a short id or a dashed uuid string fails here."""
    assert REPORT_JSON_NAME == "report.json"
    assert REPORT_MD_NAME == "report.md"
    assert RUN_ID_LENGTH == 32


def test_the_set_sha_pattern_is_the_published_one() -> None:
    """Catches a pattern that widens the accepted `--set-sha` - uppercase hex, a length
    other than 7 to 64, or a `.*` a caller could put a path in. The pattern text is
    asserted exactly because the CLI's boundary table below is only as strong as the
    pattern it is measuring, and both halves have to name the same 7 and 64."""
    assert SET_SHA_PATTERN.pattern == r"[0-9a-f]{7,64}"


# --------------------------------------------------------------------------
# build_report: identity, counts, per-case records
# --------------------------------------------------------------------------


def test_build_report_fills_every_identity_field() -> None:
    """Catches a builder that defaults an identity field instead of carrying the caller's
    (a `run_kind` hardcoded to `manual`, a `set_sha_source` always `content`) and one that
    crosses two arguments. No two values in the identity table are alike, so a field wired
    to a neighbour's argument shows up as a wrong value rather than as a coincidence."""
    report = build(MIXED)

    assert report.started_at == STARTED
    assert report.finished_at == FINISHED
    assert report.set_name == IDENTITY["set_name"]
    assert report.set_sha == IDENTITY["set_sha"]
    assert report.set_sha_source == IDENTITY["set_sha_source"]
    assert report.run_kind == IDENTITY["run_kind"]
    assert report.git_sha == IDENTITY["git_sha"]
    assert report.tenant == IDENTITY["tenant"]
    assert report.target == IDENTITY["target"]


def test_the_run_id_is_a_lowercase_hex_of_the_published_length() -> None:
    """Catches an `id` that is a dashed uuid string, an uppercase hex or a truncation:
    `16d` stores it as the `eval_runs` key and a report names it as the run a reader is
    looking at, so its shape is part of the contract and not an implementation detail."""
    report = build(CLEAN)

    assert len(report.id) == RUN_ID_LENGTH
    assert report.id == report.id.lower()
    assert int(report.id, 16) >= 0


def test_each_build_gets_its_own_run_id() -> None:
    """Catches an `id` derived from something stable - a constant, the set SHA, a hash of
    the counts - which would make every run of the same set collide on `16d`'s key and
    make two runs indistinguishable in a report directory kept for history."""
    assert build(CLEAN).id != build(CLEAN).id


def test_each_count_equals_the_matching_run_result_property() -> None:
    """Catches the report and the CLI's summary line disagreeing about one run (decision
    E): a builder that recounts the records instead of reading `RunResult`'s properties
    can drift from the line the gate prints. The six-case run has no two counts alike, so
    a builder filling `failed` from `passed`, or counting a leak as a violation, fails
    here rather than matching by accident."""
    report = build(MIXED)

    assert report.metrics["cases"] == MIXED.cases == 6
    assert report.metrics["passed"] == MIXED.passed_count == 2
    assert report.metrics["failed"] == MIXED.failed_count == 4
    assert report.metrics["residency_violations"] == MIXED.residency_violations == 1
    assert report.metrics["leaked"] is MIXED.leaked
    assert report.metrics["leaked"] is True


def test_a_run_with_a_failed_case_is_not_a_passed_run() -> None:
    """Catches a `passed` flag that only reads the assertion counts, or that is filled by
    the caller: a report saying `passed: true` beside four failed cases is the one field a
    reader trusts without reading the rest, and `16d` stores it as a column."""
    assert build(MIXED).passed is False
    assert build(CLEAN).passed is True


def test_one_case_report_per_case_in_file_order() -> None:
    """Catches a builder that groups, sorts or de-duplicates the per-case rows: the golden
    set's order is the order a reader diffs two runs in, and a report holding fewer rows
    than the run had cases shrinks the denominator row 18's thresholds are measured
    against."""
    report = build(MIXED)

    assert [case.case_id for case in report.cases] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
    ]


def test_a_case_report_carries_the_measured_fields_the_assertions_and_the_verdict() -> None:
    """Catches a per-case row that drops what only the run can observe. Latency, cost,
    provider and region are measured once and never again, the assertion results are what
    row 17b's taxonomy keys off, and the residency verdict is the record of #38 - a report
    missing any of them is a run that has to be paid for twice. Asserted against an
    outcome whose four values are all distinct from the defaults around them."""
    outcome = make_outcome(
        latency_ms=1840, cost_usd=Decimal("0.0142"), provider="anthropic_eu", region="eu"
    )
    run = make_run("alpha", verdict="fail", category=EvalCategory.ACCESS_AUTO, outcome=outcome)

    case = build(RunResult(runs=(run,))).cases[0]

    assert case.case_id == "alpha"
    assert case.category == "access_auto"
    assert case.passed is False
    assert case.latency_ms == 1840
    assert case.cost_usd == Decimal("0.0142")
    assert case.provider == "anthropic_eu"
    assert case.region == "eu"
    assert case.assertions == run.assertions
    assert case.residency == run.residency
    assert case.error is None


def test_a_case_whose_turn_failed_carries_null_assertions_and_its_error() -> None:
    """Catches a broken turn rendered as an EMPTY assertion list, which reads as "ten
    assertions, none failed" to row 17b's taxonomy and to any reader counting failures per
    assertion name. `null` and `[]` are different claims: one says the target never
    answered, the other says it answered and nothing was wrong. The error message is the
    only record of which it was."""
    report = build(ALL_BROKEN)
    document = json.loads(report_json(report))

    assert report.cases[0].assertions is None
    assert report.cases[0].residency is None
    assert report.cases[0].error == "turn failed: HTTP 502 for alpha"
    assert document["cases"][0]["assertions"] is None
    assert document["cases"][0]["error"] == "turn failed: HTTP 502 for alpha"


def test_the_per_category_breakdown_counts_each_category_once_and_omits_none() -> None:
    """Catches a breakdown built from a fixed list of categories (the twelve, most of them
    zero) or from the first case's category alone, and one that loses the category with no
    passes. Four categories, three of them mixed differently, so a builder that counts a
    case under the wrong bucket or double-counts one shows a wrong number here."""
    result = RunResult(
        runs=(
            make_run("a", verdict="pass", category=EvalCategory.HELPDESK),
            make_run("b", verdict="fail", category=EvalCategory.HELPDESK),
            make_run("c", verdict="pass", category=EvalCategory.POLICY_QA),
            make_run("d", verdict="residency", category=EvalCategory.ACCESS_AUTO),
            make_run("e", verdict="broken", category=EvalCategory.CROSS_TENANT),
        )
    )

    categories = build(result).metrics["categories"]

    assert categories == {
        "helpdesk": {"passed": 1, "failed": 1},
        "policy_qa": {"passed": 1, "failed": 0},
        "access_auto": {"passed": 0, "failed": 1},
        "cross_tenant": {"passed": 0, "failed": 1},
    }


def test_the_metrics_key_set_is_pinned_including_tags() -> None:
    """Catches a `metrics` key added silently, and `tags` specifically forgotten
    from the mapping `build_report` assembles: `metrics` is a plain `Mapping` with
    no schema of its own, so nothing but this test stops a consumer reading
    `report.metrics["tags"]` from a `KeyError` on a build that never wired it."""
    report = build(MIXED)

    assert set(report.metrics) == {
        "cases",
        "passed",
        "failed",
        "residency_violations",
        "leaked",
        "categories",
        "tags",
    }


def test_a_case_report_carries_the_tags_the_taxonomy_computed() -> None:
    """Catches `CaseReport.tags` left empty or hand-derived instead of calling
    `taxonomy.tags_for(run)`: the `decision` failure on gamma and delta produces
    exactly `wrong_decision`, the leak on zeta produces exactly `tenant_leak`, and
    the two passed cases carry nothing - so a builder that always emits `()` or a
    fixed placeholder tag fails on every one of these rows but the passed two."""
    report = build(MIXED)
    by_id = {case.case_id: case for case in report.cases}

    assert by_id["alpha"].tags == ()
    assert by_id["beta"].tags == ()
    assert by_id["gamma"].tags == ("wrong_decision",)
    assert by_id["delta"].tags == ("wrong_decision",)
    assert by_id["epsilon"].tags == ("residency_violation",)
    assert by_id["zeta"].tags == ("tenant_leak",)


def test_a_broken_turns_case_report_is_tagged_by_its_failure_kind() -> None:
    """Catches a broken turn reaching the report with an empty tag tuple - the
    defect row 17b's taxonomy exists to prevent, since a turn that never answered
    has no assertions to tag from and only its failure kind can tag it."""
    result = RunResult(
        runs=(
            make_run("alpha", verdict="broken", failure_kind=TurnFailureKind.TIMEOUT),
            make_run("beta", verdict="broken", failure_kind=TurnFailureKind.PROVIDER_ERROR),
        )
    )

    report = build(result)

    assert report.cases[0].tags == ("timeout",)
    assert report.cases[1].tags == ("provider_error",)


def test_the_tags_metric_counts_over_failed_cases_only_and_omits_zero_counts() -> None:
    """Catches a `tags` count taken over every case, which would count a passed
    case's empty tuple as nothing but would also let a stray placeholder tag on a
    passed case go uncaught, and a count that lists every one of the eighteen
    words at zero - the fifteen unused ones would swamp the three that fired and
    make the report's own taxonomy block impossible to scan."""
    report = build(MIXED)

    assert report.metrics["tags"] == {
        "wrong_decision": 2,
        "residency_violation": 1,
        "tenant_leak": 1,
    }


def test_a_clean_runs_tags_metric_is_an_empty_mapping() -> None:
    """Catches a `tags` key omitted entirely on a clean run rather than present
    and empty: the key-set test above pins its PRESENCE, this pins its EMPTY
    value, so a console reading `report.metrics["tags"]` need not special-case a
    run where nothing failed."""
    assert build(CLEAN).metrics["tags"] == {}


# --------------------------------------------------------------------------
# report_json
# --------------------------------------------------------------------------


def test_cost_usd_is_serialised_as_a_json_string() -> None:
    """Catches a `Decimal` serialised through a float, which is the one way a report can
    report a cost the ledger never computed: `0.0000001` becomes `1e-07` and a sum over a
    run stops being exact. The JSON TEXT is asserted, not the parsed value, because
    `json.loads` of a float would compare equal to the `Decimal` and hide the defect."""
    outcome = make_outcome(cost_usd=Decimal("0.0000001"))
    report = build(RunResult(runs=(make_run("alpha", outcome=outcome),)))

    text = report_json(report)
    parsed = json.loads(text)["cases"][0]["cost_usd"]

    assert '"cost_usd": "0.0000001"' in text
    assert isinstance(parsed, str)
    assert parsed == "0.0000001"


def test_a_case_with_no_measurements_serialises_them_as_null_and_not_as_zero() -> None:
    """Catches a report that fills an unmeasured cost or latency with `0`, which a reader
    and `16d`'s columns take as "this run was free and instant" rather than "the target
    never answered". The broken turn is exactly where that substitution happens."""
    document = json.loads(report_json(build(ALL_BROKEN)))

    assert document["cases"][0]["latency_ms"] is None
    assert document["cases"][0]["cost_usd"] is None
    assert document["cases"][0]["provider"] is None
    assert document["cases"][0]["region"] is None


def test_the_json_document_carries_every_identity_field_by_its_published_name() -> None:
    """Catches a serialiser that renames or drops a field on the way out - an alias, an
    exclusion of `None`, a nested `run` object - while `RunReport` itself still looks
    right. `docs/eval-report.md` publishes these key names and `16d` reads them, so the
    document's own keys are the contract rather than the model's attributes."""
    document = json.loads(report_json(build(MIXED)))

    assert set(document) == {
        "id",
        "started_at",
        "finished_at",
        "set_name",
        "set_sha",
        "set_sha_source",
        "run_kind",
        "git_sha",
        "tenant",
        "target",
        "passed",
        "metrics",
        "cases",
    }
    assert set(document["cases"][0]) == {
        "case_id",
        "category",
        "passed",
        "latency_ms",
        "cost_usd",
        "provider",
        "region",
        "assertions",
        "residency",
        "error",
        "tags",
    }


# --------------------------------------------------------------------------
# report_markdown
# --------------------------------------------------------------------------


def test_the_markdown_names_the_run_and_its_counts() -> None:
    """Catches a Markdown document that is a bare failures table: the file is opened by a
    human who has to know WHICH run this was - the set, the tenant, the target and the
    SHA - before the failures mean anything, and criterion 3 names the identity header and
    the counts as part of it."""
    markdown = report_markdown(build(MIXED))

    assert IDENTITY["set_name"] in markdown
    assert IDENTITY["set_sha"] in markdown
    assert IDENTITY["tenant"] in markdown
    assert IDENTITY["target"] in markdown
    assert FAILURES_HEADING in markdown
    assert counted(markdown, "cases", 6)
    assert counted(markdown, "passed", 2)
    assert counted(markdown, "failed", 4)
    assert counted(markdown, "residency", 1)


def test_the_markdown_names_every_failed_case_and_the_assertion_that_failed() -> None:
    """Catches a failures section that names the case and stops there, leaving the reader
    to open the JSON to learn WHAT failed - the one question the document exists to answer.
    The passed cases are asserted absent, so a renderer that simply lists every case (which
    would satisfy any "the failed ids appear" check) fails here instead."""
    markdown = report_markdown(build(MIXED))
    section = failures_section(markdown)

    for case_id in ("gamma", "delta", "epsilon", "zeta"):
        assert case_id in section
    assert "decision" in section
    assert "cross_tenant_leak" in section
    assert "alpha" not in markdown
    assert "beta" not in markdown


def test_the_markdown_failures_table_carries_a_tags_column() -> None:
    """Catches a failures table still rendered with the old three columns: the
    header row is the contract a reader parses by position, so a tags column
    added after `why` instead of before it, or omitted altogether, leaves
    `report.md` disagreeing with the header it prints."""
    markdown = report_markdown(build(MIXED))
    section = failures_section(markdown)

    assert "| case | category | tags | why |" in section
    gamma_line = next(line for line in section.splitlines() if line.startswith("| gamma "))
    assert "wrong_decision" in gamma_line


def test_a_run_with_failures_renders_a_taxonomy_block_of_counts() -> None:
    """Catches #40's aggregation living only in the JSON: a reader of `report.md`
    counting a column by eye over two hundred cases is not an aggregate (decision
    F), so the counts have to be rendered as their own section too."""
    markdown = report_markdown(build(MIXED))

    assert "## Taxonomy" in markdown
    assert counted(markdown, "wrong_decision", 2)
    assert counted(markdown, "residency_violation", 1)
    assert counted(markdown, "tenant_leak", 1)


def test_a_clean_run_renders_no_taxonomy_block() -> None:
    """Catches a `## Taxonomy` heading always rendered, which on a clean run is an
    empty heading with nothing under it - the same defect criterion 3 refuses for
    the failures table, applied to the new block."""
    markdown = report_markdown(build(CLEAN))

    assert "## Taxonomy" not in markdown


def test_a_clean_run_renders_one_line_and_no_empty_failures_table() -> None:
    """Catches the naive renderer that always emits a table: a header row and a separator
    with nothing under them tells a reader "something failed, the rows are missing". One
    line, no table pipes, and no case id anywhere - so a renderer listing the passed cases
    under the failures heading fails too."""
    section = failures_section(report_markdown(build(CLEAN))).strip()

    assert len(section.splitlines()) == 1
    assert "|" not in section
    for case_id in ("alpha", "beta", "gamma"):
        assert case_id not in section


def test_a_run_where_every_turn_failed_still_renders_both_documents() -> None:
    """Catches the renderer that reaches into `case.assertions.results` for every failed
    row: with a half-broken target no case carries assertions at all, so that renderer
    raises at the END of a paid-for run and the whole run is lost - the exact shape
    criterion 8 and risk 3 name. Both documents are produced and each names the error."""
    report = build(ALL_BROKEN)

    markdown = report_markdown(report)
    document = json.loads(report_json(report))

    assert "alpha" in markdown
    assert "beta" in markdown
    assert "HTTP 502" in markdown
    assert [case["assertions"] for case in document["cases"]] == [None, None]


# --------------------------------------------------------------------------
# write_reports
# --------------------------------------------------------------------------


def test_write_reports_writes_exactly_the_two_published_files(tmp_path: Path) -> None:
    """Catches a writer that adds a per-case file, a `latest` symlink or a timestamped
    copy beside the two the gate reads (the false claim `16b`'s docstrings carry: this
    plan writes two files with constant names, never one per case). The contents are
    compared with the two renderers, so a writer that serialises the model a second way of
    its own fails here."""
    report = build(MIXED)

    write_reports(tmp_path, report)

    assert sorted(path.name for path in tmp_path.iterdir()) == [REPORT_JSON_NAME, REPORT_MD_NAME]
    assert (tmp_path / REPORT_JSON_NAME).read_text(encoding="utf-8") == report_json(report)
    assert (tmp_path / REPORT_MD_NAME).read_text(encoding="utf-8") == report_markdown(report)


def test_a_rerun_overwrites_the_two_files_rather_than_appending(tmp_path: Path) -> None:
    """Catches a writer opened in append mode or one that suffixes a second run's files:
    `make eval` names one directory and runs into it repeatedly (decision B), so the
    second run has to leave two files that parse, not two files with two documents in
    them. The second report is the clean run, so its content differs from the first."""
    write_reports(tmp_path, build(MIXED))
    second = build(CLEAN)

    write_reports(tmp_path, second)

    assert sorted(path.name for path in tmp_path.iterdir()) == [REPORT_JSON_NAME, REPORT_MD_NAME]
    document = json.loads((tmp_path / REPORT_JSON_NAME).read_text(encoding="utf-8"))
    assert document["id"] == second.id
    assert len(document["cases"]) == 3


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------


def test_an_unknown_field_is_refused_on_both_levels_of_the_report() -> None:
    """Catches `extra="forbid"` missing from either model, which is how a renamed field
    reaches `16d` and `docs/eval-report.md` unnoticed: the old key loads as an extra, the
    new one takes its default, and every consumer reads a report that validates and means
    something else."""
    payload = json.loads(REPORT_FIXTURE.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError):
        RunReport.model_validate({**payload, "duration_ms": 91_677})
    with pytest.raises(ValidationError):
        CaseReport.model_validate({**payload["cases"][0], "judge_score": 0.9})


def test_the_set_sha_source_and_run_kind_vocabularies_are_the_published_ones() -> None:
    """Catches either field typed as a free string. `set_sha_source` exists so no reader is
    ever told a content hash is a commit (decision A), and a third word - `unknown`, `sha`,
    `local` - means neither; `run_kind` is what row 18 filters CI runs by. Both accepted
    vocabularies are asserted whole, so a value quietly added later shows up here."""
    payload = json.loads(REPORT_FIXTURE.read_text(encoding="utf-8"))

    for source in ("commit", "content"):
        assert (
            RunReport.model_validate({**payload, "set_sha_source": source}).set_sha_source == source
        )
    for kind in ("ci", "nightly", "manual"):
        assert RunReport.model_validate({**payload, "run_kind": kind}).run_kind == kind
    with pytest.raises(ValidationError):
        RunReport.model_validate({**payload, "set_sha_source": "unknown"})
    with pytest.raises(ValidationError):
        RunReport.model_validate({**payload, "run_kind": "local"})


def test_the_worked_report_fixture_parses_field_by_field() -> None:
    """Catches the model drifting from the report `docs/eval-report.md` publishes: a
    renamed key, a `cost_usd` read as a float, an `assertions` block that no longer holds
    all eleven results, a residency verdict that lost its allow list. Every field is read
    back, so a model that silently drops one fails here rather than at the first run
    against a real Helpdesk - the drift check `16a`'s and `16b`'s fixtures both have."""
    report = RunReport.model_validate(json.loads(REPORT_FIXTURE.read_text(encoding="utf-8")))

    assert report.id == "9f2c1d8b4a6e47c3a1b05e7d2f83c4a6"
    assert len(report.id) == RUN_ID_LENGTH
    assert report.started_at == datetime(2026, 3, 4, 9, 12, 31, 204512, tzinfo=UTC)
    assert report.finished_at == datetime(2026, 3, 4, 9, 14, 2, 881044, tzinfo=UTC)
    assert report.set_name == "helpdesk-golden-v1.jsonl"
    assert report.set_sha == "e4b2c0d9a71f3856cbb4d20e9f61a7c845d3e0b2"
    assert SET_SHA_PATTERN.fullmatch(report.set_sha) is not None
    assert report.set_sha_source == "commit"
    assert report.run_kind == "ci"
    assert report.git_sha == "b71f0c3a9d5e42188a6c0f37d21be9054c8a7f6d"
    assert report.tenant == "acme"
    assert report.target == "https://helpdesk.example"
    assert report.passed is False
    assert report.metrics["cases"] == 4
    assert report.metrics["passed"] == 1
    assert report.metrics["failed"] == 3
    assert report.metrics["residency_violations"] == 1
    assert report.metrics["leaked"] is False
    assert report.metrics["categories"] == {
        "access_auto": {"passed": 1, "failed": 0},
        "helpdesk": {"passed": 0, "failed": 1},
        "policy_qa": {"passed": 0, "failed": 1},
        "cross_tenant": {"passed": 0, "failed": 1},
    }

    alpha, beta, gamma, delta = report.cases
    assert [case.case_id for case in report.cases] == ["alpha", "beta", "gamma", "delta"]
    assert [case.category for case in report.cases] == [
        "access_auto",
        "helpdesk",
        "policy_qa",
        "cross_tenant",
    ]
    assert [case.passed for case in report.cases] == [True, False, False, False]

    assert alpha.latency_ms == 1840
    assert alpha.cost_usd == Decimal("0.0142")
    assert alpha.provider == "anthropic_eu"
    assert alpha.region == "eu"
    assert alpha.assertions is not None
    assert tuple(result.name for result in alpha.assertions.results) == ASSERTION_NAMES
    assert alpha.residency is not None
    assert alpha.residency.allowed == ("eu", "self")
    assert alpha.residency.violation is False
    assert alpha.error is None

    assert beta.assertions is not None
    failed = [
        result for result in beta.assertions.results if result.status is AssertionStatus.FAILED
    ]
    assert [result.name for result in failed] == ["decision"]
    assert failed[0].expected == "auto_approved"
    assert failed[0].actual == "escalated"

    assert gamma.assertions is not None
    assert gamma.assertions.passed is True
    assert gamma.residency is not None
    assert gamma.residency.observed == "us"
    assert gamma.residency.violation is True

    assert delta.assertions is None
    assert delta.residency is None
    assert delta.latency_ms is None
    assert delta.cost_usd is None
    assert delta.error == "turn failed: HTTP 502 from https://helpdesk.example/v1/eval/turn"


def test_the_report_models_are_frozen() -> None:
    """Catches a mutable report: `write_reports` renders the same object twice (once as
    JSON, once as Markdown) and `16d` will read it a third time, so a step that normalised
    a region or sorted a case list in place would make the two files disagree about one
    run depending on which was written first."""
    report = build(CLEAN)

    with pytest.raises(ValueError):
        report.passed = True  # type: ignore[misc]
    with pytest.raises(ValueError):
        report.cases[0].case_id = "omega"  # type: ignore[misc]
