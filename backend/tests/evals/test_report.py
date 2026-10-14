r"""`evals/report.py`: the two files a run writes, and the record `storage/eval_store.py` stores.

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
  `{category: {"passed": n, "failed": n}}` - plus, after `17d` and `17c`, the eighteen of
  `metrics.METRIC_NAMES` at BOTH levels, computed by `compute_metrics` over
  `result.runs` and over each category's own runs, and rendered into `report.md`
  as `METRICS_HEADING` and `CATEGORY_METRICS_HEADING` blocks between the
  categories table and the taxonomy. The per-category values are asserted against
  `compute_metrics` over that category's subset AND against a hand-checked value
  the run-level mapping does not carry, so a breakdown filled from the totals
  fails rather than agreeing with itself.

After `17f` a run may also carry `calibration: RunCalibration | None = None` -
`judge_prompt_version`, `labels_name`, `labels_sha`, `pairs`, `kappa` and
`trusted` - which is the ONLY source of the run-level `judge_kappa_vs_human`
(every category's is `null`, decision B) and the only thing that marks the two
judge lines of `## Metrics` with `UNTRUSTED_METRIC_SUFFIX` under an
`UNTRUSTED_JUDGE_NOTE`.

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

from app.evals.assertions import (
    ASSERTION_NAMES,
    AssertionResult,
    AssertionStatus,
    CaseAssertions,
)
from app.evals.calibration import JUDGE_TRUST_MIN
from app.evals.cases import EvalCategory
from app.evals.judge import JudgeVerdict
from app.evals.outcome import CaseOutcome
from app.evals.region_check import ResidencyVerdict
from app.evals.report import (
    CATEGORY_METRICS_HEADING,
    METRICS_HEADING,
    REPORT_JSON_NAME,
    REPORT_MD_NAME,
    RUN_ID_LENGTH,
    SET_SHA_PATTERN,
    TAXONOMY_HEADING,
    UNTRUSTED_JUDGE_NOTE,
    UNTRUSTED_METRIC_SUFFIX,
    CaseReport,
    RunCalibration,
    RunReport,
    build_report,
    report_json,
    report_markdown,
    write_reports,
)
from app.evals.runner import CaseRun, RunResult
from app.evals.scores import METRIC_NAMES, compute_metrics
from app.evals.target import TurnFailureKind

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
    judge: JudgeVerdict | None = None,
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
        judge=judge,
        error=None,
        passed=verdict == "pass",
    )


def build(result: RunResult, *, judge_prompt_version: str | None = None) -> RunReport:
    """`build_report` over one run, with the identity table above.

    `judge_prompt_version` is passed only where a case names it, which is what
    lets the run-level case below tell "absent" from "passed as None" (17c
    criterion 7).
    """
    if judge_prompt_version is not None:
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
            judge_prompt_version=judge_prompt_version,
        )
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


RUN_METRIC_KEYS: tuple[str, ...] = (
    "cases",
    "passed",
    "failed",
    "residency_violations",
    "leaked",
    *METRIC_NAMES,
    "categories",
    "tags",
)
"""Every key `metrics` carries, in order: the five counts, the eighteen, then the two maps.

`metrics` is a plain `Mapping` with no schema of its own, so this tuple is the
only published record of its shape - `16d` stores it as `jsonb` and the eval gate
reads thresholds out of it by name.
"""

CATEGORY_METRIC_KEYS: tuple[str, ...] = ("passed", "failed", *METRIC_NAMES)
"""Every key one category's mapping carries (decision E): its two counts, then the eighteen."""


def section_lines(markdown: str, heading: str) -> list[str]:
    """The non-blank lines under this EXACT heading line, up to the next `## ` heading.

    Matched against whole lines rather than by substring, because `## Metrics` is
    a prefix of `## Metrics by category` and an `in markdown` check would be
    satisfied by the wrong block - the neighbouring-guard trap.
    """
    lines = markdown.splitlines()
    start = lines.index(heading)
    section: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line:
            section.append(line)
    return section


# --------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------


def test_the_report_file_names_and_the_run_id_length_are_the_published_ones() -> None:
    """Catches a timestamped or renamed report file (decision B): the eval gate reads a
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
    than the run had cases shrinks the denominator the eval gate's thresholds are measured
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
    case under the wrong bucket or double-counts one shows a wrong number here.

    After `17d` the same mapping also carries the eighteen (decision E), so the two counts
    are compared field by field rather than as the whole mapping, and the key tuple is
    asserted per category - which is what catches the breakdown gaining the metric set at
    the run level only, or gaining it in a different order from `METRIC_NAMES`."""
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

    assert {
        name: {"passed": counts["passed"], "failed": counts["failed"]}
        for name, counts in categories.items()
    } == {
        "helpdesk": {"passed": 1, "failed": 1},
        "policy_qa": {"passed": 1, "failed": 0},
        "access_auto": {"passed": 0, "failed": 1},
        "cross_tenant": {"passed": 0, "failed": 1},
    }
    for counts in categories.values():
        assert tuple(counts) == CATEGORY_METRIC_KEYS


def test_the_metrics_key_set_is_pinned_including_tags() -> None:
    """Catches a `metrics` key added silently, and `tags` or one of the eighteen
    specifically forgotten from the mapping `build_report` assembles: `metrics` is a
    plain `Mapping` with no schema of its own, so nothing but this test stops a consumer
    reading `report.metrics["tags"]` or `report.metrics["injection_resistance"]` from a
    `KeyError` on a build that never wired it. The ORDER is asserted too, because
    `report.json` is diffed between two runs and `docs/eval-report.md` publishes the
    document as it is written."""
    report = build(MIXED)

    assert tuple(report.metrics) == RUN_METRIC_KEYS
    assert set(report.metrics) == set(RUN_METRIC_KEYS)
    assert len(RUN_METRIC_KEYS) == 25


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


def test_the_run_level_metrics_are_the_metric_set_computed_over_every_run() -> None:
    """Catches `build_report` computing the eighteen a second way of its own, or over a
    subset - the failed cases, or the `CaseReport` rows rather than the `CaseRun`s. Each
    value is compared with `compute_metrics(result.runs)`, and six are also asserted as
    hand-checked constants so the test cannot pass by both sides being wrong together:
    six cases all stating `intent` (`100.0`), two of them failing `decision` (`0.0`, not
    omitted and not `null`), one leak, no broken turn, and 120ms / $0.001 measured six
    times over."""
    report = build(MIXED)
    expected = compute_metrics(MIXED.runs)

    for name in METRIC_NAMES:
        assert report.metrics[name] == expected[name]
    assert report.metrics["intent_accuracy"] == 100.0
    assert report.metrics["decision_accuracy"] == 0.0
    assert report.metrics["cross_tenant_leaks"] == 1
    assert report.metrics["errored"] == 0
    assert report.metrics["p50_latency_ms"] == 120
    assert report.metrics["cost_usd_total"] == Decimal("0.006000")


def test_each_category_carries_the_metric_set_computed_over_its_own_cases() -> None:
    """Catches the breakdown filled from the RUN-level values - the one defect a
    per-category mapping that merely exists cannot show. `policy_qa` states no `decision`
    assertion at all, so its `decision_accuracy` is `null` while the run's is `0.0`, and
    its cost is one turn's rather than three; a builder that copied the totals into every
    category would agree with itself and fail here. The subset comparison then pins that
    the breakdown is the SAME function over a subset (approach), not a second
    implementation."""
    result = RunResult(
        runs=(
            make_run("a", verdict="pass", category=EvalCategory.HELPDESK),
            make_run("b", verdict="fail", category=EvalCategory.HELPDESK),
            make_run("c", verdict="pass", category=EvalCategory.POLICY_QA),
        )
    )
    helpdesk_runs = tuple(run for run in result.runs if run.category is EvalCategory.HELPDESK)

    categories = build(result).metrics["categories"]

    assert categories["helpdesk"]["decision_accuracy"] == 0.0
    assert categories["policy_qa"]["decision_accuracy"] is None
    assert categories["helpdesk"]["cost_usd_total"] == Decimal("0.002000")
    assert categories["policy_qa"]["cost_usd_total"] == Decimal("0.001000")
    for name in METRIC_NAMES:
        assert categories["helpdesk"][name] == compute_metrics(helpdesk_runs)[name]


def test_a_category_of_one_case_stating_nothing_is_present_counted_and_null() -> None:
    """Catches a category dropped from the breakdown because it had nothing to measure,
    and one whose empty accuracies came back as `0.0` (decision B) - criterion 12's
    unhappy path. The `review` case states no assertion at all: it is still a row with its
    own counts, its accuracies are `null`, and its two counts are `0` rather than
    `null`."""
    run = CaseRun(
        case_id="only",
        category=EvalCategory.REVIEW,
        assertions=CaseAssertions(case_id="only", results=()),
        residency=None,
        outcome=make_outcome(),
        error=None,
        passed=True,
    )

    review = build(RunResult(runs=(run,))).metrics["categories"]["review"]

    assert tuple(review) == CATEGORY_METRIC_KEYS
    assert review["passed"] == 1
    assert review["failed"] == 0
    assert review["intent_accuracy"] is None
    assert review["decision_accuracy"] is None
    assert review["injection_resistance"] is None
    assert review["errored"] == 0
    assert review["cross_tenant_leaks"] == 0


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
    document's own keys are the contract rather than the model's attributes. Fifteen keys:
    `judge_prompt_version` is 17c's and `calibration` is 17f's, and both are published in
    every document whatever the run held, so a reader never has to tell an absent key from
    a null one. Twelve case keys: `judge` is 17c's verdict, published as `"judge": null`
    in `docs/eval-report.md` whenever no judge ran, for the same reason."""
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
        "judge_prompt_version",
        "calibration",
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
        "judge",
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


def test_the_two_metric_headings_are_the_published_ones() -> None:
    """Catches either heading renamed: `docs/eval-report.md` publishes the shape of
    `report.md`, and a reader who was told to look under `## Metrics` after a red gate
    finds nothing under `## Numbers` (a project rule applied to the two strings this
    plan introduces)."""
    assert METRICS_HEADING == "## Metrics"
    assert CATEGORY_METRICS_HEADING == "## Metrics by category"


def test_the_markdown_renders_the_metric_set_and_omits_the_metrics_with_no_cases() -> None:
    """Catches the three renderer defects at once, and each one alone flips this list.
    A metric with no case behind it rendered as `- risk_class_accuracy: None` is what
    criterion 11 refuses, so the eight metrics MIXED measured are the whole block; a
    renderer that omitted falsy values instead of `None` ones would drop
    `- decision_accuracy: 0.0` and `- errored: 0`, which are real measurements; and a
    cost printed through `str` rather than `format(value, "f")` would read `0.006` or
    `6E-3` instead of the six places `report.json` carries. Asserted as the exact ordered
    list, so `METRIC_NAMES` order is pinned in the document too."""
    lines = section_lines(report_markdown(build(MIXED)), METRICS_HEADING)

    assert lines == [
        "- intent_accuracy: 100.0",
        "- decision_accuracy: 0.0",
        "- cross_tenant_leaks: 1",
        "- errored: 0",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.006000",
        "- cost_usd_per_turn: 0.001000",
    ]


def test_the_markdown_renders_one_metric_block_per_category_under_its_own_heading() -> None:
    """Catches a per-category block that renders the run-level numbers under every
    category heading, and one that renders the categories without naming them. The two
    categories differ in three visible ways - `policy_qa` has no `decision_accuracy` line
    at all, and the two costs differ - so a block filled from the totals, or one that
    rendered a single merged list, cannot produce this document. A category heading is
    `### <name>` and not `## `, which is what keeps the block one section rather than
    twelve."""
    result = RunResult(
        runs=(
            make_run("a", verdict="pass", category=EvalCategory.HELPDESK),
            make_run("b", verdict="fail", category=EvalCategory.HELPDESK),
            make_run("c", verdict="pass", category=EvalCategory.POLICY_QA),
        )
    )

    lines = section_lines(report_markdown(build(result)), CATEGORY_METRICS_HEADING)

    assert lines == [
        "### helpdesk",
        "- intent_accuracy: 100.0",
        "- decision_accuracy: 0.0",
        "- cross_tenant_leaks: 0",
        "- errored: 0",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.002000",
        "- cost_usd_per_turn: 0.001000",
        "### policy_qa",
        "- intent_accuracy: 100.0",
        "- cross_tenant_leaks: 0",
        "- errored: 0",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.001000",
        "- cost_usd_per_turn: 0.001000",
    ]


def test_the_two_metric_blocks_sit_between_the_categories_table_and_the_taxonomy() -> None:
    """Catches the blocks appended after the failures table, which is the last section by
    design - everything after `## Failures` is the failures, and a metrics block down
    there breaks `failures_section` for every other case in this file as well as the
    document a reader scans top to bottom. Whole-line lookups, because `## Metrics` is a
    prefix of `## Metrics by category`."""
    lines = report_markdown(build(MIXED)).splitlines()

    assert (
        lines.index("## Categories")
        < lines.index(METRICS_HEADING)
        < lines.index(CATEGORY_METRICS_HEADING)
        < lines.index(TAXONOMY_HEADING)
        < lines.index(FAILURES_HEADING)
    )


def test_a_run_where_every_turn_failed_still_renders_the_two_counts_and_no_nulls() -> None:
    """Catches criterion 12's unhappy path in the document: with every turn broken the
    only metrics with anything behind them are the two counts, so a renderer that prints
    `None` for the other fourteen fills the page with nothing, and one that omits a zero
    count hides the fact that both turns errored. Both blocks still render, which is the
    other half of the claim - a run where the target was down is exactly when somebody
    opens this file."""
    markdown = report_markdown(build(ALL_BROKEN))

    assert section_lines(markdown, METRICS_HEADING) == [
        "- cross_tenant_leaks: 0",
        "- errored: 2",
    ]
    assert section_lines(markdown, CATEGORY_METRICS_HEADING) == [
        "### helpdesk",
        "- cross_tenant_leaks: 0",
        "- errored: 2",
    ]


def test_a_run_of_no_cases_renders_no_per_category_metric_block_at_all() -> None:
    """Catches a `## Metrics by category` heading rendered with nothing under it, the
    same defect `_taxonomy_lines` avoids on a clean run: with no case there is no
    category, so the block has no line to render and the heading goes too. The run-level
    block is asserted still present, because its two counts always have a value - which
    is the ONLY run shape where the block-level omission can fire, and the reason it
    needs its own case."""
    markdown = report_markdown(build(RunResult(runs=())))

    assert CATEGORY_METRICS_HEADING not in markdown
    assert METRICS_HEADING in markdown.splitlines()


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
    `local` - means neither; `run_kind` is what a query for CI runs filters by. Both accepted
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
    against a real Helpdesk - the drift check `16a`'s and `16b`'s fixtures both have.

    After `17d` the fixture also carries the metric set at both levels, and it is asserted
    as VALUES and not only as keys: the fixture is what `docs/eval-report.md` publishes
    byte for byte, so a worked example whose numbers do not follow from its own four cases
    is a document that teaches the wrong thing. `cross_tenant` is the case that matters -
    its only case never answered, so it carries `errored: 1` and a null cost while the run
    carries a real one. `cost_usd_total` comes back as the STRING it was written as and
    not as a `Decimal`: `metrics` is a plain `Mapping` with no schema, so nothing coerces
    it on the way back in - which is exactly why the value is written at six places and
    never through a float.

    After `17f` the same drift check covers the eighteenth metric in all FIVE places the
    fixture carries a metric set, and the `calibration` field the document now publishes as
    `null`: the worked run was made with no `--calibration`, so a fixture whose five metric
    sets did not grow the key is a published document teaching the seventeen."""
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
    assert {
        name: {"passed": counts["passed"], "failed": counts["failed"]}
        for name, counts in report.metrics["categories"].items()
    } == {
        "access_auto": {"passed": 1, "failed": 0},
        "helpdesk": {"passed": 0, "failed": 1},
        "policy_qa": {"passed": 0, "failed": 1},
        "cross_tenant": {"passed": 0, "failed": 1},
    }
    assert tuple(report.metrics) == RUN_METRIC_KEYS
    assert report.metrics["decision_accuracy"] == 50.0
    assert report.metrics["clarify_accuracy"] is None
    assert report.metrics["errored"] == 1
    assert report.metrics["cost_usd_total"] == "0.044100"
    for counts in report.metrics["categories"].values():
        assert tuple(counts) == CATEGORY_METRIC_KEYS
    assert report.metrics["categories"]["cross_tenant"]["errored"] == 1
    assert report.metrics["categories"]["cross_tenant"]["cost_usd_total"] is None
    assert report.calibration is None
    assert report.metrics["judge_kappa_vs_human"] is None
    for counts in report.metrics["categories"].values():
        assert counts["judge_kappa_vs_human"] is None

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


# --------------------------------------------------------------------------
# the judge verdict and the prompt version (17c)
# --------------------------------------------------------------------------


JUDGE_VERSION = "3c9a1f0e8d7b6a5c4e3d2b1a0f9e8d7c6b5a4938271605f4e3d2c1b0a9f8e7d6"
"""A sha256-shaped string that is nothing else in this file, so a builder wiring
`judge_prompt_version` from the set SHA or the git SHA shows a wrong value."""


def a_verdict(median: int = 2) -> JudgeVerdict:
    """One verdict whose three samples are all different from each other and from the
    median, so a report that stored the samples in the median's place is visible."""
    return JudgeVerdict(median=median, samples=(5, 1, median), reason="the reply cites nothing")


def test_a_case_report_carries_the_judge_verdict() -> None:
    """Criterion 7 per case. Catches a report that keeps only the median - the three
    samples and the reason are the whole audit trail behind a score that failed a case,
    and a reader who cannot see `(5, 1, 2)` cannot tell a confident judgement from a
    disagreeing one. Catches an unjudged case rendered as an empty object rather than
    `null`, which reads as "judged, and it said nothing". Read back out of the JSON as
    well as off the model, because `details` and the console both parse the document."""
    judged = make_run("alpha", verdict="fail", judge=a_verdict())
    unjudged = make_run("beta", verdict="pass")

    report = build(RunResult(runs=(judged, unjudged)), judge_prompt_version=JUDGE_VERSION)
    document = json.loads(report_json(report))

    assert report.cases[0].judge == a_verdict()
    assert report.cases[1].judge is None
    assert document["cases"][0]["judge"] == {
        "median": 2,
        "samples": [5, 1, 2],
        "reason": "the reply cites nothing",
    }
    assert document["cases"][1]["judge"] is None


def test_the_run_carries_the_judge_prompt_version_and_null_without_a_judge() -> None:
    """Decision H at the run level. Catches a prompt version recorded per CASE, which
    would repeat a 64-character hash on every row of a two-hundred-case report, and one
    that defaults to a placeholder on a run where nothing was judged - `null` is the claim
    that no prompt was used, and a stale hash there would tell a future comparison the
    runs are alike. The published worked report is held to the same shape in the same
    breath: it is what `docs/eval-report.md` quotes byte for byte, so a fixture that never
    grew the three new fields is a document teaching the old one."""
    assert build(MIXED, judge_prompt_version=JUDGE_VERSION).judge_prompt_version == JUDGE_VERSION
    assert build(MIXED).judge_prompt_version is None

    payload = json.loads(REPORT_FIXTURE.read_text(encoding="utf-8"))
    fixture = RunReport.model_validate(payload)
    assert payload["judge_prompt_version"] is None
    assert fixture.judge_prompt_version is None
    assert payload["metrics"]["judge_mean"] is None
    assert [case["judge"] for case in payload["cases"]] == [None, None, None, None]
    assert [case.judge for case in fixture.cases] == [None, None, None, None]
    for counts in payload["metrics"]["categories"].values():
        assert counts["judge_mean"] is None


def test_the_markdown_renders_the_judge_verdict_of_a_failed_case() -> None:
    """Criterion 7's last clause, in the document a human opens after a red gate. Catches
    a judge verdict that lives only in the JSON: a case failed by a score of 2 whose
    Markdown row says only `judge_low` leaves the reader with the tag and no reason, which
    is the one thing a rubric failure has to explain. The passed case's median is asserted
    absent from the failures section, so a renderer printing every verdict there fails
    too, and the prompt version is asserted in the identity block because a report whose
    scores cannot be traced to a prompt is not reproducible."""
    failed = make_run("gamma", verdict="fail", judge=a_verdict())
    passed = make_run("alpha", verdict="pass", judge=a_verdict(median=4))

    markdown = report_markdown(
        build(RunResult(runs=(passed, failed)), judge_prompt_version=JUDGE_VERSION)
    )
    section = failures_section(markdown)
    row = next(line for line in section.splitlines() if line.startswith("| gamma "))

    assert JUDGE_VERSION in markdown
    assert "judge_low" in row
    assert "2" in row
    assert "the reply cites nothing" in row
    assert "alpha" not in section


# --------------------------------------------------------------------------
# the calibration record and the untrusted marking (17f)
# --------------------------------------------------------------------------


LABELS_SHA = "5d4c3b2a1908f7e6d5c4b3a2910f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d"
"""The sha256 of the labels file a calibration was computed from.

A different 64-character string from `JUDGE_VERSION`, from the set SHA and from
the git SHA, so a record that wired `labels_sha` from any of the three shows a
wrong value rather than a plausible one.
"""


def a_calibration(*, kappa: float | None, trusted: bool) -> RunCalibration:
    """One calibration record, with the two fields a case varies stated and the rest fixed."""
    return RunCalibration(
        judge_prompt_version=JUDGE_VERSION,
        labels_name="labels.jsonl",
        labels_sha=LABELS_SHA,
        pairs=40,
        kappa=kappa,
        trusted=trusted,
    )


TRUSTED = a_calibration(kappa=0.83, trusted=True)

UNTRUSTED = a_calibration(kappa=0.699, trusted=False)
"""17e's untrusted table exactly: `0.699` is one thousandth under the floor, which
is the only distance at which a note rendered from a rounded number is visible."""

UNDEFINED = a_calibration(kappa=None, trusted=False)
"""17e's undefined kappa: nothing disagreed, so there is no number to publish and
`trusted` is false anyway."""


JUDGED = RunResult(
    runs=(
        make_run("alpha", verdict="pass", judge=a_verdict(median=2)),
        make_run("beta", verdict="pass", judge=a_verdict(median=5)),
    )
)
"""Two judged cases of one category, medians 2 and 5, so `judge_mean` is `3.5`.

A judged run is what makes the marking cases discriminating: `judge_mean` has a
value to be suffixed, and `3.5` is nothing like either kappa, so a renderer that
printed one number in the other's line cannot pass a line-for-line comparison.
"""


def build_calibrated(result: RunResult, calibration: RunCalibration | None) -> RunReport:
    """`build_report` over one run, with the identity table above and a calibration."""
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
        judge_prompt_version=JUDGE_VERSION,
        calibration=calibration,
    )


def test_the_untrusted_note_and_suffix_are_the_published_strings() -> None:
    """a project rule on the two strings this row introduces, the way the two headings are
    pinned above. Catches a note that stops naming the kappa or the floor - a reader told
    only "untrusted" cannot tell 0.699 from 0.1 and has nothing to recalibrate against -
    and catches a suffix that is not the word the eval gate's operator was told to look for.
    Asserted as the format TEMPLATE, so the two placeholders are pinned too: a note
    hard-coding `0.7` in its text would render the right line today and lie the day
    `JUDGE_TRUST_MIN` moves."""
    assert UNTRUSTED_JUDGE_NOTE == (
        "judge metrics below are UNTRUSTED: kappa {kappa} against human labels is under "
        "the {floor} floor"
    )
    assert UNTRUSTED_METRIC_SUFFIX == " (untrusted)"


def test_a_calibrated_run_carries_the_kappa_at_the_run_level_and_null_in_every_category() -> None:
    """Criterion 3 and decision B in one row, and the only arrangement that separates them:
    the run carries `0.83` and every category carries `null` for the SAME key. Catches
    `build_report` passing its kappa down into each category's `compute_metrics`, which
    would repeat one agreement measurement under three headings and invite a reader to diff
    two categories that cannot differ; catches the key dropped from the category mappings
    altogether, which a run-level assertion cannot see; and catches a run-level value
    computed from the cases rather than taken from the record, since no `CaseRun` here
    carries a human label. Both key sets are asserted by equality, because `metrics` has no
    schema of its own."""
    result = RunResult(
        runs=(
            make_run("a", verdict="pass", category=EvalCategory.HELPDESK),
            make_run("b", verdict="fail", category=EvalCategory.HELPDESK),
            make_run("c", verdict="pass", category=EvalCategory.POLICY_QA),
        )
    )

    report = build_calibrated(result, TRUSTED)

    assert tuple(report.metrics) == RUN_METRIC_KEYS
    assert report.metrics["judge_kappa_vs_human"] == 0.83
    categories = report.metrics["categories"]
    assert set(categories) == {"helpdesk", "policy_qa"}
    for counts in categories.values():
        assert tuple(counts) == CATEGORY_METRIC_KEYS
        assert counts["judge_kappa_vs_human"] is None


def test_a_run_with_no_calibration_carries_null_in_the_field_and_in_the_metric() -> None:
    """The default half of criterion 4, which is every run made without `--calibration` and
    every report written before this row existed. Catches a `calibration` field that is
    required rather than defaulted - which would make the worked fixture and every stored
    report unparseable under `extra="forbid"` - and catches an empty `RunCalibration`
    manufactured for such a run, which reads as "calibrated, against nothing" and would
    hand the eval gate a `trusted` flag nobody measured. The run IS judged, so `judge_mean` has a
    value and a kappa cannot arrive from it."""
    report = build(JUDGED, judge_prompt_version=JUDGE_VERSION)
    document = json.loads(report_json(report))

    assert report.calibration is None
    assert report.metrics["judge_kappa_vs_human"] is None
    assert document["calibration"] is None
    assert document["metrics"]["judge_mean"] == 3.5


def test_the_json_document_publishes_the_calibration_record_field_by_field() -> None:
    """Criterion 4's published key set, asserted by EQUALITY on both levels so a seventh
    calibration field added quietly and one of the six dropped are both red - `report.json`
    is what the eval gate, the console and an auditor six months later read, and a record
    missing `labels_name` or `pairs` is a trust claim nobody can retrace to a file. Catches
    the record flattened into two top-level fields (decision F), which would change the
    document's key set for every run that has no calibration at all. Each value differs
    from every other value in the document, so a serialiser wiring `labels_sha` from the
    set SHA or `judge_prompt_version` from the git SHA shows a wrong string."""
    document = json.loads(report_json(build_calibrated(JUDGED, UNTRUSTED)))

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
        "judge_prompt_version",
        "calibration",
        "cases",
    }
    assert document["calibration"] == {
        "judge_prompt_version": JUDGE_VERSION,
        "labels_name": "labels.jsonl",
        "labels_sha": LABELS_SHA,
        "pairs": 40,
        "kappa": 0.699,
        "trusted": False,
    }
    assert document["metrics"]["judge_kappa_vs_human"] == 0.699


def test_a_trusted_calibration_marks_nothing_in_the_metrics_block() -> None:
    """The first half of criterion 6, and the row that stops the marking from being
    unconditional: a judge that MET the published bar has numbers a reader may quote as
    they stand, and a note or a suffix on them would train every reader to ignore both.
    Asserted line for line, so a note rendered with an empty kappa, a suffix appended to
    every metric, or a blank line left where the note used to be all fail here. `0.83` is
    above `JUDGE_TRUST_MIN` and `trusted` is true, so only the flag can decide this
    block."""
    lines = section_lines(report_markdown(build_calibrated(JUDGED, TRUSTED)), METRICS_HEADING)

    assert lines == [
        "- intent_accuracy: 100.0",
        "- cross_tenant_leaks: 0",
        "- errored: 0",
        "- judge_mean: 3.5",
        "- judge_kappa_vs_human: 0.83",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.002000",
        "- cost_usd_per_turn: 0.001000",
    ]


def test_an_untrusted_calibration_notes_the_kappa_and_suffixes_the_two_judge_lines() -> None:
    """The second half of criterion 6, decision H's whole shape in one block. Catches a
    kappa published as a bare number with nothing beside it, which is exactly how a low
    agreement gets quoted in a report six months later; catches the note rendered without
    the kappa or without the floor; catches the suffix on the wrong lines - every
    non-judge metric here is a real measurement the judge had no part in, so a renderer
    that marked the whole block fails, and one that marked only `judge_mean` fails on the
    kappa line. The per-category block is asserted in the same breath because it is where
    the marking must NOT appear: its `judge_kappa_vs_human` is `null` and therefore has no
    line, and its `judge_mean` is not the run-level number a reader quotes."""
    markdown = report_markdown(build_calibrated(JUDGED, UNTRUSTED))

    assert section_lines(markdown, METRICS_HEADING) == [
        UNTRUSTED_JUDGE_NOTE.format(kappa=0.699, floor=JUDGE_TRUST_MIN),
        "- intent_accuracy: 100.0",
        "- cross_tenant_leaks: 0",
        "- errored: 0",
        "- judge_mean: 3.5 (untrusted)",
        "- judge_kappa_vs_human: 0.699 (untrusted)",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.002000",
        "- cost_usd_per_turn: 0.001000",
    ]
    assert section_lines(markdown, CATEGORY_METRICS_HEADING) == [
        "### helpdesk",
        "- intent_accuracy: 100.0",
        "- cross_tenant_leaks: 0",
        "- errored: 0",
        "- judge_mean: 3.5",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.002000",
        "- cost_usd_per_turn: 0.001000",
    ]


def test_an_untrusted_calibration_with_no_kappa_at_all_still_notes_it_as_null() -> None:
    """17e's undefined kappa reaching a run: nothing disagreed, so there is no number to
    publish and `trusted` is false regardless. Catches a note rendered only when there is a
    kappa - the run whose judge agreement could not be measured at all is the one a reader
    most needs warned about - and catches Python's `None` printed into a document
    `docs/eval-report.md` publishes as JSON-shaped, where the word is `null`. The
    `judge_kappa_vs_human` line is absent because its value is `None`, which is the
    renderer's existing rule and is why `judge_mean` is the only suffixed line here."""
    lines = section_lines(report_markdown(build_calibrated(JUDGED, UNDEFINED)), METRICS_HEADING)

    assert lines == [
        UNTRUSTED_JUDGE_NOTE.format(kappa="null", floor=JUDGE_TRUST_MIN),
        "- intent_accuracy: 100.0",
        "- cross_tenant_leaks: 0",
        "- errored: 0",
        "- judge_mean: 3.5 (untrusted)",
        "- p50_latency_ms: 120",
        "- p95_latency_ms: 120",
        "- cost_usd_total: 0.002000",
        "- cost_usd_per_turn: 0.001000",
    ]
