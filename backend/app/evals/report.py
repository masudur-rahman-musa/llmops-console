r"""The two files one run writes: `report.json` and `report.md` (#39, part).

Pure and clockless. `build_report` is handed the two timestamps rather than
reading a clock (decision F), so `runner.run_set` stays the loop `16b` shipped
and every case in `tests/evals/test_report.py` is deterministic.

Three things the module holds that no single field states:

* every count is DERIVED from `RunResult`'s own properties and never accepted
  from a caller (decision E), so the report and the one line `ops-eval` prints
  cannot disagree about one run
  (`test_each_count_equals_the_matching_run_result_property`);
* `assertions` is `None` exactly where `CaseRun.assertions` is - a turn that
  never answered, which is not the same claim as an empty list of assertions
  (`test_a_case_whose_turn_failed_carries_null_assertions_and_its_error`);
* `cost_usd` is serialised as a JSON STRING and never through a float, so a cost
  round-trips to the number the ledger computed
  (`test_cost_usd_is_serialised_as_a_json_string`);
* every failed case carries at least one tag from `evals/taxonomy.py`'s closed
  vocabulary (#40), counted in `metrics["tags"]` and rendered in both the
  failures table and the taxonomy block
  (`test_a_case_report_carries_the_tags_the_taxonomy_computed`).
* `metrics` carries the SAME eighteen keys on every run, whatever the run held
  (#40), so a gate reads a threshold against a key that is always there and two
  reports diff (`test_the_metrics_key_set_is_pinned_including_tags`); a metric
  with no case behind it is `null` there and is left out of `report.md`
  entirely (`test_the_markdown_renders_the_metric_set_and_omits_the_metrics_with_no_cases`).
* a run handed a `calibration.json` publishes what it was told about its judge -
  the `RunCalibration` record, the kappa as the eighteenth metric, and, when that
  calibration is not trusted, one `UNTRUSTED_JUDGE_NOTE` and a suffix on the two
  judge lines of `report.md` (17f decisions F and H,
  `test_the_json_document_publishes_the_calibration_record_field_by_field`,
  `test_an_untrusted_calibration_notes_the_kappa_and_suffixes_the_two_judge_lines`).
  Nothing here applies the trust floor: 17e decided it and this module renders it.

The two file names are constants and nothing a caller supplies shapes a path
(decision B): `make eval` names one directory, the eval gate reads a known path
inside it, and a rerun overwrites
(`test_a_rerun_overwrites_the_two_files_rather_than_appending`).

`storage/eval_store.py` reads this record: migration `003` and the `eval_runs` and
`eval_cases` rows are built from the fields published here and in
`docs/eval-report.md`, and from no other source.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.evals.assertions import AssertionResult, AssertionStatus, CaseAssertions
from app.evals.calibration import JUDGE_TRUST_MIN
from app.evals.judge import JUDGE_LOW_SCORE, JUDGE_MAX_SCORE, JudgeVerdict
from app.evals.region_check import ResidencyVerdict
from app.evals.runner import CaseRun, RunResult
from app.evals.scores import METRIC_NAMES, compute_metrics
from app.evals.taxonomy import TAG_ORDER, tags_for

REPORT_JSON_NAME: Final[str] = "report.json"
"""The machine-readable report, the file the eval gate reads."""

REPORT_MD_NAME: Final[str] = "report.md"
"""The human-readable report, the file somebody opens after a red gate.

Both names are fixed rather than timestamped (decision B) and pinned exactly by
`test_the_report_file_names_and_the_run_id_length_are_the_published_ones`."""

RUN_ID_LENGTH: Final[int] = 32
"""Characters in a run id: a `uuid4` hex, which `16d` stores as the `eval_runs` key.

Enforced on the field itself, so a shortened or dashed id is a validation error
rather than a key two runs could share."""

SET_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{7,64}")
r"""What `--set-sha` may be: 7 to 64 lowercase hex characters, inclusive.

Used with `re.fullmatch` and never `^...$` with `.match`, because in Python `$`
also matches before a trailing newline. The value is written
into a report and into `16d`'s `eval_runs` row, so it is checked before the seed
is reset (`test_a_malformed_set_sha_is_refused_before_the_target_is_touched`)."""

REPORT_INDENT: Final[int] = 2
"""How `report.json` is indented. Fixed so the worked report in
`docs/eval-report.md` stays byte-identical to `tests/evals/fixtures/report.json`."""

SetShaSource = Literal["commit", "content"]
"""Where `set_sha` came from: the caller's commit, or the set file's own bytes.

Recorded beside the SHA (decision A) so no reader is ever told a content hash is
a commit (`test_the_set_sha_source_and_run_kind_vocabularies_are_the_published_ones`)."""

RunKind = Literal["ci", "nightly", "manual"]
"""Why the run happened. A gate run and a nightly run are told apart by it."""

FAILURES_HEADING: Final[str] = "## Failures"
"""Where the per-case rows live in `report.md`, and the last section of it.

A clean run says so in ONE line under this heading rather than printing a table
with no rows, which reads as "something failed and the rows are missing"
(`test_a_clean_run_renders_one_line_and_no_empty_failures_table`)."""

CLEAN_RUN_LINE: Final[str] = "Every case passed and every turn stayed in region."
"""The one line a clean run renders in place of the failures table."""

TAXONOMY_HEADING: Final[str] = "## Taxonomy"
"""Where the per-tag counts live in `report.md`, immediately before the failures.

Rendered only when something failed, for the reason the failures table is: a
heading with nothing under it reads as counts that went missing
(`test_a_clean_run_renders_no_taxonomy_block`). It sits BEFORE the failures
because everything after that heading is the failures
(`test_a_run_with_failures_renders_a_taxonomy_block_of_counts`)."""

METRICS_HEADING: Final[str] = "## Metrics"
"""Where the fixed metric set lives in `report.md` (#40), one line per metric.

`docs/eval-report.md` publishes this heading, so a reader sent to look under it
after a red gate finds the numbers there
(`test_the_two_metric_headings_are_the_published_ones`)."""

CATEGORY_METRICS_HEADING: Final[str] = "## Metrics by category"
"""Where the same set, per category, lives - the breakdown decision E stores in
`metrics["categories"]`. It is a prefix-sharing neighbour of `METRICS_HEADING`,
so both are matched as whole lines and never by substring
(`test_the_two_metric_blocks_sit_between_the_categories_table_and_the_taxonomy`)."""


JUDGE_METRIC_NAMES: Final[tuple[str, ...]] = ("judge_mean", "judge_kappa_vs_human")
"""The two metrics a judge produced, and the only two an untrusted calibration marks.

Every other name in `METRIC_NAMES` is measured from the turn's own record and had
no judge in it, so marking the whole block would train a reader to ignore all
eighteen (17f decision H,
`test_an_untrusted_calibration_notes_the_kappa_and_suffixes_the_two_judge_lines`)."""

UNTRUSTED_JUDGE_NOTE: Final[str] = (
    "judge metrics below are UNTRUSTED: kappa {kappa} against human labels is under "
    "the {floor} floor"
)
"""The ONE line `## Metrics` carries when the run's calibration is not trusted.

A format template rather than a finished sentence, so both numbers are rendered
from the values that decided them: a note hard-coding `0.7` would read correctly
today and lie the day `JUDGE_TRUST_MIN` moves, and a note naming no kappa leaves a
reader unable to tell `0.699` from `0.1`
(`test_the_untrusted_note_and_suffix_are_the_published_strings`). `{kappa}` is the
word `null` when the artifact carried none - `report.md` sits beside a JSON
document and Python's `None` is not what that document says
(`test_an_untrusted_calibration_with_no_kappa_at_all_still_notes_it_as_null`)."""

UNTRUSTED_METRIC_SUFFIX: Final[str] = " (untrusted)"
"""What each judge metric line ends with under an untrusted calibration.

Beside the number a reader is about to quote, which is what "greyed out" means in
a document with no colour (17f decision H,
`test_the_untrusted_note_and_suffix_are_the_published_strings`)."""


def _new_run_id() -> str:
    """A fresh `uuid4` hex, one per built report.

    Nothing about the run derives it: two runs of the same set on the same
    commit are two rows in `16d`'s table and two directories a reader keeps
    (`test_each_build_gets_its_own_run_id`).
    """
    return uuid.uuid4().hex


class CaseReport(BaseModel):
    """One case's row: what it was, what it cost, and what was wrong with it.

    `latency_ms`, `cost_usd`, `provider` and `region` are measured once by the
    turn and never again, so they are carried rather than recomputed; they are
    `None`, never `0`, on a case whose turn failed
    (`test_a_case_with_no_measurements_serialises_them_as_null_and_not_as_zero`).

    `tags` is `17b`'s failure taxonomy (#40), computed by `taxonomy.tags_for` from
    this case's own record and empty on a case that passed
    (`test_a_case_report_carries_the_tags_the_taxonomy_computed`). It is derived
    here rather than in the runner because the tags are a READING of the record
    the runner measured, and `16d` stores it as the `eval_cases.tags` column.

    `judge` is `17c`'s verdict, carried whole rather than reduced to its median:
    the three samples and the median sample's reason are the audit trail behind a
    case an LLM failed, and a reader who cannot see them cannot tell a confident
    judgement from three that disagreed
    (`test_a_case_report_carries_the_judge_verdict`). It is `null` and never `{}`
    on a case nothing judged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    passed: bool
    tags: tuple[str, ...]
    latency_ms: int | None
    cost_usd: Decimal | None
    provider: str | None
    region: str | None
    assertions: CaseAssertions | None
    residency: ResidencyVerdict | None
    judge: JudgeVerdict | None
    error: str | None

    @field_serializer("cost_usd")
    def _serialise_cost(self, value: Decimal | None) -> str | None:
        """Write the cost as its plain decimal text, never through a float.

        `format(value, "f")` and not `str(value)`: Decimal's own text switches to
        scientific notation below `1e-6`, so a real per-token cost would reach a
        report as `1E-7` and stop matching the ledger's own column
        (`test_cost_usd_is_serialised_as_a_json_string`).
        """
        return None if value is None else format(value, "f")


class RunCalibration(BaseModel):
    """What a run was told about its judge: which labels, how well they agreed, trust.

    17e's `CalibrationReport` narrowed to the six fields a RUN publishes. The rows,
    the drop count and the clock stay in `calibration.json`, which is the audit
    trail; what a report owes its reader is the number, the flag, and enough to
    find the file the two came from - `labels_name` and `labels_sha` are how a
    stored run is retraced six months later, and `pairs` is how a reader tells a
    kappa over forty labels from one over four hundred
    (`test_the_json_document_publishes_the_calibration_record_field_by_field`).

    `judge_prompt_version` is repeated here rather than read off `RunReport`: the
    run's own field is the prompt this run SCORED with, and this one is the prompt
    the calibration was COMPUTED under. `cli/evals.py` refuses a run where the two
    differ (17f decision D), and the two fields are what makes that refusal
    checkable in a stored document rather than only in the process that made it.

    `kappa` is `None` exactly where 17e's is - chance agreement was total - and
    `trusted` is then `False` whatever a comparison would have done with it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_prompt_version: str
    labels_name: str
    labels_sha: str
    pairs: int
    kappa: float | None
    trusted: bool


class RunReport(BaseModel):
    """One run: its identity, its counts, and one `CaseReport` per case.

    The identity fields are the columns `16d`'s `eval_runs` row needs, and their
    names are the published contract rather than an implementation detail -
    `docs/eval-report.md` publishes them and
    `test_the_json_document_carries_every_identity_field_by_its_published_name`
    pins the document's own key set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_new_run_id, min_length=RUN_ID_LENGTH, max_length=RUN_ID_LENGTH)
    started_at: datetime
    finished_at: datetime
    set_name: str
    set_sha: str
    set_sha_source: SetShaSource
    run_kind: RunKind
    git_sha: str | None
    tenant: str
    target: str
    passed: bool
    metrics: Mapping[str, Any]
    judge_prompt_version: str | None
    """The sha256 of the prompt file the judge scored this run with, or `null`
    when nothing was judged (`17c` decision H). At the RUN level and not per case,
    because one process reads one prompt file and a hash repeated on two hundred
    rows is the same fact two hundred times
    (`test_the_run_carries_the_judge_prompt_version_and_null_without_a_judge`)."""
    calibration: RunCalibration | None = None
    """The calibration this run was handed, or `null` when it was given none (17f).

    Defaulted rather than required, which is what keeps every report written
    before this row - the worked fixture included - parseable under
    `extra="forbid"`, and what stops an uncalibrated run manufacturing an empty
    record that reads as "calibrated, against nothing"
    (`test_a_run_with_no_calibration_carries_null_in_the_field_and_in_the_metric`).
    `storage/eval_store.py` reads `kappa` off this field and never out of the `metrics`
    mapping, so `mypy --strict` still sees the type going into a
    `double precision` column."""
    cases: tuple[CaseReport, ...]


def _case_report(run: CaseRun) -> CaseReport:
    """One per-case record turned into its report row.

    The four measured fields come from the turn's own `CaseOutcome`, which is
    `None` exactly where the assertions are, so a broken turn reports nothing
    measured rather than reporting zeros.
    """
    outcome = run.outcome
    return CaseReport(
        case_id=run.case_id,
        category=str(run.category),
        passed=run.passed,
        tags=tags_for(run),
        latency_ms=None if outcome is None else outcome.latency_ms,
        cost_usd=None if outcome is None else outcome.cost_usd,
        provider=None if outcome is None else outcome.provider,
        region=None if outcome is None else outcome.region,
        assertions=run.assertions,
        residency=run.residency,
        judge=run.judge,
        error=run.error,
    )


def _categories(result: RunResult) -> dict[str, dict[str, object]]:
    """The two counts and the whole metric set per category this run actually used.

    Built from the run rather than from `EvalCategory`'s twelve, so a report does
    not carry ten empty buckets, and a category with no passes is still present
    (`test_the_per_category_breakdown_counts_each_category_once_and_omits_none`).

    The metrics are `compute_metrics` over that category's own records and never
    the run-level values narrowed or copied (17d decision E), which is what stops
    the breakdown drifting from the totals: one function, two subsets
    (`test_each_category_carries_the_metric_set_computed_over_its_own_cases`).
    """
    grouped: dict[str, list[CaseRun]] = {}
    counts: dict[str, dict[str, int]] = {}
    for run in result.runs:
        name = str(run.category)
        grouped.setdefault(name, []).append(run)
        tally = counts.setdefault(name, {"passed": 0, "failed": 0})
        tally["passed" if run.passed else "failed"] += 1
    return {name: {**counts[name], **compute_metrics(runs)} for name, runs in grouped.items()}


def _tag_counts(cases: list[CaseReport]) -> dict[str, int]:
    """How many FAILED cases carried each tag, in `TAG_ORDER` and omitting zeros.

    Failed cases only, because a passed case carries no tags at all and a count
    over every case would say the same thing more slowly. Zeros are omitted so a
    reader is not scanning fifteen unused words to find the three that fired
    (`test_the_tags_metric_counts_over_failed_cases_only_and_omits_zero_counts`),
    and a clean run leaves the key present and empty rather than absent
    (`test_a_clean_runs_tags_metric_is_an_empty_mapping`).
    """
    counts: dict[str, int] = {}
    for case in cases:
        if case.passed:
            continue
        for tag in case.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return {tag.value: counts[tag.value] for tag in TAG_ORDER if tag.value in counts}


def build_report(
    result: RunResult,
    *,
    started_at: datetime,
    finished_at: datetime,
    set_name: str,
    set_sha: str,
    set_sha_source: str,
    run_kind: str,
    git_sha: str | None,
    tenant: str,
    target: str,
    judge_prompt_version: str | None = None,
    calibration: RunCalibration | None = None,
) -> RunReport:
    """Turn one run and its identity into the record both files render.

    Every count is read off `RunResult`'s properties (decision E). `metrics` is a
    mapping rather than a model, which is how `17b`'s `tags` and `17d`'s fixed
    metric set were both added without changing this schema or `16d`'s column.

    The metric set is `compute_metrics` over the run's own records, spread into
    the mapping as eighteen flat keys, and computed a second time per category
    inside `_categories` - the same function over a subset, so the breakdown and
    the totals cannot drift
    (`test_the_run_level_metrics_are_the_metric_set_computed_over_every_run`).

    The eighteenth of those keys is the exception, and the only one that comes
    from a caller: `judge_kappa_vs_human` is the supplied calibration's own kappa
    and is passed at the RUN level only, so every category reports `null` for it
    by construction rather than through a second code path (17f decision B,
    `test_a_calibrated_run_carries_the_kappa_at_the_run_level_and_null_in_every_category`).

    The per-case rows are built BEFORE `metrics`, because `metrics["tags"]` is
    counted from the tags those rows carry: one reading of the taxonomy per case,
    so the counts and the rows cannot disagree about the same failure
    (`test_the_tags_metric_counts_over_failed_cases_only_and_omits_zero_counts`).

    `set_sha_source` and `run_kind` arrive as plain strings and are validated by
    the model's own vocabularies here, which is why the report is built through
    `model_validate`: a caller that has not been through the CLI's `choices` and
    its `SET_SHA_PATTERN` still cannot put a fourth word into a stored row
    (`test_the_set_sha_source_and_run_kind_vocabularies_are_the_published_ones`).
    """
    cases = [_case_report(run) for run in result.runs]
    payload: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "set_name": set_name,
        "set_sha": set_sha,
        "set_sha_source": set_sha_source,
        "run_kind": run_kind,
        "git_sha": git_sha,
        "tenant": tenant,
        "target": target,
        "passed": result.failed_count == 0,
        "judge_prompt_version": judge_prompt_version,
        "calibration": calibration,
        "metrics": {
            "cases": result.cases,
            "passed": result.passed_count,
            "failed": result.failed_count,
            "residency_violations": result.residency_violations,
            "leaked": result.leaked,
            **compute_metrics(
                result.runs, judge_kappa=None if calibration is None else calibration.kappa
            ),
            "categories": _categories(result),
            "tags": _tag_counts(cases),
        },
        "cases": cases,
    }
    return RunReport.model_validate(payload)


def report_json(report: RunReport) -> str:
    """`report.json`'s text: the whole record, indented, newline-terminated."""
    return report.model_dump_json(indent=REPORT_INDENT) + "\n"


def _failed_results(assertions: CaseAssertions) -> list[AssertionResult]:
    """The assertions that failed, in the fixed order `assert_case` reports them."""
    return [result for result in assertions.results if result.status is AssertionStatus.FAILED]


def _cell(text: str) -> str:
    """One table cell, with any pipe in it escaped so the row still renders."""
    return text.replace("|", r"\|")


def _reason(case: CaseReport) -> str:
    """Why this case failed, in the words a reader needs before opening the JSON.

    A broken turn reports its error; a case that answered reports every assertion
    that failed and, when the turn left its region, the residency verdict too - a
    violation with every assertion passing has no failed assertion to name
    (`test_the_markdown_names_every_failed_case_and_the_assertion_that_failed`).

    A low judge median is named here too, with the median sample's own reason,
    because a case an LLM failed has nothing else to explain it - a row reading
    `judge_low` and no more leaves the reader with the tag alone
    (`test_the_markdown_renders_the_judge_verdict_of_a_failed_case`). A verdict at
    or above `JUDGE_LOW_SCORE` is not a reason and is not printed: it failed for
    something else, and the score that did not fail it is in `report.json`.
    """
    if case.assertions is None:
        return case.error or "the turn never answered"

    reasons = []
    for result in _failed_results(case.assertions):
        if result.expected is None and result.actual is None:
            reasons.append(result.name)
        else:
            reasons.append(f"{result.name} (expected {result.expected}, actual {result.actual})")
    if case.residency is not None and case.residency.violation:
        allowed = ", ".join(case.residency.allowed)
        reasons.append(f"residency: served in {case.residency.observed}, allowed {allowed}")
    if case.judge is not None and case.judge.median < JUDGE_LOW_SCORE:
        reasons.append(
            f"judge: median {case.judge.median} of {JUDGE_MAX_SCORE}, {case.judge.reason}"
        )
    return "; ".join(reasons) if reasons else "failed with no stated reason"


def _identity_lines(report: RunReport) -> list[str]:
    """The header a reader needs before any count means anything: which run was this.

    The judge prompt's sha256 is one of them (`17c` criterion 7): a score nobody
    can trace to the wording that produced it is not reproducible, and a run that
    judged nothing says `none` rather than carrying a stale hash
    (`test_the_run_carries_the_judge_prompt_version_and_null_without_a_judge`,
    `test_the_markdown_renders_the_judge_verdict_of_a_failed_case`).
    """
    return [
        f"# Eval run {report.id}",
        "",
        f"- set: {report.set_name}",
        f"- set sha: {report.set_sha} ({report.set_sha_source})",
        f"- run kind: {report.run_kind}",
        f"- git sha: {report.git_sha if report.git_sha is not None else 'unknown'}",
        f"- tenant: {report.tenant}",
        f"- target: {report.target}",
        f"- judge prompt: {report.judge_prompt_version or 'none'}",
        f"- started: {report.started_at.isoformat()}",
        f"- finished: {report.finished_at.isoformat()}",
        f"- result: {'passed' if report.passed else 'FAILED'}",
    ]


def _count_lines(report: RunReport) -> list[str]:
    """The counts, each beside its own word, so a reader is not counting rows."""
    metrics = report.metrics
    return [
        "## Counts",
        "",
        f"- cases: {metrics['cases']}",
        f"- passed: {metrics['passed']}",
        f"- failed: {metrics['failed']}",
        f"- residency: {metrics['residency_violations']} violations",
        f"- leaked: {'yes' if metrics['leaked'] else 'no'}",
    ]


def _category_lines(report: RunReport) -> list[str]:
    """The per-category breakdown, one row per category the run used."""
    lines = ["## Categories", "", "| category | passed | failed |", "| --- | --- | --- |"]
    categories: Mapping[str, Mapping[str, object]] = report.metrics["categories"]
    lines += [
        f"| {name} | {counts['passed']} | {counts['failed']} |"
        for name, counts in categories.items()
    ]
    return lines


def _metric_value(value: object) -> str:
    """One metric as `report.md` prints it: a cost through `format`, anything else plainly.

    `format(value, "f")` and not `str(value)` for a `Decimal`, for the reason
    `CaseReport._serialise_cost` does the same: Decimal's own text goes
    scientific below six places, so a real per-turn cost would print as `1E-7`
    beside a `report.json` that carries `0.000000`
    (`test_the_markdown_renders_the_metric_set_and_omits_the_metrics_with_no_cases`).
    """
    return format(value, "f") if isinstance(value, Decimal) else str(value)


def _metric_lines(metrics: Mapping[str, object], *, untrusted: bool = False) -> list[str]:
    """One line per metric that has a case behind it, in `METRIC_NAMES` order.

    A metric whose value is `None` is OMITTED and never rendered as `- name: None`
    (17d criterion 11): a document a human opens after a red gate should carry the
    numbers the run measured, not fourteen words with nothing beside them. The
    test is `is None` and not falsiness - `0.0` accuracy and `0` errored are
    measurements and both render.

    `untrusted` suffixes the two `JUDGE_METRIC_NAMES` lines and nothing else (17f
    decision H). It is passed only for the RUN-level block: the per-category
    blocks report the judge's mean over a subset and never the number a reader
    quotes out of a gate, and their `judge_kappa_vs_human` is `None` and so has no
    line at all
    (`test_an_untrusted_calibration_notes_the_kappa_and_suffixes_the_two_judge_lines`).
    """
    return [
        f"- {name}: {_metric_value(metrics[name])}"
        + (UNTRUSTED_METRIC_SUFFIX if untrusted and name in JUDGE_METRIC_NAMES else "")
        for name in METRIC_NAMES
        if metrics.get(name) is not None
    ]


def _untrusted_note(calibration: RunCalibration | None) -> list[str]:
    """The one note an untrusted calibration puts at the top of `## Metrics`, or nothing.

    Rendered from the calibration's own `trusted` flag and never from a comparison
    made here: 17e is where `JUDGE_TRUST_MIN` is applied, and a second `>=` in this
    module is a second place the floor can be got wrong. A kappa of `None` still
    notes, as the word `null`, because a run whose judge agreement could not be
    measured at all is the one a reader most needs warned about
    (`test_an_untrusted_calibration_with_no_kappa_at_all_still_notes_it_as_null`).
    """
    if calibration is None or calibration.trusted:
        return []
    kappa = "null" if calibration.kappa is None else calibration.kappa
    return [UNTRUSTED_JUDGE_NOTE.format(kappa=kappa, floor=JUDGE_TRUST_MIN)]


def _metrics_lines(report: RunReport) -> list[str]:
    """The run-level metric set (#40), or nothing at all when nothing rendered.

    Under an untrusted calibration the block opens with `UNTRUSTED_JUDGE_NOTE` and
    the two judge lines carry `UNTRUSTED_METRIC_SUFFIX`; under a trusted one, and
    under no calibration at all, neither appears anywhere
    (`test_a_trusted_calibration_marks_nothing_in_the_metrics_block`).

    The empty case cannot be reached by any run that HAS a case, because
    `cross_tenant_leaks` and `errored` are counts and never `None`; the guard is
    kept for symmetry with `_taxonomy_lines` and because an empty block is the
    defect `_category_lines` was written to avoid. Untestable by design at this
    level - `test_a_run_of_no_cases_renders_no_per_category_metric_block_at_all`
    exercises the same guard on the block below, which is the only one a run shape
    can empty.
    """
    untrusted = _untrusted_note(report.calibration)
    lines = _metric_lines(report.metrics, untrusted=bool(untrusted))
    return [METRICS_HEADING, "", *untrusted, *lines] if lines else []


def _category_metric_lines(report: RunReport) -> list[str]:
    """The same metric set per category, each under its own `###` sub-heading.

    A `###` and not a `##`, so the whole breakdown is one section of the document
    rather than twelve, and so `_taxonomy_lines` and the failures still read as
    the top-level sections a reader scans for
    (`test_the_markdown_renders_one_metric_block_per_category_under_its_own_heading`).

    Rendered from `metrics["categories"]` rather than from the runs, which is what
    makes the document and `report.json` the same numbers by construction.

    The per-category `if rendered` is untestable by design, for the reason
    `_metrics_lines`'s guard is: a category exists because a case is in it, and
    `cross_tenant_leaks` and `errored` are counts that always have a value, so no
    run shape produces a category with nothing to render. It is kept rather than
    dropped because it is what stops a bare `### name` heading if a later metric
    set is ever all-nullable. The BLOCK-level guard below is the reachable one
    (`test_a_run_of_no_cases_renders_no_per_category_metric_block_at_all`).
    """
    categories: Mapping[str, Mapping[str, object]] = report.metrics["categories"]
    lines: list[str] = []
    for name, values in categories.items():
        rendered = _metric_lines(values)
        if rendered:
            lines += [f"### {name}", *rendered]
    return [CATEGORY_METRICS_HEADING, "", *lines] if lines else []


def _taxonomy_lines(report: RunReport) -> list[str]:
    """The failure taxonomy as counts (#40), or nothing at all on a clean run.

    Decision F: the tags are also a column of the failures table, but a reader
    counting a column by eye over two hundred cases is not an aggregate. Nothing
    is rendered when no case failed, heading included, for the reason the
    failures table is not (`test_a_clean_run_renders_no_taxonomy_block`).
    """
    counts: Mapping[str, int] = report.metrics["tags"]
    if not counts:
        return []
    return [TAXONOMY_HEADING, "", *(f"- {tag}: {count}" for tag, count in counts.items())]


def _failure_lines(report: RunReport) -> list[str]:
    """The failures section, and the last section of the document.

    A clean run renders ONE line here and no table at all, which is the whole
    reason this section is last: everything after the heading is the failures,
    and a reader who sees a header row expects rows under it.
    """
    failed = [case for case in report.cases if not case.passed]
    if not failed:
        return [FAILURES_HEADING, "", CLEAN_RUN_LINE]
    return [
        FAILURES_HEADING,
        "",
        "| case | category | tags | why |",
        "| --- | --- | --- | --- |",
        *(
            f"| {_cell(case.case_id)} | {_cell(case.category)} "
            f"| {_cell(', '.join(case.tags))} | {_cell(_reason(case))} |"
            for case in failed
        ),
    ]


def report_markdown(report: RunReport) -> str:
    """`report.md`'s text: who ran what, the counts, the categories, the failures.

    Rendered with f-strings and a joined list rather than a template engine
    (decision C): one document is not a dependency ask.

    A section that renders no lines contributes no blank paragraph either: the
    taxonomy is empty on a clean run, and joining it in anyway would leave two
    blank lines where a heading used to be
    (`test_a_clean_run_renders_no_taxonomy_block`).
    """
    sections = [
        _identity_lines(report),
        _count_lines(report),
        _category_lines(report),
        _metrics_lines(report),
        _category_metric_lines(report),
        _taxonomy_lines(report),
        _failure_lines(report),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections if lines) + "\n"


def write_reports(out: Path, report: RunReport) -> None:
    """Write both files into `out`, overwriting whatever a previous run left.

    Exactly two files, both named by the constants above and neither derived from
    a case id, a tenant or a clock (decision B), so the eval gate reads a known
    path and a rerun into the same directory leaves two documents that parse
    (`test_write_reports_writes_exactly_the_two_published_files`).

    Nothing is caught here: a write that fails is the caller's to report, and the
    CLI turns it into `EXIT_USAGE` naming the path rather than a traceback at the
    end of a paid-for run.
    """
    (out / REPORT_JSON_NAME).write_text(report_json(report), encoding="utf-8")
    (out / REPORT_MD_NAME).write_text(report_markdown(report), encoding="utf-8")
