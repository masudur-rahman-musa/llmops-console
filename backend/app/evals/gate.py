r"""`ops-eval gate`: the design's five rules over one published run (#41, #42).

Pure, and deliberately so. Everything here reads a `report.json` that already
exists and a `baseline.json` that is checked into the repo, so the same two files
give the same verdict on a laptop and in the eval gate workflow - no clock, no
database, no gateway. The command that shells over it lives in `cli/evals.py`;
what lives here is the comparison, the verdict and the markdown that verdict
becomes.

Every number in this module is in POINTS, 0-100. `scores.py` emits percentages
(`_percentage` answers `100 * passed / total` at `ACCURACY_SCALE` places) and
the design is written in points, so the baseline is written in points too
(decision A, a design note). That is not a formatting preference: the file shipped
before this row stated `refusal_recall_min: 0.95` as a FRACTION, and a gate
comparing a points report against a fraction floor passes a run at one percent
recall. `BASELINE_MIN` and `BASELINE_MAX` are what refuse the old file rather
than silently accept it
(`test_a_baseline_carrying_an_unknown_key_or_a_number_outside_the_range_is_refused`).

No judge metric is a gate rule (decision F). #36 asks for the judge metrics to be
excluded from the gate when the kappa is under 17e's trust floor; `RULE_ORDER`
contains neither `judge_mean` nor `judge_kappa_vs_human`, so the exclusion holds
by construction and a trusted and an untrusted calibration produce an equal
verdict (`test_a_trusted_and_an_untrusted_calibration_give_the_same_verdict`).
`gate_comment` is where that is SAID, because the comment is where the merge
decision is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.evals.report import JUDGE_METRIC_NAMES, RunReport

BASELINE_MIN: Final[float] = 0.0
"""The lowest a threshold or a reference metric may be, inclusive.

a project rule on a file an operator edits: every number in `baseline.json`
lands in a comparison that decides whether a PR merges, and a negative floor
makes a rule that can never fail. Pinned with `BASELINE_MAX` by
`test_the_gate_bounds_and_rule_names_are_the_published_ones`, and the inclusive
end by `test_a_baseline_sitting_exactly_on_a_bound_is_accepted`."""

BASELINE_MAX: Final[float] = 100.0
"""The highest a threshold or a reference metric may be, inclusive.

Inclusive because the shipped `injection_resistance_min` IS `100.0`: an exclusive
upper bound would make the gate unloadable the day the workflow first runs it
(`test_a_baseline_sitting_exactly_on_a_bound_is_accepted`)."""

MAX_SET_NAME_LENGTH: Final[int] = 128
"""The cap on `golden_set`, the one free-text string in the file.

a project rule again: the name is quoted into the PR comment the workflow posts, and
a path with no bound is a comment with no bound
(`test_a_golden_set_name_at_the_cap_is_accepted_and_one_character_over_is_refused`)."""

NO_BREACH_COUNT: Final[float] = 0.0
"""What the two count rules compare against: `> 0` is a breach, `0` is clean.

A constant rather than a literal in the comparison because it is also the
threshold a `RuleResult` carries and the comment renders, and the two must be the
same number (`test_each_rule_decides_only_at_its_own_inclusive_boundary`)."""

RULE_DECISION_ACCURACY: Final[str] = "decision_accuracy"
"""The drop rule's metric. Its threshold is not a constant: it is the baseline's
own reference accuracy less `max_drop_points`."""

RULE_REFUSAL_RECALL: Final[str] = "refusal_recall"
"""The refusal rule's metric, floored by `thresholds.refusal_recall_min`."""

RULE_INJECTION_RESISTANCE: Final[str] = "injection_resistance"
"""The injection rule's metric, floored by `thresholds.injection_resistance_min`."""

RULE_CROSS_TENANT_LEAKS: Final[str] = "cross_tenant_leaks"
"""The leak rule's count. Any leak at all is a breach, and it is a residency breach."""

RULE_RESIDENCY_VIOLATIONS: Final[str] = "residency_violations"
"""The residency rule's count, which `build_report` writes into the count block
rather than into `METRIC_NAMES` - the one rule here that is not a metric name."""

RULE_ORDER: Final[tuple[str, ...]] = (
    RULE_DECISION_ACCURACY,
    RULE_REFUSAL_RECALL,
    RULE_INJECTION_RESISTANCE,
    RULE_CROSS_TENANT_LEAKS,
    RULE_RESIDENCY_VIOLATIONS,
)
"""The five rules the design names, in the order a verdict reports them.

Fixed for `METRIC_NAMES`' reason: the workflow renders these strings into a PR
comment and reads them out of a workflow, so a reorder makes two gate runs of the
same report diff. Every name is a key a real report carries, which is asserted
against the imported `METRIC_NAMES` and the count block by
`test_every_rule_name_is_a_key_the_report_actually_carries` - a rule named for
`accuracy_min`, the key decision A drops, would gate on nothing forever.
"""

_COUNT_RULES: Final[frozenset[str]] = frozenset(
    {RULE_CROSS_TENANT_LEAKS, RULE_RESIDENCY_VIOLATIONS}
)
"""The two rules that are counts rather than percentages, and the two that make
`GateVerdict.residency_breached` true. Both are `> 0` breaches, both outrank a
threshold breach in `cli/evals.py`'s exit code (decision D)."""

_Points = Annotated[float, Field(ge=BASELINE_MIN, le=BASELINE_MAX)]
"""A number in the baseline: a percentage in points, bounded inclusively at both ends."""


class GateInputInvalid(Exception):
    """One of the gate's two input files could not be used, and the message names it.

    `CalibrationSetInvalid`'s shape and its reasons: a plain exception rather than
    a `GatewayError`, because an unloadable file is an operator error reported to
    whoever ran the harness and never an answer to a caller, and
    `tests/gateway/test_error_contract.py` walks `GatewayError.__subclasses__()`
    asserting one class per wire code.

    Two subclasses rather than one class with a flag, so `cli/evals.py`'s refusal
    block can say which of the two files it refused without parsing the reason.
    """

    def __init__(self, reason: str, *, path: Path) -> None:
        self.reason = reason
        self.path = path
        super().__init__(f"{path}: {reason}")


class BaselineInvalid(GateInputInvalid):
    """`--baseline` could not be read, parsed, or validated.

    Every way the file can be wrong arrives as this one class, which is what lets
    the CLI print one line and answer `EXIT_USAGE` rather than let an `OSError`,
    a `JSONDecodeError` or a pydantic `ValidationError` reach an operator as a
    traceback (`test_a_baseline_path_that_does_not_exist_is_refused_and_the_message_names_it`,
    `test_a_baseline_that_is_not_json_is_refused_as_the_same_typed_error`,
    `test_a_valid_json_file_that_is_not_a_baseline_is_refused_as_baseline_invalid`).
    """


class ReportInvalid(GateInputInvalid):
    """`--report` could not be read, parsed, or validated as a `RunReport`.

    `BaselineInvalid`'s three shapes over the other file: a mistyped path, a
    half-written artifact from a killed run, and a document from some other schema
    (`test_a_malformed_report_or_baseline_is_one_line_on_stderr_and_exit_two`).
    """


class BaselineThresholds(BaseModel):
    """The three numbers the design states as thresholds, all in points.

    `extra="forbid"` is what refuses the fraction-era `accuracy_min` rather than
    reading past it: a file half of whose numbers nobody reads is worse than a
    file that will not load, because everyone believes the unread ones are being
    enforced (`test_a_baseline_carrying_an_unknown_key_or_a_number_outside_the_range_is_refused`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_drop_points: _Points
    refusal_recall_min: _Points
    injection_resistance_min: _Points


class BaselineMetrics(BaseModel):
    """The reference the drop rule measures from (decision B).

    One field today. It is a BLOCK rather than a bare `decision_accuracy` key so
    a second gated reference - a nightly median, another accuracy - is a field
    here and not a change of shape to the whole file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_accuracy: _Points


class Baseline(BaseModel):
    """`backend/evals/baseline.json`, validated whole before any comparison runs.

    `rule` is read from the file's `_rule` key, which carries a project rule's own
    instruction in the file it governs. It is a real field rather than an ignored
    extra because `extra="forbid"` admits no ignored extras
    (`test_the_baseline_reads_the_shipped_files_prose_key_under_its_own_name`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str = Field(alias="_rule")
    golden_set: Annotated[str, Field(max_length=MAX_SET_NAME_LENGTH)]
    thresholds: BaselineThresholds
    metrics: BaselineMetrics


class RuleResult(BaseModel):
    """One rule's answer: what the run measured, what it had to clear, and which won.

    `actual` is `None` when the report carried `null` for the metric, and that is
    a BREACH (decision C): a golden set that gates refusal recall and no longer
    states a refusal case has lost the cases, and reading `null` as "nothing
    against us" merges that green forever
    (`test_a_null_gated_metric_breaches_its_own_rule_and_no_other`). `None` rather
    than `0.0` so the comment and the CLI can say "no evidence" instead of
    printing a number nobody measured.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    actual: float | None
    threshold: float
    passed: bool


class GateVerdict(BaseModel):
    """The whole answer: all five rules, always, plus whether a boundary was breached.

    All five and never only the failures (criterion 2): a reviewer meeting a red
    gate needs to see that the other four were measured, and a gate that
    short-circuits on the first breach sends them back for a second round
    (`test_all_five_rules_breached_at_once_are_all_reported_in_rule_order`).

    `residency_breached` is carried rather than recomputed by callers because it
    is what decides `EXIT_RESIDENCY` over `EXIT_FAILED` (decision D), and a second
    `> 0` in `cli/evals.py` is a second place that ranking can be got wrong.
    """

    model_config = ConfigDict(frozen=True)

    results: tuple[RuleResult, ...]
    residency_breached: bool

    @property
    def passed(self) -> bool:
        """Whether every rule cleared. The gate's whole question."""
        return all(result.passed for result in self.results)

    @property
    def breaches(self) -> tuple[RuleResult, ...]:
        """The rules that failed, in `RULE_ORDER`. Empty on a clean run."""
        return tuple(result for result in self.results if not result.passed)


VERDICT_PASSED_LINE: Final[str] = "gate passed: {total} of {total} rules clear"
"""The ONE line a clean gate prints, and the first line of its comment.

A format template rather than a finished sentence for `UNTRUSTED_JUDGE_NOTE`'s
reason: the count comes from `RULE_ORDER`, so a sixth rule is reported as six
without anyone remembering to edit a string
(`test_gate_on_a_run_that_breaches_nothing_exits_zero_with_one_line_on_stdout`)."""

VERDICT_FAILED_LINE: Final[str] = "gate FAILED: {failed} of {total} rules breached ({names})"
"""The ONE line a red gate prints, and the first line of its comment.

It names the breached rules, so the workflow's log says what broke without opening
the artifact, and the same string is what the comment opens with - the two are
rendered once and never by separate code that can disagree about whether the run
passed
(`test_the_comment_file_carries_the_verdict_line_and_the_rules_while_stdout_stays_one_line`)."""

NO_EVIDENCE: Final[str] = "no evidence"
"""What a `null` metric renders as, in the reason line and in the comment's table.

Decision C's second half: the two failures have to read differently, because
"refusal recall was 94.9" and "no case stated a refusal expectation" have
different fixes, and printing `None` or `0.0` for the second sends a reviewer
looking for a regression that never happened
(`test_a_null_gated_metric_exits_one_and_says_no_evidence_rather_than_a_number`)."""

JUDGE_EXCLUSION_NOTE: Final[str] = (
    "The judge metrics ({names}) are excluded from every gate rule: an LLM judge "
    "scores against a rubric and is calibrated against humans, so it reports and "
    "never decides (#36)."
)
"""The sentence every comment carries, whatever the run's calibration says.

Decision F stated where the merge decision is made. Without it a reviewer reading
a green gate beside a judge mean of 2.1 assumes the judge was consulted; with it,
"excluded when the kappa is under the floor" is visibly a property of the rule
set rather than of a branch somebody can delete
(`test_the_comment_states_that_the_judge_metrics_are_excluded_from_the_gate`)."""

CALIBRATION_NOTE: Final[str] = (
    "Judge calibration: kappa {kappa} against {labels_name} over {pairs} pairs, {trust}."
)
"""What a comment says about the calibration the run published, when it had one.

The kappa is rendered from the report's own record rather than restated as a
word, because it is the only place a reviewer can learn the judge disagreed with
the humans - the verdict cannot tell them, by decision F
(`test_only_the_comment_text_tells_a_trusted_calibration_from_an_untrusted_one`)."""

NO_CALIBRATION_NOTE: Final[str] = (
    "Judge calibration: none supplied, so no agreement with human labels is known."
)
"""What a comment says when the run carried no calibration at all. Said rather
than omitted: a silent comment and a trusted one look identical to a reviewer."""

TRUSTED_WORD: Final[str] = "trusted"
UNTRUSTED_WORD: Final[str] = "NOT trusted"
"""The two verdicts 17e's `JUDGE_TRUST_MIN` already decided. Read off
`RunCalibration.trusted` and never recompared here, for the reason
`report._untrusted_note` gives: a second comparison is a second place the floor
can be got wrong."""


def _one_line(exc: ValidationError) -> str:
    """One pydantic `ValidationError` as ONE line: the first problem, and how many more.

    `str(exc)` runs to a line per field plus a documentation URL for each - forty
    for a file that is JSON and nothing else - and both refusals below travel to a
    CI log as a single stderr line the eval gate workflow reads
    (`test_a_malformed_report_or_baseline_is_one_line_on_stderr_and_exit_two`).
    The first error is the one reported, named by its field, exactly as
    `gateway/errors.py` renders a request validation failure.
    """
    problems = exc.errors()
    if not problems:
        return "did not validate"
    first = problems[0]
    location = first.get("loc") or ()
    field = ".".join(str(part) for part in location) or "the document"
    message = str(first.get("msg", "is not valid")).replace("\n", " ")
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{field}: {message}{more}"


def load_baseline(path: Path) -> Baseline:
    """Read `baseline.json`, or raise `BaselineInvalid` naming the path.

    The shape `cases.load_golden_set` and `calibration.read_calibration` already
    have, and for the same reason: every way the file can be wrong becomes ONE
    typed refusal, so the CLI's pre-check block gains an exception class and no
    new structure.

    The range checks are the MODEL's, not a pass afterwards: `_Points` bounds
    every threshold and the reference accuracy inclusively, so a file is either
    wholly valid or wholly refused and nothing compares against a number that got
    halfway through validation (criterion 8).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc

    try:
        return Baseline.model_validate(payload)
    except ValidationError as exc:
        raise BaselineInvalid(f"is not a baseline: {_one_line(exc)}", path=path) from exc


def load_report(path: Path) -> RunReport:
    """Read a run's `report.json` back, or raise `ReportInvalid` naming the path.

    `load_baseline`'s three guards over the other input. The document is validated
    through `RunReport` itself and never by picking the five gated keys out of it:
    a reader that took the keys alone would gate a hand-written file whose numbers
    no run produced
    (`test_a_malformed_report_or_baseline_is_one_line_on_stderr_and_exit_two`).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc

    try:
        return RunReport.model_validate(payload)
    except ValidationError as exc:
        raise ReportInvalid(f"is not a run report: {_one_line(exc)}", path=path) from exc


def _actual(value: Any) -> float | None:  # noqa: ANN401 - `metrics` is a Mapping[str, Any]
    """One metric as a rule reads it: a float, or `None` where the report said null."""
    return None if value is None else float(value)


def evaluate(report: RunReport, baseline: Baseline) -> GateVerdict:
    """Answer the five rules over one published run. Pure: no I/O, no clock.

    The same report and baseline give an equal verdict every time
    (`test_evaluating_one_report_twice_gives_an_equal_verdict`), which is what
    lets the workflow run the gate over a stored artifact and get the answer CI got.

    Every comparison is `<` for a threshold and `>` for a count, exactly as SPEC
    §12.6 writes them, so a run landing ON a floor or dropping EXACTLY
    `max_drop_points` is green - the boundary is inclusive in the run's favour
    (`test_each_rule_decides_only_at_its_own_inclusive_boundary`). A `null`
    breaches (decision C).

    Only the five rules' own keys are read out of `report.metrics`, each of them a
    `METRIC_NAMES` entry or one of `build_report`'s counts
    (`test_evaluate_reads_no_metric_key_outside_the_published_set`): a key this
    module invented would be a `KeyError` on every real run, or worse a `.get`
    default gating against nothing.
    """
    floors = {
        RULE_DECISION_ACCURACY: baseline.metrics.decision_accuracy
        - baseline.thresholds.max_drop_points,
        RULE_REFUSAL_RECALL: baseline.thresholds.refusal_recall_min,
        RULE_INJECTION_RESISTANCE: baseline.thresholds.injection_resistance_min,
    }
    results = []
    for name in RULE_ORDER:
        actual = _actual(report.metrics[name])
        threshold = NO_BREACH_COUNT if name in _COUNT_RULES else floors[name]
        if actual is None:
            passed = False
        elif name in _COUNT_RULES:
            passed = actual <= threshold
        else:
            passed = actual >= threshold
        results.append(RuleResult(name=name, actual=actual, threshold=threshold, passed=passed))
    return GateVerdict(
        results=tuple(results),
        residency_breached=any(
            not result.passed for result in results if result.name in _COUNT_RULES
        ),
    )


def verdict_line(verdict: GateVerdict) -> str:
    """The ONE line the command prints and the comment opens with (decision E).

    Rendered once and used twice, so the summary a workflow reads and the markdown
    a reviewer reads cannot disagree about whether the run passed
    (`test_the_comment_file_carries_the_verdict_line_and_the_rules_while_stdout_stays_one_line`).
    """
    total = len(verdict.results)
    if verdict.passed:
        return VERDICT_PASSED_LINE.format(total=total)
    breaches = verdict.breaches
    return VERDICT_FAILED_LINE.format(
        failed=len(breaches),
        total=total,
        names=", ".join(result.name for result in breaches),
    )


def _rendered(value: float | None) -> str:
    """A measured value as both outputs print it, or `NO_EVIDENCE` for a null."""
    return NO_EVIDENCE if value is None else str(value)


def reason_line(result: RuleResult) -> str:
    """One breached rule as the command prints it on stderr, one line each.

    A null says `NO_EVIDENCE` and carries no number at all, so the two failures a
    reviewer can meet - "this got worse" and "the cases that measured it are
    gone" - do not read the same
    (`test_a_null_gated_metric_exits_one_and_says_no_evidence_rather_than_a_number`).
    """
    if result.actual is None:
        return f"{result.name}: {NO_EVIDENCE}, and no case in this run stated it"
    if result.name in _COUNT_RULES:
        return f"{result.name}: {_rendered(result.actual)}, and any at all is a breach"
    return f"{result.name}: {_rendered(result.actual)} is under {result.threshold}"


def _calibration_line(report: RunReport) -> str:
    """What the comment says about the run's judge calibration, always exactly one line."""
    calibration = report.calibration
    if calibration is None:
        return NO_CALIBRATION_NOTE
    return CALIBRATION_NOTE.format(
        kappa="null" if calibration.kappa is None else calibration.kappa,
        labels_name=calibration.labels_name,
        pairs=calibration.pairs,
        trust=TRUSTED_WORD if calibration.trusted else UNTRUSTED_WORD,
    )


def _category_lines(report: RunReport) -> list[str]:
    """The report's own per-category table, one row per category the run used.

    Carried into the comment because it is what turns "decision accuracy fell two
    points" into "policy_qa fell over" - the design asks for it by name
    (`test_the_comment_carries_the_reports_per_category_table`).
    """
    categories: Any = report.metrics["categories"]
    lines = ["| category | passed | failed |", "| --- | --- | --- |"]
    lines += [
        f"| {name} | {counts['passed']} | {counts['failed']} |"
        for name, counts in categories.items()
    ]
    return lines


def gate_comment(verdict: GateVerdict, report: RunReport) -> str:
    """The markdown the workflow posts on the PR, and `--comment` writes to disk.

    It opens with `verdict_line`, then all five rules with their actual against
    their threshold - the four that passed included, because that is how a
    reviewer knows they were measured - then the run's per-category table, then
    the two judge lines: what the calibration said, and that the judge decides
    nothing here (decision F).

    Rendered from BOTH arguments and never from the verdict alone: the categories
    and the calibration exist only on the report, and a comment built from the
    verdict would post identical markdown for a run whose judge was trustworthy
    and one whose kappa was 0.42
    (`test_only_the_comment_text_tells_a_trusted_calibration_from_an_untrusted_one`).
    """
    lines = [
        "## Eval gate",
        "",
        verdict_line(verdict),
        "",
        f"- set: {report.set_name}",
        f"- set sha: {report.set_sha} ({report.set_sha_source})",
        f"- run: {report.id}",
        "",
        "| rule | actual | threshold | verdict |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {result.name} | {_rendered(result.actual)} | {result.threshold} | "
        f"{'pass' if result.passed else 'BREACH'} |"
        for result in verdict.results
    ]
    lines += ["", *_category_lines(report), ""]
    lines += [
        _calibration_line(report),
        "",
        JUDGE_EXCLUSION_NOTE.format(names=", ".join(JUDGE_METRIC_NAMES)),
    ]
    return "\n".join(lines) + "\n"
