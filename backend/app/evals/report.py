r"""The two files one conversation run writes: `report.json` and `report.md`.

Pure and clockless. `build_report` is handed the two timestamps, so the runner stays a
plain loop and every test of this module is deterministic.

Things the module holds that no single field states:

* every count and score is DERIVED from the saved `CaseLine` records and never accepted
  from a caller, so the report and the one line `ops-eval` prints cannot disagree;
* `check` is `None` exactly where the saved line's is: a case that never answered, which
  is not the same claim as a case whose checks all passed;
* the three fields a per-turn eval measured (`latency_ms`, `cost_usd`, `provider`) are
  kept as `None` on every case. A conversation answer carries none of them, and a zero
  there would read as a free, instant success;
* a failed case is tagged with each check its first failed run broke, once each, in
  `FAILURE_TAGS` order. A passed case, or one that never answered, has none;
* `tenant` is `ALL_TENANTS` for a conversation run: the cases name their own tenants;
* a case played several times is ONE `CaseReport`: it passed when every run passed, it
  carries its pass rate and 95% range, and an unstable one is listed in its own section
  and never under the failures.

The two file names are constants and nothing a caller supplies shapes a path.
`storage/eval_store.py` builds the `eval_runs` and `eval_cases` rows from this record.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.evals.calibration import CalibrationReport
from app.evals.checks import CaseCheck
from app.evals.runner import CaseLine, JudgedReply
from app.evals.scores import compute_metrics, is_unstable, pass_rate, runs_by_case, wilson_range

REPORT_JSON_NAME: Final[str] = "report.json"
"""The machine-readable report, the file the eval gate reads."""

REPORT_MD_NAME: Final[str] = "report.md"
"""The human-readable report, the file somebody opens after a red gate."""

RUN_ID_LENGTH: Final[int] = 32
"""Characters in a run id: a `uuid4` hex, the `eval_runs` key."""

SET_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{7,64}")
"""What a set sha may be: 7 to 64 lowercase hex, matched with `fullmatch`."""

REPORT_INDENT: Final[int] = 2

ALL_TENANTS: Final[str] = "all"
"""The tenant a conversation run records: each case names its own."""

SetShaSource = Literal["commit", "content"]
"""Where `set_sha` came from: the caller's commit, or the cases file's own bytes."""

RunKind = Literal["ci", "nightly", "manual"]
"""Why the run happened."""

FAILURES_HEADING: Final[str] = "## Failures"
"""Where the failed cases live in `report.md`, and the last section of it. A clean run
has no such heading."""

JUDGE_HEADING: Final[str] = "## Replies the judge failed"
METRICS_HEADING: Final[str] = "## Metrics"
CATEGORIES_HEADING: Final[str] = "## Cases by category"
UNSTABLE_HEADING: Final[str] = "## Unstable cases"
"""Cases that both passed and failed across their runs. Left out of `check_score`."""

FAILURE_TAGS: Final[tuple[str, ...]] = ("wrong_tool", "early_change", "made_up_name", "judge_low")
"""The tags a failed conversation case can carry, in the order it carries them."""

RunCalibration = CalibrationReport
"""What a run publishes about its judge: the six fields of a `calibration.json`."""


def _new_run_id() -> str:
    """A fresh `uuid4` hex, one per built report."""
    return uuid.uuid4().hex


class CaseReport(BaseModel):
    """One case's row: what it was, whether it passed, and why not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    passed: bool
    tags: tuple[str, ...] = ()
    latency_ms: int | None = None
    cost_usd: Decimal | None = None
    provider: str | None = None
    error: str | None
    check: CaseCheck | None
    judged: tuple[JudgedReply, ...]
    runs: int = 1
    pass_rate: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    """The 95% Wilson range of `pass_rate`, in percent. `None` on a report from before
    repeat runs, as are `pass_rate` and the low end."""
    unstable: bool = False

    @field_serializer("cost_usd")
    def _serialise_cost(self, value: Decimal | None) -> str | None:
        """Write a cost as plain decimal text, never through a float."""
        return None if value is None else format(value, "f")


class RunReport(BaseModel):
    """One run: its identity, its metrics, and one `CaseReport` per case."""

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
    judge_prompt_version: str | None = None
    """The sha256 of the prompt file the judge graded this run with, or `null` when
    nothing was judged."""
    calibration: RunCalibration | None = None
    """What the run was told about its judge, or `null` when it was given nothing."""
    cases: tuple[CaseReport, ...]


def _failure_tags(line: CaseLine) -> tuple[str, ...]:
    """The checks one failed run broke, in `FAILURE_TAGS` order.

    A run that never answered (it has an `error`) gets none: the target failed, not a
    check. A run with no `check` can still be `judge_low`. Tests pin this.
    """
    if line.passed or line.error is not None:
        return ()
    broken: set[str] = set()
    if line.check is not None:
        if not line.check.tools_in_order:
            broken.add("wrong_tool")
        if any(not turn.no_early_change for turn in line.check.turns):
            broken.add("early_change")
        if any(not turn.no_made_up_name for turn in line.check.turns):
            broken.add("made_up_name")
    if any(reply.verdict == "fail" for reply in line.judged):
        broken.add("judge_low")
    return tuple(tag for tag in FAILURE_TAGS if tag in broken)


def _case_report(runs: Sequence[CaseLine]) -> CaseReport:
    """One case's row from all its runs. The first failed run explains a failure."""
    shown = next((line for line in runs if not line.passed), runs[0])
    passes = sum(1 for line in runs if line.passed)
    low, high = wilson_range(passes, len(runs))
    return CaseReport(
        case_id=shown.case_id,
        category=shown.category,
        passed=passes == len(runs),
        tags=_failure_tags(shown),
        error=shown.error,
        check=shown.check,
        judged=tuple(reply for line in runs for reply in line.judged),
        runs=len(runs),
        pass_rate=pass_rate(passes, len(runs)),
        range_low=low,
        range_high=high,
        unstable=is_unstable(runs),
    )


def build_report(
    lines: Sequence[CaseLine],
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
    """Turn the saved case lines and the run's identity into the record both files render.

    `set_sha_source` and `run_kind` arrive as plain strings and are checked by the
    model's own vocabularies, so a caller that skipped the CLI's checks still cannot put
    a fourth word into a stored row. The run passes when no stable case failed.
    """
    cases = [_case_report(runs) for runs in runs_by_case(lines).values()]
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
        "passed": all(case.passed or case.unstable for case in cases),
        "judge_prompt_version": judge_prompt_version,
        "calibration": calibration,
        "metrics": compute_metrics(lines),
        "cases": cases,
    }
    return RunReport.model_validate(payload)


def report_json(report: RunReport) -> str:
    """`report.json`'s text: the whole record, indented, newline-terminated."""
    return report.model_dump_json(indent=REPORT_INDENT) + "\n"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _why_it_failed(case: CaseReport) -> str:
    """The reason a case failed, in a clause per broken check."""
    if case.error is not None:
        return _one_line(case.error)
    if case.check is None:
        return "no result"
    reasons: list[str] = []
    if not case.check.tools_in_order:
        reasons.append("the right tools were not called in order")
    for turn in case.check.turns:
        if not turn.no_early_change:
            reasons.append(f"turn {turn.turn} made a change before the yes")
        if not turn.no_made_up_name:
            reasons.append(f"turn {turn.turn} named a number or role no tool returned")
    return "; ".join(reasons) or "failed"


def _identity_lines(report: RunReport) -> list[str]:
    return [
        f"# Eval run {report.id}",
        "",
        f"- set: {report.set_name}",
        f"- set sha: {report.set_sha} ({report.set_sha_source})",
        f"- run kind: {report.run_kind}",
        f"- target: {report.target}",
        f"- git sha: {report.git_sha if report.git_sha is not None else 'unknown'}",
        f"- started: {report.started_at.isoformat()}",
        f"- finished: {report.finished_at.isoformat()}",
        f"- result: {'passed' if report.passed else 'failed'}",
    ]


def _metric_lines(report: RunReport) -> list[str]:
    lines = [METRICS_HEADING, ""]
    for name, value in report.metrics.items():
        lines.append(f"- {name}: {'null' if value is None else value}")
    if report.calibration is not None:
        kappa = report.calibration.kappa
        trust = "trusted" if report.calibration.trusted else "NOT trusted"
        lines.append(
            f"- judge calibration: kappa {'null' if kappa is None else kappa} against "
            f"{report.calibration.labels_name} over {report.calibration.pairs} pairs, {trust}"
        )
    return lines


def _category_lines(report: RunReport) -> list[str]:
    counts: dict[str, list[int]] = {}
    for case in report.cases:
        passed_failed = counts.setdefault(case.category, [0, 0])
        passed_failed[0 if case.passed else 1] += 1
    lines = [CATEGORIES_HEADING, "", "| category | passed | failed |", "| --- | --- | --- |"]
    lines += [f"| {name} | {p} | {f} |" for name, (p, f) in sorted(counts.items())]
    return lines


def _judge_lines(report: RunReport) -> list[str]:
    failed = [
        (case.case_id, reply)
        for case in report.cases
        for reply in case.judged
        if reply.verdict == "fail"
    ]
    if not failed:
        return []
    lines = [JUDGE_HEADING, ""]
    lines += [
        f"- {case_id} turn {reply.turn}: {_one_line(reply.reason or 'no reason given')}"
        for case_id, reply in failed
    ]
    return lines


def _unstable_lines(report: RunReport) -> list[str]:
    unstable = [case for case in report.cases if case.unstable]
    if not unstable:
        return []
    lines = [UNSTABLE_HEADING, ""]
    lines += [
        f"- {case.case_id} ({case.category}): passed {case.pass_rate}% of {case.runs} runs, "
        f"95% range {case.range_low} to {case.range_high}"
        for case in unstable
    ]
    return lines


def _failure_lines(report: RunReport) -> list[str]:
    failed = [case for case in report.cases if not case.passed and not case.unstable]
    if not failed:
        return []
    lines = [FAILURES_HEADING, ""]
    lines += [f"- {case.case_id} ({case.category}): {_why_it_failed(case)}" for case in failed]
    return lines


def report_markdown(report: RunReport) -> str:
    """`report.md`'s text: identity, metrics, categories, judge fails, unstable cases, then
    the failures last."""
    sections = [
        _identity_lines(report),
        _metric_lines(report),
        _category_lines(report),
        _judge_lines(report),
        _unstable_lines(report),
        _failure_lines(report),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections if lines) + "\n"


def write_reports(out: Path, report: RunReport) -> None:
    """Write both files into `out`, overwriting what a previous run left.

    Exactly two files, named by the constants above. Nothing is caught here: a write
    that fails is the caller's to report as a usage error naming the path.
    """
    (out / REPORT_JSON_NAME).write_text(report_json(report), encoding="utf-8")
    (out / REPORT_MD_NAME).write_text(report_markdown(report), encoding="utf-8")
