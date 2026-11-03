r"""The four files one run writes: `report.json`, `report.md`, `needs-a-look.md` and `comment.md`.

Pure and clockless. `build_report` is handed the two timestamps, so the runner stays a plain
loop and every test of this module is deterministic.

Things the module holds that no single field states:

* every score is DERIVED from the finished plays and never accepted from a caller, so the
  report and the one line `ops-eval` prints cannot disagree;
* a case is ONE `CaseReport` however many times it was played: it passed when every scored
  play passed, and its tags are the rule ids its plays broke, once each, in the order first
  broken;
* the run passed when no play failed: a needs a look play broke no rule;
* `tenant` is `ALL_TENANTS` for a run: the cases name their own tenants.

The file names are constants and nothing a caller supplies shapes a path. `storage/eval_store.py`
builds the `eval_runs` and `eval_cases` rows from this record.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evals.results import (
    EVAL_BUG,
    FAIL,
    NEEDS_A_LOOK,
    NOT_SCORED,
    PlayRecord,
    compute_scores,
)

REPORT_JSON_NAME: Final[str] = "report.json"
"""The machine-readable report, the file the eval gate reads."""

REPORT_MD_NAME: Final[str] = "report.md"
"""The human-readable report, the file somebody opens after a red gate."""

NEEDS_A_LOOK_MD_NAME: Final[str] = "needs-a-look.md"
"""The plays that broke no rule and matched no listed end state, for a person to read."""

COMMENT_MD_NAME: Final[str] = "comment.md"
"""The text the workflow posts on the pull request."""

RUN_ID_LENGTH: Final[int] = 32
"""Characters in a run id: a `uuid4` hex, the `eval_runs` key."""

SET_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{7,64}")
"""What a set sha may be: 7 to 64 lowercase hex, matched with `fullmatch`."""

REPORT_INDENT: Final[int] = 2

ALL_TENANTS: Final[str] = "all"
"""The tenant a run records: each case names its own."""

SetShaSource = Literal["commit", "content"]
"""Where `set_sha` came from: the caller's commit, or the cases file's own bytes."""

RunKind = Literal["ci", "nightly", "manual"]
"""Why the run happened."""


def _new_run_id() -> str:
    """A fresh `uuid4` hex, one per built report."""
    return uuid.uuid4().hex


class CaseReport(BaseModel):
    """One case's row: what it was, whether it passed, and the plays behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    passed: bool
    tags: tuple[str, ...] = ()
    plays: tuple[PlayRecord, ...]


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
    """The sha256 of the prompt file the judge answered with, or `null` when nothing was
    judged."""
    cases: tuple[CaseReport, ...]


def _case_report(case_id: str, plays: Sequence[PlayRecord]) -> CaseReport:
    scored = [p for p in plays if p.result not in NOT_SCORED]
    tags: list[str] = []
    for play in plays:
        tags.extend(rule for rule in play.broken if rule not in tags)
    return CaseReport(
        case_id=case_id,
        category=plays[0].owner,
        passed=bool(scored) and all(p.result == "pass" for p in scored),
        tags=tuple(tags),
        plays=tuple(plays),
    )


def build_report(
    plays: Sequence[PlayRecord],
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
    trusted_check_types: frozenset[str] = frozenset(),
    judge_prompt_version: str | None = None,
) -> RunReport:
    """Turn the finished plays and the run's identity into the record the files render.

    `set_sha_source` and `run_kind` arrive as plain strings and are checked by the model's
    own vocabularies, so a caller that skipped the CLI's checks still cannot put a fourth
    word into a stored row.
    """
    grouped: dict[str, list[PlayRecord]] = {}
    for play in plays:
        grouped.setdefault(play.case_id, []).append(play)
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
        "passed": not any(p.result == FAIL for p in plays),
        "judge_prompt_version": judge_prompt_version,
        "metrics": compute_scores(plays, trusted_check_types=trusted_check_types),
        "cases": [_case_report(case_id, ps) for case_id, ps in grouped.items()],
    }
    return RunReport.model_validate(payload)


def report_json(report: RunReport) -> str:
    """`report.json`'s text: the whole record, indented, newline-terminated."""
    return report.model_dump_json(indent=REPORT_INDENT) + "\n"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _show(value: object) -> str:
    return "none" if value is None else str(value)


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
    return ["## Scores", "", *[f"- {name}: {_show(v)}" for name, v in report.metrics.items()]]


def _owner_lines(report: RunReport) -> list[str]:
    counts: dict[str, list[int]] = {}
    for case in report.cases:
        passed_other = counts.setdefault(case.category, [0, 0])
        passed_other[0 if case.passed else 1] += 1
    lines = ["## Cases by owner", "", "| owner | passed | not passed |", "| --- | --- | --- |"]
    lines += [f"| {name} | {p} | {o} |" for name, (p, o) in sorted(counts.items())]
    return lines


def _failure_lines(report: RunReport) -> list[str]:
    failed = [
        (case.case_id, play) for case in report.cases for play in case.plays if play.result == FAIL
    ]
    if not failed:
        return []
    lines = ["## Failures", ""]
    lines += [f"- {case_id} play {play.play}: {', '.join(play.broken)}" for case_id, play in failed]
    return lines


def _eval_bug_lines(report: RunReport) -> list[str]:
    """The plays the eval itself broke, apart from the failures: they say nothing about the
    help desk, and each says why the eval broke it."""
    bugs = [
        (case.case_id, play)
        for case in report.cases
        for play in case.plays
        if play.result == EVAL_BUG
    ]
    if not bugs:
        return []
    lines = ["## Eval bugs (not scored)", ""]
    lines += [
        f"- {case_id} play {play.play}: {_one_line(play.fault_reason) or 'no reason kept'}"
        for case_id, play in bugs
    ]
    return lines


def report_markdown(report: RunReport) -> str:
    """`report.md`'s text: identity, scores, cases by owner, the failures, then the eval
    bugs last."""
    sections = [
        _identity_lines(report),
        _metric_lines(report),
        _owner_lines(report),
        _failure_lines(report),
        _eval_bug_lines(report),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections if lines) + "\n"


def needs_a_look_markdown(report: RunReport) -> str:
    """`needs-a-look.md`'s text: each play that needs a look, with its conversation."""
    lines = ["# Needs a look", ""]
    found = False
    for case in report.cases:
        for play in case.plays:
            if play.result != NEEDS_A_LOOK:
                continue
            found = True
            lines.append(f"## {case.case_id} play {play.play}")
            if play.flagged:
                lines.append(f"The judge flagged: {', '.join(play.flagged)}")
            else:
                lines.append("No rule broken, and the end state matches none that is listed.")
            if play.fault is not None:
                lines.append(
                    f"Fault, as the judge sees it: {play.fault}. {_one_line(play.fault_reason)}"
                )
            for turn in play.conversation:
                lines.append(f"- person: {_one_line(turn.message)}")
                lines.append(f"- help desk: {_one_line(turn.answer.reply)}")
            lines.append("")
    if not found:
        lines.append("Nothing needs a look.")
    return "\n".join(lines).rstrip() + "\n"


def comment_markdown(report: RunReport) -> str:
    """`comment.md`'s text: the two gated scores and the counts, for the pull request."""
    metrics = report.metrics
    return (
        "## Eval run\n\n"
        f"- result: {'passed' if report.passed else 'failed'}\n"
        f"- set: {report.set_name} ({report.set_sha})\n\n"
        "| score | run |\n| --- | --- |\n"
        f"| safety | {_show(metrics.get('safety'))} |\n"
        f"| outcome_every_play | {_show(metrics.get('outcome_every_play'))} |\n"
        f"| outcome_at_least_once | {_show(metrics.get('outcome_at_least_once'))} |\n\n"
        f"- needs a look: {_show(metrics.get('needs_a_look'))}\n"
        f"- setup wrong: {_show(metrics.get('setup_wrong'))}\n"
        f"- eval bug: {_show(metrics.get('eval_bug'))}\n"
    )


def write_reports(out: Path, report: RunReport) -> None:
    """Write the four files into `out`, overwriting what a previous run left.

    Nothing is caught here: a write that fails is the caller's to report as a usage error
    naming the path.
    """
    (out / REPORT_JSON_NAME).write_text(report_json(report), encoding="utf-8")
    (out / REPORT_MD_NAME).write_text(report_markdown(report), encoding="utf-8")
    (out / NEEDS_A_LOOK_MD_NAME).write_text(needs_a_look_markdown(report), encoding="utf-8")
    (out / COMMENT_MD_NAME).write_text(comment_markdown(report), encoding="utf-8")
