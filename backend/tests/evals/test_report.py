"""`evals/report.py`: the report a run writes.

Pure: finished plays go in, a `RunReport` and four files come out.

Contract these tests are written against:

* `build_report(plays, *, started_at, finished_at, set_name, set_sha, set_sha_source,
  run_kind, git_sha, tenant, target, trusted_check_types=frozenset()) -> RunReport` with
  `metrics` from `results.compute_scores` and one `CaseReport` per case (`case_id`,
  `category`, `passed`, `tags`, `plays`).
* `report_json(report)`, `report_markdown(report)`, `needs_a_look_markdown(report)`,
  `write_reports(out, report)`, which writes `report.json`, `report.md`, `needs-a-look.md`
  and `comment.md` into `out`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.evals.report import (
    COMMENT_MD_NAME,
    NEEDS_A_LOOK_MD_NAME,
    REPORT_JSON_NAME,
    REPORT_MD_NAME,
    RunReport,
    build_report,
    report_json,
    write_reports,
)
from app.evals.results import PlayRecord

from .builders import play_record

STARTED = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
FINISHED = datetime(2026, 11, 2, 9, 20, tzinfo=UTC)


def build(plays: tuple[PlayRecord, ...]) -> RunReport:
    return build_report(
        plays,
        started_at=STARTED,
        finished_at=FINISHED,
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target="https://helpdesk.example",
    )


PLAYS = (
    play_record("alpha", 1, "pass"),
    play_record("alpha", 2, "pass"),
    play_record("beta", 1, "fail", broken=["early_change"]),
    play_record("gamma", 1, "needs_a_look"),
)


def test_the_run_writes_its_four_files(tmp_path: Path) -> None:
    """Catches a run that leaves out the needs-a-look list or the gate comment, or writes
    them under another name: the workflow posts `comment.md` and a person reads
    `needs-a-look.md`. Exactly four files, so a stray leftover of the old format shows."""
    write_reports(tmp_path, build(PLAYS))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "comment.md",
        "needs-a-look.md",
        "report.json",
        "report.md",
    ]
    assert (REPORT_JSON_NAME, REPORT_MD_NAME, NEEDS_A_LOOK_MD_NAME, COMMENT_MD_NAME) == (
        "report.json",
        "report.md",
        "needs-a-look.md",
        "comment.md",
    )


def test_the_json_round_trips_through_the_model() -> None:
    """Catches a report file the gate cannot read back: what is written must validate as the
    same report."""
    report = build(PLAYS)

    assert RunReport.model_validate_json(report_json(report)) == report


def test_the_needs_a_look_list_names_only_the_cases_that_need_a_look(tmp_path: Path) -> None:
    """Catches a list that holds every case or the failures: gamma needs a look, alpha passed
    and beta failed, so only gamma is named."""
    write_reports(tmp_path, build(PLAYS))

    text = (tmp_path / NEEDS_A_LOOK_MD_NAME).read_text(encoding="utf-8")
    assert "gamma" in text
    assert "alpha" not in text
    assert "beta" not in text


def test_the_comment_carries_the_two_gated_scores(tmp_path: Path) -> None:
    """Catches a comment that shows the old score names or none: the pull request reader
    sees `safety` and `outcome_every_play`."""
    write_reports(tmp_path, build(PLAYS))

    text = (tmp_path / COMMENT_MD_NAME).read_text(encoding="utf-8")
    assert "safety" in text
    assert "outcome_every_play" in text
    assert "check_score" not in text


def test_the_metrics_are_the_new_scores_and_none_of_the_old() -> None:
    """Catches a report still carrying the retired scores next to the new ones."""
    metrics = json.loads(report_json(build(PLAYS)))["metrics"]

    assert metrics["safety"] == 75.0
    assert metrics["outcome_every_play"] == 33.33
    assert "check_score" not in metrics
    assert "judge_score" not in metrics


def test_a_run_with_a_failed_play_is_not_passed() -> None:
    """Catches `passed` true while a rule was broken. beta failed one play."""
    assert build(PLAYS).passed is False


@pytest.mark.parametrize("result", ["pass", "needs_a_look"])
def test_a_run_with_no_failed_play_is_passed(result: str) -> None:
    """Catches `passed` false for a needs a look play: that result breaks no rule."""
    assert build((play_record("alpha", 1, result),)).passed is True


def test_eval_bugs_are_listed_apart_with_their_reason(tmp_path: Path) -> None:
    """Catches an eval bug hidden among the failures or the plays to look at, or listed with
    no reason: the report names the case, its play and the reason under its own eval bug
    heading, the bug is not named in the failures or in the needs-a-look list, and the
    comment counts it."""
    plays = (
        *PLAYS,
        play_record(
            "delta",
            2,
            "eval_bug",
            fault_reason="the person stopped answering after the first reply",
        ),
    )
    report = build(plays)

    write_reports(tmp_path, report)
    text = (tmp_path / REPORT_MD_NAME).read_text(encoding="utf-8")
    found = re.search(r"^#+ .*eval bug", text, re.IGNORECASE | re.MULTILINE)
    assert found is not None
    heading = found.start()
    listed = text[heading:]

    assert "delta" in listed
    assert "play 2" in listed
    assert "the person stopped answering after the first reply" in listed
    assert "delta" not in text[:heading]
    assert "delta" not in (tmp_path / NEEDS_A_LOOK_MD_NAME).read_text(encoding="utf-8")
    assert report.metrics["eval_bug"] == 1
    assert "eval bug" in (tmp_path / COMMENT_MD_NAME).read_text(encoding="utf-8").lower()


def test_a_run_with_an_eval_bug_play_is_not_failed_for_it() -> None:
    """Catches an eval bug counted as a failed rule: it broke no rule, so the run still
    passes."""
    assert build((play_record("alpha", 1, "pass"), play_record("beta", 1, "eval_bug"))).passed


def test_a_needs_a_look_play_shows_its_fault_answer(tmp_path: Path) -> None:
    """Catches a fault answer stored and never shown: the needs-a-look list says "fault" and
    gives the reason beside the play."""
    plays = (
        play_record(
            "gamma", 1, "needs_a_look", fault="case", fault_reason="the case wants two things"
        ),
    )

    write_reports(tmp_path, build(plays))

    text = (tmp_path / NEEDS_A_LOOK_MD_NAME).read_text(encoding="utf-8")
    assert "fault" in text.lower()
    assert "the case wants two things" in text


def test_a_fail_shows_its_fault_answer(tmp_path: Path) -> None:
    """Catches a fault answer kept on a failed play and never shown: the failures section of
    `report.md` gives the judge's fault and reason beside the play, and a failed play with no
    fault answer shows no such line."""
    plays = (
        play_record(
            "beta",
            1,
            "fail",
            broken=["early_change"],
            fault="eval",
            fault_reason="the person agreed but did not mark a yes",
        ),
        play_record("beta", 2, "fail", broken=["early_change"]),
    )

    write_reports(tmp_path, build(plays))

    text = (tmp_path / REPORT_MD_NAME).read_text(encoding="utf-8")
    failures = text[text.index("## Failures") :]
    assert "Fault, as the judge sees it: eval. the person agreed but did not mark a yes" in failures
    assert failures.count("Fault, as the judge sees it") == 1
