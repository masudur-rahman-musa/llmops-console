"""`evals/calibration.py`: two-class kappa, the trust floor and the 40-label floor.

Plain objects: the judge is a fake with a `grade(rubric, reply)` method, so no request is
made. Labels are `ConversationLabel` records (the eight-field line of
`backend/evals/labels/conversation-labels.jsonl`).

Contract these tests are written against:

* constants `MIN_CALIBRATION_LABELS = 40`, `JUDGE_TRUST_MIN = 0.7`,
  `CALIBRATION_JSON_NAME = "calibration.json"`.
* `ConversationLabel(case_id, turn, rubric, reply, verdict, votes, reason, source)`:
  frozen, `extra="forbid"`; `verdict` and each vote are `"pass"` or `"fail"`, `votes` has
  exactly 3 entries, `source` is `"model-reference"` or `"human"`.
* `load_labels(path) -> tuple[ConversationLabel, ...]`, raising `CalibrationSetInvalid`.
* `cohen_kappa(labels, judged) -> float | None`: unweighted, two classes; `None` when chance
  agreement is total (kappa is undefined).
* `is_trusted(kappa) -> bool`: `kappa >= JUDGE_TRUST_MIN`; `None` is not trusted.
* `calibrate(labels, *, judge, labels_name, labels_sha) -> CalibrationReport` where `judge` is a
  `judge.Grader` (a Protocol with `grade(rubric, reply) -> JudgeVerdict | None`), with
  `kappa`, `pairs`, `trusted`, `judge_prompt_version`, `labels_name`, `labels_sha`;
  raises `CalibrationRefused` when fewer than 40 labels were judged, BEFORE any request is
  made when the file itself holds fewer than 40.
* `write_calibration(out_dir, report)` writes `out_dir / "calibration.json"`;
  `read_calibration(path) -> CalibrationReport`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from app.evals.calibration import (
    CALIBRATION_JSON_NAME,
    JUDGE_TRUST_MIN,
    MIN_CALIBRATION_LABELS,
    CalibrationRefused,
    CalibrationSetInvalid,
    ConversationLabel,
    calibrate,
    cohen_kappa,
    is_trusted,
    load_labels,
    read_calibration,
    write_calibration,
)
from app.evals.judge import JudgeVerdict, judge_prompt_version

Verdict = Literal["pass", "fail"]


def label(index: int, verdict: Verdict) -> ConversationLabel:
    return ConversationLabel(
        case_id=f"conv-{index % 33 + 1:03d}",
        turn=1,
        rubric=f"rubric {index}",
        reply=f"reply {index}",
        verdict=verdict,
        votes=(verdict, verdict, verdict),
        reason="because",
        source="model-reference",
    )


class ScriptedJudge:
    """Answers per reply text from a map, and counts the calls. Unknown replies get None."""

    def __init__(self, answers: dict[str, Verdict | None]) -> None:
        self.answers = answers
        self.calls = 0

    def grade(self, rubric: str, reply: str) -> JudgeVerdict | None:
        self.calls += 1
        verdict = self.answers.get(reply)
        if verdict is None:
            return None
        return JudgeVerdict(verdict=verdict, votes=(verdict,) * 3, reasons=("r", "r", "r"))


def table(
    both_pass: int, label_pass_only: int, judge_pass_only: int, both_fail: int
) -> tuple[list[ConversationLabel], ScriptedJudge]:
    """Labels and the judge's matching answers for a 2x2 table of counts."""
    p: Verdict = "pass"
    f: Verdict = "fail"
    rows: list[tuple[Verdict, Verdict]] = (
        [(p, p)] * both_pass
        + [(p, f)] * label_pass_only
        + [(f, p)] * judge_pass_only
        + [(f, f)] * both_fail
    )
    labels = [label(i, truth) for i, (truth, _) in enumerate(rows)]
    judge = ScriptedJudge({f"reply {i}": judged for i, (_, judged) in enumerate(rows)})
    return labels, judge


def test_the_floors_are_pinned() -> None:
    """Catches a trust floor of 0.75 or 0.65 and a label floor of 39 or 41: the owner set
    0.7 and 40, and the judge model is chosen by them."""
    assert MIN_CALIBRATION_LABELS == 40
    assert JUDGE_TRUST_MIN == 0.7
    assert CALIBRATION_JSON_NAME == "calibration.json"


def test_two_class_kappa_and_the_trust_floor() -> None:
    """Catches a weighted or ordinal kappa and a plain percent-agreement. The table is
    18 both pass, 2 label-only pass, 3 judge-only pass, 17 both fail: agreement 35/40 =
    0.875, chance (20*21 + 20*19)/1600 = 0.5, so kappa is exactly 0.75. Percent agreement
    would give 0.875, so only the chance correction gives 0.75."""
    labels, judge = table(18, 2, 3, 17)

    report = calibrate(labels, judge=judge, labels_name="l.jsonl", labels_sha="a" * 64)

    assert report.kappa == pytest.approx(0.75, abs=1e-12)
    assert report.pairs == 40
    assert report.trusted is True
    assert report.judge_prompt_version == judge_prompt_version()


def test_kappa_on_a_plain_list_pair() -> None:
    """Catches an inverted sign or a swapped pair of arguments: full agreement is 1.0, full
    disagreement on a balanced table is -1.0."""
    assert cohen_kappa(["pass", "fail"] * 10, ["pass", "fail"] * 10) == pytest.approx(1.0)
    assert cohen_kappa(["pass", "fail"] * 10, ["fail", "pass"] * 10) == pytest.approx(-1.0)


def test_kappa_is_none_when_chance_agreement_is_total() -> None:
    """Catches a divide by zero: when every label and every answer is pass, chance agreement
    is 1 and kappa is undefined. It must be none and not trusted, never 1.0 or 0.0."""
    assert cohen_kappa(["pass"] * 40, ["pass"] * 40) is None
    assert is_trusted(None) is False


@pytest.mark.parametrize(
    ("kappa", "trusted"), [(0.7, True), (0.71, True), (0.69, False), (0.0, False)]
)
def test_the_trust_floor_is_inclusive_at_seven_tenths(kappa: float, trusted: bool) -> None:
    """Catches `>` instead of `>=` (0.7 would not be trusted) and a floor drifted to 0.69
    (0.69 would be trusted)."""
    assert is_trusted(kappa) is trusted


def test_fewer_than_forty_labels_are_refused_before_any_request() -> None:
    """Catches a calibration that runs on 39 labels (a kappa over too few pairs reads as
    trust) and one that refuses only after paying for the judge: the file holds 39, so no
    judge call may be made."""
    labels, judge = table(20, 0, 0, 19)

    with pytest.raises(CalibrationRefused):
        calibrate(labels, judge=judge, labels_name="l.jsonl", labels_sha="a" * 64)

    assert judge.calls == 0


def test_exactly_forty_labels_are_enough() -> None:
    """Catches an off-by-one at the floor: 40 labels calibrate."""
    labels, judge = table(20, 0, 0, 20)

    report = calibrate(labels, judge=judge, labels_name="l.jsonl", labels_sha="a" * 64)

    assert report.pairs == 40


def test_labels_the_judge_could_not_grade_do_not_count_toward_the_floor() -> None:
    """Catches a calibration that counts a label the judge never answered: 40 labels but one
    unusable judge answer leaves 39 pairs, which is refused."""
    labels, judge = table(20, 0, 0, 20)
    judge.answers["reply 0"] = None

    with pytest.raises(CalibrationRefused):
        calibrate(labels, judge=judge, labels_name="l.jsonl", labels_sha="a" * 64)


def test_the_calibration_file_round_trips(tmp_path: Path) -> None:
    """Catches an artifact the gate cannot read back, or one written under another name:
    the gate and the report read `calibration.json` for the kappa and the trust flag."""
    labels, judge = table(18, 2, 3, 17)
    report = calibrate(labels, judge=judge, labels_name="l.jsonl", labels_sha="b" * 64)

    write_calibration(tmp_path, report)
    read_back = read_calibration(tmp_path / CALIBRATION_JSON_NAME)

    assert read_back == report
    assert json.loads((tmp_path / CALIBRATION_JSON_NAME).read_text())["kappa"] == pytest.approx(
        0.75
    )


def write_label_lines(path: Path, *documents: dict[str, object]) -> Path:
    path.write_text("\n".join(json.dumps(d) for d in documents) + "\n", encoding="utf-8")
    return path


GOOD_LINE: dict[str, object] = {
    "case_id": "conv-001",
    "turn": 1,
    "rubric": "r",
    "reply": "x",
    "verdict": "pass",
    "votes": ["pass", "pass", "pass"],
    "reason": "because",
    "source": "model-reference",
}


def test_a_good_label_line_loads_with_all_eight_fields(tmp_path: Path) -> None:
    """Catches a loader that drops `votes` or `source`: a person's later `human` label must
    stay distinguishable from the placeholder."""
    loaded = load_labels(write_label_lines(tmp_path / "l.jsonl", GOOD_LINE))

    assert loaded[0].model_dump(mode="json") == GOOD_LINE


@pytest.mark.parametrize(
    "change",
    [
        {"verdict": "maybe"},
        {"verdict": 4},
        {"votes": ["pass", "pass"]},
        {"votes": ["pass", "pass", "maybe"]},
        {"source": "intern"},
        {"extra": 1},
        {"case_id": "conv-12"},
        {"turn": 0},
    ],
    ids=["verdict word", "score", "two votes", "vote word", "source", "extra", "case id", "turn"],
)
def test_a_bad_label_line_is_refused(tmp_path: Path, change: dict[str, object]) -> None:
    """Catches the old 1-to-5 format loading as a label, a label with an invented source, and
    a loader that accepts extra or malformed fields: each row changes exactly one field of a
    good line."""
    path = write_label_lines(tmp_path / "l.jsonl", {**GOOD_LINE, **change})

    with pytest.raises(CalibrationSetInvalid):
        load_labels(path)
