"""`evals/labels.py`: the labels file the project reads and never writes.

One line is one label: `case_id`, `play` (1 to 10), `check_id`, `question`, the
`conversation` it judged (turns with their tool calls and results), `answer`, `reason`,
`labelled_by` (`model` or `musa`) and `labelled_on`. A bad file stops the load with the line
number of the bad row. No network, no sleep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import LabelsInvalid
from app.evals.labels import OWNER_LABEL, load_labels

from .label_builders import label_document, write_labels

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = REPO_ROOT / "backend" / "evals" / "labels"


def test_a_label_row_loads_with_its_conversation(tmp_path: Path) -> None:
    """Catches a loader that drops the tool results from the conversation, or refuses one of
    the three answers: all ten fields must come back, the tool call with its args, result
    and error code."""
    path = write_labels(
        tmp_path / "labels.jsonl",
        label_document("case-001", 1, "reports_from_tool", answer="yes"),
        label_document("case-002", 2, "plain_words", answer="no", labelled_by="model"),
        label_document("case-003", 3, "plain_words", answer="does_not_apply"),
    )

    rows = load_labels(path)

    assert [r.answer for r in rows] == ["yes", "no", "does_not_apply"]
    assert [r.labelled_by for r in rows] == ["musa", "model", "musa"]
    first = rows[0]
    assert (first.case_id, first.play, first.check_id) == ("case-001", 1, "reports_from_tool")
    assert first.question.startswith("Check: reports_from_tool")
    assert first.reason == "The reply repeats the number the tool gave."
    assert str(first.labelled_on) == "2026-11-02"
    turn = first.conversation[0]
    assert turn.person == "Can I get the wiki editor role?"
    assert turn.reply == "I filed request REQ-0001."
    call = turn.tool_calls[0]
    assert call.name == "create_access_request"
    assert call.args == {"role": "wiki.editor"}
    assert call.ok is True
    assert call.error_code is None
    assert call.result == {"number": "REQ-0001"}


def test_the_ids_pass_at_their_limits(tmp_path: Path) -> None:
    """Catches a `check_id` cap set one too low: 64 characters loads, plays 1 and 10 load."""
    path = write_labels(
        tmp_path / "labels.jsonl",
        label_document("case-001", 1, "a" * 64),
        label_document("case-001", 10, "a" * 64),
    )

    assert [r.play for r in load_labels(path)] == [1, 10]


@pytest.mark.parametrize(
    ("second", "line", "needle"),
    [
        (label_document(extra_field="x"), 2, "extra_field"),
        (label_document(answer="maybe"), 2, "answer"),
        (label_document(labelled_by="claude"), 2, "labelled_by"),
        (label_document(play=0), 2, "play"),
        (label_document(play=11), 2, "play"),
        (label_document(check_id="a" * 65), 2, "check_id"),
        (label_document(check_id="Plain Words"), 2, "check_id"),
        (label_document(check_id="plain_words\n"), 2, "check_id"),
        (label_document(check_id="plain_words\r\n"), 2, "check_id"),
        (label_document(check_id="plain_words "), 2, "check_id"),
        (label_document(case_id="case-001", play=1, check_id="reports_from_tool"), 2, "duplicate"),
    ],
    ids=[
        "extra field",
        "answer maybe",
        "labelled_by claude",
        "play 0",
        "play 11",
        "check_id 65 characters",
        "check_id with a capital and a space",
        "check_id trailing newline",
        "check_id trailing CRLF",
        "check_id trailing space",
        "duplicate case, play and check",
    ],
)
def test_bad_rows_stop_the_load_with_their_line(
    tmp_path: Path, second: dict[str, object], line: int, needle: str
) -> None:
    """Catches a loader that skips a bad row, names the wrong line, or lets a key-like
    `check_id` through with a trailing newline, CRLF or space. The first row is good, so only
    line 2 can be named; the duplicate row names the SECOND line, not the first."""
    path = write_labels(tmp_path / "labels.jsonl", label_document(), second)

    with pytest.raises(LabelsInvalid) as raised:
        load_labels(path)

    message = str(raised.value)
    assert message.startswith(f"{path}:{line}: ")
    assert needle in message


def test_a_bad_first_line_is_line_one(tmp_path: Path) -> None:
    """Catches a line counter that starts at 0: the first row is line 1."""
    path = write_labels(tmp_path / "labels.jsonl", label_document(answer="maybe"))

    with pytest.raises(LabelsInvalid) as raised:
        load_labels(path)

    assert str(raised.value).startswith(f"{path}:1: ")


def test_the_same_check_may_repeat_across_cases_and_plays(tmp_path: Path) -> None:
    """Catches a duplicate rule that is too wide: the key is the three fields together, so
    another play, another case, or another check of the same case is not a duplicate."""
    path = write_labels(
        tmp_path / "labels.jsonl",
        label_document("case-001", 1, "plain_words"),
        label_document("case-001", 2, "plain_words"),
        label_document("case-002", 1, "plain_words"),
        label_document("case-001", 1, "reports_from_tool"),
    )

    assert len(load_labels(path)) == 4


def test_a_missing_file_is_an_error_and_an_empty_file_is_no_rows(tmp_path: Path) -> None:
    """Catches a missing file read as no labels (a typo in `--labels` would then calibrate
    nothing in silence), and an empty file refused (the shipped file is empty)."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert load_labels(empty) == ()
    with pytest.raises(LabelsInvalid):
        load_labels(tmp_path / "nope.jsonl")


def test_the_shipped_labels_file_loads_and_the_old_one_is_gone() -> None:
    """Catches a shipped row the owner did not label (only the owner's rows count in a
    calibration), a shipped file that no longer loads, and the old labels file left beside
    the new one."""
    rows = load_labels(LABELS_DIR / "labels.jsonl")
    assert rows, "the shipped labels file is empty"
    assert {row.labelled_by for row in rows} == {OWNER_LABEL}
    assert not (LABELS_DIR / "conversation-labels.jsonl").exists()
