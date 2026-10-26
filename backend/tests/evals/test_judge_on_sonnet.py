"""The judge on Sonnet: shipped files only, no model call, no wait."""

from __future__ import annotations

import re
from pathlib import Path

from app.evals.calibration import load_labels

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent
LABELS_DIR = BACKEND_DIR / "evals" / "labels"
MAKEFILE = REPO_DIR / "Makefile"

MIN_LABELS = 40


def test_the_haiku_calibration_no_longer_ships() -> None:
    """Catches a tree that still ships the calibration made on Haiku: the gate would
    trust a Sonnet judge on a Haiku score, instead of reporting the score and letting
    it decide nothing until the owner calibrates again."""
    assert not (LABELS_DIR / "calibration.json").exists()


def test_the_eval_comment_counts_45_cases_played_eval_repeats_times() -> None:
    """Catches a Makefile that keeps the old 33 count in the comment or in the `eval:`
    help line, or whose comment above `eval:` does not say 45 and name EVAL_REPEATS."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    eval_index = next(i for i, line in enumerate(lines) if line.startswith("eval:"))
    start = next(i for i, line in enumerate(lines) if line.startswith("# The eval gate"))
    comment = "\n".join(line for line in lines[start:eval_index] if line.startswith("#"))
    help_line = lines[eval_index]

    assert "33 conversation" not in "\n".join(lines)
    assert re.search(r"\b45\b", comment)
    assert "EVAL_REPEATS" in comment
    assert re.search(r"\b45\b", help_line)


def test_the_labels_file_still_ships() -> None:
    """Catches a change that deletes the labels along with the old calibration: the
    owner's calibrate run on Sonnet would have no input."""
    path = LABELS_DIR / "conversation-labels.jsonl"

    assert path.exists()
    assert len(load_labels(path)) >= MIN_LABELS
