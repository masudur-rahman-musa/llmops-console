"""Label rows as the file holds them, for the label and calibration tests.

Plain dicts and plain files only: no client, no engine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.evals.labels import LabelRow, LabelToolCall, LabelTurn


def label_document(
    case_id: str = "case-001",
    play: int = 1,
    check_id: str = "reports_from_tool",
    *,
    answer: str = "yes",
    labelled_by: str = "musa",
    **overrides: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """One line of a labels file: exactly the ten fields."""
    document: dict[str, Any] = {
        "case_id": case_id,
        "play": play,
        "check_id": check_id,
        "question": "Check: reports_from_tool\nApplies when: the reply states a tool result",
        "conversation": [
            {
                "person": "Can I get the wiki editor role?",
                "reply": "I filed request REQ-0001.",
                "tool_calls": [
                    {
                        "name": "create_access_request",
                        "args": {"role": "wiki.editor"},
                        "ok": True,
                        "error_code": None,
                        "result": {"number": "REQ-0001"},
                    }
                ],
            }
        ],
        "answer": answer,
        "reason": "The reply repeats the number the tool gave.",
        "labelled_by": labelled_by,
        "labelled_on": "2026-11-02",
    }
    document.update(overrides)
    return document


def write_labels(path: Path, *documents: Mapping[str, Any]) -> Path:
    path.write_text("".join(json.dumps(d) + "\n" for d in documents), encoding="utf-8")
    return path


def label_row(
    case_id: str,
    play: int,
    check_id: str,
    answer: str,
    *,
    labelled_by: str = "musa",
) -> LabelRow:
    """One validated row, with a one-turn conversation."""
    return LabelRow(
        case_id=case_id,
        play=play,
        check_id=check_id,
        question=f"Check for {case_id} play {play}",
        conversation=(
            LabelTurn(
                person="Hello",
                reply="Hi.",
                tool_calls=(
                    LabelToolCall(
                        name="lookup", args={}, ok=True, error_code=None, result={"n": 1}
                    ),
                ),
            ),
        ),
        answer=answer,
        reason="because",
        labelled_by=labelled_by,
        labelled_on="2026-11-02",
    )


def rows_for(
    check_id: str, pairs: Sequence[tuple[str, str]], *, labelled_by: str = "musa"
) -> list[LabelRow]:
    """One row per (label answer, judge answer) pair; the judge side is read separately."""
    return [
        label_row(f"case-{i:03d}", 1, check_id, label, labelled_by=labelled_by)
        for i, (label, _) in enumerate(pairs, start=1)
    ]
