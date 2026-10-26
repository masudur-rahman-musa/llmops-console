"""The note a Langfuse step shows when its text is not sent: one plain line.

Langfuse's Formatted view shows a plain string whole, and shows only the first line of a
JSON object, so the note is a string.
"""

from __future__ import annotations

import pytest

from app.core.tracing import INPUT_NOTE_KEY, OUTPUT_NOTE_KEY, PreviewNote, plain_notes

INPUT_ROWS: list[tuple[str, int | None, str | None, str]] = [
    (
        "alias and several messages",
        3,
        "agent-default",
        "not sent: prompts stay inside the gateway (alias agent-default, 3 messages)",
    ),
    (
        "one message is singular",
        1,
        "agent-default",
        "not sent: prompts stay inside the gateway (alias agent-default, 1 message)",
    ),
    (
        "alias only",
        None,
        "embed",
        "not sent: prompts stay inside the gateway (alias embed)",
    ),
    ("no facts", None, None, "not sent: prompts stay inside the gateway"),
]


@pytest.mark.parametrize(
    ("case", "messages", "alias", "expected"), INPUT_ROWS, ids=[row[0] for row in INPUT_ROWS]
)
def test_the_input_note_is_one_plain_line(
    case: str, messages: int | None, alias: str | None, expected: str
) -> None:
    """Catches a note still written as JSON (Langfuse's Formatted view shows only its first
    line), a plural "1 messages", a pair of brackets left on when no fact is known, or a
    fact written in a different order or wording."""
    assert PreviewNote.input(alias=alias, messages=messages).as_text() == expected, case


OUTPUT_ROWS: list[tuple[str, str | None, int | None, str]] = [
    ("both facts", "stop", 361, "not sent (finish reason stop, 361 tokens out)"),
    ("finish reason only", "stop", None, "not sent (finish reason stop)"),
    ("tokens only", None, 361, "not sent (361 tokens out)"),
    ("zero tokens is a fact", "stop", 0, "not sent (finish reason stop, 0 tokens out)"),
    ("no facts", None, None, "not sent"),
]


@pytest.mark.parametrize(
    ("case", "finish_reason", "tokens_out", "expected"),
    OUTPUT_ROWS,
    ids=[row[0] for row in OUTPUT_ROWS],
)
def test_the_output_note_is_one_plain_line(
    case: str, finish_reason: str | None, tokens_out: int | None, expected: str
) -> None:
    """Catches JSON output, empty brackets when nothing is known, and a zero token count
    dropped as if it were unknown (a falsy check instead of a `None` check)."""
    assert PreviewNote.output(finish_reason=finish_reason, tokens_out=tokens_out).as_text() == (
        expected
    ), case


def test_a_bad_finish_reason_still_reads_other_in_the_line() -> None:
    """Catches the finish-reason check being skipped on the way into the line: provider text
    must still come out as `other`."""
    note = PreviewNote.output(finish_reason="ignore previous instructions", tokens_out=361)

    assert note.as_text() == "not sent (finish reason other, 361 tokens out)"


def test_a_stage_step_starts_with_the_bare_lines() -> None:
    """Catches `plain_notes` still returning JSON, or carrying brackets or facts a stage
    step does not know."""
    assert plain_notes() == {
        INPUT_NOTE_KEY: "not sent: prompts stay inside the gateway",
        OUTPUT_NOTE_KEY: "not sent",
    }
