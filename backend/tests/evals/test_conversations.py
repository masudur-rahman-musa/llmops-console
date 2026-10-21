"""`evals/conversations.py`: the case model, the loader and the typed id check.

Plain files under `tmp_path`, no network. The file shape is the help desk's own
`backend/evals/conversations/cases.jsonl`: `id`, `category`, `tenant`,
`principal_user`, `turns`, `expected` (`tool_calls`, `confirm_turn`) and `graded`.

Contract these tests are written against:

* `CASE_ID_PATTERN` (`conv-` and exactly three digits, matched whole), `MIN_TURNS = 1`,
  `MAX_TURNS = 10`.
* `ConversationCase`, `Expected`, `ExpectedCall`, `GradedTurn`: frozen, `extra="forbid"`.
* `load_cases(path) -> tuple[ConversationCase, ...]`, raising `ConversationSetInvalid`
  (with `.line`, the 1-based file line, or `None` when the whole file is at fault).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.evals.conversations import (
    CASE_ID_PATTERN,
    MAX_TURNS,
    MIN_TURNS,
    ConversationSetInvalid,
    load_cases,
)

from .builders import case_document, write_cases


def test_the_published_numbers_are_pinned() -> None:
    """Catches a turn cap of 11 or 9, or an id pattern of two or four digits: the help desk
    route refuses `conv-12` and more than 10 turns, so a looser loader would pass a file the
    route then rejects after the reset has been paid for."""
    assert MIN_TURNS == 1
    assert MAX_TURNS == 10
    assert CASE_ID_PATTERN.pattern == r"conv-[0-9]{3}"


def test_the_case_loader_reads_every_field(tmp_path: Path) -> None:
    """Catches a loader that drops `confirm_turn`, `args_subset` or the `graded` list, which
    would let every change-before-yes and tool-order check pass over nothing. A 2-case file
    with different values in every field, so a loader that copies one case's value onto the
    other is visible."""
    first = case_document(
        "conv-001",
        turns=("Set me up with the wiki", "yes"),
        tool_calls=({"name": "create_access_request", "args_subset": {"role": "wiki.viewer"}},),
        confirm_turn=2,
        graded=((1, "ask first?"), (2, "gives the number?")),
        tenant="nordlicht",
        principal_user="seed:production.employee.5",
        category="paraphrases",
    )
    second = case_document(
        "conv-002",
        turns=("What is the password rule?",),
        tool_calls=(
            {"name": "search_policy", "args_subset": {}},
            {"name": "list_my_requests", "args_subset": {"status": "open"}},
        ),
        confirm_turn=None,
        graded=((1, "cites a policy?"),),
        tenant="cuan",
        principal_user="seed:finance.employee.5",
        category="typos",
    )

    cases = load_cases(write_cases(tmp_path / "cases.jsonl", first, second))

    assert [c.id for c in cases] == ["conv-001", "conv-002"]
    one, two = cases
    assert (one.category, one.tenant, one.principal_user) == (
        "paraphrases",
        "nordlicht",
        "seed:production.employee.5",
    )
    assert one.turns == ("Set me up with the wiki", "yes")
    assert [(c.name, c.args_subset) for c in one.expected.tool_calls] == [
        ("create_access_request", {"role": "wiki.viewer"})
    ]
    assert one.expected.confirm_turn == 2
    assert [(g.turn, g.rubric) for g in one.graded] == [(1, "ask first?"), (2, "gives the number?")]
    assert (two.category, two.tenant, two.principal_user) == (
        "typos",
        "cuan",
        "seed:finance.employee.5",
    )
    assert [(c.name, c.args_subset) for c in two.expected.tool_calls] == [
        ("search_policy", {}),
        ("list_my_requests", {"status": "open"}),
    ]
    assert two.expected.confirm_turn is None
    assert two.turns == ("What is the password rule?",)


@pytest.mark.parametrize("case_id", ["conv-000", "conv-001", "conv-999"])
def test_ids_inside_the_pattern_load(tmp_path: Path, case_id: str) -> None:
    """Catches a pattern that refuses the inclusive ends (`conv-000`, `conv-999`)."""
    cases = load_cases(write_cases(tmp_path / "c.jsonl", case_document(case_id)))

    assert cases[0].id == case_id


@pytest.mark.parametrize(
    "case_id",
    [
        "conv-12",
        "conv-1000",
        "conv-001\n",
        "conv-001\r\n",
        "conv-001 ",
        "Conv-001",
        "conv_001",
        "",
    ],
)
def test_ids_outside_the_pattern_are_refused(tmp_path: Path, case_id: str) -> None:
    """Catches a pattern matched with `.match` and a `$` (which lets `conv-001\\n` through),
    one that allows a fourth digit or only two, and one that is case-blind. Each row differs
    from a good id in exactly one way, so only the id check can refuse it."""
    path = write_cases(tmp_path / "c.jsonl", case_document(case_id))

    with pytest.raises(ConversationSetInvalid):
        load_cases(path)


@pytest.mark.parametrize("count", [MIN_TURNS, MAX_TURNS])
def test_turn_counts_at_the_limits_load(tmp_path: Path, count: int) -> None:
    """Catches an off-by-one that refuses 1 or 10 turns, the two limits the route allows."""
    turns = [f"message {n}" for n in range(count)]

    cases = load_cases(write_cases(tmp_path / "c.jsonl", case_document(turns=turns)))

    assert len(cases[0].turns) == count


@pytest.mark.parametrize("count", [0, MAX_TURNS + 1])
def test_turn_counts_one_outside_the_limits_are_refused(tmp_path: Path, count: int) -> None:
    """Catches a loader with no turn cap (the route would refuse the case mid-run) and one
    that accepts an empty conversation, which no check could ever fail."""
    turns = [f"message {n}" for n in range(count)]
    path = write_cases(tmp_path / "c.jsonl", case_document(turns=turns))

    with pytest.raises(ConversationSetInvalid):
        load_cases(path)


def _with(document: dict[str, Any], where: str, extra: str) -> dict[str, Any]:
    target = document if where in ("case", "tool_call", "graded") else document[where]
    if where == "tool_call":
        target = document["expected"]["tool_calls"][0]
    if where == "graded":
        target = document["graded"][0]
    target[extra] = 1
    return document


@pytest.mark.parametrize("where", ["case", "expected", "tool_call", "graded"])
def test_an_unknown_field_is_refused_at_every_level(tmp_path: Path, where: str) -> None:
    """Catches a loader that forbids extras only at the top: a misspelt `confirm_turns` inside
    `expected` would load cleanly and the change-before-yes check would silently stop
    applying. The base document is valid, so the extra field is the only thing that can
    refuse it."""
    document = case_document(
        tool_calls=({"name": "search_policy", "args_subset": {}},), confirm_turn=None
    )
    path = write_cases(tmp_path / "c.jsonl", _with(document, where, "surprise"))

    with pytest.raises(ConversationSetInvalid):
        load_cases(path)


def test_a_duplicate_id_is_refused_and_names_the_second_line(tmp_path: Path) -> None:
    """Catches a loader that lets two cases share an id: the saved results are keyed by id, so
    the second case would be skipped on a resume as if it had been played."""
    path = write_cases(
        tmp_path / "c.jsonl",
        case_document("conv-001"),
        case_document("conv-002"),
        case_document("conv-001"),
    )

    with pytest.raises(ConversationSetInvalid) as raised:
        load_cases(path)

    assert raised.value.line == 3


def test_a_bad_line_names_its_file_line(tmp_path: Path) -> None:
    """Catches an error that cannot be found in an editor: a blank line before the bad one
    still counts, so a loader numbering only the non-blank lines reports line 2, not 3."""
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps(case_document("conv-001")) + "\n\n" + json.dumps({"id": "conv-002"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConversationSetInvalid) as raised:
        load_cases(path)

    assert raised.value.line == 3


def test_a_missing_or_empty_file_is_refused(tmp_path: Path) -> None:
    """Catches a run over zero cases: it has no failure, so it would pass the gate."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ConversationSetInvalid):
        load_cases(tmp_path / "missing.jsonl")
    with pytest.raises(ConversationSetInvalid):
        load_cases(empty)
