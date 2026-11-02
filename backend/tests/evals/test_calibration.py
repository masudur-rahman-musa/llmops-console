"""`evals/calibration.py`, `JudgeClient.label` and the trust a run reads from `calibration.json`.

The judge is asked about the rows the owner labelled (`labelled_by` is `musa`), one ask per
row, and its answers are counted per check type with Cohen's kappa. A fake judge answers by
question text, so no test needs a gateway for the counting; the gateway tests use an
`httpx.MockTransport`. No network, no sleep.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.errors import CalibrationInvalid
from app.evals.calibration import (
    TRUSTED_KAPPA,
    Calibration,
    calibrate,
    calibration_json,
    cohens_kappa,
    is_trusted,
    judge_model_from_routing,
)
from app.evals.judge import (
    CheckAnswer,
    JudgeClient,
    prompt_text,
    render_conversation,
    render_label_conversation,
)
from app.evals.labels import LabelRow, LabelToolCall, LabelTurn
from app.evals.report import build_report
from app.evals.results import trusted_check_types

from .builders import call, check_record, play_record, played
from .label_builders import label_row

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTING = REPO_ROOT / "backend" / "config" / "routing.yaml"
GATEWAY = "https://gateway.example"
PROMPT_HASH = "p" * 64
LABELS_HASH = "l" * 64
MADE_ON = date(2026, 11, 2)


class AnswerByQuestion:
    """A label judge that answers from a table keyed by the row's question, and keeps every
    question it was asked. A question not in the table is unanswered (`None`)."""

    def __init__(self, answers: Mapping[str, str | None]) -> None:
        self._answers = dict(answers)
        self.asked: list[str] = []

    def label(self, question: str, conversation: Sequence[LabelTurn]) -> CheckAnswer | None:
        self.asked.append(question)
        word = self._answers.get(question)
        if word is None:
            return None
        return CheckAnswer(answer=word, quote="", reason="r")


def run(rows: Sequence[LabelRow], judge: AnswerByQuestion) -> Calibration:
    return calibrate(
        rows,
        judge,
        judge_model="anthropic/claude-sonnet-5-5",
        judge_prompt_sha256=PROMPT_HASH,
        labels_sha256=LABELS_HASH,
        made_on=MADE_ON,
    )


def rows_and_judge(
    check_id: str, pairs: Sequence[tuple[str, str | None]], *, start: int = 1
) -> tuple[list[LabelRow], dict[str, str | None]]:
    """One musa row per (label, judge answer) pair, and the judge table that gives the answer."""
    rows: list[LabelRow] = []
    table: dict[str, str | None] = {}
    for offset, (label, judged) in enumerate(pairs):
        row = label_row(f"{check_id}-{start + offset:03d}", 1, check_id, label)
        rows.append(row)
        table[row.question] = judged
    return rows, table


# --- kappa ----------------------------------------------------------------------------


def test_unweighted_kappa_over_the_three_answers() -> None:
    """Catches a weighted kappa (the three answers have no order, so a quadratic weight would
    score yes against does_not_apply as a near miss) and a kappa that is not rounded to four
    decimals. Four pairs where agreement is 0.75 and chance agreement is 0.3125 give
    0.636363..., stored as 0.6364."""
    pairs = [
        ("yes", "yes"),
        ("no", "no"),
        ("does_not_apply", "does_not_apply"),
        ("does_not_apply", "yes"),
    ]

    rows, table = rows_and_judge("plain_words", pairs)
    result = run(rows, AnswerByQuestion(table)).check_types["plain_words"]

    assert result.kappa == 0.6364
    assert cohens_kappa(pairs) == pytest.approx(0.6363636, abs=1e-6)


def test_total_chance_agreement_is_not_trusted() -> None:
    """Catches a division by zero, or a kappa of 1.0 when every label and every answer is
    `yes`: chance agreement is 1, so there is no kappa and the type is not trusted."""
    pairs = [("yes", "yes")] * 5

    rows, table = rows_and_judge("plain_words", pairs)
    result = run(rows, AnswerByQuestion(table)).check_types["plain_words"]

    assert cohens_kappa(pairs) is None
    assert result.kappa is None
    assert result.trusted is False
    assert result.agreed == 5


def test_no_pairs_has_no_kappa() -> None:
    """Catches a kappa of 0 or a crash with nothing to compare: a type the judge answered
    nothing for has no kappa and is not trusted."""
    assert cohens_kappa([]) is None
    assert is_trusted(cohens_kappa([])) is False


def test_kappa_per_check_type() -> None:
    """Catches one kappa over all the rows (the old way): two types in one calibration must
    be counted apart. `reports_from_tool` has 5 yes and 5 no labels and the judge differs on
    one yes: 10 labels, 9 agreed, kappa 0.8, trusted. `plain_words` is yes,yes,no,no against
    yes,no,yes,no: kappa 0.0, not trusted. Pooled, the two would give a different number."""
    tool_rows, tool_table = rows_and_judge(
        "reports_from_tool",
        [("yes", "yes")] * 4 + [("yes", "no")] + [("no", "no")] * 5,
    )
    words_rows, words_table = rows_and_judge(
        "plain_words",
        [("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")],
    )

    calibration = run([*tool_rows, *words_rows], AnswerByQuestion({**tool_table, **words_table}))

    tool = calibration.check_types["reports_from_tool"]
    words = calibration.check_types["plain_words"]
    assert (tool.labels, tool.answered, tool.agreed, tool.kappa, tool.trusted) == (
        10,
        10,
        9,
        0.8,
        True,
    )
    assert (words.labels, words.answered, words.agreed, words.kappa, words.trusted) == (
        4,
        4,
        2,
        0.0,
        False,
    )
    assert set(calibration.check_types) == {"reports_from_tool", "plain_words"}


def test_trusted_starts_at_0_7() -> None:
    """Catches a bar at the wrong place: 0.7 is trusted (`>=`, not `>`), 0.699 is not, and no
    kappa is not."""
    assert TRUSTED_KAPPA == 0.7
    assert is_trusted(0.7) is True
    assert is_trusted(0.699) is False
    assert is_trusted(None) is False


def test_a_type_at_exactly_0_7_is_trusted_in_the_calibration() -> None:
    """Catches trust decided with `>` instead of `>=`: 20 pairs with 17 agreeing and a chance
    agreement of 0.5 give a kappa of exactly 0.7, which is trusted."""
    pairs = [("yes", "yes")] * 9 + [("yes", "no")] * 2 + [("no", "no")] * 8 + [("no", "yes")] * 1
    assert cohens_kappa(pairs) == pytest.approx(0.7)

    rows, table = rows_and_judge("plain_words", pairs)
    result = run(rows, AnswerByQuestion(table)).check_types["plain_words"]

    assert result.kappa == 0.7
    assert result.trusted is True


def test_an_unanswered_check_is_dropped() -> None:
    """Catches an unanswered check counted as a disagreement or as a `no`: 4 labels with 1
    unanswered give labels 4, answered 3, and the pair is left out of agreed and kappa."""
    rows, table = rows_and_judge(
        "plain_words", [("yes", "yes"), ("no", "no"), ("yes", "yes"), ("no", None)]
    )

    result = run(rows, AnswerByQuestion(table)).check_types["plain_words"]

    assert (result.labels, result.answered, result.agreed) == (4, 3, 3)
    assert result.kappa == 1.0


def test_only_rows_labelled_by_musa_are_asked() -> None:
    """Catches a calibration that counts the model's own labels: with 3 `model` rows and 4
    `musa` rows the judge is asked 4 times, about the musa rows only, and the labels counted
    are 4."""
    musa, table = rows_and_judge("plain_words", [("yes", "yes")] * 4, start=1)
    model = [
        label_row(f"model-{i}", 1, "plain_words", "yes", labelled_by="model") for i in range(3)
    ]
    judge = AnswerByQuestion({**table, **{r.question: "yes" for r in model}})

    calibration = run([*model, *musa], judge)

    assert len(judge.asked) == 4
    assert sorted(judge.asked) == sorted(r.question for r in musa)
    assert sum(t.labels for t in calibration.check_types.values()) == 4


def test_calibrate_with_no_musa_rows_is_refused() -> None:
    """Catches a calibration that writes an empty result when nobody has labelled anything:
    only model rows (or no rows) raise, and the judge is never asked."""
    judge = AnswerByQuestion({})
    only_model = [label_row("c", 1, "plain_words", "yes", labelled_by="model")]

    with pytest.raises(CalibrationInvalid):
        run(only_model, judge)
    with pytest.raises(CalibrationInvalid):
        run([], judge)
    assert judge.asked == []


# --- the file ---------------------------------------------------------------------------


def test_calibration_json_names_the_judge_model_and_both_hashes() -> None:
    """Catches a file that leaves out the judge model or either hash, or one that is not
    indented and newline-ended: the file shows which judge and which prompt and labels the
    numbers belong to."""
    rows, table = rows_and_judge("plain_words", [("yes", "yes"), ("no", "no")])
    calibration = run(rows, AnswerByQuestion(table))

    text = calibration_json(calibration)
    document = json.loads(text)

    assert text.endswith("\n")
    assert "\n  " in text
    assert document["judge_model"] == "anthropic/claude-sonnet-5-5"
    assert document["judge_prompt_sha256"] == PROMPT_HASH
    assert document["labels_sha256"] == LABELS_HASH
    assert document["made_on"] == "2026-11-02"
    assert document["check_types"]["plain_words"] == {
        "labels": 2,
        "answered": 2,
        "agreed": 2,
        "kappa": 1.0,
        "trusted": True,
    }


def test_the_judge_model_is_the_first_candidate_of_the_judge_alias() -> None:
    """Catches a judge model typed in code, the second candidate, or the wrong alias: it is
    read from the routing file as `provider/model`."""
    assert judge_model_from_routing(ROUTING) == "anthropic/claude-sonnet-5-5"


# --- what a run trusts ------------------------------------------------------------------


def write_calibration(path: Path, prompt_hash: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "judge_model": "anthropic/claude-sonnet-5-5",
                "judge_prompt_sha256": prompt_hash,
                "labels_sha256": LABELS_HASH,
                "made_on": "2026-11-02",
                "check_types": {
                    "cites_policy": {
                        "labels": 4,
                        "answered": 4,
                        "agreed": 4,
                        "kappa": 1.0,
                        "trusted": True,
                    },
                    "same_language": {
                        "labels": 4,
                        "answered": 4,
                        "agreed": 2,
                        "kappa": 0.0,
                        "trusted": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_run_trusts_only_a_calibration_of_the_same_judge_prompt(tmp_path: Path) -> None:
    """Catches a run that trusts a calibration made with an older judge prompt, or one that
    trusts nothing when the hash matches. Matching hash: only the trusted type. Another hash,
    no file, no path: nothing, so reply quality is empty."""
    path = write_calibration(tmp_path / "calibration.json", "current")
    plays = [
        play_record(
            "case-001",
            1,
            "pass",
            checks=[
                check_record("cites_policy", "yes"),
                check_record("same_language", "no"),
            ],
        )
    ]

    def reply_quality(trusted: frozenset[str]) -> Any:  # noqa: ANN401
        report = build_report(
            plays,
            started_at=datetime(2026, 11, 2, 9, 0, tzinfo=UTC),
            finished_at=datetime(2026, 11, 2, 9, 5, tzinfo=UTC),
            set_name="cases.jsonl",
            set_sha="a" * 40,
            set_sha_source="commit",
            run_kind="ci",
            git_sha=None,
            tenant="all",
            target="https://helpdesk.example",
            trusted_check_types=trusted,
        )
        return report.metrics["reply_quality"]

    same = trusted_check_types(path, judge_prompt_sha256="current")
    assert same == frozenset({"cites_policy"})
    assert reply_quality(same) == 100.0
    stale = trusted_check_types(path, judge_prompt_sha256="other")
    assert stale == frozenset()
    assert reply_quality(stale) is None
    assert trusted_check_types(tmp_path / "nope.json", judge_prompt_sha256="current") == frozenset()
    assert trusted_check_types(None, judge_prompt_sha256="current") == frozenset()


# --- asking the judge about a label's conversation -----------------------------------------


def label_turns() -> tuple[LabelTurn, ...]:
    return (
        LabelTurn(
            person="Can I get the wiki editor role?",
            reply="Shall I file it?",
            tool_calls=(
                LabelToolCall(
                    name="search_policy",
                    args={"q": "wiki"},
                    ok=True,
                    error_code=None,
                    result={"hits": ["tool-result-marker-7"]},
                ),
            ),
        ),
        LabelTurn(person="Yes.", reply="Filed.", tool_calls=()),
    )


def test_a_label_conversation_renders_like_a_played_one_without_world_facts() -> None:
    """Catches a label rendering with different tags or tool lines than the run judge's own
    (the calibration would then test a different prompt than the run uses), or one that
    invents world facts the label does not carry."""
    turns = label_turns()
    same_played = [
        played(
            "Can I get the wiki editor role?",
            "Shall I file it?",
            call("search_policy", {"q": "wiki"}, result={"hits": ["tool-result-marker-7"]}),
        ),
        played("Yes.", "Filed."),
    ]
    expected = render_conversation(same_played).split("\nWorld facts before")[0]

    rendered = render_label_conversation(turns)

    assert rendered == expected
    assert rendered.startswith("<conversation>")
    assert rendered.endswith("</conversation>")
    assert "World facts" not in rendered


def test_the_judge_is_asked_about_a_label_as_one_request() -> None:
    """Catches a label question sent to the wrong alias, with the conversation outside the
    tags, or with an instruction in place of the prompt file: the alias is `judge`, the
    system message is the judge prompt, and the user message is the row's question and the
    conversation between the tags."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = json.dumps({"answer": "yes", "quote": "q", "reason": "r"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = JudgeClient(
        GATEWAY, client=httpx.Client(transport=httpx.MockTransport(handler)), api_key="k"
    )

    result = judge.label("Check: x\nApplies when: y", label_turns())

    assert result == CheckAnswer(answer="yes", quote="q", reason="r")
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "judge"
    user = body["messages"][1]["content"]
    assert user.startswith("Check: x\nApplies when: y")
    assert render_label_conversation(label_turns()) in user
    assert "tool-result-marker-7" in user
    assert "World facts" not in user
    assert body["messages"][0]["content"] == prompt_text()


def test_a_judge_that_cannot_be_used_leaves_the_label_unanswered() -> None:
    """Catches a label question that raises or guesses on a gateway failure: it is tried twice
    and then unanswered."""
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(500)

    judge = JudgeClient(
        GATEWAY, client=httpx.Client(transport=httpx.MockTransport(handler)), api_key="k"
    )

    assert judge.label("Check: x", label_turns()) is None
    assert len(seen) == 2
