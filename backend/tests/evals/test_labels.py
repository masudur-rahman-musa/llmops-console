"""`evals/labels.py`: the label maker. `agent-strong` grades each reply three times; only
unanimous replies become reference labels; up to 40.

Plain objects: the grader is a fake, so no request is made, except one case that drives the
real gateway grader through `httpx.MockTransport`.

Contract these tests are written against:

* constants `LABEL_ALIAS = "agent-strong"`, `LABEL_VOTES = 3`, `LABEL_TARGET = 40`,
  `LABEL_SOURCE = "model-reference"`.
* `LabelVote(verdict, reason)`; `LabelCandidate(case_id, turn, rubric, reply, conversation)`
  where `conversation` is the text the maker sees (every user turn, every reply, every tool
  call with its arguments and result).
* `candidates_from(case, answer) -> list[LabelCandidate]`: one per `graded` entry, `turn` and
  `rubric` copied from it, `reply` word for word from the answer.
* `make_labels(candidates, grader, *, limit=LABEL_TARGET) -> tuple[ConversationLabel, ...]`
  where `grader.grade(candidate)` returns the votes (3 when it worked; fewer when a sample
  was unusable). A reply is kept only when exactly 3 votes agree. `candidates` is consumed
  lazily and no more is graded once `limit` labels are kept.
* `write_labels(path, labels)` writes one JSON line per label, in file order, and refuses
  (`FileExistsError`) to overwrite an existing file: the labels are made once.
* `GatewayLabelGrader(base_url, *, client, api_key)`: the real grader, three requests per
  reply to `/v1/chat/completions` naming the alias `agent-strong`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import httpx
import pytest

from app.evals.labels import (
    LABEL_ALIAS,
    LABEL_SOURCE,
    LABEL_TARGET,
    LABEL_VOTES,
    GatewayLabelGrader,
    LabelCandidate,
    LabelVote,
    candidates_from,
    make_labels,
    write_labels,
)

from .builders import answer, call, make_case, turn

Verdict = Literal["pass", "fail"]

EIGHT_FIELDS = {"case_id", "turn", "rubric", "reply", "verdict", "votes", "reason", "source"}


def candidate(index: int) -> LabelCandidate:
    return LabelCandidate(
        case_id=f"conv-{index % 33 + 1:03d}",
        turn=1,
        rubric=f"rubric {index}",
        reply=f"reply {index}",
        conversation=f"whole conversation {index}",
    )


class Grader:
    """Votes per reply text; a reply not in the map is unanimous pass. Counts the calls."""

    def __init__(self, split: dict[str, tuple[Verdict, ...]] | None = None) -> None:
        self.split = split or {}
        self.graded: list[str] = []

    def grade(self, item: LabelCandidate) -> tuple[LabelVote, ...]:
        self.graded.append(item.reply)
        verdicts = self.split.get(item.reply, ("pass",) * LABEL_VOTES)
        return tuple(LabelVote(verdict=v, reason="because") for v in verdicts)


def test_the_published_constants_are_pinned() -> None:
    """Catches a maker on the cheap alias (its labels would be graded by the model being
    checked), one that grades twice, and a target of 39 or 41 labels."""
    assert LABEL_ALIAS == "agent-strong"
    assert LABEL_VOTES == 3
    assert LABEL_TARGET == 40
    assert LABEL_SOURCE == "model-reference"


def test_only_unanimous_replies_become_labels() -> None:
    """Catches a maker that keeps a split reply (a 2-1 vote is not a reference) or that stops
    short: the first, third and fifth replies are split, so the next ones are used until 40
    are kept, and a maker that fills 40 slots from the first 40 replies keeps only 37."""
    split: dict[str, tuple[Verdict, ...]] = {
        "reply 0": ("pass", "fail", "pass"),
        "reply 2": ("fail", "fail", "pass"),
        "reply 4": ("pass", "pass", "fail"),
    }
    grader = Grader(split)

    labels = make_labels((candidate(i) for i in range(100)), grader)

    assert len(labels) == LABEL_TARGET
    assert {"reply 0", "reply 2", "reply 4"}.isdisjoint({label.reply for label in labels})
    assert labels[0].reply == "reply 1"
    assert len(grader.graded) == 43


def test_each_label_has_the_eight_fields_and_the_model_reference_source() -> None:
    """Catches a label missing `votes` or `source`, one that copies the verdict from the first
    vote when the votes disagree, and a `source` of `human` on a model's grade."""
    split: dict[str, tuple[Verdict, ...]] = {"reply 1": ("fail", "fail", "fail")}

    labels = make_labels([candidate(0), candidate(1)], Grader(split))

    written = [label.model_dump(mode="json") for label in labels]
    assert all(set(line) == EIGHT_FIELDS for line in written)
    assert [(line["verdict"], line["votes"]) for line in written] == [
        ("pass", ["pass", "pass", "pass"]),
        ("fail", ["fail", "fail", "fail"]),
    ]
    assert {line["source"] for line in written} == {"model-reference"}
    assert written[0]["rubric"] == "rubric 0"
    assert written[0]["reply"] == "reply 0"
    assert written[0]["turn"] == 1


def test_a_reply_with_fewer_than_three_usable_votes_is_dropped() -> None:
    """Catches a maker that keeps a reply whose third sample was unusable: two agreeing votes
    are not three, so the reply is dropped and the next used."""
    grader = Grader({"reply 0": ("pass", "pass")})

    labels = make_labels([candidate(0), candidate(1)], grader)

    assert [label.reply for label in labels] == ["reply 1"]


def test_it_stops_asking_once_the_limit_is_kept() -> None:
    """Catches a maker that grades every candidate it is given: each grade is three paid
    requests, and the pool is far larger than 40. The candidates are a generator that fails
    if read past the 40th."""

    def pool() -> Iterator[LabelCandidate]:
        for index in range(LABEL_TARGET):
            yield candidate(index)
        raise AssertionError("candidate 41 was read after 40 labels were kept")

    labels = make_labels(pool(), Grader())

    assert len(labels) == LABEL_TARGET


def test_fewer_unanimous_replies_than_the_limit_give_what_there_is() -> None:
    """Catches a maker that raises or pads when the pool runs dry: it returns the labels it
    has, and the calibration step is what refuses a short file."""
    labels = make_labels([candidate(0), candidate(1), candidate(2)], Grader())

    assert len(labels) == 3


def test_candidates_come_from_the_graded_entries_word_for_word() -> None:
    """Catches a candidate whose reply is paraphrased or taken from the wrong turn, whose
    rubric belongs to another entry, or whose conversation hides what the maker needs: the
    whole conversation (both user turns, the tool call with its arguments and its result)
    is in the text the maker sees."""
    case = make_case(
        "conv-005",
        turns=("reset my password", "yes"),
        graded=((1, "asks first?"), (2, "says what happened?")),
    )
    reply_one = "I can reset it. Say yes."
    reply_two = "Done. Check your email, REQ-0007."
    played = answer(
        "conv-005",
        turn(1, reply_one),
        turn(
            2,
            reply_two,
            call("reset_own_password", {"mode": "email"}, result={"request_id": "REQ-0007"}),
        ),
    )

    found = candidates_from(case, played)

    assert [(c.case_id, c.turn, c.rubric, c.reply) for c in found] == [
        ("conv-005", 1, "asks first?", reply_one),
        ("conv-005", 2, "says what happened?", reply_two),
    ]
    for item in found:
        for needed in (
            "reset my password",
            "yes",
            reply_one,
            reply_two,
            "reset_own_password",
            "email",
            "REQ-0007",
        ):
            assert needed in item.conversation


def test_labels_are_written_once_and_never_overwritten(tmp_path: Path) -> None:
    """Catches a second label run that replaces the first: the labels cost paid model calls
    and the owner said they are never made again. The first write keeps file order; the
    second raises and leaves the file as it was."""
    path = tmp_path / "conversation-labels.jsonl"
    labels = make_labels([candidate(0), candidate(1)], Grader())

    write_labels(path, labels)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_labels(path, make_labels([candidate(5)], Grader()))

    assert path.read_text(encoding="utf-8") == before
    lines = [json.loads(line) for line in before.splitlines()]
    assert [line["reply"] for line in lines] == ["reply 0", "reply 1"]
    assert all(set(line) == EIGHT_FIELDS for line in lines)


def test_the_gateway_grader_asks_agent_strong_three_times_with_the_whole_conversation() -> None:
    """Catches a grader on the wrong alias, one that asks once and copies the answer, and one
    that leaves the conversation out (the maker must see the tool calls and results, which
    the judge being checked never sees). Answers are the judge's `reason` then `verdict`."""
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        text = json.dumps({"reason": "It asks first.", "verdict": "pass"})
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    grader = GatewayLabelGrader(
        "https://gateway.example",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        api_key="sk-not-real",
    )

    votes = grader.grade(candidate(0))

    assert [(v.verdict, v.reason) for v in votes] == [("pass", "It asks first.")] * 3
    assert len(requests) == 3
    assert {r["model"] for r in requests} == {"agent-strong"}
    for body in requests:
        sent = json.dumps(body)
        assert "whole conversation 0" in sent
        assert "rubric 0" in sent
        assert "reply 0" in sent
