"""`evals/calibration.py`: the labels file, the weighted kappa, the trust flag (#36).

No network, no key and no gateway. Every case that needs a judge drives a real
`JudgeClient` over an `httpx.Client(transport=httpx.MockTransport(...))`, exactly
as `tests/evals/test_judge.py` drives 17c's contract, so the whole module is
pinned before it exists.

The contract these cases are written against, before it exists:

* the constants `MIN_CALIBRATION_LABELS = 40`, `MAX_LABEL_ID_LENGTH = 128`,
  `MAX_LABEL_TEXT_LENGTH = 20000`, `KAPPA_SCALE = 3`, `JUDGE_TRUST_MIN = 0.7`
  and `CALIBRATION_JSON_NAME = "calibration.json"` - the values a project rule
  requires a test to pin exactly.
* `CalibrationLabel(id, output, human_score, judge_rubric, note=None)`, frozen
  and `extra="forbid"`, with `id` checked against the gateway's own
  `IDENTIFIER_PATTERN` at `MAX_LABEL_ID_LENGTH`, both text fields capped at
  `MAX_LABEL_TEXT_LENGTH` and `human_score` bounded
  `JUDGE_MIN_SCORE..JUDGE_MAX_SCORE` INCLUSIVE.
* `load_labels(path) -> tuple[CalibrationLabel, ...]`, `cases.load_golden_set`'s
  own shape: blank lines and `#` comments skipped, the FILE's line number
  reported, every per-line failure wrapped in `CalibrationSetInvalid`, loaded
  whole or not at all - and the `MIN_CALIBRATION_LABELS` floor applied in the
  LOADER, before a request can be made.
* `weighted_kappa(human, judge) -> float | None`, quadratic weights
  `w(i, j) = (i - j)**2 / 16`, `1 - (sum w*O) / (sum w*E)`, quantized at
  `KAPPA_SCALE` places with `ROUND_HALF_UP`, and `None` when `sum w*E` is zero.
* `calibrate(labels, *, judge, labels_name, labels_sha) -> CalibrationReport`,
  reaching the judge through `JudgeClient.score_reply(rubric, reply)` once per
  label in FILE order, dropping the labels it could not score, and raising
  `CalibrationRefused` when fewer than `MIN_CALIBRATION_LABELS` pairs survive.
* `write_calibration(out, report)`, writing exactly `CALIBRATION_JSON_NAME`.

Two arrangements are what keep these cases from passing on a neighbour's guard:

* every loader rejection row is a file of FORTY well-formed labels plus the one
  bad line, so the `MIN_CALIBRATION_LABELS` floor - which raises the same
  exception type - cannot be what refuses it, and the reported line number is
  asserted so the refusal is known to be about that line; and
* the "too few labels" case is driven through `calibrate(load_labels(path), ...)`
  with a recording transport, so a floor moved out of the loader and into
  `calibrate` shows up as requests that were paid for before the refusal.

The kappa vectors are hand-computed and asserted as values rather than as bands,
because no library in this repo can cross-check the formula.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from opsconsole.evals.calibration import (
    CALIBRATION_JSON_NAME,
    JUDGE_TRUST_MIN,
    KAPPA_MAX,
    KAPPA_MIN,
    KAPPA_SCALE,
    MAX_LABEL_ID_LENGTH,
    MAX_LABEL_TEXT_LENGTH,
    MAX_LABELS_NAME_LENGTH,
    MIN_CALIBRATION_LABELS,
    CalibrationRefused,
    CalibrationReport,
    CalibrationSetInvalid,
    calibrate,
    load_labels,
    read_calibration,
    weighted_kappa,
    write_calibration,
)
from opsconsole.evals.cases import EvalCase, EvalCategory, ExpectedOutcome
from opsconsole.evals.judge import (
    JUDGE_ALIAS,
    JUDGE_MAX_SCORE,
    JUDGE_MIN_SCORE,
    JudgeClient,
    JudgeVerdict,
    judge_prompt_version,
)
from opsconsole.evals.outcome import CaseOutcome

BASE_URL = "https://gateway.example"

API_KEY = "sk-eval-judge-not-a-real-key"

TOKEN_PATTERN = re.compile(r"label-(\d{3})")
"""How the doubled gateway tells which label a request is about.

Every label's `output` and `judge_rubric` carry its own zero-padded index, so the
double answers per LABEL rather than per call number - a `calibrate` that scored
the labels in the wrong order, or scored one of them twice, cannot be given the
right answer by accident.
"""


# --------------------------------------------------------------------------
# the labels file
# --------------------------------------------------------------------------


def label_payload(index: int, human_score: int = 3) -> dict[str, Any]:
    """One well-formed label line as a document, carrying its own index token."""
    token = f"label-{index:03d}"
    return {
        "id": token,
        "output": f"reply for {token}",
        "human_score": human_score,
        "judge_rubric": f"rubric for {token}",
    }


def label_line(index: int, human_score: int = 3, **overrides: object) -> str:
    """One label line, with any field replaced by the value a case is testing."""
    document = label_payload(index, human_score)
    document.update(overrides)
    return json.dumps(document)


def line_without(index: int, field: str) -> str:
    """One label line missing exactly one required field."""
    document = label_payload(index)
    del document[field]
    return json.dumps(document)


def good_lines(
    count: int = MIN_CALIBRATION_LABELS, humans: Sequence[int] | None = None
) -> list[str]:
    """`count` well-formed label lines, in index order, optionally scored to a table."""
    if humans is None:
        return [label_line(index) for index in range(count)]
    return [label_line(index, human) for index, human in enumerate(humans)]


def write_lines(tmp_path: Path, lines: Sequence[str], *, name: str = "labels.jsonl") -> Path:
    """Those lines as a file, exactly as written - no trailing surprises."""
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def refused_file(tmp_path: Path, *extra: str) -> Path:
    """Forty well-formed labels followed by the bad line under test.

    The floor is satisfied by the forty, so the only control that can refuse this
    file is the per-line one the case names, and the bad line's number is
    `MIN_CALIBRATION_LABELS + 1` for every case that uses this.
    """
    return write_lines(tmp_path, good_lines() + list(extra))


BAD_LINE = MIN_CALIBRATION_LABELS + 1
"""The 1-based file line every `refused_file` case puts its bad line on."""


# --------------------------------------------------------------------------
# the doubled gateway
# --------------------------------------------------------------------------


def answer(score: int, reason: str = "The reply meets the rubric.") -> dict[str, Any]:
    """One complete OpenAI-compatible chat answer carrying the judge's JSON."""
    return {
        "id": "chatcmpl-judge",
        "object": "chat.completion",
        "model": JUDGE_ALIAS,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"score": score, "reason": reason}),
                },
                "finish_reason": "stop",
            }
        ],
    }


class Gateway:
    """A recording gateway that answers each label with the score it was given.

    `None` for a label means every sample of it fails: the double answers 500,
    which 17c retries exactly once and then abandons, so that label reaches
    `calibrate` as an unscorable one. `requests` is what the ordering and the
    "no request was made" cases assert on, because no report can tell a run that
    asked from a run that did not.
    """

    def __init__(self, scores: Sequence[int | None]) -> None:
        self.scores = list(scores)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        score = self.scores[self.index_of(request)]
        if score is None:
            return httpx.Response(500, json={"error": "the judge is down"})
        return httpx.Response(200, json=answer(score))

    def index_of(self, request: httpx.Request) -> int:
        """Which label a request is about, read off its own text."""
        body = json.loads(request.content)
        found = TOKEN_PATTERN.search(body["messages"][-1]["content"])
        if found is None:
            raise AssertionError(f"no label token in the judge request: {body}")
        return int(found.group(1))

    def tokens(self) -> list[str]:
        """The label each request was about, in the order the requests were made."""
        return [f"label-{self.index_of(request):03d}" for request in self.requests]

    def client(self) -> JudgeClient:
        return JudgeClient(
            BASE_URL, client=httpx.Client(transport=httpx.MockTransport(self)), api_key=API_KEY
        )


Table = Mapping[tuple[int, int], int]
"""A contingency table: how many labels carry each (human score, judge score)."""


def table_pairs(table: Table) -> list[tuple[int, int]]:
    """That table expanded into one (human, judge) pair per label."""
    pairs: list[tuple[int, int]] = []
    for (human, judge), count in table.items():
        pairs.extend([(human, judge)] * count)
    return pairs


def calibrated(
    tmp_path: Path, pairs: Sequence[tuple[int, int | None]]
) -> tuple[CalibrationReport, Gateway]:
    """Write those labels, score them through a doubled gateway, and calibrate."""
    lines = [label_line(index, human) for index, (human, _) in enumerate(pairs)]
    path = write_lines(tmp_path, lines)
    gateway = Gateway([judge for _, judge in pairs])
    report = calibrate(
        load_labels(path),
        judge=gateway.client(),
        labels_name=path.name,
        labels_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return report, gateway


TRUSTED_TABLE: Table = {(1, 1): 17, (5, 5): 17, (1, 5): 3, (5, 1): 3}
"""Forty labels whose kappa is EXACTLY the trust floor. Arithmetic in the case."""

UNTRUSTED_TABLE: Table = {(1, 1): 16, (5, 5): 18, (1, 5): 3, (5, 1): 3}
"""The same forty with one agreement moved, which lands just under it."""


# --------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------


def test_the_calibration_constants_are_the_published_ones() -> None:
    """Catches any of the six being widened to make a file load or a kappa pass: a floor
    silently dropped to 12 turns a dozen labels into published calibration, a trust
    threshold nudged to 0.65 turns an untrusted judge into a gate's evidence, and a
    `KAPPA_SCALE` of 2 cannot separate 0.699 from 0.700 at all. Asserted as exact values
    rather than as bands, and `CALIBRATION_JSON_NAME` with them because
    17f reads the artifact by that name."""
    assert MIN_CALIBRATION_LABELS == 40
    assert MAX_LABEL_ID_LENGTH == 128
    assert MAX_LABEL_TEXT_LENGTH == 20000
    assert KAPPA_SCALE == 3
    assert JUDGE_TRUST_MIN == 0.7
    assert CALIBRATION_JSON_NAME == "calibration.json"


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------


def test_a_well_formed_file_loads_in_file_order(tmp_path: Path) -> None:
    """Catches a loader that returns the labels in some other order - a dict, a set or a
    sort by id - which would pair every human score against another label's judge score
    and produce a kappa nobody can reproduce. Also catches a loader that drops the
    optional `note` or silently coerces `human_score` to a string."""
    path = write_lines(tmp_path, good_lines(humans=[1, 2, 3, 4, 5] * 8))

    labels = load_labels(path)

    assert len(labels) == MIN_CALIBRATION_LABELS
    assert [label.id for label in labels] == [f"label-{index:03d}" for index in range(40)]
    assert [label.human_score for label in labels] == [1, 2, 3, 4, 5] * 8
    assert labels[0].judge_rubric == "rubric for label-000"
    assert labels[0].output == "reply for label-000"
    assert labels[0].note is None


def test_a_fifth_field_is_refused(tmp_path: Path) -> None:
    """Criterion 2, and the whole reason for `extra="forbid"`: catches a model that lets an
    unknown key through, where `{"human_score": 4, "score": 2}` - a field an author
    invented or misspelt - loads cleanly and the label is calibrated against a number
    nobody meant. The forty good lines above it mean the floor cannot be what refused
    this, and the line number proves the refusal is about the fifth field's line."""
    path = refused_file(tmp_path, label_line(40, 3, rating=2))

    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(path)

    assert refusal.value.line == BAD_LINE
    assert refusal.value.path == path


def test_a_duplicate_id_is_refused(tmp_path: Path) -> None:
    """Catches a loader with no `seen` set: two labels sharing an id are two rows a reader
    cannot tell apart in `calibration.json`, and a re-labelled file that accidentally
    repeats a block would double-weight those pairs in the kappa. The duplicate is the
    41st line, so the floor is met and only the uniqueness check can refuse it."""
    path = refused_file(tmp_path, label_line(0))

    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(path)

    assert refusal.value.line == BAD_LINE


def test_a_missing_judge_rubric_is_refused(tmp_path: Path) -> None:
    """Catches `judge_rubric` given a default of `None` or `""`: the judge would then be
    asked to grade every reply against an empty standard, three samples each, and the
    resulting kappa would measure the judge against a rubric no human ever used. Required
    and not optional is the whole claim, and `note` above proves optional fields exist."""
    path = refused_file(tmp_path, line_without(40, "judge_rubric"))

    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(path)

    assert refusal.value.line == BAD_LINE


def test_a_comment_and_a_blank_line_are_skipped_and_the_file_line_is_still_reported(
    tmp_path: Path,
) -> None:
    """Two claims one arrangement can hold: the file loads DESPITE the two non-label lines,
    and the number in the refusal is the number an author counts to in an editor. Catches a
    loader that counts only the label lines - here that would say 41 for a line on 43, and
    an author would fix the wrong record - and catches one that treats `#` as JSON."""
    lines = ["# forty helpdesk replies, labelled by hand", "", *good_lines()]
    path = write_lines(tmp_path, [*lines, label_line(40, 9)])

    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(path)

    assert refusal.value.line == 43
    assert len(load_labels(write_lines(tmp_path, lines, name="clean.jsonl"))) == 40


def test_nothing_is_returned_when_a_later_line_is_bad(tmp_path: Path) -> None:
    """Whole or not at all, `load_golden_set`'s own property. Catches a loader that yields
    the good labels it collected before the bad line: forty of forty-one labels is a
    calibration that silently excluded the record an author most wants to look at, and it
    would pass the floor while doing it."""
    path = refused_file(tmp_path, label_line(40, 0))

    with pytest.raises(CalibrationSetInvalid):
        load_labels(path)


# --------------------------------------------------------------------------
# the five bounds, at the value and one past it
# --------------------------------------------------------------------------


MAX_ID = "a" * MAX_LABEL_ID_LENGTH
OVER_CAP_ID = "a" * (MAX_LABEL_ID_LENGTH + 1)

MAX_TEXT = "t" * MAX_LABEL_TEXT_LENGTH
OVER_CAP_TEXT = "t" * (MAX_LABEL_TEXT_LENGTH + 1)


def test_the_id_length_boundary_is_inclusive(tmp_path: Path) -> None:
    """Catches an off-by-one at the cap in either direction: a check written `>=` refuses an
    id of exactly 128 characters, and a check on a wider constant lets a longer one into
    `calibration.json`'s rows. 128 and not the resolver's 64 (a label id is not a gateway
    identifier), which is why the cap is asserted as its own integer."""
    at_cap = write_lines(tmp_path, [*good_lines()[:-1], label_line(39, 3, id=MAX_ID)])

    assert load_labels(at_cap)[-1].id == MAX_ID

    over_cap = refused_file(tmp_path, label_line(40, 3, id=OVER_CAP_ID))
    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(over_cap)
    assert refusal.value.line == BAD_LINE


def test_the_output_length_boundary_is_inclusive(tmp_path: Path) -> None:
    """a project rule on the value that is SENT: catches a cap applied to the wrong field
    or not at all, which puts a megabyte of reply into a judge request that the gateway
    then refuses per sample, after the other labels have been paid for. Exactly 20000
    characters is a legitimate long reply and must load."""
    at_cap = write_lines(tmp_path, [*good_lines()[:-1], label_line(39, 3, output=MAX_TEXT)])

    assert load_labels(at_cap)[-1].output == MAX_TEXT

    over_cap = refused_file(tmp_path, label_line(40, 3, output=OVER_CAP_TEXT))
    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(over_cap)
    assert refusal.value.line == BAD_LINE


def test_the_judge_rubric_length_boundary_is_inclusive(tmp_path: Path) -> None:
    """The same cap on the OTHER string the judge is sent, asserted separately because one
    field capped and the other not is the likeliest shape of this defect - and the rubric
    is the half a `MAX_LABEL_TEXT_LENGTH` written once, for `output` only, would leave
    unbounded."""
    at_cap = write_lines(tmp_path, [*good_lines()[:-1], label_line(39, 3, judge_rubric=MAX_TEXT)])

    assert load_labels(at_cap)[-1].judge_rubric == MAX_TEXT

    over_cap = refused_file(tmp_path, label_line(40, 3, judge_rubric=OVER_CAP_TEXT))
    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(over_cap)
    assert refusal.value.line == BAD_LINE


def test_the_human_score_bounds_are_inclusive(tmp_path: Path) -> None:
    """Catches bounds written `>` and `<`, which would refuse the two ends of the very scale
    the rubric defines: a 1 is the score a bad reply gets and a 5 the score a good one
    gets, and a file of nothing but 2, 3 and 4 is not a calibration set."""
    path = write_lines(tmp_path, good_lines(humans=[JUDGE_MIN_SCORE, JUDGE_MAX_SCORE] * 20))

    assert [label.human_score for label in load_labels(path)] == [1, 5] * 20


@pytest.mark.parametrize(
    ("label", "human_score"),
    [("below the scale", JUDGE_MIN_SCORE - 1), ("above the scale", JUDGE_MAX_SCORE + 1)],
    ids=["a score of zero", "a score of six"],
)
def test_a_human_score_outside_the_scale_is_refused(
    label: str, human_score: int, tmp_path: Path
) -> None:
    """Catches an unbounded `human_score`: a 0 or a 6 is a typo, and the weighted kappa
    would take it as a real category - the weight of a (6, 1) cell is 25/16, larger than
    the formula's own maximum, so one mistyped digit moves the whole agreement. The bounds
    are the judge's own 1..5, imported rather than restated."""
    path = refused_file(tmp_path, label_line(40, human_score))

    with pytest.raises(CalibrationSetInvalid) as refusal:
        load_labels(path)

    assert refusal.value.line == BAD_LINE


# --------------------------------------------------------------------------
# the floor, applied in the loader
# --------------------------------------------------------------------------


def test_a_file_of_thirty_nine_labels_is_refused_before_any_request(tmp_path: Path) -> None:
    """Criterion 8, and its ordering half. Catches a floor checked inside `calibrate` after
    the scoring loop, which is 39 labels x 3 samples of real spend before the refusal, and
    catches no floor at all - a kappa over a dozen pairs read as calibration is worse than
    no calibration. Driven through `calibrate(load_labels(...))` precisely so a floor that
    moved downstream is visible as recorded requests rather than as the same exception."""
    path = write_lines(tmp_path, good_lines(MIN_CALIBRATION_LABELS - 1))
    gateway = Gateway([3] * (MIN_CALIBRATION_LABELS - 1))

    with pytest.raises(CalibrationSetInvalid):
        calibrate(
            load_labels(path),
            judge=gateway.client(),
            labels_name=path.name,
            labels_sha="0" * 64,
        )

    assert gateway.requests == []


def test_a_file_of_forty_labels_is_accepted(tmp_path: Path) -> None:
    """The inclusive half of the floor: catches a check written `<=`, which would refuse the
    exact count the plan, the README and 17f all name as sufficient and leave the tool
    unable to load the only file anyone was told to produce."""
    path = write_lines(tmp_path, good_lines(MIN_CALIBRATION_LABELS))

    assert len(load_labels(path)) == MIN_CALIBRATION_LABELS


# --------------------------------------------------------------------------
# the weighted kappa
# --------------------------------------------------------------------------


def test_perfect_agreement_is_one() -> None:
    """The first of the three hand-computed vectors (criterion 3). Every observed cell is on
    the diagonal, so `sum w*O` is 0 and kappa is 1 - 0. Catches a formula that forgot to
    subtract from one and reports 0 for perfect agreement, which would grey out a judge
    that agrees with every human label. The vector spreads over 1, 2, 4 and 5 so the
    expected term is non-zero (`sum w*E` is 1.25) and the None branch cannot answer it."""
    assert weighted_kappa([1, 2, 4, 5], [1, 2, 4, 5]) == 1.0


def test_the_inverted_table_is_minus_one() -> None:
    """The vector risk 1 names as the one that fails for any implementation that forgot the
    weights or the expected term. Four pairs, each 1 against 5 or 5 against 1: `sum w*O` is
    4 (weight 16/16 each), the marginals are two and two on both sides so `sum w*E` is 2,
    and kappa is 1 - 4/2. Catches a kappa clamped at 0, an unweighted count of agreements
    (which would say 0 here) and a missing expected term (which would say -3)."""
    assert weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) == -1.0


def test_one_step_of_disagreement_is_zero_point_nine_two_nine() -> None:
    """The third vector, and the only one whose value depends on the quadratic weight being
    a quadratic. Three pairs agree and one is 4 against 5: `sum w*O` is 1/16, `sum w*E` is
    56/64 = 0.875, and 1 - (1/16)/0.875 is 13/14 = 0.92857..., which is 0.929 at three
    places. Catches LINEAR weights, which give 1 - (1/4)/(14/8) = 0.857 here and agree with
    the other two vectors exactly - this is the case that separates them."""
    assert weighted_kappa([1, 2, 3, 4], [1, 2, 3, 5]) == 0.929


def test_the_kappa_rounds_half_up_and_not_to_the_nearest_even() -> None:
    """Criterion 4, and the one case in this file that sees it. The three pairs (1, 2),
    (3, 3) and (4, 5) give `sum w*O` = 2/16 = 0.125; the marginals are one each over
    {1, 3, 4} and {2, 3, 5}, so `sum w*E` = (1/3) * 32/16 = 2/3, and kappa is exactly
    1 - (0.125)/(2/3) = 1 - 3/16 = 0.8125. Half-up that is 0.813; `round(0.8125, 3)` is
    0.812, because Python rounds a tie to the nearest EVEN digit. Catches `round()` on a
    float, which is what an implementation reaches for first, and a `quantize` left on its
    default rounding - a hundredth of a kappa either side of 0.7 is a judge trusted or not
    trusted by a rounding mode."""
    assert weighted_kappa([1, 3, 4], [2, 3, 5]) == 0.813


def test_a_table_with_no_expected_disagreement_has_no_kappa() -> None:
    """Criterion 5 at the function. Both raters answered 3 every time, so the only occupied
    cell is (3, 3) and `sum w*E` is 0: the agreement is real and completely uninformative.
    Catches a 1.0 returned for it - perfect agreement that proves nothing must never reach
    a gate as trust - and catches a `ZeroDivisionError` reaching the caller as a traceback
    over a calibration that was already paid for."""
    assert weighted_kappa([3] * 8, [3] * 8) is None


# --------------------------------------------------------------------------
# calibrate: the trust flag, the drops, the artifact
# --------------------------------------------------------------------------


def test_a_constant_table_reports_no_kappa_and_is_not_trusted(tmp_path: Path) -> None:
    """The undefined case as an ARTIFACT, which the function-level case cannot see: `kappa`
    is null, `trusted` is false because `kappa is None`, and `reason` says which case it
    was. Catches `trusted` computed as `kappa >= JUDGE_TRUST_MIN` on a `None` - a
    `TypeError` at best and, if written `not kappa < 0.7`, a trusted judge nothing
    measured - and catches a null kappa published with no reason beside it, which is a
    reader's only clue why the column is empty."""
    report, _ = calibrated(tmp_path, [(3, 3)] * MIN_CALIBRATION_LABELS)

    assert report.kappa is None
    assert report.trusted is False
    assert report.reason is not None
    assert report.reason != ""


def test_a_kappa_of_exactly_the_trust_floor_is_trusted(tmp_path: Path) -> None:
    """The inclusive half of criterion 6, hand-constructed. Forty labels: 17 agree at 1, 17
    agree at 5, 3 are human 1 against judge 5 and 3 are human 5 against judge 1. Every
    disagreement weighs 16/16, so `sum w*O` = 6; both marginals are 20 and 20, so
    `sum w*E` = (20*20 + 20*20)/40 = 20; kappa is 1 - 6/20 = 0.700 exactly. Catches a
    threshold read with `>`, which greys out a judge that met the published bar exactly -
    the boundary no band-style assertion can see."""
    report, _ = calibrated(tmp_path, table_pairs(TRUSTED_TABLE))

    assert report.kappa == 0.7
    assert report.trusted is True
    assert report.reason is None


def test_a_kappa_below_the_trust_floor_is_not_trusted(tmp_path: Path) -> None:
    """The other half, one label away from the case above: the same six disagreements with
    one agreement moved from (1, 1) to (5, 5), so the marginals become 19 and 21 on both
    sides and `sum w*E` = (19*21 + 21*19)/40 = 19.95. Kappa is 1 - 6/19.95 = 0.6992481...,
    which is 0.699 at three places and below the floor. Catches a threshold compared at two
    decimal places, where both this table and the one above read 0.70 and an untrusted
    judge is published as trusted."""
    report, _ = calibrated(tmp_path, table_pairs(UNTRUSTED_TABLE))

    assert report.kappa == 0.699
    assert report.trusted is False


def test_six_unscorable_labels_of_forty_six_are_dropped_and_counted(tmp_path: Path) -> None:
    """Criterion 7 and decision D. Forty scorable labels - the exact table whose kappa is
    0.700 - followed by six the judge answers 500 for, each retried once and abandoned.
    Catches an unscorable label counted as a 0, which the weights would take as a category
    off the scale, and catches one counted as the HUMAN's own number: six extra (3, 3)
    pairs would move both marginals and the kappa off 0.700, so the value asserted here is
    what proves the six were dropped rather than filled in. `dropped` is asserted because
    it is how a reader tells a clean forty from a lucky forty of forty-six."""
    pairs: list[tuple[int, int | None]] = [
        (human, judge) for human, judge in table_pairs(TRUSTED_TABLE)
    ]
    pairs += [(3, None)] * 6

    report, gateway = calibrated(tmp_path, pairs)

    assert report.pairs == MIN_CALIBRATION_LABELS
    assert report.dropped == 6
    assert len(report.rows) == MIN_CALIBRATION_LABELS
    assert report.kappa == 0.7
    assert gateway.tokens().count("label-040") == 2


def test_seven_unscorable_labels_of_forty_six_refuses_and_writes_nothing(tmp_path: Path) -> None:
    """The floor applied to what SURVIVES (criterion 8, decision B). Thirty-nine usable pairs
    is one short of forty, so nothing is published. Catches a floor checked only against the
    file's record count, which would publish a kappa over 39 pairs the moment a judge got
    flaky, and the empty directory catches a report written before the count is checked -
    `write_calibration` is never reached, because the refusal happens while its own argument
    is being evaluated."""
    pairs: list[tuple[int, int | None]] = [
        (human, judge) for human, judge in table_pairs(TRUSTED_TABLE)[:39]
    ]
    pairs += [(3, None)] * 7
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(CalibrationRefused):
        write_calibration(out, calibrated(tmp_path, pairs)[0])

    assert list(out.iterdir()) == []


def test_the_artifact_carries_exactly_the_published_keys(tmp_path: Path) -> None:
    """Criterion 9, asserted by EQUALITY so a field quietly dropped and a field quietly added
    are both red - 17f and row 18 read this document, and a missing `judge_prompt_version`
    is what lets a calibration made under one prompt be worn as trust by a run made under
    another. Catches a writer that names the file something else or writes two files, and
    catches rows sorted or de-duplicated: one row per SCORED label, in file order, carrying
    the human's number and the judge's median for that same label."""
    out = tmp_path / "out"
    out.mkdir()
    pairs = table_pairs(TRUSTED_TABLE)
    report, gateway = calibrated(tmp_path, pairs)

    write_calibration(out, report)

    document = json.loads((out / CALIBRATION_JSON_NAME).read_text(encoding="utf-8"))
    assert [path.name for path in out.iterdir()] == [CALIBRATION_JSON_NAME]
    assert set(document) == {
        "judge_prompt_version",
        "labels_name",
        "labels_sha",
        "computed_at",
        "pairs",
        "dropped",
        "kappa",
        "reason",
        "trusted",
        "rows",
    }
    assert document["judge_prompt_version"] == judge_prompt_version()
    assert document["labels_name"] == "labels.jsonl"
    assert document["dropped"] == 0
    assert [row["id"] for row in document["rows"]] == [
        f"label-{index:03d}" for index in range(len(pairs))
    ]
    assert [row["human_score"] for row in document["rows"]] == [human for human, _ in pairs]
    assert [row["judge_median"] for row in document["rows"]] == [judge for _, judge in pairs]
    assert gateway.tokens()[0] == "label-000"


# --------------------------------------------------------------------------
# the `score_reply` seam (decision E)
# --------------------------------------------------------------------------


def no_wire(request: httpx.Request) -> httpx.Response:
    """Any request at all is a failure for the two seam cases below."""
    raise AssertionError(f"the judge opened the wire: {request.url}")


class RecordingJudge(JudgeClient):
    """A judge whose `score_reply` records its two strings and never opens the wire.

    A subclass rather than a duck type, so the case fails if `calibrate` reaches
    the judge by any route other than `score_reply` - the sampling loop it would
    otherwise run is still inherited and would hit `no_wire`.
    """

    def __init__(self) -> None:
        super().__init__(
            BASE_URL,
            client=httpx.Client(transport=httpx.MockTransport(no_wire)),
            api_key=API_KEY,
        )
        self.calls: list[tuple[str, str]] = []

    def score_reply(self, rubric: str, reply: str) -> JudgeVerdict | None:
        self.calls.append((rubric, reply))
        return JudgeVerdict(median=4, samples=(4, 4, 4), reason="It names the resource.")


def test_calibrate_reaches_the_judge_through_score_reply_with_the_labels_own_strings(
    tmp_path: Path,
) -> None:
    """Decision E's seam, from the calibration side. Catches a `calibrate` that builds a
    synthetic `EvalCase` and calls `score(case, outcome)` - which would make two records'
    required fields part of this module's contract - and catches the rubric and the reply
    handed over the wrong way round, which would grade every rubric against a reply and
    score a whole calibration on the wrong text. One call per label, in file order."""
    path = write_lines(tmp_path, good_lines())
    judge = RecordingJudge()

    report = calibrate(load_labels(path), judge=judge, labels_name=path.name, labels_sha="0" * 64)

    assert judge.calls == [
        (f"rubric for label-{index:03d}", f"reply for label-{index:03d}") for index in range(40)
    ]
    assert [row.judge_median for row in report.rows] == [4] * 40


def test_score_still_delegates_to_score_reply() -> None:
    """The other half of decision E, which keeps every 17c case true: `score` is two lines
    over `score_reply`. Catches the sampling logic left duplicated in `score` - two copies
    of the retry rule drift, and the calibration path would then be scored by a second
    implementation of the median nobody tests - and catches a `score` that passes the
    outcome's reply as the rubric."""
    judge = RecordingJudge()
    case = EvalCase(
        id="policy-qa-001",
        category=EvalCategory.POLICY_QA,
        tenant="acme",
        principal_user="dana.okafor",
        messages=[{"role": "user", "content": "What is the access policy?"}],
        expected=ExpectedOutcome(),
        judge_rubric="Score 5 only if the reply names the access level.",
    )

    verdict = judge.score(case, CaseOutcome(reply="Read access is active."))

    assert judge.calls == [
        ("Score 5 only if the reply names the access level.", "Read access is active.")
    ]
    assert verdict is not None
    assert verdict.median == 4


# --------------------------------------------------------------------------
# read_calibration: the artifact on its way back INTO a run (17f)
# --------------------------------------------------------------------------


def written_artifact(tmp_path: Path, **overrides: object) -> Path:
    """A real `calibration.json`, written by `write_calibration`, with fields replaced.

    Built from the trusted table through `calibrate` rather than hand-typed, so a
    case that expects the reader to ACCEPT a document is reading the exact bytes
    this module writes - a round trip that assumed a hand-written shape would go
    green while the two halves drifted apart.
    """
    report, _ = calibrated(tmp_path, table_pairs(TRUSTED_TABLE))
    out = tmp_path / "written"
    out.mkdir()
    write_calibration(out, report)
    path = out / CALIBRATION_JSON_NAME
    if overrides:
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(overrides)
        path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_artifact_bounds_are_the_published_ones() -> None:
    """a project rule on the three bounds this row adds, each of which guards a value that
    lands in a stored column: the kappa in `eval_runs.judge_kappa` (`double precision`) and
    the labels name in two `jsonb` documents. Catches a cap widened to make a real
    operator's file load, and catches the kappa bounds written as the trust floor by
    mistake - `-1.0` is a perfectly legal weighted kappa (a judge that inverts the human
    every time) and refusing it would hide the most informative calibration there is."""
    assert MAX_LABELS_NAME_LENGTH == 128
    assert KAPPA_MIN == -1.0
    assert KAPPA_MAX == 1.0


def test_read_calibration_round_trips_what_write_calibration_wrote(tmp_path: Path) -> None:
    """The pair this row exists to close: 17e writes the artifact, 17f reads it back, and
    nothing else in the repo has ever read one. Catches a reader that returns the raw
    document or a differently shaped record, catches a `computed_at` that comes back naive
    or as a string - it is the only field whose JSON form is not its Python one - and
    catches rows re-ordered or de-duplicated on the way in, which would silently change
    which human score sits against which judge median."""
    report, _ = calibrated(tmp_path, table_pairs(TRUSTED_TABLE))
    out = tmp_path / "written"
    out.mkdir()
    write_calibration(out, report)

    read_back = read_calibration(out / CALIBRATION_JSON_NAME)

    assert read_back == report
    assert read_back.kappa == 0.7
    assert read_back.trusted is True
    assert read_back.judge_prompt_version == judge_prompt_version()
    assert read_back.computed_at == report.computed_at
    assert [row.id for row in read_back.rows] == [row.id for row in report.rows]


@pytest.mark.parametrize("kappa", [-1.0, 1.0], ids=["the floor", "the ceiling"])
def test_a_kappa_at_either_bound_is_accepted(kappa: float, tmp_path: Path) -> None:
    """The INCLUSIVE half of criterion 8's first bound. `-1.0` is total inversion and `1.0`
    is total agreement, and both are values `weighted_kappa` itself produces - two cases
    above assert exactly those numbers - so a check written `-1 < kappa < 1` would refuse
    an artifact this repo can write. Catches a bound compared with `<`/`>` rather than with
    `<=`/`>=`, the boundary no band-style assertion can see."""
    path = written_artifact(tmp_path, kappa=kappa)

    assert read_calibration(path).kappa == kappa


@pytest.mark.parametrize(
    "kappa", [-1.001, 1.001], ids=["one thousandth under", "one thousandth over"]
)
def test_a_kappa_one_thousandth_past_either_bound_is_refused(kappa: float, tmp_path: Path) -> None:
    """The other side of the same bound, one `KAPPA_SCALE` step out so nothing but the
    check can separate the two rows. Catches an unvalidated number reaching
    `eval_runs.judge_kappa`, where a hand-edited `9.9` would publish a judge as
    superhumanly agreeing and a gate would read it as evidence. The path is named because
    an operator passing two files needs to know which one was refused."""
    path = written_artifact(tmp_path, kappa=kappa)

    with pytest.raises(CalibrationSetInvalid) as raised:
        read_calibration(path)

    assert str(path) in str(raised.value)


def test_a_labels_name_of_exactly_the_cap_is_accepted_and_one_over_is_refused(
    tmp_path: Path,
) -> None:
    """Criterion 8's second bound, both sides in one row because one character is the whole
    difference between them. `labels_name` comes from a file an operator names and lands
    unbounded in the `metrics` and report `jsonb` columns, which a project rule covers
    even though a `jsonb` column has no length of its own. Catches a cap applied to the
    wrong field - the sha and the prompt version are fixed-length and would never fire -
    and catches a cap compared with `<`, which would refuse the name that exactly meets
    it."""
    at_cap = "n" * MAX_LABELS_NAME_LENGTH
    over_cap = "n" * (MAX_LABELS_NAME_LENGTH + 1)

    assert read_calibration(written_artifact(tmp_path, labels_name=at_cap)).labels_name == at_cap

    other = tmp_path / "over"
    other.mkdir()
    path = written_artifact(other, labels_name=over_cap)
    with pytest.raises(CalibrationSetInvalid) as raised:
        read_calibration(path)

    assert str(path) in str(raised.value)


def test_an_artifact_that_is_not_json_is_refused_naming_the_path(tmp_path: Path) -> None:
    """The half-written file, which is what a calibration killed mid-write leaves behind.
    Catches a `JSONDecodeError` escaping to the operator as a traceback rather than as the
    typed refusal `cli.py`'s pre-check block already catches and prints as one line."""
    path = tmp_path / "calibration.json"
    path.write_text('{"judge_prompt_version": "abc",', encoding="utf-8")

    with pytest.raises(CalibrationSetInvalid) as raised:
        read_calibration(path)

    assert str(path) in str(raised.value)


def test_an_artifact_with_a_field_the_model_forbids_is_refused(tmp_path: Path) -> None:
    """`extra="forbid"` reached through the READER, which is criterion 1's "validated by
    17e's `CalibrationReport`". Catches a reader that picks the fields it wants out of the
    document - a `kappa` and a `trusted` and nothing else - which would accept a file from
    a future or hand-edited schema and publish trust computed by something nobody can
    identify. The document is otherwise the exact one this module writes, so the extra key
    is the only thing that can refuse it."""
    path = written_artifact(tmp_path, note="hand edited to look better")

    with pytest.raises(CalibrationSetInvalid) as raised:
        read_calibration(path)

    assert str(path) in str(raised.value)


def test_an_artifact_that_does_not_exist_is_refused_naming_the_path(tmp_path: Path) -> None:
    """The mistyped `--calibration`, the commonest way the flag is used wrong. Catches an
    `OSError` escaping as a traceback, and catches a missing file read as "no calibration
    supplied" - a run that silently drops the trust it was asked to publish is the one
    defect this flag exists to prevent."""
    missing = tmp_path / "nowhere" / CALIBRATION_JSON_NAME

    with pytest.raises(CalibrationSetInvalid) as raised:
        read_calibration(missing)

    assert str(missing) in str(raised.value)
