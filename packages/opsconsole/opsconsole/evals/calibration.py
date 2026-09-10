"""Calibrating the judge against human labels: the file, the kappa, the trust flag (#36).

A judge nobody checked is an opinion with a version hash. This module scores a
file of human-graded replies through the SAME judge a run uses - 17c's fixed
prompt, `JUDGE_TEMPERATURE`, `JUDGE_SAMPLES` samples and the median - and reports
how well the two agree, as one number a later gate can refuse to trust.

Four properties are worth more than the arithmetic.

* **Quadratic weights, not plain agreement.** A judge that answers 4 where a
  human said 5 is not the same mistake as one that answers 1, and unweighted
  kappa cannot tell them apart. The weight of a disagreement is
  `(i - j)**2 / (JUDGE_MAX_SCORE - JUDGE_MIN_SCORE)**2`, so one step of the 1..5
  scale costs a sixteenth of what the full span costs
  (`test_one_step_of_disagreement_is_zero_point_nine_two_nine` is the vector that
  separates quadratic weights from linear ones; linear weights answer 0.857 there
  and agree with both other vectors exactly).
* **`Sigma w*E == 0` is `None` and never `1.0`** (decision C). Both raters
  constant means the agreement is real and completely uninformative, and a gate
  reading `1.0` there would trust a judge nothing measured
  (`test_a_table_with_no_expected_disagreement_has_no_kappa`,
  `test_a_constant_table_reports_no_kappa_and_is_not_trusted`).
* **`MIN_CALIBRATION_LABELS` is applied twice** (decision B): to the file's
  record count in `load_labels`, before a single request is paid for, and to the
  pairs that SURVIVE the drops in `calibrate`. A kappa over a dozen pairs read as
  calibration is worse than no calibration
  (`test_a_file_of_thirty_nine_labels_is_refused_before_any_request`,
  `test_seven_unscorable_labels_of_forty_six_refuses_and_writes_nothing`).
* **A label the judge could not score is DROPPED and counted** (decision D),
  never scored as a zero and never as the human's own number - the first is a
  category off the scale the weights would take seriously, the second is an
  agreement nobody observed
  (`test_six_unscorable_labels_of_forty_six_are_dropped_and_counted`).

Nothing here opens a socket of its own: the `JudgeClient` is handed in, so the
whole module is drivable against `httpx.MockTransport` with no gateway up,
exactly as `evals/judge.py` and `evals/target.py` are. It imports `judge.py` and
`judge.py` does not import back - `evals/cli.py` is the only caller of either.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError, ValidationInfo, field_validator

from opsconsole.evals.judge import (
    JUDGE_MAX_SCORE,
    JUDGE_MIN_SCORE,
    JudgeClient,
    judge_prompt_version,
)
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.resolver import IDENTIFIER_PATTERN

MIN_CALIBRATION_LABELS: Final[int] = 40
"""How many labels a calibration needs, at both ends (decision B).

The file must carry this many records before anything is sent, and this many
pairs must survive the judge's drops before an artifact is written. Forty is the
number #36 states; what matters here is that it is a floor rather than a hint,
because 17f reads a written `calibration.json` as trust and publishes it
(`test_the_calibration_constants_are_the_published_ones`)."""

MAX_LABEL_ID_LENGTH: Final[int] = 128
"""The longest `id` a label line may name, inclusive.

a project rule: the id is written into `calibration.json`'s rows, which has no
cap of its own. Wider than the resolver's `MAX_IDENTIFIER_LENGTH` on purpose - a
label id names a graded reply and is not a tenant or an alias - so the cap is its
own integer while the VOCABULARY is still the gateway's `IDENTIFIER_PATTERN`
(`test_the_id_length_boundary_is_inclusive`)."""

MAX_LABEL_TEXT_LENGTH: Final[int] = 20000
"""The longest `output` and the longest `judge_rubric`, in characters, inclusive.

The two strings that are SENT. Uncapped, a pasted transcript becomes a judge
request the gateway refuses per sample, three times, after the earlier labels
have been paid for
(`test_the_output_length_boundary_is_inclusive`,
`test_the_judge_rubric_length_boundary_is_inclusive`)."""

KAPPA_SCALE: Final[int] = 3
"""Decimal places the kappa is published at (decision F).

Three and not the metric set's `ACCURACY_SCALE`, because the threshold is `0.7`
on a scale running `-1` to `1`: two places cannot separate `0.699` from `0.700`,
which is exactly the pair `JUDGE_TRUST_MIN` has to decide between
(`test_a_kappa_below_the_trust_floor_is_not_trusted`)."""

JUDGE_TRUST_MIN: Final[float] = 0.7
"""The lowest kappa that is still trusted, INCLUSIVE.

Read with `>=` and never `>`: a judge that met the published bar exactly is
trusted, and `0.699` is not (`test_a_kappa_of_exactly_the_trust_floor_is_trusted`,
`test_a_kappa_below_the_trust_floor_is_not_trusted`). A `None` kappa is never
trusted, whatever the comparison would do with it."""

MAX_LABELS_NAME_LENGTH: Final[int] = 128
"""The longest `labels_name` a READ artifact may carry, inclusive.

a project rule on the one string in this document an operator chose: it is the
name of the file they calibrated from, and 17f copies it into `report.json` and
into the `eval_runs.metrics` `jsonb` - neither of which has a length of its own.
`MAX_LABEL_ID_LENGTH`'s integer and not its constant, because the two cap
different things and sharing one would tie a file name to an id vocabulary
(`test_a_labels_name_of_exactly_the_cap_is_accepted_and_one_over_is_refused`)."""

KAPPA_MIN: Final[float] = -1.0
"""The lowest kappa a read artifact may carry, INCLUSIVE.

`-1.0` is a real weighted kappa - a judge that inverts the human every time - and
the most informative calibration there is, so the floor is the scale's own end
and never `JUDGE_TRUST_MIN` (`test_a_kappa_at_either_bound_is_accepted`)."""

KAPPA_MAX: Final[float] = 1.0
"""The highest kappa a read artifact may carry, INCLUSIVE.

Both bounds are checked on the way IN rather than trusted from the file: the
number lands in `eval_runs.judge_kappa`, a `double precision` column with no
bounds of its own, so a hand-edited `9.9` would otherwise publish a judge as
superhumanly agreeing and the eval gate's comment would carry it as evidence
(`test_a_kappa_one_thousandth_past_either_bound_is_refused`)."""

CALIBRATION_JSON_NAME: Final[str] = "calibration.json"
"""The one file a calibration writes, named by a constant and never derived from
a label id or a clock - `read_calibration` and `ops-eval run --calibration` read
it back by this name
(`test_the_artifact_carries_exactly_the_published_keys`,
`test_read_calibration_round_trips_what_write_calibration_wrote`)."""

CALIBRATION_INDENT: Final[int] = 2
"""How `calibration.json` is indented, `report.py`'s `REPORT_INDENT` again: the
artifact is read by a human debugging a bad kappa before it is read by anything
else."""

UNDEFINED_KAPPA_REASON: Final[str] = (
    "no expected disagreement: one of the two raters gave the same score to every label, "
    "so chance agreement is total and the kappa is undefined"
)
"""Why a published kappa is `null` (decision C).

The artifact's only clue for a reader who finds an empty column, and the
distinction a design note draws one level down: "nothing measured this" is not
"everything failed it" (`test_a_constant_table_reports_no_kappa_and_is_not_trusted`)."""


class CalibrationSetInvalid(Exception):
    """A labels file could not be loaded.

    `EvalSetInvalid`'s shape and its reasons: a plain exception rather than a
    `GatewayError`, because an unloadable file is an operator error reported to
    whoever ran the harness and never an answer to a caller, and
    `tests/gateway/test_error_contract.py` walks `GatewayError.__subclasses__()`
    asserting one class per wire code.

    `line` is the 1-based FILE line, comments and blank lines included, or `None`
    when the file itself is at fault - which is what the
    `MIN_CALIBRATION_LABELS` refusal is
    (`test_a_comment_and_a_blank_line_are_skipped_and_the_file_line_is_still_reported`).
    """

    def __init__(self, reason: str, *, path: Path, line: int | None = None) -> None:
        self.reason = reason
        self.path = path
        self.line = line
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {reason}")


class CalibrationRefused(Exception):
    """Too few labels survived the judge for a calibration to mean anything.

    Separate from `CalibrationSetInvalid` because the file was fine: what failed
    is the judge, and the operator's next move is to look at the gateway rather
    than at the file (decision B,
    `test_seven_unscorable_labels_of_forty_six_refuses_and_writes_nothing`).
    """


def _checked_label_id(value: str) -> str:
    """Validate a label id: the gateway's vocabulary at this module's own cap.

    The pattern is imported rather than restated so a label id and a case id
    cannot drift into two vocabularies, and `re.fullmatch` and never `.match`: `$` also matches before a trailing newline in Python, and
    this value is written into `calibration.json`.
    """
    if len(value) > MAX_LABEL_ID_LENGTH:
        raise IdentifierInvalid(f"id: longer than {MAX_LABEL_ID_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid("id: not a lowercase slug")
    return value


class CalibrationLabel(BaseModel):
    """One human-graded reply: what was answered, what a human scored it, the rubric.

    `extra="forbid"`, so a misspelt or invented field is a refusal rather than a
    silently dropped column - a line carrying `{"human_score": 4, "score": 2}`
    would otherwise calibrate against a number nobody meant
    (`test_a_fifth_field_is_refused`).

    `judge_rubric` is inline on the label and not looked up in a golden set
    (decision A): `calibrate` then needs one file, with no pinned set SHA and no
    dependency on a set the labels may outlive. `note` is the one optional field,
    for the human who wants to say why they scored what they did; it is never
    sent to the judge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    output: str
    human_score: int
    judge_rubric: str
    note: str | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _checked_label_id(value)

    @field_validator("output", "judge_rubric")
    @classmethod
    def _check_text(cls, value: str, info: ValidationInfo) -> str:
        """a project rule on the two strings that are SENT, capped at one constant.

        Both fields on one validator so the pair cannot drift into two caps - one
        written for `output` only would leave the rubric unbounded, which is the
        likeliest shape of this defect
        (`test_the_output_length_boundary_is_inclusive`,
        `test_the_judge_rubric_length_boundary_is_inclusive`).
        """
        if len(value) > MAX_LABEL_TEXT_LENGTH:
            raise ValueError(f"{info.field_name}: longer than {MAX_LABEL_TEXT_LENGTH} characters")
        return value

    @field_validator("human_score")
    @classmethod
    def _check_human_score(cls, value: int) -> int:
        """The judge's own bounds, inclusive, imported rather than restated.

        A 0 or a 6 is a typo, and the weights would take it as a real category -
        a `(6, 1)` cell weighs 25/16, larger than the formula's own maximum, so
        one mistyped digit moves the whole agreement
        (`test_a_human_score_outside_the_scale_is_refused`).
        """
        if not JUDGE_MIN_SCORE <= value <= JUDGE_MAX_SCORE:
            raise ValueError(f"human_score: outside {JUDGE_MIN_SCORE}..{JUDGE_MAX_SCORE}")
        return value


class CalibrationRow(BaseModel):
    """One scored label as the artifact records it: the human's number and the judge's.

    The audit trail a bad kappa is debugged from - without it the artifact is a
    single number with nothing behind it. One row per SCORED label, in file
    order, so a reader can find the record beside the id
    (`test_the_artifact_carries_exactly_the_published_keys`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    human_score: int
    judge_median: int


class CalibrationReport(BaseModel):
    """One calibration: what was scored, how well it agreed, and whether to trust it.

    `judge_prompt_version` is 17c's sha256 of the prompt file's bytes, and it is
    the field that keeps this document honest across time: a calibration computed
    under one prompt and read against a run made under another is a stale number
    worn as trust. Nothing checks that HERE - `read_calibration` validates the
    document and `evals/cli.py` refuses the mismatch, before the run starts (17f
    decision D, `test_a_calibration_computed_against_another_prompt_is_refused_before_the_run`).

    `kappa` is `None` when chance agreement was total, and `reason` says so;
    `trusted` is `False` whenever `kappa` is `None`, whatever the comparison
    would have done with it (decision C).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_prompt_version: str
    labels_name: str
    labels_sha: str
    computed_at: datetime
    pairs: int
    dropped: int
    kappa: float | None
    reason: str | None
    trusted: bool
    rows: tuple[CalibrationRow, ...]


def load_labels(path: Path) -> tuple[CalibrationLabel, ...]:
    """Read a labels file, in file order, or raise.

    `cases.load_golden_set`'s shape, deliberately: blank lines and lines whose
    first non-space character is `#` are skipped so a file can be commented, the
    reported line number is still the FILE's - which is what makes it findable in
    an editor - and every per-line failure becomes a `CalibrationSetInvalid`
    naming that line with the original as its `__cause__`, so a caller catches one
    type to report one file problem. Loaded whole or not at all
    (`test_nothing_is_returned_when_a_later_line_is_bad`): forty of forty-one
    labels is a calibration that silently excluded the record an author most
    wants to look at.

    The `MIN_CALIBRATION_LABELS` floor is HERE and not in `calibrate` (decision
    B): downstream it would cost 39 labels times `JUDGE_SAMPLES` of real spend
    before the refusal (`test_a_file_of_thirty_nine_labels_is_refused_before_any_request`
    asserts that no request was made).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    labels: list[CalibrationLabel] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CalibrationSetInvalid(
                f"is not valid JSON: {exc.msg}", path=path, line=number
            ) from exc
        try:
            label = CalibrationLabel.model_validate(payload)
        except (ValidationError, IdentifierInvalid) as exc:
            raise CalibrationSetInvalid(
                f"is not a valid label: {exc}", path=path, line=number
            ) from exc
        if label.id in seen:
            raise CalibrationSetInvalid(
                f"repeats the id {label.id!r}, first seen on line {seen[label.id]}",
                path=path,
                line=number,
            )
        seen[label.id] = number
        labels.append(label)

    if len(labels) < MIN_CALIBRATION_LABELS:
        raise CalibrationSetInvalid(
            f"carries {len(labels)} labels, fewer than the {MIN_CALIBRATION_LABELS} "
            "a calibration needs",
            path=path,
        )
    return tuple(labels)


def _disagreement(left: int, right: int) -> int:
    """The numerator of the quadratic weight `w(i, j)`.

    The weight itself is `(i - j)**2 / (JUDGE_MAX_SCORE - JUDGE_MIN_SCORE)**2`,
    that denominator being 16 on the 1..5 scale. It is carried here as the
    numerator alone because the denominator appears in BOTH sums of
    `1 - (Sigma w*O) / (Sigma w*E)` and cancels out of the ratio exactly - which
    is also why no vector in `tests/evals/test_calibration.py` can pin its value:
    every choice of denominator gives the same kappa. What the vectors do pin is
    that the weight is quadratic (`test_one_step_of_disagreement_is_zero_point_nine_two_nine`).
    """
    return (left - right) ** 2


def weighted_kappa(human: Sequence[int], judge: Sequence[int]) -> float | None:
    """Cohen's kappa with quadratic weights over the 1..5 scale, or `None`.

    `1 - (Sigma w*O) / (Sigma w*E)`, computed as a ratio of INTEGERS and only
    then divided: both sums carry the same weight denominator and the same
    number of pairs, so the whole expression reduces to
    `1 - N * Sigma_pairs (h - j)**2 / Sigma_ij n_h(i) n_j(j) (i - j)**2` with no
    floating point anywhere. That matters at exactly one place - the tie in
    `test_the_kappa_rounds_half_up_and_not_to_the_nearest_even`, where a kappa of
    0.8125 computed through two inexact divisions lands a digit either side of
    the tie and rounds by luck.

    Quantized at `KAPPA_SCALE` places with `ROUND_HALF_UP` through `Decimal` and
    never through `round()`, which is half-to-even - the house rule every number
    the harness publishes follows (`metrics._percentage` is the same three lines).
    The answer is a `float` because that is what `report.py` serialises as a JSON
    number; the arithmetic that produced it never was one.

    `None` when `Sigma w*E` is zero: one of the two raters gave every label the
    same score, so chance agreement is total and the ratio is undefined
    (`test_a_table_with_no_expected_disagreement_has_no_kappa`).
    """
    pairs = len(human)
    observed = sum(_disagreement(left, right) for left, right in zip(human, judge, strict=True))
    human_counts = Counter(human)
    judge_counts = Counter(judge)
    expected = sum(
        human_count * judge_count * _disagreement(left, right)
        for left, human_count in human_counts.items()
        for right, judge_count in judge_counts.items()
    )
    if expected == 0:
        return None
    kappa = Decimal(1) - Decimal(observed * pairs) / Decimal(expected)
    return float(kappa.quantize(Decimal(1).scaleb(-KAPPA_SCALE), rounding=ROUND_HALF_UP))


def calibrate(
    labels: Sequence[CalibrationLabel],
    *,
    judge: JudgeClient,
    labels_name: str,
    labels_sha: str,
) -> CalibrationReport:
    """Score every label through the judge and report the agreement.

    One `score_reply(rubric, reply)` per label, in file order, on the caller's own
    client (decision E): calibration has no `EvalCase` and no `CaseOutcome`, and
    building synthetic ones would make two records' required fields part of this
    module's contract for no reason
    (`test_calibrate_reaches_the_judge_through_score_reply_with_the_labels_own_strings`).

    A label the judge could not score - 17c's unusable sample, retried once and
    then `None` - is dropped and counted, never scored as a zero and never as the
    human's own number (decision D). What survives is checked against
    `MIN_CALIBRATION_LABELS` again, and a run that falls below it raises
    `CalibrationRefused` and writes nothing: a floor checked only against the
    file's record count would publish a kappa over 39 pairs the moment a judge
    got flaky (`test_seven_unscorable_labels_of_forty_six_refuses_and_writes_nothing`).
    """
    rows: list[CalibrationRow] = []
    dropped = 0
    for label in labels:
        verdict = judge.score_reply(label.judge_rubric, label.output)
        if verdict is None:
            dropped += 1
            continue
        rows.append(
            CalibrationRow(id=label.id, human_score=label.human_score, judge_median=verdict.median)
        )

    if len(rows) < MIN_CALIBRATION_LABELS:
        raise CalibrationRefused(
            f"only {len(rows)} of {len(labels)} labels could be scored, fewer than the "
            f"{MIN_CALIBRATION_LABELS} a calibration needs"
        )

    kappa = weighted_kappa([row.human_score for row in rows], [row.judge_median for row in rows])
    return CalibrationReport(
        judge_prompt_version=judge_prompt_version(),
        labels_name=labels_name,
        labels_sha=labels_sha,
        computed_at=datetime.now(UTC),
        pairs=len(rows),
        dropped=dropped,
        kappa=kappa,
        reason=None if kappa is not None else UNDEFINED_KAPPA_REASON,
        trusted=kappa is not None and kappa >= JUDGE_TRUST_MIN,
        rows=tuple(rows),
    )


def read_calibration(path: Path) -> CalibrationReport:
    """Read one `calibration.json` back, or raise `CalibrationSetInvalid` naming the path.

    The other half of `write_calibration`, and the only reader this artifact has:
    17f's `ops-eval run --calibration` hands a run what a calibration measured, so
    every way the file can be wrong has to become ONE typed refusal the CLI's
    existing pre-check block prints as a single line - a missing file, a
    half-written one, a document from a schema nobody can identify, and a number
    outside the scale
    (`test_an_artifact_that_does_not_exist_is_refused_naming_the_path`,
    `test_an_artifact_that_is_not_json_is_refused_naming_the_path`,
    `test_an_artifact_with_a_field_the_model_forbids_is_refused`).

    Validated through `CalibrationReport` itself with its `extra="forbid"`, and
    never by picking the two fields a run needs out of the document: a reader that
    took `kappa` and `trusted` alone would accept a hand-edited file and publish
    trust computed by something nobody can retrace
    (`test_read_calibration_round_trips_what_write_calibration_wrote`).

    The two bounds are applied AFTER that validation, because both are facts about
    values the model's own types admit: `kappa` is a `float` and `labels_name` a
    `str`, and both land in a stored column - `eval_runs.judge_kappa` and the
    `metrics` `jsonb` - that has no bound of its own.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationSetInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalibrationSetInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc

    try:
        report = CalibrationReport.model_validate(payload)
    except ValidationError as exc:
        raise CalibrationSetInvalid(f"is not a calibration: {exc}", path=path) from exc

    if report.kappa is not None and not KAPPA_MIN <= report.kappa <= KAPPA_MAX:
        raise CalibrationSetInvalid(
            f"carries a kappa of {report.kappa}, outside {KAPPA_MIN}..{KAPPA_MAX}", path=path
        )
    if len(report.labels_name) > MAX_LABELS_NAME_LENGTH:
        raise CalibrationSetInvalid(
            f"names a labels file longer than {MAX_LABELS_NAME_LENGTH} characters", path=path
        )
    return report


def write_calibration(out: Path, report: CalibrationReport) -> None:
    """Write exactly `CALIBRATION_JSON_NAME` into `out`, overwriting what is there.

    One file, named by a constant and never derived from a label id or a clock,
    so 17f reads a known path and a rerun into the same directory leaves one
    document that parses. Nothing is caught here: a write that fails is the
    caller's to report as a usage error naming the path, rather than a traceback
    over a calibration already paid for
    (`test_the_artifact_carries_exactly_the_published_keys`).
    """
    (out / CALIBRATION_JSON_NAME).write_text(
        report.model_dump_json(indent=CALIBRATION_INDENT) + "\n", encoding="utf-8"
    )
