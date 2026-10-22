r"""The 57-case fixture: its composition, its checksums, the floors and README drift.

Two of these tests guard the fixture rather than the engine. That is deliberate:
precision and recall are only meaningful if the ground truth is real, and a
fixture whose "card number" fails Luhn or whose eleven negatives quietly became
ten would move both numbers without any detector changing.

The floors are asserted as exact literals (decision C, a project rule), not as
a band: `>= 0.95` and `>= 0.90` are the numbers #21 and the design name.

Precision and recall are scored on what `redact` EMITS, not on `detect_all`'s
raw candidates - see `emitted_detections` in `conftest.py` for why, and
`test_the_score_measures_what_the_engine_emits_not_raw_candidates` below for
the case that forces the distinction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.redaction import Detection, Kind, NameDirectory, redact
from app.redaction.detectors import (
    compact_separators,
    detect_all,
    iban_ok,
    luhn_ok,
    ppsn_ok,
    steuer_id_ok,
)

from .conftest import (
    MISSES_PATH,
    README_PATH,
    Case,
    ExpectedSpan,
    directory_for,
    emitted_detections,
    load_report_module,
    pooled_score,
    readme_table_block,
    round_metric,
    score_by_kind,
    score_cases,
)
from .heldout import build_heldout_cases

# Decision F, pinned exactly: without the eleven negatives precision is 1.0 by
# construction, so the count is part of what makes the number mean anything.
# Round 2 of this plan's review grew the positives from 40 to 42: the fixture had
# no space-grouped IBAN and no IPv4-mapped IPv6, the two forms whose patterns
# matched only a PREFIX of the value and left the rest in the clear. A score
# cannot see what the fixture omits, so the two rows are the fix, not the
# pattern change alone.
# An earlier change decision I adds the tenth kind's ground truth: four positive PERSON
# cases (a single given name, a full name winning over its own first token, a
# name carrying regex metacharacters, and a name overlapping an email address)
# and one negative, a directory name sitting inside a longer word. 42/10 becomes
# 46/11, and the negative is what stops PERSON buying recall at precision's
# expense.
EXPECTED_CASE_COUNT = 57
EXPECTED_POSITIVE_CASES = 46
EXPECTED_NEGATIVE_CASES = 11

# The negative shapes decision F names by hand. Ids, not texts, so a case can be
# reworded without this list rotting, but cannot be deleted.
REQUIRED_NEGATIVE_IDS = (
    "neg-bare-number",
    "neg-semantic-version",
    "neg-order-id",
    "neg-sixteen-digits-fail-luhn",
    "neg-date-without-cue",
    "neg-time-of-day",
    "neg-short-numeric-run",
    "neg-directory-name-inside-a-longer-word",
)


def test_fixture_has_forty_six_positive_and_eleven_negative_cases(
    cases: tuple[Case, ...],
) -> None:
    """Catches a fixture that grew its positives without its negatives, which
    raises precision for free - which is exactly what adding a kind whose only
    check is membership in a list invites."""
    positives = [case for case in cases if case.spans]
    negatives = [case for case in cases if not case.spans]

    assert len(cases) == EXPECTED_CASE_COUNT
    assert len(positives) == EXPECTED_POSITIVE_CASES
    assert len(negatives) == EXPECTED_NEGATIVE_CASES


def test_fixture_case_ids_are_unique(cases: tuple[Case, ...]) -> None:
    """Catches a duplicated id, which makes a failure report point at two rows and
    hides one of them."""
    ids = [case.id for case in cases]

    assert len(set(ids)) == len(ids)


def test_fixture_covers_every_detector_kind(cases: tuple[Case, ...]) -> None:
    """Catches a kind with no ground truth at all: its recall would be unmeasured
    while the pooled number still looked healthy."""
    covered = {span.kind for case in cases for span in case.spans}

    assert covered == set(Kind)


@pytest.mark.parametrize("case_id", REQUIRED_NEGATIVE_IDS)
def test_required_negative_shape_is_present_and_has_no_spans(
    cases: tuple[Case, ...], case_id: str
) -> None:
    """Catches the removal of a negative that a widened pattern would start
    matching - the exact rows decision F names as the precision pressure."""
    by_id = {case.id: case for case in cases}

    assert case_id in by_id
    assert by_id[case_id].spans == ()


def test_every_person_span_belongs_to_a_case_that_names_a_directory(
    cases: tuple[Case, ...],
) -> None:
    """Catches a PERSON case added with ground truth and no `directory` key.

    The matcher's input is not the text, so such a case is unreachable by any correct
    engine: every one of its spans scores as a false negative and recall drops for a
    reason no detector change can fix. `test_fixture_covers_every_detector_kind` forces
    the case to exist and says nothing about what it needs to be scored.
    """
    for case in cases:
        if any(span.kind is Kind.PERSON for span in case.spans):
            assert case.directory, case.id


def test_the_directory_field_is_only_on_the_cases_that_need_one(
    cases: tuple[Case, ...],
) -> None:
    """Catches a directory pasted onto every case, which would make the nine pattern kinds
    scored against a matcher no request of theirs compiles: a name that happened to sit in
    another case's text would become a false positive there and quietly lower precision
    for a change to a case it has nothing to do with."""
    with_directory = {case.id for case in cases if case.directory}
    with_person = {
        case.id for case in cases if any(span.kind is Kind.PERSON for span in case.spans)
    }

    assert with_person <= with_directory
    assert with_directory - with_person == {"neg-directory-name-inside-a-longer-word"}


def test_the_scorer_hands_each_case_its_own_compiled_directory() -> None:
    """Catches a scorer that passes `None` for every case, and one that pools every case's
    names into a single directory.

    `None` for all is the failure that would make the PERSON cases score zero recall while
    the engine was correct; pooling is the one that would charge unrelated cases false
    positives. Both are invisible to the floors, which move for a dozen other reasons, so
    the seam is asserted directly against a recording detector over two hand-built cases.
    """
    named = Case(id="named", text="x", spans=(), directory=("Anna Schmidt",))
    plain = Case(id="plain", text="x", spans=())
    seen: list[tuple[str, tuple[str, ...] | None]] = []

    def record(text: str, directory: NameDirectory | None) -> list[Detection]:
        if directory is None:
            seen.append((text, None))
        else:
            seen.append((text, tuple(d.value for d in directory.detect("cc Anna Schmidt"))))
        return []

    score_cases([named, plain], record)

    assert seen == [("x", ("Anna Schmidt",)), ("x", None)]


def test_every_checksummed_fixture_value_actually_passes_its_checksum(
    cases: tuple[Case, ...],
) -> None:
    """Catches a hand-typed specimen: a card that fails Luhn or an IBAN that fails
    mod-97 is ground truth no correct detector can ever reach, so recall would be
    capped below the floor by the fixture itself.

    The separators are stripped through `compact_separators`, the same function
    the detectors use, so a space-grouped IBAN row is checked the way the engine
    checks it rather than by a second definition of "compact".
    """
    checks = {
        Kind.CARD: (luhn_ok, True),
        Kind.IBAN: (iban_ok, True),
        Kind.STEUER_ID: (steuer_id_ok, False),
        Kind.PPSN: (ppsn_ok, False),
    }
    for case in cases:
        for span in case.spans:
            entry = checks.get(span.kind)
            if entry is None:
                continue
            check, compact = entry
            value = compact_separators(span.value) if compact else span.value
            assert check(value) is True, f"{case.id}: {span.value}"


def test_recall_meets_the_floor(cases: tuple[Case, ...]) -> None:
    """Catches a detector set that misses a whole value form (the dotted German
    date, the spaced national phone number) while still scoring well on the easy
    rows."""
    score = score_cases(cases, emitted_detections)

    assert score.recall >= 0.95


def test_precision_meets_the_floor(cases: tuple[Case, ...]) -> None:
    """Catches an over-broad PHONE or IBAN pattern that eats ordinary numbers -
    the risk decision F's eleven negatives exist to expose."""
    score = score_cases(cases, emitted_detections)

    assert score.precision >= 0.90


def test_scoring_counts_a_mislabelled_span_as_both_a_miss_and_a_false_positive() -> None:
    """Catches a scorer that matches on offsets only.

    A right span with the wrong kind is not a detection: the placeholder would
    carry the wrong label and an earlier change would store it under the wrong class.
    Scored against a hand-built case, not the real fixture, so the row measures
    the scorer and cannot pass because the engine happened to find nothing.
    """
    case = Case(
        id="synthetic",
        text="x",
        spans=(ExpectedSpan(kind=Kind.CARD, start=0, end=1, value="x"),),
    )

    def mislabel(text: str, directory: NameDirectory | None) -> list[Detection]:
        return [Detection(kind=Kind.EMAIL, start=0, end=1, value="x")]

    score = score_cases([case], mislabel)

    assert (score.true_positives, score.false_positives, score.false_negatives) == (0, 1, 1)


def test_scoring_counts_an_exact_match_once() -> None:
    """Catches a scorer that can never score a true positive at all, which would
    make the row above pass for the wrong reason."""
    case = Case(
        id="synthetic",
        text="x",
        spans=(ExpectedSpan(kind=Kind.CARD, start=0, end=1, value="x"),),
    )

    def exact(text: str, directory: NameDirectory | None) -> list[Detection]:
        return [Detection(kind=Kind.CARD, start=0, end=1, value="x")]

    score = score_cases([case], exact)

    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 0, 0)
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_redact_and_detect_all_agree_on_every_positive_case(cases: tuple[Case, ...]) -> None:
    """Catches a `redact` that re-scans or filters differently from the resolved
    span list it reports, which would make the map disagree with the output."""
    for case in cases:
        result = redact(case.text, directory=directory_for(case))
        for found in result.detections:
            assert case.text[found.start : found.end] == found.value


def test_round_metric_rounds_to_three_decimals() -> None:
    """Catches a rounding change on one side of the README drift comparison: the
    table and the floor assertions must go through the same function or the drift
    test fails for a reason that is not drift."""
    assert round_metric(0.9666666) == 0.967
    assert round_metric(1.0) == 1.0
    assert round_metric(0.9) == 0.9


def test_readme_table_matches_the_computed_numbers() -> None:
    """Catches a hand-edited README number (decision D): the committed block must equal
    what `render_table` produces from the HELD-OUT set as it stands today, so
    a README that went back to scoring the tuning fixture would differ."""
    per_kind = score_by_kind(build_heldout_cases(), emitted_detections)
    render_table = load_report_module().render_table

    expected: str = render_table(per_kind, pooled_score(per_kind))

    assert readme_table_block(README_PATH.read_text(encoding="utf-8")) == expected.strip("\n")


def test_readme_carries_both_metric_names_in_the_generated_block() -> None:
    """Catches a table that reports one number or drops a kind: #99 asks for precision,
    recall and F2 for each kind, and a block with only recall would still match
    `render_table` if the script were wrong in the same way."""
    block = readme_table_block(README_PATH.read_text(encoding="utf-8"))

    assert "Precision" in block
    assert "Recall" in block
    assert "F2" in block
    for kind in Kind:
        assert f"| {kind.value} |" in block


# --------------------------------------------------------------------------
# the report script itself
# --------------------------------------------------------------------------

# Decision D's guarantee is "generated plus a drift test", and the drift test
# above computes the numbers itself rather than through the script. So the half a
# human actually runs -
# `uv run --project backend python backend/scripts/redaction_report.py` - had no
# test reaching it at all: `compute`, `replace_block` and `main` were 25 uncovered
# statements in a module this diff created.


def test_the_report_script_agrees_that_the_committed_readme_is_current() -> None:
    """Catches the script and the drift test drifting apart from each other.

    `--check` writes nothing and returns 0 only when regenerating the block would
    change nothing, so this row exercises `compute`, `render_table`,
    `replace_block` and `main` end to end against the committed README.
    """
    report = load_report_module()

    assert report.main(["--check"]) == 0


def _all_row_edited(readme: str) -> str:
    """The README with the pooled `all` row of the generated table replaced."""
    lines = readme.splitlines()
    for number, line in enumerate(lines):
        if line.startswith("| all |"):
            lines[number] = "| all | 0.500 | 0.500 | 0.500 | 0 |"
            return "\n".join(lines) + "\n"
    pytest.fail("README.md has no `| all |` row")


def test_the_report_script_reports_an_out_of_date_readme(tmp_path: Path) -> None:
    """Catches a `--check` that returns 0 unconditionally, which would make the row
    above pass no matter what the README said. The misses page is a current copy, so the
    README edit is the only thing that can turn the second answer into 1."""
    report = load_report_module()
    readme = README_PATH.read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme, encoding="utf-8")
    misses = tmp_path / "redaction-misses.md"
    misses.write_text(MISSES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(report, "README_PATH", target)
    monkeypatched.setattr(report, "MISSES_PATH", misses)
    try:
        assert report.main(["--check"]) == 0
        target.write_text(_all_row_edited(readme), encoding="utf-8")
        assert report.main(["--check"]) == 1
    finally:
        monkeypatched.undo()


def test_the_report_script_rewrites_only_the_block_between_the_markers(tmp_path: Path) -> None:
    """Catches a `replace_block` that drops the markers or the surrounding prose.

    Run against a copy, never the real README: a test that rewrote the committed
    file would make the drift test pass by editing its own subject.
    """
    report = load_report_module()
    original = README_PATH.read_text(encoding="utf-8")
    edited = _all_row_edited(original)
    assert edited != original
    target = tmp_path / "README.md"
    target.write_text(edited, "utf-8")
    misses = tmp_path / "redaction-misses.md"
    misses.write_text(MISSES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(report, "README_PATH", target)
    monkeypatched.setattr(report, "MISSES_PATH", misses)
    try:
        assert report.main([]) == 0
    finally:
        monkeypatched.undo()

    assert target.read_text(encoding="utf-8") == original


def test_the_score_measures_what_the_engine_emits_not_raw_candidates() -> None:
    """Catches the scoring subject being switched back to `detect_all`.

    A space-grouped IBAN legitimately produces `CARD` and `PHONE` sub-spans as raw
    candidates - the digit tail of an IBAN looks like both - and the overlap merge
    drops them. Scored on raw candidates the engine would be charged two false
    positives for values it never emits; scored on what it emits there is exactly
    one detection, the whole IBAN.
    """
    text = "pay DE12 5001 0517 0648 4898 90 today"

    raw_kinds = sorted({d.kind for d in detect_all(text)})
    emitted = emitted_detections(text)

    assert raw_kinds == sorted({Kind.IBAN, Kind.CARD, Kind.PHONE})
    assert [(d.kind, d.value) for d in emitted] == [(Kind.IBAN, "DE12 5001 0517 0648 4898 90")]
