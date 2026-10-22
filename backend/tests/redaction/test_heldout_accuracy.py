r"""The held-out redaction set: its shape, its honesty, its per-kind numbers and the README.

The 57-case fixture is the tuning set, so its score says little. The held-out set is 600
cases built in memory from templates nobody tuned a detector against.
It has no floor on purpose: a floor would invite tuning to it.

The 600 cases are scored once per session in `heldout_scores`, so this file stays well
under a second of work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from app.redaction import Kind
from app.redaction.detectors import (
    compact_separators,
    iban_ok,
    luhn_ok,
    ppsn_ok,
    steuer_id_ok,
)

from .conftest import (
    FIXTURE_PATH,
    MISSES_PATH,
    README_PATH,
    Case,
    KindScore,
    Problem,
    emitted_detections,
    f2_score,
    list_problems,
    load_report_module,
    pooled_score,
    readme_table_block,
    score_by_kind,
)
from .heldout import (
    HELDOUT_SEED,
    TEMPLATES_PATH,
    build_heldout_cases,
    template_fixed_texts,
)

DETECTOR_TESTS_PATH = Path(__file__).resolve().parent / "test_detectors.py"


class HeldoutScores(NamedTuple):
    per_kind: Mapping[Kind, KindScore]
    pooled: KindScore
    problems: Sequence[Problem]


@pytest.fixture(scope="session")
def heldout() -> tuple[Case, ...]:
    return build_heldout_cases()


@pytest.fixture(scope="session")
def heldout_scores(heldout: tuple[Case, ...]) -> HeldoutScores:
    per_kind = score_by_kind(heldout, emitted_detections)
    return HeldoutScores(
        per_kind=per_kind,
        pooled=pooled_score(per_kind),
        problems=list_problems(heldout, emitted_detections),
    )


def row_cells(table: str, label: str) -> list[str]:
    """The cells of the table row whose first cell is `label`."""
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    pytest.fail(f"no row for {label!r} in the table:\n{table}")


def empty_scores() -> dict[Kind, KindScore]:
    return {
        kind: KindScore(kind=kind, true_positives=0, false_positives=0, false_negatives=0)
        for kind in Kind
    }


# --------------------------------------------------------------------------
# the set itself
# --------------------------------------------------------------------------


def test_the_seed_and_template_file_are_the_ones_the_adr_names() -> None:
    """Catches a changed seed or a moved template file: a design note publishes seed 82 and the
    templates path, and a different seed would quietly give different published numbers."""
    assert HELDOUT_SEED == 82
    assert TEMPLATES_PATH.name == "heldout-templates.json"
    assert TEMPLATES_PATH.parent.name == "redaction"
    assert TEMPLATES_PATH.exists()


def test_the_heldout_set_has_fifty_positives_per_kind_and_one_hundred_negatives(
    heldout: tuple[Case, ...],
) -> None:
    """Catches a set that is lopsided (a kind with 10 cases scores on noise), has fewer
    negatives (precision would be free), or whose positives carry more than one span (the
    per-kind counts below would stop meaning 50 values)."""
    positives = [case for case in heldout if case.spans]
    negatives = [case for case in heldout if not case.spans]

    assert len(heldout) == 600
    assert len(negatives) == 100
    assert all(len(case.spans) == 1 for case in positives)
    assert Counter(case.spans[0].kind for case in positives) == {kind: 50 for kind in Kind}
    assert len({case.id for case in heldout}) == 600


def test_the_heldout_set_is_the_same_on_every_build() -> None:
    """Catches a generator that reads the clock or an unseeded random source: the README
    numbers and the misses page are checked into the repo, so two builds must agree."""
    assert build_heldout_cases() == build_heldout_cases()


def test_every_generated_checksummed_value_passes_its_checksum(
    heldout: tuple[Case, ...],
) -> None:
    """Catches a generator that makes a card that fails Luhn or an IBAN that fails mod-97:
    ground truth no correct detector can reach would be charged as a miss forever.

    Every kind is checked with its spaces taken out: a tax id or PPS number printed in
    groups is still the same number, and the set writes some that way on purpose, since
    people do. Whether the engine finds the grouped form is what the score measures."""
    checks: dict[Kind, tuple[Callable[[str], bool], bool]] = {
        Kind.CARD: (luhn_ok, True),
        Kind.IBAN: (iban_ok, True),
        Kind.STEUER_ID: (steuer_id_ok, True),
        Kind.PPSN: (ppsn_ok, True),
    }
    checked = Counter[Kind]()
    for case in heldout:
        for span in case.spans:
            entry = checks.get(span.kind)
            if entry is None:
                continue
            check, compact = entry
            value = compact_separators(span.value) if compact else span.value
            assert check(value) is True, f"{case.id}: {span.value}"
            checked[span.kind] += 1

    assert checked == {kind: 50 for kind in checks}


def leaks(fixed_texts: Sequence[str], haystacks: Sequence[str]) -> list[str]:
    """The fixed texts that appear (ignoring case) in any haystack."""
    lowered = [haystack.lower() for haystack in haystacks]
    return [text for text in fixed_texts if any(text.lower() in haystack for haystack in lowered)]


def test_no_heldout_template_appears_in_the_tuning_fixture_or_detector_tests() -> None:
    """Catches a held-out template copied from the tuning fixture or the detector tests:
    a detector tuned to that sentence would score well on a set that is no longer held out.
    Each fixed text is at least 10 characters so a short word cannot hide a copy."""
    fixed = template_fixed_texts()
    tuning = [
        FIXTURE_PATH.read_text(encoding="utf-8"),
        DETECTOR_TESTS_PATH.read_text(encoding="utf-8"),
    ]

    assert len(fixed) > 0
    assert all(len(text) >= 10 for text in fixed)
    assert leaks(fixed, tuning) == []


def test_the_leak_check_catches_a_fixed_text_found_in_a_tuning_source() -> None:
    """Catches a leak check that always answers 'none': a fixed text planted inside a
    haystack, in any letter case, must be reported."""
    fixed = ["please call me back soon", "an unrelated sentence here"]

    found = leaks(fixed, ['{"text": "Hello. PLEASE CALL ME BACK SOON, thanks"}'])

    assert found == ["please call me back soon"]


# --------------------------------------------------------------------------
# the scoring
# --------------------------------------------------------------------------


def test_f2_weighs_recall_four_times() -> None:
    """Catches F1 or a swapped formula: half precision with full recall is 0.833 and full
    precision with half recall is 0.556, so the two must differ and recall must win."""
    assert round(f2_score(0.5, 1.0), 3) == 0.833
    assert round(f2_score(1.0, 0.5), 3) == 0.556


def test_a_kind_never_reported_has_precision_shown_as_na() -> None:
    """Catches precision shown as 0 or 1 for a kind the engine never reported, and a recall
    that is not 0.000 when it missed everything it was meant to find. Every other kind is
    perfect, so only the kind under test can make its row read `n/a`."""
    per_kind = empty_scores()
    per_kind[Kind.IPV6] = KindScore(
        kind=Kind.IPV6, true_positives=0, false_positives=0, false_negatives=5
    )
    pooled = pooled_score(per_kind)
    render_table = load_report_module().render_table

    table: str = render_table(per_kind, pooled)

    assert per_kind[Kind.IPV6].precision is None
    assert per_kind[Kind.IPV6].recall == 0.0
    assert row_cells(table, Kind.IPV6.value) == [Kind.IPV6.value, "n/a", "0.000", "0.000", "5"]


def test_the_table_has_a_row_per_kind_in_enum_order_then_an_all_row() -> None:
    """Catches a table that drops a kind, reorders the kinds, leaves out the pooled row, or
    counts the wrong number as misses (it must be the false negatives, here 1)."""
    per_kind = empty_scores()
    per_kind[Kind.EMAIL] = KindScore(
        kind=Kind.EMAIL, true_positives=3, false_positives=1, false_negatives=1
    )
    pooled = pooled_score(per_kind)

    table: str = load_report_module().render_table(per_kind, pooled)

    lines = table.strip().splitlines()
    assert lines[0] == "| Kind | Precision | Recall | F2 | Misses |"
    labels = [line.strip().strip("|").split("|")[0].strip() for line in lines[2:]]
    assert labels == [*(kind.value for kind in Kind), "all"]
    assert row_cells(table, Kind.EMAIL.value)[1:] == ["0.750", "0.750", "0.750", "1"]
    assert row_cells(table, "all")[1:] == ["0.750", "0.750", "0.750", "1"]


# --------------------------------------------------------------------------
# the README and the misses page
# --------------------------------------------------------------------------


def test_readme_table_matches_the_heldout_numbers(heldout_scores: HeldoutScores) -> None:
    """Catches a hand-edited README number: the block must equal the table rendered from the
    held-out scores as they stand, with all 10 kinds and the `all` row."""
    render_table = load_report_module().render_table
    expected: str = render_table(heldout_scores.per_kind, heldout_scores.pooled)
    block = readme_table_block(README_PATH.read_text(encoding="utf-8"))

    assert block == expected.strip("\n")
    for kind in Kind:
        row_cells(block, kind.value)
    row_cells(block, "all")


def test_the_misses_page_lists_every_miss_and_false_find(
    heldout_scores: HeldoutScores,
) -> None:
    """Catches a misses page that was not regenerated after a detector change, or one that
    shows only some of the problems: it must equal what the report script renders."""
    render_misses = load_report_module().render_misses
    expected: str = render_misses(heldout_scores.problems)

    assert MISSES_PATH.read_text(encoding="utf-8") == expected


def test_readme_no_longer_presents_the_tuning_fixture_as_the_quality_number() -> None:
    """Catches the old '1.000 on 57 cases' claim surviving beside the new table: the
    generated block must not name the 57-case tuning set next to its numbers."""
    block = readme_table_block(README_PATH.read_text(encoding="utf-8"))

    assert "Precision" in block
    assert "57 cases" not in block
    assert "57-case" not in block


def test_the_report_script_checks_the_misses_page_too(tmp_path: Path) -> None:
    """Catches a `--check` that looks only at the README: the README copy is current (proved
    by the first call returning 0), so only the stale misses page can make the second call
    return 1. A plain run then rewrites the page back."""
    report = load_report_module()
    readme = tmp_path / "README.md"
    readme.write_text(README_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    misses = tmp_path / "redaction-misses.md"
    misses.write_text(MISSES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    original = misses.read_text(encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setattr(report, "README_PATH", readme)
    patch.setattr(report, "MISSES_PATH", misses)
    try:
        assert report.main(["--check"]) == 0
        misses.write_text(original + "\nstale line\n", encoding="utf-8")
        assert report.main(["--check"]) == 1
        assert report.main([]) == 0
    finally:
        patch.undo()

    assert misses.read_text(encoding="utf-8") == original
