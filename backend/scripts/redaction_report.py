#!/usr/bin/env python
"""Regenerate the README's redaction accuracy table from the test fixture.

The numbers in the README are generated, never typed. A
hand-written accuracy figure rots the first time a pattern changes and nothing
goes red, so the table is produced here and
`backend/tests/redaction/test_fixture_accuracy.py::test_readme_table_matches_the_computed_numbers`
compares the committed block against what this module renders. Both sides round
through `backend/tests/redaction/conftest.py::round_metric`, so the table and the
asserted floors cannot disagree by a half-ulp.

Run it after any change to a detector or to the fixture:

    uv run --project backend python backend/scripts/redaction_report.py

It rewrites only the text between the two markers in `README.md` and prints the
counts to stderr. With `--check` it writes nothing and exits 1 on a difference,
which is the same comparison the drift test makes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "backend" / "tests"
README_PATH = REPO_ROOT / "README.md"

TABLE_START = "<!-- redaction-table:start -->"
TABLE_END = "<!-- redaction-table:end -->"


def render_table(precision: float, recall: float) -> str:
    """The markdown block that sits between the two README markers.

    Both numbers arrive already rounded by `round_metric`; the format string only
    fixes how many decimals are shown, so a value that rounded to 1.0 prints as
    `1.000` rather than `1.0`.
    """
    return "\n".join(
        (
            "| Metric | Value |",
            "| --- | --- |",
            f"| Precision | {precision:.3f} |",
            f"| Recall | {recall:.3f} |",
        )
    )


def compute() -> tuple[float, float, int, int, int]:
    """Score the fixture with the shipped detectors.

    Imports the test helpers rather than reimplementing the scorer: two scorers
    would be two definitions of precision, and the README would eventually report
    the other one.
    """
    sys.path.insert(0, str(TESTS_ROOT))
    from redaction.conftest import (
        emitted_detections,
        load_cases,
        round_metric,
        score_cases,
    )

    cases = load_cases()
    score = score_cases(cases, emitted_detections)
    return (
        round_metric(score.precision),
        round_metric(score.recall),
        len(cases),
        score.true_positives + score.false_negatives,
        sum(1 for case in cases if not case.spans),
    )


def replace_block(readme: str, table: str) -> str:
    """Swap the text between the markers, leaving the markers themselves in place."""
    start = readme.index(TABLE_START) + len(TABLE_START)
    end = readme.index(TABLE_END)
    return f"{readme[:start]}\n{table}\n{readme[end:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if README.md is out of date instead of rewriting it",
    )
    args = parser.parse_args(argv)

    precision, recall, case_count, span_count, negative_count = compute()
    table = render_table(precision, recall)
    readme = README_PATH.read_text(encoding="utf-8")
    updated = replace_block(readme, table)

    print(
        f"{case_count} cases ({negative_count} negative), {span_count} expected spans: "
        f"precision {precision:.3f}, recall {recall:.3f}",
        file=sys.stderr,
    )
    if args.check:
        if updated != readme:
            print("README.md redaction table is out of date", file=sys.stderr)
            return 1
        return 0
    README_PATH.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
