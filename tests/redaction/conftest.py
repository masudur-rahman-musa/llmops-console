r"""Fixtures and scoring helpers for the redaction engine tests.

Everything here is in-process: one JSON file read and pure CPU work. There is no
backing service, so a project rule (session-scoped fixtures, isolation by
namespace) has nothing to apply to. The case list is read once at session scope
anyway, because it is immutable data and re-reading it 57 times buys nothing.

Two helpers carry the load-bearing claims:

* `load_cases` asserts `text[start:end] == value` for every span, so an offset
  that rots when a case's prose is edited is a collection-time failure rather
  than a silent drop in recall.
* `placeholders_in_order` walks the ORIGINAL text and the redacted text in
  lockstep, so it both extracts the placeholder that replaced each detection and
  asserts every character outside a detection survived verbatim. A helper that
  simply regex-scanned the output for `<KIND_N>` would discard exactly the thing
  the ordering and preservation claims are about.

`round_metric` is the ONE rounding in this plan: `test_fixture_accuracy.py` and
`scripts/redaction_report.py` both take their numbers through it, so the README
table and the asserted floors cannot disagree by a half-ulp.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from opsconsole.redaction import Detection, Kind, NameDirectory, RedactionResult, redact

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "cases.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
REPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "redaction_report.py"

# The README block the drift test compares against (contract: "README drift").
TABLE_START = "<!-- redaction-table:start -->"
TABLE_END = "<!-- redaction-table:end -->"

# `<KIND_N>`: the kind's own name, an underscore, then a 1-based index.
PLACEHOLDER_PATTERN = re.compile(r"<([A-Z0-9]+(?:_[A-Z0-9]+)*)_([0-9]+)>")


@dataclass(frozen=True, slots=True)
class ExpectedSpan:
    """One ground-truth detection: the kind and the exact half-open offsets."""

    kind: Kind
    start: int
    end: int
    value: str


@dataclass(frozen=True, slots=True)
class Case:
    """One fixture row. `spans` is empty for the eleven negative cases.

    `directory` is the tenant display names this case is scored with, and it is
    empty for every case that is not about `Kind.PERSON`.
    The default is what keeps every existing fixture row unchanged: a case with
    no `directory` key is scored with no matcher at all, which is exactly how
    `detect_all` and `redact` behave for a caller that passes none.
    """

    id: str
    text: str
    spans: tuple[ExpectedSpan, ...]
    directory: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Score:
    """Pooled counts over every case, plus the two derived metrics.

    Both metrics are 0.0 when their denominator is zero. That never happens for
    this fixture - `test_fixture_has_forty_six_positive_and_eleven_negative_cases`
    holds it at 46 positive cases, and a positive case carries at least one
    expected span - and a silent `ZeroDivisionError` in a scorer is worse than a
    floor test that fails loudly at 0.0.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        expected = self.true_positives + self.false_negatives
        return self.true_positives / expected if expected else 0.0


def round_metric(value: float) -> float:
    """Round a metric to three decimals.

    The one rounding shared by the floor assertions and the README table; see
    this module's docstring.
    """
    return round(value, 3)


def load_cases(path: Path = FIXTURE_PATH) -> tuple[Case, ...]:
    """Read `cases.json`, checking every offset against the text it indexes.

    Raises `AssertionError` when `text[start:end] != value`, which is the only
    thing standing between a mistyped offset and a fixture that quietly scores
    itself against spans nobody wrote.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = document["cases"]
    cases: list[Case] = []
    for raw in raw_cases:
        case_id = str(raw["id"])
        text = str(raw["text"])
        spans: list[ExpectedSpan] = []
        for raw_span in raw["spans"]:
            span = ExpectedSpan(
                kind=Kind(raw_span["kind"]),
                start=int(raw_span["start"]),
                end=int(raw_span["end"]),
                value=str(raw_span["value"]),
            )
            assert text[span.start : span.end] == span.value, (
                f"{case_id}: span {span.kind.value} [{span.start}:{span.end}] indexes "
                f"{text[span.start : span.end]!r}, not {span.value!r}"
            )
            spans.append(span)
        cases.append(
            Case(
                id=case_id,
                text=text,
                spans=tuple(spans),
                directory=tuple(str(name) for name in raw.get("directory", ())),
            )
        )
    return tuple(cases)


def directory_for(case: Case) -> NameDirectory | None:
    """The compiled matcher one case is scored with, or `None` when it names no names.

    `None` and not an empty `NameDirectory`, so a case with no `directory` key is
    scored through exactly the code path every caller that passes no directory
    takes; an empty matcher would exercise a branch the gateway never reaches for
    a tenant with no names.
    """
    if not case.directory:
        return None
    return NameDirectory.compile(case.directory)


def score_cases(
    cases: Sequence[Case],
    detect: Callable[[str, NameDirectory | None], Sequence[Detection]],
) -> Score:
    """Pool true/false positives and false negatives over every case.

    A detection is a true positive only when `(kind, start, end)` equals an
    expected span exactly: a right span with the wrong kind label counts once as
    a false positive and once as a false negative, which is the whole reason
    decision B stores spans rather than an expected output string.

    Each case is scored with ITS OWN compiled directory.
    Scoring `Kind.PERSON` against a directory pooled over the whole fixture would
    charge every other case a false positive for a name it was never given, and
    would measure a matcher no request ever compiles.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for case in cases:
        expected = {(span.kind, span.start, span.end) for span in case.spans}
        found = {(d.kind, d.start, d.end) for d in detect(case.text, directory_for(case))}
        true_positives += len(expected & found)
        false_positives += len(found - expected)
        false_negatives += len(expected - found)
    return Score(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def placeholders_in_order(original: str, result: RedactionResult) -> list[str]:
    r"""Return the placeholder that replaced each detection, in span order.

    Walks `original` and `result.text` together: the text between two
    detections must be copied verbatim, and what sits in a detection's place
    must match `PLACEHOLDER_PATTERN`. Asserting the untouched text here is what
    makes a nested, truncated or interleaved substitution visible at all.
    """
    tokens: list[str] = []
    original_pos = 0
    output_pos = 0
    for detection in result.detections:
        gap = original[original_pos : detection.start]
        assert result.text[output_pos : output_pos + len(gap)] == gap, (
            f"text outside a detection was rewritten: expected {gap!r} at "
            f"offset {output_pos} of the redacted text"
        )
        output_pos += len(gap)
        match = PLACEHOLDER_PATTERN.match(result.text, output_pos)
        assert match is not None, (
            f"no placeholder at offset {output_pos} of the redacted text for "
            f"{detection.kind.value} [{detection.start}:{detection.end}]"
        )
        tokens.append(match.group(0))
        output_pos = match.end()
        original_pos = detection.end
    tail = original[original_pos:]
    assert result.text[output_pos:] == tail, (
        f"trailing text was rewritten: expected {tail!r}, got {result.text[output_pos:]!r}"
    )
    return tokens


def emitted_detections(text: str, directory: NameDirectory | None = None) -> tuple[Detection, ...]:
    """What `redact` actually replaces, which is what precision and recall measure.

    NOT `detect_all`. Raw candidates are an internal stage: inside a
    space-grouped IBAN they legitimately include a `CARD` and a `PHONE` sub-span
    that the overlap merge then drops, so scoring them would report false
    positives for values the engine never emits and never stores. The resolved
    list is what a caller sees, what the placeholder map holds and what an earlier change
    persists, so it is the honest subject. `detect_all` keeps its own per-kind
    rejection rows in `test_detectors.py`; see
    `test_the_score_measures_what_the_engine_emits_not_raw_candidates`.

    `directory` is the case's own compiled matcher, defaulted so the rows that
    call this helper directly with one argument keep reading as they did.
    """
    return redact(text, directory=directory).detections


def load_report_module() -> ModuleType:
    """Load `scripts/redaction_report.py` by file path.

    `scripts/` has no `__init__.py` and is excluded from the built wheel
    (`[tool.hatch.build.targets.wheel] only-include`), so `import scripts.x`
    does not resolve from an installed checkout the way `opsconsole.x` does.
    Loading by path is the smallest thing that works without adding a package.
    """
    if not REPORT_SCRIPT_PATH.exists():
        pytest.fail(f"{REPORT_SCRIPT_PATH} does not exist yet")
    spec = importlib_util.spec_from_file_location("scripts.redaction_report", REPORT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail(f"{REPORT_SCRIPT_PATH} is not importable as a module")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_render_table() -> Callable[[float, float], str]:
    """The one function the drift test compares the committed README block against."""
    render = getattr(load_report_module(), "render_table", None)
    if render is None:
        pytest.fail(f"{REPORT_SCRIPT_PATH} defines no render_table")
    return cast(Callable[[float, float], str], render)


def readme_table_block(readme: str) -> str:
    """Extract the generated table between the two README markers."""
    start = readme.find(TABLE_START)
    end = readme.find(TABLE_END)
    assert start != -1, f"README.md has no {TABLE_START} marker"
    assert end != -1, f"README.md has no {TABLE_END} marker"
    assert start < end, f"README.md has {TABLE_END} before {TABLE_START}"
    return readme[start + len(TABLE_START) : end].strip("\n")


@pytest.fixture(scope="session")
def cases() -> tuple[Case, ...]:
    """The 57 fixture cases, read and offset-checked once per session."""
    return load_cases()
