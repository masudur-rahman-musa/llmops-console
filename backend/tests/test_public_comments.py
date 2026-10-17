"""Guard tests: no comment or docstring in backend/app/ or web/src/ names an
inside reference (plan/spec/ADR/decision number, test function, dev/ path, or
old gateway module path).

File reads from disk only — no running service, no db marker.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).parent.parent
APP_ROOT = BACKEND_ROOT / "app"
WEB_SRC = BACKEND_ROOT.parent / "web" / "src"

# ---------------------------------------------------------------------------
# patterns that must not appear in comments / docstrings
# ---------------------------------------------------------------------------

_INSIDE_REF_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"plan \d", re.IGNORECASE),
    re.compile(r"row \d", re.IGNORECASE),
    re.compile(r"`\d\d?[a-z]`"),  # e.g. `16c`, `17b`
    re.compile(r"decision [A-F]"),
    re.compile(r"SPEC §"),
    re.compile(r"REQUIREMENTS"),
    re.compile(r"ADR \d"),
    re.compile(r"test_[a-z0-9_]+"),
    re.compile(r"dev/(adr|ops|docs|notes|plans|disputes)"),
    re.compile(r"gateway/"),
    re.compile(r"tests/gateway"),
    re.compile(r"gateway\.(app|errors|context|routes)"),
    re.compile(r"app\.gateway\."),
    re.compile(r"opsconsole\."),
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_in_text(text: str) -> list[str]:
    """Return a list of pattern descriptions that match anywhere in text."""
    hits: list[str] = []
    for pat in _INSIDE_REF_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def _py_comment_strings(source: str) -> list[str]:
    """Extract every docstring and inline comment text from Python source."""
    texts: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return texts

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            texts.append(node.value.value)

    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            texts.append(stripped[1:])

    return texts


def _ts_comment_strings(source: str) -> list[str]:
    """Extract every // and /* */ comment text from TypeScript/TSX source."""
    texts: list[str] = []

    # /* ... */ (multi-line)
    for m in re.finditer(r"/\*(.*?)\*/", source, re.DOTALL):
        texts.append(m.group(1))

    # // ... (single-line)
    for m in re.finditer(r"//(.*)$", source, re.MULTILINE):
        texts.append(m.group(1))

    return texts


# ---------------------------------------------------------------------------
# test 1 — the scanner itself is proven on a planted bad string
# ---------------------------------------------------------------------------


def test_the_bad_pattern_scanner_flags_a_planted_inside_reference() -> None:
    bad_lines = [
        "plan 16a was the one that added this",
        "pinned by `test_x`",
        "see dev/adr/ for the decision",
        "old path was gateway/app.py",
    ]
    for line in bad_lines:
        hits = _find_in_text(line)
        assert hits, f"Scanner missed a planted bad line: {line!r}"


# ---------------------------------------------------------------------------
# test 2 — no inside ref of the plan/spec/ADR kind in backend/app/
# ---------------------------------------------------------------------------

_PLAN_SPEC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"plan \d", re.IGNORECASE),
    re.compile(r"row \d", re.IGNORECASE),
    re.compile(r"`\d\d?[a-z]`"),
    re.compile(r"decision [A-F]"),
    re.compile(r"SPEC §"),
    re.compile(r"REQUIREMENTS"),
    re.compile(r"ADR \d"),
]


def test_no_backend_app_comment_names_a_plan_spec_adr_decision_or_row() -> None:
    violations: list[str] = []
    for py_file in sorted(APP_ROOT.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        for text in _py_comment_strings(source):
            for pat in _PLAN_SPEC_PATTERNS:
                if pat.search(text):
                    rel = py_file.relative_to(BACKEND_ROOT.parent)
                    violations.append(f"{rel}: pattern {pat.pattern!r} in: {text[:120]!r}")
                    break

    assert violations == [], (
        f"{len(violations)} comment(s) name a plan/spec/ADR/decision:\n"
        + "\n".join(violations[:40])
    )


# ---------------------------------------------------------------------------
# test 3 — no test name / dev path / old module path in backend/app/
# ---------------------------------------------------------------------------

_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"test_[a-z0-9_]+"),
    re.compile(r"dev/(adr|ops|docs|notes|plans|disputes)"),
    re.compile(r"gateway/"),
    re.compile(r"tests/gateway"),
    re.compile(r"gateway\.(app|errors|context|routes)"),
    re.compile(r"app\.gateway\."),
    re.compile(r"opsconsole\."),
]


def test_no_backend_app_comment_names_a_test_a_dev_path_or_an_old_module_path() -> None:
    violations: list[str] = []
    for py_file in sorted(APP_ROOT.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        for text in _py_comment_strings(source):
            for pat in _PATH_PATTERNS:
                if pat.search(text):
                    rel = py_file.relative_to(BACKEND_ROOT.parent)
                    violations.append(f"{rel}: pattern {pat.pattern!r} in: {text[:120]!r}")
                    break

    assert violations == [], (
        f"{len(violations)} comment(s) name a test/dev path/old module path:\n"
        + "\n".join(violations[:40])
    )


# ---------------------------------------------------------------------------
# test 4 — no inside ref in web/src/ comments
# ---------------------------------------------------------------------------


def test_no_web_src_comment_names_an_inside_reference() -> None:
    violations: list[str] = []
    for ts_file in sorted(WEB_SRC.rglob("*")):
        if ts_file.suffix not in {".ts", ".tsx"}:
            continue
        if "__pycache__" in ts_file.parts or "node_modules" in ts_file.parts:
            continue
        source = ts_file.read_text(encoding="utf-8")
        for text in _ts_comment_strings(source):
            hits = _find_in_text(text)
            if hits:
                rel = ts_file.relative_to(BACKEND_ROOT.parent)
                violations.append(f"{rel}: patterns {hits} in: {text[:120]!r}")

    assert violations == [], (
        f"{len(violations)} web comment(s) name an inside reference:\n" + "\n".join(violations[:40])
    )
