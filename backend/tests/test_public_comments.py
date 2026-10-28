"""Guard tests: no comment or docstring in backend/app/ or web/src/ points at
private working material (plan/spec/ADR/decision number, test function, a private
folder, or an old gateway module path).

The names this guard looks for are built from fragments, so the file does not
contain them itself.

File reads from disk only — no running service, no db marker.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).parent.parent
APP_ROOT = BACKEND_ROOT / "app"
WEB_SRC = BACKEND_ROOT.parent / "web" / "src"


def _j(*parts: str) -> str:
    """Join fragments, so a banned name never appears whole in this file."""
    return "".join(parts)


_PRIVATE_DIRS = _j("(adr|ops|docs|", "notes|plans|disputes)")
_DEV_PATH = _j("dev", "/") + _PRIVATE_DIRS

# ---------------------------------------------------------------------------
# patterns that must not appear in comments / docstrings
# ---------------------------------------------------------------------------

_INSIDE_REF_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"plan \d", re.IGNORECASE),
    re.compile(r"row \d", re.IGNORECASE),
    re.compile(r"`\d\d?[a-z]`"),  # e.g. `16c`, `17b`
    re.compile(r"decision [A-F]"),
    re.compile(r"SPEC §"),
    re.compile(_j("REQUIRE", "MENTS")),
    re.compile(r"ADR \d"),
    re.compile(r"test_[a-z0-9_]+"),
    re.compile(_DEV_PATH),
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
        f"see {_j('dev', '/')}adr/ for the decision",
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
    re.compile(_j("REQUIRE", "MENTS")),
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
    re.compile(_DEV_PATH),
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


# ---------------------------------------------------------------------------
# test 5 — no tracked text file carries a planning pointer or a placeholder
# ---------------------------------------------------------------------------

REPO_ROOT = BACKEND_ROOT.parent

_PLANNING_POINTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bplans? \d", re.IGNORECASE),
    re.compile(r"\bADR[ -]?\d"),
    re.compile("§"),
    re.compile(r"\bSPEC\s+\d"),
    re.compile(r"\b(?:standing|hard) rule \d", re.IGNORECASE),
    re.compile(_j("STA", "TUS\\.md")),
    re.compile(_j("what", "-broke")),
    re.compile(r"(?<![A-Za-z0-9_-])" + _j("his", r"\.md")),
    re.compile(r"(?<![A-Za-z0-9_-])" + _j("notes", r"\.md")),
    re.compile(_j("switches", r"\.md")),
    re.compile(_j("dev", "-process")),
    re.compile(_j("restructure", " plan"), re.IGNORECASE),
    re.compile(_j("test", "-mode")),
    re.compile(_j("auto", "mode")),
    re.compile(_DEV_PATH.replace("(", "(?:") + r"\b"),
    re.compile(r"\ban earlier change\b", re.IGNORECASE),
    re.compile(r"\ba design note\b", re.IGNORECASE),
    re.compile(r"\bthe project notes\b", re.IGNORECASE),
    re.compile(r"\bproject(?:'s)? rules?\b", re.IGNORECASE),
]

# Files that name these strings on purpose: this guard (its patterns are the
# strings), lockfiles (generated) and the secret scanner's baseline and ignore
# files (they hold fingerprints, not prose).
_PLANNING_SCAN_SKIP_NAMES = {
    "uv.lock",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
    ".gitleaksignore",
}


def _is_planning_scan_skipped(rel: Path) -> bool:
    if rel.name in _PLANNING_SCAN_SKIP_NAMES or rel.name.endswith(".lock"):
        return True
    if "gitleaks" in rel.name and "baseline" in rel.name:
        return True
    return rel == Path(__file__).resolve().relative_to(REPO_ROOT)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [Path(p.decode()) for p in out.split(b"\0") if p]


def _planning_hits(text: str) -> list[str]:
    return [pat.pattern for pat in _PLANNING_POINTER_PATTERNS if pat.search(text)]


def test_the_planning_pointer_scanner_flags_planted_lines() -> None:
    planted = [
        "see plan 12 for why",
        "recorded in ADR 0009",
        "per SPEC 4.2",
        "standing rule 7",
        f"written up in {_j('STA', 'TUS')}.md",
        f"a {_j('dev', '/')}plans/ file",
        "per the project's rule",
        "pinned by an earlier change",
        "as a design note says",
    ]
    for line in planted:
        assert _planning_hits(line), f"Scanner missed a planted line: {line!r}"
    assert _planning_hits("a pricing plan for each tenant") == []


def test_no_tracked_text_file_carries_a_planning_pointer_or_placeholder() -> None:
    try:
        files = _tracked_files()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    violations: list[str] = []
    for rel in files:
        if _is_planning_scan_skipped(rel):
            continue
        path = REPO_ROOT / rel
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            hits = _planning_hits(line)
            if hits:
                violations.append(f"{rel}:{lineno}: {hits}: {line.strip()[:100]!r}")

    assert violations == [], (
        f"{len(violations)} line(s) carry a planning pointer or placeholder:\n"
        + "\n".join(violations[:60])
    )
