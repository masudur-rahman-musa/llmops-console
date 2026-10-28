r"""An earlier change: the last folder-restructure move. This file checks the END STATE of the tree.

Text and filesystem reads over the repository itself plus the git index; no build, no
service, no network, and no import of any `app`, `infra` or `tests` module - the whole point
is to inspect files on disk so the file stays runnable standalone from its new home. While the
restructure is open every case here is red; once the move lands they go green and stay the
guard against a reference left pointing at a folder that no longer exists.

This module spells the removed-root tokens out to ban them, so it is deliberately the opposite
of a document that must avoid such strings.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

#: This file's own location, never the working directory - `pytest` runs from anywhere.
#: The file sits at `infra/tests/infra/<file>`, so parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

ROOT_ALLOWLIST = REPO_ROOT / ".github" / "root-allowlist"
CHECK_ROOT = REPO_ROOT / ".github" / "scripts" / "check_root.py"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: The temporary-section marker the last plan deletes from the allowlist.
TEMPORARY_MARKER = "TEMPORARY"

#: Every token the TEMPORARY section let sit at the root; none may survive as its own entry.
TEMPORARY_ENTRIES = [
    "services/",
    "packages/",
    "tests/",
    "project-evals/",
    "project-obs/",
    "scripts/",
    "models/",
    "Dockerfile",
    "pyproject.toml",
    "uv.lock",
    "routing.yaml",
    "prices.yaml",
    "CUSTOMIZING.md",
]

#: Tracked files whose job is to RECORD the move, so they are allowed to name an old root.
GREP_EXCLUDE_PREFIXES = ("dev/plans/",)
GREP_EXCLUDE_FILES = (
    "dev/docs/restructure.md",
    "dev/docs/log.md",
    # This module spells every banned token out on purpose, so it cannot scan itself.
    "infra/tests/infra/test_restructure_final.py",
)

#: Each removed root and how a surviving reference to it is matched in a tracked file. A plain
#: string is a literal substring; a compiled pattern is matched with `.search`. Case-sensitive
#: throughout, matching `git grep`'s default: `CUSTOMIZING.md` catches only the old root, never
#: the new lowercase `docs/customizing.md`.
REMOVED_ROOT_TOKENS: list[tuple[str, str | re.Pattern[str]]] = [
    ("project-obs/", "project-obs/"),
    ("project-evals/ci", "project-evals/ci"),
    ("project-evals/load", "project-evals/load"),
    ("project-evals/tradeoff", "project-evals/tradeoff"),
    # The repo-root `tests/` split: anchored so the new home `infra/tests/infra` (preceded by
    # `/`) is NOT a hit, only a bare root `tests/infra|ci|load` is.
    ("tests/infra", re.compile(r"(^|[^\w/])tests/infra")),
    ("tests/ci", re.compile(r"(^|[^\w/])tests/ci")),
    ("tests/load", re.compile(r"(^|[^\w/])tests/load")),
    ("docs/architecture.md", "docs/architecture.md"),
    ("docs/runbook.md", "docs/runbook.md"),
    ("docs/threat-model.md", "docs/threat-model.md"),
    ("CUSTOMIZING.md", "CUSTOMIZING.md"),
    # The eight moved scripts at their old root home. Matched by name, not as a bare `scripts/`,
    # so a generic tree diagram's `scripts/` and the companion repo's own `scripts/seed.py` are
    # not false hits - only a dangling pointer to one of the files THIS plan moved.
    (
        "moved scripts/<file>",
        re.compile(
            r"(^|[^\w/])scripts/(aws_evidence\.sh|server_helpdesk_data\.sh|server_secrets\.sh"
            r"|tradeoff_run\.sh|git-pre-push-guard\.sh|redaction_report\.py|select_tests\.py"
            r"|tradeoff_policy\.py)"
        ),
    ),
    ("./models", "./models"),
]

#: Directories that moved; each must exist at its new home after the plan.
MOVED_DIRS = [
    "infra/tests/",
    "infra/load/",
    "infra/tradeoff/",
    "infra/scripts/",
    "infra/k8s/kind/",
    "dev/ops/",
    "dev/adr/",
]

#: Files that moved; each must exist at its new home after the plan.
MOVED_FILES = [
    "infra/k8s/monitoring/alerts/rules.yml",
    "ARCHITECTURE.md",
    "docs/customizing.md",
]

#: Old roots that must be gone entirely. `project-evals/` is NOT here - part of it stays.
REMOVED_DIRS = [
    "project-obs/",
    "scripts/",
    "tests/",
    "services/",
    "models/",
]

ALERT_RULES = REPO_ROOT / "infra" / "k8s" / "monitoring" / "alerts" / "rules.yml"

#: Directories a filesystem walk never descends into.
WALK_SKIP = {".git", ".venv"}


def allowlist_entries() -> list[str]:
    """Each stripped, non-comment, non-empty line of the root allowlist."""
    lines: list[str] = []
    for raw in ROOT_ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def tracked_files() -> list[str]:
    """Every path git tracks, as repo-relative strings."""
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def is_excluded(path: str) -> bool:
    """A tracked path whose only job is to record the move, so it may name an old root."""
    return path.startswith(GREP_EXCLUDE_PREFIXES) or path in GREP_EXCLUDE_FILES


def offending_lines(token: str | re.Pattern[str]) -> list[str]:
    """Every `path:line` in a tracked, non-excluded text file that names `token`."""
    hits: list[str] = []
    for path in tracked_files():
        if is_excluded(path):
            continue
        full = REPO_ROOT / path
        try:
            text = full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if isinstance(token, re.Pattern):
                matched = token.search(line) is not None
            else:
                matched = token in line
            if matched:
                hits.append(f"{path}:{number}: {line.strip()}")
    return hits


def walk_matches(name: str, under: str | None = None) -> list[str]:
    """Every file named `name` under REPO_ROOT (optionally inside a dir called `under`)."""
    found: list[str] = []
    for candidate in REPO_ROOT.rglob(name):
        parts = candidate.relative_to(REPO_ROOT).parts
        if any(skip in parts for skip in WALK_SKIP):
            continue
        if under is not None and under not in parts:
            continue
        found.append(str(candidate.relative_to(REPO_ROOT)))
    return found


def test_the_root_allowlist_has_no_temporary_section() -> None:
    """Catches a last restructure plan that moves the files but leaves the allowlist's
    TEMPORARY block in place, which would keep every old root (services/, packages/, the bare
    config files) legal at the repository root even after nothing uses it."""
    text = ROOT_ALLOWLIST.read_text(encoding="utf-8")
    assert TEMPORARY_MARKER not in text, f"{ROOT_ALLOWLIST.name} still carries a TEMPORARY block"

    entries = set(allowlist_entries())
    leftover = [entry for entry in TEMPORARY_ENTRIES if entry in entries]
    assert leftover == [], f"allowlist still permits removed roots: {leftover}"


def test_check_root_passes() -> None:
    """Catches a root left with a folder or file the trimmed allowlist no longer permits: the
    repository's own gate is the judge, so a stray old root that nothing else notices still
    reddens here."""
    completed = subprocess.run(
        ["python3", str(CHECK_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`python3 .github/scripts/check_root.py` exited {completed.returncode}:\n"
        f"{completed.stdout}{completed.stderr}"
    )


def test_no_committed_file_names_a_removed_root() -> None:
    """Catches any tracked file still pointing at a folder the move deleted - a compose host
    path at ./models, a doc link to docs/runbook.md, a config under project-obs/ - which would
    leave a dangling reference nothing resolves. The plan-and-log files that only record the
    move are excluded, so a hit here is a real pointer that was never repointed."""
    found: list[str] = []
    for label, token in REMOVED_ROOT_TOKENS:
        for hit in offending_lines(token):
            found.append(f"[{label}] {hit}")
    assert found == [], "references to removed roots survive:\n" + "\n".join(found)


def test_every_moved_root_exists() -> None:
    """Catches a move that lands a folder or file at the wrong new home, or deletes an old root
    without recreating it - infra/tests/ never made, ARCHITECTURE.md left only at the root, the
    alert rules copied nowhere - and catches an old root that was never removed at all."""
    missing_new: list[str] = []
    for relative in MOVED_DIRS:
        if not (REPO_ROOT / relative).is_dir():
            missing_new.append(f"dir {relative}")
    for relative in MOVED_FILES:
        if not (REPO_ROOT / relative).is_file():
            missing_new.append(f"file {relative}")
    assert missing_new == [], f"moved roots missing at their new home: {missing_new}"

    still_present = [relative for relative in REMOVED_DIRS if (REPO_ROOT / relative).exists()]
    assert still_present == [], f"old roots not removed: {still_present}"


def test_precommit_and_ci_run_check_root() -> None:
    """Catches the root check being defined but never wired in: a pre-commit config with no
    local check-root hook, or a CI workflow that never runs .github/scripts/check_root.py, would
    let a bad root land because no gate actually runs the check."""
    precommit = PRECOMMIT.read_text(encoding="utf-8")
    assert "- repo: local" in precommit, "pre-commit config has no `- repo: local` section"
    assert "check-root" in precommit, "pre-commit config defines no `check-root` hook"

    ci = CI_WORKFLOW.read_text(encoding="utf-8")
    assert ".github/scripts/check_root.py" in ci, (
        "CI workflow does not run .github/scripts/check_root.py"
    )


def test_the_one_alert_rules_copy() -> None:
    """Catches the alert rules being copied to their new home while the old project-obs/ copy is
    left behind, which would leave two files that could drift apart - the move must leave exactly
    the new one and no copy under any project-obs/ directory."""
    assert ALERT_RULES.is_file(), f"{ALERT_RULES.relative_to(REPO_ROOT)} does not exist"

    old_copies = walk_matches("rules.yml", under="project-obs")
    assert old_copies == [], f"an old alert-rules copy survives under project-obs/: {old_copies}"
