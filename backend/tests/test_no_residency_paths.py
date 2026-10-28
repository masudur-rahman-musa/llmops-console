"""No shipped file names residency, and every path the changed docs name exists.

Both cases read files from the repo and nothing else: no app, no service. They are
the whole-tree sweep that the per-feature cases cannot be, because a leftover word
in a Helm-style manifest or a dashboard is nobody's unit test.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

REMOVED = re.compile(r"residency|eu_regions|allow_regions|X-Gateway-Region|\b451\b", re.IGNORECASE)

SCANNED_TREES = ("backend/app", "backend/config", "infra")
SCANNED_FILES = ("Makefile",)
MIGRATION_GLOB = "backend/migrations/005*"
# Migration 005 has to name the table it drops. That name is the one allowed word.
DROPPED_TABLE = "residency_log"

SKIPPED_PARTS = {".terraform", "__pycache__", "node_modules", ".venv", "tests"}
SKIPPED_NAMES = {".terraform.lock.hcl", "terraform.tfstate"}
TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".json", ".sql", ".md", ".sh", ".tf", ".toml", ".js",
    ".txt", ".hcl", ".ini", ".cfg", ".env", ".conf", ".tpl", ".tfvars", "",
}  # fmt: skip


def _is_text_file(path: Path) -> bool:
    if path.name in SKIPPED_NAMES or SKIPPED_PARTS & set(path.relative_to(REPO).parts):
        return False
    return path.is_file() and path.suffix in TEXT_SUFFIXES


def _shipped_files() -> Iterator[Path]:
    for tree in SCANNED_TREES:
        yield from sorted(path for path in (REPO / tree).rglob("*") if _is_text_file(path))
    for name in SCANNED_FILES:
        yield REPO / name
    yield from sorted(REPO.glob(MIGRATION_GLOB))


def test_no_shipped_file_names_residency() -> None:
    """Catches a leftover in any file that ships: a dashboard panel, a manifest comment, a
    metric name, the Makefile. The per-feature cases only look where the removed
    code was; this sweep finds the file nobody thought to list. Terraform `region` is
    untouched because the pattern never matches the bare word, and `infra/tests` (which
    state the absence) is outside the sweep. Files are counted so an empty walk fails."""
    files = list(_shipped_files())
    assert len(files) > 50
    hits: list[str] = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name.startswith("005"):
            text = text.replace(DROPPED_TABLE, "")
        for number, line in enumerate(text.splitlines(), start=1):
            if REMOVED.search(line):
                hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()[:100]}")

    assert hits == []


DOCS = ("README.md", "docs/customizing.md")
ROOTS = ("backend/", "infra/", "docs/", "web/", ".github/")
ROOT_FILES = {"Makefile", "README.md", "ARCHITECTURE.md", "docker-compose.yml"}
# Written by a run, or living in the companion repo: named in a doc, never in this tree.
NOT_IN_THIS_TREE = {"docs/tradeoff.md", "backend/evals/conversations/cases.jsonl"}

BACKTICKED = re.compile(r"`([^`\s]+)`")
LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")


def _candidate(token: str) -> str | None:
    """The repo path a token names, or None when it is not a plain repo path."""
    token = token.split("#", 1)[0].rstrip(".,;:)")
    token = re.sub(r":\d+(-\d+)?$", "", token)
    if not token or any(mark in token for mark in "*{}<>$|=") or "://" in token:
        return None
    if "/.env" in token or token in NOT_IN_THIS_TREE:
        return None
    if token.startswith(ROOTS) or token in ROOT_FILES:
        return token
    return None


def _named_paths(doc: str) -> set[str]:
    text = (REPO / doc).read_text(encoding="utf-8")
    found: set[str] = set()
    for token in BACKTICKED.findall(text):
        path = _candidate(token)
        if path is not None:
            found.add(path)
    for target in LINK_TARGET.findall(text):
        if target.startswith(("http", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        # The docs write links from the repo root as often as from their own folder.
        path = _candidate(target)
        if path is None:
            resolved = Path(os.path.normpath((REPO / doc).parent / target))
            if resolved.is_relative_to(REPO):
                path = _candidate(str(resolved.relative_to(REPO)))
        if path is not None:
            found.add(path)
    return found


def test_every_path_the_changed_docs_name_exists() -> None:
    """Catches a doc that still points at a file that was deleted (`test_residency_property.py`,
    `ops-purge-residency`, the residency dashboard) or at a test the README guarantee should
    link (`backend/tests/api/test_fallback.py`). Moving a file updates every reference in the
    same diff. Each doc must name at least one path, so a parser that
    finds nothing cannot pass."""
    missing: list[str] = []

    for doc in DOCS:
        paths = _named_paths(doc)
        assert paths, f"{doc} names no repo path"
        missing += [f"{doc}: {path}" for path in sorted(paths) if not (REPO / path).exists()]

    assert missing == []
