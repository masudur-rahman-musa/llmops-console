"""The public docs and the committed eval reports say nothing about residency.

File reads only: no app, no service.

* No public doc names residency, the 451, the old region settings or EU.
* No committed `report.json` carries a `region` key, and each still loads.
* Every repo path the changed docs name exists.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

# Residency words match in any case; the bare EU only as capitals, so a tenant named
# `acme-eu` in an example is not a hit.
REMOVED = re.compile(r"(?i:residency|eu_regions|allow_regions|X-Gateway-Region)|\b451\b|\bEU\b")

CHANGED_DOCS = (
    "README.md",
    "web/README.md",
    "docs/customizing.md",
    "docs/azure.md",
    "docs/aws-evidence/README.md",
    "docs/media/README.md",
    "docs/eval-report.md",
)
ROOTS = ("backend/", "infra/", "docs/", "web/", ".github/")
ROOT_FILES = {"Makefile", "README.md", "ARCHITECTURE.md", "docker-compose.yml"}
# Written by a run, or living in the companion repo.
NOT_IN_THIS_TREE = {"docs/tradeoff.md", "backend/evals/conversations/cases.jsonl"}

BACKTICKED = re.compile(r"`([^`\s]+)`")
LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")


def public_docs() -> list[Path]:
    return [REPO / "README.md", REPO / "web" / "README.md", *sorted((REPO / "docs").rglob("*.md"))]


def has_key(node: Any, key: str) -> bool:  # noqa: ANN401
    """Whether a JSON value carries `key` as an object key at any depth."""
    if isinstance(node, dict):
        return key in node or any(has_key(value, key) for value in node.values())
    if isinstance(node, list):
        return any(has_key(item, key) for item in node)
    return False


def test_no_public_doc_names_residency() -> None:
    """Catches a leftover residency, 451, region-setting or EU line in the README, the web
    README or any page under docs/, which would describe a feature the gateway no longer has."""
    docs = public_docs()
    assert len(docs) >= 8
    hits = [
        f"{path.relative_to(REPO)}:{number}: {line.strip()[:100]}"
        for path in docs
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if REMOVED.search(line)
    ]
    assert hits == []


def test_no_committed_report_carries_region() -> None:
    """Catches a committed report.json that still has a `region` key on a case or in its
    metrics, or one whose edit left it unreadable as JSON. An old report does not follow the
    current report format, so it is read as plain JSON only.

    Only the files git tracks: `backend/evals/reports/` is gitignored, so what sits there
    is one machine's old runs, absent from CI and a fresh clone."""
    listed = subprocess.run(
        ["git", "ls-files", "--", "backend/evals/reports"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    reports = sorted(REPO / name for name in listed if name.endswith("report.json"))
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert not has_key(data, "region"), f"{path.relative_to(REPO)} still has a region key"


def candidate(token: str) -> str | None:
    """The repo path a token names, or None when it is not a plain repo path."""
    token = token.split("#", 1)[0].rstrip(".,;:)")
    token = re.sub(r":\d+(-\d+)?$", "", token)
    if not token or any(mark in token for mark in "*{}<>$|=") or "://" in token:
        return None
    if "/.env" in token or token.endswith(".tfvars"):
        return None
    if token in NOT_IN_THIS_TREE:
        return None
    if token.startswith(ROOTS) or token in ROOT_FILES:
        return token
    return None


def named_paths(doc: str) -> set[str]:
    text = (REPO / doc).read_text(encoding="utf-8")
    found = {path for token in BACKTICKED.findall(text) if (path := candidate(token))}
    for target in LINK_TARGET.findall(text):
        if target.startswith(("http", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        path = candidate(target)
        if path is None:
            resolved = Path(os.path.normpath((REPO / doc).parent / target))
            if resolved.is_relative_to(REPO):
                path = candidate(str(resolved.relative_to(REPO)))
        if path is not None:
            found.add(path)
    return found


def test_every_path_the_changed_docs_name_exists() -> None:
    """Catches a changed doc that still points at a file this change deleted (the residency
    page, its guards) or at a path that never existed. Each doc must name a path, so a parser
    that finds nothing cannot pass."""
    missing: list[str] = []
    for doc in CHANGED_DOCS:
        paths = named_paths(doc)
        missing += [f"{doc}: {path}" for path in sorted(paths) if not (REPO / path).exists()]
    assert missing == []
    assert sum(len(named_paths(doc)) for doc in CHANGED_DOCS) >= 7
