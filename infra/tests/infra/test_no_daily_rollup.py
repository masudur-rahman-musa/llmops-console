"""The daily rollup is gone from the repo.

Daily cost is summed from the ledger on each request, so the materialised view
`cost_ledger_daily`, its refresh command, its make target and its CronJob are removed. These
cases are the check that no file still names them and that no build still ships the job.

The build comes from the session fixture in `conftest.py` (`built` is the root build,
which holds the `platform` directory), so no build runs twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

REMOVED_NAMES = (
    "ops-refresh-rollup",
    "refresh-rollup",
    "rollup-refresh",
    "cost_ledger_daily",
    "ledger-maintenance.yaml",
)
"""Every name the removal takes out, as literals."""

CHECKED_FILES = (
    REPO_ROOT / "Makefile",
    REPO_ROOT / "backend" / "pyproject.toml",
    REPO_ROOT / "README.md",
)
CHECKED_DIRECTORIES = (
    REPO_ROOT / "infra" / "k8s",
    REPO_ROOT / ".github",
)

REMOVED_CRONJOB = "rollup-refresh"


def files_to_scan() -> list[Path]:
    """Every file the removal must leave clean: the named files and each directory's files."""
    found = [path for path in CHECKED_FILES if path.is_file()]
    for directory in CHECKED_DIRECTORIES:
        found.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
    return found


def test_nothing_names_the_removed_rollup() -> None:
    """Catches a leftover reference to the rollup: a `make` target, a script entry, a
    kustomization line, a workflow step or a doc that still names the removed view, command or
    CronJob (any one left behind is a broken path or a dead job). The required files are
    asserted to exist first, so an empty scan cannot pass."""
    for required in (REPO_ROOT / "Makefile", REPO_ROOT / "backend" / "pyproject.toml"):
        assert required.is_file(), required
    for path in CHECKED_FILES:
        assert path.is_file(), path
    scanned = files_to_scan()
    assert scanned != []

    hits = [
        (str(path.relative_to(REPO_ROOT)), name)
        for path in scanned
        for name in REMOVED_NAMES
        if name in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert hits == []
    assert not (REPO_ROOT / "infra" / "k8s" / "platform" / "ledger-maintenance.yaml").exists()


def cronjob_names(documents: list[dict[str, Any]]) -> list[str]:
    """The name of every CronJob in one built stream."""
    return [
        str(document.get("metadata", {}).get("name"))
        for document in documents
        if document.get("kind") == "CronJob"
    ]


def test_the_platform_build_has_no_rollup_job(
    built: list[dict[str, Any]],
) -> None:
    """Catches a `platform` build that still ships the `rollup-refresh` CronJob (the
    file deleted from disk but still listed). The stream is checked to hold the other
    platform CronJobs, so an empty or failed parse cannot pass: the Postgres backup job
    must still be there."""
    names = cronjob_names(built)
    assert "postgres-backup" in names
    assert REMOVED_CRONJOB not in names
