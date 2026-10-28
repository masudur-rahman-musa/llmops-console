r"""The root check: the repository root holds only what the allowlist names.

Text and filesystem reads over the repository itself; no build, no service, no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: This file's own location, never the working directory - `pytest` runs from anywhere.
#: The file sits at `infra/tests/infra/<file>`, so parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

CHECK_ROOT = REPO_ROOT / ".github" / "scripts" / "check_root.py"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


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
