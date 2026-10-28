"""Three public scripts sit at their public homes, and nothing names their old paths.

Every case reads committed files or runs a script for a few seconds; none needs
a service, so none carries the `db` marker. Old-path needles are built from
parts so this file is not a hit for its own search.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

HOMES = {
    "check_root.py": Path(".github") / "scripts" / "check_root.py",
    "tradeoff_policy.py": Path("infra") / "scripts" / "tradeoff_policy.py",
    "redaction_report.py": Path("backend") / "scripts" / "redaction_report.py",
}
OLD_DIR = "/".join(("dev", "scripts"))


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def test_each_script_sits_at_its_public_home() -> None:
    """Catches a move that leaves a script in its old folder (or copies instead of
    moving), so the public repo would lose a tool CI and the README rely on.
    """
    for name, home in HOMES.items():
        assert (REPO_ROOT / home).is_file(), home
        assert not (REPO_ROOT / OLD_DIR / name).exists(), (OLD_DIR, name)


@pytest.mark.parametrize("name", sorted(HOMES))
def test_no_live_file_names_an_old_script_path(name: str) -> None:
    """Catches a moved script whose old path is still named by a live file
    (workflow, hook, README, test), which would break the build.
    """
    slash_form = f"{OLD_DIR}/{name}"
    path_form = " / ".join(('"dev"', '"scripts"', f'"{name}"'))
    for needle in (slash_form, path_form):
        result = _git("grep", "-n", "-F", "-e", needle, "--", ".")
        assert result.returncode == 1, (needle, result.stdout)


def test_every_caller_names_the_new_path() -> None:
    """Catches a caller that was never repointed, or repointed to a wrong home,
    so CI, the hook, the overlay script or the README would run a missing file.
    """
    root_check = HOMES["check_root.py"].as_posix()
    expectations = {
        Path(".github") / "workflows" / "ci.yml": (root_check, 1),
        Path(".pre-commit-config.yaml"): (root_check, 1),
        Path("infra") / "scripts" / "tradeoff_run.sh": (HOMES["tradeoff_policy.py"].as_posix(), 1),
        Path("README.md"): (HOMES["redaction_report.py"].as_posix(), 1),
    }
    for file, (needle, count) in expectations.items():
        text = (REPO_ROOT / file).read_text(encoding="utf-8")
        assert text.count(needle) >= count, (file, needle, text.count(needle))


def test_root_check_passes_from_its_new_home() -> None:
    """Catches a moved root check that cannot find the repo root from its new
    folder, or that rejects the tree it guards.
    """
    result = subprocess.run(
        ["python3", str(HOMES["check_root.py"])],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_redaction_report_check_runs_as_a_fresh_process() -> None:
    """Catches a moved report that still points `TESTS_ROOT` at the gone root
    `tests/`, so the README's own command fails in a fresh process.
    """
    result = subprocess.run(
        [sys.executable, str(HOMES["redaction_report.py"]), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout[-500:], result.stderr[-500:])
