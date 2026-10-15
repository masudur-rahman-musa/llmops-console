"""The fast-test tooling this plan adds is pinned by reading the files it touches.

Every case reads a committed file (Makefile, pyproject, ci.yml, rules.md); none
runs a service, so none carries the `db` marker.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RULES = REPO_ROOT / "dev" / "docs" / "rules.md"


def _makefile_recipe(name: str) -> str:
    """The recipe text of Make target `name`.

    Anchored on a start-of-line `name:` so a target name that only shows up
    inside another recipe is not matched. Collects the tab-indented lines that
    follow until the next unindented line.
    """
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    target = re.compile(rf"^{re.escape(name)}:")
    recipe: list[str] = []
    collecting = False
    for line in lines:
        if target.match(line):
            collecting = True
            continue
        if collecting:
            if line.startswith(("\t", " ")) or line.strip() == "":
                recipe.append(line)
            else:
                break
    return "\n".join(recipe)


def test_slow_marker_is_registered() -> None:
    """Catches removing the `slow` marker from pyproject, which would make
    `pytest -m "not slow"` warn on an unknown marker.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("slow") for marker in markers), markers


def test_fast_target_runs_non_slow_in_parallel() -> None:
    """Catches a `fast` target that drops xdist (`-n`) or stops excluding slow
    tests (`-m "not slow"`), so the fast run would no longer be fast.
    """
    recipe = _makefile_recipe("fast")
    assert "-n" in recipe, recipe
    assert '-m "not slow"' in recipe, recipe


def test_slowest_target_prints_durations() -> None:
    """Catches a `test-slowest` target that does not pass `--durations`, so it
    would not surface the slow tests it exists to find.
    """
    recipe = _makefile_recipe("test-slowest")
    assert "--durations" in recipe, recipe


def test_full_test_target_sets_the_three_urls() -> None:
    """Catches a `test` target missing any of the three backing-service URL env
    names, so the db tests would run without their store wired up.
    """
    recipe = _makefile_recipe("test")
    for name in (
        "OPSCONSOLE_TEST_DATABASE_URL",
        "OPSCONSOLE_TEST_REDIS_URL",
        "OPSCONSOLE_TEST_QDRANT_URL",
    ):
        assert name in recipe, (name, recipe)


def test_ci_check_job_runs_the_backing_tests() -> None:
    """Catches a CI `check` job that lacks the postgres or qdrant service, or the
    two backing URL env names, so the db tests would never have a store in CI.
    """
    data = yaml.safe_load(CI.read_text(encoding="utf-8"))
    check = data["jobs"]["check"]
    services = check["services"]
    assert "postgres" in services, services
    assert "qdrant" in services, services
    env = check["env"]
    assert "OPSCONSOLE_TEST_DATABASE_URL" in env, env
    assert "OPSCONSOLE_TEST_QDRANT_URL" in env, env


def _rule_block(text: str, number: int) -> str:
    """The text of rule `number`, up to the next numbered rule or end of file."""
    lines = text.splitlines()
    start = re.compile(rf"^\s*{number}\.")
    nxt = re.compile(r"^\s*\d+\.")
    block: list[str] = []
    collecting = False
    for line in lines:
        if start.match(line):
            collecting = True
            block.append(line)
            continue
        if collecting:
            if nxt.match(line):
                break
            block.append(line)
    return "\n".join(block)


def test_rules_name_the_two_minute_fast_run() -> None:
    """Catches rules.md that does not add rule 70 naming `make fast` and the
    two-minute budget, so the fast run would have no written rule behind it.
    """
    block = _rule_block(RULES.read_text(encoding="utf-8"), 70)
    assert block, "no rule numbered 70 found in rules.md"
    assert "make fast" in block, block
    assert any(phrase in block for phrase in ("two minutes", "2 minutes")), block


def test_every_tool_path_exists() -> None:
    """Catches a stale name: a tooling file this plan references that does not
    exist, or the `backend/tests` path the Make recipes run against going away.
    """
    for path in (MAKEFILE, PYPROJECT, CI, RULES):
        assert path.exists(), path
    assert (REPO_ROOT / "backend" / "tests").exists()
