"""The fast-test tooling is pinned by reading the files it touches.

Every case reads a committed file (Makefile, pyproject, ci.yml); none
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
TRACING = REPO_ROOT / "backend" / "tests" / "api" / "test_tracing.py"


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


def test_make_test_runs_in_parallel() -> None:
    """Catches a `test` recipe that drops `-n auto`, so the full suite (backing tests
    included) would run on one core instead of across workers.
    """
    recipe = _makefile_recipe("test")
    assert "-n auto" in recipe, recipe


def test_make_test_runs_the_infra_tests() -> None:
    """Catches a `test` recipe that never calls `test-infra`, or a `test-infra` recipe
    that does not run `pytest infra/tests` in parallel, so the infra tests would stay
    out of `make test`.
    """
    test_recipe = _makefile_recipe("test")
    assert re.search(r"(\$\(MAKE\)|\bmake)\s+test-infra\b", test_recipe), test_recipe
    infra_recipe = _makefile_recipe("test-infra")
    assert infra_recipe.strip(), "no `test-infra` target in the Makefile"
    assert re.search(r"\bpytest\s+infra/tests\b", infra_recipe), infra_recipe
    assert "-n auto" in infra_recipe, infra_recipe


def test_infra_tests_carry_no_coverage_gate() -> None:
    """Catches a `test-infra` recipe that copies the backend coverage flags, which
    would fail the infra run on a coverage floor measured over backend code.
    """
    recipe = _makefile_recipe("test-infra")
    assert recipe.strip(), "no `test-infra` target in the Makefile"
    assert "pytest" in recipe, recipe
    assert "--cov" not in recipe, recipe
    assert "--cov-fail-under" not in recipe, recipe


def test_every_tool_path_exists() -> None:
    """Catches a stale name: a tooling file referenced here that does not
    exist, or the `backend/tests` path the Make recipes run against going away.
    """
    for path in (MAKEFILE, PYPROJECT, CI, TRACING):
        assert path.exists(), path
    assert (REPO_ROOT / "backend" / "tests").exists()
