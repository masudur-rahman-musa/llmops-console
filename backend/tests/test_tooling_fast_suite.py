"""The fast-test tooling this plan adds is pinned by reading the files it touches.

Every case reads a committed file (Makefile, pyproject, ci.yml, rules.md); none
runs a service, so none carries the `db` marker.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RULES = REPO_ROOT / "dev" / "docs" / "rules.md"
TRACING = REPO_ROOT / "backend" / "tests" / "api" / "test_tracing.py"
AFTER_TIMING = REPO_ROOT / "dev" / "ops" / "test-timing-after.txt"

# The two tracing tests an earlier change's timing found at about seven seconds each. Plan
# 57 bounds the OTLP export retry in `conftest.py` so both drop under a second;
# either that (they are absent above 2s in the after timing) or a `slow` marker
# satisfies this file.
SLOW_TRACING_TESTS = (
    "test_a_fallback_is_recorded_on_the_trace_as_a_count",
    "test_a_gateway_error_raised_in_middleware_keeps_its_status_and_code",
)


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


def test_make_test_runs_in_parallel() -> None:
    """Catches a `test` recipe that drops `-n auto`, so the full suite (backing tests
    included) would run on one core instead of across workers.
    """
    recipe = _makefile_recipe("test")
    assert "-n auto" in recipe, recipe


def test_make_test_runs_the_infra_tests() -> None:
    """Catches a `test` recipe that never calls `test-infra`, or a `test-infra` recipe
    that does not run `pytest infra/tests` in parallel, so the infra tests would stay
    out of `make test` (the gap an earlier change closes).
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


def _carries_slow_marker(source: str, test_name: str) -> bool:
    """True when the named function in `source` is decorated `@pytest.mark.slow`."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Attribute) and target.attr == "slow":
                    return True
    return False


def _slowest_seconds(timing: str, test_name: str) -> float:
    """The largest duration the after-timing records for one test, or 0.0 if absent.

    A `--durations` line reads `<seconds>s <phase> <nodeid>`, so this reads every
    setup, call and teardown row the named test has and keeps the worst.
    """
    pattern = re.compile(rf"^\s*([\d.]+)s\s+\w+\s+\S*::{re.escape(test_name)}\b", re.MULTILINE)
    return max((float(value) for value in pattern.findall(timing)), default=0.0)


def test_the_two_slow_tracing_tests_are_fast_or_slow_marked() -> None:
    """Catches the two 7-second tracing tests coming back: each must either carry the
    `slow` marker, or be absent above 2s from the recorded after timing. Read against the
    tracing source and the project notes together, so marking one slow or
    making it fast both satisfy the file, but doing neither does not.
    """
    tracing_source = TRACING.read_text(encoding="utf-8")
    timing = AFTER_TIMING.read_text(encoding="utf-8")
    for name in SLOW_TRACING_TESTS:
        slow_marked = _carries_slow_marker(tracing_source, name)
        slowest = _slowest_seconds(timing, name)
        assert slow_marked or slowest <= 2.0, (name, slow_marked, slowest)


def test_after_timing_records_before_and_after() -> None:
    """Catches an after-timing file that does not report the before wall, the after wall
    and the test count side by side, which is the whole point of recording it.
    """
    text = AFTER_TIMING.read_text(encoding="utf-8").lower()
    assert "before" in text, text
    assert "after" in text, text
    assert "wall" in text, text
    assert re.search(r"\d[\d,]*\s+passed", text), text


def test_every_tool_path_exists() -> None:
    """Catches a stale name: a tooling file this plan references that does not
    exist, or the `backend/tests` path the Make recipes run against going away.
    """
    for path in (MAKEFILE, PYPROJECT, CI, RULES, TRACING, AFTER_TIMING):
        assert path.exists(), path
    assert (REPO_ROOT / "backend" / "tests").exists()
