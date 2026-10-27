"""`make console-check`: the live junk check on the running console.

An earlier change. Every case reads files as text and starts nothing; the check itself
needs the local stack and is never part of `make verify` or CI.
"""

from __future__ import annotations

import re
from pathlib import Path

from infra.test_hetzner_module import read

REPO_ROOT = Path(__file__).resolve().parents[3]
MAKEFILE = REPO_ROOT / "Makefile"
WEB = REPO_ROOT / "web"
LIVE_CONFIG = WEB / "vitest.live.config.ts"
NORMAL_CONFIG = WEB / "vitest.config.ts"
CLOSE_SKILL = REPO_ROOT / ".claude" / "skills" / "close" / "SKILL.md"
TEST_MODE_GUIDE = REPO_ROOT / ".claude" / "guides" / "test-mode.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DASHBOARD_ROUTES = REPO_ROOT / "backend" / "app" / "api" / "routes" / "dashboard.py"


def recipe(target: str) -> str:
    """The text of `target:`: its own line and the tab-indented lines after it."""
    lines = read(MAKEFILE).splitlines()
    start = next(
        (i for i, line in enumerate(lines) if re.match(rf"{re.escape(target)}:", line)), None
    )
    assert start is not None, f"Makefile has no {target} target"
    body = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            body.append(line)
        else:
            break
    return "\n".join(body)


def paragraph_with(text: str, needle: str) -> str:
    """The blank-line-separated paragraph of `text` that holds `needle`."""
    found = [part for part in re.split(r"\n\s*\n", text) if needle in part]
    assert found, f"no paragraph names {needle!r}"
    return found[0]


def test_console_check_target_runs_the_live_config() -> None:
    """Catches a `console-check` target that runs the normal suite instead of the live
    config, or starts it without `web/.env.local`, so it has no session secret to sign with."""
    body = recipe("console-check")
    assert re.search(r"--config[ =]vitest\.live\.config\.ts\b", body), body
    assert "-C web" in body, body
    assert re.search(r"(?:^|[\s;])(?:\.|source) web/\.env\.local\b", body), body
    # The env file is exported before vitest starts, so the child process sees the secret.
    assert body.find("web/.env.local") < body.find("vitest"), body
    assert "set -a" in body, body


def test_console_check_is_not_in_verify_or_ci() -> None:
    """Catches the live check wired into `make verify` or a CI workflow, where no stack runs
    and it would fail every build."""
    verify = recipe("verify")
    assert "console-check" not in verify, verify
    assert "vitest.live" not in verify, verify
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        text = read(workflow)
        assert "console-check" not in text, workflow.name
        assert "vitest.live" not in text, workflow.name


def test_the_normal_run_skips_the_live_folder() -> None:
    """Catches a normal vitest config that still picks up `tests/live/`, so `pnpm test` and
    `make verify` would try to reach a running console."""
    text = read(NORMAL_CONFIG)
    assert re.search(r"exclude:\s*\[[^\]]*[\"']tests/live/\*\*[\"']", text), text
    live = read(LIVE_CONFIG)
    assert re.search(r"include:\s*\[\s*[\"']tests/live/\*\*/\*\.test\.ts[\"']\s*\]", live), live


def test_close_runs_console_check() -> None:
    """Catches a close skill that never runs the check, runs it without saying when (a plan
    touching `web/` or the console routes), lets a failure pass, or skips it under a fast
    test-mode."""
    part = paragraph_with(read(CLOSE_SKILL), "make console-check")
    assert "web/" in part, part
    assert "backend/app/api/routes/dashboard.py" in part, part
    assert "every test-mode" in part, part
    assert re.search(r"failure stops the close", part, re.IGNORECASE), part
    guide = paragraph_with(read(TEST_MODE_GUIDE), "make console-check")
    assert "every mode" in guide, guide


def test_every_path_named_exists() -> None:
    """Catches a path the new target, configs or skill line names that is not there, such as
    a live test, the setup file or the shared junk check."""
    for path in (
        LIVE_CONFIG,
        WEB / "tests" / "live" / "check.ts",
        WEB / "tests" / "live" / "console-check.test.ts",
        WEB / "tests" / "junk.ts",
        WEB / "tests" / "setup.ts",
        WEB / "tests" / "fixtures" / "messy.ts",
        DASHBOARD_ROUTES,
        CLOSE_SKILL,
    ):
        assert path.is_file(), f"{path.relative_to(REPO_ROOT)} does not exist"
    setup = re.search(r"setupFiles:\s*\[\s*[\"']([^\"']+)[\"']", read(NORMAL_CONFIG))
    assert setup is not None, "vitest.config.ts names no setup file"
    assert (WEB / setup.group(1)).is_file(), setup.group(1)
