r"""An earlier change: ruff checks and formats the whole repo, with the project's own settings.

The settings live in `backend/pyproject.toml` (line length 100, the selected rules, the
`.claude/**` E501 ignore). Every place that runs ruff - the pre-commit hooks, the Makefile's
`lint` and `format`, the CI format check and the Claude edit hook - passes
`--config backend/pyproject.toml` and runs from the repo root over `.`, so `infra/`, `dev/`
and `.claude/` are held to the same rules as `backend/`.

Text and YAML reads of the four tooling files, plus ruff subprocesses run from the repo root
through `uv run --project backend`. No services, no import of any `app` or `infra` module.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

#: This file's own location, never the working directory - `pytest` runs from anywhere.
#: The file sits at `infra/tests/ci/<file>`, so parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
MAKEFILE = REPO_ROOT / "Makefile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
EDIT_HOOK = REPO_ROOT / ".claude" / "hooks" / "format_and_lint.sh"

#: The settings file every ruff call names, spelled from the repo root.
CONFIG_PATH = "backend/pyproject.toml"

#: The four files that run ruff.
TOOLING_FILES = [PRE_COMMIT, MAKEFILE, CI_WORKFLOW, EDIT_HOOK]

#: The project's line length, from `backend/pyproject.toml`.
LINE_LENGTH = 100

#: The ruff call every subprocess here starts with.
RUFF = ["uv", "run", "--project", "backend", "ruff"]

RUFF_PRE_COMMIT_REPO = "https://github.com/astral-sh/ruff-pre-commit"


def _load_yaml(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document


def _run(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _check_stdin(filename: str, source: str) -> subprocess.CompletedProcess[str]:
    """`ruff check` over `source` on stdin, as if it were the file `filename`."""
    args = ["check", "--config", CONFIG_PATH, "--no-cache", "--stdin-filename", filename, "-"]
    return _run([*RUFF, *args], stdin=source)


def _string_line(length: int) -> str:
    """One lint-clean line `x = "aaa..."` of exactly `length` characters."""
    line = 'x = "' + "a" * (length - 6) + '"'
    assert len(line) == length
    return line


def _config_value(tokens: list[str]) -> str | None:
    """The value after `--config`, or None when the flag is absent or last."""
    for index, token in enumerate(tokens):
        if token == "--config" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _ruff_hook_args() -> dict[str, list[str]]:
    """Every ruff hook's args, by hook id, across every ruff-pre-commit repo entry."""
    document = _load_yaml(PRE_COMMIT)
    found: dict[str, list[str]] = {}
    for repo in document["repos"]:
        if repo.get("repo") != RUFF_PRE_COMMIT_REPO:
            continue
        for hook in repo.get("hooks") or []:
            found[str(hook["id"])] = [str(arg) for arg in hook.get("args") or []]
    return found


def _make_variables(text: str) -> dict[str, str]:
    """The Makefile's simple `NAME = value` assignments."""
    variables: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*(.*)", line)
        if match:
            variables[match.group(1)] = match.group(2).strip()
    return variables


def _expand(line: str, variables: dict[str, str]) -> str:
    """Replace every `$(NAME)` defined in the Makefile by its value."""
    return re.sub(
        r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)",
        lambda match: variables.get(match.group(1), match.group(0)),
        line,
    )


def _recipe_lines(text: str, target: str) -> list[str]:
    """Every tab-indented recipe line of `target`, scanned to the target's end."""
    lines = text.splitlines()
    recipes: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith(f"{target}:"):
            continue
        for recipe in lines[index + 1 :]:
            if not recipe.startswith("\t"):
                break
            recipes.append(recipe.strip())
    return recipes


def _ci_run_lines() -> list[str]:
    """Every command line of every step's `run` in ci.yml, echo-only lines left out.

    A line that only prints a command (an `echo`) would match a command fragment
    without running it, so it is dropped.
    """
    document = _load_yaml(CI_WORKFLOW)
    lines: list[str] = []
    for job in document["jobs"].values():
        for step in job.get("steps") or []:
            run = step.get("run")
            if not isinstance(run, str):
                continue
            for line in run.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "echo ")):
                    continue
                lines.append(stripped)
    return lines


def _ruff_format_paths(tokens: list[str]) -> list[str]:
    """The positional path arguments after `ruff format`, flag values left out."""
    start = next(
        index + 2
        for index in range(len(tokens) - 1)
        if tokens[index] == "ruff" and tokens[index + 1] == "format"
    )
    paths: list[str] = []
    skip_next = False
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--config":
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        paths.append(token)
    return paths


def _is_ruff_format(tokens: list[str]) -> bool:
    return any(
        tokens[index] == "ruff" and tokens[index + 1] == "format"
        for index in range(len(tokens) - 1)
    )


def _run_hook(target: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(REPO_ROOT)
    return subprocess.run(
        ["bash", str(EDIT_HOOK)],
        cwd=REPO_ROOT,
        input=json.dumps({"tool_input": {"file_path": str(target)}}),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_pre_commit_ruff_hooks_use_backend_config() -> None:
    """Catches a pre-commit ruff hook that runs without `--config backend/pyproject.toml`
    (so files outside `backend/` get ruff's defaults) or a check hook that lost `--fix`."""
    hooks = _ruff_hook_args()
    assert "ruff-check" in hooks, hooks
    assert "ruff-format" in hooks, hooks
    for hook_id in ("ruff-check", "ruff-format"):
        assert _config_value(hooks[hook_id]) == CONFIG_PATH, (hook_id, hooks[hook_id])
    assert "--fix" in hooks["ruff-check"], hooks["ruff-check"]


def test_makefile_lint_and_format_use_backend_config() -> None:
    """Catches a `lint` or `format` recipe that still runs ruff from inside `backend/`
    (`--directory backend`), names no config, or runs over a path other than `.`."""
    text = MAKEFILE.read_text(encoding="utf-8")
    variables = _make_variables(text)
    for target, subcommand in (("lint", "check"), ("format", "format")):
        ruff_lines = [line for line in _recipe_lines(text, target) if "ruff" in line]
        assert ruff_lines, f"no ruff recipe line under `{target}:`"
        for raw in ruff_lines:
            line = _expand(raw, variables)
            tokens = shlex.split(line)
            assert f"ruff {subcommand}" in line, (target, line)
            assert _config_value(tokens) == CONFIG_PATH, (target, line)
            assert tokens[-1] == ".", (target, line)
            assert "--directory" not in tokens, (target, line)


def test_ci_format_check_covers_whole_repo() -> None:
    """Catches a CI format check still limited to `backend` or run without the project
    config, so a badly formatted file in `infra/` or `dev/` would pass CI."""
    format_lines = [line for line in _ci_run_lines() if _is_ruff_format(shlex.split(line))]
    assert format_lines, "no `ruff format` step in ci.yml"
    assert any(
        "ruff format --config backend/pyproject.toml --check ." in line
        and shlex.split(line)[-1] == "."
        for line in format_lines
    ), format_lines
    for line in format_lines:
        tokens = shlex.split(line)
        assert _config_value(tokens) == CONFIG_PATH, line
        assert _ruff_format_paths(tokens) == ["."], line


def test_ruff_check_clean_with_project_settings() -> None:
    """Catches a tree that still holds lint errors outside `backend/` once the project's
    rules apply to the whole repo."""
    result = _run([*RUFF, "check", "--config", CONFIG_PATH, "--no-cache", "."])
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


def test_ruff_format_clean_with_project_settings() -> None:
    """Catches a tree that still holds files ruff would reformat outside `backend/` once
    the format check covers the whole repo."""
    result = _run([*RUFF, "format", "--config", CONFIG_PATH, "--no-cache", "--check", "."])
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


def test_long_line_outside_backend_fails() -> None:
    """Catches a line length other than exactly 100 for files outside `backend/`: 101 must
    fail with E501 (a looser limit or E501 dropped passes it), 100 must pass (ruff's default
    88, used when the config is not applied, fails it)."""
    over = _check_stdin("infra/tests/x.py", _string_line(LINE_LENGTH + 1) + "\n")
    assert over.returncode == 1, over.stdout + over.stderr
    assert "E501" in over.stdout, over.stdout
    at_limit = _check_stdin("infra/tests/x.py", _string_line(LINE_LENGTH) + "\n")
    assert at_limit.returncode == 0, at_limit.stdout + at_limit.stderr


def test_dev_scripts_are_skipped_from_root() -> None:
    """Catches a `dev/**` exclude that no longer matches when ruff runs from the repo root,
    so `make lint` and `make format` would start reading the private helper scripts."""
    result = _check_stdin("dev/scripts/x.py", _string_line(LINE_LENGTH + 1) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_claude_ignore_resolves_from_root() -> None:
    """Catches a `.claude/**` E501 ignore that no longer matches when ruff runs from the
    repo root (a glob resolved against `backend/`), so long hook lines would start failing."""
    result = _check_stdin(".claude/hooks/x.py", _string_line(LINE_LENGTH + 1) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_edit_hook_formats_python_file(tmp_path: Path) -> None:
    """Catches an edit hook that still looks for a root `pyproject.toml` (gone since
    an earlier change) and so exits 0 without formatting anything."""
    target = tmp_path / "badly_spaced.py"
    target.write_text("x=1\ny =  [1,2]\n", encoding="utf-8")
    result = _run_hook(target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert target.read_text(encoding="utf-8") == "x = 1\ny = [1, 2]\n"


def test_edit_hook_reports_unfixable_issue(tmp_path: Path) -> None:
    """Catches an edit hook that never reaches `ruff check` (the root `pyproject.toml`
    guard) or swallows its failure, so an over-long line is let through silently."""
    target = tmp_path / "too_long.py"
    target.write_text(_string_line(LINE_LENGTH + 1) + "\n", encoding="utf-8")
    result = _run_hook(target)
    assert result.returncode == 2, result.stdout + result.stderr
    assert str(target) in result.stderr, result.stderr


def test_named_config_paths_exist() -> None:
    """Catches a tooling file that names a `pyproject.toml` path which does not exist from
    the repo root (the hook's bare `pyproject.toml`, or a misspelt `backend/` path)."""
    assert (REPO_ROOT / CONFIG_PATH).is_file()
    assert EDIT_HOOK.is_file()
    for path in TOOLING_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                continue
            for named in re.findall(r"[\w./-]*pyproject\.toml", line):
                assert (REPO_ROOT / named).is_file(), (path.name, named, line)
