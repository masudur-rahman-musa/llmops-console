r"""An earlier change: the repo root keeps only what a tool must find there.

Two compose overlays move next to the kits they run (`infra/load/compose.yml`,
`infra/tradeoff/compose.yml`), the owner's values move from a root `.env` to `backend/.env`,
the root check refuses a root `.env`, the allowlist names only two compose files, and
`.editorconfig` goes.

Text, YAML and filesystem reads over the repository, plus two subprocesses run against
throwaway copies in `tmp_path`: one `docker compose ... config --quiet` (skipped when docker is
not on PATH) and one run of the real `.github/scripts/check_root.py` inside a fresh git repo. No
import of any `app`, `infra` or `tests` module. The compose call passes `--quiet` so it never
prints a resolved value, even by accident.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

#: This file's own location, never the working directory - `pytest` runs from anywhere.
#: The file sits at `infra/tests/infra/<file>`, so parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_COMPOSE = REPO_ROOT / "docker-compose.yml"
LOAD_OVERLAY = REPO_ROOT / "infra" / "load" / "compose.yml"
OLD_LOAD_OVERLAY = REPO_ROOT / "docker-compose.load.yml"
TRADEOFF_OVERLAY = REPO_ROOT / "infra" / "tradeoff" / "compose.yml"
OLD_TRADEOFF_OVERLAY = REPO_ROOT / "docker-compose.tradeoff.yml"
SERVER_SECRETS = REPO_ROOT / "infra" / "scripts" / "server_secrets.sh"
CHECK_ROOT = REPO_ROOT / ".github" / "scripts" / "check_root.py"
ROOT_ALLOWLIST = REPO_ROOT / ".claude" / "root-allowlist"
EDITORCONFIG = REPO_ROOT / ".editorconfig"

#: The gateway's `env_file`, in this exact order: the demo values, then the owner's over them.
GATEWAY_ENV_FILE: list[dict[str, Any]] = [
    {"path": "backend/.env.demo", "required": False},
    {"path": "backend/.env", "required": False},
]

#: The one assignment of the deploy script's env file, with its new default.
SERVER_ENV_FILE_LINE = 'env_file="${SERVER_ENV_FILE:-backend/.env}"'

#: The compose `${...}` values that used to come from a root `.env`; scrubbed for the config run.
COMPOSE_VARS = (
    "POSTGRES_PASSWORD",
    "MODELS_DIR",
    "LLAMA_MODEL_FILE",
    "LLAMA_EMBED_MODEL_FILE",
)

#: The load run's compose call, overlay relative to the project folder (the root).
LOAD_CONFIG_ARGS = [
    "docker",
    "compose",
    "-f",
    "docker-compose.yml",
    "-f",
    "infra/load/compose.yml",
    "--profile",
    "load",
    "config",
    "--quiet",
]

#: The only compose file names the root may hold.
ALLOWED_COMPOSE_NAMES = ("docker-compose.yml", "docker-compose.override.yml")

#: Files whose compose file paths must all exist.
COMPOSE_NAMING_FILES = [
    REPO_ROOT / "Makefile",
    REPO_ROOT / "infra" / "scripts" / "tradeoff_run.sh",
    REPO_ROOT / "infra" / "tradeoff" / "README.md",
    REPO_ROOT / "README.md",
]

#: A compose FILE path token. No space in the class, so the `docker compose` command (with a
#: space) is never a hit; `"`, `$`, `{` and `}` are in it so `"$REPO_ROOT/x.yml` is one token.
COMPOSE_PATH = re.compile(r"[\w./$\"{}-]*(?:docker-compose[\w.-]*\.ya?ml|compose\.ya?ml)")

#: Prefixes stripped from a token before it is read as a path from the repo root.
PATH_PREFIXES = ('"', "$REPO_ROOT/", "${REPO_ROOT}/", "./")


def load_yaml(path: Path) -> dict[str, Any]:
    """One YAML mapping, or a failure naming the file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} is not a YAML mapping"
    return data


def build_contexts(compose: dict[str, Any]) -> list[str]:
    """Every service's build context, whether `build:` is a string or a mapping."""
    contexts: list[str] = []
    for service in (compose.get("services") or {}).values():
        build = (service or {}).get("build")
        if isinstance(build, str):
            contexts.append(build)
        elif isinstance(build, dict) and isinstance(build.get("context"), str):
            contexts.append(build["context"])
    return contexts


def normalise(token: str) -> str:
    """Strip quote and `$REPO_ROOT/` style prefixes until none is left."""
    changed = True
    while changed:
        changed = False
        for prefix in PATH_PREFIXES:
            if token.startswith(prefix):
                token = token[len(prefix) :]
                changed = True
    return token.rstrip('"')


def test_load_overlay_lives_with_its_kit() -> None:
    """Catches a build that copies the load overlay to `infra/load/` but leaves the root copy."""
    assert LOAD_OVERLAY.is_file(), f"{LOAD_OVERLAY} is missing"
    assert not OLD_LOAD_OVERLAY.exists(), f"{OLD_LOAD_OVERLAY} is still at the root"


def test_tradeoff_overlay_lives_with_its_kit() -> None:
    """Catches a build that copies the tradeoff overlay to `infra/tradeoff/` but leaves the root
    copy."""
    assert TRADEOFF_OVERLAY.is_file(), f"{TRADEOFF_OVERLAY} is missing"
    assert not OLD_TRADEOFF_OVERLAY.exists(), f"{OLD_TRADEOFF_OVERLAY} is still at the root"


def test_gateway_reads_backend_env_files() -> None:
    """Catches a gateway still reading the root `.env`, the two files swapped, or either one
    required."""
    gateway = (load_yaml(BASE_COMPOSE).get("services") or {}).get("gateway") or {}
    assert gateway.get("env_file") == GATEWAY_ENV_FILE


def test_compose_config_without_any_env(tmp_path: Path) -> None:
    """Catches a gateway whose `backend/` env file is required, so a fresh checkout cannot parse.

    Neither `backend/.env.demo` nor `backend/.env` is copied, as on a fresh checkout.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    assert LOAD_OVERLAY.is_file(), f"{LOAD_OVERLAY} is missing"
    shutil.copyfile(BASE_COMPOSE, tmp_path / "docker-compose.yml")
    (tmp_path / "infra" / "load").mkdir(parents=True)
    shutil.copyfile(LOAD_OVERLAY, tmp_path / "infra" / "load" / "compose.yml")
    # Empty build-context folders, so a missing context is never why the run fails.
    for source in (BASE_COMPOSE, LOAD_OVERLAY):
        for context in build_contexts(load_yaml(source)):
            (tmp_path / context).mkdir(parents=True, exist_ok=True)
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "backend" / ".env").exists()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in COMPOSE_VARS and not key.startswith("COMPOSE_")
    }
    completed = subprocess.run(
        LOAD_CONFIG_ARGS,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`{' '.join(LOAD_CONFIG_ARGS)}` exited {completed.returncode}:\n{completed.stderr}"
    )


def test_server_secrets_default_env_file() -> None:
    """Catches a deploy script still defaulting to the root `.env`, or a second `env_file=`
    overriding the new default."""
    lines = [
        line.strip()
        for line in SERVER_SECRETS.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("env_file=")
    ]
    assert lines == [SERVER_ENV_FILE_LINE]


def _check_root(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run the real root check against `repo`."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["CLAUDE_PROJECT_DIR"] = str(repo)
    return subprocess.run(
        [sys.executable, str(CHECK_ROOT)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_check_refuses_root_env(tmp_path: Path) -> None:
    """Catches a root check that still sees only git's file list, so a gitignored root `.env`
    passes."""
    repo = tmp_path / "repo"
    (repo / ".claude").mkdir(parents=True)
    shutil.copyfile(ROOT_ALLOWLIST, repo / ".claude" / "root-allowlist")
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)

    clean = _check_root(repo)
    assert clean.returncode == 0, f"stray on a clean copy:\n{clean.stdout}{clean.stderr}"

    (repo / ".env").write_text("PLACEHOLDER=1\n", encoding="utf-8")
    dirty = _check_root(repo)
    assert dirty.returncode == 1, f"exit {dirty.returncode} with a root .env:\n{dirty.stdout}"
    assert re.search(r"(^|[^\w.])\.env($|[^\w.])", dirty.stdout), dirty.stdout


def test_allowlist_names_only_two_compose_files() -> None:
    """Catches an allowlist still carrying the `docker-compose*.yml` glob that lets a new overlay
    land at the root."""
    lines = [
        line.strip()
        for line in ROOT_ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for name in ALLOWED_COMPOSE_NAMES:
        assert name in lines, f"{name} is not in the allowlist"
    patterns = [
        line
        for line in lines
        if line.startswith("docker-compose") and any(char in line for char in "*?[")
    ]
    assert patterns == []


def test_no_editorconfig() -> None:
    """Catches a build that forgets to delete `.editorconfig`."""
    assert not EDITORCONFIG.exists()


def test_named_compose_paths_exist() -> None:
    """Catches a moved overlay whose Makefile, script or README still names its old root path."""
    named: list[tuple[str, str]] = []
    for source in COMPOSE_NAMING_FILES:
        text = source.read_text(encoding="utf-8")
        for match in COMPOSE_PATH.finditer(text):
            # A path after a make variable (`$(EVAL_HELPDESK)/docker-compose.yml`) names the
            # help desk repo's file, not one in this repo.
            if text[: match.start()].endswith(")"):
                continue
            named.append((source.relative_to(REPO_ROOT).as_posix(), normalise(match.group(0))))
    assert named, "no compose file path found in any of the files"
    missing = [(source, path) for source, path in named if not (REPO_ROOT / path).is_file()]
    assert missing == []
