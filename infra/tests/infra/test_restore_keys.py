"""`infra/evals/restore_keys.sh`: put the gateway and the help desk back on their normal keys.

An earlier change. `make eval` and `make numbers` start both on a run's throwaway keys. The restore
force-recreates them with the base compose files only, and with the run's key variables
unset, so no throwaway key outlives the run in a container. A stub `docker` on `PATH` records
every call, its directory and which key variables it could see. Nothing real is started.
The Makefile cases read the recipe text, and run the `trap` line it carries against a stub
restore script, so the exit codes are the real shell's.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "infra" / "evals" / "restore_keys.sh"
MAKEFILE = REPO_ROOT / "Makefile"

RUN_KEYS = (
    "OPSCONSOLE_API_KEYS",
    "OPSCONSOLE_ADMIN_KEY_SHA256",
    "HELPDESK_GATEWAY_API_KEYS",
    "OPSCONSOLE_EVAL_JUDGE_KEY",
)
HELPDESK_SERVICES = ("helpdesk-agent-api", "helpdesk-mcp-server", "helpdesk-scheduler")

# Records one line per call: the directory, the arguments, then each run key it could see.
STUB_DOCKER = """#!/usr/bin/env bash
{
  printf 'CALL pwd=%s args=%s\\n' "$PWD" "$*"
  for name in OPSCONSOLE_API_KEYS OPSCONSOLE_ADMIN_KEY_SHA256 \\
      HELPDESK_GATEWAY_API_KEYS OPSCONSOLE_EVAL_JUDGE_KEY; do
    if [ -n "${!name+x}" ]; then printf 'SEES %s\\n' "$name"; fi
  done
} >> "$STUB_LOG"
exit "${STUB_EXIT:-0}"
"""


@dataclass
class Call:
    pwd: str
    args: list[str]

    def files(self) -> list[str]:
        """Every value given to `-f`."""
        return [self.args[i + 1] for i, arg in enumerate(self.args) if arg == "-f"]


@dataclass
class Ran:
    code: int
    calls: list[Call]
    seen_keys: list[str]
    helpdesk: Path


def run_restore(tmp_path: Path, *, docker_exit: int = 0) -> Ran:
    assert SCRIPT.is_file(), "infra/evals/restore_keys.sh does not exist"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(STUB_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    helpdesk = tmp_path / "helpdesk"
    (helpdesk / "backend").mkdir(parents=True)
    (helpdesk / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    log = tmp_path / "docker.log"
    log.write_text("", encoding="utf-8")

    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.update({name: "throwaway-run-key" for name in RUN_KEYS})
    env.update({"STUB_LOG": str(log), "STUB_EXIT": str(docker_exit)})
    done = subprocess.run(
        ["bash", str(SCRIPT), str(helpdesk)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    calls: list[Call] = []
    seen: list[str] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("CALL "):
            match = re.fullmatch(r"CALL pwd=(.*?) args=(.*)", line)
            assert match is not None, line
            calls.append(Call(pwd=match.group(1), args=match.group(2).split()))
        elif line.startswith("SEES "):
            seen.append(line.removeprefix("SEES "))
    return Ran(code=done.returncode, calls=calls, seen_keys=seen, helpdesk=helpdesk)


def test_restore_recreates_gateway_and_help_desk_on_base_compose(tmp_path: Path) -> None:
    """Catches a restore that rebuilds with the eval or numbers overlay (the throwaway keys
    stay wired in), that recreates only one of the two stacks, that forgets
    `--force-recreate` (compose then keeps the running container and its old keys), that
    leaves out one of the three help desk services or the `app` profile, or that points the
    help desk call at this repo's compose file instead of the help desk's own."""
    ran = run_restore(tmp_path)

    assert ran.code == 0
    everything = " ".join(arg for call in ran.calls for arg in call.args)
    assert "infra/evals" not in everything
    assert "infra/local" not in everything
    for call in ran.calls:
        assert call.pwd.count("infra/") == 0
        assert {Path(path).name for path in call.files()} <= {"docker-compose.yml"}

    gateway_calls = [call for call in ran.calls if "gateway" in call.args]
    assert len(gateway_calls) == 1
    gateway = gateway_calls[0]
    assert "--force-recreate" in gateway.args
    assert not set(HELPDESK_SERVICES) & set(gateway.args)
    assert str(ran.helpdesk) not in " ".join([gateway.pwd, *gateway.args])
    assert [Path(path).name for path in gateway.files()] in (["docker-compose.yml"], [])

    helpdesk_calls = [call for call in ran.calls if set(HELPDESK_SERVICES) & set(call.args)]
    assert len(helpdesk_calls) == 1
    helpdesk = helpdesk_calls[0]
    assert set(HELPDESK_SERVICES) <= set(helpdesk.args)
    assert "gateway" not in helpdesk.args
    assert "--force-recreate" in helpdesk.args
    assert "--profile" in helpdesk.args
    assert helpdesk.args[helpdesk.args.index("--profile") + 1] == "app"
    assert str(ran.helpdesk) in " ".join([helpdesk.pwd, *helpdesk.args])


def test_restore_runs_without_the_run_keys(tmp_path: Path) -> None:
    """Catches a restore that inherits the run's key variables (compose would read them and
    put the throwaway keys straight back into the recreated containers): the stub docker
    must see none of the four, though the script was started with all four set."""
    ran = run_restore(tmp_path)

    assert ran.code == 0
    assert ran.calls != []
    assert ran.seen_keys == []


def test_restore_fails_when_docker_fails(tmp_path: Path) -> None:
    """Catches a restore that swallows a failed `docker compose` (a missing `set -e`, a
    `|| true`) and reports success while the stack still runs on the throwaway keys."""
    ran = run_restore(tmp_path, docker_exit=1)

    assert ran.calls != []
    assert ran.code != 0


# --- the Makefile hands the restore to both targets -----------------------------------


def recipe(target: str) -> str:
    """The target's recipe lines joined, backslash continuations folded into one line."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"{target}:", line))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            body.append(line.strip())
        elif line.strip() == "" or line.startswith("#"):
            continue
        else:
            break
    return re.sub(r"\\\n", " ", "\n".join(body))


def trap_statement(target: str) -> str:
    """The `trap ... EXIT` statement of a recipe, with make's `$$` turned into shell `$`."""
    text = recipe(target)
    match = re.search(r"\btrap\s+.+?\s+EXIT\b", text, flags=re.DOTALL)
    assert match is not None, f"the {target} recipe sets no EXIT trap"
    statement = match.group(0)
    assert "infra/evals/restore_keys.sh" in statement
    assert text.index(statement) < text.index(" up "), f"{target}: trap is set after the first up"
    return re.sub(r"\$\([A-Z_]+\)", "helpdesk-dir", statement.replace("$$", "$"))


def run_trap(
    tmp_path: Path, statement: str, *, restore_exit: int, body_exit: int
) -> tuple[int, list[str]]:
    """Run the trap statement, then exit `body_exit`, over a stub restore script."""
    stub = tmp_path / "infra" / "evals" / "restore_keys.sh"
    stub.parent.mkdir(parents=True)
    stub.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> restore.log\nexit {restore_exit}\n', encoding="utf-8"
    )
    stub.chmod(0o755)
    done = subprocess.run(
        ["bash", "-c", f"{statement}\nexit {body_exit}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    log = tmp_path / "restore.log"
    return done.returncode, log.read_text(encoding="utf-8").splitlines() if log.exists() else []


@pytest.mark.parametrize("target", ["eval", "numbers"])
def test_eval_and_numbers_restore_on_exit(target: str, tmp_path: Path) -> None:
    """Catches a recipe with no EXIT trap (a failed run leaves the throwaway keys in place),
    one set after the first `up` (a failure in between skips it), a restore that is not given
    the help desk directory, a trap that hides the run's own failure behind the restore's
    success, and one that hides a failed restore behind a passing run."""
    statement = trap_statement(target)

    code, calls = run_trap(tmp_path / "pass", statement, restore_exit=0, body_exit=0)
    assert (code, len(calls)) == (0, 1)
    assert calls[0].split() != []

    code, calls = run_trap(tmp_path / "failed-run", statement, restore_exit=0, body_exit=3)
    assert (code, len(calls)) == (3, 1)

    code, calls = run_trap(tmp_path / "failed-restore", statement, restore_exit=1, body_exit=0)
    assert code != 0
    assert len(calls) == 1

    code, _ = run_trap(tmp_path / "both", statement, restore_exit=1, body_exit=3)
    assert code != 0
