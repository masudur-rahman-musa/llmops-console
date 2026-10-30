r"""`infra/scripts/tradeoff_run.sh` and the artefacts around it: the three-option measurement run.

The run itself is a human action (`terraform apply` against a real account, a GPU
instance that bills by the hour), so what can be proven before it happens is the tooling, in
two halves.

The script is EXECUTED, against fake `terraform`, `aws`, `kubectl`, `ssh`, `docker`, `openssl`,
`uv`, `ops-eval`, `ops-migrate`, `redis-cli` and `curl` binaries placed first on `PATH` in
`tmp_path`, exactly the harness `test_aws_evidence.py` uses. Each fake appends
`<tool> <argv joined by space>` to `$FAKE_LOG`, exits 1 when `FAKE_FAIL_<TOOL>` is set, and
prints the canned output the script parses. Two fakes do more: `ops-eval conversations` writes
`report.json` and `report.md` into its `--out` directory even when it is made to fail, which is
which is the point of the check, and `docker` records the two credential variables from its own
environment into `$FAKE_ENV_DIR`, which is how the tenant key is seen reaching the harness
without the script ever writing it down. `uv run <tool>` re-executes `<tool>`, so a recipe
written either way is logged. `python3` stays REAL (it is re-executed through `$REAL_PYTHON3`)
because the policy build and the account-id redaction are its work, and faking it would make
both unobservable. `TRADEOFF_TEST_ROOT` (a `..._TEST_...` name no application setting reads) sends
the evidence under `tmp_path`, so no case writes into the repository,
and `HOME` is pointed there too.

The compose overlay, the manifest, the Makefile, the workflows and the two READMEs are read as
TEXT or as parsed documents, because nothing else in this repository builds them. Every path is
taken from this file's own `__file__`, and every read goes through `read()`, so a file that is not
written yet fails the one case that needs it instead of the collection.

`read`, `load_yaml`, `recipe` and the token-scan patterns come from
`test_hetzner_module.py` rather than being copied, spelled as the package `__init__` files make
it. The two happy-path runs are made ONCE per module and read by several
cases; every script run is well under a second.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from infra.test_hetzner_module import (
    ASSIGNMENT,
    SECRETISH_NAME,
    TOKEN_LITERAL,
    load_yaml,
    read,
    recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "infra" / "scripts" / "tradeoff_run.sh"
POLICY_SCRIPT = REPO_ROOT / "infra" / "scripts" / "tradeoff_policy.py"
OVERLAY_DIR = REPO_ROOT / "infra" / "tradeoff"
MANIFEST = REPO_ROOT / "infra" / "tradeoff" / "tradeoff.json"
COMPOSE_OVERLAY = REPO_ROOT / "infra" / "tradeoff" / "compose.yml"
BASE_COMPOSE = REPO_ROOT / "docker-compose.yml"
RUN_DOCS = REPO_ROOT / "docs" / "tradeoff-run"
RUN_README = RUN_DOCS / "README.md"
README = REPO_ROOT / "README.md"
MAKEFILE = REPO_ROOT / "Makefile"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: A real-looking commit SHA: 40 lowercase hex, with letters so `.upper()` changes it.
VALID_SHA = "3f9c2a7e1b4d6f8a0c2e4b6d8f0a1c3e5b7d9f1a"

#: The three options, in the order the manifest lists them.
OPTION_IDS = ("cheap-api", "strong-api", "gpu-vllm")

#: The option every non-GPU case runs, and the target the stub is reached at.
OPTION = "cheap-api"
TARGET_URL = "http://localhost:8090"

#: The small conversation file this repo ships for the stand-in target and the
#: tenant the gateway policy is built for.
SET_PATH = "backend/evals/kind-cases.jsonl"
SET_NAME = "kind-cases.jsonl"
TENANT = "evals"

#: The account ID every fake output carries, and what a capture must say instead.
ACCOUNT_ID = "123456789012"
ACCOUNT_PLACEHOLDER = "ACCOUNT_ID"

#: What the fake `openssl rand -hex 32` returns, standing in for the run's tenant key.
FAKE_TENANT_KEY = "b" * 64

#: The captures, plus the option's two report files.
CAPTURES = (
    "terraform-output.json",
    "nvidia-smi.txt",
    "pods.txt",
    "vllm-metrics.txt",
    "report.json",
    "report.md",
)

#: The host ports `docker-compose.yml` publishes today.
#: Asserted against the file as well, so this list cannot silently go stale. The backing
#: services sit on shifted numbers - 15432, 16379, 16333 - so this stack and the
#: companion Helpdesk stack can be up together; only the gateway's 8080 is stock, because that
#: is the address the Helpdesk dials. The console's 3001 joined when it moved into compose; it
#: clashes with no Helpdesk port.
BASE_HOST_PORTS = frozenset({8080, 16379, 15432, 9091, 16333, 8083, 3001, 9090, 3002})

#: What the companion Helpdesk repository's own compose publishes. This stack may not hold any
#: of them: one of the two would then fail at `up`, which is the whole reason for the shift.
HELPDESK_STACK_HOST_PORTS = frozenset({5432, 6379, 6333, 8082, 8000, 8081, 3000})

#: The stub's own port, its `PORT` default, and the only host port the overlay may add.
STUB_PORT = 8090

#: The stub image the overlay's one added service is built from.
STUB_BUILD_DIR = "backend/evals/target-stub"

#: The environment variable the overlay points at the mounted effective policy.
POLICY_ENV = "OPSCONSOLE_ROUTING_POLICY_PATH"
PROVIDERS_ENV = "OPSCONSOLE_PROVIDERS_DIR"

#: The three targets and the subcommand each must reach.
MAKE_TARGETS = (
    ("tradeoff-option", "option"),
    ("tradeoff-gpu", "gpu"),
    ("tradeoff-gpu-cost", "cost"),
)

#: The placeholder `tradeoff.example.json` ships, which neither self-hosted option may keep.
PLACEHOLDER_HOURLY = "0.0100"

#: The conclusion marker in the README's tradeoff section, matched as a WORD and in any case. The
#: section is published in sentence case, and a bare substring would also count the
#: "depending" of an ordinary sentence as a second marker.
PENDING_MARKER = re.compile(r"(?i)\bpending\b")

#: A source date in a manifest note: the day the rate was read is wanted.
NOTE_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

#: Tracing on in any spelling a bash script accepts: `set -x`, `set -ex`, `set -o xtrace`.
XTRACE = re.compile(r"(?m)^\s*set\s+(-[a-z]*x|-o\s+xtrace)")

DATE_NAME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

#: One fake for every tool, dispatched on its own name. the project's stub run exercised
#: every branch below against a script written to the agreed contract.
FAKE_TOOL = r"""#!/usr/bin/env bash
tool=$(basename "$0")
printf '%s %s\n' "$tool" "$*" >> "$FAKE_LOG"
fail_var="FAKE_FAIL_$(printf '%s' "$tool" | tr '[:lower:]-' '[:upper:]_')"
has() {
  local want=$1 arg
  shift
  for arg in "$@"; do
    [ "$arg" = "$want" ] && return 0
  done
  return 1
}
flag() {
  local want=$1 prev="" arg
  shift
  for arg in "$@"; do
    if [ "$prev" = "$want" ]; then printf '%s' "$arg"; return 0; fi
    case "$arg" in "$want"=*) printf '%s' "${arg#*=}"; return 0 ;; esac
    prev=$arg
  done
  return 1
}
case "$tool" in
  python3)
    exec "$REAL_PYTHON3" "$@"
    ;;
  uv)
    if [ "${1:-}" = "run" ]; then
      shift
      exec "$@"
    fi
    ;;
  docker)
    n=$(find "$FAKE_ENV_DIR" -type f | wc -l | tr -d ' ')
    {
      printf 'OPSCONSOLE_API_KEYS=%s\n' "${OPSCONSOLE_API_KEYS:-}"
      printf 'GATEWAY_API_KEY=%s\n' "${GATEWAY_API_KEY:-}"
    } > "$FAKE_ENV_DIR/docker-$n.env"
    ;;
  openssl)
    if has rand "$@"; then printf '%s\n' "$FAKE_TENANT_KEY"; fi
    ;;
  terraform)
    if has output "$@"; then
      cat <<'JSON'
{"instance_id":{"value":"i-0abcdef"},"public_ip":{"value":"203.0.113.10"},"role_arn":{"value":"arn:aws:iam::123456789012:role/aioc-gpu-measure"}}
JSON
    fi
    ;;
  aws)
    case "${1:-} ${2:-}" in
      "ce get-cost-and-usage") printf '%s\n' '{"ResultsByTime":[]}' ;;
    esac
    ;;
  ssh)
    case " $* " in
      *k3s.yaml*)
        printf '%s\n' 'apiVersion: v1'
        printf '%s\n' 'clusters: [{cluster: {server: "https://127.0.0.1:6443"}, name: default}]'
        ;;
      *) printf '%s\n' 'NVIDIA-SMI 560.35.03  Driver Version: 560.35.03  CUDA Version: 12.6' ;;
    esac
    ;;
  kubectl)
    if has get "$@"; then
      printf '%s\n' 'vllm-6d4f Running arn:aws:eks:eu-central-1:123456789012:pod/vllm'
    elif has exec "$@"; then
      printf '%s\n' 'NVIDIA-SMI 560.35.03  Driver Version: 560.35.03  CUDA Version: 12.6'
    fi
    ;;
  curl)
    printf '%s\n' 'vllm:num_requests_running 0'
    ;;
  ops-eval)
    if has conversations "$@"; then
      out=$(flag --out "$@") || out=""
      if [ -n "$out" ]; then
        mkdir -p "$out"
        printf '%s\n' '{"summary":{"total":10,"passed":9}}' > "$out/report.json"
        printf '%s\n' '# eval report' > "$out/report.md"
      fi
    fi
    ;;
esac
if [ -n "${!fail_var:-}" ]; then
  exit 1
fi
exit 0
"""

FAKE_NAMES = (
    "terraform",
    "aws",
    "kubectl",
    "ssh",
    "docker",
    "openssl",
    "uv",
    "ops-eval",
    "ops-migrate",
    "redis-cli",
    "curl",
    "python3",
)


@dataclass(frozen=True)
class Run:
    """One script invocation against the fakes, and where it could have left traces."""

    completed: subprocess.CompletedProcess[str]
    root: Path
    home: Path
    env_dir: Path
    log_path: Path
    started: date
    finished: date

    def log(self) -> list[str]:
        """The fake call log, one line per tool call; empty when nothing was called."""
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def exposure(self) -> str:
        """Everything the harness could have been handed: docker's argv and its environment."""
        recorded = [path.read_text(encoding="utf-8") for path in sorted(self.env_dir.iterdir())]
        return "\n".join([*self.log(), *recorded])

    def evidence(self) -> Path:
        """Where the script writes under the test root."""
        return self.root / "docs" / "tradeoff-run"


def invoke(base: Path, arguments: Sequence[str], /, **switches: str) -> Run:
    """Run the script under `base` with the fakes first on `PATH` and `switches` in env."""
    base.mkdir(parents=True, exist_ok=True)
    fakes = base / "bin"
    fakes.mkdir()
    for name in FAKE_NAMES:
        fake = fakes / name
        fake.write_text(FAKE_TOOL, encoding="utf-8")
        fake.chmod(0o755)
    root, home, env_dir = (base / name for name in ("root", "home", "env"))
    for directory in (root, home, env_dir):
        directory.mkdir()
    log_path = base / "calls.log"
    real_python = subprocess.run(
        ["/usr/bin/env", "python3", "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    environment = {
        "PATH": f"{fakes}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(home),
        "TRADEOFF_TEST_ROOT": str(root),
        "FAKE_LOG": str(log_path),
        "FAKE_ENV_DIR": str(env_dir),
        "FAKE_TENANT_KEY": FAKE_TENANT_KEY,
        "REAL_PYTHON3": real_python,
        **switches,
    }
    assert SCRIPT.is_file(), f"{SCRIPT} does not exist"
    started = datetime.now(UTC).date()
    completed = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return Run(
        completed=completed,
        root=root,
        home=home,
        env_dir=env_dir,
        log_path=log_path,
        started=started,
        finished=datetime.now(UTC).date(),
    )


def succeeded(run: Run) -> None:
    """Fail with the script's own output when it did not exit 0."""
    assert run.completed.returncode == 0, (
        f"exit {run.completed.returncode}\nstdout:\n{run.completed.stdout}\n"
        f"stderr:\n{run.completed.stderr}\ncalls:\n" + "\n".join(run.log())
    )


def calls(run: Run, tool: str) -> list[tuple[int, list[str]]]:
    """`(log index, argv tokens)` of every call to `tool`, in call order."""
    found: list[tuple[int, list[str]]] = []
    for index, line in enumerate(run.log()):
        tokens = line.split()
        if tokens and tokens[0] == tool:
            found.append((index, tokens[1:]))
    return found


def runs(tokens: list[str], *words: str) -> bool:
    """`words` appear consecutively anywhere in `tokens`: `docker compose up` counts."""
    width = len(words)
    return any(tokens[i : i + width] == list(words) for i in range(len(tokens) - width + 1))


def flag(tokens: list[str], name: str) -> str | None:
    """The value after `name` in `tokens`, or after `name=`; None when absent."""
    for position, token in enumerate(tokens):
        if token == name and position + 1 < len(tokens):
            return tokens[position + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


def policy_builds(run: Run) -> list[tuple[int, list[str]]]:
    """Every `python3 infra/scripts/tradeoff_policy.py <id> <out> <dir>` call, in call order."""
    return [
        (index, tokens)
        for index, tokens in calls(run, "python3")
        if any(token.endswith("tradeoff_policy.py") for token in tokens)
    ]


def eval_runs(run: Run) -> list[tuple[int, list[str]]]:
    """Every `ops-eval conversations ...` call, direct or through `uv run`."""
    return [
        (index, tokens) for index, tokens in calls(run, "ops-eval") if "conversations" in tokens
    ]


def flushes(run: Run) -> list[int]:
    """Every log line that both names redis and names the `response` cache prefix.

    Written as "names both", not "the first redis line": the flush may go
    through `docker compose exec redis redis-cli` or through `redis-cli` directly, and a
    `redis-cli` line that does not name the prefix is a connection check, not a flush.
    """
    return [
        index
        for index, line in enumerate(run.log())
        if "redis" in line.lower() and "response" in line
    ]


def terraform_calls(run: Run, verb: str) -> list[tuple[int, list[str]]]:
    """Terraform calls running `verb` against the GPU module directory."""
    return [
        (index, tokens)
        for index, tokens in calls(run, "terraform")
        if verb in tokens
        and any(
            token.startswith("-chdir=") and token.rstrip("/").endswith("infra/terraform/aws-gpu")
            for token in tokens
        )
    ]


def dated_dir(run: Run) -> Path:
    """The one `<UTC date>` directory the run wrote, allowing for a run across midnight."""
    evidence = run.evidence()
    assert evidence.is_dir(), f"the run wrote no {evidence}"
    dated = [
        path for path in evidence.iterdir() if path.is_dir() and DATE_NAME.fullmatch(path.name)
    ]
    assert len(dated) == 1, f"{len(dated)} dated directories: {dated}"
    allowed = {run.started.isoformat(), run.finished.isoformat()}
    assert dated[0].name in allowed, f"{dated[0].name} is not today's UTC date {sorted(allowed)}"
    return dated[0]


def host_ports(path: Path) -> set[int]:
    """Every host port a compose file publishes, from `ports:` in any of its forms."""
    document = load_yaml(path)
    services = document.get("services") or {}
    found: set[int] = set()
    for service in services.values():
        for entry in (service or {}).get("ports") or []:
            if isinstance(entry, dict):
                published = entry.get("published")
                if published is not None:
                    found.add(int(published))
                continue
            parts = str(entry).split(":")
            if len(parts) >= 2:
                found.add(int(parts[-2].rsplit("/", 1)[0]))
    return found


def readme_section(text: str, word: str) -> list[str]:
    """The lines under the ONE markdown heading whose text names `word`.

    Every heading is scanned rather than the first match taken; prose
    elsewhere in the file naming the same word is not a heading and is not read.
    """
    lines = text.splitlines()
    headings = [
        (index, len(line) - len(line.lstrip("#")))
        for index, line in enumerate(lines)
        if line.startswith("#") and word in line.casefold()
    ]
    assert len(headings) == 1, f"{len(headings)} headings name {word!r}: {headings}"
    start, level = headings[0]
    body: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.lstrip("#")
        if line.startswith("#") and 0 < len(line) - len(stripped) <= level:
            break
        body.append(line)
    return body


@pytest.fixture(scope="module")
def option_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """`option cheap-api --target <url>` against fakes that all succeed, made ONCE."""
    return invoke(tmp_path_factory.mktemp("option"), ["option", OPTION, "--target", TARGET_URL])


@pytest.fixture(scope="module")
def gpu_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """`gpu <sha>` against fakes that all succeed, made ONCE."""
    return invoke(tmp_path_factory.mktemp("gpu"), ["gpu", VALID_SHA])


# ---------------------------------------------------------------------------------------------
# Pre-checks


@pytest.mark.parametrize(
    "sha",
    [
        pytest.param(VALID_SHA[:-1], id="39-hex"),
        pytest.param(VALID_SHA + "a", id="41-hex"),
        pytest.param(VALID_SHA.upper(), id="uppercase"),
        pytest.param(VALID_SHA + "\n", id="trailing-newline"),
        pytest.param(VALID_SHA + "\r\n", id="crlf"),
        pytest.param(VALID_SHA + " ", id="trailing-space"),
    ],
)
def test_gpu_refuses_a_malformed_sha_before_any_tool(tmp_path: Path, sha: str) -> None:
    """The refused rows; the accepted 40-lowercase row is `gpu_run`. Catches
    a check written as `grep -Eq '^[0-9a-f]{40}$'`, which accepts the trailing-newline row
    because grep matches per line, a `{39,41}` bound or a `[0-9a-fA-F]` class, and a check
    placed after `terraform apply` - every fake succeeds, so without the check the run completes
    and a real one would have started an hourly-billed GPU instance for a tag that does not
    exist."""
    run = invoke(tmp_path, ["gpu", sha])

    assert run.completed.returncode != 0, f"{sha!r} was accepted"
    assert run.log() == [], f"{sha!r} reached a tool: {run.log()}"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param("2026-9-01", "2026-09-02", id="short-month-start"),
        pytest.param("2026-09-01", "2026-09-1", id="short-day-end"),
        pytest.param("2026-09-01\n", "2026-09-02", id="trailing-newline-start"),
        pytest.param("2026-09-01", "2026-09-01", id="start-equals-end"),
        pytest.param("2026-09-02", "2026-09-01", id="start-after-end"),
    ],
)
def test_cost_refuses_a_malformed_or_empty_window_before_any_tool(
    tmp_path: Path, start: str, end: str
) -> None:
    """Catches a date check on the start only (the short-day end passes), a
    grep-per-line check (the newline row), `-le` in place of a strict comparison (the equal row,
    a zero-day window Cost Explorer rejects), and no ordering check at all. The fake `aws`
    succeeds, so without the check `cost` would call it and exit 0."""
    run = invoke(tmp_path, ["cost", start, end])

    assert run.completed.returncode != 0, f"{start!r} {end!r} was accepted"
    assert run.log() == [], f"{start!r} {end!r} reached a tool: {run.log()}"


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param([], id="no-subcommand"),
        pytest.param(["measure", OPTION], id="unknown-subcommand"),
        pytest.param(["option"], id="option-without-id"),
        pytest.param(["gpu"], id="gpu-without-sha"),
        pytest.param(["cost", "2026-09-01"], id="cost-without-end"),
    ],
)
def test_a_missing_or_unknown_subcommand_is_refused(tmp_path: Path, arguments: list[str]) -> None:
    """The argument shape behind. Catches a dispatcher whose default
    branch falls through to `option` or `gpu`, and a missing positional expanding to an empty
    string that a later `terraform apply` never sees refused."""
    run = invoke(tmp_path, arguments)

    assert run.completed.returncode != 0, f"{arguments} was accepted"
    assert run.log() == [], f"{arguments} reached a tool: {run.log()}"


def test_option_refuses_a_malformed_id_before_any_tool(tmp_path: Path) -> None:
    """The id shapes the overlay path AND the evidence directory, so `option`
    refuses it before it can become either. Catches a script that passes the id straight to
    `tradeoff_policy.py` and lets the reports directory be created from `../..` first - the
    policy script's own check comes too late to stop the directory."""
    run = invoke(tmp_path, ["option", "../etc", "--target", TARGET_URL])

    assert run.completed.returncode != 0, "a traversing option id was accepted"
    assert run.log() == [], f"the id reached a tool: {run.log()}"


# ---------------------------------------------------------------------------------------------
# `option`


def test_option_builds_flushes_migrates_runs_and_tears_down_in_that_order(
    option_run: Run,
) -> None:
    """Catches the policy built after the harness is up (the gateway then
    loads the shipped `routing.yaml` and every option measures the same three candidates), the
    cache flush placed after the first `ops-eval` call or left out entirely (option
    two reads option one's answers and reports a near-zero cost at a latency no provider
    served), `ops-migrate` never run (the ledger has no tables and every turn's cost is lost),
    and a teardown that never happens, which leaves the stack holding the ports the next
    option's run needs."""
    succeeded(option_run)
    log = option_run.log()
    builds = policy_builds(option_run)
    ups = [(i, t) for i, t in calls(option_run, "docker") if runs(t, "up")]
    downs = [(i, t) for i, t in calls(option_run, "docker") if runs(t, "down")]
    migrates = calls(option_run, "ops-migrate")
    evaluations = eval_runs(option_run)
    flush = flushes(option_run)

    assert len(builds) == 1, f"{len(builds)} policy builds: {log}"
    assert ups, f"the harness is never brought up: {log}"
    assert flush, f"the response cache is never flushed: {log}"
    assert migrates, f"ops-migrate is never run: {log}"
    assert len(evaluations) == 1, f"{len(evaluations)} ops-eval conversations calls: {log}"
    assert downs, f"the harness is never torn down: {log}"

    assert builds[0][0] < ups[0][0] < flush[0] < migrates[0][0] < evaluations[0][0]
    assert evaluations[0][0] < downs[-1][0]


def test_option_runs_the_named_set_as_the_evals_tenant_with_the_policy_it_built(
    option_run: Run,
) -> None:
    """The flags, and the "the id shapes both". The `--policy` value is
     required to be the SAME path the build wrote, so the two scripts are proven wired together
     rather than each being correct alone. Catches `--policy` left off (the gateway's shipped
     policy is used and the option is not what it says), a hard-coded reports directory that
     overwrites the previous option's files, the Helpdesk set named instead of the stand-in one
    , and a tenant other than `evals`, whose rules the overlays edit."""
    succeeded(option_run)
    builds = policy_builds(option_run)
    evaluations = eval_runs(option_run)
    assert len(builds) == 1 and len(evaluations) == 1, option_run.log()
    arguments = evaluations[0][1]
    built_for = [token for token in builds[0][1] if not token.endswith("tradeoff_policy.py")]

    assert str(flag(arguments, "--cases")).endswith(SET_PATH)
    assert flag(arguments, "--target") == TARGET_URL
    assert built_for[0] == OPTION, f"the policy was built for {built_for[0]!r}"
    assert "--set" not in arguments and "--tenant" not in arguments and "--policy" not in arguments
    out = str(flag(arguments, "--out"))
    assert Path(out).parts[-4:] == ("docs", "tradeoff-run", "reports", OPTION), out


def test_a_failing_eval_still_tears_down_and_reports_the_targets_own_exit_code(
    tmp_path: Path,
) -> None:
    """A run with failed cases is a MEASUREMENT, not a broken tool: its two report
    files are written and the table reads them. Catches `ops-eval ... || true`, which hides a
    target that never answered at all, a `set -e` script that exits before the teardown and
    leaves the stack up for the next option, and an exit code invented by the wrapper."""
    run = invoke(tmp_path, ["option", OPTION, "--target", TARGET_URL], FAKE_FAIL_OPS_EVAL="1")
    downs = [(i, t) for i, t in calls(run, "docker") if runs(t, "down")]
    evaluations = eval_runs(run)
    reports = sorted(path.name for path in run.evidence().rglob("report.*"))

    assert run.completed.returncode == 1, f"exit {run.completed.returncode}, not the target's 1"
    assert evaluations, f"ops-eval was never called: {run.log()}"
    assert downs, f"no teardown after the failed run: {run.log()}"
    assert downs[-1][0] > evaluations[0][0]
    assert reports == ["report.json", "report.md"], reports


def test_the_generated_tenant_key_reaches_the_harness_and_no_file(
    option_run: Run,
) -> None:
    """The `make load` shape. The gateway is handed the SHA-256 digest and the
    stub the key itself, both through the environment. Catches a key hard-coded in the script
    or committed to `.env` (one credential shared by every run, now in git history), the raw key
    written into the evidence directory that is about to be committed, and an `echo` of it for
    debugging. The two positive assertions come first, so a script that generates no key at all
    cannot pass this case by leaking nothing."""
    succeeded(option_run)
    digest = hashlib.sha256(FAKE_TENANT_KEY.encode("utf-8")).hexdigest()
    exposure = option_run.exposure()
    written = [
        str(path.relative_to(option_run.root))
        for path in sorted(option_run.root.rglob("*"))
        if path.is_file() and FAKE_TENANT_KEY in path.read_text(encoding="utf-8", errors="replace")
    ]

    assert digest in exposure, "the gateway was never handed the key's digest"
    assert FAKE_TENANT_KEY in exposure, "the stub was never handed the key"
    assert written == [], f"the tenant key is in {written}"
    assert FAKE_TENANT_KEY not in option_run.completed.stdout, "the key is on stdout"
    assert FAKE_TENANT_KEY not in option_run.completed.stderr, "the key is on stderr"


# ---------------------------------------------------------------------------------------------
# `gpu`


def test_gpu_applies_connects_deploys_waits_measures_and_destroys_in_that_order(
    gpu_run: Run,
) -> None:
    """Catches `apply -auto-approve` (the owner never reviews the Terraform plan for an
    instance that bills by the hour), the kubeconfig fetched before the instance exists, the
    manifests applied before the kubeconfig is there, the measurement started before the fetch
    Job and the rollout have finished (vLLM answers 503 and the option scores zero), and a
    destroy that is not the single last call - an explicit destroy plus the trap would show as
    two, and a destroy before the capture would lose the evidence the run exists for."""
    succeeded(gpu_run)
    log = gpu_run.log()
    applies = terraform_calls(gpu_run, "apply")
    fetches = calls(gpu_run, "ssh")
    deploys = [
        (i, t)
        for i, t in calls(gpu_run, "kubectl")
        if runs(t, "apply") and any(token.endswith("infra/k8s/gpu") for token in t)
    ]
    waits = [
        (i, t) for i, t in calls(gpu_run, "kubectl") if "wait" in t or runs(t, "rollout", "status")
    ]
    evaluations = eval_runs(gpu_run)
    destroys = [(i, t) for i, t in terraform_calls(gpu_run, "destroy") if "-auto-approve" in t]

    assert len(applies) == 1, f"{len(applies)} terraform apply calls: {log}"
    assert "-auto-approve" not in applies[0][1], "apply does not wait for the owner"
    assert fetches, f"the kubeconfig is never fetched: {log}"
    assert len(deploys) == 1, f"{len(deploys)} `kubectl apply -k infra/k8s/gpu` calls: {log}"
    assert len(waits) >= 2, f"{len(waits)} waits, expected the fetch Job and the rollout: {log}"
    assert len(evaluations) == 1, f"{len(evaluations)} ops-eval conversations calls: {log}"
    assert len(destroys) == 1, f"{len(destroys)} destroy -auto-approve calls: {log}"

    assert applies[0][0] < fetches[0][0] < deploys[0][0] < waits[0][0]
    assert waits[1][0] < evaluations[0][0] < destroys[0][0]
    assert destroys[0][0] == len(log) - 1, f"a call after destroy: {log[destroys[0][0] :]}"


def test_gpu_measures_the_gpu_option(gpu_run: Run) -> None:
    """The "the `gpu-vllm` option run". Catches the GPU subcommand measuring whatever
    id it was last given or defaulting to `cheap-api`, which would apply a GPU instance, destroy
    it, and publish a row measured against an API provider."""
    succeeded(gpu_run)
    builds = policy_builds(gpu_run)
    evaluations = eval_runs(gpu_run)
    assert len(builds) == 1 and len(evaluations) == 1, gpu_run.log()
    out = str(flag(evaluations[0][1], "--out"))

    built_for = [token for token in builds[0][1] if not token.endswith(".py")]
    assert built_for[:1] == ["gpu-vllm"], built_for
    assert Path(out).name == "gpu-vllm", out


def test_a_failing_kubectl_still_destroys_the_instance(tmp_path: Path) -> None:
    """The one tested guard for a failed run. Catches a script with `set -e` and a destroy
    at the end but no EXIT trap: the first failing `kubectl` exits it and leaves a GPU instance
    billing until the module's own 240-minute self-shutdown. The fakes for terraform and ssh
    succeed, so the only thing that can stop the destroy is the missing trap."""
    run = invoke(tmp_path, ["gpu", VALID_SHA], FAKE_FAIL_KUBECTL="1")
    applies = terraform_calls(run, "apply")
    destroys = [(i, t) for i, t in terraform_calls(run, "destroy") if "-auto-approve" in t]

    assert run.completed.returncode != 0, "a failed kubectl reported success"
    assert len(applies) == 1, f"no apply before the failure: {run.log()}"
    assert destroys, f"no destroy after the failure: {run.log()}"
    assert destroys[-1][0] > applies[0][0]


def test_the_capture_writes_every_file_under_todays_utc_date(gpu_run: Run) -> None:
    """Catches a local-time directory name, a format other than `YYYY-MM-DD`,
    evidence written under the repository instead of the test root, and any capture skipped -
    the run is destroyed minutes later, so a capture missed is a measurement that cannot be
    repeated without paying for the instance again."""
    succeeded(gpu_run)
    directory = dated_dir(gpu_run)
    missing = [name for name in CAPTURES if not (directory / name).is_file()]

    assert missing == [], f"{directory} lacks {missing}"


def test_every_captured_account_id_is_redacted(gpu_run: Run) -> None:
    """The capture is committed to a public repository, and `terraform output`
    and `kubectl get` both carry the account id in a real run. Catches captures written straight
    from the command, and a redaction that empties the file instead of substituting, which the
    placeholder half makes visible."""
    succeeded(gpu_run)
    directory = dated_dir(gpu_run)
    leaking = [
        str(path.relative_to(directory))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ACCOUNT_ID in path.read_text(encoding="utf-8", errors="replace")
    ]
    unredacted = [
        name
        for name in ("terraform-output.json", "pods.txt")
        if ACCOUNT_PLACEHOLDER not in read(directory / name)
    ]

    assert leaking == [], f"the account id is in {leaking}"
    assert unredacted == [], f"no {ACCOUNT_PLACEHOLDER} in {unredacted}"


def test_the_kubeconfig_never_touches_the_owners_default_context(gpu_run: Run) -> None:
    """Catches the k3s kubeconfig written to `~/.kube/config`, which rewrites the
    owner's default context to a cluster that is destroyed minutes later, and a `mktemp -d`
    directory the EXIT trap never removes."""
    succeeded(gpu_run)

    assert not (gpu_run.home / ".kube").exists(), "the run touched ~/.kube"


# ---------------------------------------------------------------------------------------------
# `cost`


def test_cost_queries_the_whole_account_daily_by_service(tmp_path: Path) -> None:
    """Catches a `--filter` on a project tag (the tag is not active before
    the run, so the filter returns nothing and the GPU hour looks free), `MONTHLY` granularity
    (which cannot separate the measurement day), `BlendedCost`, no `SERVICE` group-by, and a
    one-day window refused. `cost` runs after the instance is gone, so it must touch neither
    terraform nor kubectl."""
    run = invoke(tmp_path, ["cost", "2026-09-01", "2026-09-02"])
    succeeded(run)
    queries = [t for _i, t in calls(run, "aws") if runs(t, "ce", "get-cost-and-usage")]
    cost = run.evidence() / "cost.json"

    assert len(queries) == 1, f"{len(queries)} get-cost-and-usage calls: {run.log()}"
    assert flag(queries[0], "--granularity") == "DAILY"
    assert flag(queries[0], "--metrics") == "UnblendedCost"
    assert "Key=SERVICE" in " ".join(queries[0])
    assert flag(queries[0], "--filter") is None, "the cost query is filtered"
    assert calls(run, "terraform") == [] and calls(run, "kubectl") == [], run.log()
    assert json.loads(read(cost)) == {"ResultsByTime": []}


# ---------------------------------------------------------------------------------------------
# The scripts as text


def test_both_scripts_are_executable_and_never_trace_or_name_the_default_kubeconfig() -> None:
    """Catches `set -x` or `set -euxo pipefail`, which prints every expanded
    command - the tenant key among them - to the owner's terminal and scrollback, `.kube/config`
    named anywhere, and either file committed without its execute bit, where `make
    tradeoff-option` fails with permission denied."""
    text = read(SCRIPT)

    assert XTRACE.search(text) is None, "the script turns on tracing"
    assert ".kube/config" not in text
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable"
    assert POLICY_SCRIPT.is_file(), f"{POLICY_SCRIPT} does not exist"
    assert os.access(POLICY_SCRIPT, os.X_OK), f"{POLICY_SCRIPT} is not executable"


def test_no_file_this_plan_writes_carries_a_token_shaped_literal() -> None:
    """Catches a provider key pasted into an overlay while
    debugging, a digest left in the run README, and a tenant key committed with the evidence.
    Each path is asserted to exist first, so an empty scan cannot pass as a clean one, and the
    scan is anchored on a secret-shaped NAME: a bare long-run ban would
    redden on the model names and the commit sha that is also needed."""
    roots = (OVERLAY_DIR, SCRIPT, POLICY_SCRIPT, RUN_DOCS)
    for path in roots:
        assert path.exists(), f"{path} does not exist"
    files = sorted(
        candidate
        for root in roots
        for candidate in ([root] if root.is_file() else root.rglob("*"))
        if candidate.is_file()
    )
    assert files, "the scan found no files"
    findings = [
        f"{path.relative_to(REPO_ROOT)}: {name}"
        for path in files
        for name, value in ASSIGNMENT.findall(path.read_text(encoding="utf-8", errors="replace"))
        if SECRETISH_NAME.fullmatch(name) and TOKEN_LITERAL.fullmatch(value)
    ]

    assert findings == [], f"token-shaped literals: {findings}"


# ---------------------------------------------------------------------------------------------
# The compose overlay


def test_the_overlay_adds_exactly_the_stub_service() -> None:
    """Catches a second gateway or a duplicated platform service added here
    instead of merged (one harness: the shipped compose file plus this overlay), and
    the stub pulled as an image instead of built from the Dockerfile beside it, which would need
    a published image for a throwaway target."""
    base = set(load_yaml(BASE_COMPOSE).get("services") or {})
    overlay = load_yaml(COMPOSE_OVERLAY).get("services") or {}
    added = sorted(set(overlay) - base)

    assert len(added) == 1, f"the overlay adds {added}"
    build = overlay[added[0]].get("build")
    context = build.get("context") if isinstance(build, dict) else build
    assert str(context).rstrip("/").endswith(STUB_BUILD_DIR), f"built from {context}"


def test_the_overlay_mounts_the_effective_policy_read_only_and_points_the_gateway_at_it() -> None:
    """Catches the policy mounted read-write (the gateway process could
    rewrite the file the measurement is defined by), the mounts and
    `OPSCONSOLE_ROUTING_POLICY_PATH` or `OPSCONSOLE_PROVIDERS_DIR` naming different paths - the
    gateway then silently loads the shipped files and every option measures the same
    candidates - and `extra_hosts` left off, without which the GPU
    option's `host.docker.internal` does not resolve and every turn fails to connect."""
    gateway = (load_yaml(COMPOSE_OVERLAY).get("services") or {}).get("gateway") or {}
    environment = gateway.get("environment") or {}
    if isinstance(environment, list):
        environment = dict(
            entry.split("=", 1) for entry in environment if isinstance(entry, str) and "=" in entry
        )
    mounts = [str(entry) for entry in gateway.get("volumes") or []]
    read_only = [entry for entry in mounts if entry.endswith(":ro")]
    target = str(environment.get(POLICY_ENV, ""))
    providers = str(environment.get(PROVIDERS_ENV, ""))

    assert target, f"{POLICY_ENV} is not set on the gateway"
    assert providers, f"{PROVIDERS_ENV} is not set on the gateway"
    assert len(read_only) == len(mounts) == 2, f"gateway volumes: {mounts}"
    assert sorted(entry.split(":")[1] for entry in read_only) == sorted([target, providers]), (
        f"{read_only} do not mount {target} and {providers}"
    )
    assert "host.docker.internal:host-gateway" in [
        str(entry) for entry in gateway.get("extra_hosts") or []
    ], f"extra_hosts: {gateway.get('extra_hosts')}"


def test_the_overlay_publishes_no_port_the_base_stack_already_publishes() -> None:
    """Catches the stub published on 8080, which the gateway
    already holds - compose then refuses to start the whole harness and the run is lost at `up`.
    The base file's own port set is asserted against the literal list too, so this case cannot go
    stale by reading a file that changed underneath it. It also refuses any port the companion
    Helpdesk stack publishes: the two stacks are meant to run at once, this one being the
    gateway the other calls, so a shared host port means only ever having one of them."""
    base = host_ports(BASE_COMPOSE)
    overlay = host_ports(COMPOSE_OVERLAY)

    assert base == set(BASE_HOST_PORTS), f"docker-compose.yml now publishes {sorted(base)}"
    assert overlay & base == set(), f"the overlay reuses {sorted(overlay & base)}"
    assert overlay <= {STUB_PORT}, f"the overlay publishes {sorted(overlay)}"

    clashing = sorted((base | overlay) & set(HELPDESK_STACK_HOST_PORTS))
    assert clashing == [], (
        f"this stack publishes {clashing}, which the companion Helpdesk stack also publishes; "
        "whichever comes up second fails at `up`"
    )


# ---------------------------------------------------------------------------------------------
# The manifest, the Makefile, the workflows and the two READMEs


def test_the_manifest_names_the_three_options_in_order_against_the_stand_in_set() -> None:
    """Catches the
    example manifest copied across with its `helpdesk-golden-v1.jsonl` set name (the
    Helpdesk set does not exist yet), the options reordered so the table's rows do not match
    the README table, an absolute report path or one resolving outside the repository,
    and above all the self-hosted option keeping the example's `0.0100` placeholder - a
    fabricated hourly rate produces a fabricated USD per 1,000 turns, which is the one number
    the whole table exists to publish.

    The path claim is "resolves inside the repository", not "carries no `..`": the manifest
    is read from `project-evals/`, so the shipped
    `../docs/tradeoff-run/reports/<option>/report.json` is the correct spelling there and
    lands inside the tree. An absolute path, or one climbing out of it, still reddens."""
    document = json.loads(read(MANIFEST))
    options = document["options"]

    assert document["set_name"] == SET_NAME
    assert [option["id"] for option in options] == list(OPTION_IDS)
    for option in options:
        report = str(option["report"])
        assert not report.startswith("/"), report
        resolved = (MANIFEST.parent / report).resolve()
        assert resolved.is_relative_to(REPO_ROOT), report
        assert Path(report).name == "report.json", report
    self_hosted = [option for option in options if option["kind"] == "self_hosted"]
    assert len(self_hosted) == 1, [option["id"] for option in self_hosted]
    for option in self_hosted:
        assert str(option["hourly_usd"]) != PLACEHOLDER_HOURLY, f"{option['id']} keeps the example"
        assert NOTE_DATE.search(str(option["notes"])) is not None, option["notes"]


@pytest.mark.parametrize(("target", "subcommand"), MAKE_TARGETS)
def test_each_tradeoff_target_calls_the_script(target: str, subcommand: str) -> None:
    """Catches a target that only prints its usage (an `echo` naming the script is
    excluded), the subcommands swapped between targets, and `||` in a recipe,
    which reports success for a measurement run that failed."""
    lines = [
        line for line in recipe(read(MAKEFILE), target) if not line.lstrip("@-").startswith("echo")
    ]
    calling = [line for line in lines if f"infra/scripts/tradeoff_run.sh {subcommand}" in line]

    assert calling, f"{target}: {lines}"
    assert all("||" not in line for line in lines), f"{target}: {lines}"


def test_no_workflow_names_the_measurement_tooling() -> None:
    """Catches a dispatch workflow, or a step in any workflow,
    calling a target or the script directly: that needs a write-capable AWS role on the
    repository and an approval gate for an apply, which is kept off the repository."""
    assert WORKFLOWS.is_dir(), f"{WORKFLOWS} does not exist"
    files = sorted(path for path in WORKFLOWS.rglob("*") if path.is_file())
    assert files, f"{WORKFLOWS} holds no workflow"
    forbidden = (*(target for target, _subcommand in MAKE_TARGETS), "tradeoff_run.sh")
    naming = [
        f"{path.name}: {word}"
        for path in files
        for word in forbidden
        if word in path.read_text(encoding="utf-8")
    ]

    assert naming == [], f"workflows naming the measurement tooling: {naming}"


def test_the_run_readme_states_the_methodology_the_numbers_have_to_be_read_with() -> None:
    """Ten cases move `check_score` in ten-point steps and one
    sequential stream measures no concurrency, so a table published without those two sentences
    implies a precision it has not got. Catches a README that lists the commands and leaves the
    methodology out, and one whose cost table is missing entirely."""
    text = read(RUN_README)
    folded = text.casefold()
    missing = [word for word in (SET_NAME, "sequential") if word.casefold() not in folded]

    assert missing == [], f"the run README does not name {missing}"
    assert all(option_id in text for option_id in OPTION_IDS), text[:200]
    assert "|" in text, "the run README carries no cost table"


def test_readme_tradeoff_section_links_the_table_and_marks_the_conclusion_pending() -> None:
    """The heading and the link land with the tooling and the run replaces ONE line. Catches the
    section added with a conclusion already written (a claim about numbers nobody has measured),
    more than one pending line (the post-run edit then misses one and the README ships half
    pending), and a heading with no link to the generated table. The marker is matched
    case-insensitively: the section is published English, and it is written in sentence
    case, so the count of one is the claim and the casing is not."""
    body = readme_section(read(README), "tradeoff")
    pending = [line for line in body if PENDING_MARKER.search(line)]

    assert any("docs/tradeoff.md" in line for line in body), body
    assert len(pending) == 1, f"{len(pending)} pending lines in the tradeoff section: {pending}"


def test_each_api_label_names_the_model_its_overlay_routes_to() -> None:
    """The label of every `api` option in `tradeoff.json` names the model its overlay picks.

    Catches a label left on an old model after the overlay moved on, so the published table
    credits one model with another's numbers.
    """
    tradeoff_dir = Path(__file__).resolve().parents[3] / "infra" / "tradeoff"
    options = json.loads((tradeoff_dir / "tradeoff.json").read_text(encoding="utf-8"))["options"]
    api_options = [option for option in options if option["kind"] == "api"]
    assert api_options
    for option in api_options:
        overlay = (tradeoff_dir / f"{option['id']}.yaml").read_text(encoding="utf-8")
        found = re.search(r"model:\s*claude-([a-z]+)-(\d+)-(\d+)", overlay)
        assert found is not None, option["id"]
        family, major, minor = found.groups()
        assert f"{family.capitalize()} {major}.{minor}" in option["label"], option
