r"""`scripts/aws_evidence.sh` and `infra/k8s/aws`: the one-off AWS evidence run, before it runs.

The run itself is a human action against a real account, so what can be proven
here is the tooling around it, in two halves.

The script is EXECUTED, against fake `terraform`, `aws`, `kubectl` and `curl` binaries placed
first on `PATH` in `tmp_path`. Each fake appends `<tool> <argv joined by space>` to
`$FAKE_LOG`, exits 1 when `FAKE_FAIL_<TOOL>` is set, and prints the canned output the script
parses. `kubectl ... -f -` copies its stdin to `$FAKE_STDIN_DIR/kubectl-<N>.stdin`, which is
how the Secret is read without the script ever writing it anywhere. `python3` and `mktemp` stay
real. `AWS_EVIDENCE_TEST_ROOT` (a `_TEST_` name no application setting reads, a project rule)
sends the evidence under `tmp_path`, so no case writes into the repository. `HOME` is pointed
into `tmp_path` too, which is what lets a case see that `~/.kube` was never touched. `TMPDIR` is
NOT overridden: macOS `mktemp -d` ignores it, so the kubeconfig case pins what holds on both
platforms (a directory that existed at the call, is gone after exit, and is outside `HOME` and
the evidence tree) rather than a prefix. The happy-path run is made ONCE per module and read by
several cases; every script run is well under a second.

The overlay, the Makefile and the CI job are read as TEXT, because `kustomize` is not installed
on the development machine and CI's `k8s` job is the copy that builds. Assertions
on `kustomization.yaml` go through the PARSED document, never the raw text: its comments are
expected to describe the files it leaves out.

Every path is taken from this file's own `__file__`, and every read goes through `read()`, so a
file the plan has not written yet fails the one case that needs it instead of the collection.
`read`, `load_yaml`, `steps` and `recipe` come from `test_hetzner_module.py`, and `is_printer`
from `test_aws_module.py`, rather than being copied.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.infra.test_aws_module import is_printer
from tests.infra.test_hetzner_module import load_yaml, read, recipe, steps

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "aws_evidence.sh"
AWS_OVERLAY = REPO_ROOT / "infra" / "k8s" / "aws"
AWS_KUSTOMIZATION = AWS_OVERLAY / "kustomization.yaml"
EMPTYDIR_PATCH = AWS_OVERLAY / "emptydir-patch.yaml"
BACKUPS_SA = AWS_OVERLAY / "backups-sa.yaml"
IRSA_JOB = AWS_OVERLAY / "irsa-job.yaml"
APPS_KUSTOMIZATION = REPO_ROOT / "infra" / "k8s" / "apps" / "kustomization.yaml"
ROOT_KUSTOMIZATION = REPO_ROOT / "infra" / "k8s" / "kustomization.yaml"
MAKEFILE = REPO_ROOT / "Makefile"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "ci.yml"

#: A real-looking commit SHA: 40 lowercase hex, with letters so `.upper()` changes it.
VALID_SHA = "3f9c2a7e1b4d6f8a0c2e4b6d8f0a1c3e5b7d9f1a"

#: Section 1's six overlay resources, in the form kustomize reads them.
EXPECTED_RESOURCES = (
    "../base",
    "../platform/qdrant.yaml",
    "../apps/gateway.yaml",
    "../apps/llama.yaml",
    "../apps/llama-embed.yaml",
    "backups-sa.yaml",
)

#: The kind overlay's llama patch, reused by path and never copied (criterion 20).
KIND_LLAMA_PATCH = "../../../project-evals/ci/kind/llama-model-init-patch.yaml"

#: The captures section 1 names under `docs/aws-evidence/<UTC date>/`.
CAPTURES = ("terraform-output.json", "nodes.txt", "pods.txt", "smoke.json", "irsa.txt")

#: The captures whose fake source carries the account ID, so each must show the redaction.
REDACTED_CAPTURES = ("terraform-output.json", "nodes.txt", "pods.txt", "irsa.txt")

#: The account ID every fake output carries, and what the capture must say instead.
ACCOUNT_ID = "123456789012"
ACCOUNT_PLACEHOLDER = "ACCOUNT_ID"

#: The fake RDS-managed password. Must reach the Secret on stdin and nowhere else.
DB_PASSWORD = "FAKE-DB-PASSWORD-x9"

#: The ARN the fake `aws rds describe-db-instances` names for the master user secret.
RDS_SECRET_ARN = "arn:aws:secretsmanager:eu-central-1:123456789012:secret:rds!db-abc"

#: The role ARN the fake `aws iam get-role` prints; IRSA needs it on the service account.
BACKUPS_ROLE_ARN = "arn:aws:iam::123456789012:role/aioc-evidence-backups"

REDIS_URL = (
    "OPSCONSOLE_REDIS_URL=rediss://master.aioc-evidence.abc123.euc1.cache.amazonaws.com:6379/0"
)
GATEWAY_IMAGE = "ghcr.io/developermusa/ai-operations-console/gateway"

#: The literals the Secret on stdin carries (criteria 6 to 8).
SECRET_LITERALS = (
    "gateway-secrets",
    "OPSCONSOLE_DATABASE_URL",
    "sslmode=require",
    "OPSCONSOLE_API_KEYS",
    "OPSCONSOLE_ADMIN_KEY_SHA256",
    "OPSCONSOLE_BUDGET_OVERRIDE_SECRET",
    DB_PASSWORD,
)

#: The tenant key's digest as `OPSCONSOLE_API_KEYS` carries it.
TENANT_DIGEST = re.compile(r"evals:[0-9a-f]{64}")

#: Tracing on in any spelling a bash script accepts: `set -x`, `set -ex`, `set -o xtrace`.
XTRACE = re.compile(r"(?m)^\s*set\s+(-[a-z]*x|-o\s+xtrace)")

DATE_NAME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

#: The stale claim on the root kustomization (criterion 26), whitespace-normalised.
STALE_ROOT_CLAIM = "deploys this same kustomization unchanged"

#: One fake for all four tools, dispatched on its own name. a project rule's stub run
#: exercised every branch below against a correct script.
FAKE_TOOL = r"""#!/usr/bin/env bash
tool=$(basename "$0")
printf '%s %s\n' "$tool" "$*" >> "$FAKE_LOG"
fail_var="FAKE_FAIL_$(printf '%s' "$tool" | tr '[:lower:]' '[:upper:]')"
if [ -n "${!fail_var:-}" ]; then
  exit 1
fi
has() {
  local want=$1 arg
  shift
  for arg in "$@"; do
    [ "$arg" = "$want" ] && return 0
  done
  return 1
}
case "$tool" in
  terraform)
    if has output "$@"; then
      cat <<'JSON'
{"region":{"value":"eu-central-1"},"cluster_name":{"value":"aioc-evidence"},"rds_endpoint":{"value":"aioc-evidence-postgres.abc123.eu-central-1.rds.amazonaws.com:5432"},"redis_endpoint":{"value":"master.aioc-evidence.abc123.euc1.cache.amazonaws.com"},"backup_bucket":{"value":"aioc-evidence-backups-123456789012"},"cluster_endpoint":{"value":"https://ABC.gr7.eu-central-1.eks.amazonaws.com"}}
JSON
    fi
    ;;
  aws)
    case "$1 $2" in
      "rds describe-db-instances")
        printf '%s\n' 'arn:aws:secretsmanager:eu-central-1:123456789012:secret:rds!db-abc' ;;
      "secretsmanager get-secret-value")
        printf '%s\n' '{"username":"opsconsole","password":"FAKE-DB-PASSWORD-x9"}' ;;
      "iam get-role")
        printf '%s\n' 'arn:aws:iam::123456789012:role/aioc-evidence-backups' ;;
      "ce get-cost-and-usage")
        printf '%s\n' '{"ResultsByTime":[]}' ;;
      "eks update-kubeconfig")
        prev=""
        path=""
        for arg in "$@"; do
          [ "$prev" = "--kubeconfig" ] && path=$arg
          case "$arg" in --kubeconfig=*) path=${arg#--kubeconfig=} ;; esac
          prev=$arg
        done
        if [ -n "$path" ] && [ -d "$(dirname "$path")" ]; then
          : > "$path"
          printf '%s\n' "$path" >> "$FAKE_KUBECONFIG_SEEN"
        fi
        ;;
    esac
    ;;
  kubectl)
    prev=""
    for arg in "$@"; do
      if [ "$prev" = "-f" ] && [ "$arg" = "-" ]; then
        n=$(find "$FAKE_STDIN_DIR" -type f | wc -l | tr -d ' ')
        cat > "$FAKE_STDIN_DIR/kubectl-$n.stdin"
      fi
      prev=$arg
    done
    if has get "$@"; then
      printf '%s\n' 'node-1 Ready 123456789012'
    elif has logs "$@"; then
      printf '%s\n' 'arn:aws:sts::123456789012:assumed-role/aioc-evidence-backups/x'
    fi
    ;;
  curl)
    case " $* " in
      *"/readyz"*) printf '%s\n' '{"checks":{}}' ;;
      *"/v1/chat/completions"*) printf '%s\n' '{"choices":[{"message":{"content":"pong"}}]}' ;;
    esac
    ;;
esac
exit 0
"""

FAKE_NAMES = ("terraform", "aws", "kubectl", "curl")


@dataclass(frozen=True)
class Run:
    """One script invocation against the fakes, and where it could have left traces."""

    completed: subprocess.CompletedProcess[str]
    root: Path
    home: Path
    stdin_dir: Path
    log_path: Path
    seen_path: Path
    started: date
    finished: date

    def log(self) -> list[str]:
        """The fake call log, one line per tool call; empty when nothing was called."""
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def stdins(self) -> list[str]:
        """Every stdin a `kubectl ... -f -` call received."""
        return [path.read_text(encoding="utf-8") for path in sorted(self.stdin_dir.iterdir())]

    def seen_kubeconfigs(self) -> list[str]:
        """Every kubeconfig path the fake `aws` found a directory for at call time."""
        if not self.seen_path.exists():
            return []
        return self.seen_path.read_text(encoding="utf-8").splitlines()

    def evidence(self) -> Path:
        """Where the script writes under the test root."""
        return self.root / "docs" / "aws-evidence"


def invoke(base: Path, arguments: Sequence[str], /, **switches: str) -> Run:
    """Run the script under `base` with the four fakes first on `PATH` and `switches` in env."""
    base.mkdir(parents=True, exist_ok=True)
    fakes = base / "bin"
    fakes.mkdir()
    for name in FAKE_NAMES:
        fake = fakes / name
        fake.write_text(FAKE_TOOL, encoding="utf-8")
        fake.chmod(0o755)
    root, home, stdin_dir = (base / name for name in ("root", "home", "stdin"))
    for directory in (root, home, stdin_dir):
        directory.mkdir()
    log_path, seen_path = base / "calls.log", base / "kubeconfigs.seen"
    environment = {
        "PATH": f"{fakes}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(home),
        "AWS_EVIDENCE_TEST_ROOT": str(root),
        "FAKE_LOG": str(log_path),
        "FAKE_STDIN_DIR": str(stdin_dir),
        "FAKE_KUBECONFIG_SEEN": str(seen_path),
        **switches,
    }
    assert SCRIPT.is_file(), f"{SCRIPT} does not exist"
    started = datetime.now(UTC).date()
    completed = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return Run(
        completed=completed,
        root=root,
        home=home,
        stdin_dir=stdin_dir,
        log_path=log_path,
        seen_path=seen_path,
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
    """`words` appear consecutively anywhere in `tokens`: `kubectl -n apps set env` counts."""
    width = len(words)
    return any(tokens[i : i + width] == list(words) for i in range(len(tokens) - width + 1))


def option(tokens: list[str], name: str) -> str | None:
    """The value after `name` in `tokens`, or after `name=`; None when absent."""
    for position, token in enumerate(tokens):
        if token == name and position + 1 < len(tokens):
            return tokens[position + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


def terraform_calls(run: Run, verb: str) -> list[tuple[int, list[str]]]:
    """Terraform calls running `verb` against the AWS module directory."""
    return [
        (index, tokens)
        for index, tokens in calls(run, "terraform")
        if verb in tokens
        and any(
            token.startswith("-chdir=") and token.rstrip("/").endswith("infra/terraform/aws")
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
    assert len(dated) == 1, f"{len(dated)} dated evidence directories: {dated}"
    allowed = {run.started.isoformat(), run.finished.isoformat()}
    assert dated[0].name in allowed, f"{dated[0].name} is not today's UTC date {sorted(allowed)}"
    return dated[0]


def documents(path: Path) -> list[dict[str, Any]]:
    """Every mapping document in a multi-document YAML file."""
    return [doc for doc in yaml.safe_load_all(read(path)) if isinstance(doc, dict) and doc]


def identity(document: dict[str, Any]) -> tuple[str, str, str]:
    """`(kind, name, namespace)` of a manifest."""
    metadata = document.get("metadata") or {}
    return (str(document.get("kind")), str(metadata.get("name")), str(metadata.get("namespace")))


def the_one(path: Path, kind: str, name: str) -> dict[str, Any]:
    """The single document of `kind` named `name` in `path`."""
    found = [doc for doc in documents(path) if identity(doc)[:2] == (kind, name)]
    assert len(found) == 1, f"{path.name} holds {len(found)} {kind} {name}"
    return found[0]


def volume(document: dict[str, Any], name: str) -> dict[str, Any]:
    """The pod template's volume `name`."""
    volumes = document.get("spec", {}).get("template", {}).get("spec", {}).get("volumes") or []
    matched = [entry for entry in volumes if isinstance(entry, dict) and entry.get("name") == name]
    assert len(matched) == 1, f"{identity(document)} has {len(matched)} volumes named {name}"
    return matched[0]


def image_entry(path: Path, name: str) -> dict[str, Any]:
    """The `images:` entry `name` of a kustomization."""
    images = load_yaml(path).get("images") or []
    matched = [entry for entry in images if isinstance(entry, dict) and entry.get("name") == name]
    assert len(matched) == 1, f"{path} has {len(matched)} image entries named {name}"
    return matched[0]


def pipeline_tail(line: str) -> str:
    """Everything after the first `| ` of a recipe line."""
    assert "| " in line, f"no pipe in {line}"
    return line.split("| ", 1)[1]


@pytest.fixture(scope="module")
def happy(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """`run <VALID_SHA>` against fakes that all succeed, made ONCE for the module."""
    return invoke(tmp_path_factory.mktemp("happy"), ["run", VALID_SHA])


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
def test_run_refuses_a_malformed_sha_before_any_tool(tmp_path: Path, sha: str) -> None:
    """Criteria 1 and 2, the refused rows; the accepted 40-lowercase row is the happy path.
    Catches a check written as `grep -Eq '^[0-9a-f]{40}$'`, which accepts the trailing-newline
    row because grep matches per line, a `{39,41}` or `[0-9a-fA-F]` class, and a check placed
    after `terraform apply` - every fake succeeds, so without the check the run completes."""
    run = invoke(tmp_path, ["run", sha])

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
    """Criteria 14 and 15. Catches a date check on the start only (the short-day end passes),
    a grep-per-line check (the newline row), `-le` in place of a strict comparison (the equal
    row, a zero-day window Cost Explorer rejects), and no ordering check at all (the reversed
    row). The fake `aws` succeeds, so without the check `cost` would call it and exit 0."""
    run = invoke(tmp_path, ["cost", start, end])

    assert run.completed.returncode != 0, f"{start!r} {end!r} was accepted"
    assert run.log() == [], f"{start!r} {end!r} reached a tool: {run.log()}"


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param([], id="no-subcommand"),
        pytest.param(["deploy", VALID_SHA], id="unknown-subcommand"),
        pytest.param(["run"], id="run-without-sha"),
        pytest.param(["cost", "2026-09-01"], id="cost-without-end"),
    ],
)
def test_a_missing_or_unknown_subcommand_is_refused(tmp_path: Path, arguments: list[str]) -> None:
    """Criterion 1's input validation for the argument shape. Catches a dispatcher whose default
    branch falls through to `run`, and a missing positional expanding to an empty string that a
    later `terraform apply` never sees refused."""
    run = invoke(tmp_path, arguments)

    assert run.completed.returncode != 0, f"{arguments} was accepted"
    assert run.log() == [], f"{arguments} reached a tool: {run.log()}"


# ---------------------------------------------------------------------------------------------
# The run


def test_run_applies_connects_deploys_and_destroys_in_that_order(happy: Run) -> None:
    """Criterion 3. Catches `apply -auto-approve` (the owner never reviews the plan), the
    kubeconfig fetched before `output -json` has named the cluster, a `kubectl` call placed
    before the kubeconfig exists, and destroy run before capture finishes or twice (an explicit
    destroy plus the trap), which shows as a destroy that is not the single last call."""
    succeeded(happy)
    log = happy.log()
    applies = terraform_calls(happy, "apply")
    outputs = [(i, t) for i, t in terraform_calls(happy, "output") if "-json" in t]
    connects = [(i, t) for i, t in calls(happy, "aws") if runs(t, "eks", "update-kubeconfig")]
    kubectl = calls(happy, "kubectl")
    destroys = [(i, t) for i, t in terraform_calls(happy, "destroy") if "-auto-approve" in t]

    assert len(applies) == 1, f"{len(applies)} terraform apply calls: {log}"
    assert "-auto-approve" not in applies[0][1], "apply does not wait for the owner"
    assert outputs, f"no terraform output -json: {log}"
    assert len(connects) == 1, f"{len(connects)} update-kubeconfig calls: {log}"
    assert option(connects[0][1], "--region") == "eu-central-1"
    assert option(connects[0][1], "--name") == "aioc-evidence"
    assert kubectl, f"no kubectl call: {log}"
    assert len(destroys) == 1, f"{len(destroys)} destroy -auto-approve calls: {log}"

    assert applies[0][0] < outputs[0][0] < connects[0][0] < kubectl[0][0]
    assert kubectl[-1][0] < destroys[0][0]
    assert destroys[0][0] == len(log) - 1, f"a call after destroy: {log[destroys[0][0] :]}"


def test_a_failing_kubectl_still_destroys_the_cluster(tmp_path: Path) -> None:
    """Criterion 4. Catches a script with `set -e` and a destroy at the end but no EXIT trap:
    the first failing `kubectl` exits it and leaves EKS, RDS and ElastiCache billing. The fakes
    for terraform and aws succeed, so the only thing that can stop destroy is the missing trap."""
    run = invoke(tmp_path, ["run", VALID_SHA], FAKE_FAIL_KUBECTL="1")
    applies = terraform_calls(run, "apply")
    destroys = [(i, t) for i, t in terraform_calls(run, "destroy") if "-auto-approve" in t]

    assert run.completed.returncode != 0, "a failed kubectl reported success"
    assert len(applies) == 1, f"no apply before the failure: {run.log()}"
    assert destroys, f"no destroy after the failure: {run.log()}"
    assert destroys[-1][0] > applies[0][0]


def test_the_kubeconfig_lives_in_a_temp_dir_removed_on_exit(happy: Run) -> None:
    """Criterion 5. Catches `--kubeconfig ~/.kube/aioc` or no `--kubeconfig` at all (the
    owner's default context rewritten to a cluster about to be destroyed), a kubeconfig written
    into the evidence folder that is about to be committed, a path whose directory was never
    created, and a temp directory the EXIT trap never removes."""
    succeeded(happy)
    connects = [t for _i, t in calls(happy, "aws") if runs(t, "eks", "update-kubeconfig")]
    assert len(connects) == 1, f"{len(connects)} update-kubeconfig calls"
    path = option(connects[0], "--kubeconfig")
    assert path is not None, "update-kubeconfig writes the default kubeconfig"
    kubeconfig = Path(path)

    assert kubeconfig.is_absolute(), f"{path} is relative to wherever the owner stands"
    assert happy.seen_kubeconfigs() == [path], "the kubeconfig directory did not exist at call"
    assert not kubeconfig.parent.exists(), f"{kubeconfig.parent} survived the run"
    assert ".kube" not in kubeconfig.parts, f"{path} is under a .kube directory"
    assert not kubeconfig.is_relative_to(happy.home), f"{path} is under HOME"
    assert not kubeconfig.is_relative_to(happy.root), f"{path} is in the evidence tree"
    assert not (happy.home / ".kube").exists(), "the run touched ~/.kube"


def test_the_secret_is_built_from_the_rds_secret_and_piped_on_stdin(happy: Run) -> None:
    """Criteria 6 to 8. Catches the password fetched from a guessed secret name instead of the
    ARN the RDS instance names, a `DATABASE_URL` without `sslmode=require` (RDS then refuses or
    downgrades), a Secret missing one of the four keys, the raw tenant key where its 64-hex
    digest belongs, and a Secret written to a file and applied with `-f <path>`."""
    succeeded(happy)
    fetches = [
        " ".join(t)
        for _i, t in calls(happy, "aws")
        if runs(t, "secretsmanager", "get-secret-value")
    ]
    secrets = [text for text in happy.stdins() if "gateway-secrets" in text]

    assert len(fetches) == 1, f"{len(fetches)} get-secret-value calls"
    assert RDS_SECRET_ARN in fetches[0], f"the secret fetched is not the RDS one: {fetches[0]}"
    assert len(secrets) == 1, f"{len(secrets)} kubectl stdins carry gateway-secrets"
    missing = [literal for literal in SECRET_LITERALS if literal not in secrets[0]]
    assert missing == [], f"the Secret lacks {missing}"
    assert TENANT_DIGEST.search(secrets[0]) is not None, "no evals:<sha256> in the Secret"


def test_each_run_generates_a_fresh_tenant_key(happy: Run, tmp_path: Path) -> None:
    """Criterion 8's "fresh per run". Catches a key hard-coded in the script or derived from the
    SHA, which makes every evidence run share one credential that is now in git history."""
    succeeded(happy)
    second = invoke(tmp_path, ["run", VALID_SHA])
    succeeded(second)

    def digest(run: Run) -> str:
        found = {m.group(0) for text in run.stdins() for m in TENANT_DIGEST.finditer(text)}
        assert len(found) == 1, f"{len(found)} tenant digests in one run: {found}"
        return found.pop()

    assert digest(happy) != digest(second)


def test_the_gateway_is_pointed_at_elasticache_over_tls_and_the_sha_image(happy: Run) -> None:
    """Criterion 9. Catches `redis://` (ElastiCache has transit encryption on and refuses it),
    the endpoint without `:6379/0`, an image tagged with a short SHA or `latest`, and an image
    path other than the one `build.yml` pushes."""
    succeeded(happy)
    kubectl = [t for _i, t in calls(happy, "kubectl")]
    set_env = [t for t in kubectl if runs(t, "set", "env")]
    set_image = [t for t in kubectl if runs(t, "set", "image")]
    image = re.compile(rf"(?:^|=){re.escape(GATEWAY_IMAGE)}:{VALID_SHA}$")

    assert any(REDIS_URL in t for t in set_env), f"no {REDIS_URL} in {set_env}"
    assert any(image.search(token) for t in set_image for token in t), f"image: {set_image}"


def test_the_backups_service_account_carries_the_role_arn_and_the_job_runs(happy: Run) -> None:
    """Criterion 11. Catches the annotation missing or naming a different key, so the IRSA Job
    runs with the node role and the evidence proves nothing, and the overlay or the Job never
    being handed to `kubectl` at all."""
    succeeded(happy)
    kubectl = [" ".join(t) for _i, t in calls(happy, "kubectl")]
    annotation = f"eks.amazonaws.com/role-arn={BACKUPS_ROLE_ARN}"

    assert any("annotate" in line and annotation in line for line in kubectl), kubectl
    assert any("infra/k8s/aws" in line for line in kubectl), "the overlay is never applied"
    assert any("irsa" in line for line in kubectl), "the IRSA Job is never run"


def test_capture_writes_the_five_files_under_todays_utc_date(happy: Run) -> None:
    """Criterion 12. Catches a local-time directory name, a date format other than
    `YYYY-MM-DD`, evidence written under the repository instead of the test root, and any of
    the five captures skipped."""
    succeeded(happy)
    directory = dated_dir(happy)
    missing = [name for name in CAPTURES if not (directory / name).is_file()]

    assert missing == [], f"{directory} lacks {missing}"


def test_every_captured_account_id_is_redacted(happy: Run) -> None:
    """Criterion 13. Catches captures written straight from the command, where the fake account
    ID reaches a file meant for a public repository, and a redaction that empties the file
    instead of substituting, which the placeholder check below makes visible."""
    succeeded(happy)
    directory = dated_dir(happy)
    leaking = [
        str(path.relative_to(directory))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ACCOUNT_ID in path.read_text(encoding="utf-8", errors="replace")
    ]
    unredacted = [
        name for name in REDACTED_CAPTURES if ACCOUNT_PLACEHOLDER not in read(directory / name)
    ]

    assert leaking == [], f"the account ID is in {leaking}"
    assert unredacted == [], f"no {ACCOUNT_PLACEHOLDER} in {unredacted}"


def test_the_database_password_never_leaves_stdin(happy: Run) -> None:
    """Criterion 7's "never stdout". Catches `echo "$DATABASE_URL"` for debugging, the secret
    JSON captured as evidence, and a Secret manifest written into the evidence folder. The
    secret case above proves the password did reach the Secret, so this scan is not vacuous."""
    succeeded(happy)
    written = [
        str(path.relative_to(happy.root))
        for path in sorted(happy.root.rglob("*"))
        if path.is_file() and DB_PASSWORD in path.read_text(encoding="utf-8", errors="replace")
    ]

    assert DB_PASSWORD not in happy.completed.stdout, "the password is on stdout"
    assert DB_PASSWORD not in happy.completed.stderr, "the password is on stderr"
    assert written == [], f"the password is in {written}"


def test_cost_queries_the_whole_account_daily_by_service(tmp_path: Path) -> None:
    """Criteria 15 and 16. Catches a `--filter` on the project tag (decision F: the tag is not
    active before the run, so the filter misses it), `MONTHLY` granularity, `BlendedCost`, no
    `SERVICE` group-by, a one-day window refused, and `cost` touching terraform or kubectl."""
    run = invoke(tmp_path, ["cost", "2026-09-01", "2026-09-02"])
    succeeded(run)
    queries = [t for _i, t in calls(run, "aws") if runs(t, "ce", "get-cost-and-usage")]
    cost = run.evidence() / "cost.json"

    assert len(queries) == 1, f"{len(queries)} get-cost-and-usage calls: {run.log()}"
    assert option(queries[0], "--granularity") == "DAILY"
    assert option(queries[0], "--metrics") == "UnblendedCost"
    assert "Key=SERVICE" in " ".join(queries[0])
    assert option(queries[0], "--filter") is None, "cost is filtered"
    assert calls(run, "terraform") == [] and calls(run, "kubectl") == [], run.log()
    assert json.loads(read(cost)) == {"ResultsByTime": []}


# ---------------------------------------------------------------------------------------------
# The script as text


def test_the_script_is_executable_never_traces_and_never_names_the_default_kubeconfig() -> None:
    """Criteria 5 and 7. Catches `set -x` or `set -euxo pipefail` (every expanded command, the
    database URL included, printed to the owner's terminal and scrollback), `.kube/config`
    named anywhere, and the file committed without its execute bit, where `make aws-evidence`
    fails with permission denied."""
    text = read(SCRIPT)

    assert XTRACE.search(text) is None, "the script turns on tracing"
    assert ".kube/config" not in text
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable"


# ---------------------------------------------------------------------------------------------
# The overlay


def test_the_overlay_holds_exactly_the_six_resources() -> None:
    """Criteria 17 and 18, on the parsed list. Catches `../apps` whole, which
    brings `ingress.yaml`'s Traefik objects and `helpdesk.yaml`'s unpublished images, `../platform`
    whole, which brings in-cluster postgres and redis and the cert-manager issuers, a duplicate
    entry, and the backups ServiceAccount left out."""
    resources = load_yaml(AWS_KUSTOMIZATION).get("resources")
    assert isinstance(resources, list), "the overlay has no resources list"

    assert len(resources) == len(EXPECTED_RESOURCES), resources
    assert set(resources) == set(EXPECTED_RESOURCES), resources


def test_the_overlay_reuses_the_kind_llama_patch_by_path() -> None:
    """Criterion 20. Catches a copy of the kind patch under `infra/k8s/aws/`, which drifts the
    first time the kind one is fixed, the reference aimed at the wrong Deployment, and the
    emptyDir patch not wired into the overlay at all."""
    patches = load_yaml(AWS_KUSTOMIZATION).get("patches")
    assert isinstance(patches, list), "the overlay has no patches list"
    by_path = {str(entry.get("path")): entry for entry in patches if isinstance(entry, dict)}
    assert AWS_OVERLAY.is_dir(), f"{AWS_OVERLAY} does not exist"
    copies = [path for path in AWS_OVERLAY.rglob("*") if path.name == Path(KIND_LLAMA_PATCH).name]

    assert KIND_LLAMA_PATCH in by_path, f"patches: {sorted(by_path)}"
    target = by_path[KIND_LLAMA_PATCH].get("target") or {}
    assert (target.get("kind"), target.get("name")) == ("Deployment", "llama"), target
    assert "emptydir-patch.yaml" in by_path, f"patches: {sorted(by_path)}"
    assert copies == [], f"a copy of the kind patch: {copies}"


def test_the_overlay_deletes_exactly_the_two_llama_claims_by_inline_patch() -> None:
    """Criterion 19's deletions. Catches a deleted claim left out (EKS has no EBS CSI driver, so
    it stays Pending and the pod never schedules), a delete aimed at a claim in the wrong
    namespace, and an extra `$patch: delete` removing an object the run needs. The deletes are
    inline `patch:` entries, never in `emptydir-patch.yaml`: kustomize 5.5.0 (CI's pin) panics
    on a patch file where any document follows a delete, so a delete there is also refused."""
    patches = load_yaml(AWS_KUSTOMIZATION).get("patches")
    assert isinstance(patches, list), "the overlay has no patches list"
    inline = [
        doc
        for entry in patches
        if isinstance(entry, dict) and isinstance(entry.get("patch"), str)
        for doc in yaml.safe_load_all(entry["patch"])
        if isinstance(doc, dict)
    ]
    deleted = {identity(doc) for doc in inline if doc.get("$patch") == "delete"}
    in_file = [identity(doc) for doc in documents(EMPTYDIR_PATCH) if "$patch" in doc]

    assert deleted == {
        ("PersistentVolumeClaim", "llama-models", "apps"),
        ("PersistentVolumeClaim", "llama-embed-models", "apps"),
    }
    assert in_file == [], f"emptydir-patch.yaml carries a $patch directive: {in_file}"


def test_the_emptydir_patch_drops_qdrants_claim_template_for_an_emptydir() -> None:
    """Criterion 19 for qdrant. Catches the StatefulSet keeping `volumeClaimTemplates` (a PVC
    with no provisioner, so qdrant stays Pending), and the template removed without a `storage`
    emptyDir in its place, where the container's mount names a volume that does not exist."""
    qdrant = the_one(EMPTYDIR_PATCH, "StatefulSet", "qdrant")
    spec = qdrant.get("spec") or {}

    assert identity(qdrant)[2] == "platform"
    assert "volumeClaimTemplates" in spec and spec["volumeClaimTemplates"] is None
    assert "emptyDir" in volume(qdrant, "storage")


@pytest.mark.parametrize("name", ["llama", "llama-embed"])
def test_the_emptydir_patch_swaps_each_llama_models_volume(name: str) -> None:
    """Criterion 19 for the two llama Deployments. Catches `emptyDir` added beside the claim
    without nulling `persistentVolumeClaim` (a strategic merge keeps both, which the API
    rejects as two volume sources), and the patch naming a volume other than `models`."""
    deployment = the_one(EMPTYDIR_PATCH, "Deployment", name)
    models = volume(deployment, "models")

    assert identity(deployment)[2] == "apps"
    assert "emptyDir" in models
    assert "persistentVolumeClaim" in models and models["persistentVolumeClaim"] is None


def test_the_overlay_pins_the_same_llama_image_as_the_apps_kustomization() -> None:
    """Criterion 21. Catches the overlay floating on `server` or pinning a different build than
    the k3s cluster runs, so the evidence is for a server nobody deploys, and an entry that is
    missing so the bare `llama` name is pulled from Docker Hub."""
    aws = image_entry(AWS_KUSTOMIZATION, "llama")
    apps = image_entry(APPS_KUSTOMIZATION, "llama")

    assert aws.get("newName") == "ghcr.io/ggml-org/llama.cpp"
    assert str(apps.get("newTag") or "") != "", "the apps llama entry has no tag"
    assert aws.get("newTag") == apps.get("newTag")


def test_the_backups_service_account_is_platform_backups() -> None:
    """Criterion 17's ServiceAccount and decision G. Catches it in `apps` or named `backup`,
    where 26a's role trust names `system:serviceaccount:platform:backups` and the IRSA Job's
    token is refused."""
    identities = {identity(doc) for doc in documents(BACKUPS_SA)}

    assert ("ServiceAccount", "backups", "platform") in identities, identities


def test_the_irsa_job_runs_as_backups_on_a_pinned_cli_image() -> None:
    """Criterion 11 and section 6's pin. Catches `amazon/aws-cli:latest` or an untagged image,
    a Job with no `serviceAccountName` (it then runs as `default` with no role), and the Job in
    a namespace other than the service account's."""
    job = the_one(IRSA_JOB, "Job", "irsa-evidence")
    pod = job.get("spec", {}).get("template", {}).get("spec", {})
    images = [str(container.get("image")) for container in pod.get("containers") or []]

    assert identity(job)[2] == "platform"
    assert pod.get("serviceAccountName") == "backups"
    assert images, "the Job has no container"
    unpinned = [
        image
        for image in images
        if not re.fullmatch(r"amazon/aws-cli:[0-9]+\.[0-9]+\.[0-9]+", image)
    ]
    assert unpinned == [], f"unpinned images: {unpinned}"


def test_the_root_kustomization_no_longer_claims_aws_deploys_it_unchanged() -> None:
    """Criterion 26 and decision B. Catches the comment left as it was, now false: the AWS run
    deploys a narrowed overlay. Whitespace is normalised, so a reflowed line cannot hide it."""
    text = " ".join(read(ROOT_KUSTOMIZATION).split())

    assert STALE_ROOT_CLAIM not in text


# ---------------------------------------------------------------------------------------------
# Makefile and CI


def test_k8s_aws_check_builds_the_overlay_with_k8s_checks_own_flags() -> None:
    """Criterion 22. Catches the build without `--load-restrictor=LoadRestrictionsNone` (the
    kind patch's `../../../` path is then refused), `|| true` swallowing a failure, a second
    line so only the last exit code counts, and kubeconform flags that drift from `k8s-check`,
    `-strict` or a schema location dropped."""
    makefile = read(MAKEFILE)
    aws = recipe(makefile, "k8s-aws-check")
    base = recipe(makefile, "k8s-check")
    assert len(aws) == 1, f"k8s-aws-check has {len(aws)} recipe lines: {aws}"
    assert len(base) == 1, f"k8s-check has {len(base)} recipe lines: {base}"

    assert (
        "kustomize build --load-restrictor=LoadRestrictionsNone infra/k8s/aws | kubeconform"
        in (aws[0])
    )
    assert "||" not in aws[0]
    assert pipeline_tail(aws[0]) == pipeline_tail(base[0])


@pytest.mark.parametrize(
    ("target", "command"),
    [
        ("aws-evidence", "scripts/aws_evidence.sh run"),
        ("aws-evidence-cost", "scripts/aws_evidence.sh cost"),
    ],
)
def test_the_evidence_targets_call_the_script(target: str, command: str) -> None:
    """Criterion 24. Catches a target that only prints its usage (an `echo` naming the script
    is excluded, a project rule), and the two subcommands swapped between the targets."""
    lines = [
        line for line in recipe(read(MAKEFILE), target) if not line.lstrip("@-").startswith("echo")
    ]

    assert any(command in line for line in lines), f"{target}: {lines}"


def test_the_k8s_job_runs_k8s_aws_check_once_unconditionally() -> None:
    """Criterion 23. Catches the step missing, so the overlay's kustomize build is proven
    nowhere, the step given an `if` or `continue-on-error` that makes it inert, and a second
    copy. An `echo` step naming the command is excluded."""
    jobs = load_yaml(WORKFLOW)["jobs"]
    assert isinstance(jobs, dict) and "k8s" in jobs, "ci.yml declares no k8s job"
    running = [
        step
        for step in steps(jobs["k8s"])
        if "make k8s-aws-check" in str(step.get("run", "")) and not is_printer(step)
    ]

    assert len(running) == 1, f"{len(running)} steps run make k8s-aws-check"
    assert "if" not in running[0]
    assert "continue-on-error" not in running[0]


def test_no_workflow_runs_the_evidence_tooling() -> None:
    """Criterion 24 and decision A. Catches an `aws-evidence.yml` dispatch workflow, or a step in
    any workflow, calling either target or the script directly: that needs a write-capable AWS
    role on the repository, which a project rule keeps off it."""
    assert WORKFLOWS.is_dir(), f"{WORKFLOWS} does not exist"
    files = sorted(path for path in WORKFLOWS.rglob("*") if path.is_file())
    assert files, f"{WORKFLOWS} holds no workflow"
    naming = [
        path.name
        for path in files
        if "aws-evidence" in (text := path.read_text(encoding="utf-8")) or "aws_evidence" in text
    ]

    assert naming == [], f"workflows naming the evidence tooling: {naming}"
