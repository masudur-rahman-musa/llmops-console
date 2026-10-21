#!/usr/bin/env bash
# The self-hosted versus API measurement run, started by the owner from their own shell.
#
#   infra/scripts/tradeoff_run.sh option <option id> --target <eval target url>
#     build that option's effective routing policy, bring the measurement harness up
#     against it, flush the response cache, migrate, play the conversation cases, tear the
#     harness down. The option's two report files land under
#     docs/tradeoff-run/reports/<option id>/.
#
#   infra/scripts/tradeoff_run.sh gpu <commit sha>
#     terraform apply on infra/terraform/aws-gpu (the owner reads the plan and types yes),
#     fetch the cluster's kubeconfig over SSH, deploy infra/k8s/gpu, wait for the model
#     and the rollout, measure the gpu-vllm option, capture the evidence under
#     docs/tradeoff-run/<UTC date>/, then terraform destroy.
#
#   infra/scripts/tradeoff_run.sh cost <start YYYY-MM-DD> <end YYYY-MM-DD>
#     write cost.json (in docs/tradeoff-run) from Cost Explorer, once the spend has landed
#     (about a day after the run). End is exclusive, as Cost Explorer reads it.
#
# Once apply has started, the EXIT trap tears the harness down and destroys the stack
# whatever happens next: a failed step, Ctrl-C, or a clean finish. A g5.xlarge left
# running bills around 1.00 USD an hour. If destroy itself fails, the trap says so
# loudly and exits non-zero, and `terraform -chdir=infra/terraform/aws-gpu destroy` has
# to be run by hand. The module's 240-minute self-shutdown and its 15 USD budget are the
# other two guards.
#
# Credentials never touch disk or the terminal. The tenant key is generated per run,
# handed to the gateway as a digest and to the stub as itself, both through the
# environment, and written nowhere (the `make load` shape). The cluster's kubeconfig
# lives in a mktemp directory removed on exit, never the owner's default one. Tracing is
# never switched on here: it would print every expanded command, the key among them.
#
# Captured files pass through `redact`, which replaces every standalone 12-digit number
# (an AWS account ID) with ACCOUNT_ID, since the evidence is committed.
#
# The cluster is reached through an SSH local forward, which is all the module's security
# group opens: 6443 for the Kubernetes API and 30800 for the vLLM NodePort the harness
# dials as `host.docker.internal`. The forward adds the owner's round trip to every
# measured GPU turn; `docs/tradeoff-run/README.md` says so beside the numbers, because it
# inflates that option's latency and therefore its cost per 1,000 turns rather than
# flattering it.
#
# Pinned by infra/tests/infra/test_tradeoff_run.py, which runs this file against fake terraform,
# aws, kubectl, ssh, docker, openssl, uv and curl binaries. TRADEOFF_TEST_ROOT moves the
# evidence folder for those tests and is read by nothing else.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODULE_DIR="$REPO_ROOT/infra/terraform/aws-gpu"
GPU_OVERLAY="$REPO_ROOT/infra/k8s/gpu"
POLICY_BUILDER="$REPO_ROOT/infra/scripts/tradeoff_policy.py"
EVIDENCE_ROOT="${TRADEOFF_TEST_ROOT:-$REPO_ROOT}/docs/tradeoff-run"

COMPOSE=(docker compose -f "$REPO_ROOT/docker-compose.yml" -f "$REPO_ROOT/infra/tradeoff/compose.yml" --profile tradeoff --profile local-model)

# The small conversation file this repo ships, played against the stub target. The help
# desk's own 33 cases live in the companion repository, so waiting for it means no table.
CASE_SET="backend/evals/kind-cases.jsonl"
TENANT="evals"

# The option the `gpu` subcommand measures, and the one overlay that enables the vllm
# provider. Never an argument: a GPU instance applied and destroyed to measure an API
# provider is an expensive way to publish a wrong row.
GPU_OPTION="gpu-vllm"

# The stub inside the harness, on the port `infra/tradeoff/compose.yml` publishes.
GPU_TARGET="http://localhost:8090"

# Bash `=~` is POSIX ERE over the whole string: `$` does not match before a trailing
# newline, so `sha\n` is refused.
OPTION_ID_PATTERN='^[a-z][a-z0-9-]{0,31}$'
SHA_PATTERN='^[0-9a-f]{40}$'
DATE_PATTERN='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'

# The Kubernetes API and the vLLM NodePort, forwarded over the one port the security
# group opens. 30800 is `infra/k8s/gpu/vllm.yaml`'s `vllm-node` Service, written out
# again here because a test compares the two.
API_PORT=6443
NODE_PORT=30800

# Cost Explorer is a global API served from us-east-1 only.
COST_EXPLORER_REGION="us-east-1"

usage() {
  printf 'usage: %s option <option id> --target <url>\n       %s gpu <40 lowercase hex sha>\n       %s cost <YYYY-MM-DD start> <YYYY-MM-DD end>\n' "$0" "$0" "$0" >&2
  exit 2
}

die() {
  printf 'tradeoff-run: %s\n' "$*" >&2
  exit 1
}

redact() {
  python3 -c 'import re, sys; sys.stdout.write(re.sub(r"(?<![0-9])[0-9]{12}(?![0-9])", "ACCOUNT_ID", sys.stdin.read()))'
}

# The tenant key's digest, computed here rather than with `openssl dgst` so the key stays
# on a pipe and out of every argument list.
sha256_of_stdin() {
  python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
}

# State the EXIT trap reads.
workdir=""
applied=""
harness=""
forward=""

finish() {
  local status=$?
  if [ -n "$harness" ]; then
    "${COMPOSE[@]}" down --volumes --remove-orphans || status=1
  fi
  if [ -n "$forward" ]; then
    # The forward may already have exited; there is nothing to stop then.
    kill "$forward" 2>/dev/null || true
  fi
  if [ -n "$applied" ]; then
    printf 'tradeoff-run: destroying the instance\n' >&2
    if ! terraform -chdir="$MODULE_DIR" destroy -auto-approve; then
      printf 'tradeoff-run: DESTROY FAILED, the instance is still billing. Run: terraform -chdir=%s destroy\n' "$MODULE_DIR" >&2
      status=1
    fi
  fi
  if [ -n "$workdir" ]; then
    rm -rf -- "$workdir"
  fi
  exit "$status"
}

output_value() {
  printf '%s' "$outputs" | python3 -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]]["value"])' "$1"
}

# One option measured end to end. Answers the eval runner's own exit code: a run with
# failed cases still wrote its two report files and still counts as measured, and only a
# harness that never answered should read as a broken tool.
run_option() {
  local option_id=$1 target=$2
  [[ $option_id =~ $OPTION_ID_PATTERN ]] || die "the option id must match ${OPTION_ID_PATTERN}"

  local reports_dir="$EVIDENCE_ROOT/reports/$option_id"
  mkdir -p "$reports_dir"

  # The effective policy goes beside the kubeconfig, in a directory removed on exit: it is
  # derived from two committed files and is evidence of nothing on its own.
  [ -n "$workdir" ] || workdir="$(mktemp -d)"
  local policy_file="$workdir/$option_id.yaml"
  python3 "$POLICY_BUILDER" "$option_id" "$policy_file"

  local key digest
  key="$(openssl rand -hex 32)"
  digest="$(printf '%s' "$key" | sha256_of_stdin)"

  export TRADEOFF_POLICY_FILE="$policy_file"
  export OPSCONSOLE_API_KEYS="$TENANT:$digest"
  export GATEWAY_API_KEY="$key"

  harness=yes
  trap finish EXIT
  trap 'exit 130' INT TERM

  "${COMPOSE[@]}" up -d --build --wait gateway eval-target-stub

  # The semantic tier is not keyed by the resolved model, so the second option measured
  # would read the first's answers and report a cost no provider charged and a latency
  # none served. The exact tier is safe and flushed with it anyway.
  "${COMPOSE[@]}" exec -T redis sh -c 'redis-cli --scan --pattern "response:*" | xargs -r redis-cli del'

  # Every served turn writes a ledger row, and the cost column is read from those rows.
  uv run ops-migrate

  local status=0
  uv run ops-eval conversations \
    --target "$target" \
    --cases "$CASE_SET" \
    --out "$reports_dir" || status=$?

  printf 'tradeoff-run: %s measured into %s (ops-eval exited %s)\n' "$option_id" "$reports_dir" "$status" >&2
  return "$status"
}

cmd_option() {
  local option_id=$1
  shift
  [ "${1:-}" = "--target" ] || usage
  [ -n "${2:-}" ] || usage
  [[ $option_id =~ $OPTION_ID_PATTERN ]] || die "the option id must match ${OPTION_ID_PATTERN}"
  run_option "$option_id" "$2"
}

cmd_gpu() {
  local sha=$1
  [[ $sha =~ $SHA_PATTERN ]] || die "the sha must be 40 lowercase hex characters"

  local evidence_dir
  evidence_dir="$EVIDENCE_ROOT/$(date -u +%Y-%m-%d)"

  workdir="$(mktemp -d)"
  export KUBECONFIG="$workdir/kubeconfig"
  trap finish EXIT
  trap 'exit 130' INT TERM

  applied=yes
  terraform -chdir="$MODULE_DIR" apply

  outputs="$(terraform -chdir="$MODULE_DIR" output -json)"
  local address
  address="$(output_value public_ip)"

  # The k3s admin file is written 0600 on the host and holds cluster-admin, so it is read
  # over SSH into the throwaway directory and rewritten to dial the forwarded port.
  ssh -o StrictHostKeyChecking=accept-new "ubuntu@$address" sudo cat /etc/rancher/k3s/k3s.yaml >"$KUBECONFIG"
  ssh -o StrictHostKeyChecking=accept-new -N \
    -L "$API_PORT:127.0.0.1:$API_PORT" \
    -L "$NODE_PORT:127.0.0.1:$NODE_PORT" \
    "ubuntu@$address" &
  forward=$!

  # `gpu/networkpolicies.yaml` names an `apps` policy, so the namespace has to exist on a
  # bare host. Written as a client-side render piped to apply rather than `create ... ||
  # true`, which reports success for every other failure too.
  kubectl create namespace apps --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -k "$GPU_OVERLAY"

  # The weights are about 15GB before vLLM loads them, and loading them plus warming CUDA
  # graphs is minutes more.
  kubectl -n gpu wait --for=condition=complete job/vllm-model-fetch --timeout=30m
  kubectl -n gpu rollout status deployment/vllm --timeout=20m

  local status=0
  run_option "$GPU_OPTION" "$GPU_TARGET" || status=$?

  mkdir -p "$evidence_dir"
  printf '%s\n' "$outputs" | redact >"$evidence_dir/terraform-output.json"
  kubectl -n gpu exec deployment/vllm -- nvidia-smi | redact >"$evidence_dir/nvidia-smi.txt"
  kubectl get pods --all-namespaces -o wide | redact >"$evidence_dir/pods.txt"
  curl --fail --silent --show-error --max-time 30 "http://127.0.0.1:$NODE_PORT/metrics" |
    redact >"$evidence_dir/vllm-metrics.txt"
  local report
  for report in report.json report.md; do
    redact <"$EVIDENCE_ROOT/reports/$GPU_OPTION/$report" >"$evidence_dir/$report"
  done

  printf 'tradeoff-run: captured %s (image %s)\n' "$evidence_dir" "$sha" >&2
  return "$status"
}

cmd_cost() {
  local start=$1 end=$2
  [[ $start =~ $DATE_PATTERN ]] || die "the start must be YYYY-MM-DD"
  [[ $end =~ $DATE_PATTERN ]] || die "the end must be YYYY-MM-DD"
  [[ $start < $end ]] || die "the start must be before the end"

  # Unfiltered: a cost allocation tag cannot be active before its first resource exists, so
  # a tag filter would miss the run. The account must carry no other spend that day.
  local result
  result="$(aws ce get-cost-and-usage --region "$COST_EXPLORER_REGION" \
    --time-period "Start=$start,End=$end" \
    --granularity DAILY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --output json)"
  mkdir -p "$EVIDENCE_ROOT"
  printf '%s\n' "$result" | redact >"$EVIDENCE_ROOT/cost.json"
  printf 'tradeoff-run: wrote %s\n' "$EVIDENCE_ROOT/cost.json" >&2
}

main() {
  case "${1:-}" in
    option)
      [ "$#" -eq 4 ] || usage
      shift
      cmd_option "$@"
      ;;
    gpu)
      [ "$#" -eq 2 ] || usage
      cmd_gpu "$2"
      ;;
    cost)
      [ "$#" -eq 3 ] || usage
      cmd_cost "$2" "$3"
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"
