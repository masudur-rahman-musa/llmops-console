#!/usr/bin/env bash
# The one-off AWS evidence run, started by the owner from their own shell.
#
#   scripts/aws_evidence.sh run <commit sha>
#     terraform apply on infra/terraform/aws (the owner reads the plan and types yes),
#     deploy infra/k8s/aws onto the new EKS cluster against RDS and ElastiCache,
#     smoke test the gateway, prove the IRSA role with one S3 write, capture the
#     evidence under docs/aws-evidence/<UTC date>/, then terraform destroy.
#
#   scripts/aws_evidence.sh cost <start YYYY-MM-DD> <end YYYY-MM-DD>
#     write docs/aws-evidence/cost.json from Cost Explorer, once the spend has
#     landed (about a day after the run). End is exclusive, as Cost Explorer reads it.
#
# Once apply has started, the EXIT trap destroys the stack whatever happens next:
# a failed step, Ctrl-C, or a clean finish. A stack left running bills around
# 0.20 USD an hour. If destroy itself fails, the trap says so loudly and exits
# non-zero, and `terraform -chdir=infra/terraform/aws destroy` has to be run by hand.
#
# Credentials never touch disk or the terminal. The kubeconfig lives in a mktemp
# directory removed on exit, never the owner's default one. The database password
# flows from Secrets Manager through a pipe into `kubectl apply -f -` and is never
# held in a shell variable. The tenant key is fresh per run and dies with the cluster.
# Tracing is never switched on in this file: it would print every expanded command.
#
# Captured files pass through `redact`, which replaces every standalone 12-digit
# number (an AWS account ID) with ACCOUNT_ID, since the evidence is committed.
#
# Pinned by tests/infra/test_aws_evidence.py, which runs this file against fake
# terraform, aws, kubectl and curl binaries. AWS_EVIDENCE_TEST_ROOT moves the
# evidence folder for those tests and is read by nothing else.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_DIR="$REPO_ROOT/infra/terraform/aws"
OVERLAY_DIR="$REPO_ROOT/infra/k8s/aws"
EVIDENCE_ROOT="${AWS_EVIDENCE_TEST_ROOT:-$REPO_ROOT}/docs/aws-evidence"

GATEWAY_IMAGE="ghcr.io/masudur-rahman-musa/ai-operations-console/gateway"
# Bash `=~` is POSIX ERE over the whole string: `$` does not match before a
# trailing newline, so `sha\n` is refused.
SHA_PATTERN='^[0-9a-f]{40}$'
DATE_PATTERN='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
# The port-forward's local end, the same one deploy.yml's smoke uses.
LOCAL_PORT=18080
# Cost Explorer is a global API served from us-east-1 only.
COST_EXPLORER_REGION="us-east-1"

usage() {
  printf 'usage: %s run <40 lowercase hex sha>\n       %s cost <YYYY-MM-DD start> <YYYY-MM-DD end>\n' "$0" "$0" >&2
  exit 2
}

die() {
  printf 'aws-evidence: %s\n' "$*" >&2
  exit 1
}

redact() {
  python3 -c 'import re, sys; sys.stdout.write(re.sub(r"(?<![0-9])[0-9]{12}(?![0-9])", "ACCOUNT_ID", sys.stdin.read()))'
}

# Builds the gateway-secrets manifest. Reads the RDS master credential JSON on
# stdin, writes the manifest on stdout. The admin key and the budget override
# secret are generated here and never leave this process: only the admin key's
# digest enters the Secret, so /admin is closed for the whole run.
SECRET_MANIFEST_PY='
import hashlib
import json
import os
import secrets
import sys
import urllib.parse

credential = json.load(sys.stdin)
user = urllib.parse.quote(credential["username"], safe="")
password = urllib.parse.quote(credential["password"], safe="")
endpoint = os.environ["RDS_ENDPOINT"]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


manifest = {
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {"name": "gateway-secrets", "namespace": "apps"},
    "type": "Opaque",
    "stringData": {
        # The database name is infra/terraform/aws/data.tf db_name.
        "OPSCONSOLE_DATABASE_URL": f"postgresql://{user}:{password}@{endpoint}/opsconsole?sslmode=require",
        "OPSCONSOLE_API_KEYS": "evals:" + digest(os.environ["TENANT_KEY"]),
        "OPSCONSOLE_ADMIN_KEY_SHA256": digest(secrets.token_urlsafe(32)),
        "OPSCONSOLE_BUDGET_OVERRIDE_SECRET": secrets.token_urlsafe(32),
    },
}
json.dump(manifest, sys.stdout)
'

# Checks the smoke answers and writes them as one JSON document on stdout.
SMOKE_PY='
import json
import os
import sys

answer = json.loads(os.environ["CHAT_ANSWER"])
choices = answer.get("choices") or [{}]
content = (choices[0].get("message") or {}).get("content")
if not content:
    sys.exit("smoke: the chat turn returned no choices[0] content")
json.dump({"readyz": json.loads(os.environ["READYZ_ANSWER"]), "chat": answer}, sys.stdout, indent=2)
print()
'

# State the EXIT trap reads.
workdir=""
applied=""
forward=""

finish() {
  local status=$?
  if [ -n "$forward" ]; then
    # The port-forward may already have exited; there is nothing to stop then.
    kill "$forward" 2>/dev/null || true
  fi
  if [ -n "$applied" ]; then
    printf 'aws-evidence: destroying the stack\n' >&2
    if ! terraform -chdir="$MODULE_DIR" destroy -auto-approve; then
      printf 'aws-evidence: DESTROY FAILED, the stack is still billing. Run: terraform -chdir=%s destroy\n' "$MODULE_DIR" >&2
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

cmd_run() {
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
  local region cluster rds_endpoint redis_endpoint bucket
  region="$(output_value region)"
  cluster="$(output_value cluster_name)"
  rds_endpoint="$(output_value rds_endpoint)"
  redis_endpoint="$(output_value redis_endpoint)"
  bucket="$(output_value backup_bucket)"

  aws eks update-kubeconfig --region "$region" --name "$cluster" --kubeconfig "$KUBECONFIG"

  # The overlay is built to a file rather than piped, so the only manifest ever
  # read from stdin is the Secret. The file holds no credential.
  kubectl kustomize --load-restrictor=LoadRestrictionsNone "$OVERLAY_DIR" >"$workdir/overlay.yaml"
  kubectl apply -f "$workdir/overlay.yaml"

  # The RDS identifier is the endpoint's first label; the module names both.
  local secret_arn tenant_key
  secret_arn="$(aws rds describe-db-instances --region "$region" \
    --db-instance-identifier "${rds_endpoint%%.*}" \
    --query 'DBInstances[0].MasterUserSecret.SecretArn' --output text)"
  tenant_key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  aws secretsmanager get-secret-value --region "$region" --secret-id "$secret_arn" \
    --query SecretString --output text |
    RDS_ENDPOINT="$rds_endpoint" TENANT_KEY="$tenant_key" python3 -c "$SECRET_MANIFEST_PY" |
    kubectl apply -f -

  # Explicit env wins over the ConfigMap's in-cluster redis:// address.
  kubectl -n apps set image deployment/gateway "gateway=$GATEWAY_IMAGE:$sha"
  kubectl -n apps set env deployment/gateway "OPSCONSOLE_REDIS_URL=rediss://$redis_endpoint:6379/0"

  # The role name is `<name>-backups` in backups.tf, and the cluster is named `<name>`.
  local role_arn
  role_arn="$(aws iam get-role --role-name "$cluster-backups" --query Role.Arn --output text)"
  kubectl -n platform annotate serviceaccount backups "eks.amazonaws.com/role-arn=$role_arn" --overwrite

  kubectl -n platform rollout status statefulset/qdrant --timeout=10m
  kubectl -n apps rollout status deployment/gateway --timeout=10m
  # The model fetch is about 2 GB before llama.cpp loads it.
  kubectl -n apps rollout status deployment/llama --timeout=30m

  # Creates the ledger tables in RDS, so the smoke's turn lands a row there.
  kubectl -n apps exec deployment/gateway -- ops-migrate

  kubectl -n apps port-forward deployment/gateway "$LOCAL_PORT:8080" >/dev/null 2>&1 &
  forward=$!

  local readyz="" attempt
  for attempt in $(seq 1 30); do
    if readyz="$(curl --fail --silent --max-time 5 "http://127.0.0.1:$LOCAL_PORT/readyz")"; then
      break
    fi
    readyz=""
    printf 'aws-evidence: /readyz not ready yet (attempt %s)\n' "$attempt" >&2
    sleep 2
  done
  [ -n "$readyz" ] || die "smoke: /readyz never answered 200"

  # The key goes to curl on stdin, so it is not in the process list.
  local body answer
  body='{"model":"agent-default","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
  answer="$(printf 'Authorization: Bearer %s\n' "$tenant_key" |
    curl --fail --silent --show-error --max-time 120 \
      -X POST "http://127.0.0.1:$LOCAL_PORT/v1/chat/completions" \
      -H @- \
      -H "X-Tenant: evals" \
      -H "X-Data-Class: pii" \
      -H "Content-Type: application/json" \
      --data "$body")"
  local smoke
  smoke="$(READYZ_ANSWER="$readyz" CHAT_ANSWER="$answer" python3 -c "$SMOKE_PY")"

  kubectl -n platform create configmap irsa-evidence "--from-literal=BUCKET=$bucket"
  kubectl apply -f "$OVERLAY_DIR/irsa-job.yaml"
  kubectl -n platform wait --for=condition=complete job/irsa-evidence --timeout=10m

  mkdir -p "$evidence_dir"
  printf '%s\n' "$outputs" | redact >"$evidence_dir/terraform-output.json"
  kubectl get nodes -o wide | redact >"$evidence_dir/nodes.txt"
  kubectl get pods --all-namespaces -o wide | redact >"$evidence_dir/pods.txt"
  printf '%s\n' "$smoke" | redact >"$evidence_dir/smoke.json"
  kubectl -n platform logs job/irsa-evidence | redact >"$evidence_dir/irsa.txt"

  printf 'aws-evidence: captured %s\n' "$evidence_dir" >&2
}

cmd_cost() {
  local start=$1 end=$2
  [[ $start =~ $DATE_PATTERN ]] || die "the start must be YYYY-MM-DD"
  [[ $end =~ $DATE_PATTERN ]] || die "the end must be YYYY-MM-DD"
  [[ $start < $end ]] || die "the start must be before the end"

  # Unfiltered: a cost allocation tag cannot be active before its first resource
  # exists, so a tag filter would miss the run. The account must carry no other spend.
  local result
  result="$(aws ce get-cost-and-usage --region "$COST_EXPLORER_REGION" \
    --time-period "Start=$start,End=$end" \
    --granularity DAILY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --output json)"
  mkdir -p "$EVIDENCE_ROOT"
  printf '%s\n' "$result" | redact >"$EVIDENCE_ROOT/cost.json"
  printf 'aws-evidence: wrote %s\n' "$EVIDENCE_ROOT/cost.json" >&2
}

main() {
  case "${1:-}" in
    run)
      [ "$#" -eq 2 ] || usage
      cmd_run "$2"
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
