#!/usr/bin/env bash
# The kind e2e's checks while the local model is out of use. Each one goes
# through the deployed gateway and needs no model: the process is up, its
# dependencies are reachable, auth refuses a missing key, the redactor hides an
# email. The key is the throwaway CI one whose digest
# `gateway-secrets.yaml` carries; the cluster lives for one run.
#
#   infra/k8s/kind/smoke.sh http://127.0.0.1:18080
set -euo pipefail

base="${1:?usage: smoke.sh <gateway base url>}"
evals_key="ci-e2e-kind-throwaway-key"
failed=0

check() {  # check <name> <expected status> <curl args...>
  local name=$1 want=$2; shift 2
  local got
  got=$(curl -s -o /tmp/smoke-body -w '%{http_code}' "$@")
  if [ "$got" = "$want" ]; then
    echo "ok   $name ($got)"
  else
    echo "FAIL $name: wanted $want, got $got"; head -c 400 /tmp/smoke-body; echo
    failed=1
  fi
}

check "process is up" 200 "$base/healthz"
check "dependencies reachable" 200 "$base/readyz"
check "a request with no key is refused" 401 "$base/v1/models"
check "the evals tenant lists its aliases" 200 "$base/v1/models" \
  -H "Authorization: Bearer $evals_key" -H "X-Tenant: evals"

check "the redactor answers" 200 -X POST "$base/v1/redact" \
  -H "Authorization: Bearer $evals_key" -H "X-Tenant: evals" -H "X-Data-Class: internal" \
  -H "Content-Type: application/json" -d '{"text": "Mail anna.berg@example.com today"}'
if grep -q "anna.berg@example.com" /tmp/smoke-body; then
  echo "FAIL the redactor returned the raw email"; failed=1
else
  echo "ok   the email is gone from the redacted text"
fi

exit "$failed"
