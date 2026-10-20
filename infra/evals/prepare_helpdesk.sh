#!/usr/bin/env bash
# Put the help desk's local database and Qdrant back to the seed before an eval
# run: `make eval` calls this, never by hand. Usage: prepare_helpdesk.sh <help desk dir>
#
# Runs the help desk's own reset (drop, migrate, checkpointer, seed, reviews),
# minus its ingest step: that step still names a script the help desk has
# since removed. The ingest runs here instead, one tenant and corpus version at
# a time, the way `infra/scripts/server_helpdesk_data.sh` does. It redacts each
# chunk through this gateway's `/v1/redact`, so it needs the gateway up and the
# run's keys in HELPDESK_GATEWAY_API_KEYS.
set -euo pipefail

cd "$1/backend"

# The reset reads DATABASE_URL from the environment, so it gets `.env.demo`.
uv run --env-file .env.demo python - <<'PY'
import sys

sys.path.insert(0, "../dev/scripts")
import e2e_reset

run_step = e2e_reset._step
e2e_reset._step = lambda name, *args, **kwargs: (
    None if name == "ingest.py" else run_step(name, *args, **kwargs)
)
e2e_reset.main()
PY

# Real environment wins over `.env.demo` and `.env` in the help desk's
# settings, so a laptop's own gateway URL and keys are not used here.
export GATEWAY_URL="http://localhost:8080/v1"
export GATEWAY_API_KEYS="$HELPDESK_GATEWAY_API_KEYS"
export POLICY_DIR="$PWD/data/policies"
for tenant_dir in data/policies/*/; do
  tenant=$(basename "$tenant_dir")
  for version_dir in "$tenant_dir"*/; do
    version=$(basename "$version_dir")
    # On macOS onnxruntime can abort (134) as the process exits, after every
    # document is written; each document commits on its own, so that is fine.
    rc=0
    uv run python -m helpdesk.mcp_server.cli --tenant "$tenant" --version "$version" || rc=$?
    if [ "$rc" != 0 ] && [ "$rc" != 134 ]; then exit "$rc"; fi
  done
done
echo "prepare_helpdesk: reset, seeded and ingested"
