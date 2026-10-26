#!/usr/bin/env bash
# Put the gateway and the help desk back on their normal keys after a run:
# `make eval` and `make numbers` call this on exit, pass or fail, never by hand.
# Usage: restore_keys.sh <help desk dir>
#
# Both runs start the gateway and the help desk on throwaway keys made for the
# run, through the `infra/evals/` and `infra/local/` overlays. Left that way,
# every help desk chat gets a 401 afterwards. This recreates them from each
# repo's own base compose file, so they read their `.env.demo` and `.env` again.
# The run's key variables are unset first: compose would otherwise read them
# and put the throwaway keys straight back. `--no-deps` leaves the databases
# running as they are: only the containers holding keys are recreated.
set -euo pipefail

helpdesk_dir=$(cd "$1" && pwd)
unset OPSCONSOLE_API_KEYS OPSCONSOLE_ADMIN_KEY_SHA256 HELPDESK_GATEWAY_API_KEYS \
  OPSCONSOLE_EVAL_JUDGE_KEY NUMBERS_TENANT_KEY NUMBERS_ADMIN_KEY

cd "$(dirname "$0")/../.."
docker compose -f docker-compose.yml up -d --force-recreate --no-deps --wait gateway
docker compose --project-directory "$helpdesk_dir" -f "$helpdesk_dir/docker-compose.yml" \
  --profile app up -d --force-recreate --no-deps --wait \
  helpdesk-agent-api helpdesk-mcp-server helpdesk-scheduler
echo "restore_keys: gateway and help desk are back on their normal keys"
