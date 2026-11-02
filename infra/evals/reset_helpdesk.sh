#!/usr/bin/env bash
# Put the help desk's local database back to the seed, its policy rows included, and check
# that policy search can work. `ops-eval conversations` runs this before every play
# (`--reset-command`), and `prepare_helpdesk.sh` runs it first with `--no-check`.
# Usage: HELPDESK_DIR=<help desk dir> reset_helpdesk.sh [--no-check]
#    or: reset_helpdesk.sh [--no-check] <help desk dir>
#
# The work is in `reset_helpdesk.py`, run in the help desk's own Python so it can call the
# help desk's reset. It keeps the policy rows across the reset instead of running the slow
# ingest again, and exits 1 when policy search would fail, which stops the eval run.
set -euo pipefail

flags=()
if [ "${1:-}" = "--no-check" ]; then
  flags+=(--no-check)
  shift
fi
here="$(cd "$(dirname "$0")" && pwd)"

cd "${1:-${HELPDESK_DIR:?set HELPDESK_DIR or pass the help desk dir}}/backend"

# The reset reads DATABASE_URL from the environment, so it gets `.env.demo`.
uv run --env-file .env.demo python "$here/reset_helpdesk.py" ${flags[@]+"${flags[@]}"}
