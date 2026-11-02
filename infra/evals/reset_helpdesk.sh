#!/usr/bin/env bash
# Put the help desk's local database back to the seed. `ops-eval conversations` runs this
# before every play (`--reset-command`), and `prepare_helpdesk.sh` runs it first.
# Usage: HELPDESK_DIR=<help desk dir> reset_helpdesk.sh   (or: reset_helpdesk.sh <dir>)
#
# Runs the help desk's own reset (drop, migrate, checkpointer, seed, reviews), minus its
# ingest step: on macOS that ingest can abort (134) as it exits, after every document is
# written, and the reset counts the abort as a failure. The ingest runs once per eval run
# in `prepare_helpdesk.sh`, not once per play.
set -euo pipefail

cd "${1:-${HELPDESK_DIR:?set HELPDESK_DIR or pass the help desk dir}}/backend"

# The reset reads DATABASE_URL from the environment, so it gets `.env.demo`.
uv run --env-file .env.demo python - <<'PY'
import sys

sys.path.insert(0, "scripts")
import e2e_reset

e2e_reset._ingest = lambda *args, **kwargs: None
e2e_reset.main()
PY
