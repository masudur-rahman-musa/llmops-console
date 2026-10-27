r"""The command that owns the ledger's schema.

`make migrate` applies the SQL files. The gateway can also migrate as it starts, when
`OPSCONSOLE_MIGRATE_ON_START` is on: `app.main.migrate_at_start` runs
the same files under a Postgres advisory lock, so replicas starting together
apply each file once. This command stays for a deploy that wants to run the DDL
as its own step, and with the switch off `/readyz`'s `postgres` probe is still
the only thing that says whether the server is reachable.

The entry point reads `OPSCONSOLE_DATABASE_URL` through `Settings` and
answers a process exit code. An unconfigured database is a non-zero code with a message,
never a quiet success: in a deploy pipeline "migrated nothing, exit 0" is the
worst available answer, because the next step starts the gateway against a
database with no table in it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.core.config import Settings
from app.storage.db import DEFAULT_SCHEMA, apply_migrations, build_pool

NO_DATABASE_URL = 2
"""The exit code for "nothing is configured", distinct from 1 so a CronJob's
alerting can tell a misconfiguration from a failed statement."""


def _parser(description: str) -> argparse.ArgumentParser:
    """The one option the command takes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help=f"the schema holding the ledger tables (default: {DEFAULT_SCHEMA})",
    )
    return parser


def _database_url() -> str | None:
    """Where Postgres is, or `None` when nothing configured one."""
    return Settings().database_url


def migrate(argv: Sequence[str] | None = None) -> int:
    """Apply every migration this database has not seen. `make migrate`.

    Safe to run twice: the second run applies nothing and leaves the data alone,
    which is what makes it a deploy step rather than a thing someone has to
    remember whether they already did.
    """
    options = _parser("Apply the ledger migrations.").parse_args(argv)
    url = _database_url()
    if url is None:
        print("OPSCONSOLE_DATABASE_URL is not set; nothing was migrated", file=sys.stderr)
        return NO_DATABASE_URL

    pool = build_pool(url)
    try:
        with pool.connection() as conn:
            applied = apply_migrations(conn, schema=options.schema)
    finally:
        pool.close()

    if applied:
        print(f"applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("already up to date; nothing was applied")
    return 0
