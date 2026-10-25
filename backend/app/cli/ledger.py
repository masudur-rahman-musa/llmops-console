r"""The two commands that own the ledger's schema and its rollup.

`make migrate` applies the SQL files; `make refresh-rollup` recomputes the materialised
view the console's cost page reads. Neither runs on app start, deliberately: a
gateway that migrated its own database would race every other replica coming up
at the same time, and a deploy would have no step at which someone could look at
the DDL before it ran. `create_app` therefore assumes the table is already there,
and `/readyz`'s `postgres` probe is what says whether the server is reachable.

Both entry points read `OPSCONSOLE_DATABASE_URL` through `Settings` and
answer a process exit code. An unconfigured database is a non-zero code with a message,
never a quiet success: in a deploy pipeline "migrated nothing, exit 0" is the
worst available answer, because the next step starts the gateway against a
database with no table in it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.core.config import Settings
from app.storage.cost_ledger import COST_LEDGER_DAILY, refresh_daily_rollup
from app.storage.db import DEFAULT_SCHEMA, apply_migrations, build_pool

NO_DATABASE_URL = 2
"""The exit code for "nothing is configured", distinct from 1 so a CronJob's
alerting can tell a misconfiguration from a failed statement. One case per entry
point, because each reads the setting for itself."""


def _parser(description: str) -> argparse.ArgumentParser:
    """The one option all three commands share."""
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


def refresh_rollup(argv: Sequence[str] | None = None) -> int:
    """Recompute the daily cost rollup. `make refresh-rollup`.

    Safe to run at any time and as often as an operator likes: the view is
    derived entirely from `cost_ledger`, so a refresh answers the same rows as
    the one before it unless a request was served in between. The CronJob that
    calls this on a schedule belongs to the platform plan.
    """
    options = _parser(f"Refresh the {COST_LEDGER_DAILY} materialised view.").parse_args(argv)
    url = _database_url()
    if url is None:
        print("OPSCONSOLE_DATABASE_URL is not set; nothing was refreshed", file=sys.stderr)
        return NO_DATABASE_URL

    pool = build_pool(url)
    try:
        concurrently = refresh_daily_rollup(pool, schema=options.schema)
    finally:
        pool.close()

    how = "concurrently" if concurrently else "for the first time, exclusively"
    print(f"refreshed {options.schema}.{COST_LEDGER_DAILY} {how}")
    return 0
