"""Put the help desk's local data back to the seed before a play, its policies included.

`reset_helpdesk.sh` runs this in the help desk's own Python (`uv run` in its `backend/`), so
it can call the help desk's own reset: drop, migrate, checkpointer, seed, reviews. It skips
that reset's ingest: on macOS the ingest can abort (134) as it exits, and it is slow.

The reset drops the whole `public` schema, and `policy_documents` and `policy_chunks` with
it. Qdrant keeps the vectors, and each one names its document row by id, so a search that
hits a vector whose row is gone fails. So the rows are kept: before the drop they are copied
into a schema of their own that the drop does not touch, and after the seed they are copied
back. The seed gives every tenant the same id each time, so the copied rows fit.

Then a check: every current vector in each tenant's policy collection must name a document
row that exists, and there must be document rows at all. When that fails the script exits 1
with one line per problem, and the eval run stops instead of playing every case against a
policy search that cannot work. `--no-check` skips it, for the first reset of an eval run,
which comes before the ingest has filled anything.

Nothing from the help desk or SQLAlchemy is imported at the top, so the parts that decide
what to copy and what counts as broken can be tested without the help desk.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

KEEP_SCHEMA = "eval_kept_policies"
"""Where the policy rows wait while `public` is dropped and rebuilt."""

KEPT_TABLES = ("policy_documents", "policy_chunks")
"""The policy tables, documents first, because each chunk points at its document."""

SCROLL_PAGE = 256
"""How many vectors one Qdrant scroll call reads."""


def keep_statements(source: str = "public", keep: str = KEEP_SCHEMA) -> list[str]:
    """Copy every policy table from `source` into `keep`, replacing the last copy."""
    statements = [f"CREATE SCHEMA IF NOT EXISTS {keep}"]
    for table in reversed(KEPT_TABLES):
        statements.append(f"DROP TABLE IF EXISTS {keep}.{table}")
    for table in KEPT_TABLES:
        statements.append(f"CREATE TABLE {keep}.{table} AS TABLE {source}.{table}")
    return statements


def restore_statement(
    table: str, columns: Sequence[str], source: str = "public", keep: str = KEEP_SCHEMA
) -> str:
    """Copy one kept table back, column by column, so the order of columns does not matter."""
    names = ", ".join(f'"{column}"' for column in columns)
    return f"INSERT INTO {source}.{table} ({names}) SELECT {names} FROM {keep}.{table}"


def policy_problems(
    vector_documents: Mapping[str, set[str]],
    row_documents: Mapping[str, set[str]],
    played: set[str] | None = None,
) -> list[str]:
    """What would make a policy search fail, one line each; empty when search can work.

    `vector_documents` holds, per tenant slug, the document ids its current vectors name.
    `row_documents` holds, per tenant slug, the ids of its `policy_documents` rows.
    `played` names the tenants the eval ingests and plays; only their gaps count, so old
    vectors in a tenant no case uses never stop a run. `None` checks every tenant.
    """
    problems = []
    if not any(row_documents.values()):
        problems.append("the help desk has no policy_documents rows, so every policy search fails")
    for slug in sorted(vector_documents):
        if played is not None and slug not in played:
            continue
        missing = vector_documents[slug] - row_documents.get(slug, set())
        if missing:
            problems.append(
                f"{slug}: {len(missing)} documents named in Qdrant have no policy_documents "
                "row, so a policy search that finds them fails"
            )
    return problems


async def _keep(database_url: str) -> None:
    """Copy the policy rows aside, but only when there are some.

    An empty `public` copy would wipe a good kept copy, so a reset that stopped half way
    cannot lose the rows for every reset after it.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            exists = await connection.scalar(text("SELECT to_regclass('public.policy_documents')"))
            if exists is None:
                return
            count = await connection.scalar(text("SELECT count(*) FROM public.policy_documents"))
            if not count:
                return
            for statement in keep_statements():
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


async def _restore(database_url: str) -> None:
    """Copy the kept policy rows back into the fresh tables, when a copy exists."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            for table in KEPT_TABLES:
                kept = await connection.scalar(text(f"SELECT to_regclass('{KEEP_SCHEMA}.{table}')"))
                if kept is None:
                    return
                columns = (
                    await connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_schema = :schema AND table_name = :table "
                            "ORDER BY ordinal_position"
                        ),
                        {"schema": KEEP_SCHEMA, "table": table},
                    )
                ).scalars()
                await connection.execute(text(restore_statement(table, list(columns))))
    finally:
        await engine.dispose()


async def _check(database_url: str) -> list[str]:
    """Read which documents the vectors name and which rows exist, and compare them."""
    from helpdesk.mcp_server.services.rag.vectors import collection_name, create_client
    from helpdesk.shared.core.config import load_settings
    from qdrant_client import models
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            slugs = list((await connection.execute(text("SELECT slug FROM tenants"))).scalars())
            rows = await connection.execute(
                text(
                    "SELECT t.slug, d.id::text FROM policy_documents d "
                    "JOIN tenants t ON t.id = d.tenant_id"
                )
            )
            row_documents: dict[str, set[str]] = {slug: set() for slug in slugs}
            for slug, document_id in rows:
                row_documents[slug].add(document_id)
    finally:
        await engine.dispose()

    client = create_client(load_settings())
    current = models.Filter(
        must=[models.FieldCondition(key="is_current", match=models.MatchValue(value=True))]
    )
    vector_documents: dict[str, set[str]] = {}
    try:
        for slug in slugs:
            name = collection_name(slug)
            if not await client.collection_exists(name):
                continue
            named: set[str] = set()
            offset = None
            while True:
                points, offset = await client.scroll(
                    name,
                    scroll_filter=current,
                    limit=SCROLL_PAGE,
                    offset=offset,
                    with_payload=["document_id"],
                    with_vectors=False,
                )
                named.update(str(point.payload["document_id"]) for point in points if point.payload)
                if offset is None:
                    break
            vector_documents[slug] = named
    finally:
        await client.close()
    # The eval ingests one policy folder per tenant, so those are the tenants it plays.
    played = {slug for slug in slugs if (Path("data/policies") / slug).is_dir()}
    return policy_problems(vector_documents, row_documents, played)


def main(argv: list[str] | None = None) -> int:
    """Keep the policy rows, reset, put them back, then check that policy search can work."""
    parser = argparse.ArgumentParser(description="Reset the help desk's data for one eval play.")
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the policy check, for the reset that comes before the ingest",
    )
    args = parser.parse_args(argv)

    import asyncio

    sys.path.insert(0, "scripts")
    import e2e_reset
    from helpdesk.shared.core.config import load_settings

    database_url = load_settings().database_url
    # The same refusal the reset itself makes, before this script copies anything.
    e2e_reset.check_local_database_url(database_url)

    asyncio.run(_keep(database_url))
    e2e_reset._ingest = lambda *args, **kwargs: asyncio.run(_restore(database_url))
    e2e_reset.main([])

    if args.no_check:
        return 0
    problems = asyncio.run(_check(database_url))
    for problem in problems:
        print(f"reset_helpdesk: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
