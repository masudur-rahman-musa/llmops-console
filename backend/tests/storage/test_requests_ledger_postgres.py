"""The `prompt_version` ledger column and the newest-requests read, against a real server.

The `db` file of an earlier change: the column, the INSERT that fills it and the newest-first capped
read are things only PostgreSQL can show. Every case runs the SHIPPED migrations into a
throwaway schema (the fixtures in tests/storage/conftest.py, unchanged), so a column added
to the SQL file and not to the writer's INSERT, or the reverse, is a failure here.

The contract these cases are written against, before it exists:

* `migrations/006_ledger_prompt_version.sql` adds a nullable `text` column
  `cost_ledger.prompt_version`;
* `CostLedgerRow.prompt_version: str | None` (default None) is written by
  `PostgresCostLedgerWriter`;
* `PostgresConsoleReader.recent_requests() -> tuple[RequestRow, ...]` answers the newest
  `ROW_LIMIT` ledger rows, newest first, with `trace_url` None (the route builds the link).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from psycopg import sql

from app.schemas.dashboard import RequestRow
from app.storage import CostLedgerRow, PostgresCostLedgerWriter
from app.storage.cost_ledger import COST_LEDGER
from app.storage.dashboard_reads import ROW_LIMIT, PostgresConsoleReader

from .conftest import Pool, requires_postgres

pytestmark = [requires_postgres, pytest.mark.db]

NEWEST = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PROMPT_VERSION = "9f2c4be07a1d3c58e6b0a4d2f1c7e9835b6a0d4c2e8f1a7b3c5d9e0f2a4b6c8d"


def a_row(**overrides: object) -> CostLedgerRow:
    """One well-formed row, with the named fields replaced."""
    fields: dict[str, object] = {
        "ts": NEWEST,
        "tenant": "nordlicht",
        "alias": "agent-default",
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "input_tokens": 137,
        "output_tokens": 29,
        "cached": False,
        "cost_usd": Decimal("0.000028"),
        "price_version": "anthropic/claude-haiku-5-5@2026-10-01",
        "estimated": False,
        "session": None,
        "trace_id": TRACE_ID,
    }
    fields.update(overrides)
    return CostLedgerRow.model_validate(fields)


def stored_prompt_versions(pool: Pool, schema: str) -> list[str | None]:
    """The `prompt_version` of every ledger row, oldest first."""
    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("select prompt_version from {} order by ts").format(
                sql.Identifier(schema, COST_LEDGER)
            )
        ).fetchall()
    return [None if row[0] is None else str(row[0]) for row in rows]


def test_migration_adds_nullable_prompt_version(pool: Pool, migrated_schema: str) -> None:
    """Catches a migration that never adds the column, adds it NOT NULL (every request sent
    without the header would then be a lost ledger row), or types it other than text. A raw
    INSERT that names no `prompt_version` must also land and read back null, which is what
    keeps every older writer working."""
    with pool.connection() as conn:
        found = conn.execute(
            "select is_nullable, data_type from information_schema.columns "
            "where table_schema = %s and table_name = %s and column_name = 'prompt_version'",
            (migrated_schema, COST_LEDGER),
        ).fetchall()
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, trace_id) "
                "VALUES (%s, 'nordlicht', 'agent-default', 'anthropic', 'claude-haiku-5-5', "
                "1, 1, false, 0, 'v', false, null, %s)"
            ).format(sql.Identifier(migrated_schema, COST_LEDGER)),
            (NEWEST, TRACE_ID),
        )

    assert [tuple(row) for row in found] == [("YES", "text")]
    assert stored_prompt_versions(pool, migrated_schema) == [None]


def test_writer_stores_prompt_version(pool: Pool, migrated_schema: str) -> None:
    """Catches an INSERT that leaves the new column out of its list (the value is lost and
    every row reads back null), or binds it into the wrong text column: the row with a
    value and the row with none are both written and both read back as written."""
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row(ts=NEWEST - timedelta(minutes=1), prompt_version=None))
    writer.write(a_row(ts=NEWEST, prompt_version=PROMPT_VERSION))
    writer.flush()

    assert stored_prompt_versions(pool, migrated_schema) == [None, PROMPT_VERSION]
    with pool.connection() as conn:
        tenant, model = conn.execute(
            sql.SQL("select tenant, model from {} where prompt_version is not null").format(
                sql.Identifier(migrated_schema, COST_LEDGER)
            )
        ).fetchall()[0]
    assert (tenant, model) == ("nordlicht", "claude-haiku-5-5")


def test_recent_requests_are_newest_first_and_capped(pool: Pool, migrated_schema: str) -> None:
    """Catches a read in insertion order (the rows go in oldest first), one without the
    LIMIT (101 rows are stored), one that keeps the oldest 100 instead of the newest, and
    one that drops a column on the way out. `input_tokens` carries the row's age in
    minutes, so the order is read off the rows themselves."""
    assert ROW_LIMIT == 100
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    for minutes in range(100, -1, -1):
        writer.write(
            a_row(
                ts=NEWEST - timedelta(minutes=minutes),
                input_tokens=minutes,
                prompt_version=PROMPT_VERSION if minutes == 0 else None,
                cached=minutes == 1,
                cost_usd=Decimal("0.000028") if minutes == 0 else Decimal("0"),
            )
        )
    writer.flush()

    read = PostgresConsoleReader(pool, schema=migrated_schema).recent_requests()

    assert len(read) == 100
    assert [row.input_tokens for row in read] == list(range(100))
    assert 100 not in [row.input_tokens for row in read]
    assert read[0] == RequestRow(
        ts=NEWEST,
        tenant="nordlicht",
        alias="agent-default",
        provider="anthropic",
        model="claude-haiku-5-5",
        input_tokens=0,
        output_tokens=29,
        cached=False,
        cost_usd=0.000028,
        prompt_version=PROMPT_VERSION,
        trace_id=TRACE_ID,
        trace_url=None,
    )
    assert read[1].cached is True
    assert read[1].prompt_version is None
