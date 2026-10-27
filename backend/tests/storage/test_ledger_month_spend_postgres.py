"""`PostgresMonthSpendReader` against a real server.

Raw inserts into a throwaway schema with the shipped migrations, read back through the
reader. Each edge has a row on both sides: the first instant of the UTC month and the
second before it, the first instant of the next month. Each filter has a distractor that
satisfies every other condition, so removing that one filter is what changes the answer.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`. Fixtures come from
`tests/storage/conftest.py`: one session pool, one schema per test.

The contract these cases are written against, before it exists:
`PostgresMonthSpendReader(pool, *, schema)` with `tenant_month_spend(tenant, month)` and
`user_month_spend(tenant, user_id, month)`, answering micro-USD as an int, 0 when no rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from functools import partial

import pytest
from psycopg import sql

from app.storage.cost_ledger import PostgresMonthSpendReader

from .conftest import Pool, requires_postgres

pytestmark = [requires_postgres, pytest.mark.db]

MONTH = "2026-10"
USER_A = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"
USER_B = "u_00000000000000000000000000000002"


def insert_cost(
    pool: Pool,
    schema: str,
    *,
    ts: datetime,
    cost_usd: str,
    tenant: str = "nordlicht",
    cached: bool = False,
    user_id: str | None = None,
) -> None:
    """One `cost_ledger` row, raw."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cached, cost_usd, price_version, estimated, session, "
                "trace_id, user_id) "
                "VALUES (%s, %s, 'agent-default', 'anthropic', 'claude-haiku-4-5', 1, 1, "
                "%s, %s, 'anthropic/claude-haiku-4-5@2025-10-15', false, NULL, "
                "'0af7651916cd43dd8448eb211c80319c', %s)"
            ).format(sql.Identifier(schema, "cost_ledger")),
            (ts, tenant, cached, Decimal(cost_usd), user_id),
        )


def test_tenant_month_spend_sums_this_utc_month_not_cached(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches a month cut on local time or off by one second at either end, a sum that
    includes cached rows (the console's bars leave them out, so the gateway and the console
    would disagree), a sum over every tenant, and a result in dollars instead of micro-USD."""
    insert = partial(insert_cost, pool, migrated_schema)
    insert(ts=datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC), cost_usd="10.000001")
    insert(ts=datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC), cost_usd="5.000002")
    insert(ts=datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC), cost_usd="100.000000")
    insert(ts=datetime(2026, 11, 1, 0, 0, 0, tzinfo=UTC), cost_usd="200.000000")
    insert(ts=datetime(2026, 10, 15, tzinfo=UTC), cost_usd="300.000000", cached=True)
    insert(ts=datetime(2026, 10, 15, tzinfo=UTC), cost_usd="400.000000", tenant="cuan")

    reader = PostgresMonthSpendReader(pool, schema=migrated_schema)

    assert reader.tenant_month_spend("nordlicht", MONTH) == 15_000_003
    assert isinstance(reader.tenant_month_spend("nordlicht", MONTH), int)


def test_user_month_spend_counts_only_that_user(pool: Pool, migrated_schema: str) -> None:
    """Catches a user sum that drops the user filter (the tenant total), one that counts
    rows with no user as everyone's, and one that ignores the cached and month filters."""
    insert = partial(insert_cost, pool, migrated_schema)
    now = datetime(2026, 10, 10, tzinfo=UTC)
    insert(ts=now, cost_usd="1.000001", user_id=USER_A)
    insert(ts=now, cost_usd="2.000002", user_id=USER_A)
    insert(ts=now, cost_usd="50.000000", user_id=USER_B)
    insert(ts=now, cost_usd="60.000000", user_id=None)
    insert(ts=now, cost_usd="70.000000", user_id=USER_A, cached=True)
    insert(ts=datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC), cost_usd="80.000000", user_id=USER_A)
    insert(ts=now, cost_usd="90.000000", user_id=USER_A, tenant="cuan")

    reader = PostgresMonthSpendReader(pool, schema=migrated_schema)

    assert reader.user_month_spend("nordlicht", USER_A, MONTH) == 3_000_003
    assert reader.user_month_spend("nordlicht", USER_B, MONTH) == 50_000_000


def test_a_tenant_with_no_rows_spends_zero(pool: Pool, migrated_schema: str) -> None:
    """Catches a reader that answers `None` for an empty sum (SQL `sum` of no rows is NULL):
    `None` means unknown and would stop the counter being marked loaded. 0 is a real answer."""
    reader = PostgresMonthSpendReader(pool, schema=migrated_schema)

    assert reader.tenant_month_spend("nordlicht", MONTH) == 0
    assert reader.user_month_spend("nordlicht", USER_A, MONTH) == 0
