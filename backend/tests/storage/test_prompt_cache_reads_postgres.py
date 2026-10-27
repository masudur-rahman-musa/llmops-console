"""Prompt cache sums and savings over real ledger rows.

Only the SQL parts are here: the cache token sums per model, and the window, status,
answer-cache and tenant filters of the prompt saving. The price arithmetic itself is
`cost_for`'s and is pinned in tests/limits. Rows are raw inserts into the shipped
migrations' schema, like tests/storage/test_dashboard_reads_postgres.py.

The contract these cases are written against, before it exists:

* `ModelTokens.cache_read` and `.cache_write` (float, default 0), the sums of the
  ledger's `cache_read_tokens` and `cache_write_tokens` per model;
* `CostRead.prompt_cache_saved_usd` (float, default 0.0): for served rows (not cached,
  status under 500) in the window and tenant filter, the price with every cache token
  billed as plain input, minus the real price.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from psycopg import sql

from app.limits.prices import PriceTable
from app.schemas.dashboard import ModelTokens
from app.storage.dashboard_reads import PostgresConsoleReader

from .conftest import Pool, requires_postgres

pytestmark = [requires_postgres, pytest.mark.db]

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
SECOND = timedelta(seconds=1)
MILLION = 1_000_000
WINDOW_DAYS = 7
START = NOW - timedelta(days=WINDOW_DAYS)

PRICES = PriceTable.model_validate(
    {
        "prices": {
            "testprov/model-a": [
                {
                    "effective_from": "2026-01-01",
                    "input_usd_per_1m": "10.00",
                    "output_usd_per_1m": "20.00",
                    "cache_read_usd_per_1m": "1.00",
                    "cache_write_usd_per_1m": "12.50",
                }
            ]
        }
    }
)
"""One priced model; any other `provider/model` is unpriced."""


def insert_row(
    pool: Pool,
    schema: str,
    /,
    *,
    ts: datetime,
    tenant: str = "nordlicht",
    provider: str = "testprov",
    model: str = "model-a",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cached: bool = False,
    status: int = 200,
) -> None:
    """One `cost_ledger` row with cache tokens."""
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (ts, tenant, alias, provider, model, input_tokens, "
                "output_tokens, cache_read_tokens, cache_write_tokens, cached, cost_usd, "
                "price_version, estimated, session, trace_id, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(sql.Identifier(schema, "cost_ledger")),
            (
                ts,
                tenant,
                "agent-default",
                provider,
                model,
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
                cached,
                Decimal("0.001000"),
                "testprov/model-a@2026-01-01",
                False,
                None,
                "0af7651916cd43dd8448eb211c80319c",
                status,
            ),
        )


def seed_priced_rows(pool: Pool, schema: str) -> None:
    """The three served rows that make 11.00 in total and 6.50 for nordlicht."""
    hour = timedelta(hours=1)
    insert_row(pool, schema, ts=NOW - hour, cache_read=MILLION)
    insert_row(pool, schema, ts=NOW - 2 * hour, cache_write=MILLION)
    insert_row(pool, schema, ts=NOW - 3 * hour, tenant="cuan", cache_read=MILLION // 2)


def saving(pool: Pool, schema: str, *, tenant: str | None = None) -> float:
    reader = PostgresConsoleReader(pool, schema=schema)
    return reader.cost(
        days=WINDOW_DAYS, tenant=tenant, now=NOW, prices=PRICES
    ).prompt_cache_saved_usd


def test_model_tokens_sum_the_cache_tokens(pool: Pool, migrated_schema: str) -> None:
    """Catches a reader that sums only plain input and output (cache fields stay 0), reads
    the write column into `cache_read` or the reverse, or folds the cache tokens into
    `input`: input must stay 150 beside read 300 and write 50."""
    schema = migrated_schema
    hour = timedelta(hours=1)
    insert_row(pool, schema, ts=NOW - hour, input_tokens=100, output_tokens=10, cache_read=300)
    insert_row(pool, schema, ts=NOW - 2 * hour, input_tokens=50, output_tokens=10, cache_write=50)
    reader = PostgresConsoleReader(pool, schema=schema)

    assert reader.model_tokens(WINDOW_DAYS, NOW) == (
        ModelTokens(model="model-a", input=150.0, output=20.0, cache_read=300.0, cache_write=50.0),
    )


def test_prompt_saving_prices_reads_and_writes(pool: Pool, migrated_schema: str) -> None:
    """Catches a saving that ignores write cost (would be 13.50), counts reads at the full
    rate (0), clamps the write row's loss to 0, or adds the tenants wrongly: reads save
    9.00 and 4.50, a write costs 2.50 more, total 11.00."""
    seed_priced_rows(pool, migrated_schema)

    assert saving(pool, migrated_schema) == pytest.approx(11.00)


def test_prompt_saving_skips_answer_cache_errors_and_old_rows(
    pool: Pool, migrated_schema: str
) -> None:
    """Catches a saving that counts answer-cache rows, 5xx rows, or rows before the window:
    each distractor reads 1,000,000 and satisfies every other condition, so dropping any
    one filter moves the total off 11.00."""
    schema = migrated_schema
    seed_priced_rows(pool, schema)
    hour = timedelta(hours=1)
    insert_row(pool, schema, ts=NOW - hour, cache_read=MILLION, cached=True)
    insert_row(pool, schema, ts=NOW - hour, cache_read=MILLION, status=502)
    insert_row(pool, schema, ts=START - SECOND, cache_read=MILLION)

    assert saving(pool, schema) == pytest.approx(11.00)


def test_prompt_saving_follows_the_tenant_filter(pool: Pool, migrated_schema: str) -> None:
    """Catches a saving that ignores the tenant filter (would stay 11.00 and include cuan's
    4.50): nordlicht alone is 9.00 less 2.50."""
    seed_priced_rows(pool, migrated_schema)

    assert saving(pool, migrated_schema, tenant="nordlicht") == pytest.approx(6.50)


def test_an_unpriced_model_saves_nothing(pool: Pool, migrated_schema: str) -> None:
    """Catches a saving that raises on a model with no price, or that books the plain-input
    price of an unpriced model as savings: the answer is 0.0 with no error."""
    insert_row(
        pool,
        migrated_schema,
        ts=NOW - timedelta(hours=1),
        provider="ghost",
        model="nameless",
        cache_read=MILLION,
    )

    assert saving(pool, migrated_schema) == 0.0
