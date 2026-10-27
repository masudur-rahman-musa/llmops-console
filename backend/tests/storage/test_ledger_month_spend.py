"""The month-spend reader's two answers that need no database.

* a pool that cannot connect makes the reader raise `LedgerUnreadable`, which the route
  turns into 503 `BUDGET_UNAVAILABLE` rather than treating the spend as zero;
* with no database configured the reader answers `None`: there is nothing to load from.

The month and cached-row edges are in `test_ledger_month_spend_postgres.py`, which needs a
real server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn

import psycopg
import pytest

from app.storage.cost_ledger import (
    LedgerUnreadable,
    MonthSpendReader,
    NoDatabaseMonthSpendReader,
    PostgresMonthSpendReader,
)

USER = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"


class DownPool:
    """A pool double whose `connection()` raises the way a dead server does."""

    @contextmanager
    def connection(self) -> Iterator[NoReturn]:
        raise psycopg.OperationalError("connection refused")
        yield  # make it a generator


def test_a_failing_database_raises_ledger_unreadable() -> None:
    """Catches a reader that swallows the database error and answers 0 or `None`: the gateway
    would then treat an unreadable ledger as no spend and serve a tenant over its cap, the
    exact bug this plan fixes. Both methods are asserted, since they are two queries."""
    reader = PostgresMonthSpendReader(DownPool())  # type: ignore[arg-type]

    with pytest.raises(LedgerUnreadable):
        reader.tenant_month_spend("nordlicht", "2026-10")
    with pytest.raises(LedgerUnreadable):
        reader.user_month_spend("nordlicht", USER, "2026-10")


def test_the_no_database_reader_answers_none() -> None:
    """Catches a no-database reader that answers 0: a zero is a spend, and the store would
    mark the counter loaded from it. `None` means there is nothing to load from."""
    reader: MonthSpendReader = NoDatabaseMonthSpendReader()

    assert reader.tenant_month_spend("nordlicht", "2026-10") is None
    assert reader.user_month_spend("nordlicht", USER, "2026-10") is None
