"""Ledger off the reply path: queue, non-blocking writes, metrics, error handling.

An earlier change moves the Postgres ledger write off the hot path. `PostgresCostLedgerWriter`
now owns a bounded in-memory queue and a daemon worker; `write` enqueues and
returns at once, and a full queue drops and counts the row. Three cases need a
real Postgres (insert landing + metric, error metric, flush on close); those carry
`requires_postgres` and `db`, and are skipped without `OPSCONSOLE_TEST_DATABASE_URL`.

The counter under test is `gateway_ledger_writes_total{result}` in `app.core.metrics`.
"""

from __future__ import annotations

import queue as _queue_module
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, NoReturn
from unittest.mock import MagicMock, patch

import pytest

from app.storage import CostLedgerRow, PostgresCostLedgerWriter
from app.storage.cost_ledger import LEDGER_QUEUE_SIZE

from .conftest import Pool, requires_postgres

_Full = _queue_module.Full
_patch = patch

# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)


def a_row(**overrides: Any) -> CostLedgerRow:  # noqa: ANN401
    """One well-formed row."""
    fields: dict[str, Any] = {
        "ts": WHEN,
        "tenant": "nordlicht",
        "alias": "agent-default",
        "provider": "local",
        "model": "qwen2.5-7b-instruct",
        "region": "self",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached": False,
        "cost_usd": Decimal("0.001500"),
        "price_version": "local/qwen2.5-7b-instruct@2026-01-01",
        "estimated": False,
        "session": "sess-test01",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
    }
    fields.update(overrides)
    return CostLedgerRow(**fields)


class BlockingPool:
    """A pool double whose `connection()` blocks until `event` is set.

    Used to assert that `writer.write()` returns before the insert runs.
    """

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    @contextmanager
    def connection(self) -> Iterator[MagicMock]:
        self._event.wait()
        conn = MagicMock()
        conn.execute.return_value = MagicMock(fetchall=lambda: [], fetchone=lambda: None)
        yield conn


class RaisingPool:
    """A pool double whose `connection()` always raises."""

    @contextmanager
    def connection(self) -> Iterator[NoReturn]:
        raise RuntimeError("db is down")
        yield  # make it a generator


class RecordingConnection:
    """Records every execute call and does nothing else."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, query: object, params: object = None) -> MagicMock:
        self.statements.append(str(query))
        return MagicMock(fetchall=lambda: [], fetchone=lambda: None)


class RecordingPool:
    """A pool double that records statements and answers successfully."""

    def __init__(self) -> None:
        self.conn = RecordingConnection()

    @contextmanager
    def connection(self) -> Iterator[RecordingConnection]:
        yield self.conn


def _counter_value(label_result: str) -> float:
    """Current value of gateway_ledger_writes_total for one result label."""
    from app.core.metrics import registry

    for family in registry().collect():
        if family.name == "gateway_ledger_writes":
            for sample in family.samples:
                if sample.labels.get("result") == label_result:
                    return float(sample.value)
    return 0.0


# ---------------------------------------------------------------------------
# 1. Queue size constant
# ---------------------------------------------------------------------------


def test_the_ledger_queue_size_is_exactly_10000() -> None:
    """Catches an implementation that used a different bound — 1000 or
    unlimited — leaving the queue either too small for a traffic burst or
    able to grow without limit and exhaust process memory."""
    assert LEDGER_QUEUE_SIZE == 10000


# ---------------------------------------------------------------------------
# 2. write() does not block the caller
# ---------------------------------------------------------------------------


def test_the_ledger_write_does_not_block_the_caller() -> None:
    """Catches an implementation that calls pool.connection() synchronously
    inside write(), which blocks the calling thread until the insert finishes
    and puts every chat reply behind every ledger write."""
    unblock = threading.Event()
    pool = BlockingPool(unblock)
    writer = PostgresCostLedgerWriter(pool)  # type: ignore[arg-type]

    try:
        # write() must return before unblock is set.
        writer.write(a_row())
        # If we reach here without blocking, the write is non-blocking.
    finally:
        unblock.set()
        writer.close()


# ---------------------------------------------------------------------------
# 3. Successful insert lands the row and increments result="ok"  (db)
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.db
def test_a_successful_insert_counts_ok_and_lands_the_row(pool: Pool, migrated_schema: str) -> None:
    """Catches an implementation that enqueues but the worker never flushes,
    or one that counts ok before the insert rather than after."""
    from psycopg import sql

    from app.storage.cost_ledger import COST_LEDGER

    before = _counter_value("ok")
    writer = PostgresCostLedgerWriter(pool, schema=migrated_schema)
    writer.write(a_row())
    writer.flush()

    with pool.connection() as conn:
        rows = conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(migrated_schema, COST_LEDGER))
        ).fetchone()
    assert rows is not None
    assert rows[0] == 1
    assert _counter_value("ok") - before == 1.0


# ---------------------------------------------------------------------------
# 4. A failed insert is counted, not raised
# ---------------------------------------------------------------------------


def test_a_failed_ledger_insert_is_counted_not_raised() -> None:
    """Catches an implementation that lets a write exception escape to the
    caller — the chat reply would fail, and the provider already charged."""
    before = _counter_value("error")
    writer = PostgresCostLedgerWriter(RaisingPool())  # type: ignore[arg-type]
    writer.write(a_row())
    writer.flush()

    assert _counter_value("error") - before == 1.0


# ---------------------------------------------------------------------------
# 5. A full queue drops the row and counts it
# ---------------------------------------------------------------------------


def test_a_full_ledger_queue_drops_and_counts() -> None:
    """Catches an implementation that blocks or raises on a full queue instead
    of dropping, which lets a Postgres slowdown back-pressure chat replies."""
    before = _counter_value("dropped")
    pool = RecordingPool()
    writer = PostgresCostLedgerWriter(pool)  # type: ignore[arg-type]

    # Simulate a full queue without racing against the background worker.
    with _patch.object(writer._queue, "put_nowait", side_effect=_Full()):
        writer.write(a_row())

    assert _counter_value("dropped") - before == 1.0
    writer.close()


# ---------------------------------------------------------------------------
# 6. close() drains the queue before returning
# ---------------------------------------------------------------------------


def test_the_writer_flushes_pending_rows_on_close() -> None:
    """Catches an implementation that stops the worker thread immediately on
    close(), discarding rows still in the queue rather than draining them."""
    pool = RecordingPool()
    writer = PostgresCostLedgerWriter(pool)  # type: ignore[arg-type]
    writer.write(a_row())
    writer.close()

    inserts = [s for s in pool.conn.statements if "insert" in s.lower()]
    assert len(inserts) == 1, "close() returned before the queued row was written"
