"""The ledger writer's retry buffer: what is sent again, what is dropped, and what is counted.

An earlier change and a design note. A write fails in one of three places, and the place decides:

* getting a connection, or the INSERT failing on a lost connection: nothing landed, so
  the row goes to a retry buffer and is sent again;
* the database refusing the row (a data or constraint error): it will never pass, so it
  is `dropped_invalid`;
* the commit failing on leaving the `with` block: the row may be in the table, so it is
  `dropped_unsure` and never sent again, or the money would be counted twice.

No database and no waiting. `ScriptedPool` plays one scripted step per `connection()` call
and keeps what it was asked to INSERT, and every case calls `flush()` instead of sleeping.
`flush()` handles every queued row, then tries every buffered row once, and returns.
Counters are process-wide and only grow, so every case compares a before and an after.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import psycopg
import psycopg.errors
import pytest
from psycopg_pool import PoolTimeout

from app.storage import CostLedgerRow, PostgresCostLedgerWriter
from app.storage.cost_ledger import LEDGER_QUEUE_SIZE

from .conftest import Pool

WHEN = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
TRACE_ID_INDEX = 13
"""Where `trace_id` sits in the INSERT's parameters, the 14th of fourteen."""

DROPPED_RESULTS = (
    "dropped_full",
    "dropped_invalid",
    "dropped_unsure",
    "dropped_shutdown",
)


def a_row(number: int = 0) -> CostLedgerRow:
    """One well-formed row, told apart from the others by its trace id."""
    return CostLedgerRow(
        ts=WHEN,
        tenant="nordlicht",
        alias="agent-default",
        provider="local",
        model="qwen2.5-7b-instruct",
        region="self",
        input_tokens=100,
        output_tokens=50,
        cached=False,
        cost_usd=Decimal("0.001500"),
        price_version="local/qwen2.5-7b-instruct@2026-01-01",
        estimated=False,
        session="sess-test01",
        trace_id=f"{number:032x}",
    )


def counter(result: str) -> float:
    """The current value of `gateway_ledger_writes_total` for one result."""
    from app.core.metrics import registry

    for family in registry().collect():
        if family.name == "gateway_ledger_writes":
            for sample in family.samples:
                if sample.labels.get("result") == result:
                    return float(sample.value)
    return 0.0


def snapshot() -> dict[str, float]:
    """Every result value this plan names, as it stands now."""
    return {
        result: counter(result) for result in ("ok", "retry", "dropped_full", *DROPPED_RESULTS[1:])
    }


def moved(before: dict[str, float]) -> dict[str, float]:
    """What moved since `before`, zero entries left out."""
    after = snapshot()
    return {key: after[key] - before[key] for key in after if after[key] != before[key]}


@dataclass
class Step:
    """What one `connection()` call does. All three `None` is a write that works."""

    enter: Exception | None = None
    execute: Exception | None = None
    exit: Exception | None = None


class ScriptedConnection:
    """The connection a step hands out: records the INSERT, or raises the step's error."""

    def __init__(self, pool: ScriptedPool, step: Step) -> None:
        self._pool = pool
        self._step = step
        self.pending: list[str] = []

    def execute(self, query: object, params: tuple[Any, ...] | None = None) -> None:
        assert params is not None
        self._pool.attempts.append(str(params[TRACE_ID_INDEX]))
        if self._step.execute is not None:
            raise self._step.execute
        self.pending.append(str(params[TRACE_ID_INDEX]))


class ScriptedPool:
    """A pool double playing one `Step` per `connection()` call, then `default` forever.

    `attempts` is every INSERT the writer issued, landed or not. `landed` is only the rows
    whose `with` block left cleanly, which is the commit: a row is in the table only then.
    """

    def __init__(self, *steps: Step, default: Step | None = None) -> None:
        self._script = deque(steps)
        self.default = default if default is not None else Step()
        self.attempts: list[str] = []
        self.landed: list[str] = []

    def take_down(self) -> None:
        """From now on no connection can be had."""
        self.default = Step(enter=PoolTimeout("couldn't get a connection"))

    def bring_up(self) -> None:
        """From now on every write works."""
        self.default = Step()

    @contextmanager
    def connection(self) -> Iterator[ScriptedConnection]:
        step = self._script.popleft() if self._script else self.default
        if step.enter is not None:
            raise step.enter
        conn = ScriptedConnection(self, step)
        yield conn
        if step.exit is not None:
            raise step.exit
        self.landed.extend(conn.pending)


@pytest.fixture
def writers() -> Iterator[list[PostgresCostLedgerWriter]]:
    """Every writer a case builds, closed at the end so no worker thread outlives it."""
    built: list[PostgresCostLedgerWriter] = []
    yield built
    for writer in built:
        writer.close()


def writer_over(
    pool: ScriptedPool, writers: list[PostgresCostLedgerWriter]
) -> PostgresCostLedgerWriter:
    """The real writer over a scripted pool, with the pool typing narrowed at one seam."""
    writer = PostgresCostLedgerWriter(cast(Pool, pool))
    writers.append(writer)
    return writer


# --------------------------------------------------------------------------
# sent again when nothing landed
# --------------------------------------------------------------------------


def test_a_row_whose_connection_failed_is_sent_again_and_lands_once(
    writers: list[PostgresCostLedgerWriter],
) -> None:
    """Catches a writer that drops a row when it cannot get a connection (the old
    behaviour: a short outage loses every row), and one that sends it again and lands it
    twice. Exactly one row in `landed`, two attempts, `retry` once and `ok` once."""
    pool = ScriptedPool(Step(enter=PoolTimeout("couldn't get a connection")))
    writer = writer_over(pool, writers)
    before = snapshot()

    writer.write(a_row(1))
    writer.flush()
    writer.flush()

    assert pool.landed == [a_row(1).trace_id]
    assert moved(before) == {"retry": 1.0, "ok": 1.0}


@pytest.mark.parametrize(
    "error",
    [psycopg.OperationalError("server closed the connection"), psycopg.InterfaceError("closed")],
    ids=["OperationalError", "InterfaceError"],
)
def test_an_insert_lost_with_its_connection_is_sent_again(
    error: Exception, writers: list[PostgresCostLedgerWriter]
) -> None:
    """Catches a writer that sorts only a failed connect as safe to resend, so an INSERT
    cut off by a lost connection (nothing committed) is dropped; and one that resends but
    lands the row twice. Two INSERTs are attempted, one lands."""
    pool = ScriptedPool(Step(execute=error))
    writer = writer_over(pool, writers)
    before = snapshot()

    writer.write(a_row(1))
    writer.flush()
    writer.flush()

    assert pool.attempts == [a_row(1).trace_id, a_row(1).trace_id]
    assert pool.landed == [a_row(1).trace_id]
    assert moved(before) == {"retry": 1.0, "ok": 1.0}


# --------------------------------------------------------------------------
# dropped when sending again is wrong or useless
# --------------------------------------------------------------------------


def test_a_failed_commit_is_dropped_unsure_and_never_sent_again(
    writers: list[PostgresCostLedgerWriter],
) -> None:
    """Catches a writer that resends a row whose commit failed: that row may already be in
    the table, so a resend counts the money twice. The error is an `OperationalError`, the
    kind that IS resent when it comes from the INSERT, so only sorting by the stage (the
    commit) and not by the exception class passes. Exactly one INSERT is ever made."""
    pool = ScriptedPool(Step(exit=psycopg.OperationalError("commit failed")))
    writer = writer_over(pool, writers)
    before = snapshot()

    writer.write(a_row(1))
    writer.flush()
    writer.flush()

    assert pool.attempts == [a_row(1).trace_id]
    assert moved(before) == {"dropped_unsure": 1.0}


@pytest.mark.parametrize(
    "error",
    [
        psycopg.errors.CheckViolation("cost_usd must not be negative"),
        psycopg.errors.NumericValueOutOfRange("value out of range"),
    ],
    ids=["CheckViolation", "NumericValueOutOfRange"],
)
def test_a_row_the_database_refuses_is_dropped_invalid(
    error: Exception, writers: list[PostgresCostLedgerWriter]
) -> None:
    """Catches a writer that retries a row the database refused for its content (it will
    never pass, and would sit in the buffer forever), and one that counts it `retry` or
    loses it unnamed. One INSERT only, and only `dropped_invalid` moves."""
    pool = ScriptedPool(Step(execute=error))
    writer = writer_over(pool, writers)
    before = snapshot()

    writer.write(a_row(1))
    writer.flush()
    writer.flush()

    assert pool.attempts == [a_row(1).trace_id]
    assert pool.landed == []
    assert moved(before) == {"dropped_invalid": 1.0}


# --------------------------------------------------------------------------
# the buffer
# --------------------------------------------------------------------------


def test_the_buffer_holds_exactly_10000_rows(
    writers: list[PostgresCostLedgerWriter], caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a buffer that is one row too small or too large (9999 drops row 10000;
    10001 keeps the row that should be dropped) and one with no bound. While down, 10000
    rows are buffered and row 10001 is `dropped_full`; once the pool works exactly those
    10000 land. The 10000 writes fit the 10000-slot input queue, and the queue is flushed
    before row 10001, so the input queue never races to full. Logging is silenced for the
    writer: a traceback per failed attempt over tens of thousands of attempts is the
    cost, not the claim."""
    caplog.set_level(logging.CRITICAL, logger="app.storage.cost_ledger")
    assert LEDGER_QUEUE_SIZE == 10000
    pool = ScriptedPool()
    pool.take_down()
    writer = writer_over(pool, writers)
    before = snapshot()

    for number in range(LEDGER_QUEUE_SIZE):
        writer.write(a_row(number))
    writer.flush()
    assert moved(before) == {"retry": 10000.0}

    writer.write(a_row(LEDGER_QUEUE_SIZE))
    writer.flush()
    assert moved(before) == {"retry": 10000.0, "dropped_full": 1.0}

    pool.bring_up()
    writer.flush()

    assert pool.landed == [a_row(number).trace_id for number in range(LEDGER_QUEUE_SIZE)]
    assert moved(before) == {"retry": 10000.0, "dropped_full": 1.0, "ok": 10000.0}


def test_buffered_rows_land_in_the_order_written(writers: list[PostgresCostLedgerWriter]) -> None:
    """Catches a buffer that sends rows back in the wrong order (a stack, or a set), which
    would put a later row's timestamp ahead of an earlier one in the table. A fourth row
    written after the pool is back also queues behind the buffer, not in front of it."""
    pool = ScriptedPool()
    pool.take_down()
    writer = writer_over(pool, writers)
    for number in (1, 2, 3):
        writer.write(a_row(number))
    writer.flush()
    assert pool.landed == []

    pool.bring_up()
    writer.write(a_row(4))
    writer.flush()

    assert pool.landed == [a_row(number).trace_id for number in (1, 2, 3, 4)]


def test_the_retry_wait_doubles_from_one_second_to_a_thirty_second_cap() -> None:
    """Catches a wait that never grows (a tight retry loop against a database that is
    down), one that grows without a cap, and a cap of any other value (a range would admit
    it, so the exact sequence is pinned): 1, 2, 4, 8, 16, 30, 30."""
    from app.storage.cost_ledger import (
        LEDGER_RETRY_WAIT_CAP,
        LEDGER_RETRY_WAIT_START,
        next_retry_wait,
    )

    assert LEDGER_RETRY_WAIT_START == 1.0
    assert LEDGER_RETRY_WAIT_CAP == 30.0
    waits: list[float] = []
    current: float | None = None
    for _ in range(7):
        current = next_retry_wait(current)
        waits.append(current)

    assert waits == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_close_counts_rows_still_buffered_as_dropped_shutdown(
    writers: list[PostgresCostLedgerWriter],
) -> None:
    """Catches a close that discards buffered rows without a count (the quiet loss this
    plan removes), one that counts them `retry` or `dropped_full`, and one that hangs
    waiting for a database that is down. Two rows, pool down, `dropped_shutdown` +2."""
    pool = ScriptedPool()
    pool.take_down()
    writer = writer_over(pool, writers)
    before = snapshot()
    writer.write(a_row(1))
    writer.write(a_row(2))
    writer.flush()

    writer.close()

    assert moved(before) == {"retry": 2.0, "dropped_shutdown": 2.0}
    assert pool.landed == []


def test_close_sends_buffered_rows_once_before_giving_up(
    writers: list[PostgresCostLedgerWriter],
) -> None:
    """Catches a close that counts buffered rows `dropped_shutdown` without trying them
    again: a database that came back just before shutdown would lose rows it could take."""
    pool = ScriptedPool()
    pool.take_down()
    writer = writer_over(pool, writers)
    writer.write(a_row(1))
    writer.flush()
    before = snapshot()
    pool.bring_up()

    writer.close()

    assert pool.landed == [a_row(1).trace_id]
    assert moved(before) == {"ok": 1.0}


def test_a_working_database_moves_no_dropped_counter(
    writers: list[PostgresCostLedgerWriter],
) -> None:
    """Catches a writer that counts a dropped value on the happy path (which would fire
    the `LedgerWriteDropped` alert on healthy traffic). Five rows land; only `ok` moves."""
    pool = ScriptedPool()
    writer = writer_over(pool, writers)
    before = snapshot()

    for number in range(5):
        writer.write(a_row(number))
    writer.flush()

    assert pool.landed == [a_row(number).trace_id for number in range(5)]
    assert moved(before) == {"ok": 5.0}
    assert all(counter(result) == before[result] for result in DROPPED_RESULTS)
