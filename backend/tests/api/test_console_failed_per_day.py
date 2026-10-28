"""The Failed (5xx) line on `/console/requests` is counted from the cost ledger.

No backing service. The admin app is built as tests/api/test_console_layout.py builds it, with
`StubReader` and `PromStub` swapped in after startup. The stub Prometheus answers fractions
(13.0128, 1.08) for any range query, so a route that still reads them shows up as a wrong or
non-whole `errors`.

The contract these cases are written against, before it exists:

* `ConsoleReader.daily_requests(days, now)` answers `(day, answered, failed)` per UTC day;
* `RequestsPage.per_day[i].errors` is the `failed` of day i, a whole number, never null;
* the route sends no range query to Prometheus.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.prometheus_reads import QUERY_RANGE_PATH

from .conftest import NO_PROVIDER_KEYS
from .test_console_requests import PromStub, StubReader, Wired


@pytest.fixture
def wired(admin_settings: Settings) -> Iterator[Wired]:
    """The admin app with the requests stubs swapped in, Prometheus holding fractions."""
    app = create_app(admin_settings, NO_PROVIDER_KEYS)
    reader = StubReader()
    prom = PromStub()
    prom.errors = {0: 13.012805401607991, 1: 1.0833333333333335}
    with TestClient(app) as client:
        app.state.console_reader = reader
        app.state.prometheus = prom.client()
        yield Wired(client, reader, prom)


def test_failed_per_day_comes_from_the_ledger(wired: Wired) -> None:
    """Catches a route that still fills `errors` from the Prometheus estimate (fractions here),
    one that returns the ledger count as a float (3.0), and one that swaps answered and
    failed: the reader days (10, 0), (0, 3), (7, 0) give errors 0, 3, 0, all ints."""
    wired.reader.daily = [(10, 0), (0, 3), (7, 0)]

    body = wired.body(days=3)

    errors = [row["errors"] for row in body["per_day"]]
    assert errors == [0, 3, 0]
    assert all(type(value) is int for value in errors)
    assert [row["requests"] for row in body["per_day"]] == [10, 0, 7]


def test_no_range_query_is_sent(wired: Wired) -> None:
    """Catches a route that still asks Prometheus for errors per day and then discards or
    merges the answer: no request to the range path is sent at all."""
    wired.body(days=3)

    assert [path for path, _ in wired.prom.calls if path == QUERY_RANGE_PATH] == []


def test_prometheus_down_keeps_failed_counts(wired: Wired) -> None:
    """Catches a Prometheus failure that nulls or zeroes the failed counts, or fails the page:
    Prometheus answers 500, every errors value is still the ledger's whole number, the status
    is 200."""
    wired.reader.daily = [(10, 0), (0, 3), (7, 0)]
    wired.prom.mode = "500"

    response = wired.get(days=3)

    assert response.status_code == 200, response.text
    errors = [row["errors"] for row in response.json()["per_day"]]
    assert errors == [0, 3, 0]
    assert all(type(value) is int for value in errors)
