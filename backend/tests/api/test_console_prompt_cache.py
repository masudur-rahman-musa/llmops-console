"""The prompt cache fields of `/console/requests` and `/console/cost`.

No backing service: the admin app from `create_app` with the reader replaced by a stub, as
tests/api/test_console_requests.py does, and a Prometheus that answers empty.

The contract these cases are written against, before it exists:

* `ModelTokens.cache_read` and `.cache_write`; `RequestsPage.prompt_cache_share` is cache
  reads over plain input plus reads plus writes, null when that total is 0 or the ledger
  is down;
* `CostRead.prompt_cache_saved_usd`; `CostPage.prompt_cache_saved_usd` carries it, null
  when the ledger is down.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.limits.prices import PriceTable
from app.main import create_app
from app.schemas.dashboard import ModelTokens
from app.services.prometheus_reads import PrometheusClient
from app.storage.dashboard_reads import CostRead, RequestFilters, RequestsRead

from .conftest import ADMIN_KEY, NO_PROVIDER_KEYS

REQUESTS = "/console/requests"
COST = "/console/cost"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}

MIXED = (
    ModelTokens(model="claude-haiku-5-5", input=100.0, output=10.0, cache_read=300.0),
    ModelTokens(model="gpt-5-mini", input=50.0, output=10.0, cache_write=50.0),
)


class StubReader:
    """A configured reader answering fixed reads; `error` is raised by every method."""

    def __init__(self) -> None:
        self.configured = True
        self.error: Exception | None = None
        self.tokens: tuple[ModelTokens, ...] = ()
        self.saved = 0.0

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    def cost(
        self,
        *,
        days: int,
        tenant: str | None,
        now: datetime,
        prices: PriceTable,
        start: date | None = None,
        end: date | None = None,
    ) -> CostRead:
        self._check()
        return CostRead(
            days=(),
            savings=(),
            savings_usd_total=0.0,
            cost_per_request=(),
            prompt_cache_saved_usd=self.saved,
        )

    def month_spend(self, *, tenant: str | None, now: datetime) -> dict[str, float]:
        self._check()
        return {}

    def recent_requests(
        self, *, page: int, filters: RequestFilters, days: int, now: datetime
    ) -> RequestsRead:
        self._check()
        return RequestsRead(rows=(), page=1, pages=1)

    def cache_hit_rate(self, days: int, now: datetime) -> float | None:
        self._check()
        return None

    def daily_requests(self, days: int, now: datetime) -> tuple[tuple[date, int, int], ...]:
        self._check()
        return ()

    def model_tokens(self, days: int, now: datetime) -> tuple[ModelTokens, ...]:
        self._check()
        return self.tokens


def empty_prometheus(request: httpx.Request) -> httpx.Response:
    result_type = "matrix" if request.url.path.endswith("query_range") else "vector"
    return httpx.Response(
        200, json={"status": "success", "data": {"resultType": result_type, "result": []}}
    )


class Wired:
    def __init__(self, client: TestClient, reader: StubReader) -> None:
        self.client = client
        self.reader = reader

    def body(self, path: str) -> dict[str, Any]:
        response = self.client.get(path, headers=ADMIN_HEADERS)
        assert response.status_code == 200, response.text
        parsed: dict[str, Any] = response.json()
        return parsed


@pytest.fixture
def wired(admin_settings: Settings) -> Iterator[Wired]:
    app = create_app(admin_settings, NO_PROVIDER_KEYS)
    reader = StubReader()
    with TestClient(app) as client:
        app.state.console_reader = reader
        app.state.prometheus = PrometheusClient(
            httpx.Client(
                base_url="http://prometheus.test",
                transport=httpx.MockTransport(empty_prometheus),
            )
        )
        yield Wired(client, reader)


def test_the_share_is_reads_over_all_input(wired: Wired) -> None:
    """Catches a share of reads over plain input only (3.0), over input plus reads (0.75), or
    one that leaves out the write tokens: 300 over 100 + 300 + 0 + 50 + 0 + 50 is 0.6."""
    wired.reader.tokens = MIXED

    assert wired.body(REQUESTS)["prompt_cache_share"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "tokens",
    [
        (),
        (ModelTokens(model="claude-haiku-5-5", input=0.0, output=5.0),),
    ],
    ids=["no rows", "rows all zero"],
)
def test_no_input_is_a_null_share(wired: Wired, tokens: tuple[ModelTokens, ...]) -> None:
    """Catches a share that divides by zero (a 500) or answers 0 for no input: no traffic is
    not a 0% share, it is null."""
    wired.reader.tokens = tokens

    assert wired.body(REQUESTS)["prompt_cache_share"] is None


def test_a_dead_ledger_nulls_both_fields(wired: Wired) -> None:
    """Catches a route that answers 0 or leaves out either new field when Postgres is down:
    both are null beside a 200 that flags the postgres source."""
    wired.reader.error = psycopg.OperationalError("connection refused")

    requests = wired.body(REQUESTS)
    cost = wired.body(COST)

    assert requests["sources"]["postgres"] == "error"
    assert requests["prompt_cache_share"] is None
    assert cost["sources"]["postgres"] == "error"
    assert cost["prompt_cache_saved_usd"] is None


def test_the_cost_body_carries_the_saving(wired: Wired) -> None:
    """Catches a cost route that drops the reader's saving or answers a fixed number: 0.25 in
    the reader is 0.25 in the body."""
    wired.reader.saved = 0.25

    assert wired.body(COST)["prompt_cache_saved_usd"] == 0.25
