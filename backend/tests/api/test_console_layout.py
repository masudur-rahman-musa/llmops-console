"""The data the console layout reads, on `/console/requests` and `/console/cost`.

No backing service. Each case builds the admin app from `create_app`, then replaces the two
sources on `app.state` as tests/api/test_console_requests.py does, and reuses its
`StubReader` and `PromStub`. The route, its dependencies and the error handlers stay real.

The contract these cases are written against, before it exists:

* `RequestsPage.per_day`: one `{day, requests, errors}` per window day, oldest first, every
  day present. `requests` is the ledger's answered count for the UTC day and `errors` its
  status 500 and above count (`daily_requests`); both are whole numbers whatever
  Prometheus does;
* `RequestsPage.tokens` comes from `ConsoleReader.model_tokens(days, now)`; Prometheus is not
  asked for tokens any more;
* `CostPage.month_spend_usd`: the sum of the values `ConsoleReader.month_spend` answers,
  `None` when Postgres is down or unconfigured;
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.limits.prices import PriceTable
from app.main import create_app
from app.schemas.dashboard import ModelTokens
from app.storage.dashboard_reads import CostRead, UnconfiguredConsoleReader

from .conftest import ADMIN_KEY, NO_PROVIDER_KEYS
from .test_console_requests import (
    LEDGER_TOKENS,
    TOKENS_7D,
    PromStub,
    StubReader,
    Wired,
)

REQUESTS = "/console/requests"
COST = "/console/cost"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}


@pytest.fixture
def wired(admin_settings: Settings) -> Iterator[Wired]:
    """The admin app with the requests stubs swapped in after the lifespan has run."""
    app = create_app(admin_settings, NO_PROVIDER_KEYS)
    reader = StubReader()
    prom = PromStub()
    with TestClient(app) as client:
        app.state.console_reader = reader
        app.state.prometheus = prom.client()
        yield Wired(client, reader, prom)


def midnight(moment: datetime) -> datetime:
    """UTC midnight of the day `moment` falls on."""
    return datetime(moment.year, moment.month, moment.day, tzinfo=UTC)


def test_per_day_joins_ledger_answered_and_failed(wired: Wired) -> None:
    """Catches answered and failed zipped in the wrong order, a failed count taken from
    Prometheus (its answer here differs: 2.0 and 5.0), a day dropped or left null, a window
    with missing days, and a date written in another format. The three days differ in both
    columns."""
    wired.reader.daily = [(10, 1), (0, 4), (7, 0)]
    wired.prom.errors = {0: 2.0, 2: 5.0}

    body = wired.body(days=3)

    today = midnight(wired.reader.daily_asked[0][1]).date()
    assert body["per_day"] == [
        {"day": (today - timedelta(days=2)).isoformat(), "requests": 10, "errors": 1},
        {"day": (today - timedelta(days=1)).isoformat(), "requests": 0, "errors": 4},
        {"day": today.isoformat(), "requests": 7, "errors": 0},
    ]
    assert [days for days, _ in wired.reader.daily_asked] == [3]


def test_prometheus_down_keeps_per_day_requests(wired: Wired) -> None:
    """Catches a Prometheus failure that blanks the per-day counts (both come from the
    ledger), nulls the errors (they no longer depend on Prometheus), or turns the page into
    an error: the counts stay, the status is 200."""
    wired.reader.daily = [(10, 2), (0, 0), (7, 5)]
    wired.prom.mode = "500"

    response = wired.get(days=3)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": "ok", "prometheus": "error"}
    assert [(row["requests"], row["errors"]) for row in body["per_day"]] == [
        (10, 2),
        (0, 0),
        (7, 5),
    ]


def test_postgres_down_gives_no_per_day_and_no_tokens(wired: Wired) -> None:
    """Catches a dead ledger that leaves stale days or tokens in the body, invents zero days
    from Prometheus alone, or fails the page: both are empty, Prometheus answered, the
    status is 200."""
    wired.reader.error = psycopg.OperationalError("connection refused")

    response = wired.get(days=3)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": "error", "prometheus": "ok"}
    assert body["per_day"] == []
    assert body["tokens"] == []


def test_tokens_come_from_the_ledger_not_prometheus(wired: Wired) -> None:
    """Catches a route that still sends the Prometheus tokens query (whose answer here
    differs from the ledger's), merges the two sources, or ignores the reader: the reader's
    rows come back as they are, the tokens query is never sent, and the read covers the
    asked window."""
    wired.reader.tokens = (ModelTokens(model="ledger-model", input=111.0, output=22.0),)

    body = wired.body(days=7)

    assert body["tokens"] == [
        {
            "model": "ledger-model",
            "input": 111.0,
            "output": 22.0,
            "cache_read": 0.0,
            "cache_write": 0.0,
        }
    ]
    assert not any("gateway_tokens_total" in query for query in wired.prom.queries)
    assert TOKENS_7D not in wired.prom.queries
    assert [days for days, _ in wired.reader.tokens_asked] == [7]
    assert wired.reader.tokens_asked[0][1].tzinfo is not None


def test_tokens_read_the_ledger_when_prometheus_is_down(wired: Wired) -> None:
    """Catches tokens that still ride on the Prometheus read and vanish with it: Prometheus
    fails, the ledger rows are still in the body."""
    wired.prom.mode = "500"

    body = wired.body(days=7)

    assert body["tokens"] == [t.model_dump() for t in LEDGER_TOKENS]


# --- /console/cost -------------------------------------------------------------------

EMPTY_COST = CostRead(days=(), savings=(), savings_usd_total=0.0, cost_per_request=())


class CostReader:
    """A configured reader for the cost route: no days and a chosen month spend."""

    def __init__(self, month_spend: dict[str, float]) -> None:
        self.configured = True
        self.month = month_spend
        self.error: Exception | None = None

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
        if self.error is not None:
            raise self.error
        return EMPTY_COST

    def month_spend(self, *, tenant: str | None, now: datetime) -> dict[str, float]:
        if self.error is not None:
            raise self.error
        return dict(self.month)


def get_cost(admin_settings: Settings, reader: Any) -> dict[str, Any]:  # noqa: ANN401
    """The cost page body with `reader` swapped in after startup."""
    app = create_app(admin_settings, NO_PROVIDER_KEYS)
    with TestClient(app) as client:
        app.state.console_reader = reader
        response = client.get(COST, headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_month_spend_sums_every_tenant(admin_settings: Settings) -> None:
    """Catches a total of one tenant only (the first, the largest), a total of the budgets
    list instead of the reader's answer, the sum left as a per-tenant map, and a sum that
    drops the smallest tenant: 31.00 + 25.00 + 5.00 is 61.00."""
    reader = CostReader({"nordlicht": 31.0, "cuan": 25.0, "evals": 5.0})

    body = get_cost(admin_settings, reader)

    assert body["month_spend_usd"] == pytest.approx(61.0)


def test_month_spend_is_null_when_postgres_is_down(admin_settings: Settings) -> None:
    """Catches a failed ledger read shown as 0.0 (a month with no spend), and a missing
    field: a dead ledger reads null, the page is still 200."""
    reader = CostReader({"nordlicht": 31.0})
    reader.error = psycopg.OperationalError("connection refused")

    body = get_cost(admin_settings, reader)

    assert body["sources"]["postgres"] == "error"
    assert body["month_spend_usd"] is None


def test_month_spend_is_null_when_postgres_is_unconfigured(admin_settings: Settings) -> None:
    """Catches an unconfigured ledger shown as 0.0 spend: the field is present and null."""
    body = get_cost(admin_settings, UnconfiguredConsoleReader())

    assert body["sources"]["postgres"] == "unconfigured"
    assert "month_spend_usd" in body
    assert body["month_spend_usd"] is None
