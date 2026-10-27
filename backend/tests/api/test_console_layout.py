"""An earlier change: the data the console layout reads, on `/console/requests` and `/console/cost`.

No backing service. Each case builds the admin app from `create_app`, then replaces the two
sources on `app.state` as tests/api/test_console_requests.py does, and reuses its
`StubReader` and `PromStub`. The route, its dependencies and the error handlers stay real.

The contract these cases are written against, before it exists:

* `RequestsPage.per_day`: one `{day, requests, errors}` per window day, oldest first, every
  day present. `requests` is the ledger's count for the UTC day (`daily_requests`); `errors`
  is the 5xx count Prometheus holds for that day, `0.0` when Prometheus answered without a
  point for the day, and `None` for every day when Prometheus failed;
* `app.services.prometheus_reads.errors_per_day_query()`: one range query at a one-day step,
  from the UTC midnight after the first window day to the UTC midnight of tomorrow, so the
  point at midnight D+1 is the errors of day D;
* `RequestsPage.tokens` comes from `ConsoleReader.model_tokens(days, now)`; Prometheus is not
  asked for tokens any more;
* `CostPage.month_spend_usd`: the sum of the values `ConsoleReader.month_spend` answers,
  `None` when Postgres is down or unconfigured;
* `app.services.dashboard.JUDGE_NOTE` keeps "kappa 0.7" and says what 1 and 0 mean.
"""

from __future__ import annotations

import re
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
from app.services.dashboard import JUDGE_NOTE
from app.services.prometheus_reads import QUERY_RANGE_PATH, errors_per_day_query
from app.storage.dashboard_reads import CostRead, UnconfiguredConsoleReader

from .conftest import ADMIN_KEY, NO_PROVIDER_KEYS
from .test_console_requests import (
    DAY_SECONDS,
    ERRORS_PER_DAY,
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


def test_errors_per_day_query_is_pinned(wired: Wired) -> None:
    """Catches a changed metric, status filter or window in the query text, a step other than
    one day (the point-to-day mapping breaks), and a start or end that is not a UTC midnight
    one day after the first window day / tomorrow (every day's errors shift by one)."""
    assert errors_per_day_query() == ERRORS_PER_DAY

    wired.body(days=3)

    ranges = [params for path, params in wired.prom.calls if path == QUERY_RANGE_PATH]
    assert len(ranges) == 1
    params = ranges[0]
    assert params["query"] == ERRORS_PER_DAY
    assert float(params["step"]) == DAY_SECONDS
    today = midnight(wired.reader.daily_asked[0][1])
    assert float(params["start"]) == (today - timedelta(days=2) + timedelta(days=1)).timestamp()
    assert float(params["end"]) == (today + timedelta(days=1)).timestamp()
    assert float(params["start"]) % DAY_SECONDS == 0
    assert float(params["end"]) % DAY_SECONDS == 0


@pytest.mark.parametrize("days", [1, 90])
def test_errors_per_day_range_follows_the_window_edges(days: int, wired: Wired) -> None:
    """Catches a range that ignores `days` (always three or 30 points): at the inclusive
    edges 1 and 90 the range holds exactly `days` daily points, from the first window day's
    next midnight to tomorrow's midnight."""
    wired.body(days=days)

    params = next(p for path, p in wired.prom.calls if path == QUERY_RANGE_PATH)
    assert (float(params["end"]) - float(params["start"])) / DAY_SECONDS == days - 1
    today = midnight(wired.reader.daily_asked[0][1])
    assert float(params["end"]) == (today + timedelta(days=1)).timestamp()


def test_per_day_joins_ledger_requests_and_prometheus_errors(wired: Wired) -> None:
    """Catches errors shifted a day (the point at midnight D+1 filed under D+1 instead of D),
    requests and errors zipped in the wrong order, a day with no point left null or dropped
    instead of 0.0, a window with missing days, and a date written in another format. The
    three days differ in both columns, and the middle day has no Prometheus point."""
    wired.reader.daily = [10, 0, 7]
    wired.prom.errors = {0: 2.0, 2: 5.0}

    body = wired.body(days=3)

    today = midnight(wired.reader.daily_asked[0][1]).date()
    assert body["per_day"] == [
        {"day": (today - timedelta(days=2)).isoformat(), "requests": 10, "errors": 2.0},
        {"day": (today - timedelta(days=1)).isoformat(), "requests": 0, "errors": 0.0},
        {"day": today.isoformat(), "requests": 7, "errors": 5.0},
    ]
    assert [days for days, _ in wired.reader.daily_asked] == [3]


def test_prometheus_down_keeps_per_day_requests(wired: Wired) -> None:
    """Catches a Prometheus failure that blanks the per-day requests (they come from the
    ledger), reports errors as 0 (a failed read looks like a quiet day), or turns the page
    into an error: every errors value is null, the counts stay, the status is 200."""
    wired.reader.daily = [10, 0, 7]
    wired.prom.mode = "500"

    response = wired.get(days=3)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == {"postgres": "ok", "prometheus": "error"}
    assert [(row["requests"], row["errors"]) for row in body["per_day"]] == [
        (10, None),
        (0, None),
        (7, None),
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


def test_judge_note_explains_kappa_in_plain_words() -> None:
    """Catches the old one-line note that keeps "kappa 0.7" but never says what the number
    means, and a rewrite that drops the threshold: the note keeps "kappa 0.7" and holds a
    standalone 1 and a standalone 0 (what each end of the scale means)."""
    assert "kappa 0.7" in JUDGE_NOTE
    assert re.search(r"(?<![\d.])1(?![\d.])", JUDGE_NOTE)
    assert re.search(r"(?<![\d.])0(?![\d.])", JUDGE_NOTE)
