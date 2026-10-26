"""The ledger row carries a status and a latency, and a failed request leaves a row.

No backing service. A real app from `create_app` with two stub providers, the in-memory ledger,
and a fake clock that only moves when a stub provider moves it. The clock reaches the pipeline
through `app.dependency_overrides[get_clients]`, which hands the route the same `Clients` with
`clock` replaced. Nothing sleeps.

The contract these cases are written against, before it exists:

* `Clients.clock: Callable[[], float]` (seconds, monotonic), read once when `complete_chat`
  starts and again when a row is booked;
* a served request books `status=200` and `latency_ms = round((booking - start) * 1000)`;
  for a stream the booking moment is the usage event reaching the token sink;
* once the alias has resolved, a `GatewayError` whose `http_status` is 500 or above books ONE
  failed row (`book_failure`): that status, cost 0, tokens 0, not cached, the first
  candidate's provider and model. It debits no budget, no tokens and no cost counter;
* an answer below 500 books nothing, as before.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.api.deps import get_clients
from app.cache import InProcessResponseCache
from app.core.config import Settings
from app.core.errors import ProviderError
from app.limits.budget import InProcessBudgetStore
from app.limits.ratelimit import InProcessRateLimitStore, RateDimension, Taken
from app.providers.base import Adapter, ChatRequest, ChatResponse, StreamEvent, UsageTotal
from app.services.chat_accounting import Clients
from app.storage import InMemoryCostLedgerWriter

from .conftest import (
    CHAT,
    FALLBACK_ALIAS,
    FALLBACK_CLASS,
    FALLBACK_TENANT,
    FIRST_PROVIDER,
    SECOND_PROVIDER,
    TEST_KEYS,
    FallbackStubAdapter,
    MetricDelta,
    SilentSemanticCache,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    fresh_breaker,
    retryable_error,
)

COST_USD = "gateway_cost_usd_total"
FIRST_MODEL = "claude-haiku-5-5"
"""What the shipped policy resolves `evals`/`agent-default`/`internal` to first."""

START = 10.0
"""Where the fake clock begins: the moment `complete_chat` starts."""


class FakeClock:
    """A clock that moves only when a test or a stub provider says so."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


class TimedAdapter(FallbackStubAdapter):
    """The fallback stub, which spends fake time when it is called.

    A chat call spends `chat_spends` seconds (a failing call as well). A stream spends
    `before_usage` seconds just before its usage event is yielded and `after_usage` just
    after, so a latency read at the first frame, at the usage event or at the close of the
    stream are three different numbers.
    """

    def __init__(self, *, name: str, clock: FakeClock) -> None:
        super().__init__(name=name)
        self.clock = clock
        self.chat_spends = 0.0
        self.before_usage = 0.0
        self.after_usage = 0.0

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.clock.now += self.chat_spends
        return super().chat(request)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        for event in self.events:
            if isinstance(event, UsageTotal):
                self.clock.now += self.before_usage
                yield event
                self.clock.now += self.after_usage
            else:
                yield event


class RecordingBudgetStore:
    """The in-process counter with every debit kept."""

    def __init__(self) -> None:
        self.inner = InProcessBudgetStore()
        self.debits: list[tuple[str, str, int]] = []

    def spend(self, tenant: str, month: str) -> int | None:
        return self.inner.spend(tenant, month)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        self.debits.append((tenant, month, micro_usd))
        return self.inner.debit(tenant, month, micro_usd)


class RecordingRateStore:
    """The in-process buckets with every take kept; `refuse` turns every take into a 429."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.inner = InProcessRateLimitStore(clock=lambda: 1_800_000_000.0)
        self.refuse = refuse
        self.takes: list[tuple[RateDimension, int]] = []

    def take(
        self, tenant: str, dimension: RateDimension, *, cost: int, capacity: int, rate: float
    ) -> Taken | None:
        self.takes.append((dimension, cost))
        if self.refuse:
            return Taken(allowed=False, retry_after_seconds=7)
        return self.inner.take(tenant, dimension, cost=cost, capacity=capacity, rate=rate)


class World:
    """Everything one case arranges and reads."""

    def __init__(
        self,
        client: TestClient,
        clock: FakeClock,
        first: TimedAdapter,
        second: TimedAdapter,
        ledger: InMemoryCostLedgerWriter,
        budget: RecordingBudgetStore,
        rates: RecordingRateStore,
    ) -> None:
        self.client = client
        self.clock = clock
        self.first = first
        self.second = second
        self.ledger = ledger
        self.budget = budget
        self.rates = rates

    def post(self, **overrides: Any) -> Any:  # noqa: ANN401
        """One chat request as `evals` under `internal`."""
        return self.client.post(
            CHAT,
            json=chat_body(**overrides),
            headers=auth_headers(
                key=TEST_KEYS[FALLBACK_TENANT],
                tenant=FALLBACK_TENANT,
                data_class=FALLBACK_CLASS,
            ),
        )


@pytest.fixture(scope="module")
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(scope="module")
def adapters(clock: FakeClock) -> dict[str, TimedAdapter]:
    return {
        FIRST_PROVIDER: TimedAdapter(name=FIRST_PROVIDER, clock=clock),
        SECOND_PROVIDER: TimedAdapter(name=SECOND_PROVIDER, clock=clock),
    }


@pytest.fixture(scope="module")
def status_app(settings: Settings, adapters: dict[str, TimedAdapter]) -> Iterator[TestClient]:
    """One app for the module, built once."""
    stubs: dict[str, Adapter] = dict(adapters)
    with build_chat_client(settings, stubs) as client:
        yield client


@pytest.fixture
def world(
    status_app: TestClient, clock: FakeClock, adapters: dict[str, TimedAdapter]
) -> Iterator[World]:
    """Reset stubs, clock, ledger, counters and caches; route the fake clock into `Clients`."""
    clock.now = START
    for adapter in adapters.values():
        adapter.reset()
        adapter.chat_spends = 0.0
        adapter.before_usage = 0.0
        adapter.after_usage = 0.0
    app = app_of(status_app)
    ledger = InMemoryCostLedgerWriter()
    budget = RecordingBudgetStore()
    rates = RecordingRateStore()
    app.state.cost_ledger = ledger
    app.state.budget_store = budget
    app.state.rate_limit_store = rates
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = SilentSemanticCache()
    fresh_breaker(status_app)

    def with_fake_clock(request: Request) -> Clients:
        return replace(get_clients(request), clock=clock)

    app.dependency_overrides[get_clients] = with_fake_clock
    try:
        yield World(
            status_app,
            clock,
            adapters[FIRST_PROVIDER],
            adapters[SECOND_PROVIDER],
            ledger,
            budget,
            rates,
        )
    finally:
        app.dependency_overrides.pop(get_clients, None)


def test_a_served_request_books_200_and_its_latency(world: World) -> None:
    """Catches a row booked with a status other than 200 on success, a latency left null,
    a latency in seconds or in microseconds instead of milliseconds, and a latency read from
    the wall clock instead of `clients.clock`: the clock reads 10.0 at the start and 10.412
    when the provider has answered, so the row must say 412."""
    world.first.chat_spends = 0.412

    response = world.post()

    assert response.status_code == 200, response.text
    assert [(row.status, row.latency_ms) for row in world.ledger.rows] == [(200, 412)]


def test_a_stream_books_latency_at_its_end(world: World) -> None:
    """Catches a stream whose latency is read when the first frame is on its way (0 here),
    or when the stream closes (the clock moves again after the usage event: 1500 here), instead
    of when the usage arrives at the token sink: the clock reads 10.9 at that moment."""
    world.first.before_usage = 0.9
    world.first.after_usage = 1.5

    response = world.post(stream=True)

    assert response.status_code == 200, response.text
    assert [(row.status, row.latency_ms) for row in world.ledger.rows] == [(200, 900)]


def test_every_candidate_failing_books_one_failed_row(world: World, metrics: MetricDelta) -> None:
    """Catches a failure that books no row, one row per attempt (two here), a row whose status
    is not the response's, a row naming the LAST candidate that failed instead of the first,
    a row priced or carrying tokens, and a failed row that still moves the budget counter, the
    token bucket or the cost counter. Both providers fail and each spends 0.25 s, so the row is
    booked after both and says 500."""
    world.first.chat_error = retryable_error(FIRST_PROVIDER)
    world.second.chat_error = retryable_error(SECOND_PROVIDER)
    world.first.chat_spends = 0.25
    world.second.chat_spends = 0.25

    response = world.post()

    assert response.status_code == 502, response.text
    assert world.first.call_count == 1
    assert world.second.call_count == 1
    assert len(world.ledger.rows) == 1
    row = world.ledger.rows[0]
    assert row.status == response.status_code
    assert row.latency_ms == 500
    assert row.cost_usd == Decimal("0")
    assert (row.input_tokens, row.output_tokens) == (0, 0)
    assert row.cached is False
    assert (row.tenant, row.alias) == (FALLBACK_TENANT, FALLBACK_ALIAS)
    assert (row.provider, row.model) == (FIRST_PROVIDER, FIRST_MODEL)
    assert world.budget.debits == []
    assert metrics.total(COST_USD) == 0.0
    # Only the two probes of the check at the top of the request; no debit of tokens.
    assert world.rates.takes == [(RateDimension.REQUESTS, 1), (RateDimension.TOKENS, 1)]


def test_a_content_refusal_books_a_502_row(world: World) -> None:
    """Catches a failure path that books only when every candidate was tried: a content
    refusal stops at the first candidate (the second is never dialled) and is still a 502 the
    caller saw, so exactly one row with status 502 must exist."""
    world.first.chat_error = ProviderError(
        "anthropic: the provider refused the request for content policy", refused=True
    )

    response = world.post()

    assert response.status_code == 502, response.text
    assert world.second.call_count == 0
    assert [(row.status, row.provider) for row in world.ledger.rows] == [(502, FIRST_PROVIDER)]


def test_a_429_books_no_row(world: World) -> None:
    """Unhappy path. Catches a failure booking that runs for a refusal at the top of the
    request: a rate-limited request leaves the ledger empty and dials no provider."""
    world.rates.refuse = True

    response = world.post()

    assert_error_body(response, status=429, code="RATE_LIMITED")
    assert world.ledger.rows == ()
    assert world.first.call_count == 0


def test_a_422_after_the_alias_resolved_books_no_row(world: World) -> None:
    """Unhappy path. Catches a failure booking that fires for any error after the alias
    resolved (the threshold must be 500, not 400): a 422 is refused after resolution, and
    books nothing. The message is over the redaction cap, which `confidential` enforces."""
    response = world.client.post(
        CHAT,
        json=chat_body(messages=[{"role": "user", "content": "x" * 20000}]),
        headers=auth_headers(
            key=TEST_KEYS[FALLBACK_TENANT], tenant=FALLBACK_TENANT, data_class="confidential"
        ),
    )

    assert response.status_code == 422, response.text
    assert world.ledger.rows == ()


def test_the_hook_is_a_plain_callable() -> None:
    """Catches a `Clients.clock` default that is not the monotonic clock: the shipped default
    must be `time.monotonic` itself, so latency never follows the wall clock."""

    fields = {f.name: f for f in dataclasses.fields(Clients)}
    assert list(fields)[-1] == "clock"
    default: object = fields["clock"].default
    assert default is time.monotonic
