"""The monthly budget through the route: the 429, the gauge, the warning and the debit.

Everything here is settled by a real request through `TestClient` against an app
`create_app` built, because the question is never "does
`decide` answer `reject` at the cap" - tests/limits/test_budget.py settles that -
but "is the request refused with the typed body every other refusal carries,
before a provider is dialled, and does every path that books a ledger row debit
the same number".

No backing service. The app is built with no Redis URL, so
`app.state.budget_store` is the in-process twin; every case here wraps it in
`RecordingBudgetStore`, which is the only way to tell "debited zero" (what a
cache hit owes) from "did not debit at all" (a call site that was missed). The
store is reached by NAME, `app.state.budget_store`, never by shape: this app
already holds several collaborators with a `get`-and-`put` pair, and selecting
one of them by its methods is how the wrong one gets picked.

One app for the module, built once. Per-test isolation is a
fresh `budget_store`, `cost_ledger`, `response_cache` and `semantic_cache` on
`app.state` and the stubs' recorded calls cleared - never a new app.

Three arrangements are worth reading before the assertions:

* THE SPEND IS ARRANGED THROUGH THE STORE, not by sending requests until the cap
  is reached. Nine hundred requests to reach 10.00 would be a slow suite and
  would prove nothing the one-line arrangement does not.
* THE MEASUREMENT IS ARMED IMMEDIATELY BEFORE THE REQUEST BEING MEASURED. A
  cache-hit sender primes the cache first, and the priming request books a row
  and a debit of its own; `arm` replaces the ledger and clears the recorded
  debits after it, so "exactly one debit" is a statement about ONE request.
* THE EMBEDDING ADAPTER IS LIVE ON THIS APP. `agent-default` carries
  `semantic_cache: true`, so a chat request here really does embed through the
  `embed` alias against `local`; that is what makes the semantic-hit path
  reachable, and it is also what would expose an implementation that debited the
  internal embedding as a second charge.

The contract these cases are written against, before it exists:

* `create_app` builds `app.state.budget_store` - `RedisBudgetStore` with a Redis
  URL, `InProcessBudgetStore` without one;
* both routes check the tenant's spend against its `budget_usd_month` after
  `Context` resolves and BEFORE anything is resolved, redacted or dialled, raising
  `BudgetExceeded` for `BudgetDecision.REJECT`;
* the check logs one warning through `app.limits.budget` in the warn
  band, carrying `tenant`, `spend_usd` and `cap_usd`;
* every `book_cost` call site debits the counter by `to_micro_usd` of the
  `cost_usd` of the row it just booked, exactly once, zero included;
* `gateway_budget_ratio{tenant}` is set from the total `debit` answers, so it
  includes the request that has just been served.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal
from typing import Any, cast

import pytest
import redis
from fastapi.testclient import TestClient
from httpx import Response
from prometheus_client import CollectorRegistry

from app.cache import InProcessResponseCache, InProcessSemanticCache
from app.core.config import Settings
from app.limits.budget import (
    InProcessBudgetStore,
    RedisBudgetStore,
    SpendLoader,
    current_month,
    to_micro_usd,
)
from app.main import create_app
from app.model_routing.policy import Policy
from app.providers.base import Adapter
from app.services.chat_headers import CACHE_STATUS_HIT, CACHE_STATUS_MISS, CACHE_STATUS_SEMANTIC
from app.storage import CostLedgerRow, CostLedgerWriter, InMemoryCostLedgerWriter

from .conftest import (
    API_KEYS,
    CHAT,
    EMBEDDINGS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    EmbeddingStubAdapter,
    MetricDelta,
    StubAdapter,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
    stream_events,
)

BUDGET_RATIO = "gateway_budget_ratio"
"""The family name as the exposition spells it, written out rather than imported
from `core/metrics.py`: it is a dashboard contract, and a test that read the
module's own constant would follow a rename instead of failing on it."""

REQUESTS = "gateway_requests_total"

BUDGET_LOGGER = "app.limits.budget"
"""Where the warning comes from. Named so the assertion counts THIS module's
records and not whatever else the request logged."""

TENANT = "evals"
DATA_CLASS = "internal"
SESSION = "sess-budget-enforcement"
ALIAS = "agent-default"
PROVIDER = "anthropic"

EMBED_ALIAS = "embed"
EMBED_PROVIDER = "local_embed"

UNPOLICED_TENANT = "cap-key"
"""A tenant with a key in the store and no entry in `routing.yaml`. The budget
check runs before the resolver, so this is the tenant that tells a check which
looks its cap up with `policy.tenants[tenant]` from one that does not."""

CAP_USD = Decimal("40.00")
CAP = 40_000_000
"""`evals`'s shipped `budget_usd_month`, in dollars and in the counter's unit.
Asserted against the loaded policy by the first case below, so an edit to
`routing.yaml` fails there rather than making every other case here meaningless."""

WARN_AT = 32_000_000
"""Exactly 80% of `CAP`: the inclusive warn boundary, in micro-USD."""

CHAT_MICRO = 28
"""(137 * 0.10 + 29 * 0.50) / 1_000_000 in micro-USD, rounded to six places, from
`STUB_USAGE` and the shipped `anthropic/claude-haiku-5-5` price, written out as arithmetic
rather than recomputed from the table the implementation reads."""

STREAM_MICRO = 69
"""(431 * 0.10 + 52 * 0.50) / 1_000_000, rounded to six places, from `STREAM_USAGE`. Different from
`CHAT_MICRO` in both directions, so a streamed request debited from the
non-streamed usage is a different number here."""

ZERO_MICRO = 0
"""What a cache hit debits, and what a zero-priced `local` row debits. It is a
recorded call carrying zero, never an absent call - which is the whole reason
this module records debits instead of reading a total."""

COST_HEADER = "x-gateway-cost-usd"
CACHE_HEADER = "x-gateway-cache"

QUESTION = "What is the capital of Germany?"
SAME_QUESTION_REWORDED = "Which city is the capital city of Germany?"


# --------------------------------------------------------------------------
# the seams
# --------------------------------------------------------------------------


class RecordingBudgetStore:
    """The in-process twin, with every call kept.

    `debits` is what separates "this path debited zero" from "this path never
    debited", which no total can express and which is exactly the failure:
    a `book_cost` call site that was missed leaves budgets
    permanently under-reported, and on the three paths whose cost is zero it
    leaves no trace at all.
    """

    def __init__(self) -> None:
        self.inner = InProcessBudgetStore()
        self.debits: list[tuple[str, str, int]] = []
        self.reads: list[tuple[str, str]] = []

    def spend(self, tenant: str, month: str) -> int | None:
        self.reads.append((tenant, month))
        return self.inner.spend(tenant, month)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        self.debits.append((tenant, month, micro_usd))
        return self.inner.debit(tenant, month, micro_usd)

    def seeded_spend(self, tenant: str, month: str, loader: SpendLoader) -> int | None:
        self.reads.append((tenant, month))
        return self.inner.seeded_spend(tenant, month, loader)

    def arrange(self, tenant: str, month: str, micro_usd: int) -> None:
        """Put a month's spend on the counter without recording it as a debit."""
        self.inner.debit(tenant, month, micro_usd)


class UnreachableRedis:
    """A client whose every command raises, for the fail-open case."""

    def incrby(self, name: str, amount: int) -> int:
        raise redis.ConnectionError("connection refused")

    def expire(self, name: str, seconds: int) -> bool:
        raise redis.ConnectionError("connection refused")

    def get(self, name: str) -> bytes | None:
        raise redis.ConnectionError("connection refused")

    def exists(self, *names: str) -> int:
        raise redis.ConnectionError("connection refused")

    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object:
        raise redis.ConnectionError("connection refused")


def unreachable_store() -> RedisBudgetStore:
    """A real `RedisBudgetStore` over a client that cannot answer."""
    return RedisBudgetStore(cast(redis.Redis, UnreachableRedis()))


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_stub() -> StubAdapter:
    """The completion adapter under the policy provider `anthropic`."""
    return StubAdapter(name=PROVIDER)


@pytest.fixture(scope="module")
def embed_stub() -> EmbeddingStubAdapter:
    """The embedding adapter under `local`: the `embed` alias and the semantic tier."""
    return EmbeddingStubAdapter(name=EMBED_PROVIDER)


@pytest.fixture(scope="module")
def budget_client(
    settings: Settings, chat_stub: StubAdapter, embed_stub: EmbeddingStubAdapter
) -> Iterator[TestClient]:
    """One app for the module: a chat adapter and an embedding adapter, built once."""
    adapters: dict[str, Adapter] = {chat_stub.name: chat_stub, embed_stub.name: embed_stub}
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture
def budget_store(budget_client: TestClient) -> RecordingBudgetStore:
    """This test's counter, recording every call, on `app.state.budget_store`."""
    store = RecordingBudgetStore()
    app_of(budget_client).state.budget_store = store
    return store


@pytest.fixture(autouse=True)
def _fresh(
    budget_client: TestClient, chat_stub: StubAdapter, embed_stub: EmbeddingStubAdapter
) -> Iterator[None]:
    """Empty caches, a fresh ledger, a fresh counter and cleared stubs - never a new app."""
    chat_stub.requests.clear()
    embed_stub.requests.clear()
    app = app_of(budget_client)
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = InProcessSemanticCache()
    app.state.cost_ledger = InMemoryCostLedgerWriter()
    app.state.budget_store = RecordingBudgetStore()
    yield


def month() -> str:
    """The month a request made now is booked under."""
    return current_month()


def post(client: TestClient, body: Mapping[str, Any], *, tenant: str = TENANT) -> Response:
    """POST one completion, as `evals` under `internal` unless a case says otherwise."""
    response: Response = client.post(
        CHAT,
        json=dict(body),
        headers=auth_headers(
            key=TEST_KEYS[tenant], tenant=tenant, data_class=DATA_CLASS, session=SESSION
        ),
    )
    return response


def post_embed(client: TestClient, body: Mapping[str, Any]) -> Response:
    """The same, on `/v1/embeddings`."""
    response: Response = client.post(
        EMBEDDINGS,
        json=dict(body),
        headers=auth_headers(
            key=TEST_KEYS[TENANT], tenant=TENANT, data_class=DATA_CLASS, session=SESSION
        ),
    )
    return response


def question_body(text: str) -> dict[str, Any]:
    """A semantic-eligible body: `agent-default`, no temperature, the given question."""
    return chat_body(model=ALIAS, messages=[{"role": "user", "content": text}])


def eligible_body() -> dict[str, Any]:
    """The exact tier's baseline: `temperature: 0` stated explicitly."""
    return chat_body(model=ALIAS, temperature=0)


def arm(client: TestClient, store: RecordingBudgetStore) -> None:
    """Start measuring here: a fresh ledger, and the priming requests' debits forgotten."""
    app_of(client).state.cost_ledger = InMemoryCostLedgerWriter()
    store.debits.clear()


def rows_of(writer: CostLedgerWriter) -> tuple[CostLedgerRow, ...]:
    """The rows an in-memory writer holds, with the Protocol typing narrowed at one seam."""
    assert isinstance(writer, InMemoryCostLedgerWriter)
    return writer.rows


def ledger_rows(client: TestClient) -> tuple[CostLedgerRow, ...]:
    """The rows this app's ledger holds right now."""
    writer: CostLedgerWriter = app_of(client).state.cost_ledger
    return rows_of(writer)


def gauge(collector_registry: CollectorRegistry, tenant: str) -> float | None:
    """One tenant's budget ratio as a scraper would read it, or `None` for no series.

    Read absolutely rather than as a delta: a gauge carries a level, and `None`
    is what separates "publishes nothing" from "publishes 0".
    """
    return collector_registry.get_sample_value(BUDGET_RATIO, {"tenant": tenant})


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The budget module's own WARNING records, and nothing else's."""
    return [
        record
        for record in caplog.records
        if record.name == BUDGET_LOGGER and record.levelno == logging.WARNING
    ]


# --------------------------------------------------------------------------
# the arrangement itself
# --------------------------------------------------------------------------


def test_the_shipped_cap_for_this_tenant_is_the_one_these_cases_arrange_against(
    policy: Policy,
) -> None:
    """Not a behaviour: the guard that keeps every case below meaningful.

    Catches an edit to `routing.yaml` that changes `evals`'s cap, after which the
    over-cap arrangement would be under cap and every refusal case would pass by
    serving. The conversion is asserted too, because the arrangement is written in
    micro-USD and the policy in dollars."""
    assert policy.tenants[TENANT].budget_usd_month == CAP_USD
    assert to_micro_usd(CAP_USD) == CAP
    assert int(CAP * Decimal("0.8")) == WARN_AT


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------


def test_a_tenant_over_cap_is_429_budget_exceeded(
    budget_client: TestClient, budget_store: RecordingBudgetStore
) -> None:
    """The inclusive boundary, end to end.

    The spend is arranged at EXACTLY the cap, which is the row a `>` catches: a check
    written "over the cap" serves this request, and the criterion says it is the 429.
    The body is asserted through the shared `assert_error_body`, so a refusal raised as a
    bare `HTTPException` - 429 with FastAPI's `{"detail": ...}` and no typed `code` -
    fails here even though the status is right."""
    budget_store.arrange(TENANT, month(), CAP)

    response = post(budget_client, chat_body(model=ALIAS))

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_tenant_one_micro_under_the_cap_is_served(
    budget_client: TestClient, budget_store: RecordingBudgetStore
) -> None:
    """The other side of the same boundary, and the case that stops the one above passing
    against a route that refuses everybody. Catches a `>=` on the wrong quantity - a
    check comparing the spend PLUS this request's cost would refuse here, which is
    a reservation the gateway deliberately does not take."""
    budget_store.arrange(TENANT, month(), CAP - 1)

    response = post(budget_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text


def test_the_embeddings_route_refuses_over_cap_too(
    budget_client: TestClient, budget_store: RecordingBudgetStore
) -> None:
    """Catches a check wired into the chat route alone, which is the likelier half of the
    job to forget: the tenant keeps spending through `/v1/embeddings` after its cap is
    reached, and the counter keeps rising while the gateway never refuses."""
    budget_store.arrange(TENANT, month(), CAP)

    response = post_embed(budget_client, embeddings_body())

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_streamed_request_over_cap_is_refused_before_the_stream_opens(
    budget_client: TestClient, budget_store: RecordingBudgetStore, chat_stub: StubAdapter
) -> None:
    """Catches a check placed inside the serving loop rather than at the top of the route: the
    response would already be a 200 `text/event-stream` by the time the
    budget was consulted, and the client would receive a stream that carries an error in
    a frame instead of the typed 429 body."""
    budget_store.arrange(TENANT, month(), CAP)

    response = post(budget_client, chat_body(model=ALIAS, stream=True))

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert chat_stub.requests == []


def test_a_refused_request_dials_no_provider_and_books_no_row(
    budget_client: TestClient, budget_store: RecordingBudgetStore, chat_stub: StubAdapter
) -> None:
    """The requirement's other three halves in the arrangement that can see them.

    The request is exact-cache eligible and identical to one sent afterwards under a cap
    that allows it, so "no cache entry was stored" is asserted as the LATER request
    missing - a refusal that stored the answer it never got would make it a hit. Catches
    a check that runs after the provider call (the money is already spent), after
    `record_cost` (the ledger disagrees with the counter), and a refusal path that writes
    the response into the cache on its way out."""
    budget_store.arrange(TENANT, month(), CAP)

    refused = post(budget_client, eligible_body())

    assert refused.status_code == 429
    assert chat_stub.requests == [], "a provider was dialled for a request that was refused"
    assert ledger_rows(budget_client) == ()
    assert budget_store.debits == []

    app_of(budget_client).state.budget_store = RecordingBudgetStore()
    served = post(budget_client, eligible_body())

    assert served.status_code == 200, served.text
    assert served.headers[CACHE_HEADER] == CACHE_STATUS_MISS, (
        "the refused request stored an answer in the cache"
    )


def test_a_tenant_the_policy_does_not_carry_is_still_the_resolvers_403(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    metrics_registry: CollectorRegistry,
) -> None:
    """The budget check runs BEFORE the resolver, so it meets tenants the policy has never
    heard of - a key the store carries with no `routing.yaml` entry.

    Catches a cap read as `policy.tenants[tenant]`, which raises `KeyError` and turns
    today's typed 403 into an untyped 500; and catches a check that invents a cap of zero
    for such a tenant, which would answer 429 and hide the tenant refusal that is the
    correct answer. Deny-by-default is the resolver's job here, not the budget's."""
    response = post(budget_client, chat_body(model=ALIAS), tenant=UNPOLICED_TENANT)

    assert_error_body(response, status=403, code="TENANT_MISMATCH")
    assert budget_store.debits == []
    assert gauge(metrics_registry, UNPOLICED_TENANT) is None


# --------------------------------------------------------------------------
# the gauge
# --------------------------------------------------------------------------


def test_the_gauge_reads_the_ratio_after_a_served_request(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    metrics_registry: CollectorRegistry,
) -> None:
    """Catches a gauge fed BEFORE the request is debited, which lags a month's spend by one
    request and reads a flat 0 for the first one; a gauge fed with the raw micro-USD
    total rather than the fraction (7,000,000 on a panel scaled to 1); and a gauge fed
    from a second reading of the store rather than from what `debit` answered."""
    budget_store.arrange(TENANT, month(), WARN_AT)

    response = post(budget_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert gauge(metrics_registry, TENANT) == pytest.approx((WARN_AT + CHAT_MICRO) / CAP)


def test_an_unreachable_store_serves_and_moves_no_gauge(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    metrics_registry: CollectorRegistry,
) -> None:
    """The behaviour end to end, in the only arrangement that can see it.

    A served request first, so the gauge holds a real level; then the same request
    against a store that cannot answer. Catches a route that lets the store's failure
    reach the client as a 500 or a 429 on a request every provider would have served,
    and catches the over-correction: a gauge overwritten with 0.0 because "unknown" was
    read as "nothing spent", which paints every tenant healthy for exactly as long as
    Redis is down."""
    budget_store.arrange(TENANT, month(), WARN_AT)
    assert post(budget_client, chat_body(model=ALIAS)).status_code == 200
    before = gauge(metrics_registry, TENANT)
    assert before is not None and before > 0

    app_of(budget_client).state.budget_store = unreachable_store()
    served = post(budget_client, chat_body(model=ALIAS))

    assert served.status_code == 200, served.text
    assert gauge(metrics_registry, TENANT) == before


# --------------------------------------------------------------------------
# the warning
# --------------------------------------------------------------------------


def test_a_tenant_at_eighty_percent_is_served_and_warned_once(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    metrics_registry: CollectorRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warn boundary, inclusive, end to end.

    Arranged at EXACTLY 80%, which is the row a `>` catches. Catches a warn band that
    refuses instead of serving (the two decisions collapsed into one comparison), a
    warning logged twice per request from two call sites, and a message that names no
    tenant - an alert that cannot say WHOSE budget is nearly gone is one nobody can act
    on. The three values are read off the record rather than out of the formatted string,
    so the wording stays the implementation's to choose."""
    budget_store.arrange(TENANT, month(), WARN_AT)

    with caplog.at_level(logging.WARNING, logger=BUDGET_LOGGER):
        response = post(budget_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    warned = warnings_of(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    assert TENANT in warned[0].getMessage()
    carried = warned[0].__dict__
    assert carried["tenant"] == TENANT
    assert Decimal(str(carried["spend_usd"])) == Decimal(WARN_AT) / CAP * CAP_USD
    assert Decimal(str(carried["cap_usd"])) == CAP_USD

    ratio = gauge(metrics_registry, TENANT)
    assert ratio is not None and ratio >= 0.8


def test_a_tenant_just_under_eighty_percent_is_served_without_a_warning(
    budget_client: TestClient, budget_store: RecordingBudgetStore, caplog: pytest.LogCaptureFixture
) -> None:
    """One micro under the boundary, which is what stops the case above passing against an
    implementation that warns on every request. Catches a threshold applied to the ratio
    as a float where a rounding puts 0.7999999 in the warn band, and a warning emitted
    unconditionally next to the check."""
    budget_store.arrange(TENANT, month(), WARN_AT - 1)

    with caplog.at_level(logging.WARNING, logger=BUDGET_LOGGER):
        response = post(budget_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert warnings_of(caplog) == []


# --------------------------------------------------------------------------
# the debit
# --------------------------------------------------------------------------

Send = Callable[[TestClient, RecordingBudgetStore], Response]
"""One measured request, armed immediately before it.

The store is a parameter rather than a fixture read inside the sender so a
priming request's debit is never counted: `arm` clears what the priming did.
"""


def send_chat(client: TestClient, store: RecordingBudgetStore) -> Response:
    """One non-streamed completion, answered by the provider."""
    arm(client, store)
    return post(client, chat_body(model=ALIAS))


def send_stream(client: TestClient, store: RecordingBudgetStore) -> Response:
    """The same request, streamed, with the provider reporting usage."""
    arm(client, store)
    return post(client, chat_body(model=ALIAS, stream=True))


def send_exact_hit(client: TestClient, store: RecordingBudgetStore) -> Response:
    """A second identical eligible request, served from the exact cache."""
    primed = post(client, eligible_body())
    assert primed.status_code == 200, primed.text
    arm(client, store)
    hit = post(client, eligible_body())
    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_HIT, "the arrangement did not hit"
    return hit


def send_semantic_hit(client: TestClient, store: RecordingBudgetStore) -> Response:
    """The same question reworded, served from the semantic cache."""
    primed = post(client, question_body(QUESTION))
    assert primed.status_code == 200, primed.text
    arm(client, store)
    hit = post(client, question_body(SAME_QUESTION_REWORDED))
    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_SEMANTIC, "the arrangement did not hit"
    return hit


def send_embeddings(client: TestClient, store: RecordingBudgetStore) -> Response:
    """One embeddings request against the `embed` alias."""
    arm(client, store)
    return post_embed(client, embeddings_body())


PATHS: list[tuple[str, Send, int]] = [
    ("the non-streamed chat path", send_chat, CHAT_MICRO),
    ("the streamed chat path", send_stream, STREAM_MICRO),
    ("the exact-cache hit path", send_exact_hit, ZERO_MICRO),
    ("the semantic-cache hit path", send_semantic_hit, ZERO_MICRO),
    ("the embeddings path", send_embeddings, ZERO_MICRO),
]
"""The five `book_cost` call sites, and the exact
micro-USD each owes. Three of the five owe zero - two cache hits and a
zero-priced `local` row - which is precisely why the assertion below is about the
recorded CALL and not about a total: a total cannot tell a site that debited zero
from one that was never wired at all, and those three are the sites most likely
to be missed."""


@pytest.mark.parametrize(("label", "send", "expected"), PATHS, ids=[row[0] for row in PATHS])
def test_every_book_cost_site_debits_the_budget(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    label: str,
    send: Send,
    expected: int,
) -> None:
    """One row per `book_cost` call site: each debits once, and by the row it just booked.

    Catches a call site the wiring missed - the streamed sinks and the two cache-hit
    paths are the three that book a row somewhere other than the closing stretch of the
    non-streamed path - which would leave that traffic invisible to every cap for ever.
    Catches a debit priced a SECOND time rather than taken from the row (`cost_usd` and
    the debit then disagree by a rounding, and only one of them is ever audited), and a
    debit booked twice for one request. The expected figure is asserted against both the
    literal from the shipped price table and the ledger row, so a route that books a row
    of the wrong size fails on the first and one that debits the wrong number fails on
    the second."""
    response = send(budget_client, budget_store)

    assert response.status_code == 200, response.text
    rows = ledger_rows(budget_client)
    assert len(rows) == 1, rows
    assert budget_store.debits == [(TENANT, month(), expected)]
    assert to_micro_usd(rows[0].cost_usd) == expected


@pytest.fixture(scope="module")
def usageless_client(settings: Settings) -> Iterator[TestClient]:
    """A second app whose stream finishes without a `UsageTotal` frame, built once.

    The stub's scripted events are fixed at construction, so this is the one case that
    cannot share the module's app. It costs a YAML parse and no backing service.
    """
    stub = StubAdapter(name=PROVIDER, events=stream_events(usage=None))
    adapters: dict[str, Adapter] = {
        PROVIDER: stub,
        EMBED_PROVIDER: EmbeddingStubAdapter(name=EMBED_PROVIDER),
    }
    with build_chat_client(settings, adapters) as client:
        yield client


def test_a_stream_that_reports_no_usage_still_debits_its_estimated_row(
    usageless_client: TestClient,
) -> None:
    """The SIXTH `book_cost` call site, which the list of five does not include: a
    stream that ends without a usage frame books an ESTIMATED row from a second sink
    (`_estimated_cost_sink` in `services/chat_pipeline.py`).

    Catches a debit wired into the reported-usage sink alone. A provider that stops
    sending usage - or a client that goes away mid-stream - would then spend real money
    against a counter that never moves, which is the one shape of leak a cap cannot
    notice, because everything it CAN see still adds up. The row's `estimated` flag is
    asserted so a stream that quietly took the reported-usage path instead cannot make
    this pass."""
    store = RecordingBudgetStore()
    app = app_of(usageless_client)
    app.state.budget_store = store
    app.state.cost_ledger = InMemoryCostLedgerWriter()

    response = usageless_client.post(
        CHAT,
        json=chat_body(model=ALIAS, stream=True),
        headers=auth_headers(
            key=TEST_KEYS[TENANT], tenant=TENANT, data_class=DATA_CLASS, session=SESSION
        ),
    )

    assert response.status_code == 200, response.text
    rows = ledger_rows(usageless_client)
    assert len(rows) == 1, rows
    assert rows[0].estimated is True
    assert store.debits == [(TENANT, month(), to_micro_usd(rows[0].cost_usd))]


@pytest.mark.parametrize(
    ("label", "send", "expected"),
    [row for row in PATHS if row[0] != "the streamed chat path"],
    ids=[row[0] for row in PATHS if row[0] != "the streamed chat path"],
)
def test_the_debit_is_the_number_the_cost_header_reports(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    label: str,
    send: Send,
    expected: int,
) -> None:
    """The one-figure claim from the client's side. `X-Gateway-Cost-USD`, the ledger row
    and the counter are one number computed once, so a debit that
    disagrees with the header the same response carried is a gateway whose invoice and
    whose cap cannot both be right. The streamed path is excluded because its headers are
    on the wire before the usage chunk exists, which is why it carries no cost header at
    all."""
    response = send(budget_client, budget_store)

    assert response.status_code == 200, response.text
    assert budget_store.debits == [
        (TENANT, month(), to_micro_usd(Decimal(response.headers[COST_HEADER])))
    ]
    assert to_micro_usd(Decimal(response.headers[COST_HEADER])) == expected


def test_a_cache_hit_debits_zero(
    budget_client: TestClient,
    budget_store: RecordingBudgetStore,
    metrics_registry: CollectorRegistry,
) -> None:
    """A replayed answer bought nothing, so it moves the counter by nothing - and it still
    moves it, once.

    Catches an implementation that debits the REPLAYED answer's cost, which charges a
    tenant again for every hit and makes the cache actively expensive; and the opposite,
    a hit that skips the debit entirely and so never refreshes the gauge. The total after
    the hit is asserted as well, because "debited zero" and "debited nothing" are the two
    readings a total cannot separate."""
    budget_store.arrange(TENANT, month(), WARN_AT)
    primed = post(budget_client, eligible_body())
    assert primed.status_code == 200, primed.text
    before = budget_store.inner.spend(TENANT, month())
    arm(budget_client, budget_store)

    hit = post(budget_client, eligible_body())

    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_HIT
    assert budget_store.debits == [(TENANT, month(), ZERO_MICRO)]
    assert budget_store.inner.spend(TENANT, month()) == before
    assert gauge(metrics_registry, TENANT) == pytest.approx(cast(int, before) / CAP)


def test_the_debit_is_booked_under_the_requesting_tenant_and_the_current_month(
    budget_client: TestClient, budget_store: RecordingBudgetStore
) -> None:
    """Catches a debit booked under a tenant taken from the request body or from the
    `X-Tenant` header before authentication rather than from the authenticated context,
    and one booked under a month string built somewhere other than `current_month` - a
    month key that drifts is a cap that never fills."""
    arm(budget_client, budget_store)

    response = post(budget_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    tenants = {tenant for tenant, _, _ in budget_store.debits}
    months = {booked_month for _, booked_month, _ in budget_store.debits}
    assert tenants == {TENANT}
    assert months == {current_month()}


def test_the_check_reads_the_counter_before_the_provider_is_dialled(
    budget_client: TestClient, budget_store: RecordingBudgetStore
) -> None:
    """Stated as an ordering the recording store can see: the spend is read for
    a request the route goes on to serve, not only for one it refuses. Catches an
    implementation that checks the cap only inside the refusal path it never reaches
    under cap - which is a gateway whose budget is enforced by nothing until someone
    happens to be over it - and one that never reads the counter at all and refuses from
    the ledger."""
    response = post(budget_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert (TENANT, current_month()) in budget_store.reads


def test_a_served_request_is_still_counted_once(
    budget_client: TestClient, budget_store: RecordingBudgetStore, metrics: MetricDelta
) -> None:
    """The neighbouring family, because a check inserted at the top of both routes is
    exactly the kind of early return that double-counts or skips
    `gateway_requests_total`. Catches a budget check that answers before the metrics
    record is opened, which would leave a refusal uncounted and every error-rate panel
    blind to a budget incident."""
    budget_store.arrange(TENANT, month(), CAP)

    refused = post(budget_client, chat_body(model=ALIAS))

    assert refused.status_code == 429
    assert metrics.total(REQUESTS) == 1


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------

UNCONNECTED_REDIS_URL = "redis://127.0.0.1:1/0"
"""A Redis nothing listens on. `redis.Redis` connects lazily, so building the
counter costs no connection and the URL is never dialled by the two cases below."""


def wiring_settings(*, redis_url: str | None) -> Settings:
    """The test key store and the shipped policy, with the Redis URL stated either way.

    Stated rather than left to default, for the reason
    tests/gateway/test_exact_cache.py states it: `Settings` reads
    `OPSCONSOLE_REDIS_URL` from the process environment, which the Redis suite asks a
    developer to export, and the unconfigured row would then assert the opposite of
    what it says.
    """
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, redis_url=redis_url)


def test_create_app_keeps_the_in_process_budget_store_without_a_redis_url() -> None:
    """Catches a factory that requires Redis to enforce a cap at all: the whole existing
    suite, `make verify` and a bare `uvicorn` run configure no `redis_url`, so a
    Redis-only factory would leave `app.state.budget_store` absent and turn the first
    request into an `AttributeError` with no code."""
    app = create_app(wiring_settings(redis_url=None), NO_PROVIDER_KEYS)

    # The store is built on lifespan startup now, so enter it first.
    with TestClient(app):
        assert isinstance(app.state.budget_store, InProcessBudgetStore)


def test_create_app_uses_the_redis_budget_store_when_a_redis_url_is_configured() -> None:
    """Catches a factory that counts a month's spend in process memory while Redis is
    configured: two gateway replicas would then each enforce their own share of one cap,
    so a tenant would be served twice its budget and no test with one replica could see
    it except here at the wiring."""
    app = create_app(wiring_settings(redis_url=UNCONNECTED_REDIS_URL), NO_PROVIDER_KEYS)

    with TestClient(app):
        assert isinstance(app.state.budget_store, RedisBudgetStore)
