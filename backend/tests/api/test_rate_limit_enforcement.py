"""Per-tenant rate limits through the route: the 429, the order, the header and the debit.

Everything here is settled by a real request through `TestClient` against an app
`create_app` built, because the question is never "does a
bucket refuse at zero" - tests/limits/test_ratelimit.py settles that - but "is the
request refused with the typed body every other refusal carries, before a provider
is dialled, and does every path that books a ledger row debit the token bucket".

No backing service. The app is built with no Redis URL, so
`app.state.rate_limit_store` is the in-process twin; every case here wraps it in
`RecordingRateLimitStore`, which is the only way to tell "debited zero tokens"
(what a cache hit owes) from "did not debit at all" (a call site the plan missed).
The store is reached by NAME, `app.state.rate_limit_store`, never by shape: this
app already holds several collaborators, and selecting one of them by its methods
is what a project rule exists for.

THE TWIN IS ONLY EVIDENCE ABOUT THE SHIPPED STORE BECAUSE OF THE DIFFERENTIAL.
`RedisRateLimitStore` runs a Lua script no case here can exercise, and
tests/limits/test_redis_ratelimit.py asserts the two stores answer identically
over one interleaved sequence. Without that, every refusal below would be a
statement about code a cluster does not run (section 5's second risk).

a project rule: one app for the module, built once. Per-test isolation is a
fresh `rate_limit_store`, `budget_store`, `cost_ledger`, `response_cache` and
`semantic_cache` on `app.state` and the stubs' recorded calls cleared - never a
new app.

Three arrangements are worth reading before the assertions:

* THE BUCKET IS DRAINED THROUGH THE STORE, not by sending 120 requests. A hundred
  and twenty requests would be a slow suite and would prove nothing the one-line
  arrangement does not.
* THE CLOCK IS FROZEN. `InProcessRateLimitStore` takes its clock as a constructor
  argument, so a drained bucket stays drained for the length of a case and
  "refills after half a second" is arranged by assigning to a number. Nothing
  here sleeps, and no case can pass or fail on how long the suite took.
* THE EMBEDDING ADAPTER IS LIVE ON THIS APP. `agent-default` carries
  `semantic_cache: true`, so a chat request here really does embed through the
  `embed` alias against `local`; that is what makes the semantic-hit path
  reachable, and it is also what would expose an implementation that took a
  second request token for the internal embedding (a project rule's fourth
  target).

The contract these cases are written against, before it exists:

* `create_app` builds `app.state.rate_limit_store` - `RedisRateLimitStore` with a
  Redis URL, `InProcessRateLimitStore` without one;
* `check_rate_limit(state, context)` lives beside `check_budget` in
  `services/chat_pipeline.py` and is called from BOTH routes immediately BEFORE
  `check_budget`, after `Context` has authenticated the tenant. It takes
  `CHECK_COST` from the request bucket and `CHECK_COST` from the token bucket,
  at the tenant's own `rpm`/`tpm` capacity and `rpm/60`/`tpm/60` rate, and raises
  `RateLimited` carrying `retry_after_seconds` when either is refused;
* the tenant's limit is read with `.get`, so a tenant `routing.yaml` does not
  carry has no limit at all and is left to the resolver's 403;
* `debit_tokens` is called from inside `book_cost` and nowhere else, beside
  `debit_budget`, and books `0` for a cached row and `input_tokens +
  output_tokens` for every other one;
* a `None` from the store is "cannot answer": the request is SERVED and the
  failure is logged at ERROR with the tenant and the dimension (decision E);
* `RateLimited` carries `retry_after_seconds` and `gateway/errors.py` emits
  `Retry-After` in whole seconds for a typed error that carries one, and for no
  other response (decision D).
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

from app.cache import InProcessResponseCache, InProcessSemanticCache
from app.core.config import Settings
from app.limits.budget import InProcessBudgetStore, current_month, to_micro_usd
from app.limits.ratelimit import (
    SECONDS_PER_MINUTE,
    InProcessRateLimitStore,
    RateDimension,
    RedisRateLimitStore,
    Taken,
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
)

REQUESTS = "gateway_requests_total"

RATE_LIMIT_LOGGER = "app.limits.ratelimit"
"""Where a swallowed limiter failure is reported. Named so the assertion counts
THIS module's records and not whatever else the request logged."""

TENANT = "evals"
DATA_CLASS = "internal"
SESSION = "sess-rate-limit-enforcement"
ALIAS = "agent-default"
PROVIDER = "anthropic"

EMBED_ALIAS = "embed"
EMBED_PROVIDER = "local_embed"

UNPOLICED_TENANT = "cap-key"
"""A tenant with a key in the store and no entry in `routing.yaml`. The rate check
runs before the resolver, so this is the tenant that tells a check which looks its
ceiling up with `policy.tenants[tenant]` from one that does not."""

RPM = 120
TPM = 120_000
"""`evals`'s shipped ceilings. Asserted against the loaded policy by the first case
below, so an edit to `routing.yaml` fails there rather than making every other case
here meaningless."""

REQUEST_RATE = 2.0
TOKEN_RATE = 2000.0
"""`RPM / 60` and `TPM / 60`, the two refill rates (criterion 3)."""

CHECK_COST = 1
"""What one request costs each bucket at the check.

One from the request bucket is decision C: a request is countable before it is
served, so the check and the debit are one atomic take. One from the TOKEN bucket
is the probe that makes "this tenant has spent its minute's tokens" expressible at
all - the real usage is not known until the provider has answered (decision B), so
the check asks whether a single token is left and the debit follows from
`book_cost`. One token in 120,000 is what that probe costs.
"""

CHAT_TOKENS = 166
"""`STUB_USAGE`: 137 prompt + 29 completion, written out as arithmetic rather than
recomputed from the stub the implementation reads."""

STREAM_TOKENS = 483
"""`STREAM_USAGE`: 431 + 52. Different from `CHAT_TOKENS` in both directions, so a
streamed request debited from the non-streamed usage is a different number here."""

EMBED_TOKENS = 97
"""`STUB_EMBEDDING_USAGE`: 97 prompt, and `completion_tokens=0` because an
embedding has no output direction."""

ZERO_TOKENS = 0
"""What a cache hit debits. A recorded call carrying zero, never an absent call -
which is the whole reason this module records takes instead of reading a level."""

RETRY_AFTER_HEADER = "retry-after"
CACHE_HEADER = "x-gateway-cache"

FROZEN = 1_800_000_000.0
"""The instant every case runs at. Assigning to `clock[0]` is how time passes."""

QUESTION = "What is the capital of Germany?"
SAME_QUESTION_REWORDED = "Which city is the capital city of Germany?"


# --------------------------------------------------------------------------
# the seams
# --------------------------------------------------------------------------


class RecordingRateLimitStore:
    """The in-process twin on a frozen clock, with every take kept.

    `takes` is what separates "this path debited zero tokens" from "this path
    never debited", which no bucket level can express and which is exactly the
    failure a missed `book_cost` call site produces: the tenant's `tpm` is
    permanently under-counted, and on the three paths whose token cost is zero it
    leaves no trace at all.

    `drain` empties a bucket without recording a take, so a case's assertion about
    what the REQUEST did is not polluted by its arrangement.
    """

    def __init__(self, clock: list[float]) -> None:
        self.clock = clock
        self.inner = InProcessRateLimitStore(clock=lambda: clock[0])
        self.takes: list[tuple[str, RateDimension, int, int, float]] = []

    def take(
        self, tenant: str, dimension: RateDimension, *, cost: int, capacity: int, rate: float
    ) -> Taken | None:
        self.takes.append((tenant, dimension, cost, capacity, rate))
        return self.inner.take(tenant, dimension, cost=cost, capacity=capacity, rate=rate)

    def drain(self, tenant: str, dimension: RateDimension, *, capacity: int, rate: float) -> None:
        """Spend a bucket down to exactly zero, off the record."""
        self.inner.take(tenant, dimension, cost=capacity, capacity=capacity, rate=rate)

    def leave(
        self, tenant: str, dimension: RateDimension, *, capacity: int, rate: float, tokens: int
    ) -> None:
        """Spend a bucket down to exactly `tokens`, off the record."""
        self.inner.take(tenant, dimension, cost=capacity - tokens, capacity=capacity, rate=rate)

    def dimension_takes(
        self, dimension: RateDimension
    ) -> list[tuple[str, RateDimension, int, int, float]]:
        """Every recorded take on one bucket, in the order the request made them."""
        return [row for row in self.takes if row[1] is dimension]


class UnreachableRedis:
    """A client whose registered script always raises, for the fail-open case."""

    def register_script(self, script: str) -> Callable[..., object]:
        def run(keys: object, args: object) -> object:
            raise redis.ConnectionError("connection refused")

        return run


def unreachable_store() -> RedisRateLimitStore:
    """A real `RedisRateLimitStore` over a client that cannot answer."""
    return RedisRateLimitStore(cast(redis.Redis, UnreachableRedis()))


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_stub() -> StubAdapter:
    """The completion adapter under the policy provider `anthropic`."""
    return StubAdapter(name=PROVIDER)


@pytest.fixture(scope="module")
def embed_stub() -> EmbeddingStubAdapter:
    """The embedding adapter under `local_embed`: the `embed` alias and the semantic
    tier."""
    return EmbeddingStubAdapter(name=EMBED_PROVIDER)


@pytest.fixture(scope="module")
def limit_client(
    settings: Settings, chat_stub: StubAdapter, embed_stub: EmbeddingStubAdapter
) -> Iterator[TestClient]:
    """One app for the module: a chat adapter and an embedding adapter, built once."""
    adapters: dict[str, Adapter] = {chat_stub.name: chat_stub, embed_stub.name: embed_stub}
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture
def clock() -> list[float]:
    """The instant this test's buckets read, as a one-element list a case assigns into."""
    return [FROZEN]


@pytest.fixture
def limit_store(limit_client: TestClient, clock: list[float]) -> RecordingRateLimitStore:
    """This test's buckets, recording every take, on `app.state.rate_limit_store`."""
    store = RecordingRateLimitStore(clock)
    app_of(limit_client).state.rate_limit_store = store
    return store


@pytest.fixture(autouse=True)
def _fresh(
    limit_client: TestClient,
    chat_stub: StubAdapter,
    embed_stub: EmbeddingStubAdapter,
    clock: list[float],
) -> Iterator[None]:
    """Empty caches, fresh ledgers, fresh buckets and cleared stubs - never a new app."""
    chat_stub.requests.clear()
    embed_stub.requests.clear()
    app = app_of(limit_client)
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = InProcessSemanticCache()
    app.state.cost_ledger = InMemoryCostLedgerWriter()
    app.state.budget_store = InProcessBudgetStore()
    app.state.rate_limit_store = RecordingRateLimitStore(clock)
    yield


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


def drain_requests(store: RecordingRateLimitStore) -> None:
    """Empty this tenant's request bucket, off the record."""
    store.drain(TENANT, RateDimension.REQUESTS, capacity=RPM, rate=REQUEST_RATE)


def drain_tokens(store: RecordingRateLimitStore) -> None:
    """Empty this tenant's token bucket, off the record."""
    store.drain(TENANT, RateDimension.TOKENS, capacity=TPM, rate=TOKEN_RATE)


def over_budget(client: TestClient) -> None:
    """Put this tenant's whole monthly cap on its budget counter."""
    store = InProcessBudgetStore()
    store.debit(TENANT, current_month(), to_micro_usd(Decimal("10.00")))
    app_of(client).state.budget_store = store


def arm(client: TestClient, store: RecordingRateLimitStore) -> None:
    """Start measuring here: a fresh ledger, and the priming requests' takes forgotten."""
    app_of(client).state.cost_ledger = InMemoryCostLedgerWriter()
    store.takes.clear()


def rows_of(writer: CostLedgerWriter) -> tuple[CostLedgerRow, ...]:
    """The rows an in-memory writer holds, with the Protocol typing narrowed at one seam."""
    assert isinstance(writer, InMemoryCostLedgerWriter)
    return writer.rows


def ledger_rows(client: TestClient) -> tuple[CostLedgerRow, ...]:
    """The rows this app's ledger holds right now."""
    writer: CostLedgerWriter = app_of(client).state.cost_ledger
    return rows_of(writer)


# --------------------------------------------------------------------------
# the arrangement itself
# --------------------------------------------------------------------------


def test_the_shipped_rate_limit_for_this_tenant_is_the_one_these_cases_arrange_against(
    policy: Policy,
) -> None:
    """Not a behaviour: the guard that keeps every case below meaningful.

    Catches an edit to `routing.yaml` that changes `evals`'s ceilings, after which the
    drained arrangement would leave tokens in the bucket and every refusal case would
    pass by serving. It is also section 8's other half: two FIXTURE policies had their
    ceilings raised so unrelated modules cannot trip over enforcement, and this case is
    what stops that raise from quietly reaching production - the shipped file is asserted
    here and nowhere else. The derived rates are asserted too, because the arrangements
    are written in rates and the policy in per-minute ceilings."""
    limit = policy.tenants[TENANT].rate_limit

    assert limit.rpm == RPM
    assert limit.tpm == TPM
    assert limit.rpm / SECONDS_PER_MINUTE == REQUEST_RATE
    assert limit.tpm / SECONDS_PER_MINUTE == TOKEN_RATE


# --------------------------------------------------------------------------
# the refusal (criteria 1 and 2)
# --------------------------------------------------------------------------


def test_a_tenant_over_its_request_rate_is_429_rate_limited(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Criterion 1, end to end, at the inclusive boundary.

    The bucket is drained to EXACTLY zero, which is the row a `>` catches: a check written
    "below zero" serves this request and the tenant gets 121 requests a minute out of a
    policy that says 120. The body is asserted through the shared `assert_error_body`, so
    a refusal raised as a bare `HTTPException` - 429 with FastAPI's `{"detail": ...}` and
    no typed `code` - fails here even though the status is right."""
    drain_requests(limit_store)

    response = post(limit_client, chat_body(model=ALIAS))

    assert_error_body(response, status=429, code="RATE_LIMITED")


def test_a_tenant_with_one_request_token_left_is_served(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """The other side of the same boundary, and the case that stops the one above passing
    against a route that refuses everybody. Catches a check that takes more than
    `CHECK_COST` from the request bucket - a check costing two would halve every tenant's
    published `rpm` - and a comparison written `>` on the level after the subtraction."""
    limit_store.leave(TENANT, RateDimension.REQUESTS, capacity=RPM, rate=REQUEST_RATE, tokens=1)

    response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text


def test_a_tenant_over_its_token_rate_is_429_rate_limited(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Criterion 2. The REQUEST bucket is left full, so the request ceiling cannot be what
    refuses: only the token bucket is empty, and removing the token half of the check is
    the single thing that flips this to a 200. Catches a `check_rate_limit` that reads
    `rpm` and ignores `tpm` entirely, which is the likelier half to forget because it is
    the one whose cost is not known until after the provider has answered."""
    drain_tokens(limit_store)

    response = post(limit_client, chat_body(model=ALIAS))

    assert_error_body(response, status=429, code="RATE_LIMITED")


def test_the_embeddings_route_refuses_over_rate_too(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Catches a check wired into the chat route alone, which is the likelier half of the
    job to forget: the tenant keeps sending through `/v1/embeddings` after its ceiling is
    reached, and every embedding still costs a provider call the limit was meant to
    stop."""
    drain_requests(limit_store)

    response = post_embed(limit_client, embeddings_body())

    assert_error_body(response, status=429, code="RATE_LIMITED")


def test_a_streamed_request_over_rate_is_refused_before_the_stream_opens(
    limit_client: TestClient, limit_store: RecordingRateLimitStore, chat_stub: StubAdapter
) -> None:
    """Catches a check placed inside the serving loop rather than at the top of the route:
    the response would already be a 200 `text/event-stream` by the time the bucket was
    consulted, and the client would receive a stream carrying an error in a frame instead
    of the typed 429 body - which is also a response no `Retry-After` can be added to,
    because the headers are already on the wire."""
    drain_requests(limit_store)

    response = post(limit_client, chat_body(model=ALIAS, stream=True))

    assert_error_body(response, status=429, code="RATE_LIMITED")
    assert chat_stub.requests == []


def test_a_refused_request_dials_no_provider_and_books_no_row(
    limit_client: TestClient, limit_store: RecordingRateLimitStore, chat_stub: StubAdapter
) -> None:
    """Criterion 1's other three halves in the arrangement that can see them.

    The request is exact-cache eligible and identical to one sent afterwards against a
    full bucket, so "no cache entry was stored" is asserted as the LATER request MISSING -
    a refusal that stored the answer it never got would make it a hit. Catches a check
    that runs after the provider call (the call the limit exists to prevent has already
    been made), after `record_cost` (the ledger carries a row for a request that was
    refused), and a refusal path that writes the response into the cache on its way
    out."""
    drain_requests(limit_store)

    refused = post(limit_client, eligible_body())

    assert refused.status_code == 429
    assert chat_stub.requests == [], "a provider was dialled for a request that was refused"
    assert ledger_rows(limit_client) == ()

    app_of(limit_client).state.rate_limit_store = RecordingRateLimitStore(limit_store.clock)
    served = post(limit_client, eligible_body())

    assert served.status_code == 200, served.text
    assert served.headers[CACHE_HEADER] == CACHE_STATUS_MISS, (
        "the refused request stored an answer in the cache"
    )


def test_a_refused_request_takes_nothing_from_the_token_bucket(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """A request refused on `rpm` never reaches `book_cost`, so it owes the token bucket
    nothing beyond the check's own probe. Catches a `debit_tokens` wired into an
    `except`/`finally` or into the response path rather than into `book_cost`, which
    would charge a tenant's `tpm` for traffic no provider ever served and turn one
    `rpm` refusal into a cascade of `tpm` ones."""
    drain_requests(limit_store)

    refused = post(limit_client, chat_body(model=ALIAS))

    assert refused.status_code == 429
    assert all(
        take[2] == CHECK_COST for take in limit_store.dimension_takes(RateDimension.TOKENS)
    ), "a refused request debited a real usage against tpm"


def test_the_bucket_refills_and_the_next_request_is_served(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Criterion 3 through the route, with the clock injected rather than slept for.

    A tenant refused at t is served again once one whole token has refilled: at 2 tokens a
    second that is half a second, and a quarter second is deliberately not enough. Catches
    a fixed-window counter behind the route, which would refuse both of these and only
    serve on the minute boundary - the 2x burst decision A rejects, seen from the other
    side. The middle request is what stops this passing against a limiter that simply
    forgets its state between requests.

    The half second is owed from the MIDDLE request and not from `FROZEN`: a take
    consumes whether or not it was allowed (`RateLimitStore`'s docstring), so the
    refused probe at +0.25 spends the half token that had refilled by then (`REQUEST_RATE`
    is 2 tokens a second). +0.75
    is the first moment a whole token stands against the bucket again."""
    drain_requests(limit_store)
    assert post(limit_client, chat_body(model=ALIAS)).status_code == 429

    limit_store.clock[0] = FROZEN + 0.25
    too_early = post(limit_client, chat_body(model=ALIAS))

    limit_store.clock[0] = FROZEN + 0.75
    on_time = post(limit_client, chat_body(model=ALIAS))

    assert too_early.status_code == 429
    assert on_time.status_code == 200, on_time.text


# --------------------------------------------------------------------------
# the order (criterion 6)
# --------------------------------------------------------------------------


def test_the_rate_check_runs_before_the_budget_check(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Criterion 6, stated as the only arrangement that can see an ORDER rather than a
    presence: a tenant over BOTH.

    Both refusals are 429s, so the status cannot distinguish them and the typed `code` is
    the whole assertion. Catches `check_rate_limit` called after `check_budget`, which
    answers `BUDGET_EXCEEDED` and tells a client to top up its budget when what it
    actually needs to do is slow down - and, worse, gives it no `Retry-After` to slow down
    by. The second half is the control that keeps this from passing for the wrong reason:
    the SAME app with the buckets full answers `BUDGET_EXCEEDED`, so it is the ordering
    and not a rate check that swallowed the budget one that produces the first answer."""
    over_budget(limit_client)
    drain_requests(limit_store)

    over_both = post(limit_client, chat_body(model=ALIAS))

    assert_error_body(over_both, status=429, code="RATE_LIMITED")

    app_of(limit_client).state.rate_limit_store = RecordingRateLimitStore(limit_store.clock)
    over_budget_only = post(limit_client, chat_body(model=ALIAS))

    assert_error_body(over_budget_only, status=429, code="BUDGET_EXCEEDED")


# --------------------------------------------------------------------------
# Retry-After (criterion 7, decision D)
# --------------------------------------------------------------------------


def test_the_429_carries_retry_after_in_whole_seconds(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Criterion 7. At `rpm: 120` one token lands every half second, and `Retry-After` is
    defined in whole seconds, so the value is 1.

    Catches the header being absent, which leaves every client guessing and is the reason
    decision D exists. Catches a float (`0.5`), which is not a legal `Retry-After` and
    breaks a client that parses it as an integer. Catches `0`, which tells a client to
    retry immediately and turns one refusal into a storm. The rounding-up boundary itself
    is a pure function and is pinned across nine rows in tests/limits/test_ratelimit.py;
    what this case adds is that the number reaches the WIRE."""
    drain_requests(limit_store)

    response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 429
    assert response.headers[RETRY_AFTER_HEADER] == "1"


def test_a_served_response_carries_no_retry_after(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Decision D's "on this error only", first half. Catches a header added from the
    store's answer unconditionally, which would tell a client to back off from a request
    that was served - the one instruction a working gateway must not give."""
    response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert RETRY_AFTER_HEADER not in response.headers


def test_a_budget_refusal_carries_no_retry_after(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Decision D's "on this error only", second half, and the row that makes the claim
    non-trivial: `BUDGET_EXCEEDED` is also a 429.

    Catches the header added in `error_response` by STATUS - `if http_status == 429` - the
    obvious implementation, which would tell a tenant that has spent its month to come
    back in a second, for ever. The header has to come from the ERROR carrying a
    `retry_after_seconds`, and a budget refusal carries none. Nothing else in this file
    distinguishes the two implementations, because every other 429 here is a rate
    limit."""
    over_budget(limit_client)

    response = post(limit_client, chat_body(model=ALIAS))

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert RETRY_AFTER_HEADER not in response.headers


def test_a_tenant_refusal_carries_no_retry_after(limit_client: TestClient) -> None:
    """The other typed error a request can meet on this app, so "only when the error carries
    one" is asserted across two statuses rather than one. Catches a `Retry-After` derived
    from anything the request holds rather than from the error itself - a header written
    from `app.state` or from the tenant's policy would land on this 403 too, and telling a
    client to retry a tenant refusal is telling it to retry for ever."""
    response = post(limit_client, chat_body(model=ALIAS), tenant=UNPOLICED_TENANT)

    assert_error_body(response, status=403, code="TENANT_MISMATCH")
    assert RETRY_AFTER_HEADER not in response.headers


# --------------------------------------------------------------------------
# an unreachable limiter (criterion 4, decision E)
# --------------------------------------------------------------------------


def test_an_unreachable_limiter_serves_and_refuses_nobody(limit_client: TestClient) -> None:
    """Criterion 4 end to end. Catches a route that lets the store's failure reach the
    client as a 500 on a request every provider would have served, and catches the
    fail-CLOSED over-correction: reading `None` as "refused" turns a Redis blip into a
    total outage, which is precisely the failure decision E weighed and rejected. The
    served path is exercised to completion - a row is booked - so a `debit_tokens` that
    raises on a `None` from the same store is caught here too, on the path where the
    request has already been paid for."""
    app_of(limit_client).state.rate_limit_store = unreachable_store()

    response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert len(ledger_rows(limit_client)) == 1


def test_every_swallowed_failure_is_logged_with_the_tenant_and_the_dimension(
    limit_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The other half of decision E, and the half that is easy to leave out.

    Failing open means a ceiling silently stops being enforced, and decision F adds no
    metric family, so this log is the ONLY place the window is visible at all. Catches a
    bare `except redis.RedisError: return None`, which is the obvious implementation and
    leaves an operator unable to tell a quiet minute from an unenforced one. Both
    dimensions are asserted because they fail independently and the answer to "was it
    `rpm` or `tpm` that stopped being enforced" changes what an operator does next; the
    values are read off the record rather than out of the formatted string, so the
    wording stays the implementation's to choose."""
    app_of(limit_client).state.rate_limit_store = unreachable_store()

    with caplog.at_level(logging.ERROR, logger=RATE_LIMIT_LOGGER):
        response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    records = [
        record
        for record in caplog.records
        if record.name == RATE_LIMIT_LOGGER and record.levelno == logging.ERROR
    ]
    assert records != [], "a limiter stopped enforcing and nothing said so"
    assert {record.__dict__["tenant"] for record in records} == {TENANT}
    assert {record.__dict__["dimension"] for record in records} == {"req", "tok"}


# --------------------------------------------------------------------------
# a tenant the policy does not carry (criterion 5)
# --------------------------------------------------------------------------


def test_a_tenant_the_policy_does_not_carry_is_still_the_resolvers_403(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """The rate check runs before the resolver, so it meets tenants the policy has never
    heard of - a key the store carries with no `routing.yaml` entry.

    Catches a ceiling read as `policy.tenants[tenant]`, which raises `KeyError` and turns
    today's typed 403 into an untyped 500; and catches a check that invents a ceiling of
    zero for such a tenant, which would answer 429 and hide the tenant refusal that is
    the correct answer. Deny-by-default is the resolver's job here, not the limiter's
    . `takes == []` is what says the limiter did not merely happen to allow
    it - it never opened a bucket for a tenant with no published limit, so no key with an
    unpoliced name is ever written."""
    response = post(limit_client, chat_body(model=ALIAS), tenant=UNPOLICED_TENANT)

    assert_error_body(response, status=403, code="TENANT_MISMATCH")
    assert limit_store.takes == []


# --------------------------------------------------------------------------
# the check's own cost (decision C)
# --------------------------------------------------------------------------


def test_the_check_takes_one_from_each_bucket_at_the_tenants_own_ceiling(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Decision C, and the capacities that go with it.

    Catches a check that takes from the request bucket only, which would leave `tpm`
    unenforceable until a debit happened to empty it; a check that takes MORE than one,
    which silently caps a tenant below its published `rpm`; and - the one no refusal case
    can see - a check that passes some other tenant's capacity or a hardcoded rate, which
    would give every tenant the same ceiling whatever `routing.yaml` says. The whole
    five-tuple is asserted, so a swapped `capacity`/`rate` pair fails here."""
    response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert limit_store.takes[0] == (TENANT, RateDimension.REQUESTS, CHECK_COST, RPM, REQUEST_RATE)
    assert limit_store.takes[1] == (TENANT, RateDimension.TOKENS, CHECK_COST, TPM, TOKEN_RATE)


def test_one_http_request_takes_exactly_one_request_token(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """A chat request against `agent-default` embeds internally for the semantic tier, so
    the route really does resolve a second alias while serving one request.

    Catches an implementation that checks the rate again on that internal path - the
    tenant would burn two request tokens per request and reach its ceiling at 60 requests
    a minute out of a published 120, with nothing in the response to explain it (standing
    rule 30's fourth target: the embedding adapter is live on this app)."""
    response = post(limit_client, question_body(QUESTION))

    assert response.status_code == 200, response.text
    assert len(limit_store.dimension_takes(RateDimension.REQUESTS)) == 1


# --------------------------------------------------------------------------
# the token debit (criterion 2, decision B)
# --------------------------------------------------------------------------

Send = Callable[[TestClient, RecordingRateLimitStore], Response]
"""One measured request, armed immediately before it.

The store is a parameter rather than a fixture read inside the sender so a
priming request's takes are never counted: `arm` clears what the priming did.
"""


def send_chat(client: TestClient, store: RecordingRateLimitStore) -> Response:
    """One non-streamed completion, answered by the provider."""
    arm(client, store)
    return post(client, chat_body(model=ALIAS))


def send_stream(client: TestClient, store: RecordingRateLimitStore) -> Response:
    """The same request, streamed, with the provider reporting usage."""
    arm(client, store)
    return post(client, chat_body(model=ALIAS, stream=True))


def send_exact_hit(client: TestClient, store: RecordingRateLimitStore) -> Response:
    """A second identical eligible request, served from the exact cache."""
    primed = post(client, eligible_body())
    assert primed.status_code == 200, primed.text
    arm(client, store)
    hit = post(client, eligible_body())
    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_HIT, "the arrangement did not hit"
    return hit


def send_semantic_hit(client: TestClient, store: RecordingRateLimitStore) -> Response:
    """The same question reworded, served from the semantic cache."""
    primed = post(client, question_body(QUESTION))
    assert primed.status_code == 200, primed.text
    arm(client, store)
    hit = post(client, question_body(SAME_QUESTION_REWORDED))
    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_SEMANTIC, "the arrangement did not hit"
    return hit


def send_embeddings(client: TestClient, store: RecordingRateLimitStore) -> Response:
    """One embeddings request against the `embed` alias."""
    arm(client, store)
    return post_embed(client, embeddings_body())


PATHS: list[tuple[str, Send, int]] = [
    ("the non-streamed chat path", send_chat, CHAT_TOKENS),
    ("the streamed chat path", send_stream, STREAM_TOKENS),
    ("the exact-cache hit path", send_exact_hit, ZERO_TOKENS),
    ("the semantic-cache hit path", send_semantic_hit, ZERO_TOKENS),
    ("the embeddings path", send_embeddings, EMBED_TOKENS),
]
"""The five `book_cost` call sites and the exact token count each owes. The two
cache-hit rows owe zero, which is precisely why the assertion below is about the
recorded TAKE and not about a bucket level: a level cannot tell a site that
debited zero from one that was never wired at all, and those two are the sites
most likely to be missed."""


@pytest.mark.parametrize(("label", "send", "expected"), PATHS, ids=[row[0] for row in PATHS])
def test_every_book_cost_site_debits_the_token_bucket(
    limit_client: TestClient,
    limit_store: RecordingRateLimitStore,
    label: str,
    send: Send,
    expected: int,
) -> None:
    """One row per `book_cost` call site: each debits once, by the tokens of the row it just
    booked, after the check's own probe.

    Catches a call site the wiring missed - the streamed sinks and the two cache-hit paths
    book their rows somewhere other than the closing stretch of the non-streamed path -
    which would leave that traffic invisible to `tpm` for ever. Catches a debit taken from
    the REQUEST bucket by a copied `dimension` argument, which would make one large
    request exhaust a tenant's whole minute of requests. Catches a debit of the total
    tokens where the row books prompt and completion separately, and one booked twice. The
    expected figure is asserted against both the literal from the stub's usage and the
    ledger row, so a route that books a row of the wrong size fails on the first and one
    that debits the wrong number fails on the second."""
    response = send(limit_client, limit_store)

    assert response.status_code == 200, response.text
    rows = ledger_rows(limit_client)
    assert len(rows) == 1, rows
    token_takes = limit_store.dimension_takes(RateDimension.TOKENS)
    assert [row[2] for row in token_takes] == [CHECK_COST, expected]
    assert token_takes[1] == (TENANT, RateDimension.TOKENS, expected, TPM, TOKEN_RATE)
    assert expected == (
        ZERO_TOKENS if rows[0].cached else rows[0].input_tokens + rows[0].output_tokens
    )


def test_a_cache_hit_debits_zero_tokens(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """A replayed answer bought no tokens from anybody, so it moves `tpm` by nothing - and it
    still moves it, once (criterion 2).

    This is the row where the obvious implementation is wrong. A cached ledger row KEEPS
    the replayed answer's token counts on purpose
    (`test_a_cached_row_keeps_the_replayed_answers_token_counts` in
    tests/storage/test_cost_ledger.py), so a `debit_tokens` that reads `input_tokens +
    output_tokens` off the row charges a tenant's `tpm` again for every hit and makes the
    cache actively expensive in the one currency it was meant to save. Catches exactly
    that, and catches the opposite - a hit that skips the debit entirely, which no bucket
    level could tell from a hit that debited zero. The row's own non-zero counts are
    asserted so the case cannot pass because the cache happened to book an empty row."""
    primed = post(limit_client, eligible_body())
    assert primed.status_code == 200, primed.text
    arm(limit_client, limit_store)

    hit = post(limit_client, eligible_body())

    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_HIT
    row = ledger_rows(limit_client)[0]
    assert row.cached is True
    assert row.input_tokens + row.output_tokens > 0, "the arrangement cannot see the bug"
    assert [take[2] for take in limit_store.dimension_takes(RateDimension.TOKENS)] == [
        CHECK_COST,
        ZERO_TOKENS,
    ]


def test_the_debit_is_booked_under_the_requesting_tenant(
    limit_client: TestClient, limit_store: RecordingRateLimitStore
) -> None:
    """Catches a debit booked under a tenant taken from the request body or from the
    `X-Tenant` header before authentication rather than from the authenticated context -
    which would let any caller spend a neighbour's `tpm` by naming it, and would leave the
    real tenant's ceiling unenforceable."""
    response = post(limit_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
    assert {take[0] for take in limit_store.takes} == {TENANT}


# --------------------------------------------------------------------------
# the neighbouring family (decision F)
# --------------------------------------------------------------------------


def test_a_rate_refusal_is_still_counted_once(
    limit_client: TestClient, limit_store: RecordingRateLimitStore, metrics: MetricDelta
) -> None:
    """Decision F: no new metric family, because `gateway_requests_total{status}` already
    counts the 429 - which is only true if the refusal is counted.

    A check inserted at the very top of both routes is exactly the kind of early return
    that skips or double-counts the request counter. Catches a `check_rate_limit` that
    answers before the metrics record is opened, which would leave every rate refusal
    uncounted and every error-rate panel blind to a rate-limit incident - and decision F's
    whole argument for adding no family rests on this counter seeing them."""
    drain_requests(limit_store)

    refused = post(limit_client, chat_body(model=ALIAS))

    assert refused.status_code == 429
    assert metrics.total(REQUESTS) == 1


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------

UNCONNECTED_REDIS_URL = "redis://127.0.0.1:1/0"
"""A Redis nothing listens on. `redis.Redis` connects lazily, so building the
store costs no connection and the URL is never dialled by the two cases below."""


def wiring_settings(*, redis_url: str | None) -> Settings:
    """The test key store and the shipped policy, with the Redis URL stated either way.

    Stated rather than left to default, for the reason
    tests/gateway/test_exact_cache.py states it: `Settings` reads
    `OPSCONSOLE_REDIS_URL` from the process environment, which the Redis suite asks a
    developer to export, and the unconfigured row would then assert the opposite of
    what it says.
    """
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, redis_url=redis_url)


def test_create_app_keeps_the_in_process_rate_limit_store_without_a_redis_url() -> None:
    """Catches a factory that requires Redis to enforce a ceiling at all: the whole existing
    suite, `make verify` and a bare `uvicorn` run configure no `redis_url`, so a
    Redis-only factory would leave `app.state.rate_limit_store` absent and turn the first
    request into an `AttributeError` with no code."""
    app = create_app(wiring_settings(redis_url=None), NO_PROVIDER_KEYS)

    # The store is built on lifespan startup now, so enter it first.
    with TestClient(app):
        assert isinstance(app.state.rate_limit_store, InProcessRateLimitStore)


def test_create_app_uses_the_redis_rate_limit_store_when_a_redis_url_is_configured() -> None:
    """Catches a factory that keeps a tenant's buckets in process memory while Redis is
    configured: two gateway replicas would then each grant the full `rpm`, so a tenant
    would be served twice its published rate and no test with one replica could ever see
    it except here at the wiring - which is also the race decision A's Lua script exists
    to close."""
    app = create_app(wiring_settings(redis_url=UNCONNECTED_REDIS_URL), NO_PROVIDER_KEYS)

    with TestClient(app):
        assert isinstance(app.state.rate_limit_store, RedisRateLimitStore)
