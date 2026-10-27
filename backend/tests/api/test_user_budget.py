"""The per-user monthly cap inside a tenant's budget, through the routes.

Everything is settled by a real request through `TestClient` against an app
`create_app` built, with no Redis: `app.state.budget_store` is a
recording wrapper over the in-process twin. The spend is arranged through the store
rather than by sending requests until a cap is reached.

The tenant is `cuan`: its shipped `budget_usd_month` is 60.00 and its shipped
`user_budget_usd_month` 10.00 (pinned by the first case, so an edit to `routing.yaml`
fails there instead of making every over-cap arrangement below meaningless). Its
request rate (6000 a minute) keeps a module of requests clear of the rate limit.

The contract these cases are written against, before it exists:

* `BudgetStore.user_spend(tenant, user_id, month) -> int | None` and
  `BudgetStore.user_debit(tenant, user_id, month, micro_usd) -> int | None`, on both
  stores, beside `spend` and `debit`;
* the tenant cap is checked first, then the user cap, only when the request names a user
  and the tenant has a `user_budget_usd_month`; a user at or over it is 429
  `BUDGET_EXCEEDED` with `param` `X-User`;
* every booked row debits the tenant counter, and the user counter when it has a user.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator, Mapping
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.cache import InProcessResponseCache, InProcessSemanticCache
from app.core.config import Settings
from app.limits.budget import (
    BUDGET_OVERRIDE_HEADER,
    InProcessBudgetStore,
    current_month,
    to_micro_usd,
)
from app.model_routing.policy import Policy
from app.providers.base import Adapter
from app.storage import CostLedgerRow, CostLedgerWriter, InMemoryCostLedgerWriter

from .conftest import (
    API_KEYS,
    CHAT,
    EMBEDDINGS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    EmbeddingStubAdapter,
    StubAdapter,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
)

TENANT = "cuan"
ALIAS = "agent-default"
PROVIDER = "anthropic"
EMBED_PROVIDER = "local_embed"
DATA_CLASS = "internal"

USER_A = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"
USER_B = "u_00000000000000000000000000000002"

USER_CAP = 10_000_000
"""`cuan`'s shipped per-user cap in the counter's unit (10.00 USD)."""

TENANT_CAP = 60_000_000
"""`cuan`'s shipped tenant budget in the counter's unit (60.00 USD)."""

SECRET = "test-budget-override-secret-belonging-to-nobody"
"""A worthless string, not anyone's key."""


class RecordingUserBudgetStore:
    """The in-process twin with every call kept, user calls included.

    `user_unknown`, when set, makes `user_spend` answer `None` the way a Redis that
    cannot answer does.
    """

    def __init__(self) -> None:
        self.inner = InProcessBudgetStore()
        self.debits: list[tuple[str, str, int]] = []
        self.user_debits: list[tuple[str, str, str, int]] = []
        self.user_unknown = False

    def spend(self, tenant: str, month: str) -> int | None:
        return self.inner.spend(tenant, month)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        self.debits.append((tenant, month, micro_usd))
        return self.inner.debit(tenant, month, micro_usd)

    def user_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        if self.user_unknown:
            return None
        return self.inner.user_spend(tenant, user_id, month)

    def user_debit(self, tenant: str, user_id: str, month: str, micro_usd: int) -> int | None:
        self.user_debits.append((tenant, user_id, month, micro_usd))
        return self.inner.user_debit(tenant, user_id, month, micro_usd)

    def arrange(self, month: str, micro_usd: int) -> None:
        """Put a month's tenant spend on the counter without recording a debit."""
        self.inner.debit(TENANT, month, micro_usd)

    def arrange_user(self, user_id: str, month: str, micro_usd: int) -> None:
        """Put a month's spend for one user on the counter without recording a debit."""
        self.inner.user_debit(TENANT, user_id, month, micro_usd)


@pytest.fixture(scope="module")
def chat_stub() -> StubAdapter:
    """The completion adapter under the policy provider `anthropic`."""
    return StubAdapter(name=PROVIDER)


@pytest.fixture(scope="module")
def embed_stub() -> EmbeddingStubAdapter:
    """The embedding adapter under `local_embed`."""
    return EmbeddingStubAdapter(name=EMBED_PROVIDER)


@pytest.fixture(scope="module")
def user_client(chat_stub: StubAdapter, embed_stub: EmbeddingStubAdapter) -> Iterator[TestClient]:
    """One app for the module, with an override secret set, built once."""
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        budget_override_secret=SECRET,
    )
    adapters: dict[str, Adapter] = {chat_stub.name: chat_stub, embed_stub.name: embed_stub}
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh(
    user_client: TestClient, chat_stub: StubAdapter, embed_stub: EmbeddingStubAdapter
) -> Iterator[None]:
    """Empty caches, a fresh ledger and counter and cleared stubs - never a new app."""
    chat_stub.requests.clear()
    embed_stub.requests.clear()
    app = app_of(user_client)
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = InProcessSemanticCache()
    app.state.cost_ledger = InMemoryCostLedgerWriter()
    app.state.budget_store = RecordingUserBudgetStore()
    yield


@pytest.fixture
def store(user_client: TestClient) -> RecordingUserBudgetStore:
    """This test's counter, the one `_fresh` put on the app."""
    found = app_of(user_client).state.budget_store
    assert isinstance(found, RecordingUserBudgetStore)
    return found


def month() -> str:
    """The month a request made now is booked under."""
    return current_month()


def headers_for(user: str | None, *, override: str | None = None) -> dict[str, str]:
    """The `cuan`/`internal` headers, with `X-User` and the override only when given."""
    headers = auth_headers(key=TEST_KEYS[TENANT], tenant=TENANT, data_class=DATA_CLASS)
    if user is not None:
        headers["X-User"] = user
    if override is not None:
        headers[BUDGET_OVERRIDE_HEADER] = override
    return headers


def post(client: TestClient, user: str | None, *, override: str | None = None) -> Response:
    """One chat completion as `cuan`."""
    body: Mapping[str, Any] = chat_body(model=ALIAS)
    response: Response = client.post(
        CHAT, json=dict(body), headers=headers_for(user, override=override)
    )
    return response


def rows_of(client: TestClient) -> tuple[CostLedgerRow, ...]:
    """The rows this app's ledger holds right now."""
    writer: CostLedgerWriter = app_of(client).state.cost_ledger
    assert isinstance(writer, InMemoryCostLedgerWriter)
    return writer.rows


def override_token() -> str:
    """A genuine override for `cuan` this month, minted from the configured secret."""
    signature = hmac.new(
        SECRET.encode("utf-8"), f"{TENANT}:{month()}".encode(), hashlib.sha256
    ).hexdigest()
    return f"{month()}:{signature}"


def test_the_shipped_caps_for_this_tenant_are_the_ones_these_cases_arrange_against(
    policy: Policy,
) -> None:
    """Not a behaviour: the guard that keeps every case below meaningful. Catches an edit
    to `routing.yaml` that moves `cuan`'s caps, after which the arrangements would be under
    cap and the refusals would pass by serving."""
    tenant = policy.tenants[TENANT]

    assert tenant.user_budget_usd_month == Decimal("10.00")
    assert to_micro_usd(Decimal("10.00")) == USER_CAP
    assert to_micro_usd(tenant.budget_usd_month) == TENANT_CAP


def test_a_user_over_its_cap_is_refused(
    user_client: TestClient, store: RecordingUserBudgetStore, chat_stub: StubAdapter
) -> None:
    """Catches a user cap that is never checked, one checked with `>` (the spend is exactly
    the cap, which must be refused like the tenant's), and a refusal raised as a bare 429
    with no typed code or no `param` naming `X-User`. Also no provider call and no row: the
    refusal comes before anything is dialled."""
    store.arrange_user(USER_A, month(), USER_CAP)

    response = post(user_client, USER_A)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert response.json()["error"]["param"] == "X-User"
    assert USER_A not in response.text
    assert chat_stub.requests == []
    assert rows_of(user_client) == ()


def test_a_user_one_micro_under_its_cap_is_served(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """The other side of the boundary, and what stops the refusal above passing against a
    route that refuses every user: a spend one micro under the cap is served."""
    store.arrange_user(USER_A, month(), USER_CAP - 1)

    response = post(user_client, USER_A)

    assert response.status_code == 200, response.text


def test_another_user_of_the_same_tenant_goes_on(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches a user cap counted on the tenant (one heavy user locks out the whole
    company) or keyed without the user. User A is over; user B of the same tenant is
    served."""
    store.arrange_user(USER_A, month(), USER_CAP)

    assert_error_body(post(user_client, USER_A), status=429, code="BUDGET_EXCEEDED")
    response = post(user_client, USER_B)

    assert response.status_code == 200, response.text


def test_a_request_with_no_user_skips_the_user_cap(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches a request with no `X-User` counted under a shared 'no user' bucket and
    refused with it. A user is over its cap; the anonymous request is served."""
    store.arrange_user(USER_A, month(), USER_CAP)

    response = post(user_client, None)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("user_over", [False, True], ids=["user under", "user over"])
def test_the_tenant_cap_still_refuses_first(
    user_over: bool, user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches the user cap checked before the tenant cap: with BOTH over, the answer must
    be the tenant's refusal (no `param` `X-User`), so a company that is out of money is told
    that and not that one user is. The 'user under' row is the plain tenant refusal."""
    store.arrange(month(), TENANT_CAP)
    if user_over:
        store.arrange_user(USER_A, month(), USER_CAP)

    response = post(user_client, USER_A)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert response.json()["error"]["param"] != "X-User"


def test_a_served_request_debits_tenant_and_user(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches a debit of the tenant counter alone (the user cap never fills) or of the
    user counter alone (the tenant budget stops moving), and a user debit of any amount but
    the row's own cost. Both counters rise by exactly the booked row's cost."""
    response = post(user_client, USER_A)

    assert response.status_code == 200, response.text
    (row,) = rows_of(user_client)
    cost = to_micro_usd(row.cost_usd)
    assert cost > 0
    assert store.debits == [(TENANT, month(), cost)]
    assert store.user_debits == [(TENANT, USER_A, month(), cost)]
    assert store.inner.spend(TENANT, month()) == cost
    assert store.inner.user_spend(TENANT, USER_A, month()) == cost


def test_a_request_with_no_user_debits_no_user_counter(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches a debit filed under a made-up user for an anonymous request. The tenant is
    still debited."""
    response = post(user_client, None)

    assert response.status_code == 200, response.text
    assert len(store.debits) == 1
    assert store.user_debits == []


def test_a_valid_override_serves_a_user_over_its_cap(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches an override honoured for the tenant cap only. The control is the same
    request without the header, refused, so the genuine token alone flips the result."""
    store.arrange_user(USER_A, month(), USER_CAP)

    refused = post(user_client, USER_A)
    served = post(user_client, USER_A, override=override_token())

    assert_error_body(refused, status=429, code="BUDGET_EXCEEDED")
    assert served.status_code == 200, served.text


def test_a_counter_that_cannot_answer_serves(
    user_client: TestClient, store: RecordingUserBudgetStore
) -> None:
    """Catches a user cap that fails closed (a Redis blip refuses every user) or that
    crashes on `None` (a 500). The tenant counter answers, only the user counter is
    unknown."""
    store.user_unknown = True

    response = post(user_client, USER_A)

    assert response.status_code == 200, response.text


def test_the_embeddings_route_refuses_a_user_over_its_cap_too(
    user_client: TestClient, store: RecordingUserBudgetStore, embed_stub: EmbeddingStubAdapter
) -> None:
    """Catches the user check wired into the chat route alone: the user keeps spending
    through `/v1/embeddings` after the cap is reached."""
    store.arrange_user(USER_A, month(), USER_CAP)
    headers = headers_for(USER_A)

    response = user_client.post(EMBEDDINGS, json=dict(embeddings_body()), headers=headers)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert response.json()["error"]["param"] == "X-User"
    assert embed_stub.requests == []
