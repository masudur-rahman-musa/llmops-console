"""The budget decided from the ledger's month spend, through the chat route.

A real request through `TestClient` against an app `create_app` built. The counter is the
in-process store (no Redis URL), and `app.state.month_spend` is a fake reader that counts its
calls and answers what a case arranges - the ledger, without a database.

Tenant `nordlicht`: shipped `budget_usd_month` 15.00 and `user_budget_usd_month` 3.00 (pinned
by the first case, so an edit to `routing.yaml` fails there and not by making every
arrangement below meaningless).

The contract these cases are written against, before it exists:

* `Clients.month_spend: MonthSpendReader`, handed from `app.state.month_spend`;
* `check_budget` and `check_user_budget` read the month through `seeded_spend` /
  `seeded_user_spend` with a loader over that reader;
* a refusal carries the exact published messages, with the contact from
  `Settings.budget_contact`;
* `LedgerUnreadable` becomes 503 `BUDGET_UNAVAILABLE`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Iterator
from decimal import Decimal

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
from app.storage import InMemoryCostLedgerWriter
from app.storage.cost_ledger import LedgerUnreadable

from .conftest import (
    API_KEYS,
    CHAT,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    StubAdapter,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
)

TENANT = "nordlicht"
ALIAS = "agent-default"
PROVIDER = "anthropic"
USER = "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c"
CONTACT = "budget-owner@example.com"
SECRET = "test-budget-override-secret-belonging-to-nobody"
BUDGET_LOGGER = "app.limits.budget"

LEDGER_OVER = 15_070_000
LEDGER_AT_EIGHTY = 12_000_000
LEDGER_USER_AT_CAP = 3_000_000
LEDGER_LOW = 1_000_000

TENANT_MESSAGE = (
    "Your tenant nordlicht has used its monthly budget of 15.00 USD. "
    "Ask budget-owner@example.com to allow more."
)
TENANT_MESSAGE_NO_CONTACT = "Your tenant nordlicht has used its monthly budget of 15.00 USD."
USER_MESSAGE = (
    "You have used your personal monthly limit of 3.00 USD in tenant nordlicht. "
    "Ask budget-owner@example.com to allow more."
)


class FakeMonthSpend:
    """A ledger reader that answers what a case arranged and counts every call."""

    def __init__(self) -> None:
        self.tenant_spend: int | None = None
        self.user_spend: int | None = None
        self.unreadable = False
        self.tenant_calls: list[tuple[str, str]] = []
        self.user_calls: list[tuple[str, str, str]] = []

    def tenant_month_spend(self, tenant: str, month: str) -> int | None:
        self.tenant_calls.append((tenant, month))
        if self.unreadable:
            raise LedgerUnreadable("ledger down")
        return self.tenant_spend

    def user_month_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        self.user_calls.append((tenant, user_id, month))
        if self.unreadable:
            raise LedgerUnreadable("ledger down")
        return self.user_spend


@pytest.fixture(scope="module")
def chat_stub() -> StubAdapter:
    """The completion adapter under the policy provider `anthropic`."""
    return StubAdapter(name=PROVIDER)


def build(chat_stub: StubAdapter, *, contact: str) -> Iterator[TestClient]:
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        budget_override_secret=SECRET,
        budget_contact=contact,
    )
    adapters: dict[str, Adapter] = {chat_stub.name: chat_stub}
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture(scope="module")
def ledger_client(chat_stub: StubAdapter) -> Iterator[TestClient]:
    """One app with the contact set, built once."""
    yield from build(chat_stub, contact=CONTACT)


@pytest.fixture(scope="module")
def no_contact_client(chat_stub: StubAdapter) -> Iterator[TestClient]:
    """One app with the contact left empty, built once."""
    yield from build(chat_stub, contact="")


def arrange(client: TestClient, chat_stub: StubAdapter) -> FakeMonthSpend:
    """Fresh caches, ledger, empty counter, a fresh fake reader and cleared stub."""
    chat_stub.requests.clear()
    app = app_of(client)
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = InProcessSemanticCache()
    app.state.cost_ledger = InMemoryCostLedgerWriter()
    app.state.budget_store = InProcessBudgetStore()
    reader = FakeMonthSpend()
    app.state.month_spend = reader
    return reader


@pytest.fixture
def reader(ledger_client: TestClient, chat_stub: StubAdapter) -> FakeMonthSpend:
    """This test's ledger reader, on `app.state.month_spend` of the contact app."""
    return arrange(ledger_client, chat_stub)


def post(client: TestClient, *, user: str | None = None, override: str | None = None) -> Response:
    """One chat completion as `nordlicht`."""
    headers = auth_headers(key=TEST_KEYS[TENANT], tenant=TENANT, data_class="internal")
    if user is not None:
        headers["X-User"] = user
    if override is not None:
        headers[BUDGET_OVERRIDE_HEADER] = override
    response: Response = client.post(CHAT, json=chat_body(model=ALIAS), headers=headers)
    return response


def message_of(response: Response) -> str:
    return str(response.json()["error"]["message"])


def test_the_shipped_caps_for_this_tenant_are_the_ones_these_cases_arrange_against(
    policy: Policy,
) -> None:
    """Not a behaviour: catches an edit to `routing.yaml` that moves nordlicht's caps, after
    which the exact messages and the arrangements below would be about other numbers."""
    tenant = policy.tenants[TENANT]

    assert tenant.budget_usd_month == Decimal("15.00")
    assert tenant.user_budget_usd_month == Decimal("3.00")
    assert to_micro_usd(Decimal("15.00")) < LEDGER_OVER
    assert to_micro_usd(Decimal("15.00")) * 8 // 10 == LEDGER_AT_EIGHTY


def test_a_tenant_over_budget_in_the_ledger_is_refused_with_an_empty_counter(
    ledger_client: TestClient, reader: FakeMonthSpend, chat_stub: StubAdapter
) -> None:
    """Catches a gateway that decides from the empty Redis counter (the reported bug:
    nordlicht at $15.07 served), a refusal with the old generic message, and a refusal that
    still dials a provider. The ledger is asked for this tenant and this UTC month."""
    reader.tenant_spend = LEDGER_OVER

    response = post(ledger_client)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert message_of(response) == TENANT_MESSAGE
    assert chat_stub.requests == []
    assert reader.tenant_calls == [(TENANT, current_month())]


def test_the_message_names_no_contact_when_the_setting_is_empty(
    no_contact_client: TestClient, chat_stub: StubAdapter
) -> None:
    """Catches a message that always carries the second sentence (a dangling `Ask  to allow
    more.`) or a contact hard-coded in the message instead of read from the setting."""
    reader = arrange(no_contact_client, chat_stub)
    reader.tenant_spend = LEDGER_OVER

    response = post(no_contact_client)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert message_of(response) == TENANT_MESSAGE_NO_CONTACT


def test_a_user_over_their_cap_gets_the_personal_message(
    ledger_client: TestClient, reader: FakeMonthSpend, chat_stub: StubAdapter
) -> None:
    """Catches a user refusal that reuses the tenant wording, one that drops `param`
    `X-User`, one that puts the user id in the text, and a user counter that is
    not loaded from the ledger. The tenant is under its cap, so only the user check can
    refuse."""
    reader.tenant_spend = LEDGER_LOW
    reader.user_spend = LEDGER_USER_AT_CAP

    response = post(ledger_client, user=USER)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")
    assert response.json()["error"]["param"] == "X-User"
    assert message_of(response) == USER_MESSAGE
    assert USER not in response.text
    assert chat_stub.requests == []
    assert reader.user_calls == [(TENANT, USER, current_month())]


def test_a_tenant_at_eighty_percent_in_the_ledger_is_served_and_warned(
    ledger_client: TestClient, reader: FakeMonthSpend, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a warning that still reads the empty counter (no warning at 80%) and a
    check that refuses at 80%. Served, with exactly one warning that names the tenant."""
    reader.tenant_spend = LEDGER_AT_EIGHTY

    with caplog.at_level(logging.WARNING, logger=BUDGET_LOGGER):
        response = post(ledger_client)

    assert response.status_code == 200, response.text
    warnings = [
        r for r in caplog.records if r.name == BUDGET_LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert TENANT in warnings[0].getMessage()


def test_an_unreadable_ledger_refuses_with_budget_unavailable(
    ledger_client: TestClient, reader: FakeMonthSpend, chat_stub: StubAdapter
) -> None:
    """Catches a gateway that treats a ledger it cannot read as zero spend (serves a tenant
    that may be over its cap), one that answers 500, and one that lets the request reach a
    provider or book a row before refusing."""
    reader.unreadable = True

    response = post(ledger_client)

    assert_error_body(response, status=503, code="BUDGET_UNAVAILABLE")
    assert "try again" in message_of(response).lower()
    assert chat_stub.requests == []
    ledger = app_of(ledger_client).state.cost_ledger
    assert isinstance(ledger, InMemoryCostLedgerWriter)
    assert ledger.rows == ()


def test_the_ledger_is_read_once_per_month_not_per_request(
    ledger_client: TestClient, reader: FakeMonthSpend
) -> None:
    """Catches a gateway that reads the ledger on every request: the second request must
    find the counter loaded and never reach the reader."""
    reader.tenant_spend = LEDGER_LOW

    assert post(ledger_client).status_code == 200
    assert post(ledger_client).status_code == 200

    assert len(reader.tenant_calls) == 1


def test_a_valid_override_still_serves_over_a_loaded_total(
    ledger_client: TestClient, reader: FakeMonthSpend
) -> None:
    """Catches a new refusal path that skips the override check: a tenant over its loaded
    total with a good override header is served, as before."""
    reader.tenant_spend = LEDGER_OVER
    month = current_month()
    signature = hmac.new(
        SECRET.encode("utf-8"), f"{TENANT}:{month}".encode(), hashlib.sha256
    ).hexdigest()

    response = post(ledger_client, override=f"{month}:{signature}")

    assert response.status_code == 200, response.text
