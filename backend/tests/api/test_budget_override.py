"""`X-Gateway-Budget-Override`: the one way past a reached cap, and every way that is not.

The signature rules themselves are tested as a function in
tests/limits/test_budget.py; what this file settles is the half no unit test can
- that the tenant the signature is checked against is the AUTHENTICATED one and
not anything the caller sent, that the month is the gateway's and not the
header's, and that a request carrying nothing at all is still the 429 (standing
rule 12: a claim about HTTP shape is settled by a `TestClient` call against the
built app).

Two apps, because the secret is a setting and the shipped default is empty:

* `override_client` has `budget_override_secret` set, and is where the accept
  case and the refusals live;
* `no_secret_client` is the shipped default, where a token that is valid
  everywhere else has to be refused. HMAC over an empty key is still perfectly
  good HMAC, so an implementation that does not refuse an unset secret outright
  hands an override to anyone who can read this repository (decision A).

a project rule: both apps are module-scoped and built once. Per-test isolation
is a fresh `budget_store` on `app.state`, never a new app.

The contract these cases are written against, before it exists:

* `Settings.budget_override_secret` (`OPSCONSOLE_BUDGET_OVERRIDE_SECRET`),
  defaulting to the empty string;
* both routes read `X-Gateway-Budget-Override` and serve an over-cap request
  only when `verify_override` accepts it for the authenticated tenant and the
  current month;
* an accepted override changes nothing else: the request is still priced, still
  booked and still debited.
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
from app.core.settings import Settings
from app.ledger import InMemoryCostLedgerWriter
from app.limits.budget import (
    BUDGET_OVERRIDE_HEADER,
    MAX_OVERRIDE_HEADER_CHARS,
    InProcessBudgetStore,
    current_month,
    to_micro_usd,
)
from app.providers.base import Adapter

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

TENANT = "evals"
OTHER_TENANT = "cuan"
DATA_CLASS = "internal"
SESSION = "sess-budget-override"
ALIAS = "agent-default"
PROVIDER = "anthropic"
EMBED_PROVIDER = "local_embed"

CAP_USD = Decimal("10.00")
CAP = 10_000_000
"""`evals`'s shipped cap, in the counter's unit. Pinned against the loaded policy
in tests/gateway/test_budget_enforcement.py, which is the file that arranges
against it in every case."""

SECRET = "test-budget-override-secret-belonging-to-nobody"
"""A worthless string, not anyone's key. A real secret lives in
`.env` locally and in a SOPS-encrypted secret in the cluster, never here."""

OTHER_SECRET = "test-budget-override-secret-that-is-not-the-one-configured"


def signature(*, tenant: str, month: str, secret: str) -> str:
    """HMAC-SHA256 over `tenant:month`, hex, exactly as decision A specifies it."""
    return hmac.new(
        secret.encode("utf-8"), f"{tenant}:{month}".encode(), hashlib.sha256
    ).hexdigest()


def token(*, tenant: str, month: str, secret: str = SECRET, stated_month: str | None = None) -> str:
    """One override header. `stated_month` is what the header SAYS, defaulting to the
    month that was signed, so a row can carry a signature over one month and a field
    naming another."""
    stated = stated_month if stated_month is not None else month
    return f"{stated}:{signature(tenant=tenant, month=month, secret=secret)}"


def previous_month(month: str) -> str:
    """The month before this one, in the same `YYYY-MM` spelling."""
    year, number = (int(part) for part in month.split("-"))
    return f"{year - 1}-12" if number == 1 else f"{year}-{number - 1:02d}"


# --------------------------------------------------------------------------
# the two apps
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def override_settings() -> Settings:
    """The test key store, the shipped policy, and an override secret that is set."""
    return Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        budget_override_secret=SECRET,
    )


@pytest.fixture(scope="module")
def no_secret_settings() -> Settings:
    """The same, with `budget_override_secret` left at its shipped empty default."""
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH)


def adapters() -> dict[str, Adapter]:
    """A completion adapter under `anthropic` and an embedding one under `local`."""
    return {
        PROVIDER: StubAdapter(name=PROVIDER),
        EMBED_PROVIDER: EmbeddingStubAdapter(name=EMBED_PROVIDER),
    }


@pytest.fixture(scope="module")
def override_client(override_settings: Settings) -> Iterator[TestClient]:
    """The app with a secret configured, built once."""
    with build_chat_client(override_settings, adapters()) as client:
        yield client


@pytest.fixture(scope="module")
def no_secret_client(no_secret_settings: Settings) -> Iterator[TestClient]:
    """The shipped default: an app that has no override secret at all."""
    with build_chat_client(no_secret_settings, adapters()) as client:
        yield client


def over_cap(client: TestClient) -> InProcessBudgetStore:
    """Put a fresh counter on the app, already at exactly the cap, and answer it."""
    store = InProcessBudgetStore()
    store.debit(TENANT, current_month(), CAP)
    app = app_of(client)
    app.state.budget_store = store
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = InProcessSemanticCache()
    app.state.cost_ledger = InMemoryCostLedgerWriter()
    return store


def post(
    client: TestClient,
    body: Mapping[str, Any],
    *,
    override: str | None = None,
    tenant: str = TENANT,
) -> Response:
    """POST one completion, with the override header only when a case sends one."""
    headers = auth_headers(
        key=TEST_KEYS[tenant], tenant=tenant, data_class=DATA_CLASS, session=SESSION
    )
    if override is not None:
        headers[BUDGET_OVERRIDE_HEADER] = override
    response: Response = client.post(CHAT, json=dict(body), headers=headers)
    return response


def post_embed(client: TestClient, override: str | None = None) -> Response:
    """The same, on `/v1/embeddings`."""
    headers = auth_headers(
        key=TEST_KEYS[TENANT], tenant=TENANT, data_class=DATA_CLASS, session=SESSION
    )
    if override is not None:
        headers[BUDGET_OVERRIDE_HEADER] = override
    response: Response = client.post(EMBEDDINGS, json=dict(embeddings_body()), headers=headers)
    return response


# --------------------------------------------------------------------------
# the accept case, and the control that makes it mean something
# --------------------------------------------------------------------------


def test_an_over_cap_request_without_the_header_is_refused(override_client: TestClient) -> None:
    """The control. Without it every case below would pass against a gateway that never
    enforces the cap on this app at all, and the accept case would be evidence of
    nothing. Catches an app whose budget check is not wired, and an override secret whose
    mere presence disables enforcement."""
    over_cap(override_client)

    response = post(override_client, chat_body(model=ALIAS))

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_valid_override_serves_a_tenant_over_cap(override_client: TestClient) -> None:
    """The one thing the header exists for.

    Catches a header that is read but never verified against anything, and - with the
    control above - a verification that can never succeed, which is the failure an
    operator discovers during the incident the override was minted for. The debit is
    asserted too: an override is permission to spend, not permission to stop counting, so
    an implementation that skipped the booking would leave the counter frozen at the cap
    and the ratio flat for the rest of the month."""
    store = over_cap(override_client)

    response = post(
        override_client,
        chat_body(model=ALIAS),
        override=token(tenant=TENANT, month=current_month()),
    )

    assert response.status_code == 200, response.text
    spent = store.spend(TENANT, current_month())
    assert spent is not None and spent > CAP
    assert spent == CAP + to_micro_usd(Decimal(response.headers["x-gateway-cost-usd"]))


def test_a_valid_override_serves_the_embeddings_route_too(override_client: TestClient) -> None:
    """Catches an override read in the chat route alone: the two routes are refused by one
    check, so they have to be excused by one check as well, and an operator who has
    minted a token would otherwise find half their traffic still refused."""
    over_cap(override_client)

    response = post_embed(override_client, token(tenant=TENANT, month=current_month()))

    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------
# every token that is not this tenant's, this month's, ours
# --------------------------------------------------------------------------


def test_a_token_minted_for_another_tenant_is_still_refused(override_client: TestClient) -> None:
    """Section 5's second risk, and the reason the tenant comes from the authenticated
    context rather than from the token.

    The token is genuine - this gateway's secret, this month - and simply belongs to
    `cuan`. Catches an implementation that reads the tenant out of the header, out of the
    request body, or that verifies the signature against nothing but the month: any of
    those turns one captured token into a permanent bypass for every tenant on the
    gateway."""
    over_cap(override_client)

    response = post(
        override_client,
        chat_body(model=ALIAS),
        override=token(tenant=OTHER_TENANT, month=current_month()),
    )

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_token_for_last_month_is_still_refused(override_client: TestClient) -> None:
    """The other half of what the signature is scoped to. The token is this tenant's and
    this gateway's, and was valid a month ago. Catches a signature computed over the
    tenant alone, which never expires - an override handed out once during an incident
    then excuses that tenant's spending for ever, which is the difference between a
    temporary permission and a permanent hole."""
    over_cap(override_client)
    last = previous_month(current_month())

    response = post(
        override_client, chat_body(model=ALIAS), override=token(tenant=TENANT, month=last)
    )

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_token_signed_for_last_month_but_stating_this_one_is_still_refused(
    override_client: TestClient,
) -> None:
    """The sharper version of the case above: the header's month FIELD says the current
    month, so a check that reads the month out of the header and verifies the signature
    against that would accept last month's signature. The month has to come from the
    gateway's own clock on both sides of the comparison."""
    over_cap(override_client)
    last = previous_month(current_month())

    response = post(
        override_client,
        chat_body(model=ALIAS),
        override=token(tenant=TENANT, month=last, stated_month=current_month()),
    )

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_token_signed_with_another_secret_is_still_refused(
    override_client: TestClient,
) -> None:
    """The right tenant, the right month, the wrong key. Catches a comparison that checks
    only the SHAPE of the token - a `YYYY-MM` and sixty-four hex characters - which
    anyone can produce, and a comparison written against a signature the caller supplied
    on both sides."""
    over_cap(override_client)

    response = post(
        override_client,
        chat_body(model=ALIAS),
        override=token(tenant=TENANT, month=current_month(), secret=OTHER_SECRET),
    )

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


MALFORMED: list[tuple[str, str]] = [
    ("empty", ""),
    ("the month alone", "2026-09"),
    ("no separator", "2026-09abcdef"),
    ("a signature that is not hex", f"{current_month()}:{'z' * 64}"),
    ("a second separator after a valid token", f"{token(tenant=TENANT, month=current_month())}:x"),
    ("the signature alone", signature(tenant=TENANT, month=current_month(), secret=SECRET)),
]
"""The malformed values a client can actually put on the wire. A trailing newline
and a trailing CRLF are not among them - httpx refuses to send either, and the
server would strip surrounding whitespace anyway - so those two rows are checked
against `verify_override` directly, in tests/limits/test_budget.py, where they
are what a project rule is about."""


@pytest.mark.parametrize(("label", "header"), MALFORMED, ids=[row[0] for row in MALFORMED])
def test_a_malformed_override_is_still_refused(
    override_client: TestClient, label: str, header: str
) -> None:
    """The parsing rows through the route rather than through the function, because a
    header is not a string until the server has decoded it. Catches a parser that raises
    on a value with no separator - an unhandled `ValueError` from `split` is a 500 on a
    request whose correct answer is a 429 - and one that splits on the LAST colon, or on
    every colon and keeps the first two pieces, either of which accepts a token with
    something appended to it."""
    over_cap(override_client)

    response = post(override_client, chat_body(model=ALIAS), override=header)

    assert response.status_code == 429, response.text
    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_an_over_length_override_header_is_still_refused(override_client: TestClient) -> None:
    """A genuine token with ten thousand characters of padding after it. Catches a parser
    that slices a fixed prefix out of an unbounded header, which would accept this
    because its first seventy-two characters are valid, and any implementation that
    hashes the whole header before measuring it."""
    over_cap(override_client)
    padded = token(tenant=TENANT, month=current_month()) + "x" * 9_928

    assert len(padded) > MAX_OVERRIDE_HEADER_CHARS
    response = post(override_client, chat_body(model=ALIAS), override=padded)

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


# --------------------------------------------------------------------------
# the shipped default: no secret at all
# --------------------------------------------------------------------------


def test_a_gateway_with_no_override_secret_refuses_a_token_minted_from_the_empty_one(
    no_secret_client: TestClient,
) -> None:
    """The shipped default is an empty secret, and HMAC over an empty key is still valid
    HMAC. Catches an implementation that feeds the configured secret straight into
    `hmac.new` without asking whether one is configured: on every deployment that has not
    set `OPSCONSOLE_BUDGET_OVERRIDE_SECRET`, anyone who knows the scheme can mint a
    working override, and the tenant with the reached cap is the least of it."""
    over_cap(no_secret_client)

    response = post(
        no_secret_client,
        chat_body(model=ALIAS),
        override=token(tenant=TENANT, month=current_month(), secret=""),
    )

    assert_error_body(response, status=429, code="BUDGET_EXCEEDED")


def test_a_gateway_with_no_override_secret_still_serves_under_cap(
    no_secret_client: TestClient,
) -> None:
    """The control for the case above, and for the default itself: an unset secret closes
    the override, it does not close the gateway. Catches a check that refuses every
    request on an app with no secret configured, which would be a shipped default that
    breaks every deployment that never wanted overrides."""
    store = InProcessBudgetStore()
    app_of(no_secret_client).state.budget_store = store

    response = post(no_secret_client, chat_body(model=ALIAS))

    assert response.status_code == 200, response.text
