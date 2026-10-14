"""The seven `X-Gateway-*` headers on a successful chat completion (#9).

They are the operator's view of one request: which provider served it, in which
region, on which provider model, whether it was cached, whether it was redacted,
and the trace id to look it up by. Nothing else in the answer says any of that -
the body deliberately names the alias - so a header that is missing, renamed or
wrong is a claim about residency and cost that cannot be checked after the fact.

Every value is asserted exactly, and the set is asserted whole,
so an added header is a deliberate change rather than a surprise. The set grew
from six to seven when the cost ledger was wired to the route: an earlier change decision
B kept `X-Gateway-Cost-USD` off the answer until a call could really be priced,
because a header reading `0.00` is a wrong number, and it is on the answer now
that the number is the ledger row's own. It stays absent on a stream and on
every error response, which tests/gateway/test_cost_wiring.py pins.

Expectations come from `resolve()` over the shipped policy rather than from
literals alone, with one literal anchor per value so the comparison cannot become
a tautology over the resolver.
"""

from __future__ import annotations

import re
from decimal import Decimal

from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.model_routing.policy import Policy
from app.model_routing.resolver import Resolved, resolve
from app.storage import InMemoryCostLedgerWriter

from .conftest import StubAdapter, app_of, build_chat_client, chat_body, post_chat

GATEWAY_HEADER_PREFIX = "x-gateway-"

# The whole set, and only this set.
EXPECTED_HEADER_NAMES = {
    "x-gateway-provider",
    "x-gateway-region",
    "x-gateway-model",
    "x-gateway-cache",
    "x-gateway-redacted",
    "x-gateway-trace-id",
    "x-gateway-cost-usd",
}

ALIAS = "agent-default"
TENANT = "evals"
DATA_CLASS = "internal"

# A well-formed W3C traceparent, version 00, with a non-zero trace and span id.
UPSTREAM_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
UPSTREAM_TRACEPARENT = f"00-{UPSTREAM_TRACE_ID}-00f067aa0ba902b7-01"

TRACE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def first_candidate(policy: Policy) -> Resolved:
    """Where the shipped policy sends `evals`' `agent-default` under `internal`.

    The same health seam the route uses: only `anthropic` has an adapter in these
    tests, so `anthropic_eu` (disabled) and `local` (no adapter) drop out and the
    first candidate is the one the headers must describe.
    """
    candidates = resolve(
        policy, TENANT, ALIAS, DATA_CLASS, is_healthy=lambda provider: provider == "anthropic"
    )
    return candidates[0]


def gateway_headers(response: Response) -> dict[str, str]:
    """The `X-Gateway-*` headers of a response, lowercased, with a 200 asserted."""
    assert response.status_code == 200, response.text
    return {
        name.lower(): value
        for name, value in response.headers.items()
        if name.lower().startswith(GATEWAY_HEADER_PREFIX)
    }


def test_response_headers_on_success(chat_client: TestClient, policy: Policy) -> None:
    """Catches every wrong value in the set at once - a provider label taken from the
    vendor rather than the policy name, a region hardcoded to the tenant's own, a cache
    status of `hit` on a request that was never eligible for the an earlier change cache (this
    body states no `temperature: 0` and sends no `X-Gateway-Cache: allow`), or a
    redaction flag of `true` on a route that redacts nothing - each of which is an
    unverifiable claim about where a tenant's data went."""
    resolved = first_candidate(policy)

    headers = gateway_headers(post_chat(chat_client, chat_body(model=ALIAS)))

    assert headers["x-gateway-provider"] == resolved.provider
    assert headers["x-gateway-region"] == resolved.region
    assert headers["x-gateway-model"] == resolved.model
    assert headers["x-gateway-cache"] == "miss"
    assert headers["x-gateway-redacted"] == "false"
    # The literal anchors, so the three resolver-derived rows above cannot be
    # satisfied by a resolver that returned something else entirely.
    assert resolved.provider == "anthropic"
    assert resolved.region == "us"
    assert resolved.model == "claude-haiku-4-5"


def test_the_header_set_is_exactly_seven(chat_client: TestClient) -> None:
    """Catches both directions: a header quietly dropped (an operator loses the trace id
    on the one request they need it for) and one quietly added ahead of the plan that
    specifies it, which would ship an unpinned value into a published contract."""
    headers = gateway_headers(post_chat(chat_client, chat_body()))

    assert set(headers) == EXPECTED_HEADER_NAMES
    assert len(headers) == 7


def test_the_cost_header_is_the_booked_cost(chat_client: TestClient) -> None:
    """The positive claim that replaced the absence an earlier change decision B pinned. Catches a
    header that carries a number the route computed for itself rather than the ledger
    row's - the two would then drift apart on the first pricing change, and only the
    ledger would be audited - and catches a placeholder `0` or `0.00`, since the value is
    compared to the row as a `Decimal` and its six places are counted. The literal is the
    arithmetic of the fixture usage over the shipped price entry, so a header and a row
    that are consistently wrong fail here too."""
    response = post_chat(chat_client, chat_body())

    assert response.status_code == 200, response.text
    ledger = app_of(chat_client).state.cost_ledger
    assert isinstance(ledger, InMemoryCostLedgerWriter)
    row = ledger.rows[-1]
    value = response.headers["x-gateway-cost-usd"]
    assert Decimal(value) == row.cost_usd
    assert len(value.partition(".")[2]) == 6, value
    # 137 input and 29 output tokens at the shipped 1.00 / 5.00 per million.
    assert value == "0.000282"


def test_model_header_is_the_provider_model_not_the_alias(
    chat_client: TestClient, policy: Policy
) -> None:
    """Catches a header builder that copies the caller's `model` field: the body already
    names the alias, so a header repeating it tells an operator nothing about which
    provider model was actually billed and makes the two indistinguishable in a log."""
    resolved = first_candidate(policy)

    headers = gateway_headers(post_chat(chat_client, chat_body(model=ALIAS)))

    assert headers["x-gateway-model"] == resolved.model
    assert headers["x-gateway-model"] != ALIAS


def test_trace_id_header_is_the_upstream_trace_id(chat_client: TestClient) -> None:
    """Catches a route that mints its own trace id while a valid `traceparent` was sent:
    the gateway's span would then sit in a trace of its own, and the caller's request
    could never be joined to what the gateway did with it."""
    headers = gateway_headers(post_chat(chat_client, chat_body(), traceparent=UPSTREAM_TRACEPARENT))

    assert headers["x-gateway-trace-id"] == UPSTREAM_TRACE_ID


def test_trace_id_header_is_minted_when_no_traceparent_is_sent(chat_client: TestClient) -> None:
    """Catches an empty or constant trace id on a request with no upstream header: a
    constant one collapses every request in the logs into a single trace, which is worse
    than none at all because it looks correct."""
    first = gateway_headers(post_chat(chat_client, chat_body()))["x-gateway-trace-id"]
    second = gateway_headers(post_chat(chat_client, chat_body()))["x-gateway-trace-id"]

    assert TRACE_ID_PATTERN.fullmatch(first) is not None
    assert TRACE_ID_PATTERN.fullmatch(second) is not None
    assert first != second


def test_headers_describe_the_provider_that_served_the_request(
    settings: Settings, policy: Policy
) -> None:
    """Catches a header builder reading the adapter it was handed rather than the resolver
    result, or the reverse: the stub is registered under the policy name `anthropic` and
    reports a model of its own, so `X-Gateway-Provider` and `X-Gateway-Model` must still
    be the policy's name and the policy's model id."""
    stub = StubAdapter(name="anthropic")
    resolved = first_candidate(policy)

    with build_chat_client(settings, {"anthropic": stub}) as client:
        headers = gateway_headers(post_chat(client, chat_body(model=ALIAS)))

    assert stub.requests[-1].model == resolved.model
    assert headers["x-gateway-provider"] == "anthropic"
    assert headers["x-gateway-model"] == resolved.model
