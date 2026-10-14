"""Ordered fallback through the route: what falls back, what does not, and where it stops.

Everything here goes through `TestClient` against an app `create_app` built, so
every claim about a status, a header or a frame is settled by a real request
rather than by reading the route. No backing service and no
socket: the candidates are the `FallbackStubAdapter` pair in `conftest.py`, and
per-test isolation is those stubs' reset plus a fresh breaker on the session
app's state.

The contract these cases are written against, before it exists:

* both routes build the resolver's health seam from "an adapter exists for this
  provider AND `app.state.breaker.allows(provider)`", so a breaker that is open
  is invisible to `resolve()` in exactly the way a missing adapter already is;
* both routes then walk the tuple `resolve()` returned, in order, and move to the
  next candidate only on a RETRYABLE `ProviderError`;
* `app.state.breaker` is the `CircuitBreaker` built by `create_app`.

The one thing the fallback list can never be is longer than what `resolve()`
answered, which is the whole residency argument. It is
tested here rather than in the resolver, by
`test_an_eu_tenants_fallback_stays_inside_the_eu_regions`: a US provider sits in
the middle of the EU policy's candidate list with an adapter registered, and no
request in this file may ever reach it.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from httpx import Response

from app.providers.breaker import MAX_FAILURES as BREAKER_MAX_FAILURES
from app.providers.breaker import CircuitBreaker, CircuitState
from app.routing.policy import DataClass, load_policy
from app.routing.resolver import resolve

from .conftest import (
    EU_ALLOWED_REGIONS,
    EU_CLASS,
    EU_EXCLUDED_PROVIDER,
    EU_FIRST_PROVIDER,
    EU_SECOND_PROVIDER,
    EU_TENANT,
    FALLBACK_ALIAS,
    FALLBACK_CANDIDATES,
    FALLBACK_CLASS,
    FALLBACK_RESOLUTION,
    FALLBACK_TENANT,
    FIRST_PROVIDER,
    PROVIDER_MESSAGE_SENTINEL,
    SECOND_PROVIDER,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    FallbackStubAdapter,
    answer_of,
    assert_error_body,
    chat_body,
    embeddings_body,
    fragment_of,
    non_retryable_error,
    post_chat,
    post_embeddings,
    retryable_error,
)

DONE_FRAME = "data: [DONE]"


def stream_body() -> dict[str, object]:
    """The fallback alias, streamed."""
    return chat_body(model=FALLBACK_ALIAS, stream=True)


def body() -> dict[str, object]:
    """The fallback alias, not streamed."""
    return chat_body(model=FALLBACK_ALIAS)


def send(client: TestClient, request_body: dict[str, object]) -> Response:
    """POST one chat completion as the tenant the fallback pair resolves for."""
    return post_chat(
        client,
        request_body,
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )


def frame_lines(response: Response) -> list[str]:
    """Every non-empty line of a streamed 200, with the media type asserted."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";")[0] == "text/event-stream"
    return [line for line in response.text.splitlines() if line.strip()]


def data_payloads(response: Response) -> list[str]:
    """The payload of every `data:` frame, in order, the `[DONE]` sentinel included."""
    return [
        line.removeprefix("data:").strip()
        for line in frame_lines(response)
        if line.startswith("data:")
    ]


def streamed_text(response: Response) -> str:
    """The text deltas of a streamed answer, joined, so the ANSWER names its provider."""
    text = ""
    for payload in data_payloads(response):
        if payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        for choice in chunk["choices"]:
            text += str(choice["delta"].get("content") or "")
    return text


def completion_text(response: Response) -> str:
    """The content of a non-streamed completion."""
    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    return content


# --------------------------------------------------------------------------
# the arrangement every case below depends on
# --------------------------------------------------------------------------


def test_the_fallback_alias_resolves_to_the_expected_candidates() -> None:
    """Catches a policy edit that leaves this file testing one provider twice, or testing
    them in the other order: every case here reads "the FIRST candidate failed and the
    SECOND served", and if `routing.yaml` no longer answers that pair in that order the
    cases would keep passing while asserting something else. Read from `resolve()` rather
    than from the file, because the tuple the route walks is the resolver's answer.

    The resolver answers THREE candidates since an earlier change appended `openai` to
    `agent-default`, and only the first two carry a stub, so the prefix relation is
    asserted here too: a policy edit that put the unstubbed third candidate anywhere but
    last would leave the cases below falling back to a provider that cannot answer."""
    candidates = resolve(
        load_policy(SHIPPED_POLICY_PATH),
        FALLBACK_TENANT,
        FALLBACK_ALIAS,
        DataClass(FALLBACK_CLASS).value,
    )

    assert tuple(item.provider for item in candidates) == FALLBACK_RESOLUTION
    assert FALLBACK_RESOLUTION[: len(FALLBACK_CANDIDATES)] == FALLBACK_CANDIDATES


# --------------------------------------------------------------------------
# a retryable failure falls back
# --------------------------------------------------------------------------


def test_a_retryable_failure_is_served_by_the_next_candidate(
    fallback_client: TestClient, fallback_adapters: dict[str, FallbackStubAdapter]
) -> None:
    """Catches a route that still returns `candidates[0]`'s failure - the 502 an earlier change
    shipped - and a route that falls back but keeps answering the FIRST candidate's
    identity: the body is compared against the second stub's own words and the headers
    against the second provider, so a fallback that served correctly while reporting
    `X-Gateway-Provider: anthropic` (which is what a copied header dict does) is caught
    too. The call count pins that the first candidate was tried once and not retried."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.chat_error = retryable_error(FIRST_PROVIDER)

    response = send(fallback_client, body())

    assert response.status_code == 200, response.text
    assert completion_text(response) == answer_of(SECOND_PROVIDER)
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER
    assert response.headers["x-gateway-model"] != ""
    assert len(first.requests) == 1
    assert len(second.requests) == 1


def test_a_stream_failing_before_its_first_event_falls_back(
    fallback_client: TestClient, fallback_adapters: dict[str, FallbackStubAdapter]
) -> None:
    """Catches a streamed request that gets no fallback at all - a 502 where a
    non-streamed one would have been served - and one whose fallback answers from the
    second provider under the first provider's header. Nothing has been sent when the
    first candidate fails here (the route pulls event one before it returns), which is
    exactly the window decision C says is safe to retry elsewhere."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.stream_error = retryable_error(FIRST_PROVIDER)
    first.stream_error_at = 0

    response = send(fallback_client, stream_body())

    assert streamed_text(response) == fragment_of(SECOND_PROVIDER)
    assert frame_lines(response)[-1] == DONE_FRAME
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER
    assert len(first.requests) == 1
    assert len(second.requests) == 1


def test_a_stream_failing_after_its_first_event_is_one_error_frame_and_no_fallback(
    fallback_client: TestClient, fallback_adapters: dict[str, FallbackStubAdapter]
) -> None:
    """Catches a fallback loop wrapped around the whole stream: the status and the first
    frames are already on the wire, so re-attempting elsewhere would replay a partial
    answer from a second provider inside one 200 - a corrupted answer rather than a
    reported failure. The second stub receiving nothing is the assertion that says the
    loop stopped; the missing `[DONE]` is what tells the client the answer is truncated."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.stream_error = retryable_error(FIRST_PROVIDER)
    first.stream_error_at = 1

    response = send(fallback_client, stream_body())

    lines = frame_lines(response)
    assert DONE_FRAME not in lines
    payloads = data_payloads(response)
    error_frames = [json.loads(payload) for payload in payloads if "error" in json.loads(payload)]
    assert len(error_frames) == 1
    assert error_frames[0]["error"]["code"] == "PROVIDER_ERROR"
    assert PROVIDER_MESSAGE_SENTINEL not in response.text
    assert second.requests == [], "nothing already sent may be re-attempted elsewhere"
    assert len(first.requests) == 1


# --------------------------------------------------------------------------
# a non-retryable failure does not
# --------------------------------------------------------------------------


def test_a_non_retryable_failure_stops_at_the_first_candidate(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    fallback_breaker: CircuitBreaker,
) -> None:
    """Catches a loop that falls back on any `ProviderError` instead of on `retryable`
    only: a 400 cannot succeed anywhere, so the tenant would be billed twice for a request
    that was wrong the first time, and the client would wait for a second provider to
    confirm it. Catches a breaker that counts it as well - `MAX_FAILURES` such requests
    leave the provider closed here, so a breaker counting every failure would have opened
    it and taken a healthy provider out of rotation over the caller's own bad request."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.chat_error = non_retryable_error(FIRST_PROVIDER)

    for _ in range(BREAKER_MAX_FAILURES):
        response = send(fallback_client, body())
        assert_error_body(response, status=502, code="PROVIDER_ERROR")

    assert PROVIDER_MESSAGE_SENTINEL not in response.text
    assert second.requests == [], "a 4xx must not reach a second provider"
    assert len(first.requests) == BREAKER_MAX_FAILURES
    assert fallback_breaker.state(FIRST_PROVIDER) is CircuitState.CLOSED
    assert fallback_breaker.allows(FIRST_PROVIDER) is True


# --------------------------------------------------------------------------
# the ends of the list
# --------------------------------------------------------------------------


def test_every_candidate_failing_retryably_is_the_last_ones_502(
    fallback_client: TestClient, fallback_adapters: dict[str, FallbackStubAdapter]
) -> None:
    """Catches a loop that reports the FIRST failure after trying them all - an operator
    reading the log would go and look at a provider that may be perfectly healthy - and
    one that runs out of candidates without raising, which is a 200 with no body or a 500
    with no typed code. Each stub names itself in its message, so which 502 came back is
    a fact about the body rather than an inference."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.chat_error = retryable_error(FIRST_PROVIDER)
    second.chat_error = retryable_error(SECOND_PROVIDER)

    response = send(fallback_client, body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    message = response.json()["error"]["message"]
    assert SECOND_PROVIDER in message, message
    assert FIRST_PROVIDER not in message, message
    assert PROVIDER_MESSAGE_SENTINEL not in response.text
    assert len(first.requests) == 1
    assert len(second.requests) == 1


def test_every_breaker_open_is_the_resolvers_503_with_no_provider_called(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    fallback_breaker: CircuitBreaker,
) -> None:
    """Catches a route that consults the breaker inside the loop instead of in the health
    seam: every candidate would be called and the answer would be a 502 after two provider
    calls, rather than the resolver's 503 with no call at all - which is the difference
    between a breaker that protects a struggling provider and one that only re-labels the
    failure. The empty request logs are the half that says nothing was dialled."""
    for provider in FALLBACK_CANDIDATES:
        for _ in range(BREAKER_MAX_FAILURES):
            fallback_breaker.record_failure(provider)

    response = send(fallback_client, body())

    assert_error_body(response, status=503, code="ALL_PROVIDERS_FAILED")
    for provider in FALLBACK_CANDIDATES:
        assert fallback_adapters[provider].call_count == 0, provider


def test_an_open_breaker_skips_a_provider_on_the_next_request(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    fallback_breaker: CircuitBreaker,
) -> None:
    """The end-to-end statement of the whole plan: catches a route that reads the breaker
    but never RECORDS the failures the loop absorbs, which is a breaker that never opens
    and a first candidate dialled on every request forever. The sixth request is the kill
    - the first stub's call count must not move - and it is why the count is asserted both
    before and after."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.chat_error = retryable_error(FIRST_PROVIDER)

    for _ in range(BREAKER_MAX_FAILURES):
        assert send(fallback_client, body()).status_code == 200

    assert len(first.requests) == BREAKER_MAX_FAILURES
    assert fallback_breaker.state(FIRST_PROVIDER) is CircuitState.OPEN

    response = send(fallback_client, body())

    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER
    assert len(first.requests) == BREAKER_MAX_FAILURES, "the open provider was called again"
    assert len(second.requests) == BREAKER_MAX_FAILURES + 1


# --------------------------------------------------------------------------
# residency
# --------------------------------------------------------------------------


def test_an_eu_tenants_fallback_stays_inside_the_eu_regions(
    eu_client: TestClient, eu_adapters: dict[str, FallbackStubAdapter]
) -> None:
    """The residency criterion, tested through the route. Catches a fallback that
    re-resolves without the tenant's clamp, or that walks `app.state.adapters` instead of
    the tuple `resolve()` returned: the US provider sits BETWEEN the two EU candidates in
    the policy and has an adapter registered, so either mistake dials it, and its data
    would have left the region on a request the resolver had already refused it for."""
    excluded = eu_adapters[EU_EXCLUDED_PROVIDER]
    eu_adapters[EU_FIRST_PROVIDER].chat_error = retryable_error(EU_FIRST_PROVIDER)

    response = post_chat(
        eu_client,
        chat_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[EU_TENANT],
        tenant=EU_TENANT,
        data_class=EU_CLASS,
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == EU_SECOND_PROVIDER
    assert response.headers["x-gateway-region"] in EU_ALLOWED_REGIONS
    assert completion_text(response) == answer_of(EU_SECOND_PROVIDER)
    assert excluded.call_count == 0, "a fallback crossed the residency boundary"


# --------------------------------------------------------------------------
# the embeddings route takes the same seam and the same loop
# --------------------------------------------------------------------------


def test_the_embeddings_route_falls_back_on_a_retryable_failure(
    fallback_client: TestClient, fallback_adapters: dict[str, FallbackStubAdapter]
) -> None:
    """Catches step 4 being skipped: the chat route alone would leave every embedding
    request answered by `candidates[0]` and 502 on a provider the gateway already knows is
    failing, with no sign of it in any chat test. The header is asserted as well as the
    status, because an embeddings route that fell back and kept the first provider's
    headers would mislabel the tokens and the cost booked against it."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    first.embed_error = retryable_error(FIRST_PROVIDER)

    response = post_embeddings(
        fallback_client,
        embeddings_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER
    assert len(first.embed_requests) == 1
    assert len(second.embed_requests) == 1


def test_the_embeddings_route_skips_a_provider_whose_breaker_is_open(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    fallback_breaker: CircuitBreaker,
) -> None:
    """Catches an embeddings health seam that still asks only "is this an embedding
    adapter": the open provider would be dialled on every request, and the breaker the
    chat route respects would be advisory on half the traffic. The pre-opened breaker is
    what makes this about the SEAM rather than about the loop - no failure happens in this
    request at all."""
    first = fallback_adapters[FIRST_PROVIDER]
    second = fallback_adapters[SECOND_PROVIDER]
    for _ in range(BREAKER_MAX_FAILURES):
        fallback_breaker.record_failure(FIRST_PROVIDER)

    response = post_embeddings(
        fallback_client,
        embeddings_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER
    assert first.embed_requests == [], "the open provider was still dialled"
    assert len(second.embed_requests) == 1
