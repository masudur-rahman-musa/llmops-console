"""Inline redaction on `/v1/chat/completions` and `/v1/embeddings`, and its header.

The claim under test is narrow and load bearing: when the tenant's rule for the
request's data class carries `redact: true`, the ADAPTER receives only redacted
text. So every case here reads the stub's recorded outbound request rather than
the answer - a route that redacted the response and sent the raw prompt would
look identical from the body.

The decision table is swept in both directions on both routes, because either
half alone is satisfiable by a constant:

* a redact class WITH personal data -> the placeholder reaches the adapter and
  `X-Gateway-Redacted` is `true`;
* a redact class WITHOUT any -> the text is untouched and the header is `false`
  (decision C: the header is about the payload, not about the rule);
* a non-redact class WITH personal data -> the raw text reaches the adapter and
  the header is `false`.

Every case runs against a stub registered under the policy provider `local`,
which is the only provider all four data classes can reach for `evals`
(`confidential` allows `[eu, self]` and `pii` allows `[self]` alone, so a stub
under `anthropic` would make the redact rows a 451 and the test would pass for
the wrong reason).

`gateway_headers` is also called directly: that `redacted` is keyword-only and
has no default is what stops a later route from getting `false` by omission, and
a signature cannot be asserted through HTTP.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app.core.settings import Settings
from app.gateway.routes.chat import REDACTED_FALSE, REDACTED_TRUE, gateway_headers
from app.gateway.routes.embeddings import MAX_EMBEDDING_INPUT_CHARS
from app.redaction import MAX_REDACTION_INPUT_CHARS
from app.redaction.directory import DirectoryStore
from app.redaction.errors import DirectoryStoreUnavailable
from app.routing.policy import Policy
from app.routing.resolver import Resolved, resolve

from .conftest import (
    TEST_KEYS,
    EmbeddingStubAdapter,
    MetricDelta,
    StubAdapter,
    assert_error_body,
    build_chat_client,
    chat_body,
    embedding_stub_response,
    embeddings_body,
    post_chat,
    post_embeddings,
    stub_response,
)

EMAIL_A = "alice@example.com"
EMAIL_B = "bob@example.net"

WITH_PII = f"write to {EMAIL_A} about it"
REDACTED_PII = "write to <EMAIL_1> about it"
WITHOUT_PII = "which city is the capital of Germany"

ALIAS = "agent-default"
PROVIDER = "local"

EMBED_PROVIDER = "local_embed"
"""The `embed` alias's only candidate: the embedding server is its own provider,
because one llama.cpp process serves one model. The chat rows keep `local`."""

# (data class, what the caller sent, what the adapter must receive, the header)
INLINE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("confidential", WITH_PII, REDACTED_PII, "true"),
    ("pii", WITH_PII, REDACTED_PII, "true"),
    ("confidential", WITHOUT_PII, WITHOUT_PII, "false"),
    ("pii", WITHOUT_PII, WITHOUT_PII, "false"),
    ("public", WITH_PII, WITH_PII, "false"),
    ("internal", WITH_PII, WITH_PII, "false"),
)
INLINE_IDS = tuple(f"{row[0]}-{'pii' if row[1] == WITH_PII else 'clean'}" for row in INLINE_ROWS)


@pytest.fixture(scope="module")
def inline_chat_stub() -> StubAdapter:
    """The chat stub, registered under `local` so every data class can reach it."""
    return StubAdapter(name=PROVIDER, response=stub_response())


@pytest.fixture(scope="module")
def inline_chat_client(settings: Settings, inline_chat_stub: StubAdapter) -> Iterator[TestClient]:
    """One app for the chat rows, built once."""
    with build_chat_client(settings, {inline_chat_stub.name: inline_chat_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def inline_embedding_stub() -> EmbeddingStubAdapter:
    """The embedding stub, under the policy provider the `embed` alias names."""
    return EmbeddingStubAdapter(name=EMBED_PROVIDER, response=embedding_stub_response())


@pytest.fixture(scope="module")
def inline_embeddings_client(
    settings: Settings, inline_embedding_stub: EmbeddingStubAdapter
) -> Iterator[TestClient]:
    """One app for the embeddings rows, built once."""
    with build_chat_client(settings, {inline_embedding_stub.name: inline_embedding_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def inline_batch_stub() -> EmbeddingStubAdapter:
    """A second embedding stub answering THREE vectors, for the multi-input rows.

    The route refuses an answer carrying fewer vectors than the request had inputs, so a three-item batch needs a stub that answers three; widening the
    single-input stub instead would make every one-item row assert against an
    over-long answer.
    """
    return EmbeddingStubAdapter(name=EMBED_PROVIDER, response=embedding_stub_response(3))


@pytest.fixture(scope="module")
def inline_batch_client(
    settings: Settings, inline_batch_stub: EmbeddingStubAdapter
) -> Iterator[TestClient]:
    """One app for the three-input embedding rows, built once."""
    with build_chat_client(settings, {inline_batch_stub.name: inline_batch_stub}) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh_requests(
    inline_chat_stub: StubAdapter,
    inline_embedding_stub: EmbeddingStubAdapter,
    inline_batch_stub: EmbeddingStubAdapter,
) -> Iterator[None]:
    """Per-test isolation by clearing the recorded calls, never by rebuilding an app."""
    inline_chat_stub.requests.clear()
    inline_embedding_stub.requests.clear()
    inline_batch_stub.requests.clear()
    yield


def redacted_header(response: Response) -> str:
    """`X-Gateway-Redacted`, with the 200 asserted so a failure is not read as `false`."""
    assert response.status_code == 200, response.text
    return response.headers["x-gateway-redacted"]


def resolved_for(policy: Policy, data_class: str = "internal") -> Resolved:
    """Where the shipped policy sends `evals`' `agent-default`, with only `local` healthy."""
    return resolve(policy, "evals", ALIAS, data_class, is_healthy=lambda p: p == PROVIDER)[0]


# --------------------------------------------------------------------------
# the header constants and the builder's signature
# --------------------------------------------------------------------------


def test_the_two_header_values_are_the_published_words() -> None:
    """Catches a header rendered from a Python bool (`True`/`False`) or as `1`/`0`: a
    console sums this string, so a capitalised value silently counts as neither."""
    assert REDACTED_TRUE == "true"
    assert REDACTED_FALSE == "false"


@pytest.mark.parametrize(("redacted", "expected"), [(True, "true"), (False, "false")])
def test_gateway_headers_reports_the_redaction_outcome(
    policy: Policy, redacted: bool, expected: str
) -> None:
    """Catches a builder that still answers a constant `false`: both rows go through the
    same call, so an implementation ignoring the argument fails exactly one of them."""
    headers = gateway_headers(resolved_for(policy), "trace-id", redacted=redacted)

    assert headers["X-Gateway-Redacted"] == expected


def test_the_header_set_is_still_exactly_six(policy: Policy) -> None:
    """Catches a redaction outcome added as a SEVENTH header rather than filling the
    existing one: the six are a published contract (#9).

    Six, still, after an earlier change: `X-Gateway-Cost-USD` is the seventh header a non-streamed
    ANSWER carries, and the route sets it on the response once the ledger has priced the
    call. This builder runs before a provider is dialled, so it cannot know the number
    and does not carry it - which is also what keeps the header off a stream."""
    headers = gateway_headers(resolved_for(policy), "trace-id", redacted=True)

    assert len(headers) == 6
    assert "X-Gateway-Redacted" in headers


def test_the_redaction_outcome_is_keyword_only_and_required(policy: Policy) -> None:
    """Catches `redacted: bool = False` and a positional third parameter: a default is
    how a future route silently reports "not redacted" for a request it did redact, and
    only the signature can say so."""
    untyped = cast(Callable[..., dict[str, str]], gateway_headers)
    chosen = resolved_for(policy)

    with pytest.raises(TypeError):
        untyped(chosen, "trace-id")
    with pytest.raises(TypeError):
        untyped(chosen, "trace-id", True)


# --------------------------------------------------------------------------
# chat completions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("data_class", "sent", "outbound", "header"), INLINE_ROWS, ids=INLINE_IDS)
def test_the_chat_adapter_receives_only_what_the_class_allows(
    inline_chat_client: TestClient,
    inline_chat_stub: StubAdapter,
    data_class: str,
    sent: str,
    outbound: str,
    header: str,
) -> None:
    """Catches the three inversions at once: no redaction on a redact class (the raw
    address leaves the box), redaction on a non-redact class (a `public` caller gets
    placeholders it never asked for), and a header wired to the rule rather than to the
    payload - the clean rows on a redact class are what separate those last two."""
    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": sent}]),
        data_class=data_class,
    )

    assert redacted_header(response) == header
    assert len(inline_chat_stub.requests) == 1
    assert inline_chat_stub.requests[0].messages[0].content == outbound


def test_every_chat_message_is_redacted_under_one_numbering(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches a route redacting the first message only, and one calling the engine per
    message: the same address is in turns 0 and 2 with a different one between, so a
    per-message numbering sends `<EMAIL_1>` for two different people."""
    response = post_chat(
        inline_chat_client,
        chat_body(
            messages=[
                {"role": "user", "content": f"from {EMAIL_A}"},
                {"role": "assistant", "content": f"noted, cc {EMAIL_B}"},
                {"role": "user", "content": f"still {EMAIL_A}"},
            ]
        ),
        data_class="confidential",
    )

    assert redacted_header(response) == "true"
    contents = [turn.content for turn in inline_chat_stub.requests[0].messages]
    assert contents == ["from <EMAIL_1>", "noted, cc <EMAIL_2>", "still <EMAIL_1>"]


def test_tool_call_arguments_are_not_redacted(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches a route that redacts a tool call's JSON arguments: the arguments are a
    machine payload the caller's own tool parses, and rewriting a value inside that JSON
    hands the tool a placeholder it cannot act on. The user turn in the same request
    carries an address, so "nothing was redacted at all" cannot pass this."""
    response = post_chat(
        inline_chat_client,
        chat_body(
            messages=[
                {"role": "user", "content": f"write to {EMAIL_A}"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "toolu_01StubMailCall0000000",
                            "type": "function",
                            "function": {
                                "name": "send_mail",
                                "arguments": '{"to": "' + EMAIL_B + '"}',
                            },
                        }
                    ],
                },
            ]
        ),
        data_class="confidential",
    )

    assert redacted_header(response) == "true"
    outbound = inline_chat_stub.requests[0].messages
    assert outbound[0].content == "write to <EMAIL_1>"
    assert outbound[1].tool_calls[0].arguments == '{"to": "' + EMAIL_B + '"}'


def test_a_streamed_request_takes_the_same_inline_path(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches redaction wired into the non-streaming branch only: a stream reaches the
    adapter through `stream_chat`, so the raw prompt would leave the box on exactly the
    requests an SDK sends by default."""
    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": WITH_PII}], stream=True),
        data_class="confidential",
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-gateway-redacted"] == "true"
    assert inline_chat_stub.requests[0].messages[0].content == REDACTED_PII


def test_a_chat_message_at_the_cap_is_accepted_on_a_redact_class(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches an inline cap written `>=`: 16384 characters is legal, and refusing it
    would 422 a request that used to be served, which is the half of the boundary pair a
    cap alone cannot get right."""
    at_cap = "a" * MAX_REDACTION_INPUT_CHARS

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": at_cap}]),
        data_class="confidential",
    )

    assert response.status_code == 200, response.text
    assert inline_chat_stub.requests[0].messages[0].content == at_cap


def test_a_chat_message_over_the_cap_is_422_with_no_outbound_call(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches a cap enforced after the adapter was called - which sends the unredacted
    text and then reports a failure, the exact inversion the deny-by-default direction
    forbids - and a cap not enforced inline at all."""
    over_cap = "a" * (MAX_REDACTION_INPUT_CHARS + 1)

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": over_cap}]),
        data_class="confidential",
    )

    assert_error_body(response, status=422, code="VALIDATION")
    assert inline_chat_stub.requests == []


def test_a_refused_request_is_still_attributed_to_its_resolved_candidate(
    inline_chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches the resolution labelling being done only inside the fallback loop, which
    sits after redaction. Redaction can REFUSE a request - the case above is the 422 - and
    such a request would then be counted under an empty alias, provider and region while
    the resolver had already decided all three: the design's per-alias error panel would
    show nothing during a redaction-side incident, and a trace search by alias would miss
    every refused request. The resolution happens before redaction either way, so an
    unlabelled row here is a label LOST rather than one that was never known."""
    over_cap = "a" * (MAX_REDACTION_INPUT_CHARS + 1)

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": over_cap}]),
        data_class="confidential",
    )
    assert_error_body(response, status=422, code="VALIDATION")

    moved = [dict(labels) for labels in metrics.moved_series("gateway_requests_total")]
    assert len(moved) == 1, moved
    assert moved[0]["status"] == "422"
    assert moved[0]["alias"] != ""
    assert moved[0]["provider"] != ""
    assert moved[0]["region"] != ""


def test_the_redaction_cap_does_not_apply_on_a_non_redact_class(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches a 16384 cap applied to every chat request rather than to the strings this
    plan redacts: on `public` nothing is redacted, so an over-cap message must still be
    served exactly as before this plan."""
    over_cap = "a" * (MAX_REDACTION_INPUT_CHARS + 1)

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": over_cap}]),
        data_class="public",
    )

    assert response.status_code == 200, response.text
    assert inline_chat_stub.requests[0].messages[0].content == over_cap


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("data_class", "sent", "outbound", "header"), INLINE_ROWS, ids=INLINE_IDS)
def test_the_embedding_adapter_receives_only_what_the_class_allows(
    inline_embeddings_client: TestClient,
    inline_embedding_stub: EmbeddingStubAdapter,
    data_class: str,
    sent: str,
    outbound: str,
    header: str,
) -> None:
    """Catches redaction wired into the chat route alone (decision B): `confidential`
    allows region `eu`, so an unredacted embedding input leaves the box on the route
    whose batches are largest, and the same three inversions apply here."""
    response = post_embeddings(
        inline_embeddings_client, embeddings_body(input=sent), data_class=data_class
    )

    assert redacted_header(response) == header
    assert len(inline_embedding_stub.requests) == 1
    assert inline_embedding_stub.requests[0].input == (outbound,)


def test_every_embedding_input_is_redacted_under_one_numbering(
    inline_batch_client: TestClient, inline_batch_stub: EmbeddingStubAdapter
) -> None:
    """Catches a route redacting `input[0]` only, and a per-item numbering: item 0 and
    item 2 hold the same address with a different one between them, so a per-item map
    renumbers item 2 and two batch rows would rehydrate to the wrong person."""
    response = post_embeddings(
        inline_batch_client,
        embeddings_body(input=[f"from {EMAIL_A}", f"cc {EMAIL_B}", f"still {EMAIL_A}"]),
        data_class="confidential",
    )

    assert redacted_header(response) == "true"
    assert inline_batch_stub.requests[0].input == (
        "from <EMAIL_1>",
        "cc <EMAIL_2>",
        "still <EMAIL_1>",
    )


@pytest.mark.parametrize("data_class", ["confidential", "public"])
def test_an_embedding_input_over_the_routes_own_cap_is_422_first(
    inline_embeddings_client: TestClient,
    inline_embedding_stub: EmbeddingStubAdapter,
    data_class: str,
) -> None:
    """Catches an inline cap that REPLACED the route's 8192-character cap from an earlier change.

    The 16384/16385 pair is unreachable on this route: `MAX_EMBEDDING_INPUT_CHARS` is
    8192, so no input can reach the redaction cap at all, and the pair is pinned on the
    chat route instead. What this row asserts is that the tighter cap still answers
    first, on a redact class and a plain one alike, and that nothing was sent.
    """
    assert MAX_EMBEDDING_INPUT_CHARS < MAX_REDACTION_INPUT_CHARS

    response = post_embeddings(
        inline_embeddings_client,
        embeddings_body(input="a" * (MAX_EMBEDDING_INPUT_CHARS + 1)),
        data_class=data_class,
    )

    assert_error_body(response, status=422, code="VALIDATION")
    assert str(MAX_EMBEDDING_INPUT_CHARS) in response.json()["error"]["message"]
    assert inline_embedding_stub.requests == []


def test_an_embedding_input_at_the_routes_cap_is_accepted_on_a_redact_class(
    inline_embeddings_client: TestClient, inline_embedding_stub: EmbeddingStubAdapter
) -> None:
    """Catches an inline redaction step that refuses a legal 8192-character input - the
    engine accepts up to 16384, so nothing this route can carry may be rejected by the
    redaction path."""
    at_cap = "a" * MAX_EMBEDDING_INPUT_CHARS

    response = post_embeddings(
        inline_embeddings_client, embeddings_body(input=at_cap), data_class="confidential"
    )

    assert response.status_code == 200, response.text
    assert inline_embedding_stub.requests[0].input == (at_cap,)


def test_a_full_batch_of_capped_inputs_is_accepted_on_a_redact_class(
    inline_batch_client: TestClient, inline_batch_stub: EmbeddingStubAdapter
) -> None:
    """Catches a cap applied to the JOINED batch (decision F): 64 inputs of 8192
    characters is the largest legal request this route has, and a joined redaction cap
    would 422 it while every single-input row above stayed green."""
    batch = ["a" * MAX_EMBEDDING_INPUT_CHARS] * 3

    response = post_embeddings(
        inline_batch_client, embeddings_body(input=batch), data_class="confidential"
    )

    assert response.status_code == 200, response.text
    assert inline_batch_stub.requests[0].input == tuple(batch)


# --------------------------------------------------------------------------
# the tenant's directory names
# --------------------------------------------------------------------------
#
# The name is seeded through the app's own store rather than through
# `POST /admin/directory`, so a failure here points at the inline path and not at
# the endpoint, which has its own file. Seeding is a merge, so calling it twice
# with the same name changes nothing and no case has to clean up after another.
#
# The cross-tenant row is the sharp one. `nordlicht` sends the SAME text under a
# redact class and reaches the same stub (its `agent-default` override resolves
# to `local`), and the text carries an address as well - so redaction demonstrably
# ran for that request and simply did not know the name. Without that second
# tenant, "the directory was loaded for the authenticated tenant" and "one global
# directory was loaded" are indistinguishable.

DIRECTORY_NAME = "Anna Schmidt"
NAME_TEXT = f"escalate to {DIRECTORY_NAME} at {EMAIL_A}"
NAME_REDACTED = "escalate to <PERSON_1> at <EMAIL_1>"
NAME_WITHOUT_PERSON = f"escalate to {DIRECTORY_NAME} at <EMAIL_1>"

NORDLICHT_KEY = TEST_KEYS["nordlicht"]


def seed_directory(client: TestClient, tenant: str, *names: str) -> None:
    """Put names into the app's directory store, as `POST /admin/directory` would."""
    app = cast(FastAPI, client.app)
    store = cast(DirectoryStore, app.state.directory_store)
    store.put(tenant, list(names))


REDIS_FAILURE_TEXT = "Error 111 connecting to redis:6379. Connection refused."
"""The client's own words, planted as a marker: they name internal topology and
must not travel to a caller."""


class UnavailableDirectoryStore:
    """A `DirectoryStore` whose read fails the way an unreachable Redis does.

    `put` is here because the protocol has it; nothing in this file calls it. The
    failure is on `get`, because `get` is what the inline path makes on EVERY
    redacted request.
    """

    def put(self, tenant: str, names: Sequence[str]) -> None:
        raise DirectoryStoreUnavailable(REDIS_FAILURE_TEXT)

    def get(self, tenant: str) -> tuple[str, ...]:
        raise DirectoryStoreUnavailable(REDIS_FAILURE_TEXT)


@pytest.fixture
def unavailable_directory_chat_client(settings: Settings) -> Iterator[TestClient]:
    """A second app whose directory store cannot be reached.

    Its own app rather than the module's: swapping the store is the thing under
    test, and doing it on `inline_chat_client` would leave every later case in this
    file talking to a directory that refuses every read.
    """
    stub = StubAdapter(name=PROVIDER, response=stub_response())
    with build_chat_client(settings, {stub.name: stub}) as client:
        cast(FastAPI, client.app).state.directory_store = UnavailableDirectoryStore()
        yield client


def test_the_chat_adapter_receives_a_placeholder_where_a_directory_name_stood(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches an inline path that redacts with the nine pattern kinds and never loads the
    tenant's directory: the address in the same message is still replaced, so the header
    reads `true` and the response looks correct while the name this control exists to
    remove leaves the box. Both the placeholder and the header are asserted, because a
    route that redacted but reported `false` is a different bug with the same body."""
    seed_directory(inline_chat_client, "evals", DIRECTORY_NAME)

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": NAME_TEXT}]),
        data_class="confidential",
    )

    assert redacted_header(response) == "true"
    assert inline_chat_stub.requests[0].messages[0].content == NAME_REDACTED


def test_the_embedding_adapter_receives_a_placeholder_where_a_directory_name_stood(
    inline_embeddings_client: TestClient, inline_embedding_stub: EmbeddingStubAdapter
) -> None:
    """Catches the directory wired into the chat route alone: embeddings is the route
    whose batches are largest and whose text is stored in a vector database afterwards, so
    a name that survives here survives in a corpus nobody re-reads."""
    seed_directory(inline_embeddings_client, "evals", DIRECTORY_NAME)

    response = post_embeddings(
        inline_embeddings_client, embeddings_body(input=NAME_TEXT), data_class="confidential"
    )

    assert redacted_header(response) == "true"
    assert inline_embedding_stub.requests[0].input == (NAME_REDACTED,)


def test_one_tenants_directory_name_is_not_redacted_for_another_tenant(
    inline_chat_client: TestClient, inline_chat_stub: StubAdapter
) -> None:
    """Catches a directory loaded without its tenant - one process-wide matcher, or a
    matcher compiled once at startup from every tenant's names.

    `nordlicht` sends the same text under a redact class and the address in it IS still
    replaced, so redaction ran for this request and only the name was left alone. That is
    the arrangement in which removing the tenant from the load is the single thing that
    flips the result: a test asserting nothing was redacted would also pass if inline
    redaction had been switched off altogether.
    """
    seed_directory(inline_chat_client, "evals", DIRECTORY_NAME)

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": NAME_TEXT}]),
        key=NORDLICHT_KEY,
        tenant="nordlicht",
        data_class="confidential",
    )

    assert redacted_header(response) == "true"
    assert inline_chat_stub.requests[0].messages[0].content == NAME_WITHOUT_PERSON


def test_a_person_redaction_is_counted_under_the_person_kind(
    inline_chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a counter that files a directory match under an existing kind label, or
    under none: `gateway_redactions_total` is what the design's redaction panel sums, and a
    tenth kind counted as `EMAIL` would make the panel report a detector that is not the
    one that fired. The message carries an address too, so a single kindless increment
    cannot pass, and `moved_series` is what says no THIRD label appeared."""
    seed_directory(inline_chat_client, "evals", DIRECTORY_NAME)

    response = post_chat(
        inline_chat_client,
        chat_body(messages=[{"role": "user", "content": NAME_TEXT}]),
        data_class="confidential",
    )
    assert response.status_code == 200, response.text

    assert metrics.delta("gateway_redactions_total", tenant="evals", kind="PERSON") == 1
    assert metrics.delta("gateway_redactions_total", tenant="evals", kind="EMAIL") == 1
    assert len(metrics.moved_series("gateway_redactions_total")) == 2


def test_an_unreachable_directory_store_is_502_with_no_outbound_call(
    unavailable_directory_chat_client: TestClient,
) -> None:
    """Catches the failure mode decision H exists to forbid: a route that reads the
    directory in a `try` and carries on with an empty one would serve this request with
    the tenant's names in the clear, and the response would still say
    `X-Gateway-Redacted: true`. Redaction is deny-by-default like residency, so the only
    acceptable answer is the 502 PROVIDER_ERROR `filed_map_id` already gives an
    unreachable map store - and NOTHING may reach the adapter."""
    app = cast(FastAPI, unavailable_directory_chat_client.app)
    stub = cast(StubAdapter, app.state.adapters[PROVIDER])

    response = post_chat(
        unavailable_directory_chat_client,
        chat_body(messages=[{"role": "user", "content": NAME_TEXT}]),
        data_class="confidential",
    )

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert stub.requests == []


def test_a_directory_store_failure_does_not_put_the_clients_words_in_front_of_a_caller(
    unavailable_directory_chat_client: TestClient,
) -> None:
    """Catches a conversion that pastes the Redis error into the client-facing message:
    that string names an internal host and port, and the same rule already holds for the
    map store and on the provider path."""
    response = post_chat(
        unavailable_directory_chat_client,
        chat_body(messages=[{"role": "user", "content": NAME_TEXT}]),
        data_class="confidential",
    )

    assert REDIS_FAILURE_TEXT not in response.text
    assert "6379" not in response.text


def test_a_non_redact_class_is_served_while_the_directory_store_is_down(
    unavailable_directory_chat_client: TestClient,
) -> None:
    """Catches a directory read moved in front of the `redact: true` check: a `public`
    request is not redacted at all, so a store outage that took it down would turn a Redis
    incident into a total outage rather than one confined to the classes that redact. Also
    the positive control for the 502 rows above - without it, a route that answered 502 to
    everything would pass them both."""
    response = post_chat(
        unavailable_directory_chat_client,
        chat_body(messages=[{"role": "user", "content": NAME_TEXT}]),
        data_class="public",
    )

    assert response.status_code == 200, response.text
