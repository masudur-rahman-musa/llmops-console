"""`POST /v1/embeddings`: the OpenAI wire shape, the caps, base64, and the failures.

The endpoint exists so embeddings never leave the box, so the cases here are
grouped by the three things that could break that promise or the answer.

* The ANSWER: the `list` envelope, one `embedding` object per input in request
  order, `usage` carried through, and the alias-versus-provider-model split (the
  body names the alias the caller sent; only `X-Gateway-Model` carries the
  provider's id). Order is the contract: the caller matches vectors back to its
  own rows by `index` and by nothing else.
* The OUTBOUND request: what the route hands the adapter.
  `EmbeddingStubAdapter` records every `EmbeddingRequest`, so the wire-to-neutral
  mapping is asserted field by field rather than inferred from a good answer. The
  base64 case runs the REAL `OpenAICompatibleAdapter` over `httpx.MockTransport`
  and asserts the JSON that actually went out, because "the gateway renders
  base64 itself and still asks the provider for floats" is a claim about the
  outbound body that a parsed result cannot settle.
* The REFUSALS: the input cap table, the four routing failures, and the two auth
  ones. Every rejected row is arranged so the control it names is the only thing
  that can reject it, and every rejected row asserts that NO outbound call was
  made - a cap enforced after the provider was called is not a cap.

No socket is opened, no provider key is read and no backing service is started.
The successful cases answer from the stub, the base64 case from a
hand-written payload under `tests/providers/fixtures/local/`.
"""

from __future__ import annotations

import base64
import binascii
import struct
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from providers.conftest import (
    PROVIDER_DETAIL_MARKER,
    SINGLE_VECTOR,
    local_fail,
    local_ok,
    make_local_adapter,
)

from app.api.routes.embeddings import EMBEDDINGS
from app.core.config import Settings
from app.core.errors import ProviderError
from app.model_routing.policy import Policy
from app.model_routing.resolver import Resolved, resolve
from app.providers.base import EmbeddingAdapter, EmbeddingRequest
from app.schemas.embeddings import (
    EMBEDDING_OBJECT,
    LIST_OBJECT,
    MAX_EMBEDDING_INPUT_CHARS,
    MAX_EMBEDDING_INPUTS,
)
from app.services.embedding_pipeline import to_base64
from app.storage import InMemoryCostLedgerWriter

from .conftest import (
    EMBED_ALIAS,
    EMBED_PROVIDER,
    STUB_EMBEDDING_MODEL,
    STUB_EMBEDDING_USAGE,
    STUB_VECTORS,
    TEST_KEYS,
    EmbeddingStubAdapter,
    StubAdapter,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    embedding_stub_response,
    embeddings_body,
    post_embeddings,
)
from .conftest import EMBEDDINGS as CONFTEST_EMBEDDINGS

NORDLICHT_KEY = TEST_KEYS["nordlicht"]

TENANT = "evals"
DATA_CLASS = "internal"

# The provider model id the shipped policy gives the `embed` alias. Never what
# the response body may name.
PROVIDER_MODEL = "bge-small-en-v1.5"

GATEWAY_HEADER_PREFIX = "x-gateway-"

# The whole set, and only this set, exactly as the chat route's is.
EXPECTED_HEADER_NAMES = {
    "x-gateway-provider",
    "x-gateway-model",
    "x-gateway-cache",
    "x-gateway-redacted",
    "x-gateway-trace-id",
    "x-gateway-cost-usd",
}

UPSTREAM_TRACE_ID = "9c4a1b77e2d34f08b6510ac3d7e29f41"
UPSTREAM_TRACEPARENT = f"00-{UPSTREAM_TRACE_ID}-1a2b3c4d5e6f7081-01"

# Three distinct strings of three different lengths, so a route that de-dupes,
# sorts or truncates the batch is visible in the outbound request.
THREE_TEXTS = ["Berlin", "the capital of Germany", "Bundesrepublik Deutschland"]


def ok_body(response: Response) -> dict[str, Any]:
    """Assert a 200 and return its parsed body, so a failure shows the wire text."""
    assert response.status_code == 200, response.text
    parsed: dict[str, Any] = response.json()
    return parsed


def gateway_headers(response: Response) -> dict[str, str]:
    """The `X-Gateway-*` headers of a response, lowercased, with a 200 asserted."""
    assert response.status_code == 200, response.text
    return {
        name.lower(): value
        for name, value in response.headers.items()
        if name.lower().startswith(GATEWAY_HEADER_PREFIX)
    }


def first_candidate(policy: Policy) -> Resolved:
    """Where the shipped policy sends `evals`' `embed` under `internal`.

    The same health seam the route uses, expressed as the only provider these
    tests give an embedding adapter to, so the headers are compared against the
    resolver rather than against literals alone.
    """
    candidates = resolve(
        policy,
        TENANT,
        EMBED_ALIAS,
        DATA_CLASS,
        is_healthy=lambda provider: provider == EMBED_PROVIDER,
    )
    return candidates[0]


def decode_float32(value: str, size: int) -> tuple[float, ...]:
    """Decode a base64 string as `size` little-endian float32 values, strictly.

    `validate=True` so a string with whitespace or a wrong alphabet raises here
    rather than being silently truncated into numbers that look plausible.
    """
    raw = base64.b64decode(value, validate=True)
    assert len(raw) == 4 * size, f"{len(raw)} bytes is not {size} float32 values"
    return struct.unpack(f"<{size}f", raw)


# --- the stub and the constants ----------------------------------------------


def test_embedding_stub_adapter_satisfies_the_protocol() -> None:
    """Catches a stub that has drifted from the real `EmbeddingAdapter` (a `create_embedding`
    instead of an `embed`): the route's health seam is an `isinstance` against that protocol,
    so a drifted stub would make every success case in this file a 503 - and the sibling of
    `test_stub_adapter_satisfies_the_protocol` is what says so in one line rather than
    through twenty confusing failures."""
    stub = EmbeddingStubAdapter(name=EMBED_PROVIDER)

    assert isinstance(stub, EmbeddingAdapter)
    assert stub.name == EMBED_PROVIDER
    assert stub.price_key("some-model") == f"{EMBED_PROVIDER}/some-model"


def test_the_route_path_is_the_published_one() -> None:
    """Catches the route being mounted somewhere else (`/v1/embed`, or under a prefix): an
    OpenAI SDK posts to this exact path, and every case in this file would still pass
    against a path only the tests know, because they read it from the same constant."""
    assert EMBEDDINGS == "/v1/embeddings"
    assert CONFTEST_EMBEDDINGS == EMBEDDINGS


def test_the_two_object_discriminators_are_the_published_words() -> None:
    """Catches `object: "embeddings"` or `"vector"`: an OpenAI-compatible client branches on
    these two strings to decide how to read the body, so a near-miss is a body that parses
    and then hands the client nothing."""
    assert LIST_OBJECT == "list"
    assert EMBEDDING_OBJECT == "embedding"


def test_the_input_caps_are_exactly_sixty_four_and_eight_thousand_one_hundred_ninety_two() -> None:
    """Catches either cap drifting: the table below is written in terms of the constants, so
    it follows them rather than pinning them, and these are the acceptance criterion's
    numbers."""
    assert MAX_EMBEDDING_INPUTS == 64
    assert MAX_EMBEDDING_INPUT_CHARS == 8192


def test_the_neutral_embedding_request_carries_no_encoding_format() -> None:
    """Catches an `encoding_format` pushed down into the neutral contract, which is how the
    caller's base64 would reach the provider: the gateway renders base64 itself, so the
    field must stop at the wire layer and there must be nowhere below it to put it."""
    assert set(EmbeddingRequest.model_fields) == {"model", "input"}


# --- the answer ---------------------------------------------------------------


def test_successful_embeddings_response_is_openai_shaped(embeddings_client: TestClient) -> None:
    """Catches a route answering the neutral `EmbeddingResponse` straight out (an
    `embeddings` list of `{index, vector}`), which no OpenAI-compatible client can read: it
    looks for `data[].embedding` under an `object: "list"` envelope."""
    body = ok_body(post_embeddings(embeddings_client, embeddings_body()))

    assert set(body) == {"object", "data", "model", "usage"}
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert set(item) == {"object", "index", "embedding"}
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert item["embedding"] == list(STUB_VECTORS[0])


def test_response_model_is_the_alias_the_caller_named(embeddings_client: TestClient) -> None:
    """Catches a route echoing the adapter's `EmbeddingResponse.model` (or the resolved
    provider model id) into the body: that discloses which provider served the request and
    hands back a name the caller cannot send again, since only aliases are routable. The
    same published rule the chat route follows."""
    body = ok_body(post_embeddings(embeddings_client, embeddings_body()))

    assert body["model"] == EMBED_ALIAS
    assert body["model"] != STUB_EMBEDDING_MODEL
    assert body["model"] != PROVIDER_MODEL


def test_usage_is_carried_through_unchanged(embeddings_client: TestClient) -> None:
    """Catches a zero-filled or recomputed usage, and catches a third `completion_tokens`
    field written as 0 alongside the two real ones: the ledger prices from
    these numbers, so a wrong one books a real, billed call at the wrong cost."""
    body = ok_body(post_embeddings(embeddings_client, embeddings_body()))

    assert body["usage"] == {
        "prompt_tokens": STUB_EMBEDDING_USAGE.prompt_tokens,
        "total_tokens": STUB_EMBEDDING_USAGE.total_tokens,
    }
    assert body["usage"]["prompt_tokens"] == 97
    assert body["usage"]["total_tokens"] == 113


def test_a_list_input_returns_one_vector_per_item_in_request_order(settings: Settings) -> None:
    """Catches a route that renders the adapter's vectors in whatever order it iterated
    them, or that renumbers them by their position in its own loop: the caller matches
    vectors back to its rows by `index` alone, so a renumbered answer pairs a real vector
    with the wrong text and nothing reports it."""
    stub = EmbeddingStubAdapter(name=EMBED_PROVIDER, response=embedding_stub_response(3))

    with build_chat_client(settings, {EMBED_PROVIDER: stub}) as client:
        body = ok_body(post_embeddings(client, embeddings_body(input=THREE_TEXTS)))

    assert [item["index"] for item in body["data"]] == [0, 1, 2]
    assert [item["embedding"] for item in body["data"]] == [list(v) for v in STUB_VECTORS]


# --- the outbound request -----------------------------------------------------


def last_request(stub: EmbeddingStubAdapter) -> EmbeddingRequest:
    """The `EmbeddingRequest` the route handed the adapter on the most recent call."""
    assert stub.requests, "the route never called the adapter"
    return stub.requests[-1]


def test_the_adapter_is_handed_the_resolved_provider_model_not_the_alias(
    embeddings_client: TestClient, embedding_stub: EmbeddingStubAdapter
) -> None:
    """Catches a route passing the caller's alias straight into `EmbeddingRequest.model`,
    which would make the provider 404 on a name only our routing policy knows."""
    ok_body(post_embeddings(embeddings_client, embeddings_body()))

    sent = last_request(embedding_stub)
    assert sent.model != EMBED_ALIAS
    assert sent.model == PROVIDER_MODEL


def test_a_bare_string_input_reaches_the_adapter_as_a_one_item_tuple(
    embeddings_client: TestClient, embedding_stub: EmbeddingStubAdapter
) -> None:
    """Catches a route that forwards the caller's `str` union member unchanged: the neutral
    contract is a tuple of texts, and a bare string handed down would be iterated character
    by character somewhere below - one vector per letter, no error."""
    ok_body(post_embeddings(embeddings_client, embeddings_body(input="Berlin")))

    assert last_request(embedding_stub).input == ("Berlin",)


def test_inputs_reach_the_adapter_in_order(
    embeddings_client: TestClient, embedding_stub: EmbeddingStubAdapter
) -> None:
    """Catches a route that rebuilds the batch from a set, sorts it, or de-dupes it: the
    answer is positional, so any reordering below this point returns vectors the caller
    will attach to the wrong rows."""
    ok_body(post_embeddings(embeddings_client, embeddings_body(input=THREE_TEXTS)))

    assert last_request(embedding_stub).input == tuple(THREE_TEXTS)


# --- base64 -------------------------------------------------------------------


def test_to_base64_is_little_endian_float32() -> None:
    """Catches a big-endian pack (`>f`) and a float64 one (`<d`): both return a plausible
    base64 string that decodes to numbers, so a wrong byte order is silent corruption
    rather than an error - the risk this case exists for. `1.0` is asserted as its exact
    IEEE-754 bytes, which differ between the three encodings."""
    assert base64.b64decode(to_base64((1.0,)), validate=True) == b"\x00\x00\x80\x3f"
    assert decode_float32(to_base64(STUB_VECTORS[1]), len(STUB_VECTORS[1])) == STUB_VECTORS[1]


def test_float_encoding_returns_numbers_not_a_string(embeddings_client: TestClient) -> None:
    """Catches a route that renders base64 unconditionally: the default is `float`, and a
    client that did not ask for base64 gets a string where its parser expects a list of
    numbers. The anchor for the base64 cases below, which would otherwise pass against an
    always-base64 renderer."""
    body = ok_body(post_embeddings(embeddings_client, embeddings_body()))

    assert isinstance(body["data"][0]["embedding"], list)
    assert body["data"][0]["embedding"] == list(STUB_VECTORS[0])


def test_base64_encoding_round_trips_to_the_same_vector(embeddings_client: TestClient) -> None:
    """Catches a route that base64-encodes the JSON text of the vector, or the float64
    bytes: the OpenAI SDK decodes this field as little-endian float32, so anything else
    hands it numbers that are wrong without being ill-formed."""
    body = ok_body(post_embeddings(embeddings_client, embeddings_body(encoding_format="base64")))

    encoded = body["data"][0]["embedding"]
    assert isinstance(encoded, str)
    assert decode_float32(encoded, len(STUB_VECTORS[0])) == STUB_VECTORS[0]


def test_base64_still_asks_the_provider_for_floats_on_the_wire(settings: Settings) -> None:
    """The whole path, through the real adapter: catches a route that forwards the caller's
    `encoding_format` to the provider. The answer would then be a base64 string where the
    reader expects numbers, and a reader that passed it through unchanged would return a
    DOUBLE-encoded blob - which decodes cleanly to nonsense. Only asserting the outbound
    body can tell that apart from a correct answer."""
    adapter, scripted = make_local_adapter([local_ok("embeddings_single")], name=EMBED_PROVIDER)

    with build_chat_client(settings, {EMBED_PROVIDER: adapter}) as client:
        body = ok_body(post_embeddings(client, embeddings_body(encoding_format="base64")))

    assert scripted.request_count == 1
    assert scripted.body()["encoding_format"] == "float"
    assert decode_float32(body["data"][0]["embedding"], len(SINGLE_VECTOR)) == SINGLE_VECTOR
    assert body["model"] == EMBED_ALIAS


def test_an_unknown_encoding_format_is_422(embeddings_client: TestClient) -> None:
    """Catches an `encoding_format: str` widened past the two published values: an
    unrecognised one would fall through to whichever branch the renderer wrote last, and
    the caller would get the other encoding without being told."""
    response = post_embeddings(embeddings_client, embeddings_body(encoding_format="float32"))

    assert_error_body(response, status=422, code="VALIDATION")


# --- refusals: the input caps -------------------------------------------------
#
# Every row carries exactly ONE fault, with everything else on the request valid,
# so only the check it names can reject it - a row rejected by two rules would
# stay green after its own rule was deleted. `stub.requests` is empty afterwards
# in every row, which is what proves the caps run before any outbound call.

AT_ITEM_CAP: list[str] = [f"text {index}" for index in range(MAX_EMBEDDING_INPUTS)]
OVER_ITEM_CAP: list[str] = [f"text {index}" for index in range(MAX_EMBEDDING_INPUTS + 1)]
AT_CHAR_CAP = "a" * MAX_EMBEDDING_INPUT_CHARS
OVER_CHAR_CAP = "a" * (MAX_EMBEDDING_INPUT_CHARS + 1)

# Every shape a caller can put in `input`, legal or not, in one alias so the
# tables below carry a real type rather than `Any`.
WireInput = str | list[str] | list[int] | list[list[int]] | None

# The rows the CAPS reject: each is a well-typed `str` or `list[str]` that only
# the size and emptiness rules can refuse.
CAPPED_INPUTS: list[tuple[str, WireInput]] = [
    ("an empty list", []),
    ("an empty bare string", ""),
    ("an empty string inside a list", ["Berlin", ""]),
    ("one item over the item cap", OVER_ITEM_CAP),
    ("a bare string one character over the character cap", OVER_CHAR_CAP),
    ("an item one character over the character cap", ["Berlin", OVER_CHAR_CAP]),
]

# The rows the ANNOTATION rejects, before any validator of ours runs. The reasoning:
# the gateway has no tokenizer, and neither redaction nor pricing can see inside
# a token array, so OpenAI's two token forms are a shape this product cannot
# honour its guarantees on and are refused by the type rather than passed
# through.
MISTYPED_INPUTS: list[tuple[str, WireInput]] = [
    ("a token array", [1, 2, 3]),
    ("a nested token array", [[1, 2], [3, 4]]),
    ("a null input", None),
]

REJECTED_INPUTS: list[tuple[str, WireInput]] = CAPPED_INPUTS + MISTYPED_INPUTS


@pytest.mark.parametrize(
    ("case", "value"), REJECTED_INPUTS, ids=[case for case, _ in REJECTED_INPUTS]
)
def test_a_rejected_input_is_422_before_any_outbound_call(
    case: str, value: WireInput, settings: Settings
) -> None:
    """Catches caps enforced inside the route body rather than on the model, which would
    hold a CPU box for minutes on the very request the cap exists to refuse before it
    answered 422; catches an exclusive comparison that lets 65 items or 8193 characters
    through; and catches a token array accepted and handed to a provider that would embed
    the string "[1, 2, 3]". `requests == []` is the half that says the check ran FIRST."""
    stub = EmbeddingStubAdapter(name=EMBED_PROVIDER)

    with build_chat_client(settings, {EMBED_PROVIDER: stub}) as client:
        response = post_embeddings(client, embeddings_body(input=value))

    assert_error_body(response, status=422, code="VALIDATION")
    assert stub.requests == [], f"{case}: the provider was called for a rejected request"


@pytest.mark.parametrize(("case", "value"), CAPPED_INPUTS, ids=[case for case, _ in CAPPED_INPUTS])
def test_a_capped_input_names_the_field(case: str, value: WireInput, settings: Settings) -> None:
    """Catches a cap raised somewhere the shared validation handler cannot name `input` -
    a check on the whole model rather than on the field, or one that raises a bare
    `ValueError` from the route body. With up to 64 items in a batch, "the request is not
    valid" leaves the caller nothing to look at.

    Only the capped rows are here. The mistyped ones are refused by the annotation, and
    the existing `handle_request_validation_error` reports the LAST segment of the error
    location, which for a `str | tuple[str, ...]` union is the failing member's name; that
    handler is out of scope here."""
    stub = EmbeddingStubAdapter(name=EMBED_PROVIDER)

    with build_chat_client(settings, {EMBED_PROVIDER: stub}) as client:
        response = post_embeddings(client, embeddings_body(input=value))

    error = response.json()["error"]
    assert error["param"] == "input", case
    assert "input" in error["message"], case


ACCEPTED_INPUTS: list[tuple[str, WireInput]] = [
    ("a bare string", "Berlin"),
    ("a one-item list", ["Berlin"]),
    ("exactly the item cap", AT_ITEM_CAP),
    ("a bare string of exactly the character cap", AT_CHAR_CAP),
    ("an item of exactly the character cap", ["Berlin", AT_CHAR_CAP]),
]


@pytest.mark.parametrize(
    ("case", "value"), ACCEPTED_INPUTS, ids=[case for case, _ in ACCEPTED_INPUTS]
)
def test_an_input_at_the_boundary_is_accepted(
    case: str, value: WireInput, settings: Settings
) -> None:
    """The anchor for the rejection table: catches an exclusive boundary (`< 64`, `< 8192`),
    which refuses a batch that is exactly at the documented cap, and catches a validator so
    strict it rejects everything - without this row the table above would pass against one
    that always raised."""
    stub = EmbeddingStubAdapter(name=EMBED_PROVIDER)

    with build_chat_client(settings, {EMBED_PROVIDER: stub}) as client:
        ok_body(post_embeddings(client, embeddings_body(input=value)))

    assert len(stub.requests) == 1, case
    assert len(AT_ITEM_CAP) == MAX_EMBEDDING_INPUTS
    assert len(AT_CHAR_CAP) == MAX_EMBEDDING_INPUT_CHARS


def test_unknown_body_field_is_422(embeddings_client: TestClient) -> None:
    """Catches a request model left on Pydantic's default `extra="ignore"`: a misspelled
    `encoding_formats` would be dropped and the call served as floats, so a client that
    asked for base64 would parse numbers as a string."""
    response = post_embeddings(embeddings_client, embeddings_body(dimensions=256))

    assert_error_body(response, status=422, code="VALIDATION")
    assert "dimensions" in response.json()["error"]["message"]


# --- refusals: routing and the provider ---------------------------------------


def test_unknown_alias_is_404(embeddings_client: TestClient) -> None:
    """Catches a route that hands an unknown model name to an adapter anyway, turning a
    routing failure the caller can fix into a provider error it cannot."""
    response = post_embeddings(embeddings_client, embeddings_body(model="no-such-alias"))

    assert_error_body(response, status=404, code="ALIAS_UNKNOWN")


def test_a_malformed_alias_is_422_not_404(embeddings_client: TestClient) -> None:
    """Catches a route that looks the alias up in the policy dict before the identifier
    pre-check runs: a name with a trailing newline would come back as 404 "unknown alias",
    telling the caller the name does not exist rather than that it is not a legal name."""
    response = post_embeddings(embeddings_client, embeddings_body(model="embed\n"))

    assert_error_body(response, status=422, code="VALIDATION")


def test_a_candidate_without_an_embedding_adapter_is_503(settings: Settings) -> None:
    """Catches a route whose health seam is `provider in adapters` rather than "this adapter
    can embed": `local` IS registered here, with the chat-only `StubAdapter`, so the
    provider-presence seam would let the request through and the caller would get an
    `AttributeError` served as a 500 with no code. Only the `EmbeddingAdapter` check can
    turn it into the resolver's typed 503."""
    with build_chat_client(settings, {EMBED_PROVIDER: StubAdapter(name=EMBED_PROVIDER)}) as client:
        response = post_embeddings(client, embeddings_body())

    assert_error_body(response, status=503, code="ALL_PROVIDERS_FAILED")


def test_the_same_request_succeeds_with_an_embedding_capable_adapter(
    embeddings_client: TestClient,
) -> None:
    """The anchor for the case above: catches a health seam so strict that nothing satisfies
    it, which would make every embedding request a 503 while the 503 case above stayed
    green. Same alias, same tenant, same body - only the adapter differs."""
    body = ok_body(post_embeddings(embeddings_client, embeddings_body()))

    assert body["object"] == "list"


def test_provider_failure_is_502_without_the_provider_message(settings: Settings) -> None:
    """Catches a route that renders the provider's own words into the client body: the error
    fixture carries a marker string, and a client that can read a slot count or a model file
    path is being told about the box we run."""
    adapter, scripted = make_local_adapter([local_fail(400, "error_400")], name=EMBED_PROVIDER)

    with build_chat_client(settings, {EMBED_PROVIDER: adapter}) as client:
        response = post_embeddings(client, embeddings_body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert scripted.request_count == 1, "a 400 is not retried"
    assert PROVIDER_DETAIL_MARKER not in response.text


def test_an_adapter_raising_provider_error_is_502(settings: Settings) -> None:
    """Catches a route that lets a `ProviderError` reach the last-resort handler as an
    untyped bug: the caller would get a 500 with `code: null` instead of the 502 its retry
    logic is written against."""
    stub = EmbeddingStubAdapter(
        name=EMBED_PROVIDER,
        error=ProviderError(
            "local: the provider answered 503",
            retryable=True,
            provider_message=PROVIDER_DETAIL_MARKER,
        ),
    )

    with build_chat_client(settings, {EMBED_PROVIDER: stub}) as client:
        response = post_embeddings(client, embeddings_body())

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert PROVIDER_DETAIL_MARKER not in response.text


def test_embeddings_requires_auth(embeddings_client: TestClient) -> None:
    """Catches a route declared without the shared request context dependency: it would
    serve an unauthenticated caller, and with no tenant there is no policy to
    apply."""
    response = embeddings_client.post(EMBEDDINGS, json=embeddings_body())

    assert_error_body(response, status=401, code="UNAUTHORIZED")


def test_embeddings_rejects_a_tenant_that_is_not_the_keys(embeddings_client: TestClient) -> None:
    """Catches a route that resolves on the `X-Tenant` header without the key/tenant match:
    one tenant could then embed under another's policy, which is a tenant
    boundary bypassed by a header. A second auth path would pass every other case here."""
    response = embeddings_client.post(
        EMBEDDINGS,
        json=embeddings_body(),
        headers=auth_headers(key=NORDLICHT_KEY, tenant="evals", data_class="internal"),
    )

    assert_error_body(response, status=403, code="TENANT_MISMATCH")


def test_an_unknown_data_class_is_422(embeddings_client: TestClient) -> None:
    """Catches a route that reads `X-Data-Class` itself instead of taking it from the shared
    `Context`: a second reader would have to repeat the closed-enum check, and the one that
    forgot would silently downgrade an unknown class to the default and resolve on a class
    the caller never asked for."""
    response = post_embeddings(embeddings_client, embeddings_body(), data_class="secret")

    assert_error_body(response, status=422, code="VALIDATION")


def test_a_pii_request_is_served_by_the_local_embed_provider(settings: Settings) -> None:
    """Catches an embeddings route that resolves without the request's data class, or with a
    hardcoded one: the `pii` class redacts, and the request must still reach the box we run. The
    header assertions are what prove the vector was computed there and not merely that
    a 200 came back."""
    adapter, scripted = make_local_adapter([local_ok("embeddings_single")], name=EMBED_PROVIDER)

    with build_chat_client(settings, {EMBED_PROVIDER: adapter}) as client:
        served = post_embeddings(
            client,
            embeddings_body(),
            key=NORDLICHT_KEY,
            tenant="nordlicht",
            data_class="pii",
        )

    body = ok_body(served)
    assert scripted.request_count == 1, "the local provider was not actually called"
    assert served.headers["x-gateway-provider"] == EMBED_PROVIDER
    assert served.headers["x-gateway-model"] == PROVIDER_MODEL
    assert body["data"][0]["embedding"] == list(SINGLE_VECTOR)
    assert scripted.body()["model"] == PROVIDER_MODEL, (
        "the provider model id, never the alias, goes on the wire"
    )


# --- the six headers ----------------------------------------------------------


def test_response_headers_on_success(embeddings_client: TestClient, policy: Policy) -> None:
    """Catches every wrong value in the set at once - a provider label taken from the vendor
    rather than the policy name, a hardcoded value, a cache status of `hit` on a route with
    no cache, or a redaction flag of `true` on a route that redacts nothing - each of which
    is an unverifiable claim about where a tenant's text went to be embedded."""
    resolved = first_candidate(policy)

    headers = gateway_headers(post_embeddings(embeddings_client, embeddings_body()))

    assert headers["x-gateway-provider"] == resolved.provider
    assert headers["x-gateway-model"] == resolved.model
    assert headers["x-gateway-cache"] == "miss"
    assert headers["x-gateway-redacted"] == "false"
    # The literal anchors, so the three resolver-derived rows above cannot be
    # satisfied by a resolver that returned something else entirely.
    assert resolved.provider == EMBED_PROVIDER
    assert resolved.model == PROVIDER_MODEL


def test_the_header_set_is_exactly_six(embeddings_client: TestClient) -> None:
    """Catches a second header set built for this route rather than the shared
    `gateway_headers`: a copy would drift from the chat route's the first time either
    changed, and the criterion is that both come from the one function."""
    headers = gateway_headers(post_embeddings(embeddings_client, embeddings_body()))

    assert set(headers) == EXPECTED_HEADER_NAMES
    assert len(headers) == 6


def test_the_cost_header_is_the_booked_cost(embeddings_client: TestClient) -> None:
    """The positive claim that replaced the old absence of the header, on the
    route a dashboard would sum most of. Catches an embeddings path that books a row and
    forgets the header - the chat route is where the header is built, so this route
    getting it is not automatic - and catches a header that is not the row's number, since
    the value is compared to the row as a `Decimal` and its six places are counted. The
    zero here is a PRICED zero: `local_embed/bge-small-en-v1.5` ships at 0.00 per million, and
    the row's `price_version` says so rather than reading `unpriced`."""
    response = post_embeddings(embeddings_client, embeddings_body())

    assert response.status_code == 200, response.text
    ledger = app_of(embeddings_client).state.cost_ledger
    assert isinstance(ledger, InMemoryCostLedgerWriter)
    row = ledger.rows[-1]
    value = response.headers["x-gateway-cost-usd"]
    assert Decimal(value) == row.cost_usd
    assert len(value.partition(".")[2]) == 6, value
    assert row.price_version == f"{EMBED_PROVIDER}/{PROVIDER_MODEL}@2025-01-01"


def test_model_header_is_the_provider_model_not_the_alias(
    embeddings_client: TestClient, policy: Policy
) -> None:
    """Catches a header builder that copies the caller's `model` field: the body already
    names the alias, so a header repeating it tells an operator nothing about which provider
    model was actually run and makes the two indistinguishable in a log."""
    resolved = first_candidate(policy)

    headers = gateway_headers(post_embeddings(embeddings_client, embeddings_body()))

    assert headers["x-gateway-model"] == resolved.model
    assert headers["x-gateway-model"] != EMBED_ALIAS


def test_trace_id_header_is_the_upstream_trace_id(embeddings_client: TestClient) -> None:
    """Catches a route that mints its own trace id while a valid `traceparent` was sent: the
    gateway's span would sit in a trace of its own, and the caller's request could never be
    joined to what the gateway did with it."""
    headers = gateway_headers(
        post_embeddings(embeddings_client, embeddings_body(), traceparent=UPSTREAM_TRACEPARENT)
    )

    assert headers["x-gateway-trace-id"] == UPSTREAM_TRACE_ID


def test_trace_id_header_is_minted_when_no_traceparent_is_sent(
    embeddings_client: TestClient,
) -> None:
    """Catches an empty or constant trace id on a request with no upstream header: a constant
    one collapses every request in the logs into a single trace, which is worse than none at
    all because it looks correct."""
    first = gateway_headers(post_embeddings(embeddings_client, embeddings_body()))
    second = gateway_headers(post_embeddings(embeddings_client, embeddings_body()))

    assert first["x-gateway-trace-id"] != second["x-gateway-trace-id"]
    assert len(first["x-gateway-trace-id"]) == 32


def test_a_base64_string_is_not_valid_base64_by_accident() -> None:
    """Catches this file's own decoder being permissive: `decode_float32` is what every
    base64 case above trusts, so if it accepted a non-base64 string those cases would prove
    nothing about the encoding at all."""
    with pytest.raises((binascii.Error, ValueError)):
        decode_float32("not base64 at all!!", 4)
