"""`embed()` on the OpenAI-compatible adapter: both mappings, and the HTTP half.

Split the same way the chat pair next door is. `to_openai_embedding` and
`from_openai_embedding` are pure, so they run against payloads on disk with no
transport at all; the cases that have to prove the adapter REACHES the shared
`providers/transport.py` - the endpoint, the retry table, the truncation - go
through `make_local_adapter` and count requests, because an `embed()` with an
`httpx.post` loop of its own would pass every mapping case in this file and still
hammer a busy CPU box.

The `embeddings_*.json` fixtures under `tests/providers/fixtures/local/` are
HAND-WRITTEN to the published OpenAI embeddings shape, not recorded from a
llama.cpp server: none was available when these tests were written, and nothing
here proves llama.cpp puts `usage` and the vector where this reader expects them.
Each file says so in its own `_comment` key.

Two things this mapping does that the chat one does not, and both are the reason
the file exists. The outbound body always asks the provider for FLOATS whatever
the caller asked the gateway for, and the answer is ordered by each item's own
`index` rather than by the order the provider happened to serialise it: the
caller matches vectors back to its own rows by index and by nothing else, so a
reader that trusts file order returns plausible vectors against the wrong rows.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.core.errors import ProviderError
from app.providers.base import EmbeddingRequest
from app.providers.openai_compatible import (
    EMBEDDINGS_PATH,
    FLOAT32_MAX,
    OUTBOUND_ENCODING_FORMAT,
    OpenAICompatibleAdapter,
    from_openai_embedding,
    to_openai_embedding,
)
from app.providers.transport import (
    MAX_RETRIES,
    PROVIDER_MESSAGE_LIMIT,
    RETRY_BASE_DELAY,
)

from .conftest import (
    BATCH_VECTORS,
    EMBEDDING_MODEL,
    LOCAL_BASE_URL,
    LOCAL_PROVIDER_NAME,
    PROVIDER_DETAIL_MARKER,
    SINGLE_VECTOR,
    SleepRecorder,
    Step,
    load_local_fixture,
    local_fail,
    local_ok,
    make_local_adapter,
)

SINGLE = EmbeddingRequest(model=EMBEDDING_MODEL, input=("the capital of Germany",))

# Three distinct strings of three different lengths, so a mapper that de-dupes,
# sorts or truncates the batch is visible in the outbound body.
BATCH_TEXTS = ("Berlin", "the capital of Germany", "Bundesrepublik")

BATCH = EmbeddingRequest(model=EMBEDDING_MODEL, input=BATCH_TEXTS)


# --- the request direction ----------------------------------------------------


def test_a_single_input_still_travels_as_a_list() -> None:
    """Catches a mapper that sends `"input": "the capital of Germany"` for a one-item
    request: the published wire accepts a bare string, so the server would answer 200 and
    nothing downstream would notice the shape ever differed by batch size."""
    body = to_openai_embedding(SINGLE)

    assert body["input"] == ["the capital of Germany"]
    assert not isinstance(body["input"], str)


def test_a_batch_travels_as_the_list_in_request_order() -> None:
    """Catches a mapper that sorts, de-dupes or reverses the inputs: the answer is matched
    back to the caller's rows by position alone, so a reordered batch returns the right
    vectors against the wrong texts and no status says so."""
    body = to_openai_embedding(BATCH)

    assert body["input"] == list(BATCH_TEXTS)


def test_the_resolved_provider_model_id_is_sent() -> None:
    """Catches a mapper that sends the `embed` alias instead of the resolved provider model
    id, which the local server answers with a 404 on a name only our policy knows."""
    body = to_openai_embedding(SINGLE)

    assert body["model"] == EMBEDDING_MODEL
    assert body["model"] != "embed"


def test_the_outbound_body_always_asks_for_floats() -> None:
    """Catches an `encoding_format` forwarded from the caller: the gateway renders base64
    itself from the float vector, so a request that asked the PROVIDER for base64 would
    hand the reader a string where it expects numbers, and the caller would get a
    double-encoded blob rather than a vector."""
    body = to_openai_embedding(SINGLE)

    assert body["encoding_format"] == "float"
    assert OUTBOUND_ENCODING_FORMAT == "float"


def test_the_outbound_body_is_exactly_three_fields() -> None:
    """Catches a mapper that copies `max_tokens`, `stream` or a `dimensions` it invented
    into the embeddings body: llama.cpp validates this endpoint's fields rather than
    ignoring them, so every embedding request would be its 400."""
    assert set(to_openai_embedding(BATCH)) == {"model", "input", "encoding_format"}


# --- the response direction ---------------------------------------------------


def test_a_single_vector_is_read_with_its_index_and_model() -> None:
    """Catches a reader taking the vector from `data[0]["vector"]` or from a top-level
    `embedding` (neither is this wire's shape), which would make every answer unreadable."""
    response = from_openai_embedding(load_local_fixture("embeddings_single"))

    assert response.model == EMBEDDING_MODEL
    assert len(response.embeddings) == 1
    assert response.embeddings[0].index == 0
    assert response.embeddings[0].vector == SINGLE_VECTOR


def test_the_batch_is_ordered_by_each_items_own_index_not_by_file_order() -> None:
    """Catches a reader that enumerates `data` and trusts the provider's serialisation
    order: the fixture lists the items 2, 0, 1 on purpose, so a positional reader returns
    three real vectors attached to the wrong rows - plausible numbers, no error."""
    response = from_openai_embedding(load_local_fixture("embeddings_batch"))

    assert [item.index for item in response.embeddings] == [0, 1, 2]
    assert tuple(item.vector for item in response.embeddings) == BATCH_VECTORS


USAGE_CASES: list[tuple[str, str, int, int]] = [
    ("a single input", "embeddings_single", 11, 13),
    ("a three-input batch", "embeddings_batch", 23, 29),
]


@pytest.mark.parametrize(
    ("case", "fixture", "prompt", "total"),
    USAGE_CASES,
    ids=[case for case, *_ in USAGE_CASES],
)
def test_usage_maps_field_by_field(case: str, fixture: str, prompt: int, total: int) -> None:
    """Catches a reader that copies `prompt_tokens` into both fields, or reads the chat
    wire's `completion_tokens`: the two fixtures give the two counts different values for
    exactly this reason, and the ledger prices from them."""
    usage = from_openai_embedding(load_local_fixture(fixture)).usage

    assert usage.prompt_tokens == prompt, case
    assert usage.total_tokens == total, case


def test_an_absent_total_falls_back_to_the_prompt_count() -> None:
    """Catches a reader that raises (a 502 for an answer that was served and billed) or
    reports zero when the server omits `total_tokens`, which llama.cpp builds do - there
    is no second count on this wire to add, so the prompt count is the whole total."""
    payload = load_local_fixture("embeddings_single")
    del payload["usage"]["total_tokens"]

    usage = from_openai_embedding(payload).usage

    assert usage.total_tokens == 11
    assert usage.prompt_tokens == 11


def missing_usage_payloads() -> list[tuple[str, dict[str, Any]]]:
    """The two ways an embedding usage block can be unusable."""
    without_prompt = load_local_fixture("embeddings_single")
    del without_prompt["usage"]["prompt_tokens"]
    return [
        ("no usage block at all", load_local_fixture("embeddings_no_usage")),
        ("no prompt_tokens", without_prompt),
    ]


@pytest.mark.parametrize(
    ("case", "payload"),
    missing_usage_payloads(),
    ids=[case for case, _ in missing_usage_payloads()],
)
def test_a_response_without_usage_raises_rather_than_zero_filling(
    case: str, payload: dict[str, Any]
) -> None:
    """Catches the silent-zero reader (`payload.get("usage", {}).get(...) or 0`): an
    embedding call is the cheapest thing this gateway serves, so a zero booked into the
    ledger is the one nobody would query - which is precisely why decision F refuses to
    write a count this wire did not give."""
    with pytest.raises(ProviderError) as raised:
        from_openai_embedding(payload)

    assert raised.value.retryable is False, f"{case}: a malformed body is not worth a retry"
    assert raised.value.http_status == 502, case


BAD_VECTORS: list[tuple[str, list[float]]] = [
    ("an empty vector", []),
    ("a NaN component", [0.25, float("nan"), 0.5]),
    ("an infinite component", [0.25, float("inf"), 0.5]),
    ("a component beyond float32 range", [0.25, 1e300, 0.5]),
]


@pytest.mark.parametrize(("case", "vector"), BAD_VECTORS, ids=[case for case, _ in BAD_VECTORS])
def test_a_vector_the_gateway_cannot_serve_raises(case: str, vector: list[float]) -> None:
    """Catches a reader that guards the ENVELOPE and never the numbers, which every other
    check in the module does: each payload below has one item, index 0, a usage block, and
    passes the emptiness, numbering and count guards untouched. Served, an empty vector is a
    200 the caller stores against a real row; a NaN cannot be serialised by the JSON
    renderer and reaches the client as a 500 with no code; and 1e300 raises `OverflowError`
    inside the base64 packer, with the same untyped 500. All three are provider defects and
    all three must leave as the 502 a client's retry logic is written against."""
    payload = load_local_fixture("embeddings_single")
    payload["data"][0]["embedding"] = vector

    with pytest.raises(ProviderError) as raised:
        from_openai_embedding(payload)

    assert raised.value.retryable is False, case
    assert raised.value.http_status == 502, case


def test_an_empty_vector_raises() -> None:
    """The named pin for part 3 of the adapter guarantee, empty half: `[]` is what a dropped
    slot returns in place of a vector, and it is the only one of the three that would be
    SERVED rather than crash - a 200 carrying a zero-length vector."""
    payload = load_local_fixture("embeddings_single")
    payload["data"][0]["embedding"] = []

    with pytest.raises(ProviderError):
        from_openai_embedding(payload)


def test_a_non_finite_vector_component_raises() -> None:
    """The named pin for part 3, non-finite half. A badly quantised local model does emit
    `NaN`, and the JSON renderer runs with `allow_nan=False`, so an unrejected one is a 500
    with `code: null` for what is a provider defect."""
    payload = load_local_fixture("embeddings_single")
    payload["data"][0]["embedding"] = [0.25, float("nan")]

    with pytest.raises(ProviderError):
        from_openai_embedding(payload)


def test_a_vector_component_out_of_float32_range_raises() -> None:
    """The named pin for part 3, range half. `FLOAT32_MAX` is the boundary the base64 packer
    can represent, and it is inclusive: the value itself is accepted and anything above it
    is refused, which is what separates a real bound from a guess."""
    payload = load_local_fixture("embeddings_single")
    payload["data"][0]["embedding"] = [FLOAT32_MAX]
    assert from_openai_embedding(payload).embeddings[0].vector == (FLOAT32_MAX,)

    payload["data"][0]["embedding"] = [FLOAT32_MAX * 1.001]
    with pytest.raises(ProviderError):
        from_openai_embedding(payload)


def test_an_answer_with_fewer_vectors_than_inputs_raises() -> None:
    """Catches the completeness half of the adapter's guarantee being left out, which no
    check that reads the payload alone can supply: a server that drops a slot answers two
    well-formed vectors numbered 0 and 1 to a three-item request, so `data` is non-empty
    and the indices ARE exactly `0..n-1`. Every guard in `from_openai_embedding` passes,
    the caller pairs its third text with nothing, and the only place that can tell is
    `embed()`, where the request is still in hand."""
    adapter, scripted = make_local_adapter([local_ok("embeddings_batch")])
    short = load_local_fixture("embeddings_batch")
    del short["data"][0]  # the item numbered 2, so the rest are still 0 and 1
    adapter_short, _ = make_local_adapter([(200, short)])

    assert from_openai_embedding(short).embeddings[-1].index == 1, (
        "the payload alone must look valid, or this test proves nothing"
    )
    assert adapter.embed(BATCH).embeddings, "the complete answer is accepted"
    assert scripted.request_count == 1

    with pytest.raises(ProviderError) as raised:
        adapter_short.embed(BATCH)

    assert raised.value.retryable is False
    assert raised.value.http_status == 502


def test_an_answer_with_more_vectors_than_inputs_raises() -> None:
    """The other side of the count, and not symmetric with the case above: an extra vector
    cannot be numbered inside `0..n-1` without duplicating an index, so this row is what
    says the check compares against the REQUEST rather than merely against itself."""
    adapter, _ = make_local_adapter([local_ok("embeddings_batch")])

    with pytest.raises(ProviderError) as raised:
        adapter.embed(EmbeddingRequest(model=EMBEDDING_MODEL, input=("Berlin", "Bonn")))

    assert raised.value.retryable is False


def test_a_response_with_no_data_raises() -> None:
    """Catches an `IndexError` escaping the reader as a 500 with a traceback: a server that
    answers 200 with an empty `data` (llama.cpp does, when every slot is dropped) must
    leave as the `ProviderError` the routes' fallback loop reads. The twin of
    `test_a_response_with_no_choices_raises` on the chat side."""
    with pytest.raises(ProviderError) as raised:
        from_openai_embedding(load_local_fixture("embeddings_no_data"))

    assert raised.value.retryable is False


def broken_index_payloads() -> list[tuple[str, dict[str, Any]]]:
    """The three ways a provider can number its vectors so the caller cannot use them."""
    duplicated = load_local_fixture("embeddings_batch")
    duplicated["data"][0]["index"] = 0
    one_based = load_local_fixture("embeddings_batch")
    for position, item in enumerate(sorted(one_based["data"], key=lambda row: int(row["index"]))):
        item["index"] = position + 1
    return [
        ("indices 0 and 2, so row 1 has no vector", load_local_fixture("embeddings_bad_index")),
        ("two items both numbered 0", duplicated),
        ("indices counted from one", one_based),
    ]


@pytest.mark.parametrize(
    ("case", "payload"),
    broken_index_payloads(),
    ids=[case for case, _ in broken_index_payloads()],
)
def test_indices_that_are_not_zero_to_n_minus_one_raise(case: str, payload: dict[str, Any]) -> None:
    """Catches a reader that sorts by index and stops there: sorting makes a duplicated or
    a one-based numbering look orderly, and the caller then pairs a vector with a row it
    was not computed from. Only the `0..n-1` check can reject these three, because each
    payload has the right item count, a usage block and a well-formed vector in every
    item."""
    with pytest.raises(ProviderError) as raised:
        from_openai_embedding(payload)

    assert raised.value.retryable is False, case
    assert raised.value.http_status == 502, case


UNREADABLE_PAYLOADS: list[tuple[str, dict[str, Any]]] = [
    ("no data key at all", {"object": "list", "model": EMBEDDING_MODEL}),
    (
        "a vector of strings",
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": ["a", "b"]}],
            "model": EMBEDDING_MODEL,
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        },
    ),
    (
        "an index that is not a number",
        {
            "object": "list",
            "data": [{"object": "embedding", "index": "first", "embedding": [0.5]}],
            "model": EMBEDDING_MODEL,
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        },
    ),
]


@pytest.mark.parametrize(
    ("case", "payload"),
    UNREADABLE_PAYLOADS,
    ids=[case for case, _ in UNREADABLE_PAYLOADS],
)
def test_an_unreadable_payload_leaves_as_a_provider_error(
    case: str, payload: dict[str, Any]
) -> None:
    """Catches a reader with no `except (KeyError, IndexError, TypeError, ValueError)`
    around it: a `KeyError` or a Pydantic `ValidationError` reaching the route is a 500
    with a traceback instead of the 502 the routes' fallback loop acts on, and a shape change
    in a llama.cpp release is the likeliest way each of these three arrives."""
    with pytest.raises(ProviderError) as raised:
        from_openai_embedding(payload)

    assert raised.value.retryable is False, case


# --- the HTTP half ------------------------------------------------------------


def test_the_embeddings_path_is_appended_to_the_policy_base_url() -> None:
    """Catches a path carrying its own `/v1` (the shipped policy's base url already ends in
    one, so the request would go to `/v1/v1/embeddings` and 404) and catches an `embed()`
    aimed at the chat completions endpoint, which would answer a chat body for an
    embedding request."""
    adapter, scripted = make_local_adapter([local_ok("embeddings_single")])

    response = adapter.embed(SINGLE)

    assert EMBEDDINGS_PATH == "/embeddings"
    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://llama.test:8082/v1/embeddings"
    assert response.embeddings[0].vector == SINGLE_VECTOR


def test_the_body_that_goes_on_the_wire_is_the_mapped_one() -> None:
    """Catches an `embed()` that builds its own body instead of calling
    `to_openai_embedding`: every mapping case above would stay green while the request
    actually sent asked the provider for base64 or named the alias."""
    adapter, scripted = make_local_adapter([local_ok("embeddings_batch")])

    adapter.embed(BATCH)

    assert scripted.body() == {
        "model": EMBEDDING_MODEL,
        "input": list(BATCH_TEXTS),
        "encoding_format": "float",
    }


STATUS_CASES: list[tuple[str, int, str, int, bool]] = [
    ("400 bad request", 400, "error_400", 1, False),
    ("428 just under the retry row", 428, "error_400", 1, False),
    ("429 rate limited", 429, "error_429", 3, True),
    ("430 just over the retry row", 430, "error_400", 1, False),
    ("500 server error", 500, "error_500", 3, True),
    ("503 all slots busy", 503, "error_500", 3, True),
]


@pytest.mark.parametrize(
    ("case", "status", "fixture", "requests", "retryable"),
    STATUS_CASES,
    ids=[case for case, *_ in STATUS_CASES],
)
def test_the_retry_table_by_status(
    case: str,
    status: int,
    fixture: str,
    requests: int,
    retryable: bool,
    sleeps: SleepRecorder,
) -> None:
    """Catches an `embed()` that never reaches the shared transport at all - its own loop
    would retry everything or nothing - and a retry predicate written as `status >= 429` or
    `status != 200`, which the 428 and 430 rows either side of the single retryable 4xx are
    placed to expose."""
    adapter, scripted = make_local_adapter([local_fail(status, fixture)], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    assert scripted.request_count == requests, case
    assert raised.value.retryable is retryable, case
    assert len(sleeps.calls) == requests - 1, case
    assert raised.value.http_status == 502, case


def test_a_read_timeout_is_retried_twice_and_then_raises_retryable(sleeps: SleepRecorder) -> None:
    """Catches an `embed()` that only handles HTTP statuses: an unwrapped
    `httpx.ReadTimeout` reaching the route is a 500 with a traceback instead of a 502, and
    a CPU box asked for a 64-item batch being merely slow is this endpoint's single
    likeliest failure."""
    steps: list[Step] = [httpx.ReadTimeout]
    adapter, scripted = make_local_adapter(steps, sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    assert scripted.request_count == 3
    assert len(sleeps.calls) == 2
    assert MAX_RETRIES == 2, "2 retries means 3 attempts in total"
    assert raised.value.retryable is True


def test_the_local_server_being_unreachable_is_not_retried(sleeps: SleepRecorder) -> None:
    """Catches a raw `httpx.ConnectError` escaping `embed()`, which the route would answer
    as a 500 with a traceback, and catches it being retried: the model container being down
    is not a condition half a second fixes, and three attempts only delay the 502 by the
    backoff. Until an earlier change starts a llama.cpp server this is the state the shipped policy
    is actually in, so it is the failure an operator meets first."""
    steps: list[Step] = [httpx.ConnectError]
    adapter, scripted = make_local_adapter(steps, sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    assert scripted.request_count == 1
    assert sleeps.calls == []
    assert raised.value.retryable is False
    assert raised.value.http_status == 502


def test_a_429_then_a_200_succeeds_after_exactly_two_requests(sleeps: SleepRecorder) -> None:
    """Catches an `embed()` that gives up on the first busy-slot answer, and one that
    retries the full three times regardless, by pinning the count at exactly 2."""
    adapter, scripted = make_local_adapter(
        [local_fail(429, "error_429"), local_ok("embeddings_single")], sleep=sleeps
    )

    response = adapter.embed(SINGLE)

    assert scripted.request_count == 2
    assert len(sleeps.calls) == 1, "one retry means one wait"
    assert sleeps.calls == [RETRY_BASE_DELAY]
    assert response.usage.total_tokens == 13


def test_a_retry_resends_the_same_body(sleeps: SleepRecorder) -> None:
    """Catches a retry that rebuilds or mutates the request between attempts: a 64-item
    batch rebuilt from a consumed iterator would send the second attempt an empty list, and
    the caller would get one fewer vector than it asked for with no error anywhere."""
    adapter, scripted = make_local_adapter(
        [local_fail(429, "error_429"), local_ok("embeddings_batch")], sleep=sleeps
    )

    adapter.embed(BATCH)

    assert scripted.request_count == 2
    assert scripted.body(0) == scripted.body(1)
    assert scripted.body(0)["input"] == list(BATCH_TEXTS)


def test_a_non_json_success_body_is_a_provider_error(sleeps: SleepRecorder) -> None:
    """Catches the path a reverse proxy takes: a 200 carrying an HTML error page would
    otherwise raise `json.JSONDecodeError` out of `embed()`, reaching the route as a 500
    with a traceback instead of the 502 the routes' fallback loop can move past. This is the sibling
    case the chat side shipped covered and `embed()` would ship without."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>bad gateway</html>")
    )
    adapter = OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=transport,
        sleep=sleeps,
    )

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    assert raised.value.retryable is False


ERROR_FIXTURES: list[tuple[str, int, str]] = [
    ("a 400 body", 400, "error_400"),
    ("a 429 body", 429, "error_429"),
    ("a 500 body", 500, "error_500"),
]


@pytest.mark.parametrize(
    ("case", "status", "fixture"), ERROR_FIXTURES, ids=[case for case, *_ in ERROR_FIXTURES]
)
def test_the_provider_message_never_reaches_the_client_message(
    case: str, status: int, fixture: str, sleeps: SleepRecorder
) -> None:
    """Catches an `embed()` that raises `ProviderError(response.text)` of its own instead of
    letting the shared transport build the error: the server's own wording - slot counts,
    file paths, model file names - would reach the body a tenant reads."""
    payload = load_local_fixture(fixture)
    assert PROVIDER_DETAIL_MARKER in payload["error"]["message"], (
        f"{fixture}: the fixture must carry the marker, or this test proves nothing"
    )
    adapter, _ = make_local_adapter([local_fail(status, fixture)], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    error = raised.value
    assert PROVIDER_DETAIL_MARKER not in error.message, f"{case}: provider text leaked"
    assert PROVIDER_DETAIL_MARKER not in str(error), f"{case}: provider text leaked via str()"
    assert error.provider_message is not None, f"{case}: the log loses the server's reason"
    assert PROVIDER_DETAIL_MARKER in error.provider_message, case


def test_an_oversized_provider_body_is_truncated(sleeps: SleepRecorder) -> None:
    """Catches an `embed()` with a transport of its own: a reverse proxy in front of the
    model answering 502 with a megabyte of HTML would then write a megabyte into the log on
    every failed attempt, three times per request."""
    oversized = "x" * (PROVIDER_MESSAGE_LIMIT * 4)
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text=oversized))
    adapter = OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=transport,
        sleep=sleeps,
    )

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    assert raised.value.provider_message is not None
    assert len(raised.value.provider_message) == PROVIDER_MESSAGE_LIMIT


def test_the_client_message_names_the_policy_provider_and_the_status(
    sleeps: SleepRecorder,
) -> None:
    """Catches a message naming the vendor or the kind instead of the policy provider:
    `local` and `vllm` are the same kind on the same box, and an operator reading a failed
    embedding has to be able to tell which one answered."""
    adapter, _ = make_local_adapter([local_fail(400, "error_400")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.embed(SINGLE)

    assert LOCAL_PROVIDER_NAME in raised.value.message
    assert "400" in raised.value.message
