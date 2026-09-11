"""The cost ledger through the route: one row per served request, the metric and the header.

Everything here is settled by a real request through `TestClient` against an app
`create_app` built, because the question is never "does
`record_cost` price correctly" - tests/ledger/test_cost_ledger.py settles that -
but "does every path book exactly one row, naming what served, and does the
number in the row, in `gateway_cost_usd_total` and in `X-Gateway-Cost-USD` stay
one number".

No backing service. The app is built with no database URL, so
`app.state.cost_ledger` is the `InMemoryCostLedgerWriter` and the assertions read
its `rows`, exactly as tests/gateway/test_residency_log.py reads the residency
writer's. The ledger is reached by NAME, `app.state.cost_ledger`, never by shape:
this app has several collaborators with a `write`, and selecting one of them by
its methods is what a project rule exists for.

a project rule: one app for the module, built once. Per-test isolation is the
stubs' reset and fresh `response_cache`, `semantic_cache` and `cost_ledger`
objects on `app.state` - never a new app. The one case that needs a second app is
the fallback one, and it borrows the session-scoped `fallback_client` and puts
the writer it replaced back afterwards, because that app is shared with
tests/gateway/test_fallback.py and tests/gateway/test_residency_log.py.

Three arrangements are worth reading before the assertions:

* THE WRITER IS INSTALLED BY THE SENDER, immediately before the request being
  measured. A cache-hit sender primes the cache first, and that priming request
  books a row of its own; installing after it is what makes "exactly one row"
  a statement about ONE request rather than about a pair.
* THE EMBEDDING ADAPTER IS LIVE ON THE CHAT APP. `agent-default` carries
  `semantic_cache: true`, so a chat request on this app really does embed
  through the `embed` alias against `local`. That is deliberate: an
  implementation that booked the INTERNAL embedding as a second row is caught by
  the one-row cases, and it cannot be caught on an app where the semantic tier
  never runs (a project rule's fourth target).
* THE PRICES ARE THE SHIPPED ONES. `anthropic/claude-haiku-4-5` is 1.00 in and
  5.00 out per million, `local/*` is zero, so the chat rows carry a NON-ZERO
  cost written out here as arithmetic (137 * 1 + 29 * 5 = 282 millionths) rather
  than recomputed from the table the implementation reads. A zero-priced row is
  never used to prove a cost was booked: what those rows prove is the
  `price_version`, which names the key that priced them and so cannot be
  satisfied by an `unpriced` fallthrough.

The contract these cases are written against, before it exists:

* `create_app` builds `app.state.prices` (a `PriceStore` over
  `settings.prices_path`), `app.state.db_pool` (one pool, decision D) and
  `app.state.cost_ledger` (`PostgresCostLedgerWriter` with a database URL, else
  `InMemoryCostLedgerWriter`);
* `book_cost` in `gateway/routes/chat.py` calls `record_cost`, takes the row it
  answers (decision G), increments `COST_USD` and answers the header value;
* the non-streamed served path, both cache-hit paths and `/v1/embeddings` set
  `X-Gateway-Cost-USD` from it; the streamed path books from its token sink and
  from a close hook, and sets no header at all (the headers are on the wire
  before the usage chunk exists).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from opsconsole.cache import (
    SEMANTIC_THRESHOLD,
    InProcessResponseCache,
    InProcessSemanticCache,
)
from opsconsole.core.errors import ProviderError
from opsconsole.core.settings import Settings
from opsconsole.gateway.app import close_adapters, create_app
from opsconsole.gateway.routes.chat import CACHE_STATUS_HIT, CACHE_STATUS_SEMANTIC
from opsconsole.ledger import (
    CostLedgerRow,
    CostLedgerWriter,
    InMemoryCostLedgerWriter,
)
from opsconsole.limits import PriceStore, PriceTableInvalid, price_for
from opsconsole.providers.base import (
    Adapter,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamEvent,
)

from .conftest import (
    API_KEYS,
    CHAT,
    EMBEDDINGS,
    FALLBACK_ALIAS,
    FALLBACK_CLASS,
    FALLBACK_TENANT,
    FIRST_PROVIDER,
    NO_PROVIDER_KEYS,
    SECOND_PROVIDER,
    SHIPPED_POLICY_PATH,
    STREAM_FRAGMENTS,
    STREAM_USAGE,
    STUB_EMBEDDING_USAGE,
    STUB_USAGE,
    TEST_KEYS,
    FallbackStubAdapter,
    MetricDelta,
    app_of,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
    retryable_error,
    stream_events,
    stub_response,
)

COST_HEADER = "x-gateway-cost-usd"
"""The published header name, lowercased as httpx returns it. Spelled as a
literal rather than imported, for the same reason
tests/gateway/test_response_headers.py spells the other six: it is a wire
contract, and a test that read the route's own constant would follow a rename
instead of failing on it."""

CACHE_HEADER = "x-gateway-cache"
TRACE_HEADER = "x-gateway-trace-id"

COST_USD = "gateway_cost_usd_total"
REQUESTS = "gateway_requests_total"
"""Family names as the exposition spells them, read here rather than imported
from another test module."""

TENANT = "evals"
DATA_CLASS = "internal"
SESSION = "sess-cost-wiring"
ALIAS = "agent-default"
PROVIDER = "anthropic"
REGION = "us"
POLICY_MODEL = "claude-haiku-4-5"
"""What the shipped policy resolves `evals`/`agent-default`/`internal` to, best
candidate first. The model is the POLICY's id, never the stub's own answer and
never the alias."""

EMBED_ALIAS = "embed"
EMBED_PROVIDER = "local_embed"
EMBED_REGION = "self"
EMBED_MODEL = "bge-small-en-v1.5"
"""The shipped `embed` alias's only candidate, which `/v1/embeddings` serves from
and which the semantic tier embeds through."""

LOCAL_PROVIDER = "local"
"""The CHAT server on the box we run, which is not the embedding server beside it:
one llama.cpp process serves one model, so `agent-default`'s self-region candidate
and the `embed` alias's candidate are two providers with two addresses."""

FALLBACK_MODEL = "qwen2.5-3b-instruct"
"""The model of `agent-default`'s SECOND candidate, `local`. The fallback row
must name this and not `claude-haiku-4-5`. Renamed with the shipped policy when
an earlier change sized the self-hosted model to the 3 GiB ceiling its pod carries: this
is a copy of what `routing.yaml` resolves to, so it moves when that file does."""

CHAT_PRICE_VERSION = f"{PROVIDER}/{POLICY_MODEL}@2025-10-15"
EMBED_PRICE_VERSION = f"{EMBED_PROVIDER}/{EMBED_MODEL}@2025-01-01"
FALLBACK_PRICE_VERSION = f"{LOCAL_PROVIDER}/{FALLBACK_MODEL}@2025-01-01"
"""The entries in force in the shipped `prices.yaml`, spelled as the literals the
row must carry. A row that priced the ALIAS, or that fell through to `unpriced`,
carries a different string and fails - which is what makes the zero-priced
self-hosted rows worth asserting at all. The embedding row and the fallback row are
priced from DIFFERENT keys now that the two servers are two providers."""

WHEN_PRICED = date(2026, 6, 1)
"""The instant the two price-store cases price at: after every `effective_from`
in the shipped file and after the fixture entry one of them writes, so neither
case can pass because no entry was in force yet."""

CHAT_COST = Decimal("0.000282")
"""(137 * 1.00 + 29 * 5.00) / 1_000_000, from `STUB_USAGE` and the shipped
`anthropic/claude-haiku-4-5` entry, written out so no case repeats the
implementation's arithmetic."""

STREAM_COST = Decimal("0.000691")
"""(431 * 1.00 + 52 * 5.00) / 1_000_000, from `STREAM_USAGE`. Different from
`CHAT_COST` in both directions, so a streamed row priced from the non-streamed
usage is a different number here."""

ZERO_COST = Decimal("0.000000")
"""What a cache hit costs (decision B) and what a zero-priced `local` row costs.
Written at six places because `Decimal("0") == Decimal("0.000000")`: the exponent
is asserted separately wherever the scale is the claim."""

DECIMAL_PLACES = 6
"""What `X-Gateway-Cost-USD` carries, exactly. A cost formatted through a float
loses the sixth place, and the ledger and the header then disagree by a
rounding."""

QUESTION = "What is the capital of Germany?"
SAME_QUESTION_REWORDED = "Which city is the capital city of Germany?"
"""Two ways of asking one thing. The embedding stub answers a fixed vector per
text, so the pair's cosine is a fact of this file rather than an accident of the
wording."""

NEAR_A: tuple[float, ...] = (1.0, 0.0)
NEAR_B: tuple[float, ...] = (1.0, 0.2)
UNKNOWN_PROBE_VECTOR: tuple[float, ...] = (0.0, 1.0)
"""`NEAR_A` and `NEAR_B` are about 0.9806 apart, above the shipped threshold;
anything the cases did not name embeds orthogonal to both, so an unnamed probe
can never accidentally hit a stored entry."""

STREAM_ANSWER_CHARS = 33
"""`len("".join(STREAM_FRAGMENTS))`, asserted in
`test_the_stream_fixtures_carry_the_character_counts_these_cases_estimate_from`
rather than trusted, because two of the estimate cases are arithmetic over it."""

FIRST_FRAGMENT_CHARS = 6
"""`len(STREAM_FRAGMENTS[0])`, the characters that reached the client before a
mid-stream failure."""

GATEWAY_HEADERS = (
    "x-gateway-provider",
    "x-gateway-region",
    "x-gateway-model",
    "x-gateway-cache",
    "x-gateway-redacted",
)
"""The header VALUES a failed ledger write must not disturb. The trace header and
the body's `id` are excluded because they differ between any two requests, which
is why that comparison is against a baseline request rather than a literal."""


# --------------------------------------------------------------------------
# the stubs
# --------------------------------------------------------------------------


class CostChatStub:
    """The completion adapter behind this module's app: fixed answers, every call kept.

    `chat_error`, `events` and the two stream-failure knobs are set per test and
    cleared by `reset`, so no case here needs an app of its own.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []
        self.chat_error: Exception | None = None
        self.events: tuple[StreamEvent, ...] = ()
        self.stream_error: Exception | None = None
        self.stream_error_at = 0
        self.reset()

    def reset(self) -> None:
        """Forget every recorded call and every scripted failure."""
        self.requests = []
        self.chat_error = None
        self.events = stream_events()
        self.stream_error = None
        self.stream_error_at = 0

    @property
    def call_count(self) -> int:
        """How many completions a route handed this adapter."""
        return len(self.requests)

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.chat_error is not None:
            raise self.chat_error
        return stub_response()

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Yield the scripted events, raising the scripted error at its position."""
        self.requests.append(request)
        for position, event in enumerate(self.events):
            if self.stream_error is not None and position == self.stream_error_at:
                raise self.stream_error
            yield event

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


class CostEmbedStub:
    """The embedding adapter under `local`: the `embed` alias and the semantic tier both.

    A fixed vector per text rather than a model, so the cosine the semantic hit
    depends on is decided here. It satisfies `Adapter` too - `chat` and
    `stream_chat` raise - because `app.state.adapters` is a mapping of `Adapter`
    and the chat route would otherwise be able to reach a provider that cannot
    serve it.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[EmbeddingRequest] = []
        self.vectors: dict[str, tuple[float, ...]] = {}
        self.reset()

    def reset(self) -> None:
        """Forget every recorded call and go back to the fixed vector table."""
        self.requests = []
        self.vectors = {QUESTION: NEAR_A, SAME_QUESTION_REWORDED: NEAR_B}

    @property
    def call_count(self) -> int:
        """How many times a route asked this adapter to embed anything."""
        return len(self.requests)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(
            model="stub-embedder-neither-alias-nor-policy-id",
            embeddings=tuple(
                Embedding(index=index, vector=self.vectors.get(text, UNKNOWN_PROBE_VECTOR))
                for index, text in enumerate(request.input)
            ),
            usage=STUB_EMBEDDING_USAGE,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


class RaisingCostWriter:
    """A cost writer that always fails, and counts the attempts.

    The count is what stops the "nothing changed" cases from being vacuous: a
    route that stopped booking rows at all would leave every status, header and
    body identical too.
    """

    def __init__(self) -> None:
        self.calls = 0

    def write(self, row: CostLedgerRow) -> None:
        self.calls += 1
        raise RuntimeError("the cost ledger is unreachable")


# --------------------------------------------------------------------------
# the app, and the writer seam
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_stub() -> CostChatStub:
    """The completion adapter under the policy provider `anthropic`."""
    return CostChatStub(name=PROVIDER)


@pytest.fixture(scope="module")
def embed_stub() -> CostEmbedStub:
    """The embedding adapter under `local_embed`, which the shipped `embed` alias names."""
    return CostEmbedStub(name=EMBED_PROVIDER)


@pytest.fixture(scope="module")
def cost_client(
    settings: Settings, chat_stub: CostChatStub, embed_stub: CostEmbedStub
) -> Iterator[TestClient]:
    """One app for the module: a chat adapter and an embedding adapter, built once."""
    adapters: dict[str, Adapter] = {chat_stub.name: chat_stub, embed_stub.name: embed_stub}
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh(
    cost_client: TestClient, chat_stub: CostChatStub, embed_stub: CostEmbedStub
) -> Iterator[None]:
    """Empty caches, a fresh ledger and reset stubs, never a new app."""
    chat_stub.reset()
    embed_stub.reset()
    app = app_of(cost_client)
    app.state.response_cache = InProcessResponseCache()
    app.state.semantic_cache = InProcessSemanticCache()
    app.state.cost_ledger = InMemoryCostLedgerWriter()
    yield


def install(client: TestClient, writer: CostLedgerWriter) -> CostLedgerWriter:
    """Put a cost writer on the app behind this client, and hand back the one replaced."""
    app = app_of(client)
    previous: CostLedgerWriter = app.state.cost_ledger
    app.state.cost_ledger = writer
    return previous


def rows_of(writer: CostLedgerWriter) -> tuple[CostLedgerRow, ...]:
    """The rows an in-memory writer holds, with the Protocol typing narrowed at one seam."""
    assert isinstance(writer, InMemoryCostLedgerWriter)
    return writer.rows


@pytest.fixture
def fallback_ledger(fallback_client: TestClient) -> Iterator[InMemoryCostLedgerWriter]:
    """A ledger on the shared fallback app for one test, then the old one back."""
    writer = InMemoryCostLedgerWriter()
    previous = install(fallback_client, writer)
    try:
        yield writer
    finally:
        install(fallback_client, previous)


# --------------------------------------------------------------------------
# the requests
# --------------------------------------------------------------------------


def post(client: TestClient, body: Mapping[str, Any]) -> Response:
    """POST one completion as `evals` under `internal`, with a session id stated.

    The session is sent on every request in this file because the row carries it:
    a booking that filled `session` from anywhere but the request context is only
    visible when the context has one.
    """
    response: Response = client.post(
        CHAT,
        json=dict(body),
        headers=auth_headers(
            key=TEST_KEYS[TENANT], tenant=TENANT, data_class=DATA_CLASS, session=SESSION
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
    """The exact tier's baseline: `temperature: 0` stated explicitly (#22)."""
    return chat_body(model=ALIAS, temperature=0)


Send = Callable[[TestClient, CostLedgerWriter], Response]
"""One measured request, with the writer installed immediately before it.

The writer is a parameter rather than a fixture so the same five senders drive
both the row cases (an `InMemoryCostLedgerWriter`) and the failure case (a
`RaisingCostWriter`), and so a priming request's row is never counted.
"""


def send_chat(client: TestClient, writer: CostLedgerWriter) -> Response:
    """One non-streamed completion."""
    install(client, writer)
    return post(client, chat_body(model=ALIAS))


def send_stream(client: TestClient, writer: CostLedgerWriter) -> Response:
    """The same request, streamed, with the provider reporting usage."""
    install(client, writer)
    return post(client, chat_body(model=ALIAS, stream=True))


def send_exact_hit(client: TestClient, writer: CostLedgerWriter) -> Response:
    """A second identical eligible request, served from the exact cache."""
    primed = post(client, eligible_body())
    assert primed.status_code == 200, primed.text
    install(client, writer)
    hit = post(client, eligible_body())
    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_HIT, "the arrangement did not hit"
    return hit


def send_semantic_hit(client: TestClient, writer: CostLedgerWriter) -> Response:
    """The same question reworded, served from the semantic cache."""
    primed = post(client, question_body(QUESTION))
    assert primed.status_code == 200, primed.text
    install(client, writer)
    hit = post(client, question_body(SAME_QUESTION_REWORDED))
    assert hit.headers[CACHE_HEADER] == CACHE_STATUS_SEMANTIC, "the arrangement did not hit"
    return hit


def send_embeddings(client: TestClient, writer: CostLedgerWriter) -> Response:
    """One embeddings request against the `embed` alias."""
    install(client, writer)
    return post_embed(client, embeddings_body())


class Booked(NamedTuple):
    """What one path's row must say. Every field comes from a different source."""

    alias: str
    provider: str
    model: str
    region: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost_usd: Decimal
    price_version: str
    estimated: bool


CHAT_ROW = Booked(
    alias=ALIAS,
    provider=PROVIDER,
    model=POLICY_MODEL,
    region=REGION,
    input_tokens=STUB_USAGE.prompt_tokens,
    output_tokens=STUB_USAGE.completion_tokens,
    cached=False,
    cost_usd=CHAT_COST,
    price_version=CHAT_PRICE_VERSION,
    estimated=False,
)

HIT_ROW = CHAT_ROW._replace(cached=True, cost_usd=ZERO_COST)
"""A hit books zero with the REPLAYED answer's token counts and the price version
in force (decisions B and C), so it differs from `CHAT_ROW` in exactly two
fields."""

PATHS: list[tuple[str, Send, Booked]] = [
    ("the non-streamed chat path", send_chat, CHAT_ROW),
    (
        "the streamed chat path",
        send_stream,
        CHAT_ROW._replace(
            input_tokens=STREAM_USAGE.prompt_tokens,
            output_tokens=STREAM_USAGE.completion_tokens,
            cost_usd=STREAM_COST,
        ),
    ),
    ("the exact-cache hit path", send_exact_hit, HIT_ROW),
    ("the semantic-cache hit path", send_semantic_hit, HIT_ROW),
    (
        "the embeddings path",
        send_embeddings,
        Booked(
            alias=EMBED_ALIAS,
            provider=EMBED_PROVIDER,
            model=EMBED_MODEL,
            region=EMBED_REGION,
            input_tokens=STUB_EMBEDDING_USAGE.prompt_tokens,
            output_tokens=0,
            cached=False,
            cost_usd=ZERO_COST,
            price_version=EMBED_PRICE_VERSION,
            estimated=False,
        ),
    ),
]

PATH_IDS = [case for case, _, _ in PATHS]

ONE_ROW_PATHS: list[tuple[str, Send]] = [(case, send) for case, send, _ in PATHS]
"""The same five paths, without the expectation: the count case reads no field."""

HEADER_PATHS: list[tuple[str, Send]] = [
    ("the non-streamed chat path", send_chat),
    ("the exact-cache hit path", send_exact_hit),
    ("the semantic-cache hit path", send_semantic_hit),
    ("the embeddings path", send_embeddings),
]
"""The paths that can price a call before the headers go out. The stream is
absent from this list on purpose, and its absence is asserted by
`test_the_cost_header_is_absent_on_a_stream`."""

HEADER_PATH_IDS = [case for case, _ in HEADER_PATHS]


def streamed_text(response: Response) -> str:
    """The text deltas of a streamed answer, joined, so the ANSWER can be compared."""
    text = ""
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        for choice in json.loads(payload).get("choices", ()):
            text += str(choice.get("delta", {}).get("content") or "")
    return text


def answer_of(response: Response) -> object:
    """The one part of a body that two identical requests agree on, per content type."""
    if response.headers["content-type"].startswith("text/event-stream"):
        return streamed_text(response)
    body = response.json()
    if "data" in body:
        return [item["embedding"] for item in body["data"]]
    return body["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# the fixtures these cases do arithmetic over
# --------------------------------------------------------------------------


def test_the_stream_fixtures_carry_the_character_counts_these_cases_estimate_from() -> None:
    """Catches this file going stale rather than an implementation: two estimate cases
    below assert a token count derived from these lengths, so a change to
    `STREAM_FRAGMENTS` in the shared conftest would otherwise turn them into assertions
    about a number nobody chose - green or red for a reason that has nothing to do with
    the route."""
    assert len("".join(STREAM_FRAGMENTS)) == STREAM_ANSWER_CHARS
    assert len(STREAM_FRAGMENTS[0]) == FIRST_FRAGMENT_CHARS


# --------------------------------------------------------------------------
# one row per served request, on each of the five paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "send"), ONE_ROW_PATHS, ids=PATH_IDS)
def test_a_served_request_books_exactly_one_row(
    cost_client: TestClient, case: str, send: Send
) -> None:
    """The first criterion, on all five paths. Catches a booking placed on the
    non-streamed return path alone, which leaves every streamed request - the majority of
    a chat product's traffic - unbilled; catches one placed on the chat route only, which
    loses every embedding; and catches the two shapes that OVERCOUNT, since the count is
    exact: a row per candidate and, on this app specifically, a second row for the
    INTERNAL embedding the semantic tier makes on the caller's behalf. The answer is read
    off the response so a path that returned an error body cannot pass by booking one row
    on its way out."""
    writer = InMemoryCostLedgerWriter()

    response = send(cost_client, writer)

    assert response.status_code == 200, response.text
    assert answer_of(response) not in ("", [], None)
    assert len(rows_of(writer)) == 1, case


@pytest.mark.parametrize(("case", "send", "expected"), PATHS, ids=PATH_IDS)
def test_the_booked_row_names_what_served(
    cost_client: TestClient, case: str, send: Send, expected: Booked
) -> None:
    """The second criterion, field by field, on all five paths. Catches a row filled from
    the request body rather than from the resolver's answer - `model` reading
    `agent-default`, `provider` reading the tenant's home region - which turns the
    console's per-model spend into a chart of one row; catches an embedding booked with
    the chat path's token counts; and catches the `price_key` mistake the criterion names
    twice, because `price_version` is a literal built from the SERVING provider and model
    and neither the alias nor an `unpriced` fallthrough can spell it. `provider` and
    `region` are compared against the headers the SAME response carried, so the row and
    the wire contract cannot drift apart."""
    writer = InMemoryCostLedgerWriter()

    response = send(cost_client, writer)
    assert response.status_code == 200, response.text

    row = rows_of(writer)[0]
    assert row.tenant == TENANT
    assert row.session == SESSION
    assert row.trace_id == response.headers[TRACE_HEADER]
    assert row.alias == expected.alias
    assert row.provider == expected.provider
    assert row.model == expected.model
    assert row.region == expected.region
    assert row.input_tokens == expected.input_tokens
    assert row.output_tokens == expected.output_tokens
    assert row.cached is expected.cached
    assert row.cost_usd == expected.cost_usd
    assert row.price_version == expected.price_version
    assert row.estimated is expected.estimated
    assert row.ts.tzinfo is not None
    assert row.provider == response.headers["x-gateway-provider"]
    assert row.region == response.headers["x-gateway-region"]


def test_a_request_that_fell_back_books_one_row_against_the_serving_provider(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    fallback_ledger: InMemoryCostLedgerWriter,
) -> None:
    """The criterion's own sentence: the row names the candidate that SERVED. Catches a
    booking made where the first candidate is CHOSEN rather than where it answered, which
    bills a tenant for a provider that refused the request, and catches a row per ATTEMPT,
    which double-books every fallback. The two candidates are priced from different keys,
    so `price_version` fails on a row that named the right provider and priced the wrong
    one - the exact drift `price_key` passed from the serving adapter exists to
    prevent."""
    fallback_adapters[FIRST_PROVIDER].chat_error = retryable_error(FIRST_PROVIDER)

    response = fallback_client.post(
        CHAT,
        json=chat_body(model=FALLBACK_ALIAS),
        headers=auth_headers(
            key=TEST_KEYS[FALLBACK_TENANT], tenant=FALLBACK_TENANT, data_class=FALLBACK_CLASS
        ),
    )

    assert response.status_code == 200, response.text
    assert len(fallback_ledger.rows) == 1
    row = fallback_ledger.rows[0]
    assert row.provider == SECOND_PROVIDER
    assert row.model == FALLBACK_MODEL
    assert row.price_version == FALLBACK_PRICE_VERSION


# --------------------------------------------------------------------------
# the counter (third criterion)
# --------------------------------------------------------------------------


def test_the_counter_moves_by_exactly_the_rows_cost(
    cost_client: TestClient, metrics: MetricDelta
) -> None:
    """The third criterion. Catches a counter fed a number the route computed for itself
    rather than the one on the row - the whole reason `book_cost` is one function - and
    catches a wrong label set: the delta is read off an exactly-labelled series built from
    the ROW's own tenant, provider and model, so a counter labelled with the alias or with
    the tenant's home region moves a different series and this reads zero. The total is
    asserted as well, so a counter moved under two label sets at once fails rather than
    passing on the series that happened to be right. The cost is non-zero by
    construction, which a zero-priced path could not tell from a counter nobody fed."""
    writer = InMemoryCostLedgerWriter()

    response = send_chat(cost_client, writer)

    assert response.status_code == 200, response.text
    row = rows_of(writer)[0]
    assert row.cost_usd == CHAT_COST
    assert metrics.delta(
        COST_USD, tenant=row.tenant, provider=row.provider, model=row.model
    ) == pytest.approx(float(row.cost_usd))
    assert metrics.total(COST_USD) == pytest.approx(float(row.cost_usd))


def test_a_request_that_books_no_row_moves_no_cost(
    cost_client: TestClient, chat_stub: CostChatStub, metrics: MetricDelta
) -> None:
    """The third criterion's second half, "and by nothing when no row is booked". Catches
    a placeholder increment on the failure path - a zero, or worse the price of a call
    that never happened - which would draw spend on a dashboard for a request the provider
    never served. The request count is what makes the zero mean something: a request was
    made, refused and counted while the cost family stayed flat."""
    chat_stub.chat_error = ProviderError("anthropic: the provider answered 400", retryable=False)
    writer = InMemoryCostLedgerWriter()
    install(cost_client, writer)

    response = post(cost_client, chat_body(model=ALIAS))

    assert response.status_code == 502, response.text
    assert rows_of(writer) == ()
    assert metrics.total(COST_USD) == 0.0
    assert metrics.total(REQUESTS) == 1


# --------------------------------------------------------------------------
# the header (fourth criterion)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "send"), HEADER_PATHS, ids=HEADER_PATH_IDS)
def test_the_cost_header_is_the_rows_decimal_at_six_places(
    cost_client: TestClient, case: str, send: Send
) -> None:
    """The fourth criterion. Catches the header formatted through a float, which drops the
    sixth place and makes the header and the ledger disagree by a rounding on exactly the
    small numbers a per-request cost is; catches a header rendered at two places, which is
    `0.00` for almost every real request; and catches a header built from a second
    pricing rather than from the row. The digits after the point are counted, so a value
    that happens to be equal as a number but printed as `0.000282000` fails, and the
    string is compared to the row's `Decimal` as well, so a correctly-shaped wrong number
    fails too."""
    writer = InMemoryCostLedgerWriter()

    response = send(cost_client, writer)

    assert response.status_code == 200, response.text
    value = response.headers[COST_HEADER]
    row = rows_of(writer)[0]
    assert Decimal(value) == row.cost_usd
    assert len(value.partition(".")[2]) == DECIMAL_PLACES, value


def test_the_cost_header_carries_the_literal_six_place_rendering(
    cost_client: TestClient,
) -> None:
    """The same criterion with the arithmetic spelled out, so the parametrised case above
    cannot be satisfied by a route that agrees with its own wrong row. Catches a header
    and a row that are consistently wrong - both priced from the input rate in both
    directions, say - because the literal here comes from the shipped price entry and the
    fixture usage rather than from anything the route computed."""
    writer = InMemoryCostLedgerWriter()

    response = send_chat(cost_client, writer)

    assert response.status_code == 200, response.text
    assert response.headers[COST_HEADER] == "0.000282"


def test_the_cost_header_is_absent_on_a_stream(cost_client: TestClient) -> None:
    """The fourth criterion's "absent on a stream", which is forced rather than chosen: the
    headers are on the wire before the usage chunk exists. Catches an implementation that
    books early to have a number for the header, which would price every streamed request
    before a single token was counted, and catches a header set to an empty string or a
    zero rather than left off. The row is asserted too, so a route that dropped the
    header by dropping the booking fails."""
    writer = InMemoryCostLedgerWriter()

    response = send_stream(cost_client, writer)

    assert response.status_code == 200, response.text
    assert COST_HEADER not in {name.lower() for name in response.headers}
    assert len(rows_of(writer)) == 1


def test_the_cost_header_is_absent_on_an_error_response(
    cost_client: TestClient, chat_stub: CostChatStub
) -> None:
    """The rest of "and on every error response". Catches a header added in middleware or
    in a response factory rather than on the served path, which would put a cost on a 502
    the provider never charged for and teach a client to read a number that means
    nothing."""
    chat_stub.chat_error = ProviderError("anthropic: the provider answered 400", retryable=False)
    writer = InMemoryCostLedgerWriter()
    install(cost_client, writer)

    response = post(cost_client, chat_body(model=ALIAS))

    assert response.status_code == 502, response.text
    assert COST_HEADER not in {name.lower() for name in response.headers}
    assert rows_of(writer) == ()


# --------------------------------------------------------------------------
# a cache hit (fifth criterion, decisions B and C)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "send"),
    [("the exact tier", send_exact_hit), ("the semantic tier", send_semantic_hit)],
    ids=["the exact tier", "the semantic tier"],
)
def test_a_cache_hit_books_zero_with_its_tokens(
    cost_client: TestClient, case: str, send: Send
) -> None:
    """Decision B on both hit tiers. Catches a hit booked at the cost it AVOIDED, which
    puts money that was never spent into the spend column where no `WHERE cached = false`
    can find it, and catches the other direction - a hit booked with its tokens zeroed -
    which would lose the numbers an earlier change's cache-savings figure is derived from. The
    exponent is asserted because `Decimal("0") == Decimal("0.000000")`, and the price
    version is asserted because a zero cost with `unpriced` beside it is a different
    claim: it would say the model has no price rather than that the request was free."""
    writer = InMemoryCostLedgerWriter()

    response = send(cost_client, writer)

    assert response.status_code == 200, response.text
    row = rows_of(writer)[0]
    assert row.cached is True
    assert row.cost_usd == ZERO_COST
    assert row.cost_usd.as_tuple().exponent == -DECIMAL_PLACES
    assert (row.input_tokens, row.output_tokens) == (
        STUB_USAGE.prompt_tokens,
        STUB_USAGE.completion_tokens,
    )
    assert row.price_version == CHAT_PRICE_VERSION
    assert response.headers[COST_HEADER] == "0.000000"


# --------------------------------------------------------------------------
# the stream, and decision E
# --------------------------------------------------------------------------


def test_a_stream_with_usage_books_exactly_one_row(cost_client: TestClient) -> None:
    """Decision E's double-booking risk, which is the expensive one: the usage event and
    the close hook both firing writes two rows and doubles every streamed tenant's spend,
    silently and only in production. Catches that, and catches the row being flagged
    `estimated` on a stream that really did report usage - which would make a metered
    number indistinguishable from a guess. The token counts are the streaming fixture's,
    so a row filled from the non-streamed usage fails here as a wrong number."""
    writer = InMemoryCostLedgerWriter()

    response = send_stream(cost_client, writer)

    assert response.status_code == 200, response.text
    assert "[DONE]" in response.text
    assert len(rows_of(writer)) == 1
    row = rows_of(writer)[0]
    assert row.estimated is False
    assert (row.input_tokens, row.output_tokens) == (
        STREAM_USAGE.prompt_tokens,
        STREAM_USAGE.completion_tokens,
    )
    assert row.cost_usd == STREAM_COST


def test_a_stream_that_reports_no_usage_books_one_estimated_row(
    cost_client: TestClient, chat_stub: CostChatStub
) -> None:
    """The other half of decision E. Catches a close hook that books nothing, which makes
    "every request writes a row" false exactly when an operator is looking - the provider
    has already charged - and catches one that books a zero-token row, which makes a whole
    provider look free. The output tokens are the characters actually sent divided by the
    published estimator's four, so a hook that booked an empty answer fails on the
    number."""
    chat_stub.events = stream_events(usage=None)
    writer = InMemoryCostLedgerWriter()

    response = send_stream(cost_client, writer)

    assert response.status_code == 200, response.text
    assert len(rows_of(writer)) == 1
    row = rows_of(writer)[0]
    assert row.estimated is True
    assert row.output_tokens == STREAM_ANSWER_CHARS // 4
    assert row.price_version == CHAT_PRICE_VERSION


def test_a_stream_that_fails_after_its_first_frame_books_one_estimated_row(
    cost_client: TestClient, chat_stub: CostChatStub
) -> None:
    """The typed failure decision E is written for: a 200 is already on the wire, one
    fragment has reached the client and no usage event ever arrives. Catches a booking
    that only happens on the success path, which loses the row for the request an operator
    is most likely to go looking for, and catches one that estimates from the WHOLE
    scripted answer rather than from the characters that actually left - the two differ
    here, so a hook counting the wrong thing is a wrong number rather than a coincidence."""
    chat_stub.stream_error = ProviderError("anthropic: the connection dropped", retryable=True)
    chat_stub.stream_error_at = 2
    writer = InMemoryCostLedgerWriter()

    response = send_stream(cost_client, writer)

    assert response.status_code == 200, response.text
    assert "[DONE]" not in response.text
    assert len(rows_of(writer)) == 1
    row = rows_of(writer)[0]
    assert row.estimated is True
    assert row.output_tokens == FIRST_FRAGMENT_CHARS // 4


# --------------------------------------------------------------------------
# a failing writer changes nothing (the unhappy path)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "send", "expected"), PATHS, ids=PATH_IDS)
def test_a_failing_writer_changes_no_status_body_or_header(
    cost_client: TestClient, case: str, send: Send, expected: Booked
) -> None:
    """The unhappy-path criterion, on all five paths. Catches an unguarded write: a
    Postgres blip would take out a gateway whose behaviour is otherwise unchanged, and on
    the streamed path it would do so with a 200 already on the wire, truncating an answer
    the client had begun reading. Catches the subtler shape decision G exists for as well
    - a header derived from the WRITE rather than from the row - because the cost header
    must still read what the request cost when the database refused the row. The baseline
    request is what makes this a comparison rather than a restatement of the happy path,
    and the attempt count is what stops a route that silently stopped booking from
    passing."""
    good = send(cost_client, InMemoryCostLedgerWriter())
    assert good.status_code == 200, good.text

    failing = RaisingCostWriter()
    bad = send(cost_client, failing)

    assert failing.calls == 1
    assert bad.status_code == good.status_code
    assert set(bad.headers) == set(good.headers)
    for name in GATEWAY_HEADERS:
        assert bad.headers[name] == good.headers[name], name
    if COST_HEADER in {name.lower() for name in good.headers}:
        assert bad.headers[COST_HEADER] == good.headers[COST_HEADER]
        assert Decimal(bad.headers[COST_HEADER]) == expected.cost_usd
    assert answer_of(bad) == answer_of(good)


def test_a_failing_writer_still_lets_the_stream_finish_its_frames(
    cost_client: TestClient,
) -> None:
    """The streamed half of the same criterion, stated as the thing a client sees. Catches
    a booking raised inside the frame generator: the answer would stop mid-sentence with
    no `[DONE]` and no error frame, which a client cannot tell from a completed answer,
    and the status would already be 200 so nothing else could report it."""
    failing = RaisingCostWriter()

    response = send_stream(cost_client, failing)

    assert response.status_code == 200, response.text
    assert failing.calls == 1
    assert streamed_text(response) == "".join(STREAM_FRAGMENTS)
    assert response.text.endswith("data: [DONE]\n\n")


# --------------------------------------------------------------------------
# the wiring (the last criterion, decision D)
# --------------------------------------------------------------------------


def test_the_app_builds_the_in_memory_cost_ledger_without_a_database_url(
    settings: Settings,
) -> None:
    """The reason this whole file needs no backing service. Catches a factory that builds
    a pool unconditionally: `build_pool` opens connections, so every app in this session -
    and every local run without Postgres - would hold a pool retrying in the background.
    The settings fixture carries no `database_url`, which the root tests/conftest.py also
    guarantees by clearing the variable."""
    assert settings.database_url is None

    app = create_app(settings, NO_PROVIDER_KEYS)

    assert isinstance(app.state.cost_ledger, InMemoryCostLedgerWriter)
    assert app.state.db_pool is None
    close_adapters(app.state.adapters)


def test_both_postgres_writers_share_one_pool() -> None:
    """Decision D. Catches a pool per writer, which is twice the server-side connections
    for one dependency and is invisible until connection limits bite in production - no
    behaviour changes, no test fails, and the readiness probe stays green. Identity is
    asserted, not equality: two pools against the same URL compare equal on nothing useful
    and would pass any looser check."""
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        database_url="postgresql://opsconsole@127.0.0.1:5432/opsconsole",
    )

    app = create_app(settings, NO_PROVIDER_KEYS)

    try:
        pool = app.state.db_pool
        assert pool is not None
        assert app.state.residency_log._pool is pool
        assert app.state.cost_ledger._pool is pool
    finally:
        app.state.db_pool.close()
        close_adapters(app.state.adapters)


def test_create_app_refuses_a_malformed_prices_file(tmp_path: Path) -> None:
    """The startup half of the criterion, and the reason the store is built in the factory
    at all. Catches a `PriceStore` built lazily on the first request, where a typo in
    `prices.yaml` becomes a 500 on live traffic hours after the deploy that caused it
    instead of a process that refuses to start. The same rule the key store and the policy
    snapshot already follow."""
    path = tmp_path / "prices.yaml"
    path.write_text("prices: [not, a, mapping]\n", encoding="utf-8")
    settings = Settings(
        api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, prices_path=path
    )

    with pytest.raises(PriceTableInvalid):
        create_app(settings, NO_PROVIDER_KEYS)


def test_the_price_store_is_built_over_the_configured_path(tmp_path: Path) -> None:
    """Catches a factory that ignores `settings.prices_path` and hardcodes the shipped
    file: every case above would still pass, and a deployment pointing at its own price
    table would silently bill against ours. The key asserted here exists in no shipped
    `prices.yaml`, so only a store built over this path can answer it."""
    path = tmp_path / "prices.yaml"
    path.write_text(
        "prices:\n"
        "  someprovider/some-model:\n"
        "    - effective_from: 2026-01-01\n"
        '      input_usd_per_1m: "2.00"\n'
        '      output_usd_per_1m: "9.00"\n',
        encoding="utf-8",
    )
    settings = Settings(
        api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, prices_path=path
    )

    app = create_app(settings, NO_PROVIDER_KEYS)

    try:
        store = app.state.prices
        assert isinstance(store, PriceStore)
        entry = price_for(store.table, "someprovider/some-model", WHEN_PRICED)
        assert entry is not None
        assert entry.input_usd_per_1m == Decimal("2.00")
    finally:
        close_adapters(app.state.adapters)


def test_the_shipped_price_store_prices_the_key_the_chat_path_books_against(
    settings: Settings,
) -> None:
    """Catches the arrangement this whole file rests on going stale: if
    `anthropic/claude-haiku-4-5` ever leaves `prices.yaml`, every cost above becomes
    `0.000000` with `unpriced` beside it, and the equality cases would fail with no
    indication that the price table, and not the route, is what changed."""
    app = create_app(settings, NO_PROVIDER_KEYS)

    try:
        entry = price_for(app.state.prices.table, f"{PROVIDER}/{POLICY_MODEL}", WHEN_PRICED)
        assert entry is not None
        assert entry.input_usd_per_1m == Decimal("1.00")
        assert entry.output_usd_per_1m == Decimal("5.00")
    finally:
        close_adapters(app.state.adapters)


def test_the_semantic_tier_is_live_on_this_modules_app(
    cost_client: TestClient, embed_stub: CostEmbedStub
) -> None:
    """Not a claim about the route: a claim about the ARRANGEMENT the one-row cases lean
    on. If no embedding adapter were reachable, the semantic tier would never run and
    `test_a_served_request_books_exactly_one_row` would be unable to catch an
    implementation that books the internal embedding as a second row - it would pass for
    the wrong reason, which is exactly the vacuous case a project rule names. The
    embedding count is what proves the tier really ran."""
    response = post(cost_client, question_body(QUESTION))

    assert response.status_code == 200, response.text
    assert embed_stub.call_count == 1
    assert response.headers[CACHE_HEADER] not in (CACHE_STATUS_HIT, CACHE_STATUS_SEMANTIC)


def test_the_embedding_adapter_serves_the_alias_the_rows_name(
    cost_client: TestClient, embed_stub: CostEmbedStub
) -> None:
    """The same, for the embeddings path: catches a fixture whose `embed` alias resolves
    somewhere other than `local_embed`, which would make the embeddings row's provider and price
    version assertions above statements about the wrong candidate."""
    response = post_embed(cost_client, embeddings_body())

    assert response.status_code == 200, response.text
    assert embed_stub.call_count == 1
    assert response.headers["x-gateway-provider"] == EMBED_PROVIDER
    assert response.headers["x-gateway-model"] == EMBED_MODEL


def test_the_stub_vectors_sit_where_the_semantic_hit_assumes(
    embed_stub: CostEmbedStub,
) -> None:
    """The last arrangement check: catches a vector table edited into two texts that are
    NOT neighbours, which would turn every semantic-hit case above into a miss asserting
    hit-shaped things. Read off the stub rather than restated, and compared against the
    shipped threshold rather than a number spelled here."""
    first: Sequence[float] = embed_stub.vectors[QUESTION]
    second: Sequence[float] = embed_stub.vectors[SAME_QUESTION_REWORDED]
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norms = sum(a * a for a in first) ** 0.5 * sum(b * b for b in second) ** 0.5

    assert dot / norms > SEMANTIC_THRESHOLD
