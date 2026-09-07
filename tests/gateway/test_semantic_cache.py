"""The semantic tier on `POST /v1/chat/completions`: the embedding, the hit, and every failure.

Everything here is settled by a real request through `TestClient` against an app
`create_app` built. `tests/cache/test_vectors.py` already
settles what the store does with a vector; the question in this file is whether
the ROUTE embeds the right text, at the right moment, for the right requests -
and whether it survives every way that embedding can fail.

Three witnesses, and each case names which one it leans on:

* THE EMBEDDING STUB'S CALL COUNT. "Nothing was embedded" is not visible in a
  header, a body or a metric: a route that embedded on every request and simply
  never found a neighbour answers `X-Gateway-Cache: miss` exactly like a route
  that refused to embed at all. So every ineligibility row and the exact-hit row
  assert `embed_calls`, not the header.
* THE CHAT STUB'S CALL COUNT. `semantic` in a header proves nothing on its own -
  a route could serve a fresh completion and label it - so the hit pair asserts
  ONE chat call across two requests.
* THE STUB'S RECORDED `EmbeddingRequest`. The redaction row reads the outbound
  request rather than the answer, because a route that redacted only what it
  returned, and embedded the raw text, would look identical from the body.

The vectors are fixed and chosen so the arithmetic is a fact of this file rather
than an accident: `NEAR_A` and `NEAR_B` sit above `SEMANTIC_THRESHOLD` and `FAR`
sits below it, asserted directly by
`test_the_stub_vectors_sit_where_every_case_here_assumes`. Nothing restates
0.97, and nothing restates 8192: both are imported from the modules that publish
them.

Bodies here carry NO `temperature`, so the exact tier is never eligible and
cannot decide any outcome the semantic tier is being asked about. The one
exception is `test_an_exact_hit_returns_before_anything_is_embedded`, where the
exact tier is the control under test.

a project rule: one app for the module, built once, plus four more for the
arrangements a stub swap cannot express (a policy whose `embed` alias is refused
by residency, a policy with no `embed` alias at all, an app with no
embedding-capable adapter, and the `local`-only app a `pii` request reaches).
Per-test isolation is the stubs' reset and fresh `response_cache`,
`semantic_cache` and `residency_log` objects on `app.state` - never a new app,
exactly as `fresh_breaker` in this directory's `conftest.py` replaces the one
object that accumulates state.

The semantic store is reached by NAME, `app.state.semantic_cache`, never by
shape: this app now has four collaborators with a `put` and a `get`-shaped pair,
and selecting one by its methods is what a project rule exists for.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from opsconsole.cache import (
    SEMANTIC_THRESHOLD,
    InProcessResponseCache,
    InProcessSemanticCache,
    QdrantSemanticCache,
)
from opsconsole.core.errors import ProviderError
from opsconsole.core.metrics import SEMANTIC_CACHE_KIND
from opsconsole.core.settings import Settings
from opsconsole.gateway.app import build_qdrant_data_client, close_adapters, create_app
from opsconsole.gateway.probes import PROBE_TIMEOUT_SECONDS
from opsconsole.gateway.routes.chat import (
    CACHE_SPAN,
    CACHE_STATUS_HIT,
    CACHE_STATUS_MISS,
    CACHE_STATUS_SEMANTIC,
    EMBED_ALIAS,
    SEMANTIC_SPAN,
)
from opsconsole.gateway.routes.embeddings import MAX_EMBEDDING_INPUT_CHARS
from opsconsole.ledger import InMemoryResidencyLogWriter, ResidencyLogRow
from opsconsole.providers.base import (
    Adapter,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    StreamEvent,
)

from .conftest import (
    API_KEYS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    MetricDelta,
    app_of,
    build_chat_client,
    chat_body,
    post_chat,
    stream_events,
    stub_response,
)

TENANT = "evals"
DATA_CLASS = "internal"
ALIAS = "agent-default"
"""The one shipped alias carrying `semantic_cache: true`
(`tests/routing/test_policy_load.py::test_only_agent_default_carries_the_semantic_cache_switch`)."""

OFF_ALIAS = "agent-strong"
"""An alias with no `semantic_cache` key, resolvable for the same tenant and data
class, so the alias flag is the ONLY difference between it and `ALIAS`."""

PROVIDER = "anthropic"
REGION = "us"

EMBED_PROVIDER = "local"
EMBED_MODEL = "bge-small-en-v1.5"
"""The provider and the resolved model of the shipped `embed` alias's only
candidate. Stated because no header carries them: the embedding is not the
request's candidate, which is the whole content of decision G."""

REQUESTS = "gateway_requests_total"
TOKENS = "gateway_tokens_total"
CACHE_HITS = "gateway_cache_hits_total"

DIRECTION_IN = "in"
DIRECTION_OUT = "out"

CHAT_ANSWER = "Berlin is the capital of Germany."

# Two ways of asking one question, and a third question. The pair is what the
# tier exists for (#23's "the same question in different words"); the third is
# what stops every hit case here passing on a route that serves the nearest
# entry whatever its distance.
QUESTION = "What is the capital of Germany?"
SAME_QUESTION_REWORDED = "Which city is the capital city of Germany?"
DIFFERENT_QUESTION = "How do I renew a passport?"

NEAR_A: tuple[float, ...] = (1.0, 0.0)
NEAR_B: tuple[float, ...] = (1.0, 0.2)
FAR: tuple[float, ...] = (1.0, 1.0)
"""Fixed vectors, in two dimensions so the cosine is checkable by hand:
`NEAR_A`/`NEAR_B` are 1/sqrt(1.04) apart, about 0.9806, and `NEAR_A`/`FAR` are
1/sqrt(2) apart, about 0.7071. The threshold they straddle is imported, never
restated, and where they sit is asserted rather than asserted-about
(`test_the_stub_vectors_sit_where_every_case_here_assumes`)."""

UNKNOWN_PROBE_VECTOR: tuple[float, ...] = (0.0, 1.0)
"""What the stub embeds any text the cases did not name, orthogonal to `NEAR_A`
so an unnamed probe can never accidentally hit a stored one."""

EMBED_PROMPT_TOKENS = 61
"""What the embedding stub reports it consumed. Distinct from the chat stub's
`STUB_USAGE` numbers, so a booking under the wrong provider, the wrong model or
the wrong direction is a wrong NUMBER here and not a coincidence."""

STUB_EMBEDDING_MODEL = "stub-embedder-neither-alias-nor-policy-id"
"""What the embedding stub says it embedded with: neither `embed` nor the
policy's `bge-small-en-v1.5`, so a route that booked the adapter's own answer as
the `model` label is caught rather than accidentally right."""

EMAIL = "alice@example.com"
TEXT_WITH_PII = f"write to {EMAIL} about the capital of Germany"
REDACTED_PII = "write to <EMAIL_1> about the capital of Germany"
"""The same placeholder shape `tests/gateway/test_inline_redaction.py` pins for
the chat route. The raw address is what must never reach the embedding adapter."""

PII_CLASS = "pii"
LOCAL_PROVIDER = "local"
"""`evals` under `pii` allows region `self` alone, so `local` is both the chat
candidate and the `embed` candidate - which is why that row needs an adapter
that can do both."""

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
)

EMBED_FAILURE_TEXT = "local: the embedder answered 503 for request req_zqx_13c"
STORE_FAILURE_TEXT = "Error 111 connecting to qdrant:6333. Connection refused."
"""Markers, so a route that let either failure escape shows an internal host, a
port or a provider's own words in a body a caller can read."""


# --------------------------------------------------------------------------
# the stubs
# --------------------------------------------------------------------------


class ChatStub:
    """The completion adapter behind this module's apps: a fixed answer, every call kept.

    `call_count` is the load-bearing part of every hit case: whether a provider
    was dialled a second time is a fact only this object holds.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []

    def reset(self) -> None:
        """Forget every recorded call (a project rule's per-test isolation)."""
        self.requests = []

    @property
    def call_count(self) -> int:
        """How many completions a route handed this adapter."""
        return len(self.requests)

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return stub_response(content=CHAT_ANSWER)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        yield from stream_events()

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


class EmbedStub:
    """The embedding adapter behind the `embed` alias, answering one fixed vector per text.

    A mapping rather than a model: the cases need two probes at a KNOWN cosine,
    and a real embedder would make the distance an accident of the wording. Text
    the cases did not name embeds to `UNKNOWN_PROBE_VECTOR`.

    `requests` is what the redaction row and the boundary rows read; `error` is
    what the failure row sets, and `reset()` clears both, so no case here needs
    an app of its own for a failure.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[EmbeddingRequest] = []
        self.vectors: dict[str, tuple[float, ...]] = {}
        self.error: Exception | None = None
        self.reset()

    def reset(self) -> None:
        """Forget every recorded call and go back to the fixed vector table."""
        self.requests = []
        self.error = None
        self.vectors = {
            QUESTION: NEAR_A,
            SAME_QUESTION_REWORDED: NEAR_B,
            DIFFERENT_QUESTION: FAR,
        }

    @property
    def call_count(self) -> int:
        """How many times a route asked this adapter to embed anything."""
        return len(self.requests)

    @property
    def inputs(self) -> tuple[str, ...]:
        """Every text this adapter was asked to embed, in order."""
        return tuple(text for request in self.requests for text in request.input)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return EmbeddingResponse(
            model=STUB_EMBEDDING_MODEL,
            embeddings=tuple(
                Embedding(index=index, vector=self.vectors.get(text, UNKNOWN_PROBE_VECTOR))
                for index, text in enumerate(request.input)
            ),
            usage=EmbeddingUsage(
                prompt_tokens=EMBED_PROMPT_TOKENS, total_tokens=EMBED_PROMPT_TOKENS
            ),
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


class ChatAndEmbedStub(ChatStub):
    """One adapter that chats AND embeds, for the row where both resolve to `local`.

    A `pii` request may only reach region `self`, so the chat candidate and the
    `embed` candidate are the same provider and one object has to answer both.
    The embedding half is delegated rather than reimplemented, so the recorded
    requests this file reads are the same shape everywhere.
    """

    def __init__(self, *, name: str, embedder: EmbedStub) -> None:
        super().__init__(name=name)
        self.embedder = embedder

    def reset(self) -> None:
        super().reset()
        self.embedder.reset()

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return self.embedder.embed(request)


class BrokenSemanticCache:
    """A semantic store whose chosen half fails the way an unreachable Qdrant does.

    Deliberately NOT `QdrantSemanticCache` pointed at a dead port: that class
    swallows its own failures (13b decision G), so it could not tell whether the
    ROUTE survives a store that raises. The two cases using it are about the
    route.
    """

    def __init__(self, *, on_search: bool, on_put: bool) -> None:
        self.on_search = on_search
        self.on_put = on_put
        self.searches = 0
        self.puts = 0

    def search(
        self,
        tenant: str,
        vector: Sequence[float],
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> ChatResponse | None:
        self.searches += 1
        if self.on_search:
            raise RuntimeError(STORE_FAILURE_TEXT)
        return None

    def put(
        self,
        tenant: str,
        vector: Sequence[float],
        answer: ChatResponse,
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> None:
        self.puts += 1
        if self.on_put:
            raise RuntimeError(STORE_FAILURE_TEXT)


# --------------------------------------------------------------------------
# the apps
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_stub() -> ChatStub:
    """The completion adapter under the policy provider `anthropic`."""
    return ChatStub(name=PROVIDER)


@pytest.fixture(scope="module")
def embed_stub() -> EmbedStub:
    """The embedding adapter under `local`, which the shipped `embed` alias names."""
    return EmbedStub(name=EMBED_PROVIDER)


@pytest.fixture(scope="module")
def semantic_client(
    settings: Settings, chat_stub: ChatStub, embed_stub: EmbedStub
) -> Iterator[TestClient]:
    """One app for the module: a chat adapter and an embedding adapter, built once."""
    adapters: dict[str, Adapter] = {chat_stub.name: chat_stub, embed_stub.name: embed_stub}
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh(
    semantic_client: TestClient, chat_stub: ChatStub, embed_stub: EmbedStub
) -> Iterator[None]:
    """Empty caches, an empty log and reset stubs, never a new app."""
    chat_stub.reset()
    embed_stub.reset()
    state = app_of(semantic_client).state
    state.response_cache = InProcessResponseCache()
    state.semantic_cache = InProcessSemanticCache()
    state.residency_log = InMemoryResidencyLogWriter()
    yield


@pytest.fixture(scope="module")
def pii_stub() -> ChatAndEmbedStub:
    """The one adapter a `pii` request can reach: it both chats and embeds."""
    return ChatAndEmbedStub(name=LOCAL_PROVIDER, embedder=EmbedStub(name=LOCAL_PROVIDER))


@pytest.fixture(scope="module")
def pii_client(settings: Settings, pii_stub: ChatAndEmbedStub) -> Iterator[TestClient]:
    """An app whose only adapter is under `local`, the sole region a `pii` class allows."""
    with build_chat_client(settings, {pii_stub.name: pii_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def chat_only_client(settings: Settings) -> Iterator[TestClient]:
    """An app with a chat adapter and NO embedding-capable adapter anywhere.

    The shipped `embed` alias then has no healthy candidate and `resolve()`
    answers `AllProvidersUnavailable` - the 503 half of decision B, and the
    arrangement every other chat suite in this directory already runs under.
    """
    with build_chat_client(settings, {PROVIDER: ChatStub(name=PROVIDER)}) as client:
        yield client


# A policy in which the CHAT alias is allowed and the `embed` alias is refused by
# residency alone. `cuan` is an EU tenant, so the outer clamp drops `embed_us`
# even though the data class rule names `us` and even though an embedding-capable
# adapter is registered for it: nothing but residency can produce the refusal.
CHAT_EU_PROVIDER = "chat_eu"
EMBED_US_PROVIDER = "embed_us"
REFUSAL_TENANT = "cuan"

REFUSAL_POLICY = f"""
providers:
  {CHAT_EU_PROVIDER}:
    kind: anthropic
    region: eu
    base_url: https://chat-eu.invalid
  {EMBED_US_PROVIDER}:
    kind: openai
    region: us
    base_url: https://embed-us.invalid

eu_regions: [eu, self]

aliases:
  {ALIAS}:
    semantic_cache: true
    candidates:
      - {{ provider: {CHAT_EU_PROVIDER}, model: model-chat-eu }}
  {EMBED_ALIAS}:
    candidates:
      - {{ provider: {EMBED_US_PROVIDER}, model: model-embed-us }}

tenants:
  {REFUSAL_TENANT}:
    residency: eu
    data_class_rules:
      {DATA_CLASS}: {{ allow_regions: [us, eu, self] }}
    budget_usd_month: "15.00"
    rate_limit: {{ rpm: 60, tpm: 60000 }}
"""

# The same policy with the `embed` alias deleted, which is what a deployment that
# has not defined one looks like - and what the EU policy in this directory's
# `conftest.py` already is.
NO_EMBED_ALIAS_POLICY = REFUSAL_POLICY.replace(
    f"""  {EMBED_ALIAS}:
    candidates:
      - {{ provider: {EMBED_US_PROVIDER}, model: model-embed-us }}
""",
    "",
)


def written(tmp_path_factory: pytest.TempPathFactory, name: str, policy: str) -> Path:
    """One policy document on disk, written once."""
    path = tmp_path_factory.mktemp(name) / "routing.yaml"
    path.write_text(policy, encoding="utf-8")
    return path


def policy_client(path: Path, adapters: Mapping[str, Adapter]) -> TestClient:
    """An app built on the policy at `path` with the given stubs."""
    return build_chat_client(Settings(api_keys=API_KEYS, routing_policy_path=path), adapters)


@pytest.fixture(scope="module")
def refusal_stubs() -> tuple[ChatStub, EmbedStub]:
    """The chat stub and the embedding stub of the residency-refusal app."""
    return ChatStub(name=CHAT_EU_PROVIDER), EmbedStub(name=EMBED_US_PROVIDER)


@pytest.fixture(scope="module")
def refusal_client(
    tmp_path_factory: pytest.TempPathFactory, refusal_stubs: tuple[ChatStub, EmbedStub]
) -> Iterator[TestClient]:
    """An app whose `embed` alias is a 451 for this tenant, with the adapter registered."""
    chat, embed = refusal_stubs
    path = written(tmp_path_factory, "refusal-policy", REFUSAL_POLICY)
    with policy_client(path, {chat.name: chat, embed.name: embed}) as client:
        yield client


@pytest.fixture(scope="module")
def no_embed_alias_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """An app on a policy that defines no `embed` alias at all."""
    chat = ChatStub(name=CHAT_EU_PROVIDER)
    embed = EmbedStub(name=EMBED_US_PROVIDER)
    path = written(tmp_path_factory, "no-embed-alias-policy", NO_EMBED_ALIAS_POLICY)
    with policy_client(path, {chat.name: chat, embed.name: embed}) as client:
        yield client


# --------------------------------------------------------------------------
# reading a response
# --------------------------------------------------------------------------


def send(
    client: TestClient,
    text: str,
    *,
    model: str = ALIAS,
    tenant: str = TENANT,
    data_class: str = DATA_CLASS,
    **overrides: object,
) -> Response:
    """POST one completion carrying `text` as its only user message.

    No `temperature`, so `is_cacheable` refuses every body this helper builds and
    the exact tier cannot decide any outcome here (13a decision G). The cases
    that want the exact tier state `temperature=0` themselves.
    """
    return post_chat(
        client,
        chat_body(model=model, messages=[{"role": "user", "content": text}], **overrides),
        key=TEST_KEYS[tenant],
        tenant=tenant,
        data_class=data_class,
    )


def content_of(response: Response) -> str:
    """The assistant text of a completion, with a 200 asserted."""
    assert response.status_code == 200, response.text
    content: str = response.json()["choices"][0]["message"]["content"]
    return content


def cache_header_of(response: Response) -> str:
    """The `X-Gateway-Cache` value the answer carried."""
    value: str = response.headers["x-gateway-cache"]
    return value


def rows_of(client: TestClient) -> tuple[ResidencyLogRow, ...]:
    """Every residency row this app's writer holds, with its type narrowed."""
    writer = app_of(client).state.residency_log
    assert isinstance(writer, InMemoryResidencyLogWriter)
    return writer.rows


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, written out here rather than imported from the store.

    Importing `opsconsole.cache.vectors._cosine` would make the fixture check
    below agree with the implementation by construction, which is the one thing
    it must not do.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


# --------------------------------------------------------------------------
# the fixture the whole file rests on
# --------------------------------------------------------------------------


def test_the_stub_vectors_sit_where_every_case_here_assumes() -> None:
    """Catches the fixture drifting out from under every other case in this file: if
    `NEAR_B` fell below the threshold the hit cases would fail while the route was
    correct, and if `FAR` rose above it the miss case would fail while the route was
    correct - both are hours spent on the wrong module. The threshold is imported, so a
    change to the published number is answered here and not by a stale literal."""
    assert cosine(NEAR_A, NEAR_B) >= SEMANTIC_THRESHOLD
    assert cosine(NEAR_A, FAR) < SEMANTIC_THRESHOLD
    assert cosine(NEAR_A, UNKNOWN_PROBE_VECTOR) < SEMANTIC_THRESHOLD


def test_the_semantic_words_are_the_published_ones() -> None:
    """a project rule for the four names this plan introduces. Catches the header word
    and the metric label drifting apart - a console summing `cache="semantic"` while the
    counter is filed under `kind="vector"` reports a hit rate of zero forever - and
    catches the embedding alias becoming a setting (decision E), which is how a
    deployment ends up querying with vectors no stored entry shares."""
    assert CACHE_STATUS_SEMANTIC == "semantic"
    assert SEMANTIC_CACHE_KIND == "semantic"
    assert SEMANTIC_SPAN == "semantic"
    assert EMBED_ALIAS == "embed"
    assert CACHE_STATUS_SEMANTIC != CACHE_STATUS_HIT
    assert CACHE_STATUS_SEMANTIC != CACHE_STATUS_MISS


# --------------------------------------------------------------------------
# the hit
# --------------------------------------------------------------------------


def test_a_differently_worded_second_request_is_served_from_the_semantic_tier(
    semantic_client: TestClient, chat_stub: ChatStub, embed_stub: EmbedStub
) -> None:
    """The plan's first criterion. Catches a tier that stores nothing, one that stores
    under filters it cannot query back (a swapped `system_hash`/`tools_hash` pair, an
    alias taken from the resolved candidate on the way in and from the body on the way
    out), and one that looks up AFTER the provider call. `chat_stub.call_count == 1`
    across two requests is the only assertion separating those from a working tier: the
    header alone would pass on a route that dialled the provider and labelled the answer
    `semantic`. Two embeddings, because each request is looked up on its own vector."""
    first = send(semantic_client, QUESTION)
    second = send(semantic_client, SAME_QUESTION_REWORDED)

    assert content_of(first) == CHAT_ANSWER
    assert content_of(second) == CHAT_ANSWER
    assert cache_header_of(first) == CACHE_STATUS_MISS
    assert cache_header_of(second) == CACHE_STATUS_SEMANTIC
    assert chat_stub.call_count == 1
    assert embed_stub.call_count == 2


def test_a_probe_below_the_threshold_is_a_miss_and_reaches_the_provider(
    semantic_client: TestClient, chat_stub: ChatStub, embed_stub: EmbedStub
) -> None:
    """The row that stops the case above passing on a tier that serves the nearest entry
    whatever its distance - the failure mode where one caller's answer is handed to
    another caller's question, which is the only way this tier can be WRONG rather than
    merely cold. Same arrangement, same filters, one vector moved below the threshold, so
    only the comparison can decide it."""
    first = send(semantic_client, QUESTION)
    second = send(semantic_client, DIFFERENT_QUESTION)

    assert cache_header_of(first) == CACHE_STATUS_MISS
    assert cache_header_of(second) == CACHE_STATUS_MISS
    assert chat_stub.call_count == 2
    assert embed_stub.call_count == 2


def test_a_hit_carries_its_own_completion_id(
    semantic_client: TestClient, chat_stub: ChatStub
) -> None:
    """Catches a stored WIRE body replayed verbatim: the replay would carry the first
    request's `chatcmpl-` id, which is the id a trace and the ledger key their rows by,
    so two requests would merge into one row. The equal content is what stops this passing
    on a route that answers something else entirely."""
    first = send(semantic_client, QUESTION)
    second = send(semantic_client, SAME_QUESTION_REWORDED)

    assert cache_header_of(second) == CACHE_STATUS_SEMANTIC
    assert content_of(second) == content_of(first)
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["model"] == ALIAS
    assert chat_stub.call_count == 1


# --------------------------------------------------------------------------
# who may embed at all
# --------------------------------------------------------------------------

ELIGIBILITY_ROWS: list[tuple[str, dict[str, Any], int]] = [
    ("an eligible request", {}, 1),
    ("the alias flag off", {"model": OFF_ALIAS}, 0),
    ("a streamed request", {"stream": True}, 0),
    ("live tools", {"tools": list(TOOLS), "tool_choice": "auto"}, 0),
    ("tools offered with tool_choice none", {"tools": list(TOOLS), "tool_choice": "none"}, 1),
]


@pytest.mark.parametrize(
    ("case", "overrides", "embeddings"),
    ELIGIBILITY_ROWS,
    ids=[row[0] for row in ELIGIBILITY_ROWS],
)
def test_only_an_eligible_request_embeds_anything(
    case: str,
    overrides: dict[str, Any],
    embeddings: int,
    semantic_client: TestClient,
    embed_stub: EmbedStub,
) -> None:
    """The plan's second criterion, asserted on the embedding stub's call count and not on
    the header, because a header cannot tell "embedded and found nothing" from "never
    embedded" - and an embedding is a provider call that is billed and can leak. Catches
    an eligibility check placed AFTER the embedding, which is the natural place to write
    it and costs a request per stream and per tool call forever.

    The first and last rows are what make the zeroes mean something: they leave the stub
    reachable in the same app, so a row asserting zero cannot pass because nothing was
    ever wired up. The `tool_choice: none` row is also the one that stops "tools present,
    never embed" from satisfying the live-tools row."""
    response = send(semantic_client, QUESTION, **overrides)

    assert response.status_code == 200, response.text
    assert cache_header_of(response) != CACHE_STATUS_SEMANTIC
    assert embed_stub.call_count == embeddings


def test_an_exact_hit_returns_before_anything_is_embedded(
    semantic_client: TestClient, chat_stub: ChatStub, embed_stub: EmbedStub
) -> None:
    """The plan's third criterion, and the ordering claim the two tiers make together
    (13c decision A against 13a decision A). Catches a semantic lookup placed before the
    exact one, or beside it: an exact hit would then pay for an embedding on every repeat
    of the cheapest possible request, which is the traffic the exact tier exists to make
    free. `temperature: 0` is stated so the exact tier is eligible; the second request is
    byte-identical, so only the exact tier can serve it, and the header must still be the
    exact tier's word."""
    body = chat_body(model=ALIAS, temperature=0, messages=[{"role": "user", "content": QUESTION}])

    first = post_chat(semantic_client, body)
    assert cache_header_of(first) == CACHE_STATUS_MISS
    assert embed_stub.call_count == 1

    second = post_chat(semantic_client, body)

    assert cache_header_of(second) == CACHE_STATUS_HIT
    assert content_of(second) == CHAT_ANSWER
    assert chat_stub.call_count == 1
    assert embed_stub.call_count == 1, "the exact hit paid for an embedding"


# --------------------------------------------------------------------------
# what is embedded (decision A)
# --------------------------------------------------------------------------


def test_the_embedded_probe_is_the_redacted_text_and_not_the_raw_one(
    pii_client: TestClient, pii_stub: ChatAndEmbedStub
) -> None:
    """Decision A, read off the stub's recorded `EmbeddingRequest` rather than off the
    answer - a route that redacted only what it returned would look identical from the
    body. Catches the lookup placed beside the exact one, ahead of `_redacted`: the raw
    address would then be sent to an embedding provider on every request of a `pii`-class
    tenant, and nothing in the response, the headers or the metrics would show it. The
    placeholder assertion is what stops this passing on a route that embeds nothing at
    all, which would also contain no email address."""
    response = send(pii_client, TEXT_WITH_PII, tenant=TENANT, data_class=PII_CLASS)

    assert response.status_code == 200, response.text
    assert pii_stub.embedder.call_count == 1
    assert pii_stub.embedder.inputs == (REDACTED_PII,)
    assert EMAIL not in "".join(pii_stub.embedder.inputs)


def test_the_probe_is_the_final_user_message_of_the_outbound_body(
    semantic_client: TestClient, embed_stub: EmbedStub
) -> None:
    """Catches a probe built from the whole conversation, from the first user turn or from
    the assistant's last answer: a caller replaying a long history would then ask a
    different question from one asking it cold, and the tier would never hit for exactly
    the traffic #23 describes. The earlier turns carry text that would change the probe if
    it included them, so an implementation that concatenated cannot pass."""
    response = post_chat(
        semantic_client,
        chat_body(
            model=ALIAS,
            messages=[
                {"role": "user", "content": DIFFERENT_QUESTION},
                {"role": "assistant", "content": "Ask me something else."},
                {"role": "user", "content": QUESTION},
            ],
        ),
    )

    assert response.status_code == 200, response.text
    assert embed_stub.inputs == (QUESTION,)


# --------------------------------------------------------------------------
# the metrics (#24, the design)
# --------------------------------------------------------------------------


def request_labels(*, cache: str) -> dict[str, str]:
    """The six labels `gateway_requests_total` carries for a served request here."""
    return {
        "tenant": TENANT,
        "alias": ALIAS,
        "provider": PROVIDER,
        "region": REGION,
        "status": "200",
        "cache": cache,
    }


def test_a_semantic_miss_moves_no_hit_counter_and_is_labelled_miss(
    semantic_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a tier that counts a hit whenever it LOOKED - a counter that moves on both
    outcomes is a panel reporting a 100% hit rate - and catches a `cache` label set to
    `semantic` by the mere fact that the tier ran rather than by its answer. This is the
    half that makes the hit case below mean something."""
    response = send(semantic_client, QUESTION)

    assert cache_header_of(response) == CACHE_STATUS_MISS
    assert metrics.delta(REQUESTS, **request_labels(cache=CACHE_STATUS_MISS)) == 1
    assert metrics.total(CACHE_HITS) == 0


def test_a_semantic_hit_counts_its_own_kind_and_labels_the_request_semantic(
    semantic_client: TestClient, metrics_registry: CollectorRegistry
) -> None:
    """Catches a semantic hit counted under `kind="exact"`, which would make the two tiers
    indistinguishable in the one query an operator asks (which tier is paying for itself),
    and catches a hit that never reaches `gateway_cache_hits_total` at all. The `cache`
    label is the one #24's savings panel sums, so a hit left labelled `miss` is money
    reported as spent. The delta is taken AFTER the warming request, so only the second
    request's emissions are inside the measurement."""
    assert send(semantic_client, QUESTION).status_code == 200

    metrics = MetricDelta(metrics_registry)
    hit = send(semantic_client, SAME_QUESTION_REWORDED)

    assert cache_header_of(hit) == CACHE_STATUS_SEMANTIC
    assert metrics.delta(CACHE_HITS, tenant=TENANT, kind=SEMANTIC_CACHE_KIND) == 1
    assert metrics.total(CACHE_HITS) == 1
    assert metrics.delta(REQUESTS, **request_labels(cache=CACHE_STATUS_SEMANTIC)) == 1


def test_a_semantic_hit_books_the_embedding_tokens_and_no_chat_tokens(
    semantic_client: TestClient, metrics_registry: CollectorRegistry
) -> None:
    """The plan's fifth criterion. Catches a hit that re-books the tokens of the ORIGINAL
    completion: the tier would inflate cost reporting by exactly the traffic it was added
    to save, which is worse than not caching at all because it is invisible. The exact
    total is asserted rather than a band: what a hit costs is one embedding and nothing
    else, so `total` catches a second booking under any label set at all."""
    assert send(semantic_client, QUESTION).status_code == 200

    metrics = MetricDelta(metrics_registry)
    hit = send(semantic_client, SAME_QUESTION_REWORDED)

    assert cache_header_of(hit) == CACHE_STATUS_SEMANTIC
    assert (
        metrics.delta(
            TOKENS,
            tenant=TENANT,
            provider=EMBED_PROVIDER,
            model=EMBED_MODEL,
            direction=DIRECTION_IN,
        )
        == EMBED_PROMPT_TOKENS
    )
    assert metrics.total(TOKENS) == EMBED_PROMPT_TOKENS


def test_the_embedding_tokens_are_booked_under_the_embed_provider_and_model(
    semantic_client: TestClient, metrics: MetricDelta
) -> None:
    """Decision G's booking half. Catches the embedding's tokens booked under the CHAT
    candidate - `anthropic`/`claude-haiku-4-5` would then be billed for vectors it never
    computed, and an earlier change's ledger could never be reconciled with the provider's invoice -
    and catches them booked under the ALIAS rather than the resolved model, which is the
    grain the ledger prices at. The `out` assertion is the one that catches a copied chat
    call site: an embedding has no completion, and a zero `out` sample is an invented
    number a dashboard would sum."""
    response = send(semantic_client, QUESTION)
    assert response.status_code == 200, response.text
    chat_model = response.headers["x-gateway-model"]

    assert EMBED_MODEL not in {chat_model, EMBED_ALIAS}
    assert (
        metrics.delta(
            TOKENS,
            tenant=TENANT,
            provider=EMBED_PROVIDER,
            model=EMBED_MODEL,
            direction=DIRECTION_IN,
        )
        == EMBED_PROMPT_TOKENS
    )
    assert (
        metrics.delta(
            TOKENS,
            tenant=TENANT,
            provider=EMBED_PROVIDER,
            model=EMBED_MODEL,
            direction=DIRECTION_OUT,
        )
        == 0
    )
    assert (
        metrics.delta(
            TOKENS, tenant=TENANT, provider=EMBED_PROVIDER, model=chat_model, direction=DIRECTION_IN
        )
        == 0
    )


# --------------------------------------------------------------------------
# the residency log (decision G)
# --------------------------------------------------------------------------


def test_the_embedding_writes_no_second_residency_row(
    semantic_client: TestClient, embed_stub: EmbedStub
) -> None:
    """Decision G's other half, and the reason the tier does not call `record_served`.
    Catches an embedding routed through the chat route's own serving path: the request
    would write a `local`/`self` row beside its `anthropic`/`us` one, so "where did this
    tenant's data go" - the question the design's log exists to answer - would be answered
    twice for one client request and the counts would never match the requests. The stub's
    call count is what stops this passing on a route that embedded nothing at all."""
    response = send(semantic_client, QUESTION)

    assert response.status_code == 200, response.text
    assert embed_stub.call_count == 1
    assert len(rows_of(semantic_client)) == 1
    assert rows_of(semantic_client)[0].provider == PROVIDER
    assert rows_of(semantic_client)[0].region == REGION


def test_the_embedding_does_not_relabel_the_requests_candidate(
    semantic_client: TestClient, metrics: MetricDelta
) -> None:
    """The scope-label half of decision G. Catches an embedding resolution that runs
    through `label_candidate`: every eligible chat request would then be attributed to the
    embedding provider on the counter and on the span, so the per-alias latency and error
    panels of the design would show `local` serving traffic it never served. Asserted as
    the ABSENCE of a series as well as the presence of the right one, because a relabel
    moves the count rather than adding to it."""
    response = send(semantic_client, QUESTION)

    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == PROVIDER
    assert response.headers["x-gateway-region"] == REGION
    assert metrics.delta(REQUESTS, **request_labels(cache=CACHE_STATUS_MISS)) == 1
    assert metrics.total(REQUESTS) == 1


# --------------------------------------------------------------------------
# the stage span (decision D)
# --------------------------------------------------------------------------


def test_the_semantic_span_is_emitted_on_an_eligible_miss_and_on_an_eligible_hit(
    semantic_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Decision D. Catches the embedding folded into the `cache` span, where a slow
    embedder reads as a slow cache and pages the wrong team, and catches a span emitted on
    hits alone, which would make every trace carrying one a hit by construction and hide
    the cost of the lookups that found nothing. The two buffers are read separately, so a
    span left by the first request cannot stand in for the second."""
    miss = send(semantic_client, QUESTION)
    assert cache_header_of(miss) == CACHE_STATUS_MISS
    miss_names = {span.name for span in spans.get_finished_spans()}

    spans.clear()
    hit = send(semantic_client, SAME_QUESTION_REWORDED)
    assert cache_header_of(hit) == CACHE_STATUS_SEMANTIC
    hit_names = {span.name for span in spans.get_finished_spans()}

    assert SEMANTIC_SPAN in miss_names
    assert SEMANTIC_SPAN in hit_names


INELIGIBLE_SPAN_ROWS: list[tuple[str, dict[str, Any]]] = [
    ("the alias flag off", {"model": OFF_ALIAS}),
    ("a streamed request", {"stream": True}),
    ("live tools", {"tools": list(TOOLS), "tool_choice": "auto"}),
]


@pytest.mark.parametrize(
    ("case", "overrides"), INELIGIBLE_SPAN_ROWS, ids=[row[0] for row in INELIGIBLE_SPAN_ROWS]
)
def test_no_semantic_span_is_emitted_on_an_ineligible_request(
    case: str,
    overrides: dict[str, Any],
    semantic_client: TestClient,
    spans: InMemorySpanExporter,
) -> None:
    """Catches a `with` block placed around the eligibility CHECK rather than inside the
    branch it guards: every chat request would carry a `semantic` span, and an operator
    reading a trace would be told that a stream or a tool call consulted a tier it is
    forbidden to consult. The `cache` span is asserted present in the same buffer, so the
    absence cannot be satisfied by a request that emitted no span at all."""
    response = send(semantic_client, QUESTION, **overrides)
    assert response.status_code == 200, response.text

    names = {span.name for span in spans.get_finished_spans()}

    assert SEMANTIC_SPAN not in names
    assert CACHE_SPAN in names


# --------------------------------------------------------------------------
# every way it can fail (decision B)
# --------------------------------------------------------------------------


def assert_served_by_the_provider(response: Response, *, marker: str | None = None) -> None:
    """A 200 with the chat stub's own answer, labelled a miss, leaking no failure text.

    The shared assertion of every failure row: a cache tier must never change the
    status code, the body or the header of a request a provider would have
    served.
    """
    assert response.status_code == 200, response.text
    assert content_of(response) == CHAT_ANSWER
    assert cache_header_of(response) == CACHE_STATUS_MISS
    if marker is not None:
        assert marker not in response.text


def test_an_embed_alias_refused_by_residency_is_a_miss(
    refusal_client: TestClient, refusal_stubs: tuple[ChatStub, EmbedStub]
) -> None:
    """The 451 row. Catches a seam that lets `ResidencyUnavailable` escape: the tier would
    turn every chat request of an EU tenant into a 451 the moment an operator's `embed`
    alias pointed anywhere outside the tenant's regions - a residency rule about
    EMBEDDINGS refusing traffic that was itself perfectly legal. The adapter IS registered
    for the refused provider, so residency alone excludes it, and its silent call count is
    the proof no vector left the region."""
    chat, embed = refusal_stubs

    response = send(refusal_client, QUESTION, tenant=REFUSAL_TENANT)

    assert_served_by_the_provider(response)
    assert chat.call_count == 1
    assert embed.call_count == 0, "an embedding crossed the residency boundary"


def test_an_embed_alias_with_no_healthy_candidate_is_a_miss(chat_only_client: TestClient) -> None:
    """The 503 row, and the arrangement every other chat suite in this directory runs
    under. Catches a seam that lets `AllProvidersUnavailable` escape: a deployment with no
    embedding provider configured - which is the shipped default, since `local` needs a
    llama.cpp box - would answer 503 to every chat request against `agent-default`, and
    the whole existing suite would go red for a tier nobody asked for."""
    response = send(chat_only_client, QUESTION)

    assert_served_by_the_provider(response)


def test_an_embed_alias_missing_from_the_policy_is_a_miss(
    no_embed_alias_client: TestClient,
) -> None:
    """The row `AliasUnknown` hides in. Catches a seam that catches only the residency and
    availability errors: a policy with no `embed` alias - a legal document, and what this
    directory's own EU policy already is - would 404 every chat request, and the failure
    would name an alias the caller never sent. Decision B is about every failure of the
    resolution, not about the two that were easy to think of."""
    response = send(no_embed_alias_client, QUESTION, tenant=REFUSAL_TENANT)

    assert_served_by_the_provider(response)


def test_an_embedding_adapter_that_raises_is_a_miss(
    semantic_client: TestClient, chat_stub: ChatStub, embed_stub: EmbedStub
) -> None:
    """The provider-failure row. Catches an embedding call left outside a guard: the
    lookup happens before the completion, so an escaping `ProviderError` would turn a
    healthy provider's traffic into 502s whenever the embedding box wobbled - a cache tier
    causing the outage it exists to soften. The recorded call is what says the adapter was
    really reached, and the marker says the provider's own words did not reach the
    caller."""
    embed_stub.error = ProviderError(EMBED_FAILURE_TEXT, retryable=True)

    response = send(semantic_client, QUESTION)

    assert_served_by_the_provider(response, marker=EMBED_FAILURE_TEXT)
    assert embed_stub.call_count == 1
    assert chat_stub.call_count == 1


def test_a_store_that_raises_on_search_is_a_miss(
    semantic_client: TestClient, chat_stub: ChatStub
) -> None:
    """The unreachable-Qdrant row on the READ side. Catches a lookup called outside a
    guard: the search runs before the provider call, so an escaping failure takes out
    every chat request in a deployment that has Qdrant. The store's own counter is what
    stops this passing on a route that stopped calling the store at all, which would leave
    the status, the body and the header identical."""
    store = BrokenSemanticCache(on_search=True, on_put=False)
    app_of(semantic_client).state.semantic_cache = store

    response = send(semantic_client, QUESTION)

    assert_served_by_the_provider(response, marker=STORE_FAILURE_TEXT)
    assert store.searches == 1
    assert chat_stub.call_count == 1


def test_a_store_that_raises_on_put_is_swallowed_after_the_caller_is_served(
    semantic_client: TestClient, chat_stub: ChatStub
) -> None:
    """The write half, and the sharper one: the write happens after a provider has already
    answered and been billed, so an escaping failure loses a request that was fully served
    and paid for. Catches exactly that, and the second request catches a write whose
    failure aborted the store-on-serve step silently - both requests are answered, both
    miss, and the store was really asked to write."""
    store = BrokenSemanticCache(on_search=False, on_put=True)
    app_of(semantic_client).state.semantic_cache = store

    first = send(semantic_client, QUESTION)
    second = send(semantic_client, SAME_QUESTION_REWORDED)

    assert_served_by_the_provider(first, marker=STORE_FAILURE_TEXT)
    assert_served_by_the_provider(second, marker=STORE_FAILURE_TEXT)
    assert store.puts == 2
    assert chat_stub.call_count == 2


# --------------------------------------------------------------------------
# the input cap (decision F, a project rule)
# --------------------------------------------------------------------------

AT_CAP_PROBE = "a" * MAX_EMBEDDING_INPUT_CHARS
OVER_CAP_PROBE = "a" * (MAX_EMBEDDING_INPUT_CHARS + 1)

CAP_ROWS: list[tuple[str, str, int]] = [
    ("exactly the cap", AT_CAP_PROBE, 1),
    ("one character over the cap", OVER_CAP_PROBE, 0),
]


@pytest.mark.parametrize(
    ("case", "probe", "embeddings"), CAP_ROWS, ids=[row[0] for row in CAP_ROWS]
)
def test_the_probe_length_boundary_is_inclusive_and_never_a_422(
    case: str, probe: str, embeddings: int, semantic_client: TestClient, embed_stub: EmbedStub
) -> None:
    """Decision F and a project rule: the probe is a client-supplied string reaching an
    unsized store, so it is pre-checked against that store's own constant, inclusively,
    on both sides. Catches a cap written `>=`, which would refuse the longest legal input
    forever, and a cap missing entirely, which sends an over-length text to an embedder
    that will reject it on every request of that shape for as long as the caller retries.
    The 200 is the other half: the CHAT request is legal, so a probe the embedder cannot
    take is a miss and never the 422 `/v1/embeddings` would answer."""
    response = send(semantic_client, probe)

    assert response.status_code == 200, response.text
    assert cache_header_of(response) == CACHE_STATUS_MISS
    assert embed_stub.call_count == embeddings
    assert all(len(text) <= MAX_EMBEDDING_INPUT_CHARS for text in embed_stub.inputs)


def test_a_probe_at_the_cap_is_embedded_whole_and_not_truncated(
    semantic_client: TestClient, embed_stub: EmbedStub
) -> None:
    """The other way decision F can go wrong. Catches a route that truncates to the cap
    instead of refusing: two different long prompts sharing a prefix would become one
    question, and the tier would confidently serve one caller's answer to the other -
    the same failure the threshold exists to prevent, arriving through the door nobody
    watches. Asserted on the recorded input's exact length, because a truncated probe is
    still a perfectly successful request."""
    response = send(semantic_client, AT_CAP_PROBE)

    assert response.status_code == 200, response.text
    assert embed_stub.inputs == (AT_CAP_PROBE,)


# --------------------------------------------------------------------------
# wiring: which store the factory builds, and how the client is bounded
# --------------------------------------------------------------------------

UNCONNECTED_QDRANT_URL = "http://127.0.0.1:1"
"""A Qdrant nothing listens on. `httpx.Client` connects lazily, so building the
cache costs no connection and this URL is never dialled by the two cases below."""


def wiring_settings(*, qdrant_url: str | None) -> Settings:
    """The test key store and the shipped policy, with the Qdrant URL stated either way.

    Stated rather than left to default, for the reason `test_exact_cache.py`
    states it of Redis: `Settings` reads `OPSCONSOLE_QDRANT_URL` from the process
    environment, and the unconfigured row would otherwise assert the opposite of
    what it says on a machine with the stack up.
    """
    return Settings(
        api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, qdrant_url=qdrant_url
    )


def test_create_app_keeps_the_in_process_semantic_cache_without_a_qdrant_url() -> None:
    """Catches a factory that requires Qdrant to start: the whole existing suite, `make
    verify` and a bare `uvicorn` run configure no `qdrant_url`, so a Qdrant-only factory
    turns every one of them into a connection error at process start rather than a miss."""
    app = create_app(wiring_settings(qdrant_url=None), NO_PROVIDER_KEYS)

    assert isinstance(app.state.semantic_cache, InProcessSemanticCache)
    close_adapters(app.state.adapters)


def test_create_app_uses_the_qdrant_semantic_cache_when_a_qdrant_url_is_configured() -> None:
    """Catches a factory that keeps vectors in process memory while Qdrant is configured:
    two gateway replicas would each fill their own store, so a question answered on one
    pod would never hit on the other and the tier would report a hit rate that is a
    function of how the load balancer felt - invisible to every test with one replica."""
    app = create_app(wiring_settings(qdrant_url=UNCONNECTED_QDRANT_URL), NO_PROVIDER_KEYS)

    assert isinstance(app.state.semantic_cache, QdrantSemanticCache)
    close_adapters(app.state.adapters)


def test_the_semantic_cache_client_bounds_every_phase_of_the_call() -> None:
    """Catches an `httpx.Client()` built with the library default (5s connect and no read
    bound at all). This client sits IN FRONT of a chat request, not behind a readiness
    probe, so an unbounded read is the caller's whole request hanging on a cache that has
    nothing to say. The base URL is the other half: `QdrantSemanticCache` sends relative
    paths, so a client without one raises on every search instead of missing."""
    with build_qdrant_data_client(UNCONNECTED_QDRANT_URL) as client:
        assert client.timeout == httpx.Timeout(PROBE_TIMEOUT_SECONDS)
        assert client.timeout.connect == PROBE_TIMEOUT_SECONDS
        assert client.timeout.read == PROBE_TIMEOUT_SECONDS
        assert client.timeout.write == PROBE_TIMEOUT_SECONDS
        assert client.timeout.pool == PROBE_TIMEOUT_SECONDS
        assert str(client.base_url) == UNCONNECTED_QDRANT_URL
