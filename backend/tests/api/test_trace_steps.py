"""What each gateway step shows in Langfuse: the decision it took, as one plain line.

Everything goes through `TestClient` against an app `create_app` built, and the spans
come out of the shared in-memory exporter (`spans` in `conftest.py`). Step notes are the
Langfuse input and output attributes, read with the keys `app.core.tracing` publishes.

One app serves the module, built once. Per-test isolation is a fresh pair of caches, the
stubs reset and the embedder's vector put back, never a new app.

The semantic rows lean on a stub embedder whose vector the test sets. The cosine between
two vectors is then a fact of the test, chosen to be a long decimal (0.957826...) so the
"rounded down" rule is visible: rounding to nearest would read 0.958.

Things not specified are left unpinned here: the auth Input of a refused request,
and the singular form of a count of one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.api.routes.redact import REDACT
from app.cache import InProcessResponseCache, InProcessSemanticCache
from app.core.config import Settings
from app.core.errors import ProviderError
from app.core.tracing import INPUT_NOTE_KEY, OUTPUT_NOTE_KEY
from app.providers.base import (
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
    CHAT,
    EMBED_PROVIDER,
    TEST_KEYS,
    EmbeddingStubAdapter,
    StubAdapter,
    app_of,
    auth_headers,
    build_chat_client,
    chat_body,
    embedding_stub_response,
    embeddings_body,
    evals_switched_off,
    post_chat,
    post_embeddings,
)

ALIAS = "agent-default"
EMBED_ALIAS = "embed"
TENANT = "evals"
PLAIN_CLASS = "internal"
REDACT_CLASS = "pii"

EMAIL = "alice@example.com"
PHONE = "+4930123456789"
TEXT_WITH_PII = f"write to {EMAIL} or call {PHONE} about the capital"
CLEAN_TEXT = "which city is the capital of Germany"

AUTH_SPAN = "auth"
ROUTE_SPAN = "route"
CACHE_SPAN = "cache"
SEMANTIC_SPAN = "semantic"
REDACT_SPAN = "redact"
STAGE_SPANS = (AUTH_SPAN, ROUTE_SPAN, CACHE_SPAN, REDACT_SPAN, SEMANTIC_SPAN)

NOT_RECORDED = "not recorded"
NEEDED = "needed 0.97"

# Two probe vectors whose cosine is 1 / sqrt(1.09) = 0.957826..., under the 0.97 threshold.
PROBE_A: tuple[float, ...] = (1.0, 0.0)
PROBE_NEAR_MISS: tuple[float, ...] = (1.0, 0.3)
NEAR_MISS_SCORE = "0.957"


class VectorEmbedder:
    """An embedding adapter that answers every text with the vector the test last set."""

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.vector: tuple[float, ...] = PROBE_A
        self.error: Exception | None = None

    def reset(self) -> None:
        self.vector = PROBE_A
        self.error = None

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if self.error is not None:
            raise self.error
        return EmbeddingResponse(
            model="stub-embedder",
            embeddings=tuple(
                Embedding(index=index, vector=self.vector) for index, _ in enumerate(request.input)
            ),
            usage=EmbeddingUsage(prompt_tokens=3, total_tokens=3),
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


@pytest.fixture(scope="module")
def embedder() -> VectorEmbedder:
    """The embedder behind the `embed` alias, with the vector the test sets."""
    return VectorEmbedder(name=EMBED_PROVIDER)


@pytest.fixture(scope="module")
def step_client(settings: Settings, embedder: VectorEmbedder) -> Iterator[TestClient]:
    """One app: both chat providers and the embedder, so a fallback exists and the
    semantic tier is live."""
    adapters: dict[str, Adapter] = {
        "anthropic": StubAdapter(name="anthropic"),
        "openai": StubAdapter(name="openai"),
        EMBED_PROVIDER: embedder,
    }
    with build_chat_client(settings, adapters) as client:
        yield client


@pytest.fixture(scope="module")
def embeddings_step_client(settings: Settings) -> Iterator[TestClient]:
    """One app whose only adapter embeds, answering two vectors (the tests send two texts)."""
    stub = EmbeddingStubAdapter(name=EMBED_PROVIDER, response=embedding_stub_response(2))
    with build_chat_client(settings, {stub.name: stub}) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh(step_client: TestClient, embedder: VectorEmbedder) -> Iterator[None]:
    """Empty caches and a reset embedder, never a new app."""
    embedder.reset()
    state = app_of(step_client).state
    state.response_cache = InProcessResponseCache()
    state.semantic_cache = InProcessSemanticCache()
    yield


# --- reading notes ------------------------------------------------------------


def finished(spans: InMemorySpanExporter) -> tuple[ReadableSpan, ...]:
    """Every span the request finished."""
    recorded = tuple(spans.get_finished_spans())
    assert recorded, "the request emitted no span at all"
    return recorded


def step(recorded: Sequence[ReadableSpan], name: str) -> ReadableSpan:
    """The one span with this name."""
    matches = [span for span in recorded if span.name == name]
    assert len(matches) == 1, f"expected one {name!r} span, got {[s.name for s in recorded]}"
    return matches[0]


def notes(recorded: Sequence[ReadableSpan], name: str) -> tuple[str, str]:
    """The Input and Output notes of one step, as plain lines."""
    attributes = dict(step(recorded, name).attributes or {})
    raw_in, raw_out = attributes.get(INPUT_NOTE_KEY), attributes.get(OUTPUT_NOTE_KEY)
    assert isinstance(raw_in, str), f"{name} has no input note, got {raw_in!r}"
    assert isinstance(raw_out, str), f"{name} has no output note, got {raw_out!r}"
    return raw_in, raw_out


def everything_written(recorded: Sequence[ReadableSpan]) -> str:
    """Every name, attribute value and event on every span, joined."""
    parts: list[str] = []
    for span in recorded:
        parts.append(span.name)
        parts.append(span.status.description or "")
        parts.extend(str(value) for value in (span.attributes or {}).values())
        for event in span.events:
            parts.append(event.name)
            parts.extend(str(value) for value in (event.attributes or {}).values())
    return " ".join(parts)


def send(
    client: TestClient, text: str, *, data_class: str = PLAIN_CLASS, **more: object
) -> Response:
    """One chat completion carrying `text`, as the `evals` tenant."""
    return post_chat(
        client,
        chat_body(model=ALIAS, messages=[{"role": "user", "content": text}], **more),
        data_class=data_class,
    )


# --- auth ---------------------------------------------------------------------


def test_auth_says_the_tenant_accepted(
    step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an auth step left on its starting lines, or one that writes the tenant from
    the X-Tenant header instead of the one the key proved."""
    assert send(step_client, CLEAN_TEXT).status_code == 200

    assert notes(finished(spans), AUTH_SPAN) == ("API key and X-Tenant", "accepted: tenant evals")


def test_a_refused_auth_says_the_code(step_client: TestClient, spans: InMemorySpanExporter) -> None:
    """Catches a refused auth step that still reads `accepted`, shows the message text
    instead of the code, or writes the tenant the caller only claimed (a prober could plant
    text that way)."""
    claimed = "zz-claimed-tenant"
    response = step_client.post(
        CHAT, json=chat_body(), headers=auth_headers(key=None, tenant=claimed)
    )
    assert response.status_code == 401

    recorded = finished(spans)

    assert notes(recorded, AUTH_SPAN)[1] == "refused: UNAUTHORIZED"
    assert claimed not in everything_written(recorded)


# --- route --------------------------------------------------------------------


def test_route_says_alias_pick_and_fallback(
    step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a route step that names the alias only, swaps provider and model, drops the
    fallback, or writes the data class from the wrong place."""
    assert send(step_client, CLEAN_TEXT).status_code == 200

    assert notes(finished(spans), ROUTE_SPAN) == (
        f"alias {ALIAS}, data class {PLAIN_CLASS}",
        "picked anthropic claude-haiku-5-5; fallback openai gpt-5-mini",
    )


def test_an_unknown_alias_leaves_no_caller_text(
    step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a route Input written before the alias resolved: the caller's own text
    (here an alias that does not exist) would land on the trace."""
    unknown = "zz-unknown-alias"
    response = post_chat(step_client, chat_body(model=unknown), data_class=PLAIN_CLASS)
    assert response.status_code == 404

    recorded = finished(spans)

    assert notes(recorded, ROUTE_SPAN) == (NOT_RECORDED, "refused: ALIAS_UNKNOWN")
    assert unknown not in everything_written(recorded)


# --- cache --------------------------------------------------------------------


def test_cache_says_miss_then_hit(step_client: TestClient, spans: InMemorySpanExporter) -> None:
    """Catches a cache step that says `miss` on both runs (never looks at the result), or
    `hit` on both, and one that lost its Input line."""
    first = send(step_client, CLEAN_TEXT, temperature=0)
    assert first.status_code == 200
    miss = notes(finished(spans), CACHE_SPAN)

    spans.clear()
    second = send(step_client, CLEAN_TEXT, temperature=0)
    assert second.status_code == 200
    hit = notes(finished(spans), CACHE_SPAN)

    assert miss == (f"same request answered before, alias {ALIAS}", "miss")
    assert hit == (f"same request answered before, alias {ALIAS}", "hit")


# --- redact -------------------------------------------------------------------


def test_redact_says_counts_only(
    step_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches a redact line that carries a value or a placeholder (personal data on a
    trace), counts in the wrong order, or a count of kinds that were not found. Evals are
    switched off so the conversation is not on the trace and the only place a value could
    appear is a step note."""
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": TEXT_WITH_PII},
        {"role": "user", "content": "thanks"},
    ]
    with evals_switched_off(tmp_path, step_client):
        response = post_chat(
            step_client, chat_body(model=ALIAS, messages=messages), data_class=REDACT_CLASS
        )
    assert response.status_code == 200, response.text

    recorded = finished(spans)

    assert notes(recorded, REDACT_SPAN) == ("3 messages", "masked: 1 email, 1 phone")
    written = everything_written(recorded)
    for forbidden in (EMAIL, PHONE, "[EMAIL_", "<EMAIL_", "[PHONE_", "<PHONE_"):
        assert forbidden not in written, forbidden


def test_redact_with_nothing_found_says_nothing_masked(
    step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an empty `masked: ` line, or a step left on its starting lines, when the
    class redacts but the text holds nothing to mask."""
    assert send(step_client, CLEAN_TEXT, data_class=REDACT_CLASS).status_code == 200

    assert notes(finished(spans), REDACT_SPAN)[1] == "nothing masked"


# --- semantic -----------------------------------------------------------------


def test_semantic_miss_says_best_match_and_needed(
    step_client: TestClient, spans: InMemorySpanExporter, embedder: VectorEmbedder
) -> None:
    """Catches a semantic miss that hides how close it was (no best match), prints it
    rounded to nearest (0.958 reads as nearer the threshold than it was), or shows the
    score of the wrong entry."""
    assert send(step_client, CLEAN_TEXT).status_code == 200  # stores an entry at PROBE_A
    spans.clear()
    embedder.vector = PROBE_NEAR_MISS

    assert send(step_client, "another question").status_code == 200

    assert notes(finished(spans), SEMANTIC_SPAN) == (
        f"nearest stored answer, alias {ALIAS}",
        f"miss (best match {NEAR_MISS_SCORE}, {NEEDED})",
    )


def test_semantic_hit_says_score(step_client: TestClient, spans: InMemorySpanExporter) -> None:
    """Catches a semantic hit that prints no score, or the threshold in its place."""
    assert send(step_client, CLEAN_TEXT).status_code == 200
    spans.clear()

    assert send(step_client, CLEAN_TEXT).status_code == 200

    assert notes(finished(spans), SEMANTIC_SPAN)[1] == f"hit (score 1.000, {NEEDED})"


def test_semantic_with_nothing_stored_says_so(
    step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an empty store printed as `best match 0.000`: it would read as a far miss
    instead of 'nothing to compare'."""
    assert send(step_client, CLEAN_TEXT).status_code == 200

    assert notes(finished(spans), SEMANTIC_SPAN)[1] == (
        f"miss (no stored answer to compare, {NEEDED})"
    )


def test_semantic_says_when_the_question_could_not_be_embedded(
    step_client: TestClient, spans: InMemorySpanExporter, embedder: VectorEmbedder
) -> None:
    """Catches a failed embedding shown as 'no stored answer' (the store was never asked),
    and the provider's own error text reaching the trace."""
    embedder.error = ProviderError("local: req_zqx_107 quota exhausted", retryable=False)

    assert send(step_client, CLEAN_TEXT).status_code == 200

    recorded = finished(spans)

    assert notes(recorded, SEMANTIC_SPAN)[1] == "miss (the question could not be embedded)"
    assert "req_zqx_107" not in everything_written(recorded)


# --- no stage says "not sent" -------------------------------------------------


def test_no_stage_step_says_not_sent(
    step_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches any stage still on the old note. The request runs all five stages (a
    redacting class and an alias with the semantic tier), so none can be skipped."""
    with evals_switched_off(tmp_path, step_client):
        assert send(step_client, TEXT_WITH_PII, data_class=REDACT_CLASS).status_code == 200

    recorded = finished(spans)

    for name in STAGE_SPANS:
        for note in notes(recorded, name):
            assert note != "", name
            assert not note.startswith("not sent"), (name, note)
            assert note not in (NOT_RECORDED, "no decision recorded"), (name, note)


# --- embeddings and /v1/redact ------------------------------------------------


def test_embeddings_and_redact_route_steps_use_the_same_notes(
    embeddings_step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an embeddings route whose route and redact steps keep the old note, count
    messages instead of texts, or leak a value."""
    response = post_embeddings(
        embeddings_step_client,
        embeddings_body(model=EMBED_ALIAS, input=[TEXT_WITH_PII, CLEAN_TEXT]),
        data_class=REDACT_CLASS,
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)

    assert notes(recorded, ROUTE_SPAN) == (
        f"alias {EMBED_ALIAS}, data class {REDACT_CLASS}",
        "picked local_embed bge-small-en-v1.5; fallback none",
    )
    assert notes(recorded, REDACT_SPAN) == ("2 texts", "masked: 1 email, 1 phone")
    assert EMAIL not in everything_written(recorded)


def test_the_redact_endpoint_step_says_counts_only(
    step_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a `/v1/redact` step left on the old note, or one that shows the text it
    masked. The singular form of a count of one is not pinned, so either spelling passes."""
    response = step_client.post(
        REDACT,
        json={"text": TEXT_WITH_PII},
        headers=auth_headers(key=TEST_KEYS[TENANT], tenant=TENANT, data_class=PLAIN_CLASS),
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    note_in, note_out = notes(recorded, REDACT_SPAN)

    assert re.fullmatch(r"1 texts?", note_in), note_in
    assert note_out == "masked: 1 email, 1 phone"
    assert EMAIL not in everything_written(recorded)
