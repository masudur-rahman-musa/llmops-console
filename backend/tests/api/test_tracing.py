"""One trace per request: its id, its stage spans, its attributes and its failures.

Everything here goes through `TestClient` against an app `create_app` built, so
every claim about a span is settled by a real request rather than by reading the
middleware. The spans are read out of the session-scoped
`InMemorySpanExporter` in `conftest.py`, emptied per test by the `spans` fixture.

The contract these tests are written against, before it exists:

* `app.core.tracing.tracer_provider()` - the process-wide provider, built
  once, the one the exporter fixture attaches to.
* `app.core.tracing.SpanAttributes` - the closed, typed attribute model
  whose field set is exactly the ten names in `ALLOWED_ATTRIBUTE_KEYS`.
* `app.core.tracing.build_exporter(settings)` - the OTLP/HTTP exporter
  switch: `None` when no endpoint is configured (decision D).
* `app.core.config.Settings.otlp_endpoint` - optional, absent by
  default, `OPSCONSOLE_OTLP_ENDPOINT` in the environment.

Two things are deliberately NOT pinned here. The root span's NAME: the plan
fixes the four stage names and says nothing about the root's, so every case
identifies the root structurally (the only span with no local parent) and
asserts the stage names around it. And the ROUTE the trace id header rides on
`/v1/redact`: that route writes no `X-Gateway-*` header at all
(tests/gateway/test_redact_endpoint.py, `test_the_response_carries_no_gateway_headers`),
and an earlier change section 8 changes no old test, so section 6's "on the chat,
embeddings and redact routes" is read here as "the redact route emits its trace
too" and pinned by `test_the_redact_route_emits_a_trace_without_a_header`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import inject
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext, StatusCode
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.deps import emitted_trace_id
from app.api.routes.redact import REDACT
from app.cache import InProcessResponseCache
from app.core.config import Settings
from app.core.errors import ProviderError
from app.core.tracing import (
    SpanAttributes,
    annotate,
    build_exporter,
    otlp_headers,
    record_error_code,
    request_trace,
    stage_span,
    tracer_provider,
)
from app.main import create_app
from app.model_routing.errors import AllProvidersUnavailable
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    UsageTotal,
)

from .conftest import (
    API_KEYS,
    CHAT,
    FALLBACK_ALIAS,
    FALLBACK_CLASS,
    FALLBACK_TENANT,
    FIRST_PROVIDER,
    NO_PROVIDER_KEYS,
    SECOND_PROVIDER,
    SHIPPED_POLICY_PATH,
    STREAM_FRAGMENTS,
    STREAM_USAGE,
    STUB_CONTENT,
    STUB_EMBEDDING_USAGE,
    STUB_USAGE,
    TEST_KEYS,
    EmbeddingStubAdapter,
    FallbackStubAdapter,
    StubAdapter,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
    evals_switched_off,
    post_chat,
    post_embeddings,
    retryable_error,
    stream_events,
    stub_response,
)

# --- the header, the tenants and the classes each case needs -----------------

UPSTREAM_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
UPSTREAM_SPAN_ID = "00f067aa0ba902b7"
UPSTREAM_TRACEPARENT = f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01"

ALL_ZERO_TRACE_ID = "0" * 32

# The rows of tests/gateway/test_context_headers.py's rejection table that say
# something different from one another about a PARENT: each is a header a
# receiver must not join a trace on. The trailing-whitespace rows are here
# because `$`-anchored matching accepts them and `re.fullmatch` does not
#.
MALFORMED_TRACEPARENT_CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("not a traceparent at all", "garbage"),
    ("too few fields", f"00-{UPSTREAM_TRACE_ID}"),
    ("too many fields", UPSTREAM_TRACEPARENT + "-extra"),
    ("unsupported version ff", f"ff-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01"),
    ("trace id not hex", f"00-{'z' * 32}-{UPSTREAM_SPAN_ID}-01"),
    ("all-zero trace id", f"00-{ALL_ZERO_TRACE_ID}-{UPSTREAM_SPAN_ID}-01"),
    ("all-zero span id", f"00-{UPSTREAM_TRACE_ID}-{'0' * 16}-01"),
    ("uppercase trace id", f"00-{UPSTREAM_TRACE_ID.upper()}-{UPSTREAM_SPAN_ID}-01"),
    ("trailing space", UPSTREAM_TRACEPARENT + " "),
]

ALIAS = "agent-default"
EMBED_ALIAS = "embed"
TENANT = "evals"
EVALS_KEY = TEST_KEYS["evals"]
NORDLICHT_KEY = TEST_KEYS["nordlicht"]

# `evals` under `confidential` allows [eu, self] and carries `redact: true`, so a
# stub registered under the policy provider `openai` is both the resolved
# candidate and a request that actually runs redaction. Under `internal` the same
# tenant reaches `anthropic` in `us` and redaction does not apply, which is the
# pair that makes the `redact` span's ABSENCE observable.
REDACT_CLASS = "confidential"
PLAIN_CLASS = "internal"
OPENAI_PROVIDER = "openai"
EMBED_PROVIDER = "local_embed"
"""The `embed` alias's only candidate, and therefore the provider span an embeddings
request carries. Its own server on its own address: one llama.cpp process serves one
model."""
REMOTE_PROVIDER = "anthropic"

EMAIL = "alice@example.com"
TEXT_WITH_PII = f"write to {EMAIL} about it"

SESSION = "session-1"

# The stage names the design gives this plan, and the one still unbuilt.
AUTH_SPAN = "auth"
ROUTE_SPAN = "route"
CACHE_SPAN = "cache"
SEMANTIC_SPAN = "semantic"
REDACT_SPAN = "redact"
UNEMITTED_SPAN_NAMES = frozenset({"ledger"})
"""`cache` left this set in an earlier change, which emits it on every chat request; the
positive claim is `test_the_cache_span_is_emitted_on_a_miss_and_on_a_hit` in
tests/gateway/test_exact_cache.py. `semantic` left it in an earlier change, which emits it
on every ELIGIBLE chat request - the alias must carry `semantic_cache: true`, so
the embeddings row and the redact-endpoint row below still carry none; its
positive and negative claims are
`test_the_semantic_span_is_emitted_on_an_eligible_miss_and_on_an_eligible_hit`
and `test_no_semantic_span_is_emitted_on_an_ineligible_request` in
tests/gateway/test_semantic_cache.py. `ledger` stays until a call can be priced."""

ALLOWED_ATTRIBUTE_KEYS = frozenset(
    {
        "tenant",
        "session",
        "alias",
        "provider",
        "data_class",
        "cache",
        "cost_usd",
        "redacted",
        "fallback_count",
    }
)

SESSION_WIRE_KEY = "langfuse.session.id"
"""The key Langfuse reads a session from; the bare `session` key is no longer sent."""

OBSERVATION_TYPE_KEY = "langfuse.observation.type"
GENERATION_WIRE_KEYS = frozenset(
    {
        OBSERVATION_TYPE_KEY,
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.cost",
    }
)
GENERATION_FIELDS = frozenset({"model", "input_tokens", "output_tokens", "cost_usd"})
INPUT_NOTE_KEY = "langfuse.observation.input"
OUTPUT_NOTE_KEY = "langfuse.observation.output"
TAGS_KEY = "langfuse.trace.tags"
FINISH_REASON_KEY = "langfuse.observation.metadata.finish_reason"
PREVIEW_WIRE_KEYS = frozenset({INPUT_NOTE_KEY, OUTPUT_NOTE_KEY, TAGS_KEY})
ALLOWED_WIRE_KEYS = (
    (ALLOWED_ATTRIBUTE_KEYS - {"session"})
    | {SESSION_WIRE_KEY}
    | GENERATION_WIRE_KEYS
    | PREVIEW_WIRE_KEYS
    | {FINISH_REASON_KEY}
)
"""Every key a span may carry on the wire: the `SpanAttributes` keys with the session
renamed, plus the generation keys, the three preview keys and the finish
reason metadata."""

# A string that appears nowhere else in this repository, so finding it on a span
# can only mean the request's own message body was copied there.
CONTENT_SENTINEL = "zqx-prompt-sentinel-847-never-on-a-span"

# What a provider says when it fails. A client never sees it and a
# span must not carry it either: a trace backend is not the provider's confidant.
PROVIDER_MESSAGE_SENTINEL = "upstream request id req_zqx_847 quota exhausted"

# Port 9 (discard) on loopback: nothing listens, so a batch export attempt fails
# on connect. The endpoint is well formed, which is the point - the failure is at
# export time, not at parse time.
UNREACHABLE_OTLP_ENDPOINT = "http://127.0.0.1:9/v1/traces"
CONFIGURED_OTLP_ENDPOINT = "https://cloud.langfuse.com/api/public/otel/v1/traces"


# --- reading spans -----------------------------------------------------------


def finished(spans: InMemorySpanExporter) -> tuple[ReadableSpan, ...]:
    """Every span the request under test finished, in the order they ended."""
    recorded = tuple(spans.get_finished_spans())
    assert recorded, "the request emitted no span at all"
    return recorded


def context_of(span: ReadableSpan) -> SpanContext:
    """A finished span's own context, with the `None` the SDK types away."""
    span_context = span.get_span_context()
    assert span_context is not None, f"{span.name} finished without a span context"
    return span_context


def trace_id_of(span: ReadableSpan) -> str:
    """The span's trace id in the 32-hex form the wire and the header use."""
    return format(context_of(span).trace_id, "032x")


def root_of(recorded: Sequence[ReadableSpan]) -> ReadableSpan:
    """The one span of this request with no LOCAL parent.

    A continued trace's root has a parent, but a remote one, so both the new-trace
    and the continued case are the same structural question. More than one match
    means the request produced more than one trace, which is itself a failure.
    """
    roots = [span for span in recorded if span.parent is None or span.parent.is_remote]
    assert len(roots) == 1, f"expected one root span, got {[span.name for span in roots]}"
    return roots[0]


def stage_names(recorded: Sequence[ReadableSpan]) -> set[str]:
    """The names of every span but the root."""
    root = root_of(recorded)
    return {span.name for span in recorded if span is not root}


def attributes_of(span: ReadableSpan) -> Mapping[str, object]:
    """A span's attributes as a plain mapping, empty rather than `None`."""
    return dict(span.attributes or {})


def attribute_keys(recorded: Sequence[ReadableSpan]) -> set[str]:
    """Every attribute key set anywhere in this request's trace."""
    return {key for span in recorded for key in attributes_of(span)}


def span_text(span: ReadableSpan) -> str:
    """Everything a reader of this span can see, joined into one searchable string.

    Name, status description, attribute keys and values, and every event with its
    own attributes - which is where an exception message would land if the
    implementation recorded one.
    """
    parts: list[str] = [span.name, span.status.description or ""]
    for key, value in attributes_of(span).items():
        parts.append(f"{key}={value}")
    for event in span.events:
        parts.append(event.name)
        parts.extend(f"{key}={value}" for key, value in dict(event.attributes or {}).items())
    return " ".join(parts)


def trace_text(recorded: Sequence[ReadableSpan]) -> str:
    """`span_text` over the whole trace."""
    return " ".join(span_text(span) for span in recorded)


def duration_seconds(span: ReadableSpan) -> float:
    """How long the span was open, in seconds."""
    assert span.start_time is not None and span.end_time is not None, f"{span.name} never ended"
    return (span.end_time - span.start_time) / 1_000_000_000


# --- clients -----------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_chat_stub() -> StubAdapter:
    """A chat stub under `openai`, the provider every data class can reach."""
    return StubAdapter(name=OPENAI_PROVIDER, response=stub_response())


@pytest.fixture(scope="module")
def openai_chat_client(settings: Settings, openai_chat_stub: StubAdapter) -> Iterator[TestClient]:
    """One app for the redact-class chat rows, built once."""
    with build_chat_client(settings, {openai_chat_stub.name: openai_chat_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def failing_client(settings: Settings) -> Iterator[TestClient]:
    """An app whose only adapter raises a `ProviderError` carrying the sentinel."""
    stub = StubAdapter(
        name=REMOTE_PROVIDER,
        error=ProviderError(
            "anthropic: the provider answered 503",
            retryable=True,
            provider_message=PROVIDER_MESSAGE_SENTINEL,
        ),
    )
    with build_chat_client(settings, {stub.name: stub}) as client:
        yield client


class SlowStreamAdapter:
    """A streaming adapter that pauses BETWEEN frames and not before the first.

    The route pulls the first event before it returns a response, so a delay in
    front of it would be counted by a root span that ends at the return as well
    as by one that ends with the stream. Every pause here happens after that
    first event, which makes the two implementations differ by
    `FRAME_DELAY_SECONDS * DELAYED_FRAMES` instead of by nothing.
    """

    FRAME_DELAY_SECONDS = 0.02

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []

    @property
    def delayed_frames(self) -> int:
        """How many pauses one streamed answer contains."""
        return len(STREAM_FRAGMENTS) + 1

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise ProviderError(f"{self.name}: this stub streams and nothing else", retryable=False)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """`StreamStart`, then one pause before every remaining event."""
        self.requests.append(request)
        yield StreamStart()
        for fragment in STREAM_FRAGMENTS:
            time.sleep(self.FRAME_DELAY_SECONDS)
            yield TextDelta(text=fragment)
        time.sleep(self.FRAME_DELAY_SECONDS)
        yield FinishReason(reason="end_turn")
        yield UsageTotal(usage=STREAM_USAGE)

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


@pytest.fixture(scope="module")
def slow_stream_adapter() -> SlowStreamAdapter:
    """The adapter behind `slow_stream_client`."""
    return SlowStreamAdapter(name=REMOTE_PROVIDER)


@pytest.fixture(scope="module")
def slow_stream_client(
    settings: Settings, slow_stream_adapter: SlowStreamAdapter
) -> Iterator[TestClient]:
    """One app for the streamed row, built once."""
    with build_chat_client(settings, {slow_stream_adapter.name: slow_stream_adapter}) as client:
        yield client


class ToggledRaisingExporter(SpanExporter):
    """An exporter that fails on demand, so the failure is scoped to one test.

    A span processor can be added to a tracer provider and never removed, so an
    exporter that always raised would still be attached for every later test in
    the session. `raising` is turned on inside the case and off in its `finally`.
    """

    def __init__(self) -> None:
        self.raising = False
        self.exported = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self.raising:
            raise RuntimeError("the trace backend is unreachable")
        self.exported += len(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.raising = False

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


@pytest.fixture(scope="module")
def raising_exporter() -> ToggledRaisingExporter:
    """A second exporter on the process provider, attached once and left dormant."""
    exporter = ToggledRaisingExporter()
    tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture(scope="module")
def exporting_client(settings: Settings) -> Iterator[TestClient]:
    """An app built with an OTLP endpoint nothing listens on."""
    exporting_settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        otlp_endpoint=UNREACHABLE_OTLP_ENDPOINT,
    )
    assert exporting_settings.api_keys == settings.api_keys
    stub = StubAdapter(name=REMOTE_PROVIDER, response=stub_response())
    with build_chat_client(exporting_settings, {stub.name: stub}) as client:
        yield client


def post_redact(client: TestClient, *, traceparent: str | None = None) -> Response:
    """POST one `/v1/redact` body as the `evals` tenant."""
    response: Response = client.post(
        REDACT,
        json={"text": TEXT_WITH_PII},
        headers=auth_headers(
            key=EVALS_KEY, tenant=TENANT, data_class=PLAIN_CLASS, traceparent=traceparent
        ),
    )
    return response


def send_chat(client: TestClient, traceparent: str | None) -> Response:
    """One chat completion, with or without an upstream trace."""
    return post_chat(client, chat_body(model=ALIAS), traceparent=traceparent)


def send_embeddings(client: TestClient, traceparent: str | None) -> Response:
    """One embeddings request, with or without an upstream trace."""
    return post_embeddings(client, embeddings_body(model=EMBED_ALIAS), traceparent=traceparent)


# --------------------------------------------------------------------------
# the trace id: continued, minted, and the one the header reports
# --------------------------------------------------------------------------


def test_a_valid_traceparent_is_continued(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an implementation that reads the header only for the response value and
    starts a trace of its own underneath - the client's header and the exported trace
    would then be two different traces, and the Helpdesk's spans and the gateway's would
    never join. Also catches a parent recorded as a local span, which would make this
    request's root a child of nothing that exists."""
    response = send_chat(chat_client, UPSTREAM_TRACEPARENT)
    assert response.status_code == 200, response.text

    root = root_of(finished(spans))

    assert trace_id_of(root) == UPSTREAM_TRACE_ID
    assert root.parent is not None, "the incoming span id was not taken as the parent"
    assert format(root.parent.span_id, "016x") == UPSTREAM_SPAN_ID
    assert root.parent.is_remote is True


def test_an_absent_traceparent_starts_a_new_trace(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a root span that inherits whatever context the worker thread was left
    holding - two requests would share one trace id, and one tenant's request would hang
    off another's - and a root minted with the all-zero id no backend will accept."""
    assert send_chat(chat_client, None).status_code == 200
    first = trace_id_of(root_of(finished(spans)))

    spans.clear()
    assert send_chat(chat_client, None).status_code == 200
    second_root = root_of(finished(spans))

    assert second_root.parent is None
    assert trace_id_of(second_root) != first
    assert trace_id_of(second_root) != ALL_ZERO_TRACE_ID
    assert first != ALL_ZERO_TRACE_ID


@pytest.mark.parametrize(
    ("case", "traceparent"),
    MALFORMED_TRACEPARENT_CASES,
    ids=[row[0] for row in MALFORMED_TRACEPARENT_CASES],
)
def test_a_malformed_traceparent_starts_a_new_trace(
    case: str, traceparent: str, chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an implementation that hands the raw header to a propagator instead of
    parsing it first: a caller's broken, uppercase or trailing-space header would become
    a parent, attaching this request to a trace nobody can open, and the request would
    stop being traceable at all. The all-zero rows catch the ids W3C forbids being
    accepted as a parent."""
    response = send_chat(chat_client, traceparent)
    assert response.status_code == 200, response.text

    root = root_of(finished(spans))

    assert root.parent is None, f"{case}: a malformed header was taken as a parent"
    assert trace_id_of(root) not in {UPSTREAM_TRACE_ID, ALL_ZERO_TRACE_ID}


TRACE_ID_HEADER_ROWS: list[tuple[str, str, Callable[[TestClient, str | None], Response]]] = [
    ("chat-absent", "chat_client", send_chat),
    ("embeddings-absent", "embeddings_client", send_embeddings),
]


@pytest.mark.parametrize(
    ("case", "client_fixture", "send"),
    TRACE_ID_HEADER_ROWS,
    ids=[row[0] for row in TRACE_ID_HEADER_ROWS],
)
@pytest.mark.parametrize("traceparent", [None, UPSTREAM_TRACEPARENT], ids=["new", "continued"])
def test_the_trace_id_header_is_the_emitted_traces_id(
    case: str,
    client_fixture: str,
    send: Callable[[TestClient, str | None], Response],
    traceparent: str | None,
    request: pytest.FixtureRequest,
    spans: InMemorySpanExporter,
) -> None:
    """Catches the two ids drifting apart: a `trace_id_from` that still mints its own
    while the middleware starts another, or a header written from the parent rather than
    from the span that was started. An engineer would paste the id a client logged into
    the trace viewer and find nothing, on exactly the request they needed."""
    client: TestClient = request.getfixturevalue(client_fixture)

    response = send(client, traceparent)
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    assert response.headers["x-gateway-trace-id"] == trace_id_of(root_of(recorded))
    assert len({trace_id_of(span) for span in recorded}) == 1


def test_the_redact_route_emits_a_trace_without_a_header(
    gateway_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a middleware mounted on the two model routes rather than on the app: the
    one endpoint that handles raw personal data would produce no trace at all. The second
    half catches the middleware answering the gap by writing `X-Gateway-Trace-Id` here,
    which would contradict the pin that this route carries no gateway header."""
    response = post_redact(gateway_client)
    assert response.status_code == 200, response.text

    recorded = finished(spans)

    assert len({trace_id_of(span) for span in recorded}) == 1
    assert REDACT_SPAN in stage_names(recorded)
    assert not [name for name in response.headers if name.lower().startswith("x-gateway-")]


# --------------------------------------------------------------------------
# an independent tracer standing in for Helpdesk
# --------------------------------------------------------------------------


def simulated_helpdesk_turn(exporter: InMemorySpanExporter) -> tuple[str, ReadableSpan]:
    """One simulated Helpdesk request: a root span on its OWN `TracerProvider`, and the
    `traceparent` a real client would send from it.

    Never `app.core.tracing.tracer_provider()` - that is the gateway's own
    process-wide provider, and a span started on it would share the gateway's context
    trivially rather than by the header the two services actually exchange. The provider is
    never shut down: `InMemorySpanExporter.shutdown()` clears its own buffer, and this
    exporter is shared across turns so a caller can read every span it has recorded."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("simulated-helpdesk")
    carrier: dict[str, str] = {}
    with tracer.start_as_current_span("helpdesk-turn"):
        inject(carrier)
    traceparent = carrier["traceparent"]
    helpdesk_span = exporter.get_finished_spans()[-1]
    return traceparent, helpdesk_span


def test_a_span_from_a_second_tracer_shares_the_gateways_trace_id(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Proves the "Helpdesk and gateway spans share one trace" claim at the only end that
    exists today: a genuinely separate
    provider's span, propagated through a real `traceparent`, lands the gateway's root span
    and every stage span under it in that same trace."""
    helpdesk_exporter = InMemorySpanExporter()
    traceparent, helpdesk_span = simulated_helpdesk_turn(helpdesk_exporter)

    response = send_chat(chat_client, traceparent)
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    gateway_trace_id = trace_id_of(helpdesk_span)

    assert gateway_trace_id == trace_id_of(root_of(recorded))
    assert {trace_id_of(span) for span in recorded} == {gateway_trace_id}


def test_two_separate_simulated_turns_never_share_a_trace_id(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches the shared-id assertion above being trivially true - a constant header, or a
    provider that always mints the same id: two independent simulated Helpdesk turns, each
    with its own root span and its own request to the gateway, must produce two DIFFERENT
    trace ids end to end, on both the simulated and the gateway side."""
    helpdesk_exporter = InMemorySpanExporter()

    first_traceparent, first_helpdesk_span = simulated_helpdesk_turn(helpdesk_exporter)
    first_response = send_chat(chat_client, first_traceparent)
    assert first_response.status_code == 200, first_response.text
    first_gateway_trace_id = trace_id_of(root_of(finished(spans)))

    spans.clear()

    second_traceparent, second_helpdesk_span = simulated_helpdesk_turn(helpdesk_exporter)
    second_response = send_chat(chat_client, second_traceparent)
    assert second_response.status_code == 200, second_response.text
    second_gateway_trace_id = trace_id_of(root_of(finished(spans)))

    assert trace_id_of(first_helpdesk_span) != trace_id_of(second_helpdesk_span)
    assert first_gateway_trace_id != second_gateway_trace_id
    assert first_gateway_trace_id == trace_id_of(first_helpdesk_span)
    assert second_gateway_trace_id == trace_id_of(second_helpdesk_span)


# --------------------------------------------------------------------------
# the span names
# --------------------------------------------------------------------------

SPAN_NAME_ROWS: list[tuple[str, str, str, Callable[[TestClient], Response], set[str]]] = [
    (
        "chat-redact-class",
        "openai_chat_client",
        OPENAI_PROVIDER,
        lambda client: post_chat(
            client,
            chat_body(model=ALIAS, messages=[{"role": "user", "content": TEXT_WITH_PII}]),
            data_class=REDACT_CLASS,
        ),
        {
            AUTH_SPAN,
            ROUTE_SPAN,
            CACHE_SPAN,
            SEMANTIC_SPAN,
            REDACT_SPAN,
            f"provider:{OPENAI_PROVIDER}",
        },
    ),
    (
        "embeddings-plain-class",
        "embeddings_client",
        EMBED_PROVIDER,
        lambda client: post_embeddings(client, embeddings_body(model=EMBED_ALIAS)),
        {AUTH_SPAN, ROUTE_SPAN, f"provider:{EMBED_PROVIDER}"},
    ),
    (
        "redact-endpoint",
        "gateway_client",
        "",
        lambda client: post_redact(client),
        {AUTH_SPAN, REDACT_SPAN},
    ),
]


@pytest.mark.parametrize(
    ("case", "client_fixture", "provider", "send", "expected"),
    SPAN_NAME_ROWS,
    ids=[row[0] for row in SPAN_NAME_ROWS],
)
def test_the_emitted_span_names_are_exactly_the_stages_that_exist(
    case: str,
    client_fixture: str,
    provider: str,
    send: Callable[[TestClient], Response],
    expected: set[str],
    request: pytest.FixtureRequest,
    spans: InMemorySpanExporter,
) -> None:
    """Catches both directions at once: a stage that emits no span (an operator reading
    the trace cannot tell whether redaction ran or where the time went) and a span
    emitted for a stage this plan does not have, which would make a later plan's work
    look done. The redact-endpoint row also catches `route` or `provider:` spans on a
    route that resolves nothing."""
    client: TestClient = request.getfixturevalue(client_fixture)

    response = send(client)
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    assert stage_names(recorded) == expected
    assert len(recorded) == len(expected) + 1


def test_no_ledger_span_is_emitted(
    chat_client: TestClient, embeddings_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a placeholder span for the one the design stage still unbuilt: a `ledger`
    span would say a call was booked when nothing prices it yet, which makes a later
    plan's work look done. Asserted on the two shapes the exact-set rows do not cover - a
    plain-class chat and an embeddings request in the same buffer."""
    assert send_chat(chat_client, None).status_code == 200
    assert send_embeddings(embeddings_client, None).status_code == 200

    names = {span.name for span in finished(spans)}

    assert names & UNEMITTED_SPAN_NAMES == set()
    assert AUTH_SPAN in names, "the assertion above is only meaningful if spans were emitted"


def test_the_stage_spans_share_the_roots_trace_id(
    openai_chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches the plan's threadpool risk: the routes are `def`, so they run in a worker
    thread, and a stage span started there without the request's context becomes the root
    of a SECOND trace. The request would look traced and the provider call would be
    missing from the trace an engineer opens."""
    response = post_chat(
        openai_chat_client,
        chat_body(model=ALIAS, messages=[{"role": "user", "content": TEXT_WITH_PII}]),
        data_class=REDACT_CLASS,
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    root = root_of(recorded)

    assert len(recorded) == 7, [span.name for span in recorded]
    assert {trace_id_of(span) for span in recorded} == {trace_id_of(root)}
    for span in recorded:
        if span is root:
            continue
        assert span.parent is not None, f"{span.name} started its own trace"
        assert format(span.parent.trace_id, "032x") == trace_id_of(root)


def test_redact_span_is_absent_on_a_non_redact_class(
    openai_chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a `with` block placed around the `redaction_applies` CHECK rather than
    inside the branch it guards: every request would carry a `redact` span, and the trace
    would tell an auditor that a `public` or `internal` request was redacted when nothing
    touched it. The redact-class half is what stops the absence being satisfied by a
    route that never emits the span at all."""
    body = chat_body(model=ALIAS, messages=[{"role": "user", "content": TEXT_WITH_PII}])

    plain = post_chat(openai_chat_client, body, data_class=PLAIN_CLASS)
    assert plain.status_code == 200, plain.text
    plain_names = stage_names(finished(spans))

    spans.clear()
    redacted = post_chat(openai_chat_client, body, data_class=REDACT_CLASS)
    assert redacted.status_code == 200, redacted.text
    redacted_names = stage_names(finished(spans))

    assert REDACT_SPAN not in plain_names
    assert REDACT_SPAN in redacted_names


def test_the_provider_span_names_the_resolved_provider(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a span named from the body's alias instead of from the resolver's answer:
    `provider:agent-default` names something that serves nothing, and the one question a
    provider span exists to answer - which provider took this request - would be
    unanswerable from the trace."""
    response = send_chat(chat_client, None)
    assert response.status_code == 200, response.text

    names = stage_names(finished(spans))

    assert f"provider:{REMOTE_PROVIDER}" in names
    assert f"provider:{ALIAS}" not in names
    assert response.headers["x-gateway-provider"] == REMOTE_PROVIDER


# --------------------------------------------------------------------------
# the closed attribute set
# --------------------------------------------------------------------------


def test_the_attribute_key_set_is_exactly_the_allowlist(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a field added to the typed model (an `api_key`, a `user`, a `prompt`)
    and an attribute written straight onto a span past the model, either of which turns
    "never user identity" back into a review promise. The rejected `user` row is the half
    that pins the set as CLOSED rather than merely as the current shape."""
    assert set(SpanAttributes.model_fields) == ALLOWED_ATTRIBUTE_KEYS

    with pytest.raises(ValidationError):
        SpanAttributes.model_validate({"user": "alice"})

    response: Response = chat_client.post(
        CHAT,
        json=chat_body(model=ALIAS),
        headers=auth_headers(key=EVALS_KEY, tenant=TENANT, data_class=PLAIN_CLASS, session=SESSION),
    )
    assert response.status_code == 200, response.text

    keys = attribute_keys(finished(spans))

    assert keys <= ALLOWED_WIRE_KEYS, sorted(keys - ALLOWED_WIRE_KEYS)
    assert {"tenant", SESSION_WIRE_KEY, "provider"} <= keys


def test_no_message_content_reaches_a_span(
    chat_client: TestClient, chat_stub: StubAdapter, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches an implementation that puts the prompt, the answer or a header value on a
    span for a tenant whose `trace_content` switch is off (an earlier change turned `evals` on, so
    this runs on a copy with it off): the trace backend is outside this system's
    redaction guarantees. The bearer key row catches a middleware that records request
    headers wholesale, which no switch allows."""
    with evals_switched_off(tmp_path, chat_client):
        response = chat_client.post(
            CHAT,
            json=chat_body(model=ALIAS, messages=[{"role": "user", "content": CONTENT_SENTINEL}]),
            headers=auth_headers(
                key=EVALS_KEY, tenant=TENANT, data_class=PLAIN_CLASS, session=SESSION
            ),
        )
    assert response.status_code == 200, response.text
    assert chat_stub.requests[-1].messages[0].content == CONTENT_SENTINEL, (
        "the sentinel never reached the adapter, so its absence from the spans proves nothing"
    )

    text = trace_text(finished(spans))

    assert CONTENT_SENTINEL not in text
    assert EVALS_KEY not in text
    assert STUB_CONTENT not in text


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


def test_a_streamed_requests_root_span_covers_the_frames(
    slow_stream_client: TestClient,
    slow_stream_adapter: SlowStreamAdapter,
    spans: InMemorySpanExporter,
) -> None:
    """Catches a root span ended where the route RETURNS the `StreamingResponse`: the
    adapter pauses only after the first event, so such a span closes before any frame is
    sent and every streamed call would be recorded as a millisecond however long the
    answer actually took - the one measurement a streaming gateway is asked for."""
    delay = SlowStreamAdapter.FRAME_DELAY_SECONDS * slow_stream_adapter.delayed_frames

    response = post_chat(slow_stream_client, chat_body(model=ALIAS, stream=True))

    assert response.status_code == 200, response.text
    assert response.text.count("data:") == len(STREAM_FRAGMENTS) + 4
    assert duration_seconds(root_of(finished(spans))) >= delay


# --------------------------------------------------------------------------
# failures still emit their trace
# --------------------------------------------------------------------------


def test_a_refusal_records_the_code_on_the_span(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a middleware that ends the root span OK whatever the answer was: a refusal is
    the most important request in the trace store and an unset status makes it
    indistinguishable from a served one when filtering for failures. Also catches the
    refusal's own message being written onto the span, which would name the alias the caller
    asked for."""
    response = post_chat(
        chat_client, chat_body(model="no-such-alias"), key=NORDLICHT_KEY, tenant="nordlicht"
    )
    assert_error_body(response, status=404, code="ALIAS_UNKNOWN")

    recorded = finished(spans)
    root = root_of(recorded)

    assert root.status.status_code is StatusCode.ERROR
    assert root.status.description == "ALIAS_UNKNOWN"
    assert "no-such-alias" not in trace_text(recorded)
    assert attribute_keys(recorded) <= ALLOWED_WIRE_KEYS


def test_a_provider_failure_records_the_code_on_the_span(
    failing_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches `record_exception` on the raised error: the provider's own words - an
    upstream request id, a quota message - would be copied onto an exception event, which
    is the same disclosure the response body is already forbidden from making, moved to a
    place nobody reviews. The typed code is what an operator filters on instead."""
    response = post_chat(failing_client, chat_body(model=ALIAS))
    assert_error_body(response, status=502, code="PROVIDER_ERROR")

    recorded = finished(spans)
    root = root_of(recorded)

    assert root.status.status_code is StatusCode.ERROR
    assert root.status.description == "PROVIDER_ERROR"
    assert PROVIDER_MESSAGE_SENTINEL not in trace_text(recorded)


def test_an_unauthorized_request_still_emits_a_trace(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a root span started from the request CONTEXT rather than by the middleware:
    auth runs before any context exists, so an unauthenticated request would emit nothing
    and a key being probed against the gateway would leave no trace at all."""
    response = chat_client.post(CHAT, json=chat_body(model=ALIAS))
    assert_error_body(response, status=401, code="UNAUTHORIZED")

    root = root_of(finished(spans))

    assert root.status.status_code is StatusCode.ERROR
    assert root.status.description == "UNAUTHORIZED"
    assert "tenant" not in attributes_of(root), "there is no authenticated tenant to record"


# --------------------------------------------------------------------------
# the exporter switch
# --------------------------------------------------------------------------


def test_no_exporter_is_installed_without_an_endpoint(
    gateway_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches decision D being implemented as a console exporter by default, which would
    print every request's tenant and session to stdout in production, and catches
    a default endpoint compiled in - a gateway posting spans to an address nobody
    configured. The served request is the other half: with no exporter the seam is still
    there and spans are still created."""
    assert build_exporter(Settings(otlp_endpoint=None)) is None

    exporter = build_exporter(Settings(otlp_endpoint=CONFIGURED_OTLP_ENDPOINT))
    assert isinstance(exporter, OTLPSpanExporter)

    response = post_redact(gateway_client)
    assert response.status_code == 200, response.text
    assert finished(spans)


def test_an_exporter_that_raises_changes_no_response(
    exporting_client: TestClient,
    raising_exporter: ToggledRaisingExporter,
    spans: InMemorySpanExporter,
) -> None:
    """Catches an export on the request path: an exporter built per request against an
    unreachable endpoint, or a flush called inline in the middleware, turns an outage in
    the trace backend into an outage of the gateway - the observability tool taking the
    system down with it. Both halves run at once: the app is configured with an endpoint
    nothing listens on, and a second exporter raises on every span this request ends."""
    raising_exporter.raising = True
    try:
        response = post_chat(exporting_client, chat_body(model=ALIAS))
    finally:
        raising_exporter.raising = False

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == STUB_CONTENT
    assert response.headers["x-gateway-provider"] == REMOTE_PROVIDER
    assert response.headers["x-gateway-trace-id"] == trace_id_of(root_of(finished(spans)))


# --------------------------------------------------------------------------
# decision F: a typed failure raised inside middleware
# --------------------------------------------------------------------------


class RaisingMiddleware:
    """A throwaway ASGI middleware that raises a typed gateway failure.

    Pure ASGI rather than `BaseHTTPMiddleware` because that is what the root-span
    middleware is: it runs outside `ExceptionMiddleware`, which is exactly the
    position that makes decision F necessary.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            raise AllProvidersUnavailable("middleware: nothing may serve this request")
        await self.app(scope, receive, send)


def test_a_gateway_error_raised_in_middleware_keeps_its_status_and_code(
    settings: Settings,
) -> None:
    """Catches the registration Starlette gives us today: the `Exception` handler is
    lifted into `ServerErrorMiddleware`, so a `GatewayError` raised in middleware answers
    500 with `code: null` and a client's retry logic sees a gateway bug instead of the
    503 it can act on. This plan installs the first middleware, so it is the plan that
    owes the fix."""
    app = create_app(settings, NO_PROVIDER_KEYS)
    app.add_middleware(RaisingMiddleware)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/healthz")

    assert_error_body(response, status=503, code="ALL_PROVIDERS_FAILED")


# --------------------------------------------------------------------------
# the seams at their edges: a scope nobody traced, and the header parser
# --------------------------------------------------------------------------


def test_an_untraced_scope_still_runs_every_stage() -> None:
    """Catches `stage_span` and `annotate` assuming the middleware ran. A router mounted
    on an app that does not install `TracingMiddleware` - a future admin app, a unit test
    building one route - would raise inside the `with` and turn a missing trace into a
    failed request, which inverts the rule that observability never takes the system
    down."""
    scope: dict[str, object] = {"type": "http"}

    with stage_span(scope, ROUTE_SPAN) as span:
        assert span is None

    annotate(scope, SpanAttributes(tenant=TENANT))
    record_error_code(scope, "PROVIDER_ERROR")

    assert request_trace(scope) is None


def test_an_untraced_scope_still_gets_a_trace_id() -> None:
    """Catches the header contract being made conditional on tracing: without the
    fallback a route on an untraced app would answer no `X-Gateway-Trace-Id` at all, or
    crash reading a span that was never started."""
    first = emitted_trace_id({"type": "http"})
    second = emitted_trace_id({"type": "http"})

    assert len(first) == 32 and int(first, 16) >= 0
    assert first != second


OTLP_HEADER_ROWS: list[tuple[str, str | None, dict[str, str] | None]] = [
    ("absent", None, None),
    ("blank", "   ", None),
    ("one pair", "Authorization=Basic abc", {"Authorization": "Basic abc"}),
    (
        "two pairs, spaced",
        " Authorization = Basic abc , x-scope = eu ",
        {"Authorization": "Basic abc", "x-scope": "eu"},
    ),
    ("a value containing =", "Authorization=Basic ab=cd", {"Authorization": "Basic ab=cd"}),
    ("no separator", "Authorization", None),
    ("empty name", "=value", None),
]


@pytest.mark.parametrize(
    ("case", "raw", "expected"), OTLP_HEADER_ROWS, ids=[row[0] for row in OTLP_HEADER_ROWS]
)
def test_the_otlp_header_setting_is_parsed_into_pairs(
    case: str, raw: str | None, expected: dict[str, str] | None
) -> None:
    """Catches a parser that splits on every `=` rather than the first: a Langfuse basic
    credential is base64 and ends in `=` padding, so a wrong split truncates the secret
    and every export is rejected as unauthorised - with nothing on the request path to
    say so, because export failures are deliberately silent. The malformed rows catch a
    setting that yields a header with an empty name, which some exporters send anyway."""
    assert otlp_headers(raw) == expected


# --------------------------------------------------------------------------
# fallback_count
# --------------------------------------------------------------------------
#
# The last of the three attributes an earlier change declared and left unset. It is
# asserted as a VALUE on whichever span carries it rather than on a span chosen
# by name: `annotate` puts every label on the root span by design (stage spans
# take none), and pinning the carrier here would pin that choice twice.


def test_the_failed_candidate_keeps_its_own_provider_span(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    spans: InMemorySpanExporter,
) -> None:
    """The only way an operator identifies a provider shedding its traffic behind a
    WORKING fallback, and the claim `core/metrics.py` and a design note both make in place of
    the provider-error counter - which a 200 never moves, because it reaches no error
    handler. Catches a loop wrapped in one provider span, or the span moved outside the
    `try` so a failed attempt leaves none: either way both passages become false and
    nothing else in the suite would notice. The request's own `provider` attribute names
    the OTHER one, which is what makes the pair readable as "anthropic shed, openai served"
    rather than as two providers that were both used."""
    fallback_adapters[FIRST_PROVIDER].chat_error = retryable_error(FIRST_PROVIDER)

    response = post_chat(
        fallback_client,
        chat_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    names = stage_names(recorded)
    assert f"provider:{FIRST_PROVIDER}" in names
    assert f"provider:{SECOND_PROVIDER}" in names

    served = [
        attributes_of(span)["provider"] for span in recorded if "provider" in attributes_of(span)
    ]
    assert served == [SECOND_PROVIDER]


def test_a_fallback_is_recorded_on_the_trace_as_a_count(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    spans: InMemorySpanExporter,
) -> None:
    """Catches the two ways this attribute goes wrong once something finally sets it. A
    count of ATTEMPTS rather than of fallbacks reads 2 for one fallback, so every trace
    search for "requests that fell back" would match every request; and an attribute set
    on more than one span double-counts under any aggregation a backend does over a trace.
    The request is a real fallback, confirmed by the header, so a 0 here means the
    attribute was never written rather than that nothing happened."""
    fallback_adapters[FIRST_PROVIDER].chat_error = retryable_error(FIRST_PROVIDER)

    response = post_chat(
        fallback_client,
        chat_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER

    carriers = [
        attributes_of(span)["fallback_count"]
        for span in finished(spans)
        if "fallback_count" in attributes_of(span)
    ]

    assert carriers == [1]


# --------------------------------------------------------------------------
# an earlier change: the provider call is a Langfuse generation
# --------------------------------------------------------------------------

GENERATION = "generation"
COST_HEADER = "x-gateway-cost-usd"
PROVIDER_SPAN = f"provider:{REMOTE_PROVIDER}"
OPENAI_PROVIDER_SPAN = f"provider:{OPENAI_PROVIDER}"
MODEL_KEY = "gen_ai.request.model"
INPUT_TOKENS_KEY = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS_KEY = "gen_ai.usage.output_tokens"
COST_KEY = "gen_ai.usage.cost"


def span_named(recorded: Sequence[ReadableSpan], name: str) -> ReadableSpan:
    """The one span with this name."""
    matches = [span for span in recorded if span.name == name]
    assert len(matches) == 1, f"expected one {name!r} span, got {[s.name for s in recorded]}"
    return matches[0]


def is_labelled(span: ReadableSpan) -> bool:
    """True when the span carries any generation label."""
    return any(
        key.startswith("gen_ai.") or key == OBSERVATION_TYPE_KEY for key in attributes_of(span)
    )


@pytest.fixture(scope="module")
def cached_chat_stub() -> StubAdapter:
    """The adapter behind `cached_chat_client`."""
    return StubAdapter(name=REMOTE_PROVIDER, response=stub_response())


@pytest.fixture(scope="module")
def cached_chat_client(settings: Settings, cached_chat_stub: StubAdapter) -> Iterator[TestClient]:
    """One app for the cache-hit row, built once."""
    with build_chat_client(settings, {cached_chat_stub.name: cached_chat_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def no_usage_stub() -> StubAdapter:
    """A streaming stub whose stream ends without a usage event."""
    return StubAdapter(name=REMOTE_PROVIDER, events=stream_events(usage=None))


@pytest.fixture(scope="module")
def no_usage_client(settings: Settings, no_usage_stub: StubAdapter) -> Iterator[TestClient]:
    """One app for the stream-without-usage row, built once."""
    with build_chat_client(settings, {no_usage_stub.name: no_usage_stub}) as client:
        yield client


def test_the_provider_span_is_a_generation_with_model_tokens_and_cost(
    chat_client: TestClient, chat_stub: StubAdapter, spans: InMemorySpanExporter
) -> None:
    """Catches an implementation that leaves the provider call a plain span, labels it with
    the alias instead of the provider's model, or books a second price instead of the
    one `X-Gateway-Cost-USD` carries, so Langfuse's Model, Tokens and Cost stay empty or wrong."""
    response = send_chat(chat_client, None)
    assert response.status_code == 200, response.text

    provider = span_named(finished(spans), PROVIDER_SPAN)
    attributes = attributes_of(provider)

    assert attributes[OBSERVATION_TYPE_KEY] == GENERATION
    assert attributes[MODEL_KEY] == chat_stub.requests[-1].model
    assert attributes[MODEL_KEY] != ALIAS
    assert attributes[INPUT_TOKENS_KEY] == STUB_USAGE.prompt_tokens
    assert attributes[OUTPUT_TOKENS_KEY] == STUB_USAGE.completion_tokens
    assert attributes[COST_KEY] == float(response.headers[COST_HEADER])


def test_only_the_provider_span_is_a_generation(
    openai_chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a label applied to every stage span (or to the root), which would make
    Langfuse count auth, route and cache steps as model calls and double every token total."""
    response = post_chat(
        openai_chat_client,
        chat_body(model=ALIAS, messages=[{"role": "user", "content": TEXT_WITH_PII}]),
        data_class=REDACT_CLASS,
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    others = [span for span in recorded if span.name != f"provider:{OPENAI_PROVIDER}"]

    assert OBSERVATION_TYPE_KEY in attributes_of(
        span_named(recorded, f"provider:{OPENAI_PROVIDER}")
    ), "the check below is only meaningful if the provider span is the generation"
    assert {AUTH_SPAN, ROUTE_SPAN, CACHE_SPAN, REDACT_SPAN} <= {span.name for span in others}
    for span in others:
        assert not is_labelled(span), f"{span.name} carries generation labels"


def test_a_streamed_generation_carries_the_streams_usage(
    slow_stream_client: TestClient,
    slow_stream_adapter: SlowStreamAdapter,
    spans: InMemorySpanExporter,
) -> None:
    """Catches a streamed generation closed when the route returns (it would end before the
    last frame and cannot know the usage) and one filled from the plain-chat numbers, so a
    streamed call would show no tokens or the wrong ones."""
    delay = SlowStreamAdapter.FRAME_DELAY_SECONDS * slow_stream_adapter.delayed_frames

    response = post_chat(slow_stream_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text

    provider = span_named(finished(spans), PROVIDER_SPAN)
    attributes = attributes_of(provider)

    assert duration_seconds(provider) >= delay
    assert attributes[OBSERVATION_TYPE_KEY] == GENERATION
    assert attributes[MODEL_KEY] == slow_stream_adapter.requests[-1].model
    assert attributes[INPUT_TOKENS_KEY] == STREAM_USAGE.prompt_tokens
    assert attributes[OUTPUT_TOKENS_KEY] == STREAM_USAGE.completion_tokens


def test_a_stream_without_usage_ends_with_the_model_only(
    no_usage_client: TestClient, no_usage_stub: StubAdapter, spans: InMemorySpanExporter
) -> None:
    """Catches a stream that makes up zero tokens when the provider reported none: Langfuse
    would show a call that cost nothing and used nothing, instead of one with unknown usage."""
    response = post_chat(no_usage_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text

    attributes = attributes_of(span_named(finished(spans), PROVIDER_SPAN))

    assert attributes[OBSERVATION_TYPE_KEY] == GENERATION
    assert attributes[MODEL_KEY] == no_usage_stub.requests[-1].model
    assert not [key for key in attributes if key.startswith("gen_ai.usage.")]


def test_an_embeddings_call_is_a_generation_with_input_tokens(
    embeddings_client: TestClient,
    embedding_stub: EmbeddingStubAdapter,
    spans: InMemorySpanExporter,
) -> None:
    """Catches an embeddings call left a plain span, or given output tokens it does not
    have: Langfuse would leave embedding usage out of Tokens and Dashboards."""
    response = send_embeddings(embeddings_client, None)
    assert response.status_code == 200, response.text

    attributes = attributes_of(span_named(finished(spans), f"provider:{EMBED_PROVIDER}"))

    assert attributes[OBSERVATION_TYPE_KEY] == GENERATION
    assert attributes[MODEL_KEY] == embedding_stub.requests[-1].model
    assert attributes[INPUT_TOKENS_KEY] == STUB_EMBEDDING_USAGE.prompt_tokens
    assert OUTPUT_TOKENS_KEY not in attributes


def test_a_cache_hit_has_no_generation(
    cached_chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a generation recorded for a cache hit: no provider was called, so Langfuse
    would count tokens and cost that were never spent. The first, missing call is the
    guard that the label exists at all."""
    app_of(cached_chat_client).state.response_cache = InProcessResponseCache()
    body = chat_body(model=ALIAS, temperature=0)

    first = post_chat(cached_chat_client, body)
    assert first.headers["x-gateway-cache"] == "miss"
    assert [s for s in finished(spans) if OBSERVATION_TYPE_KEY in attributes_of(s)]

    spans.clear()
    second = post_chat(cached_chat_client, body)
    assert second.headers["x-gateway-cache"] == "hit"

    recorded = finished(spans)
    assert [span for span in recorded if OBSERVATION_TYPE_KEY in attributes_of(span)] == []
    assert not [span for span in recorded if span.name.startswith("provider:")]


def test_the_generation_label_set_is_closed() -> None:
    """Catches a free-form field (a `prompt`, a `user`) on the generation model, and a
    wrong wire key: the field set and the emitted keys are pinned whole."""
    from app.core.tracing import GenerationAttributes

    assert set(GenerationAttributes.model_fields) == GENERATION_FIELDS

    with pytest.raises(ValidationError):
        GenerationAttributes.model_validate({"prompt": "x"})

    assert GenerationAttributes().as_attributes() == {OBSERVATION_TYPE_KEY: GENERATION}
    assert GenerationAttributes(
        model="m", input_tokens=3, output_tokens=4, cost_usd=0.5
    ).as_attributes() == {
        OBSERVATION_TYPE_KEY: GENERATION,
        MODEL_KEY: "m",
        INPUT_TOKENS_KEY: 3,
        OUTPUT_TOKENS_KEY: 4,
        COST_KEY: 0.5,
    }


def test_the_session_goes_out_as_langfuse_session_id(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a session still sent under the bare `session` key (Langfuse's Sessions page
    stays empty) and one sent under both keys, which leaves a stale duplicate."""
    response = chat_client.post(
        CHAT,
        json=chat_body(model=ALIAS),
        headers=auth_headers(key=EVALS_KEY, tenant=TENANT, data_class=PLAIN_CLASS, session="s-1"),
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)

    assert attributes_of(root_of(recorded))[SESSION_WIRE_KEY] == "s-1"
    assert "session" not in attribute_keys(recorded)
    assert SpanAttributes(session="s-1").as_attributes()[SESSION_WIRE_KEY] == "s-1"


ROOT_NAME_ROWS: list[tuple[str, str, Callable[[TestClient], Response], str]] = [
    ("chat", "chat_client", lambda client: send_chat(client, None), "POST /v1/chat/completions"),
    (
        "embeddings",
        "embeddings_client",
        lambda client: send_embeddings(client, None),
        "POST /v1/embeddings",
    ),
    ("redact", "gateway_client", lambda client: post_redact(client), "POST /v1/redact"),
]


@pytest.mark.parametrize(
    ("case", "client_fixture", "send", "expected"),
    ROOT_NAME_ROWS,
    ids=[row[0] for row in ROOT_NAME_ROWS],
)
def test_the_trace_is_named_from_the_matched_route(
    case: str,
    client_fixture: str,
    send: Callable[[TestClient], Response],
    expected: str,
    request: pytest.FixtureRequest,
    spans: InMemorySpanExporter,
) -> None:
    """Catches the root span keeping the name `http.request` for every request, or being
    named from the raw path: Langfuse's trace list could not tell a chat from an embedding."""
    client: TestClient = request.getfixturevalue(client_fixture)

    assert send(client).status_code == 200

    name = root_of(finished(spans)).name
    assert name == expected
    assert name != "http.request"


def test_an_unmatched_path_is_named_unmatched(
    gateway_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a trace named from the raw URL: a junk path of any length would become a
    trace name (and a long one a very large one) in the trace store."""
    junk = "x" * 500
    response = gateway_client.get("/v1/" + junk)
    assert response.status_code == 404

    recorded = finished(spans)

    assert root_of(recorded).name == "GET unmatched"
    assert junk not in trace_text(recorded)


PROBE_ROWS: list[tuple[str, str, int]] = [
    ("healthz", "/healthz", 200),
    ("readyz", "/readyz", 200),
    ("metrics", "/metrics", 200),
]


@pytest.mark.parametrize(("case", "path", "status"), PROBE_ROWS, ids=[row[0] for row in PROBE_ROWS])
def test_probes_and_scrapes_are_not_traced(
    case: str,
    path: str,
    status: int,
    gateway_client: TestClient,
    spans: InMemorySpanExporter,
) -> None:
    """Catches a probe or scrape still starting a span: thousands a day would bury real
    requests and use up the free Langfuse plan. The status check catches a probe that
    stops answering as the way of not tracing it."""
    response = gateway_client.get(path)

    assert response.status_code == status
    assert spans.get_finished_spans() == ()


def test_a_real_request_after_a_probe_is_still_traced(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches the untraced-start state leaking into the next request (a flag left set, or
    a shared context): the chat after a probe would lose its trace, or the probe's would
    be exported with it."""
    assert chat_client.get("/healthz").status_code == 200
    assert send_chat(chat_client, None).status_code == 200

    recorded = finished(spans)

    assert root_of(recorded).name == "POST /v1/chat/completions"
    assert len({trace_id_of(span) for span in recorded}) == 1


# --------------------------------------------------------------------------
# an earlier change: every step carries a short Input and Output note, and the tenant is a tag
# --------------------------------------------------------------------------

INPUT_TEXT = "not sent: prompts stay inside the gateway"
OUTPUT_TEXT = "not sent"
PLAIN_INPUT = INPUT_TEXT
PLAIN_OUTPUT = OUTPUT_TEXT
PREVIEW_FIELDS = frozenset({"text", "alias", "messages", "finish_reason", "tokens_out"})
OPENAI_FINISH_REASON = "stop"
"""What the stub's `end_turn` becomes on the wire (`FINISH_REASON_BY_STOP_REASON`)."""


def note_of(span: ReadableSpan, key: str) -> str:
    """The span's Input or Output note, a plain line; fails when the key is missing."""
    raw = attributes_of(span).get(key)
    assert isinstance(raw, str), f"{span.name} has no {key} text, got {raw!r}"
    assert not raw.startswith("{"), f"{span.name} {key} is JSON, not a plain line"
    return raw


def note_text(recorded: Sequence[ReadableSpan]) -> str:
    """Every note of every span, joined: what a Langfuse reader sees in Input and Output."""
    return " ".join(
        str(attributes_of(span).get(key, ""))
        for span in recorded
        for key in (INPUT_NOTE_KEY, OUTPUT_NOTE_KEY)
    )


@pytest.fixture(scope="module")
def finished_stream_stub() -> StubAdapter:
    """A streaming stub that ends with a finish event and a usage event."""
    return StubAdapter(name=REMOTE_PROVIDER, events=stream_events())


@pytest.fixture(scope="module")
def finished_stream_client(
    settings: Settings, finished_stream_stub: StubAdapter
) -> Iterator[TestClient]:
    """One app for the stream-with-finish-and-usage rows, built once."""
    with build_chat_client(settings, {finished_stream_stub.name: finished_stream_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def bare_stream_stub() -> StubAdapter:
    """A streaming stub that sends text and then stops: no finish event, no usage event."""
    return StubAdapter(
        name=REMOTE_PROVIDER, events=(StreamStart(), TextDelta(text="Berlin"), TextDelta(text="."))
    )


@pytest.fixture(scope="module")
def bare_stream_client(settings: Settings, bare_stream_stub: StubAdapter) -> Iterator[TestClient]:
    """One app for the stream-with-neither row, built once."""
    with build_chat_client(settings, {bare_stream_stub.name: bare_stream_stub}) as client:
        yield client


def test_every_step_has_an_input_and_output_note(
    openai_chat_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches an implementation that adds the notes to the root and the provider only:
    Langfuse would still show `undefined` on every other step. Only the root and the
    provider step say `not sent`; a stage step that still starts with it has not been given
    its decision. Redaction is on so the `redact` step is among the spans checked; `evals`
    is switched off so the provider step shows its note and not the conversation."""
    with evals_switched_off(tmp_path, openai_chat_client):
        response = post_chat(
            openai_chat_client,
            chat_body(model=ALIAS, messages=[{"role": "user", "content": TEXT_WITH_PII}]),
            data_class=REDACT_CLASS,
        )
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    root = root_of(recorded)

    assert REDACT_SPAN in {span.name for span in recorded}, "the redact step must be in play"
    for span in recorded:
        for key in (INPUT_NOTE_KEY, OUTPUT_NOTE_KEY):
            note = note_of(span, key)
            if span is root or span.name == OPENAI_PROVIDER_SPAN:
                assert note.startswith("not sent"), (span.name, key)
            else:
                assert note != "", (span.name, key)
                assert not note.startswith("not sent"), (span.name, key)


def test_a_stage_step_carries_only_the_plain_notes(
    openai_chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a stage step left on the starting lines, one that copies the provider's
    facts (finish reason, tokens) or another stage's decision onto itself, and a redact
    line that names a value or a placeholder instead of a count."""
    response = post_chat(
        openai_chat_client,
        chat_body(
            model=ALIAS,
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": TEXT_WITH_PII},
            ],
        ),
        data_class=REDACT_CLASS,
    )
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    expected = {
        AUTH_SPAN: ("API key and X-Tenant", "accepted: tenant evals"),
        ROUTE_SPAN: (
            f"alias {ALIAS}, data class {REDACT_CLASS}",
            "picked openai gpt-5-mini; fallback none",
        ),
        REDACT_SPAN: ("2 messages", "masked: 1 email"),
        CACHE_SPAN: (f"same request answered before, alias {ALIAS}", "miss"),
        SEMANTIC_SPAN: (
            f"nearest stored answer, alias {ALIAS}",
            "miss (the question could not be embedded)",
        ),
    }
    for name, (want_input, want_output) in expected.items():
        span = span_named(recorded, name)
        assert note_of(span, INPUT_NOTE_KEY) == want_input, name
        assert note_of(span, OUTPUT_NOTE_KEY) == want_output, name


def test_the_provider_step_notes_say_alias_messages_finish_and_tokens(
    chat_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches a provider note that counts the messages after redaction or from the
    wrong list, reports the stub's raw `end_turn` instead of the OpenAI word, or
    reports prompt or total tokens as `tokens_out`."""
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Which city?"},
        {"role": "user", "content": "Really?"},
    ]
    with evals_switched_off(tmp_path, chat_client):
        response = post_chat(chat_client, chat_body(model=ALIAS, messages=messages))
    assert response.status_code == 200, response.text

    provider = span_named(finished(spans), PROVIDER_SPAN)

    assert note_of(provider, INPUT_NOTE_KEY) == (
        "not sent: prompts stay inside the gateway (alias agent-default, 3 messages)"
    )
    assert note_of(provider, OUTPUT_NOTE_KEY) == (
        f"not sent (finish reason {OPENAI_FINISH_REASON}, "
        f"{STUB_USAGE.completion_tokens} tokens out)"
    )


def test_the_root_step_carries_the_provider_notes(
    chat_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches a root that keeps its starting plain notes: Langfuse reads the trace's
    own Input and Output from the root, so the trace list would still show bare text."""
    with evals_switched_off(tmp_path, chat_client):
        response = send_chat(chat_client, None)
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    root = root_of(recorded)
    provider = span_named(recorded, PROVIDER_SPAN)

    assert note_of(root, INPUT_NOTE_KEY) == note_of(provider, INPUT_NOTE_KEY)
    assert note_of(root, OUTPUT_NOTE_KEY) == note_of(provider, OUTPUT_NOTE_KEY)
    assert "alias agent-default" in note_of(root, INPUT_NOTE_KEY), "must not be plain == plain"
    assert "tokens out" in note_of(root, OUTPUT_NOTE_KEY)


def test_a_streamed_provider_note_has_finish_reason_and_tokens(
    finished_stream_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches a streamed note filled from the plain-chat numbers (29, not 52), left
    without the finish reason, or carrying the stub's raw `end_turn`."""
    with evals_switched_off(tmp_path, finished_stream_client):
        response = post_chat(finished_stream_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text

    recorded = finished(spans)
    expected = (
        f"not sent (finish reason {OPENAI_FINISH_REASON}, "
        f"{STREAM_USAGE.completion_tokens} tokens out)"
    )

    assert note_of(span_named(recorded, PROVIDER_SPAN), OUTPUT_NOTE_KEY) == expected
    assert note_of(root_of(recorded), OUTPUT_NOTE_KEY) == expected


def test_a_stream_without_finish_or_usage_ends_with_text_only(
    bare_stream_client: TestClient, spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    """Catches a stream that makes up a finish reason or a token count, or leaves empty
    brackets, when the provider reported neither: the line is the bare `not sent`."""
    with evals_switched_off(tmp_path, bare_stream_client):
        response = post_chat(bare_stream_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text

    recorded = finished(spans)

    assert attributes_of(span_named(recorded, PROVIDER_SPAN))[OUTPUT_NOTE_KEY] == "not sent"
    assert note_of(span_named(recorded, PROVIDER_SPAN), OUTPUT_NOTE_KEY) == PLAIN_OUTPUT
    assert note_of(root_of(recorded), OUTPUT_NOTE_KEY) == PLAIN_OUTPUT


def test_a_cache_hit_root_says_not_sent(
    cached_chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a cache-hit root that copies a finish reason and token count from the
    earlier miss (no provider answered this request), or one that loses the alias and
    message count because no provider span ran."""
    app_of(cached_chat_client).state.response_cache = InProcessResponseCache()
    body = chat_body(model=ALIAS, temperature=0)

    first = post_chat(cached_chat_client, body)
    assert first.headers["x-gateway-cache"] == "miss"

    spans.clear()
    second = post_chat(cached_chat_client, body)
    assert second.headers["x-gateway-cache"] == "hit"

    root = root_of(finished(spans))

    assert note_of(root, INPUT_NOTE_KEY) == f"{INPUT_TEXT} (alias {ALIAS}, 1 message)"
    assert note_of(root, OUTPUT_NOTE_KEY) == PLAIN_OUTPUT


def test_an_embeddings_provider_note_has_the_alias(
    embeddings_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches an embeddings note that invents a message count (an embeddings call has
    none) or leaves out the alias the caller asked for."""
    response = send_embeddings(embeddings_client, None)
    assert response.status_code == 200, response.text

    provider = span_named(finished(spans), f"provider:{EMBED_PROVIDER}")

    assert note_of(provider, INPUT_NOTE_KEY) == f"{INPUT_TEXT} (alias {EMBED_ALIAS})"


def test_no_note_carries_message_content(
    chat_client: TestClient,
    finished_stream_client: TestClient,
    embeddings_client: TestClient,
    spans: InMemorySpanExporter,
    tmp_path: Path,
) -> None:
    """Catches a note built from the request or the answer (the last message, the answer
    text, the embeddings input) instead of from the fixed sentences and counts. Runs with
    `evals` switched off on the chat apps, because a switched-on tenant sends text on
    purpose; the embeddings input stays out whatever the switch says."""
    sentinel_messages = [{"role": "user", "content": CONTENT_SENTINEL}]
    with evals_switched_off(tmp_path, chat_client, finished_stream_client):
        sent = [
            post_chat(chat_client, chat_body(model=ALIAS, messages=sentinel_messages)),
            post_chat(
                finished_stream_client,
                chat_body(model=ALIAS, messages=sentinel_messages, stream=True),
            ),
            post_embeddings(embeddings_client, embeddings_body(input=CONTENT_SENTINEL)),
        ]
    forbidden = (CONTENT_SENTINEL, STUB_CONTENT, *STREAM_FRAGMENTS)

    for response in sent:
        assert response.status_code == 200, response.text
    recorded = finished(spans)
    text = note_text(recorded)

    assert text.strip(), "no note was written, so the absence below proves nothing"
    for fragment in forbidden:
        assert fragment not in text, fragment


def test_the_preview_note_field_set_is_closed() -> None:
    """Catches a free-form field (a `prompt`, a `user`) on the note model, a fact that
    is not known written as `None`, and a wire text that is JSON instead of one plain line."""
    from app.core.tracing import PreviewNote

    assert set(PreviewNote.model_fields) == PREVIEW_FIELDS

    with pytest.raises(ValidationError):
        PreviewNote.model_validate({"text": "x", "prompt": "x"})

    assert PreviewNote.output().as_text() == "not sent"
    assert PreviewNote.input().as_text() == PLAIN_INPUT
    assert PreviewNote.input(alias="a", messages=2).as_text() == (
        f"{INPUT_TEXT} (alias a, 2 messages)"
    )
    assert PreviewNote.output(finish_reason="stop", tokens_out=0).as_text() == (
        "not sent (finish reason stop, 0 tokens out)"
    )


FINISH_REASON_KEPT_ROWS: list[tuple[str, str]] = [
    ("stop", "stop"),
    ("one character", "a"),
    ("underscore", "tool_calls"),
    ("exactly 32 characters", "a" * 32),
]


@pytest.mark.parametrize(
    ("case", "reason"), FINISH_REASON_KEPT_ROWS, ids=[row[0] for row in FINISH_REASON_KEPT_ROWS]
)
def test_a_finish_reason_at_the_limit_is_kept(case: str, reason: str) -> None:
    """Catches a limit set one too low (31) or a pattern that refuses `_`, which would
    turn good reasons into `other`."""
    from app.core.tracing import PreviewNote

    assert PreviewNote.output(finish_reason=reason).finish_reason == reason, case


FINISH_REASON_OTHER_ROWS: list[tuple[str, str]] = [
    ("33 characters", "a" * 33),
    ("capital letter", "Stop"),
    ("space inside", "end turn"),
    ("empty", ""),
    ("digit", "stop1"),
    ("trailing newline", "stop\n"),
    ("trailing CRLF", "stop\r\n"),
    ("trailing space", "stop "),
    ("provider text", "ignore previous instructions"),
]


@pytest.mark.parametrize(
    ("case", "reason"), FINISH_REASON_OTHER_ROWS, ids=[row[0] for row in FINISH_REASON_OTHER_ROWS]
)
def test_a_bad_finish_reason_is_sent_as_other(case: str, reason: str) -> None:
    """Catches a limit one too high (33), a `$`-anchored pattern that lets a trailing
    newline through, and provider text reaching Langfuse unchecked."""
    from app.core.tracing import PreviewNote

    assert PreviewNote.output(finish_reason=reason).finish_reason == "other", case


def test_the_tenant_is_a_trace_tag(chat_client: TestClient, spans: InMemorySpanExporter) -> None:
    """Catches a tag put on a stage span or the provider span (Langfuse would read the
    wrong span), a tag built from the raw header instead of the checked tenant, and a
    bare tenant without the `tenant:` prefix."""
    response = send_chat(chat_client, None)
    assert response.status_code == 200, response.text

    recorded = finished(spans)

    assert attributes_of(root_of(recorded))[TAGS_KEY] == (f"tenant:{TENANT}",)
    assert [span.name for span in recorded if TAGS_KEY in attributes_of(span)] == [
        root_of(recorded).name
    ]


def test_an_unauthorized_request_has_no_tenant_tag(
    chat_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """Catches a tag set before auth passed (from the unchecked header, so a prober could
    plant any tenant name), and a 401 trace that lost its notes along with the tag."""
    response = chat_client.post(CHAT, json=chat_body(model=ALIAS))
    assert_error_body(response, status=401, code="UNAUTHORIZED")

    root = root_of(finished(spans))

    assert TAGS_KEY not in attributes_of(root)
    assert note_of(root, INPUT_NOTE_KEY) == PLAIN_INPUT
    assert note_of(root, OUTPUT_NOTE_KEY) == PLAIN_OUTPUT
