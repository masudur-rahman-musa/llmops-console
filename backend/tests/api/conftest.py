"""Fixtures for the request-envelope tests: keys, headers, errors, /v1/models.

a project rule: the app, its key store and the policy snapshot are built once,
in session-scoped fixtures. Per-test isolation comes from the tenant, the key and
the headers a case sends, never from rebuilding the app.

Three clients, each with its own app from `create_app`, so all three carry the
real handlers and none shares state with another:

* `gateway_client` talks to the shipped app exactly as it is served.
* `harness_client` talks to a second app plus the five routes `build_harness`
  registers under `/_harness/`: `context`, `residency`, `validated`, `boom` and
  `http-detail`. They exist because this plan adds no route that can be observed
  from the outside: nothing yet returns the `RequestContext` a header builds, and
  nothing yet raises a `GatewayError`, an `HTTPException` or a bug at request
  time (`/v1/models` filters the unresolvable aliases out rather than failing).
  Registering them on an app the factory built keeps the handlers, the
  dependencies and the state under test real; only those five endpoints are
  test-owned.
* `crashing_client` is a third app with the same five routes and
  `raise_server_exceptions` off, so the 500 handler answers instead of the
  exception reaching the test. Keeping it separate means an accidental crash in
  any other test still surfaces as the real exception rather than a masked 500.

An earlier change adds a fourth client, `chat_client`, and the `StubAdapter` behind it;
the block at the foot of this file explains why the adapter set is replaced after
`create_app` rather than built from the environment.

Every `create_app` call in this directory passes an explicit environment
(`grep -rn "create_app(" tests/`), `NO_PROVIDER_KEYS` here. `create_app`
otherwise reads `os.environ`, so a developer with a real `ANTHROPIC_API_KEY`
exported would have every app in this directory holding a live-keyed client
against `api.anthropic.com` - a suite that passes locally by making a real,
billed call and fails in CI, which is the inversion #66 exists to prevent.

The plaintext keys below are test data, not secrets: they are random-looking only
by shape, are worth nothing, and match no provider's key format.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from app.core.config import Settings
from app.core.errors import ProviderError
from app.core.metrics import registry
from app.core.tracing import tracer_provider
from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.gateway.app import create_app
from app.gateway.auth import MAX_KEY_LENGTH
from app.gateway.context import RequestContext, request_context
from app.model_routing.errors import ResidencyUnavailable, RoutingError
from app.model_routing.policy import DataClass, Policy, load_policy
from app.model_routing.resolver import resolve
from app.providers.base import (
    Adapter,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageTotal,
)
from app.providers.breaker import CircuitBreaker

# Resolved from this file, so the tests do not depend on pytest's working
# directory. The same path the shipped `Settings.routing_policy_path` defaults to.
SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "routing.yaml"

# A tenant name of exactly `MAX_IDENTIFIER_LENGTH` characters, used for the
# inclusive boundary row of the `X-Tenant` pre-check. It has a key of its own so
# the boundary case reaches the header check instead of stopping at auth.
MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH

# A key of exactly `MAX_KEY_LENGTH` characters and one character over it. Both
# are registered in the store, so the over-cap row is rejected by the length
# pre-check rather than by simply being unknown.
AT_CAP_KEY = "k" * MAX_KEY_LENGTH
OVER_CAP_KEY = "k" * (MAX_KEY_LENGTH + 1)

UNKNOWN_KEY = "test-key-belonging-to-nobody"

# tenant -> plaintext key. The store is given the sha256 digests of these.
TEST_KEYS: Mapping[str, str] = {
    "nordlicht": "test-key-nordlicht",
    "cuan": "test-key-cuan",
    "evals": "test-key-evals",
    MAX_TENANT: "test-key-max-tenant",
    "cap-key": AT_CAP_KEY,
    "over-cap-key": OVER_CAP_KEY,
}


def sha256_hex(key: str) -> str:
    """The digest form a key is stored as. Never the plaintext."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_entries(keys: Mapping[str, str]) -> str:
    """Render `tenant:sha256hex` pairs the way `OPSCONSOLE_API_KEYS` carries them."""
    return ",".join(f"{tenant}:{sha256_hex(key)}" for tenant, key in keys.items())


API_KEYS = key_entries(TEST_KEYS)

# The environment every app in this directory is built against: no provider key,
# whatever the developer's shell exports. See the module docstring.
NO_PROVIDER_KEYS: Mapping[str, str] = {}


def auth_headers(
    *,
    key: str | None,
    tenant: str | None,
    data_class: str | None = None,
    session: str | None = None,
    traceparent: str | None = None,
    authorization: str | None = None,
) -> dict[str, str]:
    """Build the request headers, omitting anything left as `None`.

    `authorization` overrides the whole header, for the rows that send something
    other than a well-formed `Bearer` value.
    """
    headers: dict[str, str] = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    elif key is not None:
        headers["Authorization"] = f"Bearer {key}"
    if tenant is not None:
        headers["X-Tenant"] = tenant
    if data_class is not None:
        headers["X-Data-Class"] = data_class
    if session is not None:
        headers["X-Session"] = session
    if traceparent is not None:
        headers["traceparent"] = traceparent
    return headers


def assert_error_body(response: Response, *, status: int, code: str | None) -> None:
    """Assert the OpenAI-shaped failure body exactly: shape, status and code.

    The key set is compared as a whole, so an extra field (a stack trace, a
    `detail` list leaking from FastAPI's default handler) fails here. `code` is
    `None` for the failures Starlette raises before any of our code runs, which
    carry the shape but no typed code.
    """
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"message", "type", "code", "param"}
    assert error["code"] == code
    assert isinstance(error["message"], str)
    assert error["message"] != ""
    assert isinstance(error["type"], str)
    assert error["type"] != ""
    assert error["param"] is None or isinstance(error["param"], str)


def resolvable_aliases(policy: Policy, tenant: str, data_class: DataClass) -> list[str]:
    """The aliases `resolve()` does not raise for, in policy order.

    `/v1/models` is specified as exactly this list, so the
    expectation is computed from the resolver rather than copied from the policy
    file. Each test that uses it also asserts one membership literally, so a
    resolver that returned nothing could not make the comparison pass.
    """
    listed: list[str] = []
    for alias in policy.aliases:
        try:
            resolve(policy, tenant, alias, data_class.value)
        except RoutingError:
            continue
        listed.append(alias)
    return listed


def build_harness(app: FastAPI) -> FastAPI:
    """Add the five observation routes described in this module's docstring."""

    @app.get("/_harness/context")
    def read_context(
        context: Annotated[RequestContext, Depends(request_context)],
    ) -> RequestContext:
        """Echo the context the header pre-checks built."""
        return context

    @app.get("/_harness/residency")
    def raise_residency() -> None:
        """Raise a `GatewayError` at request time, for the handler test."""
        raise ResidencyUnavailable("harness: nothing may serve this request")

    @app.get("/_harness/validated")
    def validated(count: int) -> dict[str, int]:
        """A required, typed query parameter, for the FastAPI validation test."""
        return {"count": count}

    @app.get("/_harness/boom")
    def boom() -> None:
        """Raise something untyped, for the last-resort handler test."""
        raise RuntimeError("harness: a bug in a route")

    @app.get("/_harness/http-detail")
    def http_detail() -> None:
        """Raise an `HTTPException` whose detail is not a string."""
        # FastAPI's own `HTTPException` types `detail` as `Any`, which is how a
        # non-string one reaches the handler at all; Starlette's types it `str`.
        raise HTTPException(status_code=409, detail={"reason": "already registered"})

    return app


@pytest.fixture(scope="session")
def settings() -> Settings:
    """One settings object: the test key store and the shipped policy."""
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH)


@pytest.fixture(scope="session")
def gateway_client(settings: Settings) -> Iterator[TestClient]:
    """The shipped app, built once."""
    with TestClient(create_app(settings, NO_PROVIDER_KEYS)) as client:
        yield client


@pytest.fixture(scope="session")
def harness_client(settings: Settings) -> Iterator[TestClient]:
    """The shipped app plus the `/_harness/` observation routes, built once."""
    with TestClient(build_harness(create_app(settings, NO_PROVIDER_KEYS))) as client:
        yield client


@pytest.fixture(scope="session")
def crashing_client(settings: Settings) -> Iterator[TestClient]:
    """The harness app with `raise_server_exceptions` off.

    `TestClient` re-raises an unhandled exception by default, which is right for
    every other test and would hide the very handler under test here: the client
    would see the `RuntimeError` instead of the 500 a real client gets.
    """
    with TestClient(
        build_harness(create_app(settings, NO_PROVIDER_KEYS)), raise_server_exceptions=False
    ) as client:
        yield client


@pytest.fixture(scope="session")
def policy() -> Policy:
    """The shipped routing policy, loaded once, for computing expectations."""
    return load_policy(SHIPPED_POLICY_PATH)


# --- chat completions ---------------------------------------------
#
# `POST /v1/chat/completions` needs an adapter behind the resolver, and the real
# one would need a provider key. `StubAdapter` stands in: it satisfies the
# `Adapter` protocol (held by `test_stub_adapter_satisfies_the_protocol`), records
# every `ChatRequest` the route hands it so the outbound shape can be asserted
# exactly, and answers with a fixed response or raises a fixed error.
#
# It is keyed by the POLICY provider name, which is what `app.state.adapters` is
# keyed by and what the resolver's health seam looks up. In the shipped
# `routing.yaml` only `anthropic` gets a stub, so `local` and `openai` have no
# adapter: that is what makes the 503 case reachable through `nordlicht`'s
# `agent-default` override, without editing the policy.
#
# `chat_client` and its stub are session-scoped and shared. A case that needs a
# differently-behaving adapter builds its own client with `build_chat_client`;
# the app is in-process and there is no backing service, so that costs a YAML
# parse.

CHAT = "/v1/chat/completions"

# The route's own numbers, chosen distinct from each other and from any fixture
# so a zero-fill, a swap or a recomputed total is visible.
STUB_USAGE = Usage(prompt_tokens=137, completion_tokens=29, total_tokens=166)

STUB_CONTENT = "The stub answered."

# What the adapter reports as the model it called. Deliberately neither the alias
# nor the policy's provider model id, so a route that echoes the adapter's answer
# into the response body, or into `X-Gateway-Model`, is caught rather than
# accidentally right.
STUB_RESPONSE_MODEL = "stub-model-neither-alias-nor-policy-id"

STUB_TOOL_CALL = ToolCall(
    id="toolu_01StubWeatherCall00000", name="get_weather", arguments='{"city": "Berlin"}'
)


# --- streaming ----------------------------------------------------
#
# `stream_chat` answers a scripted list of neutral events. The numbers and the
# strings below are deliberately different from the non-streaming ones above, so
# a route that renders a stream out of the non-streaming path - or fills the
# usage chunk from `STUB_USAGE` - is visible rather than accidentally right.
#
# `stream_error` with `stream_error_at` is what makes the mid-stream failure
# reachable: at 0 the generator raises before yielding anything (the pre-stream
# failure), above 0 it raises after that many events have already left.

STREAM_FRAGMENTS = ("Berlin", " is the ", "capital of Germany.")
STREAM_ANSWER = "Berlin is the capital of Germany."

STREAM_USAGE = Usage(prompt_tokens=431, completion_tokens=52, total_tokens=483)

STREAM_TOOL_CALL = ToolCall(
    id="toolu_01StubStreamCall00000",
    name="get_weather",
    arguments='{"city": "Berlin", "units": "metric"}',
)

# Not 0: an implementation that hardcodes the tool call index would pass with 0.
STREAM_TOOL_INDEX = 2


def stream_events(
    *,
    fragments: tuple[str, ...] = STREAM_FRAGMENTS,
    tool_calls: tuple[ToolCallDelta, ...] = (),
    finish_reason: str = "end_turn",
    usage: Usage | None = STREAM_USAGE,
) -> tuple[StreamEvent, ...]:
    """One scripted stream in the neutral vocabulary, in the order an adapter yields it."""
    events: list[StreamEvent] = [StreamStart()]
    events.extend(TextDelta(text=fragment) for fragment in fragments)
    events.extend(tool_calls)
    events.append(FinishReason(reason=finish_reason))
    if usage is not None:
        events.append(UsageTotal(usage=usage))
    return tuple(events)


def stub_response(
    *,
    content: str = STUB_CONTENT,
    tool_calls: tuple[ToolCall, ...] = (),
    finish_reason: str = "end_turn",
    usage: Usage = STUB_USAGE,
) -> ChatResponse:
    """One neutral `ChatResponse` for a stub adapter to answer with."""
    return ChatResponse(
        model=STUB_RESPONSE_MODEL,
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


class StubAdapter:
    """An in-process stand-in for a provider adapter, holding the real protocol.

    `requests` is the whole point: the route's outbound `ChatRequest` is compared
    field by field, so a wire-to-neutral mapping that drops the tool result turn,
    reorders the turns or sends the alias as the model is caught here rather than
    by an assertion about the answer.
    """

    def __init__(
        self,
        *,
        name: str,
        response: ChatResponse | None = None,
        error: Exception | None = None,
        events: Sequence[StreamEvent] | None = None,
        stream_error: Exception | None = None,
        stream_error_at: int = 0,
    ) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []
        self._response = response if response is not None else stub_response()
        self._error = error
        self._events = tuple(events) if events is not None else stream_events()
        self._stream_error = stream_error
        self._stream_error_at = stream_error_at
        self.stream_closed = False

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Yield the scripted events, raising the scripted error at its position.

        `stream_closed` records that the route released this generator, which is
        the only way a test can see the provider connection a real adapter holds
        being given back
        (`test_closing_the_frame_generator_releases_the_provider_stream`).
        """
        self.requests.append(request)
        try:
            for position, event in enumerate(self._events):
                if self._stream_error is not None and position == self._stream_error_at:
                    raise self._stream_error
                yield event
            if self._stream_error is not None and self._stream_error_at >= len(self._events):
                raise self._stream_error
        finally:
            self.stream_closed = True

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


def build_chat_client(settings: Settings, adapters: Mapping[str, Adapter]) -> TestClient:
    """A real app whose adapter set is the given stubs, keyed by policy provider name.

    `NO_PROVIDER_KEYS` means `create_app` builds nothing of its own, so no
    provider key is read whatever the shell holds (#66); replacing the dict
    afterwards is what keeps the route, the resolver and the error handlers real.
    """
    app = create_app(settings, NO_PROVIDER_KEYS)
    app.state.adapters = dict(adapters)
    return TestClient(app)


def chat_body(**overrides: object) -> dict[str, Any]:
    """The baseline OpenAI-shaped request body, with the given fields replaced."""
    body: dict[str, Any] = {
        "model": "agent-default",
        "messages": [{"role": "user", "content": "Which city?"}],
    }
    body.update(overrides)
    return body


def post_chat(
    client: TestClient,
    body: Mapping[str, Any],
    *,
    key: str = TEST_KEYS["evals"],
    tenant: str = "evals",
    data_class: str | None = "internal",
    traceparent: str | None = None,
) -> Response:
    """POST one chat completion as the `evals` tenant under `internal` by default.

    That pair is the one the shipped policy lets reach the US `anthropic`
    provider: `evals` has `residency: any`, and its `internal` rule allows `us`.
    The default data class (`confidential`) allows only `eu` and `self`, so a
    case that forgets the header would be a 451, not a success.
    """
    response: Response = client.post(
        CHAT,
        json=dict(body),
        headers=auth_headers(
            key=key, tenant=tenant, data_class=data_class, traceparent=traceparent
        ),
    )
    return response


@pytest.fixture(scope="session")
def chat_stub() -> StubAdapter:
    """The shared stub behind `chat_client`, answering a plain successful completion."""
    return StubAdapter(name="anthropic")


@pytest.fixture(scope="session")
def chat_client(settings: Settings, chat_stub: StubAdapter) -> Iterator[TestClient]:
    """The shipped app with one adapter, for the policy provider `anthropic`."""
    with build_chat_client(settings, {chat_stub.name: chat_stub}) as client:
        yield client


# --- embeddings ---------------------------------------------------
#
# `EmbeddingStubAdapter` is the second stub, not a widened first one. The
# EXISTING `StubAdapter` deliberately has no `embed`, and that absence is load
# bearing: registering it under a policy provider is what makes "this alias's
# only candidate cannot embed" reachable as the resolver's 503 without editing
# `routing.yaml` (`test_a_candidate_without_an_embedding_adapter_is_503`).
#
# The shipped policy sends `embed` to the provider `local_embed` alone, region
# `self` - its own server on its own address, because one llama.cpp process
# serves one model - so every case here registers its stub under `local_embed`.
# The chat provider `local` is a DIFFERENT adapter in this suite for the same
# reason it is a different pod in the cluster.

EMBEDDINGS = "/v1/embeddings"
"""The published route path. `test_the_route_path_is_the_published_one` asserts
this against the route module's own constant, so the two cannot drift."""

EMBED_ALIAS = "embed"

EMBED_PROVIDER = "local_embed"
"""The `embed` alias's only candidate in the shipped policy: the embedding server,
which is not the chat provider `local`. A stub registered under the chat name
leaves the alias with no healthy candidate, which is a 503 and not a test of
anything this suite is about."""

# Every component is exactly representable as a float32, so the base64 round trip
# in `test_embeddings.py` compares EQUAL rather than approximately equal: a wrong
# byte order is then the only thing that can break it, which is the whole point
# of the case (the plan's silent-corruption risk).
STUB_VECTORS: tuple[tuple[float, ...], ...] = (
    (0.5, -1.5, 2.25),
    (-0.75, 3.125, -0.0625),
    (4.5, -0.375, 1.75),
)

# Distinct from every chat number and from every provider fixture number, and the
# two counts differ from each other, so a swap or a zero-fill is visible.
STUB_EMBEDDING_USAGE = EmbeddingUsage(prompt_tokens=97, total_tokens=113)

# What the stub reports as the model it embedded with: neither the alias nor the
# policy's provider model id, so a route that echoes the adapter's answer into
# the response body or into `X-Gateway-Model` is caught rather than accidentally
# right.
STUB_EMBEDDING_MODEL = "stub-embedding-model-neither-alias-nor-policy-id"


def embedding_stub_response(count: int = 1) -> EmbeddingResponse:
    """A neutral `EmbeddingResponse` carrying `count` of the fixed vectors, numbered 0..n-1."""
    return EmbeddingResponse(
        model=STUB_EMBEDDING_MODEL,
        embeddings=tuple(
            Embedding(index=index, vector=vector)
            for index, vector in enumerate(STUB_VECTORS[:count])
        ),
        usage=STUB_EMBEDDING_USAGE,
    )


class EmbeddingStubAdapter:
    """An in-process stand-in for an embedding-capable adapter, holding the real protocol.

    `requests` is the whole point, as on `StubAdapter`: the route's outbound
    `EmbeddingRequest` is compared field by field, so a route that sends the
    alias as the model, or that flattens a list input to its first item, is
    caught here rather than by an assertion about the answer.

    It satisfies `Adapter` too - `chat` and `stream_chat` raise - because
    `app.state.adapters` is typed as a mapping of `Adapter` and the chat route
    would otherwise be able to reach a provider that cannot serve it silently.
    """

    def __init__(
        self,
        *,
        name: str,
        response: EmbeddingResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.requests: list[EmbeddingRequest] = []
        self._response = response if response is not None else embedding_stub_response()
        self._error = error

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        raise ProviderError(f"{self.name}: this stub embeds and nothing else", retryable=False)

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


def embeddings_body(**overrides: object) -> dict[str, Any]:
    """The baseline OpenAI-shaped embeddings body, with the given fields replaced."""
    body: dict[str, Any] = {"model": EMBED_ALIAS, "input": "the capital of Germany"}
    body.update(overrides)
    return body


def post_embeddings(
    client: TestClient,
    body: Mapping[str, Any],
    *,
    key: str = TEST_KEYS["evals"],
    tenant: str = "evals",
    data_class: str | None = "internal",
    traceparent: str | None = None,
) -> Response:
    """POST one embeddings request as the `evals` tenant under `internal` by default.

    The same default pair `post_chat` uses, and it reaches `local` for this alias:
    `evals` has `residency: any` and its `internal` rule allows `self`. The
    default data class (`confidential`) also allows `self`, so unlike the chat
    helper this one would not be saved by a residency refusal if the header were
    dropped - the header is sent explicitly for that reason.
    """
    response: Response = client.post(
        EMBEDDINGS,
        json=dict(body),
        headers=auth_headers(
            key=key, tenant=tenant, data_class=data_class, traceparent=traceparent
        ),
    )
    return response


@pytest.fixture(scope="session")
def embedding_stub() -> EmbeddingStubAdapter:
    """The shared stub behind `embeddings_client`, answering one fixed vector."""
    return EmbeddingStubAdapter(name=EMBED_PROVIDER)


@pytest.fixture(scope="session")
def embeddings_client(
    settings: Settings, embedding_stub: EmbeddingStubAdapter
) -> Iterator[TestClient]:
    """The shipped app with one embedding-capable adapter, for the policy provider
    `local_embed`."""
    with build_chat_client(settings, {embedding_stub.name: embedding_stub}) as client:
        yield client


# --- tracing ------------------------------------------------------
#
# One `InMemorySpanExporter`, attached once to the process-wide tracer provider
# `app.core.tracing.tracer_provider()` owns. a project rule: the
# exporter and its span processor are session-scoped and built once, and
# per-test isolation comes from `clear()` on the buffer rather than from a new
# provider, a new exporter or a new app - none of which OpenTelemetry lets you
# take back, since a span processor can be added to a provider but never
# removed.
#
# `SimpleSpanProcessor` rather than the batching one on purpose: it exports on
# `on_end`, so a finished span is readable the moment the request returns and no
# test has to flush or wait. The shipped exporter switch is a different question
# and is tested directly (`test_no_exporter_is_installed_without_an_endpoint`).


@pytest.fixture(scope="session")
def span_exporter() -> Iterator[InMemorySpanExporter]:
    """The one in-memory exporter every tracing test reads, attached once."""
    exporter = InMemorySpanExporter()
    tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.shutdown()


@pytest.fixture
def spans(span_exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    """The exporter, emptied before the test's own request runs.

    Emptied at the START rather than at the end, so spans left by a test that
    never asked for this fixture - every other file in this directory makes
    requests through the same provider - cannot be read as this test's.
    """
    span_exporter.clear()
    yield span_exporter


# --- metrics ------------------------------------------------------
#
# One `CollectorRegistry`, the module-owned one `app.core.metrics` holds,
# read through `MetricDelta`. a project rule in its shape for a registry: a
# family can be added to a registry and never taken back, and every app in this
# session feeds the same one, so per-test isolation is a DELTA rather than a new
# registry - every sample is read before the test's own requests and again after,
# and the tests assert the difference.
#
# The `_created` series prometheus_client mints beside every counter child are
# dropped on the way in. They are timestamps, not counts, and they would make a
# "one new time series" assertion count two.


SampleKey = tuple[str, tuple[tuple[str, str], ...]]
"""One time series: its sample name, and its labels as sorted pairs."""


def _sample_key(name: str, labels: Mapping[str, str]) -> SampleKey:
    """The key a sample is filed under, with the labels in a comparable order."""
    return (name, tuple(sorted(labels.items())))


class MetricDelta:
    """What one test changed in the shared registry: per series, and per family.

    Built at the start of a test, so `before` is whatever the session has already
    accumulated. Every method answers the change since then, which is what makes a
    process-wide registry safe to assert against without rebuilding it.
    """

    def __init__(self, collector_registry: CollectorRegistry) -> None:
        self._registry = collector_registry
        self._before = self._read()

    def _read(self) -> dict[SampleKey, float]:
        """Every series in the registry now, `_created` timestamps excluded."""
        samples: dict[SampleKey, float] = {}
        for family in self._registry.collect():
            for sample in family.samples:
                if sample.name.endswith("_created"):
                    continue
                samples[_sample_key(sample.name, sample.labels)] = sample.value
        return samples

    def delta(self, name: str, **labels: str) -> float:
        """How much one exactly-labelled series moved during this test.

        A series that did not exist before counts from zero, so an emission that
        creates its child during the test is measured rather than missed.
        """
        key = _sample_key(name, labels)
        now = self._read()
        return now.get(key, 0.0) - self._before.get(key, 0.0)

    def total(self, name: str) -> float:
        """The same, summed over every label tuple of one sample name.

        This is the "counted exactly once" reading: a request counted twice under
        two different label sets moves no single series by two, but moves this by
        two.
        """
        now = self._read()
        keys = set(now) | set(self._before)
        return sum(now.get(key, 0.0) - self._before.get(key, 0.0) for key in keys if key[0] == name)

    def series(self, name: str) -> set[tuple[tuple[str, str], ...]]:
        """The label tuples one sample name carries right now."""
        return {key[1] for key in self._read() if key[0] == name}

    def moved_series(self, name: str) -> set[tuple[tuple[str, str], ...]]:
        """The label tuples of one sample name whose value changed during this test.

        The cardinality reading, and the reason it is "moved" rather than "new":
        a series another test already created is still one row in Prometheus's
        index, so what a test measures is how many rows its own requests touched.
        Fifty requests that touch one series added one row; fifty that touch fifty
        added fifty, whether or not any of them existed beforehand.
        """
        now = self._read()
        keys = set(now) | set(self._before)
        return {
            key[1]
            for key in keys
            if key[0] == name and now.get(key, 0.0) != self._before.get(key, 0.0)
        }


@pytest.fixture(scope="session")
def metrics_registry() -> CollectorRegistry:
    """The one registry every metrics test reads, built once by the module."""
    return registry()


@pytest.fixture
def metrics(metrics_registry: CollectorRegistry) -> MetricDelta:
    """The registry snapshotted at the start of this test, for reading deltas."""
    return MetricDelta(metrics_registry)


# --- admin --------------------------------------------------------
#
# `/admin/*` is a second key space: the admin key is not a tenant key and a
# tenant key is not an admin key (the design). So the admin app is a fourth app,
# built once from a `Settings` carrying BOTH the tenant key
# store above and an `admin_key_sha256`, because the cases that matter present a
# working tenant key to an admin route and expect a 401.
#
# `no_admin_key_client` is the shipped default: `admin_key_sha256` left empty,
# which authorises nobody, exactly as the empty `SettingsKeyStore` does.
#
# The reload cases do NOT use these two. Rewriting the policy file is the thing
# under test there, so each of those cases owns a `tmp_path` app of its own; the
# helper that builds one lives in tests/gateway/test_admin.py. That is the one
# documented exception to the session-scoped rule, and it costs a YAML parse.

ADMIN_ROUTING = "/admin/routing"
ADMIN_ROUTING_RELOAD = "/admin/routing/reload"
"""The two admin paths, spelled out here so the requests in the tests do not
read their target from the module under test. `test_the_admin_paths_and_statuses_are_pinned`
compares these literals with the module's own constants."""

ADMIN_KEY = "test-admin-key-belonging-to-nobody"
"""Test data, not a secret: worth nothing, and the store only ever holds its
sha256 digest (`sha256_hex(ADMIN_KEY)`)."""


@pytest.fixture(scope="session")
def admin_settings() -> Settings:
    """The tenant key store, the shipped policy, and a configured admin key."""
    return Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        admin_key_sha256=sha256_hex(ADMIN_KEY),
    )


@pytest.fixture(scope="session")
def admin_client(admin_settings: Settings) -> Iterator[TestClient]:
    """An app with an admin key configured, built once."""
    with TestClient(create_app(admin_settings, NO_PROVIDER_KEYS)) as client:
        yield client


@pytest.fixture(scope="session")
def no_admin_key_client() -> Iterator[TestClient]:
    """An app with no admin key configured: the shipped default, which denies."""
    settings = Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH)
    with TestClient(create_app(settings, NO_PROVIDER_KEYS)) as client:
        yield client


# --- fallback and the breaker -------------------------------------
#
# Two things the earlier stubs cannot give this plan: a SECOND adapter, and a
# failure that can be changed between cases. `StubAdapter` is built with its
# error and answers with it forever, so a file whose nine cases each want a
# different first-candidate failure would build nine apps.
#
# `FallbackStubAdapter` is that stub with its script made public and resettable,
# and it satisfies BOTH provider protocols, so one app serves the chat cases and
# the embeddings cases. Every case reaches it through `fallback_client`, which is
# the session app with two per-test resets: the stubs' recorded
# requests and scripted failures, and a FRESH breaker on `app.state`. Rebuilding
# the app per test would be the thing rule 7 forbids; a breaker is a dict and a
# lock, and it is the only state a case has to start clean.
#
# The alias is `agent-default` and the tenant `evals` under `internal`, which the
# shipped policy resolves to exactly two candidates in this order: `anthropic`
# (us) then `local` (self). `anthropic_eu` sits between them in `routing.yaml`
# and is disabled, so the pair is also proof the loop walks the RESOLVER's tuple
# rather than the policy file's.

FALLBACK_ALIAS = "agent-default"
FALLBACK_TENANT = "evals"
FALLBACK_CLASS = "internal"

FIRST_PROVIDER = "anthropic"
SECOND_PROVIDER = "local"
FIRST_REGION = "us"
SECOND_REGION = "self"

THIRD_PROVIDER = "openai"
THIRD_REGION = "us"

FALLBACK_RESOLUTION = (FIRST_PROVIDER, SECOND_PROVIDER, THIRD_PROVIDER)
"""What `resolve()` answers for the triple above with no health seam, best first.

Three since an earlier change appended an `openai` candidate to `agent-default`: `evals` is
`residency: any` and its `internal` rule allows `us`, so the new candidate is
permitted for THIS triple even under strict. Asserted against the resolver itself
by `test_the_fallback_alias_resolves_to_the_expected_candidates`, so a policy edit
that reorders it fails there."""

FALLBACK_CANDIDATES = (FIRST_PROVIDER, SECOND_PROVIDER)
"""The prefix of `FALLBACK_RESOLUTION` that tests/gateway/test_fallback.py scripts.

One stub adapter per member and no more, which is what keeps that file's cases at
"the FIRST candidate failed and the SECOND served": the route's health seam is
"an adapter exists for this provider", so `THIRD_PROVIDER` carries no stub, is
never healthy and is never reached. The pinning test asserts both tuples and the
prefix relation between them, so neither half can drift alone."""

PROVIDER_MESSAGE_SENTINEL = "upstream req_zqx_10a quota exhausted"
"""The provider's own words on every scripted failure here. A client never sees
them, so a case that asserts a body carries no provider message has
something real to look for."""


def retryable_error(provider: str) -> ProviderError:
    """The failure a fallback is allowed to act on: a 5xx, a 429 or a timeout (#16)."""
    return ProviderError(
        f"{provider}: the provider answered 503",
        retryable=True,
        provider_message=PROVIDER_MESSAGE_SENTINEL,
    )


def non_retryable_error(provider: str) -> ProviderError:
    """The failure a fallback must NOT act on: a 4xx, or a body we could not read."""
    return ProviderError(
        f"{provider}: the provider answered 400",
        retryable=False,
        provider_message=PROVIDER_MESSAGE_SENTINEL,
    )


def answer_of(provider: str) -> str:
    """The completion content one stub answers with, naming itself.

    Distinct per provider on purpose: "the body came from the SECOND candidate" is
    then a fact about the body rather than an inference from a header.
    """
    return f"the stub {provider} answered"


def fragment_of(provider: str) -> str:
    """The single text delta one stub streams, naming itself, for the same reason."""
    return f"streamed by {provider}"


class FallbackStubAdapter:
    """A stub that can chat, stream and embed, and whose failures are set per test.

    It satisfies `Adapter` and `EmbeddingAdapter` both, which is what lets one app
    serve the chat cases and the embeddings cases: the chat route's health seam
    asks whether an adapter exists and the embeddings route's asks `isinstance`,
    so a chat-only stub under `anthropic` would make the embeddings fallback
    unreachable for a reason that has nothing to do with the breaker.

    `requests` and `embed_requests` are the point: "the second adapter received
    nothing" is the assertion that separates a route which returned a
    non-retryable failure from one which tried the next candidate first and
    happened to return the same status.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []
        self.embed_requests: list[EmbeddingRequest] = []
        self.chat_error: Exception | None = None
        self.embed_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.stream_error_at = 0
        self.reset()

    def reset(self) -> None:
        """Forget every recorded call and every scripted failure.

        The per-test isolation this file's fixtures use instead of a new app.
        """
        self.requests = []
        self.embed_requests = []
        self.chat_error = None
        self.embed_error = None
        self.stream_error = None
        self.stream_error_at = 0

    @property
    def call_count(self) -> int:
        """How many times a route handed this adapter a request, of either kind."""
        return len(self.requests) + len(self.embed_requests)

    @property
    def events(self) -> tuple[StreamEvent, ...]:
        """The scripted stream, carrying one text delta that names this stub."""
        return stream_events(fragments=(fragment_of(self.name),))

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.chat_error is not None:
            raise self.chat_error
        return stub_response(content=answer_of(self.name))

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Yield the scripted events, raising the scripted error at its position.

        Position 0 is the failure the route can still turn into a typed status,
        because it happens while the first event is being pulled; anything above 0
        is a failure with a 200 already on the wire.
        """
        self.requests.append(request)
        for position, event in enumerate(self.events):
            if self.stream_error is not None and position == self.stream_error_at:
                raise self.stream_error
            yield event

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embed_requests.append(request)
        if self.embed_error is not None:
            raise self.embed_error
        return embedding_stub_response(len(request.input))

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


def app_of(client: TestClient) -> FastAPI:
    """The `FastAPI` behind a test client, with the `ASGIApp` typing narrowed."""
    app = client.app
    assert isinstance(app, FastAPI)
    return app


def fresh_breaker(client: TestClient) -> CircuitBreaker:
    """Put a new breaker on the app this client talks to, and hand it back.

    The per-test isolation for the breaker (a project rule again): the app, its
    policy and its adapters are built once for the session, and what is replaced
    per test is the one object that accumulates state.
    """
    breaker = CircuitBreaker()
    app_of(client).state.breaker = breaker
    return breaker


@pytest.fixture(scope="session")
def fallback_adapters() -> dict[str, FallbackStubAdapter]:
    """One stub per candidate of `FALLBACK_ALIAS`, keyed by policy provider name."""
    return {name: FallbackStubAdapter(name=name) for name in FALLBACK_CANDIDATES}


@pytest.fixture(scope="session")
def fallback_app(
    settings: Settings, fallback_adapters: dict[str, FallbackStubAdapter]
) -> Iterator[TestClient]:
    """The shipped app with both candidates' adapters, built once."""
    with build_chat_client(settings, fallback_adapters) as client:
        yield client


class SilentSemanticCache:
    """A `SemanticCache` that stores nothing and finds nothing, ever.

    An earlier change wires the semantic tier into the chat route, and `agent-default` -
    the alias every case in the fallback family sends - carries
    `semantic_cache: true`. This app registers an adapter under `local`, and the
    probe resolves `embed` to `local_embed`, so the tier is live here whenever
    that adapter is registered too:
    a second identical chat request would be served from it, and the loop, the
    breaker and the residency log would stop seeing the calls those families
    exist to count.

    So the tier is silenced where it is not the subject, exactly as
    `fresh_breaker` replaces the one object that accumulates state - and nothing
    is weakened by it, because the tier's own behaviour is settled end to end in
    `tests/gateway/test_semantic_cache.py`.
    """

    def search(
        self,
        tenant: str,
        vector: Sequence[float],
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> ChatResponse | None:
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
        return None


@pytest.fixture
def fallback_client(
    fallback_app: TestClient, fallback_adapters: dict[str, FallbackStubAdapter]
) -> TestClient:
    """The session app with the stubs reset, a fresh breaker and no semantic tier."""
    for adapter in fallback_adapters.values():
        adapter.reset()
    fresh_breaker(fallback_app)
    app_of(fallback_app).state.semantic_cache = SilentSemanticCache()
    return fallback_app


@pytest.fixture
def fallback_breaker(fallback_client: TestClient) -> CircuitBreaker:
    """This test's breaker, the one the routes behind `fallback_client` read."""
    breaker = app_of(fallback_client).state.breaker
    assert isinstance(breaker, CircuitBreaker)
    return breaker


# --- the residency case -------------------------------------------
#
# The shipped policy gives an EU tenant exactly ONE permitted candidate for every
# alias, because `anthropic_eu` is disabled - so it cannot express a fallback
# that has to stay inside `eu_regions`. This policy can: `EU_TENANT` is an EU
# tenant whose data class rule allows `us` explicitly, and whose alias lists a US
# provider BETWEEN two EU ones. The US provider is excluded by the residency
# clamp alone, it has an adapter registered like the others, and a fallback that
# re-resolved, widened or walked the adapter set instead of the resolver's tuple
# would reach it.
#
# Written once into a session `tmp_path`, and the app built from it once: the
# per-test reset is the same stub reset and fresh breaker as above.

EU_TENANT = "cuan"
EU_ALIAS = "agent-default"
EU_CLASS = "internal"

EU_FIRST_PROVIDER = "eu_first"
EU_SECOND_PROVIDER = "eu_second"
EU_EXCLUDED_PROVIDER = "us_only"

EU_REGION = "eu"
EU_ALLOWED_REGIONS = ("eu", "self")

EU_POLICY = f"""
providers:
  {EU_FIRST_PROVIDER}:
    kind: anthropic
    region: {EU_REGION}
    base_url: https://eu-first.invalid
  {EU_SECOND_PROVIDER}:
    kind: openai_compatible
    region: {EU_REGION}
    base_url: https://eu-second.invalid
  {EU_EXCLUDED_PROVIDER}:
    kind: openai
    region: us
    base_url: https://us-only.invalid

eu_regions: [{", ".join(EU_ALLOWED_REGIONS)}]

aliases:
  {EU_ALIAS}:
    candidates:
      - {{ provider: {EU_FIRST_PROVIDER}, model: model-eu-first }}
      - {{ provider: {EU_EXCLUDED_PROVIDER}, model: model-us-only }}
      - {{ provider: {EU_SECOND_PROVIDER}, model: model-eu-second }}

tenants:
  {EU_TENANT}:
    residency: eu
    data_class_rules:
      {EU_CLASS}: {{ allow_regions: [us, eu, self] }}
    budget_usd_month: "15.00"
    rate_limit: {{ rpm: 1000000, tpm: 1000000000 }}
"""
"""A policy an EU tenant can fall back INSIDE. The `us` in `allow_regions` is
deliberate: it makes the residency clamp the only thing keeping
`EU_EXCLUDED_PROVIDER` out of the candidate tuple.

The rate limit is a ceiling no suite can reach, on purpose. It used to read
`rpm: 60`, which was inert while nothing enforced it; per-tenant rate limits are
now enforced on every request, and a parametrised module can cross sixty requests
in well under a minute, so a realistic ceiling here would make unrelated
fallback and residency cases fail on request count. What a DEPLOYMENT runs is
pinned separately, against `routing.yaml` itself, by
`test_the_shipped_rate_limit_for_this_tenant_is_the_one_these_cases_arrange_against`
in tests/gateway/test_rate_limit_enforcement.py - so raising a fixture cannot
quietly raise production, and this file is not where limits are tested."""


@pytest.fixture(scope="session")
def eu_policy_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`EU_POLICY` on disk, written once."""
    path = tmp_path_factory.mktemp("eu-policy") / "routing.yaml"
    path.write_text(EU_POLICY, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def eu_adapters() -> dict[str, FallbackStubAdapter]:
    """A stub for all three providers, the excluded one included."""
    return {
        name: FallbackStubAdapter(name=name)
        for name in (EU_FIRST_PROVIDER, EU_SECOND_PROVIDER, EU_EXCLUDED_PROVIDER)
    }


@pytest.fixture(scope="session")
def eu_app(
    eu_policy_path: Path, eu_adapters: dict[str, FallbackStubAdapter]
) -> Iterator[TestClient]:
    """An app on the EU policy with all three adapters, built once."""
    settings = Settings(api_keys=API_KEYS, routing_policy_path=eu_policy_path)
    with build_chat_client(settings, eu_adapters) as client:
        yield client


@pytest.fixture
def eu_client(eu_app: TestClient, eu_adapters: dict[str, FallbackStubAdapter]) -> TestClient:
    """The EU app with its stubs reset and a fresh breaker on `app.state`."""
    for adapter in eu_adapters.values():
        adapter.reset()
    fresh_breaker(eu_app)
    return eu_app
