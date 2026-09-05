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

from opsconsole.core.settings import Settings
from opsconsole.gateway.app import create_app
from opsconsole.gateway.auth import MAX_KEY_LENGTH
from opsconsole.gateway.context import RequestContext, request_context
from opsconsole.providers.base import (
    Adapter,
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageTotal,
)
from opsconsole.routing.errors import ResidencyUnavailable, RoutingError
from opsconsole.routing.policy import DataClass, Policy, load_policy
from opsconsole.routing.resolver import MAX_IDENTIFIER_LENGTH, resolve

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
