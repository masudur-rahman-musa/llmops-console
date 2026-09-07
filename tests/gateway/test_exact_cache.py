"""The exact-match cache on `POST /v1/chat/completions`: the hit, the miss, and the refusals.

Every claim here is settled by a real request through `TestClient` against an app
`create_app` built, because the question is never "does the
store work" - `tests/cache/` settles that - but "does a second identical request
reach the provider".

The witness for that is the stub adapter's `call_count`, and it is the only one
that can tell a hit from a fast miss: a header reading `hit` proves nothing on its
own, since a route could serve a fresh completion and label it. So the hit cases
assert ONE adapter call across two requests, and every refusal asserts zero.

Three arrangements are worth reading before the assertions:

* THE INELIGIBLE PAIR. `test_a_pair_with_no_temperature_both_miss` sends the same
  body twice with no `temperature` at all. It is the criterion's own boundary: an
  implementation that read an absent temperature as the provider's default zero
  would cache every request the suite has ever sent, so this case fails on the
  exact wrong implementation the eligibility table in `tests/cache/test_key.py`
  describes, one layer up.
* THE BAD HEADER. Its rows are sent on a body that is INELIGIBLE anyway, so a 422
  cannot come from the cache path deciding anything - only the header check can
  produce it - and `call_count == 0` is what places that check before the
  outbound call rather than after a served request.
* THE STREAMED LOOKUP. `test_a_streamed_request_is_not_served_from_a_stored_answer`
  fills the cache with a NON-streamed request first and then sends the identical
  body with `stream: true`. The key does not carry `stream` (the design's seven
  fields do not include it), so the entry is genuinely there to be found, and the
  only thing that can keep the stream off it is the eligibility rule (decision G).

a project rule: one app for the module, built once. Per-test isolation is the
stub's reset and a fresh `InProcessResponseCache` on `app.state`, exactly as
`fresh_breaker` in this directory's `conftest.py` replaces the one object that
accumulates state rather than rebuilding the app.

The cache is reached by NAME, `app.state.response_cache`, never by shape: this
directory now has three collaborators with a `put` and a `get`, and selecting one
of them by its methods is what a project rule exists for.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from opsconsole.cache import (
    CACHE_ALLOW,
    CACHE_HEADER,
    InProcessResponseCache,
    RedisResponseCache,
)
from opsconsole.core.settings import Settings
from opsconsole.gateway.app import close_adapters, create_app
from opsconsole.providers.base import ChatRequest, ChatResponse, StreamEvent

from .conftest import (
    API_KEYS,
    CHAT,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    MetricDelta,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    stream_events,
    stub_response,
)

TENANT = "evals"
DATA_CLASS = "internal"
ALIAS = "agent-default"
PROVIDER = "anthropic"
REGION = "us"

HIT = "hit"
MISS = "miss"

CACHE_SPAN = "cache"
"""the design's stage name for the lookup. `tests/gateway/test_tracing.py` used to
assert this span is NEVER emitted; step 4 emits it, so the positive claim lives
here and that file keeps the `ledger` half (plan §8)."""

REQUESTS = "gateway_requests_total"
TOKENS = "gateway_tokens_total"
CACHE_HITS = "gateway_cache_hits_total"

EXACT_TIER = "exact"
"""The value of `gateway_cache_hits_total`'s `kind` label for this tier (SPEC
§13.2). An earlier change adds the semantic one when it wires that tier into the route,
and the label NAME is `kind` rather than `tier`, which is what the family was
declared with while it carried no samples."""

ANSWER = "Berlin is the capital of Germany."
TOOL_BODY_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
)

CACHE_FAILURE_TEXT = "Error 111 connecting to redis:6379. Connection refused."
"""What a broken cache says when it is asked. A marker, so a route that let the
failure escape shows the internal host and port in a body a caller can read."""


class CacheStub:
    """The adapter behind this module's app: a fixed answer, and every call recorded.

    `call_count` is the load-bearing part. Every case in this file is about
    whether a provider was dialled a second time, and no header, body or metric
    can answer that - only the adapter can.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []
        self.response: ChatResponse = stub_response(content=ANSWER)
        self.events: tuple[StreamEvent, ...] = stream_events()

    def reset(self) -> None:
        """Forget every recorded call and go back to the default answer."""
        self.requests = []
        self.response = stub_response(content=ANSWER)
        self.events = stream_events()

    @property
    def call_count(self) -> int:
        """How many times a route handed this adapter a completion to run."""
        return len(self.requests)

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.response

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        yield from self.events

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


class BrokenResponseCache:
    """A cache whose every operation fails the way an unreachable Redis does.

    Deliberately NOT the shipped `RedisResponseCache` pointed at a dead port: that
    one swallows its own failures, so it could not tell whether the ROUTE survives
    a store that raises. The two cases using this class are about the route.
    """

    def get(self, tenant: str, key: str) -> ChatResponse | None:
        raise RuntimeError(CACHE_FAILURE_TEXT)

    def put(self, tenant: str, key: str, answer: ChatResponse) -> None:
        raise RuntimeError(CACHE_FAILURE_TEXT)


@pytest.fixture(scope="module")
def cache_stub() -> CacheStub:
    """The one adapter behind this module's app, under the policy provider `anthropic`."""
    return CacheStub(name=PROVIDER)


@pytest.fixture(scope="module")
def cache_client(settings: Settings, cache_stub: CacheStub) -> Iterator[TestClient]:
    """One app for the module, built once."""
    with build_chat_client(settings, {cache_stub.name: cache_stub}) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh_cache(cache_client: TestClient, cache_stub: CacheStub) -> Iterator[None]:
    """Per-test isolation: an empty cache and a reset stub, never a new app."""
    cache_stub.reset()
    app_of(cache_client).state.response_cache = InProcessResponseCache()
    yield


def post(
    client: TestClient,
    body: Mapping[str, Any],
    *,
    cache_header: str | None = None,
    key: str = TEST_KEYS[TENANT],
    tenant: str = TENANT,
) -> Response:
    """POST one completion as `evals` under `internal`, with the header only if given.

    `cache_header=None` sends NO `X-Gateway-Cache` request header, which is the
    ordinary request: the opt-in is the header's PRESENCE with the one accepted
    word, and there is no value that means "do not cache".
    """
    headers = auth_headers(key=key, tenant=tenant, data_class=DATA_CLASS)
    if cache_header is not None:
        headers[CACHE_HEADER] = cache_header
    response: Response = client.post(CHAT, json=dict(body), headers=headers)
    return response


def eligible_body(**overrides: object) -> dict[str, Any]:
    """The baseline cacheable body: `temperature: 0` stated explicitly (#22)."""
    return chat_body(model=ALIAS, temperature=0, **overrides)


def content_of(response: Response) -> str:
    """The assistant text of a completion, with a 200 asserted."""
    assert response.status_code == 200, response.text
    content: str = response.json()["choices"][0]["message"]["content"]
    return content


def cache_header_of(response: Response) -> str:
    """The `X-Gateway-Cache` value the answer carried, lowercased header name."""
    value: str = response.headers["x-gateway-cache"]
    return value


# --------------------------------------------------------------------------
# the hit
# --------------------------------------------------------------------------


def test_a_second_identical_eligible_request_calls_no_adapter(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """The plan's first criterion. Catches a cache that stores nothing, one that stores
    under a key it cannot look up again (a per-request salt, a key built from the
    REDACTED body on the way in and the inbound body on the way out), and one that looks
    up after the provider call rather than before. `call_count == 1` across two requests
    is the only assertion that separates those from a working cache: the header alone
    would pass on a route that served a fresh completion and labelled it `hit`."""
    first = post(cache_client, eligible_body())
    second = post(cache_client, eligible_body())

    assert content_of(first) == ANSWER
    assert content_of(second) == ANSWER
    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == HIT
    assert cache_stub.call_count == 1


def test_a_hit_carries_its_own_completion_id_and_the_same_answer(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """Catches decision E gone the other way - the WIRE body cached verbatim: replaying it
    replays another request's `chatcmpl-` id, which is the id the ledger and the trace
    store key their rows by, so two requests would merge into one row. The equal content
    is what stops this passing on a route that answers something else entirely."""
    first = post(cache_client, eligible_body())
    second = post(cache_client, eligible_body())

    first_body = first.json()
    second_body = second.json()

    assert content_of(second) == content_of(first)
    assert second_body["id"] != first_body["id"]
    assert second_body["model"] == ALIAS
    assert second_body["usage"] == first_body["usage"]
    assert cache_stub.call_count == 1


def test_a_hit_still_carries_the_six_gateway_headers(cache_client: TestClient) -> None:
    """Catches a hit returned before the header set is built, which would leave an
    operator unable to say which provider and region the cached answer came from - the
    one question #9's headers exist to answer - and would change the published header
    contract depending on whether a request happened to hit."""
    post(cache_client, eligible_body())
    hit = post(cache_client, eligible_body())

    assert hit.status_code == 200, hit.text
    gateway = {name.lower(): value for name, value in hit.headers.items()}
    named = {name for name in gateway if name.startswith("x-gateway-")}

    assert len(named) == 6
    assert gateway["x-gateway-provider"] == PROVIDER
    assert gateway["x-gateway-region"] == REGION
    assert gateway["x-gateway-cache"] == HIT
    assert "x-gateway-cost-usd" not in gateway


# --------------------------------------------------------------------------
# what is not eligible
# --------------------------------------------------------------------------


def test_a_pair_with_no_temperature_both_miss(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """The criterion's boundary, at the route. Catches an implementation that reads an
    ABSENT temperature as zero (`body.temperature or 0`, a wire default of 0.0): every
    request this suite has ever sent omits the field, so such a cache would serve stored
    answers to requests nobody opted in for, and every other case in this file would still
    pass. Two adapter calls is the assertion; the two `miss` headers alone would also be
    produced by a cache that stored nothing at all, which is why `call_count` is here."""
    first = post(cache_client, chat_body(model=ALIAS))
    second = post(cache_client, chat_body(model=ALIAS))

    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == MISS
    assert cache_stub.call_count == 2


def test_the_allow_header_makes_an_otherwise_ineligible_pair_cacheable(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """Catches an eligibility rule that reads the temperature and ignores the opt-in: #22
    gives a caller a second way in, and without this row the header would be a word the
    gateway validates and never acts on. Paired with the case above - same body, one
    header apart - so only the header can decide the outcome."""
    first = post(cache_client, chat_body(model=ALIAS), cache_header=CACHE_ALLOW)
    second = post(cache_client, chat_body(model=ALIAS), cache_header=CACHE_ALLOW)

    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == HIT
    assert cache_stub.call_count == 1


def test_tools_with_a_live_tool_choice_are_never_stored(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """Decision H at the route. Catches a cache that ignores the tool rule: a tool call's
    result varies per call, so the answer is not a function of the request, and replaying
    yesterday's call is a wrong answer that looks perfectly well formed. `temperature: 0`
    is stated, so eligibility is refused by the tool rule alone."""
    body = eligible_body(tools=list(TOOL_BODY_TOOLS), tool_choice="auto")

    first = post(cache_client, body)
    second = post(cache_client, body)

    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == MISS
    assert cache_stub.call_count == 2


def test_tools_offered_with_tool_choice_none_are_still_cacheable(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """The other half of decision H, and the row that stops "tools present, never cache"
    from passing the case above: `tool_choice: none` means the model may not call
    anything, so the answer is a function of the request again. Without this row the
    tool rule could be a blanket refusal nobody would notice."""
    body = eligible_body(tools=list(TOOL_BODY_TOOLS), tool_choice="none")

    first = post(cache_client, body)
    second = post(cache_client, body)

    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == HIT
    assert cache_stub.call_count == 1


def test_a_streamed_pair_is_never_stored(cache_client: TestClient, cache_stub: CacheStub) -> None:
    """Decision G. Catches a route that stores the assembled answer of a stream: the
    stored value would then be served to a NON-streamed request that shares its key, and
    #22 asks for a response cache rather than frame replay. Both frames sets are pulled,
    so the store-on-serve step really did run."""
    body = eligible_body(stream=True)

    first = post(cache_client, body)
    second = post(cache_client, body)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == MISS
    assert cache_stub.call_count == 2


def test_a_streamed_request_is_not_served_from_a_stored_answer(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """The sharper half of decision G, and the arrangement that makes it decidable: the
    non-streamed request FILLS the cache under a key the streamed one computes identically
    (the design's seven fields carry no `stream`), so the entry is genuinely there. Catches a
    lookup that runs before the eligibility check, which would answer a JSON completion to
    a caller waiting for `text/event-stream`."""
    filled = post(cache_client, eligible_body())
    assert content_of(filled) == ANSWER

    streamed = post(cache_client, eligible_body(stream=True))

    assert streamed.status_code == 200, streamed.text
    assert streamed.headers["content-type"].split(";")[0] == "text/event-stream"
    assert cache_header_of(streamed) == MISS
    assert cache_stub.call_count == 2


# --------------------------------------------------------------------------
# the request header (decision D)
# --------------------------------------------------------------------------

REJECTED_HEADER_VALUES = [
    pytest.param("yes", id="a word that is not the accepted one"),
    pytest.param("true", id="the boolean a caller would guess"),
    pytest.param("no", id="a value a caller would read as opting out"),
    pytest.param("ALLOW", id="the accepted word in uppercase"),
    pytest.param("Allow", id="the accepted word capitalised"),
    pytest.param("allow ", id="the accepted word with a trailing space"),
    pytest.param(" allow", id="the accepted word with a leading space"),
    pytest.param("allow,allow", id="the accepted word sent twice in one value"),
    pytest.param("", id="an empty header value"),
]


@pytest.mark.parametrize("value", REJECTED_HEADER_VALUES)
def test_a_cache_header_that_is_not_the_accepted_word_is_422(
    value: str, cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """Decision D. Catches a header compared with `in`, `startswith`, `.strip()` or
    `.lower()` - each row is one of those - and catches the decision going the other way,
    a typo silently meaning "do not cache", which is the failure a caller cannot see. The
    body is INELIGIBLE without the header, so nothing but the header check can produce a
    422 here, and `call_count == 0` is what places that check before any outbound call
    rather than after a request the caller has already paid for."""
    response = post(cache_client, chat_body(model=ALIAS), cache_header=value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == CACHE_HEADER
    assert cache_stub.call_count == 0


def test_the_accepted_word_is_not_refused(cache_client: TestClient, cache_stub: CacheStub) -> None:
    """The row that stops the table above being satisfied by a check that refuses every
    value: exactly `allow` is served, and it is the same word `CACHE_ALLOW` publishes."""
    response = post(cache_client, chat_body(model=ALIAS), cache_header=CACHE_ALLOW)

    assert response.status_code == 200, response.text
    assert cache_stub.call_count == 1


# --------------------------------------------------------------------------
# a cache that fails
# --------------------------------------------------------------------------


def test_a_cache_that_raises_still_serves_the_request(
    cache_client: TestClient, cache_stub: CacheStub
) -> None:
    """The plan's unhappy path: Redis unreachable is a MISS and an ordinary provider call,
    never a 5xx. Catches a route that calls the store outside a guard - the lookup happens
    before the provider call, so an escaping failure would take out every chat request in
    the deployment that has Redis, and the failing PUT happens after a provider has
    already answered, which loses a request that was fully served and billed. The second
    request proves the write path was reached at all."""
    app_of(cache_client).state.response_cache = BrokenResponseCache()

    first = post(cache_client, eligible_body())
    second = post(cache_client, eligible_body())

    assert content_of(first) == ANSWER
    assert content_of(second) == ANSWER
    assert cache_header_of(first) == MISS
    assert cache_header_of(second) == MISS
    assert cache_stub.call_count == 2
    assert CACHE_FAILURE_TEXT not in first.text


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


def test_a_miss_is_labelled_miss_and_moves_no_hit_counter(
    cache_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a `cache` label hardcoded to `hit` once the family is fed, and a hit counter
    incremented on every request: the miss half is what makes the hit half mean something,
    because a counter that moves on both is a panel that reports a 100% hit rate."""
    response = post(cache_client, eligible_body())
    assert response.status_code == 200, response.text

    assert response.headers["x-gateway-cache"] == MISS
    assert metrics.delta(REQUESTS, **request_labels(cache=MISS)) == 1
    assert metrics.total(CACHE_HITS) == 0


def test_a_hit_counts_a_cache_hit_and_labels_the_request_hit(
    cache_client: TestClient, metrics_registry: CollectorRegistry
) -> None:
    """Catches a hit counted under the label name the family was DECLARED with rather than
    the one the design publishes (`tier` for `kind`), which would leave every dashboard
    query written against the spec reading nothing; catches a hit that never reaches
    `gateway_cache_hits_total` at all; and catches a `cache` label left at `miss` on the
    served row, which is the label the savings panel of #24 sums. The delta is taken
    AFTER the warming request, so only the second request's emissions are measured."""
    assert post(cache_client, eligible_body()).status_code == 200

    metrics = MetricDelta(metrics_registry)
    hit = post(cache_client, eligible_body())

    assert hit.status_code == 200, hit.text
    assert hit.headers["x-gateway-cache"] == HIT
    assert metrics.delta(CACHE_HITS, tenant=TENANT, kind=EXACT_TIER) == 1
    assert metrics.total(CACHE_HITS) == 1
    assert metrics.delta(REQUESTS, **request_labels(cache=HIT)) == 1


def test_a_hit_books_no_tokens(
    cache_client: TestClient, metrics_registry: CollectorRegistry
) -> None:
    """Decision F. Catches a hit that re-books the tokens the ORIGINAL call reported:
    `gateway_tokens_total` is what the providers billed, so a cache hit - which bought
    nothing - would inflate cost reporting by exactly the traffic the cache was added to
    save. The delta starts after the warming request, so the first call's real tokens are
    outside the measurement and only the hit is inside it."""
    assert post(cache_client, eligible_body()).status_code == 200

    metrics = MetricDelta(metrics_registry)
    hit = post(cache_client, eligible_body())

    assert hit.status_code == 200, hit.text
    assert hit.headers["x-gateway-cache"] == HIT
    assert metrics.total(TOKENS) == 0


# --------------------------------------------------------------------------
# the stage span (the design)
# --------------------------------------------------------------------------


def test_the_cache_span_is_emitted_on_a_miss_and_on_a_hit(
    cache_client: TestClient, spans: InMemorySpanExporter
) -> None:
    """The claim `tests/gateway/test_tracing.py` used to make in reverse (plan §8). Catches
    a lookup with no span - an operator reading a trace could not tell whether the cache
    was consulted or how long it took - and catches a span emitted on one path only, which
    would make every trace that carries one a hit by construction. Both requests are read
    from the same buffer in order, so a span left by the first cannot stand in for the
    second."""
    miss = post(cache_client, eligible_body())
    assert miss.headers["x-gateway-cache"] == MISS
    miss_names = {span.name for span in spans.get_finished_spans()}

    spans.clear()
    hit = post(cache_client, eligible_body())
    assert hit.headers["x-gateway-cache"] == HIT
    hit_names = {span.name for span in spans.get_finished_spans()}

    assert CACHE_SPAN in miss_names
    assert CACHE_SPAN in hit_names


# --------------------------------------------------------------------------
# the wiring (plan decision C)
# --------------------------------------------------------------------------

UNCONNECTED_REDIS_URL = "redis://127.0.0.1:1/0"
"""A Redis nothing listens on. `redis.Redis` connects lazily, so building the
cache costs no connection and this URL is never dialled by the two cases below."""


def wiring_settings(*, redis_url: str | None) -> Settings:
    """The test key store and the shipped policy, with the Redis URL stated either way.

    Stated rather than left to default: `Settings` reads `OPSCONSOLE_REDIS_URL`
    from the process environment, which the Redis suite asks a developer to
    export, and the unconfigured row would then assert the opposite of what it
    says.
    """
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, redis_url=redis_url)


def test_create_app_keeps_the_in_process_cache_without_a_redis_url() -> None:
    """Catches a factory that requires Redis to start: the whole existing suite, `make
    verify` and a bare `uvicorn` run configure no `redis_url`, so a Redis-only factory
    turns every one of them into a connection error at process start rather than a miss."""
    app = create_app(wiring_settings(redis_url=None), NO_PROVIDER_KEYS)

    assert isinstance(app.state.response_cache, InProcessResponseCache)
    close_adapters(app.state.adapters)


def test_create_app_uses_the_redis_cache_when_a_redis_url_is_configured() -> None:
    """Catches a factory that files responses in process memory while Redis is configured:
    two gateway replicas would each warm their own cache, halving the hit rate #24 reports
    for a reason no test with one replica can see except at the wiring."""
    app = create_app(wiring_settings(redis_url=UNCONNECTED_REDIS_URL), NO_PROVIDER_KEYS)

    assert isinstance(app.state.response_cache, RedisResponseCache)
    close_adapters(app.state.adapters)
