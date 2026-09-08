"""The scrape: what `GET /metrics` exposes, and what one request adds to it.

Everything here goes through `TestClient` against an app `create_app` built, so
every claim about a counter is settled by a real request rather than by reading
the emission site. The samples are read out of the
session-scoped registry through the `metrics` fixture in `conftest.py`, which
reports DELTAS: a registry is process-wide and a family cannot be taken back, so
isolation is the difference a test makes rather than a registry of its own.

The contract these tests are written against, before it exists:

* `opsconsole.core.metrics.registry()` - the module-owned `CollectorRegistry`
  holding this gateway's families and nothing else (decision B).
* `opsconsole.core.metrics.request_metrics(scope)` - the accessor for the
  `RequestMetrics` record carried in the ASGI scope beside `RequestTrace`, filled
  where `annotate` is filled and read once by the middleware after the last byte
  (decision E). `None` when nothing started one.
* `opsconsole.gateway.routes.metrics` - `GET /metrics`, unauthenticated
  (decision D), answering the Prometheus text exposition format.

The eleven family names below are the design's, spelled as the exposition spells
them: a counter's family is its `_total` series, which is also the name a
dashboard query uses. `gateway_eval_*` is deliberately absent (decision C): the
nightly job pushes those gauges to a Pushgateway, so this process never declares
them.

Two things are deliberately NOT asserted here. The label NAMES of the three unfed
families, which have no samples to read them off and no public accessor that
would not pin an implementation detail; and the resolved provider model id, which
lives in `routing.yaml` - the token-label test compares the label against the
`X-Gateway-Model` header the same request answered, so the two cannot drift and
neither is copied from the policy.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import cast

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from httpx import Response as HttpxResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry

from opsconsole.core.errors import ProviderError
from opsconsole.core.metrics import request_metrics
from opsconsole.core.settings import Settings
from opsconsole.gateway.app import create_app
from opsconsole.gateway.routes.redact import REDACT
from opsconsole.providers.base import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    UsageTotal,
)
from opsconsole.providers.breaker import MAX_FAILURES as BREAKER_MAX_FAILURES
from opsconsole.routing.resolver import MAX_IDENTIFIER_LENGTH

from .conftest import (
    CHAT,
    FALLBACK_ALIAS,
    FALLBACK_CLASS,
    FALLBACK_TENANT,
    FIRST_PROVIDER,
    NO_PROVIDER_KEYS,
    SECOND_PROVIDER,
    STREAM_FRAGMENTS,
    STREAM_USAGE,
    STUB_EMBEDDING_USAGE,
    STUB_USAGE,
    TEST_KEYS,
    FallbackStubAdapter,
    MetricDelta,
    StubAdapter,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    embeddings_body,
    post_chat,
    post_embeddings,
    retryable_error,
    stub_response,
)

METRICS = "/metrics"
HEALTHZ = "/healthz"
READYZ = "/readyz"

# --- the eleven declared families (the design, decision C) --------------------

REQUESTS = "gateway_requests_total"
LATENCY = "gateway_request_latency_seconds"
TOKENS = "gateway_tokens_total"
REDACTIONS = "gateway_redactions_total"
PROVIDER_ERRORS = "gateway_provider_errors_total"
RESIDENCY_VIOLATIONS = "gateway_residency_violations_total"
FALLBACKS = "gateway_fallbacks_total"
CIRCUIT_STATE = "gateway_circuit_state"
COST_USD = "gateway_cost_usd_total"

LATENCY_BUCKET = f"{LATENCY}_bucket"
LATENCY_COUNT = f"{LATENCY}_count"
LATENCY_SUM = f"{LATENCY}_sum"

FED_FAMILIES = (
    REQUESTS,
    LATENCY,
    TOKENS,
    REDACTIONS,
    PROVIDER_ERRORS,
    FALLBACKS,
    CIRCUIT_STATE,
    COST_USD,
)
"""The families a request through a route in this file feeds. An earlier change took
`gateway_residency_violations_total` OUT of this tuple: under
decision E a 451 no longer moves it, and nothing else a route can do moves it
either, so a case asserting it carries samples after a request would be asserting
the behaviour that plan removed."""

CACHE_HITS = "gateway_cache_hits_total"

UNFED_FAMILIES = ("gateway_budget_ratio",)
"""The families still DECLARED AND EMPTY, which shrinks by one plan's worth each
time a subsystem lands: an earlier change filled `gateway_fallbacks_total` and
`gateway_circuit_state`, an earlier change filled `gateway_cache_hits_total`, an earlier change
filled `gateway_cost_usd_total`, and each moved its own out of here (standing
rule 33). The count below does not change - a plan here adds a value, never a
family."""

ELSEWHERE_FED_FAMILIES = (RESIDENCY_VIOLATIONS, CACHE_HITS)
"""Declared, empty after every request THIS file makes, and fed elsewhere in the
session - so neither tuple above can hold them.

`gateway_residency_violations_total` is fed by exactly one thing, an
`allowed=false` residency log row the resolver cannot produce (an earlier change decision
E); tests/gateway/test_residency_log.py writes one directly.
`gateway_cache_hits_total` is fed by a route as of an earlier change, but only by a
request eligible for the cache - a stated `temperature: 0` or the
`X-Gateway-Cache: allow` header - and no request in this file is either, which is
why every `cache` label asserted here reads `miss`.
tests/gateway/test_exact_cache.py is where its samples come from.

The distinction matters because the emptiness case below reads the PROCESS
registry: a family another module fed would fail it for a reason that has
nothing to do with the claim."""

DECLARED_FAMILIES = (
    frozenset(FED_FAMILIES) | frozenset(UNFED_FAMILIES) | frozenset(ELSEWHERE_FED_FAMILIES)
)

DECLARED_FAMILY_COUNT = 11
"""Seven fed by a request in this file, two declared and empty, and two fed only
from elsewhere in the session. The plan's own number, so a family added or
dropped fails here as well as in the set comparison."""

# The `le` values decision H fixes, in the order and the spelling
# prometheus_client renders them with, `+Inf` included. The default histogram
# tops out at 10s, which puts every ordinary streamed completion in the last
# bucket and makes a p95 panel unreadable.
DECLARED_BUCKET_BOUNDS = (
    "0.1",
    "0.25",
    "0.5",
    "1.0",
    "2.0",
    "5.0",
    "10.0",
    "20.0",
    "30.0",
    "60.0",
    "120.0",
    "+Inf",
)

# The prefixes prometheus_client's GLOBAL registry arrives carrying. Decision B
# is exactly the decision not to have them.
PLATFORM_COLLECTOR_PREFIXES = ("python_", "process_", "gc_")

# --- what the shipped policy resolves for the requests below -----------------

ALIAS = "agent-default"
EMBED_ALIAS = "embed"
TENANT = "evals"
REMOTE_PROVIDER = "anthropic"
REMOTE_REGION = "us"
LOCAL_PROVIDER = "local"
LOCAL_REGION = "self"
PLAIN_CLASS = "internal"
REDACT_CLASS = "confidential"

CACHE_MISS = "miss"
"""What `X-Gateway-Cache` says on a request the cache did not serve, and therefore
the `cache` label those requests carry. Every request in this file is one: none
states `temperature: 0` and none sends the `X-Gateway-Cache: allow` header, so
none is eligible under an earlier change. The hit label is covered in
tests/gateway/test_exact_cache.py."""

EMPTY = ""
"""Decision F: a label a request never got. Not a `none` sentinel and not a
dropped request, so the counter still reconciles with the access log."""

EMAIL = "alice@example.com"
OTHER_EMAIL = "bob@example.com"
IPV4 = "198.51.100.7"
TEXT_WITH_TWO_KINDS = f"write to {EMAIL} from {IPV4} please"

# 50 aliases the policy does not define, the last of them exactly
# `MAX_IDENTIFIER_LENGTH` characters: the inclusive boundary row of the
# identifier pre-check that decides whether a client string may become a label.
UNKNOWN_ALIAS_COUNT = 50
UNKNOWN_ALIASES = [f"unknown-alias-{index}" for index in range(UNKNOWN_ALIAS_COUNT - 1)] + [
    "u" * MAX_IDENTIFIER_LENGTH
]
OVER_CAP_ALIAS = "u" * (MAX_IDENTIFIER_LENGTH + 1)


# --- reading the exposition --------------------------------------------------


def scrape(client: TestClient) -> HttpxResponse:
    """One unauthenticated scrape of `/metrics`."""
    response: HttpxResponse = client.get(METRICS)
    return response


def type_names(body: str) -> set[str]:
    """Every family a scraper would see, read off the `# TYPE` lines.

    The `_created` series prometheus_client mints beside a counter are dropped:
    they are a child's birth timestamp, not a family of their own, and a scraper
    reads them as part of the counter they belong to.
    """
    names = {line.split()[2] for line in body.splitlines() if line.startswith("# TYPE ")}
    return {name for name in names if not name.endswith("_created")}


def sample_names(body: str) -> set[str]:
    """Every series name that actually carries a value in this body."""
    return {
        line.split("{")[0].split(" ")[0]
        for line in body.splitlines()
        if line and not line.startswith("#")
    }


def bucket_bounds(metrics: MetricDelta, *, alias: str, provider: str) -> list[str]:
    """The `le` values the latency histogram carries for one alias and provider."""
    bounds: list[str] = []
    for series in metrics.series(LATENCY_BUCKET):
        labels = dict(series)
        if labels.get("alias") == alias and labels.get("provider") == provider:
            bounds.append(labels["le"])
    return sorted(bounds, key=float)


def request_labels(
    *,
    tenant: str,
    alias: str,
    provider: str,
    region: str,
    status: str,
    cache: str,
) -> dict[str, str]:
    """The six labels the design gives `gateway_requests_total`, in one place."""
    return {
        "tenant": tenant,
        "alias": alias,
        "provider": provider,
        "region": region,
        "status": status,
        "cache": cache,
    }


def family_names(collector_registry: CollectorRegistry) -> set[str]:
    """The families registered right now, whether or not they carry a sample."""
    return {family.name for family in collector_registry.collect()}


# --- clients -----------------------------------------------------------------


@pytest.fixture(scope="module")
def local_chat_stub() -> StubAdapter:
    """A chat stub under `local`, the one provider a `confidential` request reaches."""
    return StubAdapter(name=LOCAL_PROVIDER, response=stub_response())


@pytest.fixture(scope="module")
def local_chat_client(settings: Settings, local_chat_stub: StubAdapter) -> Iterator[TestClient]:
    """One app for the inline-redaction rows, built once."""
    with build_chat_client(settings, {local_chat_stub.name: local_chat_stub}) as client:
        yield client


@pytest.fixture(scope="module")
def failing_client(settings: Settings) -> Iterator[TestClient]:
    """An app whose only adapter fails every call with a typed provider error."""
    stub = StubAdapter(
        name=REMOTE_PROVIDER,
        error=ProviderError(
            "anthropic: the provider answered 503",
            retryable=True,
            provider_message="upstream request id req_zqx_847 quota exhausted",
        ),
    )
    with build_chat_client(settings, {stub.name: stub}) as client:
        yield client


class SlowStreamAdapter:
    """A streaming adapter that pauses BETWEEN frames and not before the first.

    The route pulls the first event before it answers 200, so a pause in front of
    it would be counted by a latency observation taken at the route's return as
    well as by one taken after the last frame. Every pause here happens after that
    first event, which is what makes the two implementations land in different
    buckets instead of in the same one.
    """

    FRAME_DELAY_SECONDS = 0.05

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []

    @property
    def delayed_frames(self) -> int:
        """How many pauses one streamed answer contains."""
        return len(STREAM_FRAGMENTS) + 1

    @property
    def answer_seconds(self) -> float:
        """The least time one streamed answer can take, on the wire."""
        return self.FRAME_DELAY_SECONDS * self.delayed_frames

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
    """One app for the streamed latency row, built once."""
    with build_chat_client(settings, {slow_stream_adapter.name: slow_stream_adapter}) as client:
        yield client


# --- the poisoned emission ---------------------------------------------------
#
# The failing-emission case needs an emission that raises without touching the
# implementation. `ExplodingLabel` is a label value whose `str()` raises, and the
# harness route below writes one onto the request's OWN scope record - the object
# the call sites fill and the middleware reads - so whatever the middleware does
# with it at the end of the request raises there.
#
# The route is mounted under `/v1/` because that is the traffic decision G counts:
# on any other path there would be no emission to fail and the case would pass
# whether or not the guard exists.

POISON_PATH = "/v1/_harness/poisoned-metrics"
POISON_HEADER = "X-Harness-Poisoned"
POISON_HEADER_VALUE = "yes"
POISON_BODY = {"poisoned": True}


class ExplodingLabel:
    """A label value that cannot be rendered. Every read of it as text raises."""

    def __str__(self) -> str:
        raise RuntimeError("this label value cannot be rendered")

    def __repr__(self) -> str:
        raise RuntimeError("this label value cannot be rendered")


@pytest.fixture(scope="module")
def poison_client(settings: Settings) -> Iterator[TestClient]:
    """An app with one `/v1/` route that poisons its own metrics record."""
    app: FastAPI = create_app(settings, NO_PROVIDER_KEYS)

    @app.get(POISON_PATH)
    def poisoned_metrics(request: Request, response: Response) -> dict[str, bool]:
        """Put an unrenderable value on this request's record, and answer normally."""
        response.headers[POISON_HEADER] = POISON_HEADER_VALUE
        record = request_metrics(request.scope)
        if record is None:
            return {"poisoned": False}
        record.tenant = cast(str, ExplodingLabel())
        return {"poisoned": True}

    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------
# the endpoint and the family table
# --------------------------------------------------------------------------


def test_metrics_answers_the_exposition_content_type(gateway_client: TestClient) -> None:
    """Catches a route that returns the samples as JSON, or as a bare `text/plain`
    without the version parameter: a Prometheus scrape reads the content type before the
    body and drops a target it cannot parse, so every panel in the design would be empty
    while the endpoint answered 200 to a browser. Also catches an authenticated
    `/metrics` (decision D), since this scrape sends no key at all."""
    response = scrape(gateway_client)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert f"# TYPE {REQUESTS} counter" in response.text


def test_the_exposed_family_names_are_exactly_the_declared_set(
    gateway_client: TestClient,
) -> None:
    """Catches both directions of the closed table: a family this gateway never declared
    (prometheus_client's global registry arrives carrying process, GC and platform
    collectors, which is decision B) and a declared family that never reached the
    registry - which is the failure an earlier change would each discover as a missing
    family rather than as a missing value."""
    body = scrape(gateway_client).text

    assert type_names(body) == set(DECLARED_FAMILIES)
    assert len(DECLARED_FAMILIES) == DECLARED_FAMILY_COUNT
    for name in type_names(body):
        assert not name.startswith(PLATFORM_COLLECTOR_PREFIXES), name


def test_the_unfed_families_are_registered_and_carry_no_samples(
    gateway_client: TestClient, chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Names the family an earlier change fills in, and asserts both halves. Catches a plan that
    has to ADD the family rather than a value (the half a set comparison already sees),
    and catches a placeholder emission - a zero ratio published on every request - which
    would draw a confident flat line on a dashboard for a subsystem that does not exist
    yet. The served request is what makes the second half mean something: samples were
    being written while this one stayed empty.

    It was two until an earlier change fed `gateway_cost_usd_total` from the route, which is why
    that name is in `FED_FAMILIES` above now;
    `gateway_cache_hits_total` was a third until an earlier change, and its emptiness is now a
    claim about ELIGIBILITY rather than about a missing subsystem, asserted where that
    claim belongs (tests/gateway/test_exact_cache.py)."""
    assert post_chat(chat_client, chat_body(model=ALIAS)).status_code == 200

    body = scrape(gateway_client).text

    assert set(UNFED_FAMILIES) <= type_names(body)
    for name in UNFED_FAMILIES:
        assert metrics.series(name) == set(), name
        assert name not in sample_names(body), name
    assert metrics.total(REQUESTS) == 1, "no request was counted, so the emptiness proves nothing"


# --------------------------------------------------------------------------
# one request, one count
# --------------------------------------------------------------------------


def test_a_chat_request_is_counted_once_with_every_label(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a counter labelled from the request body instead of from the resolver
    (`provider` would read `agent-default`, and the one question the dashboard asks -
    which provider served this - would be unanswerable), a request counted at route entry
    AND at the end (every RPS panel doubled), and a hardcoded `status` label. The
    provider, region and cache labels are compared against the headers the SAME response
    carried, so the counter and the wire contract cannot disagree."""
    response = post_chat(chat_client, chat_body(model=ALIAS))
    assert response.status_code == 200, response.text

    labels = request_labels(
        tenant=TENANT,
        alias=ALIAS,
        provider=response.headers["x-gateway-provider"],
        region=response.headers["x-gateway-region"],
        status="200",
        cache=response.headers["x-gateway-cache"],
    )

    assert labels["provider"] == REMOTE_PROVIDER
    assert labels["region"] == REMOTE_REGION
    assert labels["cache"] == CACHE_MISS
    assert metrics.delta(REQUESTS, **labels) == 1
    assert metrics.total(REQUESTS) == 1
    assert metrics.delta(LATENCY_COUNT, alias=ALIAS, provider=REMOTE_PROVIDER) == 1
    assert metrics.total(LATENCY_COUNT) == 1


PROBE_PATHS = [HEALTHZ, READYZ, METRICS]


@pytest.mark.parametrize("path", PROBE_PATHS)
def test_the_probe_and_scrape_paths_are_not_counted(
    path: str, gateway_client: TestClient, chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a middleware that counts every HTTP scope. A scrape every fifteen seconds
    and a kubelet probe are not traffic: counted, they put a constant floor under every
    RPS panel and an unmovable denominator under every error ratio in the design's
    dashboard 1, so a real outage barely moves the line. The chat request afterwards is
    the control - it proves the reader would have SEEN a count, so the zero above is the
    filter working rather than the metrics never being emitted at all."""
    assert gateway_client.get(path).status_code == 200

    assert metrics.total(REQUESTS) == 0
    assert metrics.total(LATENCY_COUNT) == 0

    assert post_chat(chat_client, chat_body(model=ALIAS)).status_code == 200
    assert metrics.total(REQUESTS) == 1


def test_a_streamed_request_is_counted_once(chat_client: TestClient, metrics: MetricDelta) -> None:
    """Catches a streamed request counted twice - once when the route returns the
    `StreamingResponse` and once when the body finishes - which is the easiest mistake to
    make once the observation moves to the end of the send, and which would report
    exactly double the RPS for the path a chat product actually uses."""
    response = post_chat(chat_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text
    assert "data:" in response.text

    labels = request_labels(
        tenant=TENANT,
        alias=ALIAS,
        provider=REMOTE_PROVIDER,
        region=REMOTE_REGION,
        status="200",
        cache=CACHE_MISS,
    )

    assert metrics.delta(REQUESTS, **labels) == 1
    assert metrics.total(REQUESTS) == 1
    assert metrics.total(LATENCY_COUNT) == 1


def test_a_streamed_requests_latency_covers_the_frames(
    slow_stream_client: TestClient,
    slow_stream_adapter: SlowStreamAdapter,
    metrics: MetricDelta,
) -> None:
    """Catches a latency observation taken where the route RETURNS: the adapter pauses
    only after its first event, so such an observation lands under 0.1s however long the
    answer took, and the p95 panel a streaming gateway exists to show would report
    milliseconds for answers that take seconds. The `le=0.1` bucket is asserted at zero
    rather than the sum alone, because a sum can be right while the observation is
    duplicated."""
    answer_seconds = slow_stream_adapter.answer_seconds
    assert answer_seconds > float(DECLARED_BUCKET_BOUNDS[0]), "the pause must cross a bucket edge"

    response = post_chat(slow_stream_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text
    assert response.text.count("data:") == len(STREAM_FRAGMENTS) + 4

    pair = {"alias": ALIAS, "provider": REMOTE_PROVIDER}

    assert metrics.delta(LATENCY_COUNT, **pair) == 1
    assert metrics.delta(LATENCY_BUCKET, le=DECLARED_BUCKET_BOUNDS[0], **pair) == 0
    assert metrics.delta(LATENCY_BUCKET, le="+Inf", **pair) == 1
    assert metrics.delta(LATENCY_SUM, **pair) >= answer_seconds


def test_latency_buckets_are_the_declared_eleven(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches prometheus_client's DEFAULT buckets being taken (decision H): they top out
    at 10 seconds, so every ordinary streamed completion falls in `+Inf` and a p95 is
    unreadable exactly where it matters. a project rule: the eleven bounds are asserted
    as the exact ordered list, so a band of roughly-right values cannot pass."""
    assert post_chat(chat_client, chat_body(model=ALIAS)).status_code == 200

    assert bucket_bounds(metrics, alias=ALIAS, provider=REMOTE_PROVIDER) == list(
        DECLARED_BUCKET_BOUNDS
    )
    assert len(DECLARED_BUCKET_BOUNDS) == 12, "eleven declared bounds plus +Inf"


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def test_tokens_are_recorded_in_and_out_on_a_chat_completion(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches `total_tokens` being recorded as `out` (the two directions would sum to
    more than the provider billed, and an earlier change's ledger, built from the same numbers,
    would disagree with the dashboard), and catches one direction being emitted with the
    other's number - the fixture's prompt and completion counts are deliberately
    different, so a swap is visible."""
    response = post_chat(chat_client, chat_body(model=ALIAS))
    assert response.status_code == 200, response.text
    model = response.headers["x-gateway-model"]

    labels = {"tenant": TENANT, "provider": REMOTE_PROVIDER, "model": model}

    assert metrics.delta(TOKENS, direction="in", **labels) == STUB_USAGE.prompt_tokens
    assert metrics.delta(TOKENS, direction="out", **labels) == STUB_USAGE.completion_tokens
    assert metrics.total(TOKENS) == STUB_USAGE.prompt_tokens + STUB_USAGE.completion_tokens


def test_tokens_are_recorded_from_the_streams_usage_chunk(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a token counter blind to streaming (decision I): the tokens would be
    emitted only where the non-streamed route reads `usage`, so every tenant's usage
    would be understated by the whole of the path a chat product actually uses, and the
    ledger built from the same chunk would disagree. The streamed fixture's numbers
    differ from the non-streamed one's, so a stream priced from the wrong usage object is
    visible too."""
    response = post_chat(chat_client, chat_body(model=ALIAS, stream=True))
    assert response.status_code == 200, response.text
    model = response.headers["x-gateway-model"]

    labels = {"tenant": TENANT, "provider": REMOTE_PROVIDER, "model": model}

    assert metrics.delta(TOKENS, direction="in", **labels) == STREAM_USAGE.prompt_tokens
    assert metrics.delta(TOKENS, direction="out", **labels) == STREAM_USAGE.completion_tokens
    assert metrics.total(TOKENS) == STREAM_USAGE.prompt_tokens + STREAM_USAGE.completion_tokens


def test_tokens_are_recorded_on_an_embeddings_request(
    embeddings_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches an embeddings route that books `total_tokens` as `out`: an embedding has
    no completion, so an `out` sample here is an invented number that doubles the tenant's
    counted usage. The fixture's prompt and total counts differ, so recording the wrong
    one as `in` is caught by the same two lines."""
    response = post_embeddings(embeddings_client, embeddings_body(model=EMBED_ALIAS))
    assert response.status_code == 200, response.text
    model = response.headers["x-gateway-model"]

    labels = {"tenant": TENANT, "provider": LOCAL_PROVIDER, "model": model}

    assert metrics.delta(TOKENS, direction="in", **labels) == STUB_EMBEDDING_USAGE.prompt_tokens
    assert metrics.delta(TOKENS, direction="out", **labels) == 0
    assert metrics.total(TOKENS) == STUB_EMBEDDING_USAGE.prompt_tokens


def test_the_token_model_label_is_the_resolved_model_not_the_alias(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches the alias being used as the `model` label, and the adapter's own answer
    being used instead of the resolved id. Both break the grain an earlier change's ledger prices
    at: an alias covers several provider models, so cost per model would be unattributable
    and the two systems could never be reconciled. The stub deliberately reports a model
    that is neither the alias nor the policy's id, so an echo of the provider's answer
    fails here too."""
    response = post_chat(chat_client, chat_body(model=ALIAS))
    assert response.status_code == 200, response.text
    resolved_model = response.headers["x-gateway-model"]

    assert resolved_model != ALIAS
    assert resolved_model != response.json()["model"], "the body answers the alias, not the id"

    labelled_models = {dict(series)["model"] for series in metrics.moved_series(TOKENS)}

    assert labelled_models == {resolved_model}
    assert ALIAS not in labelled_models


# --------------------------------------------------------------------------
# redactions: one increment per placeholder minted
# --------------------------------------------------------------------------

InlineCall = Callable[[TestClient], HttpxResponse]

INLINE_REDACTION_ROWS: list[tuple[str, str, InlineCall, dict[str, float]]] = [
    (
        "chat",
        "local_chat_client",
        lambda client: post_chat(
            client,
            chat_body(model=ALIAS, messages=[{"role": "user", "content": TEXT_WITH_TWO_KINDS}]),
            data_class=REDACT_CLASS,
        ),
        {"EMAIL": 1, "IPV4": 1},
    ),
    (
        "embeddings",
        "embeddings_client",
        lambda client: post_embeddings(
            client,
            embeddings_body(model=EMBED_ALIAS, input=TEXT_WITH_TWO_KINDS),
            data_class=REDACT_CLASS,
        ),
        {"EMAIL": 1, "IPV4": 1},
    ),
]


@pytest.mark.parametrize(
    ("case", "client_fixture", "send", "expected"),
    INLINE_REDACTION_ROWS,
    ids=[row[0] for row in INLINE_REDACTION_ROWS],
)
def test_redactions_are_counted_per_kind_on_the_inline_path(
    case: str,
    client_fixture: str,
    send: InlineCall,
    expected: dict[str, float],
    request: pytest.FixtureRequest,
    metrics: MetricDelta,
) -> None:
    """Catches a redaction counter that counts REQUESTS rather than placeholders (a body
    with an address and an IP would move it by one, and the per-kind panel that tells an
    operator what class of data is arriving would be flat), and catches the counter being
    wired on one inline route only - the two rows are the two routes that redact inline.
    The `X-Gateway-Redacted` header is asserted first, so a row where redaction never ran
    cannot pass by counting nothing."""
    client: TestClient = request.getfixturevalue(client_fixture)

    response = send(client)
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-redacted"] == "true"

    for kind, count in expected.items():
        assert metrics.delta(REDACTIONS, tenant=TENANT, kind=kind) == count, kind
    assert metrics.total(REDACTIONS) == sum(expected.values())


def test_redactions_are_counted_on_the_redact_endpoint(
    gateway_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches the one endpoint that exists to handle raw personal data being left
    uninstrumented: `/v1/redact` is the ingestion path, so a dashboard blind to it would
    report near-zero redactions while the bulk of the PII the system sees goes through
    here. The counter is asserted per kind on a body carrying two different kinds, so a
    single kindless increment fails."""
    response = gateway_client.post(
        REDACT,
        json={"text": TEXT_WITH_TWO_KINDS},
        headers=auth_headers(key=TEST_KEYS["evals"], tenant=TENANT, data_class=PLAIN_CLASS),
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "write to <EMAIL_1> from <IPV4_1> please"

    assert metrics.delta(REDACTIONS, tenant=TENANT, kind="EMAIL") == 1
    assert metrics.delta(REDACTIONS, tenant=TENANT, kind="IPV4") == 1
    assert metrics.total(REDACTIONS) == 2


def test_a_repeated_value_across_turns_counts_once(
    gateway_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a counter incremented per DETECTION rather than per placeholder minted:
    the same address in three turns is one redacted value under one placeholder (plan
    07b's numbering), so counting detections would report four EMAILs for the two people
    in this body and make the per-tenant redaction rate a function of how often someone
    repeats themselves. The second, distinct address is what stops the test being passed
    by an implementation that counts one per request."""
    response = gateway_client.post(
        REDACT,
        json={
            "messages": [
                {"role": "user", "content": f"from {EMAIL}"},
                {"role": "assistant", "content": f"noted, cc {OTHER_EMAIL}"},
                {"role": "user", "content": f"still {EMAIL}, always {EMAIL}"},
            ]
        },
        headers=auth_headers(key=TEST_KEYS["evals"], tenant=TENANT, data_class=PLAIN_CLASS),
    )
    assert response.status_code == 200, response.text

    contents = [turn["content"] for turn in response.json()["messages"]]
    assert contents == [
        "from <EMAIL_1>",
        "noted, cc <EMAIL_2>",
        "still <EMAIL_1>, always <EMAIL_1>",
    ]

    assert metrics.delta(REDACTIONS, tenant=TENANT, kind="EMAIL") == 2
    assert metrics.total(REDACTIONS) == 2


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


def test_a_provider_failure_increments_the_provider_error_counter(
    failing_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a provider-error counter keyed on nothing a dashboard can use: the label
    pair is the provider and the TYPED code, so an implementation that leaves `provider`
    empty (or writes the alias into it) makes "which provider is failing" unanswerable at
    the moment it is asked. Also catches a 502 that increments the residency counter
    instead, which would fire the design's residency alert on an ordinary outage."""
    response = post_chat(failing_client, chat_body(model=ALIAS))
    assert_error_body(response, status=502, code="PROVIDER_ERROR")

    assert metrics.delta(PROVIDER_ERRORS, provider=REMOTE_PROVIDER, code="PROVIDER_ERROR") == 1
    assert metrics.total(PROVIDER_ERRORS) == 1
    assert metrics.total(RESIDENCY_VIOLATIONS) == 0
    assert metrics.total(REQUESTS) == 1


def test_a_residency_refusal_leaves_the_violation_counter_flat(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """An earlier change decision E. Catches the older reading, where `record_failure` moved
    `gateway_residency_violations_total` on every 451: a 451 is the resolver refusing
    correctly, and the design alerts this counter at greater than zero, so an EU tenant
    asking for `agent-strong` - which the shipped policy refuses by design - would page
    someone every time it happened. The counter is fed only by an `allowed=false`
    residency log row, which the resolver cannot produce; that half is pinned by
    tests/gateway/test_residency_log.py. The refusal is still counted, by
    `gateway_requests_total`, because a 451 counted nowhere is a request the RPS panel
    cannot reconcile with the access log; and it is still not a provider failure."""
    response = post_chat(
        chat_client,
        chat_body(model="agent-strong"),
        key=TEST_KEYS["nordlicht"],
        tenant="nordlicht",
    )
    assert_error_body(response, status=451, code="RESIDENCY_UNAVAILABLE")

    assert metrics.delta(RESIDENCY_VIOLATIONS, tenant="nordlicht") == 0
    assert metrics.total(RESIDENCY_VIOLATIONS) == 0
    assert metrics.total(PROVIDER_ERRORS) == 0
    assert metrics.total(REQUESTS) == 1


def test_an_unauthorized_request_is_counted_with_empty_labels(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a request DROPPED because it failed before resolution: the request counter
    would then stop reconciling with the access log, and a key being probed against the
    gateway - a burst of 401s, the one shape worth alerting on - would be invisible.
    Also catches a `none` sentinel or a tenant label taken from the unauthenticated
    caller's own `X-Tenant` header, which would let a stranger mint time series
    (decision F)."""
    response: HttpxResponse = chat_client.post(CHAT, json=chat_body(model=ALIAS))
    assert_error_body(response, status=401, code="UNAUTHORIZED")

    labels = request_labels(
        tenant=EMPTY, alias=EMPTY, provider=EMPTY, region=EMPTY, status="401", cache=EMPTY
    )

    assert metrics.delta(REQUESTS, **labels) == 1
    assert metrics.total(REQUESTS) == 1


def test_unknown_aliases_create_exactly_one_time_series(
    chat_client: TestClient, metrics: MetricDelta
) -> None:
    """The label-cardinality boundary. Catches `alias` being labelled from the request
    BODY: a registry has no cap of its own, so fifty unknown aliases would mint fifty
    permanent time series in process memory and in Prometheus's index, and anyone able to
    call the gateway could mint them without limit. The last of the fifty is exactly
    `MAX_IDENTIFIER_LENGTH` characters - the inclusive boundary the pre-check admits -
    and the over-cap one is rejected earlier still, yet both must contribute the empty
    alias and nothing else."""
    for alias in UNKNOWN_ALIASES:
        response = post_chat(chat_client, chat_body(model=alias))
        assert_error_body(response, status=404, code="ALIAS_UNKNOWN")

    added = metrics.moved_series(REQUESTS)
    assert len(added) == 1, sorted(added)
    assert dict(added.pop())["alias"] == EMPTY
    assert metrics.total(REQUESTS) == UNKNOWN_ALIAS_COUNT

    over_cap = post_chat(chat_client, chat_body(model=OVER_CAP_ALIAS))
    assert_error_body(over_cap, status=422, code="VALIDATION")

    labelled_aliases = {dict(series)["alias"] for series in metrics.moved_series(REQUESTS)}
    assert labelled_aliases == {EMPTY}


def test_a_failing_emission_changes_no_response(poison_client: TestClient) -> None:
    """Catches an unguarded emission. The route writes a label value whose `str()` raises
    onto the request's own scope record, so the observation the middleware takes after the
    last byte raises; without a guard around it that is a 500 on a request that was
    already answered correctly, which is observability causing the outage it exists to
    report. The `poisoned` field is the proof the record was really reached - a `False`
    there would mean the case tested nothing."""
    response: HttpxResponse = poison_client.get(POISON_PATH)

    assert response.status_code == 200, response.text
    assert response.json() == POISON_BODY
    assert response.headers[POISON_HEADER.lower()] == POISON_HEADER_VALUE


def test_a_second_app_registers_no_duplicate_family(
    settings: Settings, metrics_registry: CollectorRegistry
) -> None:
    """Catches the families being registered per app on prometheus_client's global
    REGISTRY, which is write-once per name: the second `create_app` in one process raises
    `Duplicated timeseries in CollectorRegistry`, which turns a whole test session red at
    once and makes a second app impossible in a worker that hosts one. Both apps are
    scraped, so a second app that answers `/metrics` from a registry of its own - the other
    way to avoid the raise - is caught by the family set the module registry still holds
    being unchanged by either construction."""
    before = family_names(metrics_registry)

    first: FastAPI = create_app(settings, NO_PROVIDER_KEYS)
    second: FastAPI = create_app(settings, NO_PROVIDER_KEYS)

    with TestClient(first) as first_client, TestClient(second) as second_client:
        first_body = scrape(first_client).text
        second_body = scrape(second_client).text

    assert family_names(metrics_registry) == before
    assert type_names(first_body) == set(DECLARED_FAMILIES)
    assert type_names(second_body) == set(DECLARED_FAMILIES)


# --------------------------------------------------------------------------
# the fallback counter and the circuit gauge
# --------------------------------------------------------------------------
#
# The two families an earlier change declared empty and this plan fills are fed from here
# on, which is why they moved out of `UNFED_FAMILIES` above. They are
# read the same way everything else in this file is - through the session
# registry - but the gauge is read ABSOLUTELY rather than as a delta: a gauge
# carries a level, not an increment, and "0 for a provider nothing failed on" is
# a claim about the value a scraper sees rather than about how far it moved.
# `get_sample_value` answers `None` for a series that does not exist, which is
# what separates "publishes 0" from "publishes nothing".

CIRCUIT_OPEN = 2.0
CIRCUIT_CLOSED = 0.0
"""The two gauge levels this file can produce, spelled as the exposition renders
them. The state enum's own members are pinned in tests/providers/test_breaker.py;
these are the numbers a dashboard query and an alert rule are written against."""


def send_fallback(client: TestClient) -> HttpxResponse:
    """One chat request as the tenant whose alias has two candidates."""
    return post_chat(
        client,
        chat_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )


def gauge_value(collector_registry: CollectorRegistry, name: str, **labels: str) -> float | None:
    """One gauge series' current level, or `None` when the series does not exist."""
    return collector_registry.get_sample_value(name, labels)


def test_a_fallback_is_counted_once_against_the_serving_provider(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    metrics: MetricDelta,
) -> None:
    """Catches the counter labelled with the provider that FAILED, which is the natural
    mistake when the failure is the thing being handled: the design's panel would then
    report fallbacks onto a provider that served nothing, and the tenant's traffic would be
    attributed to the wrong region. The total pins "once per request" as well, so a loop
    that counted each candidate it tried is caught by the same case."""
    fallback_adapters[FIRST_PROVIDER].chat_error = retryable_error(FIRST_PROVIDER)

    response = send_fallback(fallback_client)
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == SECOND_PROVIDER

    labels = {"tenant": FALLBACK_TENANT, "alias": FALLBACK_ALIAS}
    assert metrics.delta(FALLBACKS, provider=SECOND_PROVIDER, **labels) == 1
    assert metrics.delta(FALLBACKS, provider=FIRST_PROVIDER, **labels) == 0
    assert metrics.total(FALLBACKS) == 1


def test_a_request_that_needed_no_fallback_counts_none(
    fallback_client: TestClient, metrics: MetricDelta
) -> None:
    """Catches a counter incremented once per served request rather than once per request
    that actually fell back - every panel would then show a permanent 100% fallback rate,
    and the alert that says a provider is degrading could never fire on anything. Nothing
    is scripted to fail here, so the first candidate serves."""
    response = send_fallback(fallback_client)
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == FIRST_PROVIDER

    assert metrics.total(FALLBACKS) == 0


def test_the_circuit_gauge_reads_open_for_the_failed_provider_and_closed_for_the_other(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    metrics_registry: CollectorRegistry,
) -> None:
    """Catches a gauge published only when a breaker OPENS - the healthy provider's series
    would be missing, so an alert on `gateway_circuit_state != 0` would be blind to every
    provider until the moment it broke - and catches the two levels swapped or spelled as
    a 1/0 up-down gauge, which is what an operator's rule is written against. The serving
    provider is the control: it takes the same traffic and must read closed."""
    fallback_adapters[FIRST_PROVIDER].chat_error = retryable_error(FIRST_PROVIDER)

    for _ in range(BREAKER_MAX_FAILURES):
        assert send_fallback(fallback_client).status_code == 200

    assert gauge_value(metrics_registry, CIRCUIT_STATE, provider=FIRST_PROVIDER) == CIRCUIT_OPEN
    assert gauge_value(metrics_registry, CIRCUIT_STATE, provider=SECOND_PROVIDER) == CIRCUIT_CLOSED
