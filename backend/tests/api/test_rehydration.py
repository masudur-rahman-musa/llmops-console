r"""`X-Gateway-Rehydrate` on `POST /v1/chat/completions`: the opt-in, and its refusals.

Off by default is the ABSENCE of the header, so the first two cases are arranged
against the strongest wrong implementation there is: a live map for the calling
tenant, and an answer full of that map's placeholders. A route that rehydrated by
default, or that looked up "the tenant's most recent map", serves the raw address
and reddens; a route that reads the header cannot.

The refusals all share one shape - 422 `VALIDATION` or 502 `PROVIDER_ERROR`, with
the fake adapter's `requests` asserted EMPTY - because the plan's claim is not
merely "it fails" but "it fails before a provider is dialled", which is what makes
a stale id cost no tokens. `call_count` is the only witness for that, and it is
why the stub records every request it is handed.

Two arrangements are worth reading before the assertions:

* THE ID PRE-CHECK vs THE MISSING MAP. Both answer 422, so a rejection row only
  proves the pre-check exists if the STORE was never asked. Every row of
  `test_the_map_id_rejection_table` therefore asserts `map_store.gets == []`,
  while `test_an_unknown_map_id_is_422_with_no_outbound_call` asserts the
  opposite - exactly one read - so the two cases cannot both pass an
  implementation missing either control (a project rule's boundary, and the
  neighbouring-guard trap).
* THE OTHER TENANT'S ID. `test_an_id_live_for_another_tenant_is_absent` files a
  real map for `cuan` and spends its id as `evals`, so "unknown id" cannot be
  what makes it pass: a store lookup keyed by map id alone would answer `cuan`'s
  map and serve `cuan`'s data.

The map store is reached by NAME, `app.state.redaction_map_store`, never by
shape: this directory has two stores with a `put` and a `get` since an earlier change, and
selecting one of them by its methods is what a project rule exists for.

a project rule: one app for the module, built once, with per-test isolation from
resetting the stub's script and the store's recorded reads. The 502 case is the
one exception - its store must refuse every read, so it gets an app of its own
rather than leaving every later case here talking to a broken store.

An earlier change adds the streamed half, and the arrangements there are chosen the same
way:

* THE FLUSH POINTS. Each of the three has its script arranged so that only its
  own flush can decide the outcome. The `finish_reason` case asserts the ORDER of
  the frames rather than the assembled text, because a flush that happened only
  at `[DONE]` would still deliver every character - just after the client had
  been told the answer ended. The `[DONE]` case ends its script on the text delta
  itself, with no `finish_reason` and no usage chunk after it, so the flush before
  a non-text event cannot be what saves it.
* THE REFUSALS ON A STREAM. An earlier change answered 422 to the header on ANY streamed
  body before it looked at the id at all, and this plan deletes that guard; the
  four rows re-assert the remaining refusals on a streamed body, and each is
  separated from its neighbour exactly as the non-streamed four are - by whether
  the store was asked (`gets == []` for the malformed id, one read for the
  unknown one).
* THE STUB'S SCRIPT. `RehydrationStub.events` is per test for the same reason
  `response` is: every streamed case needs a different fragment sequence, and one
  app per case would be the rebuild a project rule forbids.

No provider key is read and no network is touched: the adapter is a stub and the
map store is the in-process one `create_app` builds.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.core.errors import ProviderError
from app.providers.base import (
    Adapter,
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    UsageTotal,
)
from app.redaction import (
    MAP_ID_PATTERN,
    RedactionMapStore,
    RedactionMapStoreUnavailable,
)

from .conftest import (
    CHAT,
    STREAM_TOOL_INDEX,
    STREAM_USAGE,
    TEST_KEYS,
    app_of,
    assert_error_body,
    auth_headers,
    build_chat_client,
    chat_body,
    stream_events,
    stub_response,
)

REHYDRATE_HEADER = "X-Gateway-Rehydrate"
"""The published request header. Named as a constant so a route that shipped a
near-miss spelling (`X-Gateway-Rehydration`) fails every case here rather than
silently ignoring the header on the one request a caller sent it on."""

TENANT = "evals"
OTHER_TENANT = "cuan"
DATA_CLASS = "internal"

EMAIL_A = "alice@example.com"
PERSON_A = "Alice Schmidt"
EMAIL_OTHER = "carol@example.org"

# The map the calling tenant files and then spends.
LIVE_MAPPING: Mapping[str, str] = {"<EMAIL_1>": EMAIL_A, "<PERSON_1>": PERSON_A}

# A map belonging to a DIFFERENT tenant, with the same placeholder names, so a
# lookup that ignored the tenant would answer something visibly wrong.
OTHER_MAPPING: Mapping[str, str] = {"<EMAIL_1>": EMAIL_OTHER, "<PERSON_1>": "Carol Doyle"}

# Well-formed by `MAP_ID_PATTERN` and never minted by any store: 32 lowercase hex
# characters, so the ONLY thing that can refuse it is the map being absent.
UNKNOWN_ID = "0123456789abcdef0123456789abcdef"

# The base every malformed row is built from, so each row differs from an
# ACCEPTABLE id by exactly the one defect it is named for.
WELL_FORMED_ID = "fedcba9876543210fedcba9876543210"

# The published set, lowercased as httpx returns header names. These six are the
# ones a STREAM carries: they are on the wire before the first frame, which is
# before the usage a cost could be priced from exists. This module asserts a
# rehydrated stream still carries exactly them and no more (decision G).
EXPECTED_HEADER_NAMES = {
    "x-gateway-provider",
    "x-gateway-region",
    "x-gateway-model",
    "x-gateway-cache",
    "x-gateway-redacted",
    "x-gateway-trace-id",
}

# The set a non-streamed answer carries: the six above and the cost of the call,
# which the route adds once the ledger has priced it. The same seven
# `tests/gateway/test_response_headers.py` pins on an ordinary completion.
EXPECTED_ANSWER_HEADER_NAMES = EXPECTED_HEADER_NAMES | {"x-gateway-cost-usd"}

GATEWAY_HEADER_PREFIX = "x-gateway-"

STUB_TOOL_ID = "toolu_01RehydrateCall0000000"
STUB_TOOL_NAME = "send_email"

DONE_PAYLOAD = "[DONE]"

# The two SSE headers a streamed answer carries beside the six `X-Gateway-*`
# ones. Named here so the rehydrated stream is compared against the same pair
# `tests/gateway/test_chat_streaming.py` pins on an ordinary one, rather than
# against whatever the rehydrating branch happens to build.
EXPECTED_SSE_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}

# A PROPER PREFIX of `<PERSON_1>`, the longest key in `LIVE_MAPPING`. That is the
# whole reason it is held rather than emitted, and being held is what makes the
# three flush points observable at all: no script below ever sends the `1>` that
# would complete it, so it can only leave the gateway through a flush.
HELD_PARTIAL = "<PERSON_"


class RehydrationStub:
    """A stub adapter whose ANSWER is set per test, and which records every call.

    `StubAdapter` in this directory's `conftest.py` fixes its response at
    construction, and every case here needs a different one (a placeholder in the
    content, a placeholder in a tool call, an unnamed placeholder), so one app per
    case would be the rebuild a project rule forbids. `response` and `reset()`
    are the same public-script shape `FallbackStubAdapter` uses.

    `events` is the streamed half of the same script and is set the
    same way, because each streamed case needs its own fragment sequence: one
    holding a partial placeholder at the end, one carrying a tool call, one that
    stops without a `finish_reason` at all.

    `stream_error` is raised AFTER the scripted events have been yielded, which
    is the mid-stream position an earlier change decision E is about: the 200 and the
    first frames are already on the wire, so the failure can only be a frame.

    `requests` is the load-bearing part: every refusal in this module asserts it
    is still empty, which is the only evidence a caller with a stale map id paid
    for no tokens.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []
        self.response: ChatResponse = stub_response()
        self.events: tuple[StreamEvent, ...] = stream_events()
        self.stream_error: Exception | None = None

    def reset(self) -> None:
        """Forget every recorded call and go back to the default answer and script."""
        self.requests = []
        self.response = stub_response()
        self.events = stream_events()
        self.stream_error = None

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
        if self.stream_error is not None:
            raise self.stream_error

    def healthcheck(self) -> bool:
        return True

    def price_key(self, model: str) -> str:
        return f"{self.name}/{model}"


@dataclass
class RecordingMapStore:
    """The real in-process store with every `get` recorded, so "no read" is observable.

    Wraps rather than replaces: the live cases file a map through `put` and the
    route reads it back through `get`, so the read half has to keep working.
    """

    inner: RedactionMapStore
    gets: list[tuple[str, str]] = field(default_factory=list)

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        return self.inner.put(tenant, mapping)

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        self.gets.append((tenant, map_id))
        return self.inner.get(tenant, map_id)


REDIS_FAILURE_TEXT = "Error 111 connecting to redis:6379. Connection refused."
"""The store client's own words, planted as a marker: they name an internal host
and port, and they must not travel to a caller. The same sentinel
`tests/gateway/test_redact_endpoint.py` uses on the write side."""


class UnavailableMapStore:
    """A map store whose READ fails the way an unreachable Redis does.

    The write half is irrelevant here and answers a well-formed id, so a route
    that somehow reached `put` would not be the thing that reddens the case.
    """

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        return WELL_FORMED_ID

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        raise RedactionMapStoreUnavailable(REDIS_FAILURE_TEXT)


@pytest.fixture(scope="module")
def rehydration_stub() -> RehydrationStub:
    """The one adapter behind this module's app, under the policy provider `anthropic`."""
    return RehydrationStub(name="anthropic")


@pytest.fixture(scope="module")
def rehydration_client(
    settings: Settings, rehydration_stub: RehydrationStub
) -> Iterator[TestClient]:
    """One app for the module, with its map store wrapped so reads are counted."""
    with build_chat_client(settings, {rehydration_stub.name: rehydration_stub}) as client:
        app = app_of(client)
        inner = cast(RedactionMapStore, app.state.redaction_map_store)
        app.state.redaction_map_store = RecordingMapStore(inner=inner)
        yield client


@pytest.fixture(scope="module")
def map_store(rehydration_client: TestClient) -> RecordingMapStore:
    """The recorder now sitting on `app.state.redaction_map_store`, by name."""
    return cast(RecordingMapStore, app_of(rehydration_client).state.redaction_map_store)


@pytest.fixture(scope="module")
def live_map_id(map_store: RecordingMapStore) -> str:
    """One map filed for the CALLING tenant, exactly as `/v1/redact` would have filed it."""
    return map_store.put(TENANT, LIVE_MAPPING)


@pytest.fixture(scope="module")
def other_tenant_map_id(map_store: RecordingMapStore) -> str:
    """One map filed for a DIFFERENT tenant, whose id the calling tenant then spends."""
    return map_store.put(OTHER_TENANT, OTHER_MAPPING)


@pytest.fixture(autouse=True)
def _fresh_script(
    rehydration_stub: RehydrationStub, map_store: RecordingMapStore
) -> Iterator[None]:
    """Per-test isolation from resetting the script and the record, never a new app."""
    rehydration_stub.reset()
    map_store.gets.clear()
    yield


def post_rehydrated(
    client: TestClient,
    body: Mapping[str, Any],
    *,
    map_id: str | None,
    key: str = TEST_KEYS[TENANT],
    tenant: str = TENANT,
) -> Response:
    """POST one completion as `evals` under `internal`, with the header only if given.

    `map_id=None` sends NO `X-Gateway-Rehydrate` header at all, which is what "off
    by default" means here: there is no flag, and no value that means off.
    """
    headers = auth_headers(key=key, tenant=tenant, data_class=DATA_CLASS)
    if map_id is not None:
        headers[REHYDRATE_HEADER] = map_id
    response: Response = client.post(CHAT, json=dict(body), headers=headers)
    return response


def content_of(response: Response) -> str | None:
    """The assistant `content` of a successful completion, with the 200 asserted."""
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    content: str | None = message["content"]
    return content


def tool_calls_of(response: Response) -> list[dict[str, Any]]:
    """The wire tool calls of a successful completion, with the 200 asserted."""
    assert response.status_code == 200, response.text
    calls: list[dict[str, Any]] = response.json()["choices"][0]["message"]["tool_calls"]
    return calls


def map_store_reads(client: TestClient) -> list[tuple[str, str]]:
    """Every `(tenant, map_id)` this module's store was asked for since the last reset.

    Read off `app.state.redaction_map_store` by NAME. Two stores on this app carry
    a `put` and a `get` since an earlier change, so a helper that went looking for the
    shape would pick whichever came first.
    """
    store = cast(RecordingMapStore, app_of(client).state.redaction_map_store)
    return store.gets


def gateway_headers(response: Response) -> dict[str, str]:
    """The `X-Gateway-*` headers of a response, lowercased, with the 200 asserted."""
    assert response.status_code == 200, response.text
    return {
        name.lower(): value
        for name, value in response.headers.items()
        if name.lower().startswith(GATEWAY_HEADER_PREFIX)
    }


def stream_body(**overrides: object) -> dict[str, Any]:
    """The baseline request body with `stream: true`."""
    return chat_body(stream=True, **overrides)


def data_frames(response: Response) -> list[str]:
    """The payload of every `data:` frame of a streamed 200, sentinel included.

    Written here rather than imported from `tests/gateway/test_chat_streaming.py`:
    those helpers are that module's private reading of the same wire format, and
    no test module in this repo imports from another (`grep -rn "^from test" tests/`
    finds nothing). The media type is asserted here once, so an
    `application/json` refusal cannot be read as frames by any case below.
    """
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";")[0] == "text/event-stream"
    return [
        line.removeprefix("data:").strip()
        for line in response.text.splitlines()
        if line.strip().startswith("data:")
    ]


def chunks(response: Response) -> list[dict[str, Any]]:
    """Every frame of a streamed 200 parsed as JSON, without the `[DONE]` sentinel."""
    return [json.loads(payload) for payload in data_frames(response) if payload != DONE_PAYLOAD]


def deltas(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `delta` object of every chunk that carries one choice.

    An error frame carries no `choices` at all, so the key is
    read with `get`: the frame is skipped rather than indexed, which is what
    `test_held_text_is_flushed_before_an_error_frame` needs to reach its own
    assertions about the flushed span and the single trailing frame.
    """
    return [chunk["choices"][0]["delta"] for chunk in parsed if chunk.get("choices")]


def streamed_content(response: Response) -> str:
    """The answer a client assembles: every `content` delta joined in frame order.

    This is the only reading in which a lost carry and a doubled one are both
    visible, and it is what an OpenAI SDK builds from the same frames.
    """
    return "".join(delta.get("content") or "" for delta in deltas(chunks(response)))


def content_frame_positions(parsed: list[dict[str, Any]]) -> list[int]:
    """The index of every chunk carrying a non-empty `content` delta."""
    return [
        index
        for index, chunk in enumerate(parsed)
        if chunk["choices"] and chunk["choices"][0]["delta"].get("content")
    ]


def finish_frame_positions(parsed: list[dict[str, Any]]) -> list[int]:
    """The index of every chunk carrying a `finish_reason`."""
    return [
        index
        for index, chunk in enumerate(parsed)
        if chunk["choices"] and chunk["choices"][0].get("finish_reason") is not None
    ]


# --------------------------------------------------------------------------
# off by default
# --------------------------------------------------------------------------


def test_no_header_answers_exactly_what_it_answers_today(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    map_store: RecordingMapStore,
    live_map_id: str,
) -> None:
    """Catches a route that rehydrates whenever a map exists for the tenant - by
    defaulting the feature on, or by looking up the tenant's most recent map: the map
    naming `<EMAIL_1>` IS live for this caller and its id has been minted, so the only
    thing standing between the answer and `alice@example.com` is the missing header."""
    rehydration_stub.response = stub_response(content="write to <EMAIL_1> about <PERSON_1>")

    response = post_rehydrated(rehydration_client, chat_body(), map_id=None)

    assert content_of(response) == "write to <EMAIL_1> about <PERSON_1>"
    assert EMAIL_A not in response.text
    assert PERSON_A not in response.text
    # The arrangement itself, asserted rather than assumed: this map really is
    # live for this tenant, so "the header was absent" is the only reason the
    # answer still carries placeholders.
    assert map_store.get(TENANT, live_map_id) == LIVE_MAPPING


def test_no_header_reads_the_map_store_zero_times(
    rehydration_client: TestClient, live_map_id: str
) -> None:
    """Catches an implementation that loads the map first and decides afterwards: every
    request on the hot path would pay a store round trip (a Redis call in production) for
    a feature almost no request asks for, and the cost would be invisible in a suite that
    only asserted the answer."""
    post_rehydrated(rehydration_client, chat_body(), map_id=None)

    assert map_store_reads(rehydration_client) == []


# --------------------------------------------------------------------------
# the answer, rehydrated
# --------------------------------------------------------------------------


def test_a_placeholder_in_the_answer_comes_back_as_the_raw_value(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    live_map_id: str,
) -> None:
    """Catches a route that accepts the header and returns the model's text untouched -
    the failure a caller cannot see until an end user reads `<EMAIL_1>` in their inbox:
    both placeholders the map names are asserted gone AND both raw values present, so
    substituting only the first key does not pass either."""
    rehydration_stub.response = stub_response(content="write to <EMAIL_1> about <PERSON_1>")

    response = post_rehydrated(rehydration_client, chat_body(), map_id=live_map_id)

    assert content_of(response) == f"write to {EMAIL_A} about {PERSON_A}"
    assert "<EMAIL_1>" not in response.text
    assert "<PERSON_1>" not in response.text


def test_a_placeholder_in_a_tool_calls_arguments_is_rehydrated(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    live_map_id: str,
) -> None:
    """Catches a route that substitutes into `content` only (the half of decision D that
    is easy to forget): a tool call handed `<EMAIL_1>` is a broken tool call, and the
    content here carries no placeholder at all, so a content-only implementation returns
    a 200 that looks entirely correct. The call's `id` and `name` are pinned unchanged,
    so a route that rebuilt the tool call around the substitution is caught too."""
    rehydration_stub.response = stub_response(
        content="Sending it now.",
        tool_calls=(
            ToolCall(
                id=STUB_TOOL_ID,
                name=STUB_TOOL_NAME,
                arguments='{"to": "<EMAIL_1>", "subject": "for <PERSON_1>"}',
            ),
        ),
    )

    response = post_rehydrated(rehydration_client, chat_body(), map_id=live_map_id)

    calls = tool_calls_of(response)
    assert len(calls) == 1
    assert calls[0]["id"] == STUB_TOOL_ID
    assert calls[0]["function"]["name"] == STUB_TOOL_NAME
    assert calls[0]["function"]["arguments"] == (
        f'{{"to": "{EMAIL_A}", "subject": "for {PERSON_A}"}}'
    )
    assert content_of(response) == "Sending it now."


def test_a_placeholder_the_map_does_not_name_is_left_alone(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    live_map_id: str,
) -> None:
    """Catches a route that scans the answer for the generic `<KIND_N>` shape and looks
    each hit up in the map: `<CARD_9>` is not in this map, and blanking it (or failing on
    it) rewrites text the model wrote for a reason. The named placeholder in the same
    string is asserted restored, so an implementation that substituted nothing at all
    cannot pass this row either."""
    rehydration_stub.response = stub_response(content="<EMAIL_1> paid with <CARD_9>")

    response = post_rehydrated(rehydration_client, chat_body(), map_id=live_map_id)

    assert content_of(response) == f"{EMAIL_A} paid with <CARD_9>"


def test_a_rehydrated_answer_carries_the_same_headers(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    live_map_id: str,
) -> None:
    """Catches the header decision G refuses - an `X-Gateway-Rehydrated: true` added
    because it seemed helpful - and the opposite mistake of a rehydrating branch that
    returns its own response object and drops the ones a caller's dashboard reads. The
    set is compared whole, and against a request that really did rehydrate. It is the
    seven of a non-streamed answer since an earlier change: rehydration changes what a header
    SAYS about the body, never which headers there are."""
    rehydration_stub.response = stub_response(content="write to <EMAIL_1>")

    response = post_rehydrated(rehydration_client, chat_body(), map_id=live_map_id)

    headers = gateway_headers(response)
    assert set(headers) == EXPECTED_ANSWER_HEADER_NAMES
    assert len(headers) == 7
    assert content_of(response) == f"write to {EMAIL_A}"


# --------------------------------------------------------------------------
# the stream
# --------------------------------------------------------------------------


def test_a_stream_without_the_header_is_what_it_is_today(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    map_store: RecordingMapStore,
    live_map_id: str,
) -> None:
    """Catches a route that rehydrates every stream once the streamed path can rehydrate
    at all - the easy mistake when a refusal is lifted, since the branch that used to
    raise is now the branch that substitutes. The map naming both placeholders IS live for
    this caller, so only the missing header stands between these frames and
    `alice@example.com`, and the arrangement is asserted rather than assumed at the end."""
    rehydration_stub.events = stream_events(fragments=("write to <EMAIL_1>", " about <PERSON_1>"))

    response = post_rehydrated(rehydration_client, stream_body(), map_id=None)

    assert streamed_content(response) == "write to <EMAIL_1> about <PERSON_1>"
    assert EMAIL_A not in response.text
    assert PERSON_A not in response.text
    assert map_store.get(TENANT, live_map_id) == LIVE_MAPPING


def test_a_stream_without_the_header_reads_the_map_store_zero_times(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches a streamed path that loads the map before it looks at the header - or that
    builds a rehydrator for every stream and feeds it an empty map: every streamed request
    would pay a store round trip (a Redis call in production) for a feature almost no
    stream asks for, and a suite that only read the frames could not see it."""
    rehydration_stub.events = stream_events(fragments=("write to <EMAIL_1>",))

    post_rehydrated(rehydration_client, stream_body(), map_id=None)

    assert map_store_reads(rehydration_client) == []


def test_a_streamed_placeholder_comes_back_as_the_raw_value(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches the route still refusing the combination 12b turned down (decision G), and
    catches the opposite half-fix: a 200 `text/event-stream` whose frames were never
    passed the map, which hands the caller `<EMAIL_1>` while telling it the request
    succeeded. Both placeholders are asserted gone from the WHOLE body and both raw values
    present, so substituting only the first key passes neither."""
    rehydration_stub.events = stream_events(fragments=("write to <EMAIL_1>", " about <PERSON_1>"))

    response = post_rehydrated(rehydration_client, stream_body(), map_id=live_map_id)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";")[0] == "text/event-stream"
    assert streamed_content(response) == f"write to {EMAIL_A} about {PERSON_A}"
    assert "<EMAIL_1>" not in response.text
    assert "<PERSON_1>" not in response.text
    assert data_frames(response)[-1] == DONE_PAYLOAD


def test_a_streamed_tool_calls_arguments_are_rehydrated(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches a streamed wrap that handles `TextDelta` only: this script has no text at
    all, so a content-only implementation streams a 200 that looks entirely correct and
    hands the client a tool call whose `to` field is `<EMAIL_1>`. `index`, `id` and `name`
    are pinned unchanged, so a wrap that rebuilt the call around the substitution - or
    reset its index to 0 - is caught as well. `ToolCallDelta` carries a COMPLETE call, which is why its arguments need no buffer."""
    rehydration_stub.events = stream_events(
        fragments=(),
        tool_calls=(
            ToolCallDelta(
                index=STREAM_TOOL_INDEX,
                call=ToolCall(
                    id=STUB_TOOL_ID,
                    name=STUB_TOOL_NAME,
                    arguments='{"to": "<EMAIL_1>", "subject": "for <PERSON_1>"}',
                ),
            ),
        ),
        finish_reason="tool_use",
    )

    response = post_rehydrated(rehydration_client, stream_body(), map_id=live_map_id)

    calls = [call for delta in deltas(chunks(response)) for call in (delta.get("tool_calls") or [])]
    assert len(calls) == 1
    assert calls[0]["index"] == STREAM_TOOL_INDEX
    assert calls[0]["id"] == STUB_TOOL_ID
    assert calls[0]["function"]["name"] == STUB_TOOL_NAME
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "to": EMAIL_A,
        "subject": f"for {PERSON_A}",
    }


def test_held_text_is_flushed_before_the_finish_reason(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches a wrap that flushes only at the end of the stream (decision E's rejected
    half). The text is all delivered either way, so the assembled answer cannot tell the
    two apart - the ORDER can: a client that renders as it reads paints a visibly
    truncated answer, then finishes it after `finish_reason` said the answer was over. The
    usage chunk is asserted last as well, so a flush emitted after everything is caught by
    the same case."""
    rehydration_stub.events = (
        StreamStart(),
        TextDelta(text=f"write to <EMAIL_1> or {HELD_PARTIAL}"),
        FinishReason(reason="end_turn"),
        UsageTotal(usage=STREAM_USAGE),
    )

    response = post_rehydrated(rehydration_client, stream_body(), map_id=live_map_id)
    parsed = chunks(response)

    assert streamed_content(response) == f"write to {EMAIL_A} or {HELD_PARTIAL}"
    finished = finish_frame_positions(parsed)
    assert len(finished) == 1
    assert max(content_frame_positions(parsed)) < finished[0]
    assert parsed[-1]["usage"]["total_tokens"] == STREAM_USAGE.total_tokens


def test_a_trailing_partial_placeholder_is_flushed_before_done(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches a wrap that returns without a final flush: `<PERSON_` is a proper prefix of
    a key and is therefore still held when the events run out, so it would be dropped and
    the caller would never learn that the last of what the provider wrote - and billed for
    - was thrown away. The script deliberately ends on the text delta itself, with no
    `finish_reason` and no usage chunk after it, so the flush before a non-text event
    cannot be what saves this row."""
    rehydration_stub.events = (StreamStart(), TextDelta(text=f"write to {HELD_PARTIAL}"))

    response = post_rehydrated(rehydration_client, stream_body(), map_id=live_map_id)

    assert streamed_content(response) == f"write to {HELD_PARTIAL}"
    assert data_frames(response)[-1] == DONE_PAYLOAD


def test_held_text_is_flushed_before_an_error_frame(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches an error path that returns straight from the `except` without flushing
    (decision F): the held span is text the provider produced and the caller paid for, and
    at worst it is a partial placeholder, which is what the model wrote anyway. The
    truncation report of an earlier change decision E is asserted unchanged in the same case - one
    error frame, last, and no sentinel - so a flush bolted on by re-running the whole
    stream, or one that let `[DONE]` follow the error, is caught here too."""
    rehydration_stub.events = (
        StreamStart(),
        TextDelta(text=f"write to <EMAIL_1> or {HELD_PARTIAL}"),
    )
    rehydration_stub.stream_error = ProviderError(
        "anthropic: the connection dropped", retryable=True
    )

    response = post_rehydrated(rehydration_client, stream_body(), map_id=live_map_id)
    payloads = data_frames(response)

    assert streamed_content(response) == f"write to {EMAIL_A} or {HELD_PARTIAL}"
    assert DONE_PAYLOAD not in payloads
    errors = [json.loads(payload) for payload in payloads if "error" in json.loads(payload)]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "PROVIDER_ERROR"
    assert json.loads(payloads[-1]) == errors[0], "the error frame is the last one"


def test_a_rehydrated_stream_carries_the_same_headers(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    """Catches the header decision G refuses - an `X-Gateway-Rehydrated: true` added
    because it seemed helpful - and the opposite mistake of a rehydrating branch that
    builds its own `StreamingResponse` and drops the six an operator's dashboard reads,
    or the two SSE headers without which a proxy buffers the whole answer and delivers it
    as one block. The set is compared whole, against a request that really did rehydrate.
    Six and not the answer's seven: a stream's headers are on the wire before its usage
    exists, so `X-Gateway-Cost-USD` cannot be among them."""
    rehydration_stub.events = stream_events(fragments=("write to <EMAIL_1>",))

    response = post_rehydrated(rehydration_client, stream_body(), map_id=live_map_id)

    headers = gateway_headers(response)
    assert set(headers) == EXPECTED_HEADER_NAMES
    assert len(headers) == 6
    lowered = {name.lower(): value for name, value in response.headers.items()}
    assert {name: lowered[name] for name in EXPECTED_SSE_HEADERS} == EXPECTED_SSE_HEADERS
    assert lowered["content-type"].split(";")[0] == "text/event-stream"
    assert streamed_content(response) == f"write to {EMAIL_A}"


# --------------------------------------------------------------------------
# the refusals, all before a provider is dialled
# --------------------------------------------------------------------------


def test_an_id_live_for_another_tenant_is_absent(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    other_tenant_map_id: str,
) -> None:
    """Catches a store lookup keyed by map id alone, or one that trusts the id to carry
    its own tenant: this id IS live and its map DOES name `<EMAIL_1>`, for `cuan`, so an
    implementation missing the tenant in its key answers 200 with Carol's address in
    `evals`' completion. `carol@example.org` is asserted absent from the whole body,
    which is the leak the criterion is really about."""
    rehydration_stub.response = stub_response(content="write to <EMAIL_1>")

    response = post_rehydrated(rehydration_client, chat_body(), map_id=other_tenant_map_id)

    assert_error_body(response, status=422, code="VALIDATION")
    assert EMAIL_OTHER not in response.text
    assert rehydration_stub.call_count == 0


def test_a_malformed_map_id_is_422_with_no_outbound_call(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub
) -> None:
    """Catches an id handed straight to the store: the in-process store answers `None`
    for anything it does not hold, so the request would still 422 and the missing
    pre-check would be invisible - unless the store is asserted never to have been asked,
    which is what separates this case from the unknown-id one below."""
    response = post_rehydrated(rehydration_client, chat_body(), map_id="not-a-valid-map-id")

    assert_error_body(response, status=422, code="VALIDATION")
    assert map_store_reads(rehydration_client) == []
    assert rehydration_stub.call_count == 0


def test_an_unknown_map_id_is_422_with_no_outbound_call(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    other_tenant_map_id: str,
) -> None:
    """Catches two things at once: a route that serves the answer un-rehydrated when the
    map has expired (decision B - `<PERSON_1>` reaching an end user reads as a bug with
    no header to explain it), and an error message that names WHY the id failed. The
    unknown-id body and the other-tenant body are compared whole, so a message quoting
    the id, or saying "belongs to another tenant", hands a caller an oracle for which
    ids exist."""
    unknown = post_rehydrated(rehydration_client, chat_body(), map_id=UNKNOWN_ID)
    other = post_rehydrated(rehydration_client, chat_body(), map_id=other_tenant_map_id)

    assert_error_body(unknown, status=422, code="VALIDATION")
    assert unknown.json() == other.json()
    assert map_store_reads(rehydration_client) == [
        (TENANT, UNKNOWN_ID),
        (TENANT, other_tenant_map_id),
    ]
    assert rehydration_stub.call_count == 0


def test_a_malformed_map_id_on_a_stream_is_422_with_no_outbound_call(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub
) -> None:
    """Catches a route that lifts 12b's streamed refusal and hands the id straight to the
    store on the streamed path only: the in-process store answers `None` for anything it
    does not hold, so the request would still 422 and the missing pre-check would be
    invisible. `gets == []` is the whole discriminator, exactly as on the non-streamed
    row, and the JSON content type catches a route that opened a stream first and then
    reported the refusal inside a 200."""
    response = post_rehydrated(rehydration_client, stream_body(), map_id="not-a-valid-map-id")

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.headers["content-type"].split(";")[0] == "application/json"
    assert map_store_reads(rehydration_client) == []
    assert rehydration_stub.call_count == 0


def test_an_unknown_map_id_on_a_stream_is_422_with_no_outbound_call(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub
) -> None:
    """Catches the streamed path skipping the map lookup altogether - streaming the
    answer un-rehydrated when the map has expired, which reaches an end user as
    `<PERSON_1>` in the middle of a sentence with nothing to explain it. Exactly one store
    read is asserted, which is what separates this row from the malformed one above: an
    implementation missing either control cannot satisfy both."""
    response = post_rehydrated(rehydration_client, stream_body(), map_id=UNKNOWN_ID)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.headers["content-type"].split(";")[0] == "application/json"
    assert map_store_reads(rehydration_client) == [(TENANT, UNKNOWN_ID)]
    assert rehydration_stub.call_count == 0


def test_an_id_live_for_another_tenant_is_absent_on_a_stream(
    rehydration_client: TestClient,
    rehydration_stub: RehydrationStub,
    other_tenant_map_id: str,
) -> None:
    """Catches a streamed lookup keyed by map id alone: this id IS live and its map DOES
    name `<EMAIL_1>`, for `cuan`, so an implementation missing the tenant in its key
    streams Carol's address to `evals` frame by frame - the leak is the same as on the
    non-streamed path, and a stream is where a body-wide assertion is the only way to see
    it. The stub is scripted with that placeholder so a leak would have somewhere to
    appear."""
    rehydration_stub.events = stream_events(fragments=("write to <EMAIL_1>",))

    response = post_rehydrated(rehydration_client, stream_body(), map_id=other_tenant_map_id)

    assert_error_body(response, status=422, code="VALIDATION")
    assert EMAIL_OTHER not in response.text
    assert rehydration_stub.call_count == 0


# --------------------------------------------------------------------------
# the id boundary
# --------------------------------------------------------------------------


def test_a_thirty_two_character_hex_id_is_accepted(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, live_map_id: str
) -> None:
    r"""Catches a pre-check written one character too tight - `[0-9a-f]{31}` or a `< 32`
    comparison - which would refuse every id the store ever mints. The inclusive
    boundary: exactly 32 is accepted here, 31 and 33 are refused by the table below, and
    the id is asserted to be what `MAP_ID_PATTERN` accepts so the two cases are talking
    about the same rule."""
    rehydration_stub.response = stub_response(content="write to <EMAIL_1>")

    response = post_rehydrated(rehydration_client, chat_body(), map_id=live_map_id)

    assert len(live_map_id) == 32
    assert MAP_ID_PATTERN.fullmatch(live_map_id) is not None
    assert content_of(response) == f"write to {EMAIL_A}"


@pytest.mark.parametrize(
    "map_id",
    [
        pytest.param(WELL_FORMED_ID.upper(), id="32 UPPERCASE hex characters"),
        pytest.param(WELL_FORMED_ID[:31], id="31 characters, one under the boundary"),
        pytest.param(WELL_FORMED_ID + "0", id="33 characters, one over the boundary"),
        pytest.param(WELL_FORMED_ID + "\n", id="a trailing newline"),
        pytest.param(WELL_FORMED_ID + "\r\n", id="a trailing CRLF"),
        pytest.param(WELL_FORMED_ID + " ", id="a trailing space"),
        pytest.param("g" * 32, id="32 characters that are not hex"),
        pytest.param("", id="an empty header value"),
    ],
)
def test_the_map_id_rejection_table(
    rehydration_client: TestClient, rehydration_stub: RehydrationStub, map_id: str
) -> None:
    r"""Catches a pre-check written as `re.match` against `^[0-9a-f]{32}$`, which accepts
    every one of the trailing rows: in Python `$` also matches before a final newline, and
    `.match` is happy with anything after 32 valid characters. The row
    that proves the pre-check exists at all is `map_store.gets == []` - without it the
    store would be asked and would answer "no such map", giving the same 422 for a
    different reason."""
    assert MAP_ID_PATTERN.fullmatch(map_id) is None

    response = post_rehydrated(rehydration_client, chat_body(), map_id=map_id)

    assert_error_body(response, status=422, code="VALIDATION")
    assert map_store_reads(rehydration_client) == []
    assert rehydration_stub.call_count == 0


# --------------------------------------------------------------------------
# an unreachable map store
# --------------------------------------------------------------------------


@pytest.fixture
def unavailable_store_stub() -> RehydrationStub:
    """The adapter behind the broken-store app, so "nothing was dialled" is observable."""
    return RehydrationStub(name="anthropic")


@pytest.fixture
def unavailable_store_client(
    settings: Settings, unavailable_store_stub: RehydrationStub
) -> Iterator[TestClient]:
    """A second app whose map store cannot be read.

    Its own app rather than the module's: the store swap is the whole case, and
    doing it on `rehydration_client` would leave every later case here talking to
    a store that refuses every read.
    """
    adapters = {unavailable_store_stub.name: unavailable_store_stub}
    with build_chat_client(settings, adapters) as client:
        app_of(client).state.redaction_map_store = UnavailableMapStore()
        yield client


def test_an_unreachable_map_store_is_502_with_no_outbound_call(
    unavailable_store_client: TestClient, unavailable_store_stub: RehydrationStub
) -> None:
    """Catches a route that lets `RedactionMapStoreUnavailable` escape: it is not a
    `GatewayError`, so an uncaught one is a 500 with no `code` at all and a caller cannot
    tell a retryable dependency outage from a bug in its own request. It also catches a
    route that dials the provider anyway and rehydrates with nothing, which would bill a
    request the caller asked to have names put back into."""
    response = post_rehydrated(unavailable_store_client, chat_body(), map_id=WELL_FORMED_ID)

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert unavailable_store_stub.call_count == 0


def test_an_unreachable_map_store_on_a_stream_is_502_with_no_outbound_call(
    unavailable_store_client: TestClient, unavailable_store_stub: RehydrationStub
) -> None:
    """Catches a route that reads the map AFTER the stream has opened: the status is 200
    the moment `StreamingResponse` is returned, so a store outage would reach the caller
    as an error frame inside a successful stream - which an SDK reports as a parse
    failure - and the provider would have been dialled and billed for an answer the
    caller asked to have names put back into. The JSON content type is what proves the
    refusal came before the stream, and `call_count` that it came before the provider."""
    response = post_rehydrated(unavailable_store_client, stream_body(), map_id=WELL_FORMED_ID)

    assert_error_body(response, status=502, code="PROVIDER_ERROR")
    assert response.headers["content-type"].split(";")[0] == "application/json"
    assert unavailable_store_stub.call_count == 0


def test_a_map_store_failure_does_not_put_the_clients_words_in_front_of_a_caller(
    unavailable_store_client: TestClient,
) -> None:
    """Catches a conversion that pastes the store client's error into the client-facing
    message: that string names an internal host and port, and the same rule already holds
    on the write side (`tests/gateway/test_redact_endpoint.py`) and on the provider path
    (`test_provider_message_never_reaches_the_client_message`)."""
    response = post_rehydrated(unavailable_store_client, chat_body(), map_id=WELL_FORMED_ID)

    assert REDIS_FAILURE_TEXT not in response.text
    assert "6379" not in response.text
    assert "Connection refused" not in response.text


def test_the_rehydration_stub_satisfies_the_adapter_protocol() -> None:
    """Catches this module's own stub drifting from the `Adapter` contract - a renamed
    method, a dropped `name` - which would otherwise surface as the resolver's 503 in
    every case here and read as a route bug rather than as a fixture one. The annotation
    is the check `mypy --strict` makes; the two calls are what a runtime run sees."""
    stub: Adapter = RehydrationStub(name="anthropic")

    assert stub.price_key("claude-haiku-4-5") == "anthropic/claude-haiku-4-5"
    assert stub.healthcheck() is True
