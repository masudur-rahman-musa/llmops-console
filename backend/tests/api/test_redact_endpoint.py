"""`POST /v1/redact`: the two body shapes, one shared numbering, and the refusals.

The endpoint is the ingestion path: a caller hands text over, gets it back with
every detected value replaced, and keeps a `redaction_map_id` to rehydrate with
later. So the cases here are grouped by what could break that promise.

* The ANSWER: the text shape and the messages shape, the echoed data class, and
  a map id in the store's published rendering.
* One MAP and one NUMBERING across every string in the call. `<EMAIL_1>` has to
  mean the same value in message 1 and message 4, so the rows are arranged with
  the same address in two non-adjacent messages and a second address between
  them - a per-message engine call renumbers the second address to `<EMAIL_1>`
  and reddens, while a shared one cannot.
* The REFUSALS: neither-and-both, an empty list, an unknown class, and the cap
  pair. Every refused row asserts that NO map was written, against a store that
  records its writes, because a validation that runs after the store write is
  not the "before any effect" the plan asks for.

`redact_batch` and `redaction_applies` are exercised directly as well: the route
composes them, and a claim about numbering across a 64-item batch is cheaper and
sharper as a function call than as 64 HTTP requests.

No provider key is read and no adapter is built (#66): this route resolves
nothing, so its client is built with an empty adapter set.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app.api.routes.redact import REDACT
from app.core.config import Settings
from app.model_routing.policy import DataClass, Policy
from app.redaction import (
    MAX_REDACTION_INPUT_CHARS,
    REDACTION_MAP_ID_BYTES,
    RedactionInputTooLong,
    RedactionMapStore,
    RedactionMapStoreUnavailable,
)
from app.services.redaction_flow import RedactedBatch, redact_batch, redaction_applies

from .conftest import TEST_KEYS, assert_error_body, auth_headers, build_chat_client

EMAIL_A = "alice@example.com"
EMAIL_B = "bob@example.net"
IPV4_A = "192.0.2.1"

# `<KIND_N>`: the kind, an underscore, a 1-based index. Restated here rather than
# imported from `tests/redaction/conftest.py` so this directory's expectations do
# not move when that fixture module does.
PLACEHOLDER_PATTERN = re.compile(r"<([A-Z0-9]+(?:_[A-Z0-9]+)*)_([0-9]+)>")
HEX_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

NO_ADAPTERS: Mapping[str, Any] = {}


def rehydrate(text: str, mapping: Mapping[str, str]) -> str:
    """Substitute every placeholder in `mapping` back into `text`.

    Plain replacement, exactly as an earlier change's rehydration will do it: if the map is
    a total inverse, this reproduces the caller's original string.
    """
    restored = text
    for placeholder, value in mapping.items():
        restored = restored.replace(placeholder, value)
    return restored


def placeholders(text: str) -> list[str]:
    """Every `<KIND_N>` token in `text`, in order of appearance."""
    return [match.group(0) for match in PLACEHOLDER_PATTERN.finditer(text)]


@dataclass
class RecordingStore:
    """The real store with every `put` recorded, so "no write" is observable.

    Wraps rather than replaces, because the read half has to keep working: the
    round-trip case asks the same object for the map the route just wrote.
    """

    inner: RedactionMapStore
    writes: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        self.writes.append((tenant, dict(mapping)))
        return self.inner.put(tenant, mapping)

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        return self.inner.get(tenant, map_id)


MAP_STORE_ATTRIBUTE = "redaction_map_store"
"""The `app.state` attribute `create_app` puts the placeholder map store under.

Named, not found by shape. This helper used to select "the one state object with
a `put` and a `get`" and assert there was exactly one; an earlier change's tenant
directory store has both, so every case in this file broke the moment a second
store was wired - and no grep for a name could have predicted it, because the
selection named nothing. A hardcoded attribute is a wiring
detail this suite now pins on purpose: it is the price of being able to tell two
stores of the same shape apart.
"""


def store_attribute(app: FastAPI) -> str:
    """Check the map store really is on `app.state` under its published name.

    A missing store fails here with that sentence rather than as an opaque
    `AttributeError` three fixtures later.
    """
    store = getattr(app.state, MAP_STORE_ATTRIBUTE, None)
    assert callable(getattr(store, "put", None)) and callable(getattr(store, "get", None)), (
        f"expected a redaction map store on app.state.{MAP_STORE_ATTRIBUTE}, found {store!r}"
    )
    return MAP_STORE_ATTRIBUTE


@pytest.fixture(scope="module")
def redact_client(settings: Settings) -> Iterator[TestClient]:
    """The shipped app with no adapters: `/v1/redact` calls no provider."""
    with build_chat_client(settings, NO_ADAPTERS) as client:
        yield client


@pytest.fixture(scope="module")
def redact_store(redact_client: TestClient) -> RecordingStore:
    """The app's own store, wrapped once so every write in this module is recorded."""
    app = cast(FastAPI, redact_client.app)
    name = store_attribute(app)
    recorder = RecordingStore(inner=cast(RedactionMapStore, getattr(app.state, name)))
    setattr(app.state, name, recorder)
    return recorder


@pytest.fixture(autouse=True)
def _fresh_writes(redact_store: RecordingStore) -> Iterator[None]:
    """Per-test isolation by clearing the record, never by rebuilding the app."""
    redact_store.writes.clear()
    yield


def post_redact(
    client: TestClient,
    body: Mapping[str, Any],
    *,
    key: str = TEST_KEYS["evals"],
    tenant: str = "evals",
    data_class: str | None = None,
    authorization: str | None = None,
) -> Response:
    """POST one redaction request as the `evals` tenant, with no class header by default."""
    response: Response = client.post(
        REDACT,
        json=dict(body),
        headers=auth_headers(
            key=key, tenant=tenant, data_class=data_class, authorization=authorization
        ),
    )
    return response


def message(content: str, role: str = "user") -> dict[str, str]:
    """One turn in the shape `/v1/redact` takes."""
    return {"role": role, "content": content}


# --------------------------------------------------------------------------
# the answer
# --------------------------------------------------------------------------


def test_the_route_path_is_the_published_one(redact_client: TestClient) -> None:
    """Catches a route registered at a near-miss path (`/v1/redaction`, or under the
    chat prefix): the constant and the mounted path are asserted together, so a module
    that renamed one without the other cannot pass."""
    assert REDACT == "/v1/redact"

    response = post_redact(redact_client, {"text": "nothing to see"})

    assert response.status_code == 200, response.text


def test_a_text_request_returns_redacted_text_and_a_map_id(redact_client: TestClient) -> None:
    """Catches a route that echoes the caller's text back unredacted (or returns the
    placeholder text but no map id): both halves are the endpoint's whole product, and
    the raw address is asserted absent rather than the placeholder merely present."""
    response = post_redact(redact_client, {"text": f"mail {EMAIL_A} today"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "mail <EMAIL_1> today"
    assert EMAIL_A not in body["text"]
    assert body["messages"] is None
    assert HEX_ID_PATTERN.fullmatch(body["redaction_map_id"]) is not None
    assert len(body["redaction_map_id"]) == REDACTION_MAP_ID_BYTES * 2


def test_a_messages_request_returns_every_turn_in_order_with_its_role(
    redact_client: TestClient,
) -> None:
    """Catches a route that flattens the turns into one string, drops the roles, or
    reorders them: a caller replays these messages to a model, so a lost `assistant`
    label or a swapped pair changes the conversation it sends."""
    response = post_redact(
        redact_client,
        {
            "messages": [
                message(f"I am {EMAIL_A}"),
                message("Thanks, noted.", role="assistant"),
                message(f"and {IPV4_A} is my box"),
            ]
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] is None
    assert [turn["role"] for turn in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][0]["content"] == "I am <EMAIL_1>"
    assert body["messages"][1]["content"] == "Thanks, noted."
    assert body["messages"][2]["content"] == "and <IPV4_1> is my box"


def test_one_numbering_is_shared_across_every_message(redact_client: TestClient) -> None:
    """Catches a route calling the engine once per message: the same address appears in
    message 0 and message 3 with a DIFFERENT address between them, so a per-message
    numbering makes message 2 `<EMAIL_1>` too and the map stops being an inverse."""
    response = post_redact(
        redact_client,
        {
            "messages": [
                message(f"from {EMAIL_A}"),
                message("no personal data here", role="assistant"),
                message(f"cc {EMAIL_B}"),
                message(f"still {EMAIL_A}"),
            ]
        },
    )

    assert response.status_code == 200, response.text
    contents = [turn["content"] for turn in response.json()["messages"]]
    assert contents[0] == "from <EMAIL_1>"
    assert contents[2] == "cc <EMAIL_2>"
    assert contents[3] == "still <EMAIL_1>"


def test_the_stored_map_rehydrates_every_message_exactly(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a map stored with only the last message's entries, or keyed on a tenant
    the request did not carry: every placeholder in the answer is substituted back and
    compared with the caller's own strings, so a partial map leaves a `<KIND_N>` behind."""
    originals = [f"from {EMAIL_A}", f"cc {EMAIL_B} and {IPV4_A}", f"still {EMAIL_A}"]

    response = post_redact(redact_client, {"messages": [message(text) for text in originals]})

    assert response.status_code == 200, response.text
    body = response.json()
    stored = redact_store.get("evals", body["redaction_map_id"])
    assert stored is not None
    for original, turn in zip(originals, body["messages"], strict=True):
        assert rehydrate(turn["content"], stored) == original


def test_the_map_is_written_under_the_requesting_tenant(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a store write keyed on a constant or on the data class: the id is read
    back under the OTHER tenant as well, so a global key would answer there too and hand
    one tenant's PII map to another."""
    response = post_redact(
        redact_client,
        {"text": f"mail {EMAIL_A}"},
        key=TEST_KEYS["nordlicht"],
        tenant="nordlicht",
    )

    assert response.status_code == 200, response.text
    map_id = response.json()["redaction_map_id"]
    assert [tenant for tenant, _ in redact_store.writes] == ["nordlicht"]
    assert redact_store.get("nordlicht", map_id) == {"<EMAIL_1>": EMAIL_A}
    assert redact_store.get("evals", map_id) is None


def test_the_response_carries_no_gateway_headers(redact_client: TestClient) -> None:
    """Catches a route that reused the chat route's header builder: there is no resolver
    result here, so an `X-Gateway-Provider` or `X-Gateway-Region` on this answer would be
    a claim about where data went that nothing decided."""
    response = post_redact(redact_client, {"text": f"mail {EMAIL_A}"})

    assert response.status_code == 200, response.text
    assert [name for name in response.headers if name.lower().startswith("x-gateway-")] == []


# --------------------------------------------------------------------------
# the data class
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header_class", "expected"),
    [(None, "confidential"), ("public", "public"), ("pii", "pii"), ("internal", "internal")],
)
def test_the_header_data_class_applies_when_the_body_omits_it(
    redact_client: TestClient, header_class: str | None, expected: str
) -> None:
    """Catches a response that echoes a hardcoded class, and an endpoint that ignores
    `X-Data-Class` entirely: the absent row also pins the context default, so a route
    substituting `public` for a missing header would widen every unlabelled call."""
    response = post_redact(redact_client, {"text": "nothing here"}, data_class=header_class)

    assert response.status_code == 200, response.text
    assert response.json()["data_class"] == expected


def test_the_body_data_class_wins_over_the_header(redact_client: TestClient) -> None:
    """Catches an implementation reading `X-Data-Class` and ignoring the body field
    (decision E): the two are set to DIFFERENT classes, so whichever the route echoed is
    visible - a route honouring only the header answers `pii` here."""
    response = post_redact(
        redact_client, {"text": "nothing here", "data_class": "public"}, data_class="pii"
    )

    assert response.status_code == 200, response.text
    assert response.json()["data_class"] == "public"


def test_the_endpoint_redacts_on_a_class_whose_rule_carries_no_redact_flag(
    redact_client: TestClient, policy: Policy
) -> None:
    """Catches an endpoint that honours the tenant's `redact` flag (decision D): `public`
    is asserted to be a non-redact class from the shipped policy first, so this cannot
    pass by accident, and an explicit ingestion call must never be a silent no-op."""
    assert not redaction_applies(policy, "evals", DataClass.PUBLIC)

    response = post_redact(redact_client, {"text": f"mail {EMAIL_A}", "data_class": "public"})

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "mail <EMAIL_1>"


def test_an_unknown_body_data_class_is_422_before_any_store_write(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a body class coerced onto the default instead of refused: silently
    treating `secret` as `confidential` labels a map with a class the caller never
    asked for, and an earlier change reads that label back."""
    response = post_redact(redact_client, {"text": "hello", "data_class": "secret"})

    assert_error_body(response, status=422, code="VALIDATION")
    assert redact_store.writes == []


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": None, "messages": None},
        {"text": "hello", "messages": [{"role": "user", "content": "hello"}]},
        {"messages": []},
    ],
    ids=["neither", "both-explicitly-null", "both", "empty-list"],
)
def test_a_body_that_is_not_exactly_one_shape_is_422_before_any_store_write(
    redact_client: TestClient, redact_store: RecordingStore, body: dict[str, Any]
) -> None:
    """Catches an exactly-one-of rule enforced in the route body after the map was
    already written, and a route that quietly prefers `text` when both are sent - the
    recorded write list is what separates "refused" from "refused afterwards"."""
    response = post_redact(redact_client, body)

    assert_error_body(response, status=422, code="VALIDATION")
    assert redact_store.writes == []


@pytest.mark.parametrize("shape", ["text", "messages"])
def test_a_string_at_the_cap_is_accepted(redact_client: TestClient, shape: str) -> None:
    """Catches a cap written `>=` (the inclusive half of the boundary pair): 16384
    characters is a legal input, and a route refusing it would 422 a request the plan
    says it must serve."""
    at_cap = "a" * MAX_REDACTION_INPUT_CHARS
    body = {"text": at_cap} if shape == "text" else {"messages": [message(at_cap)]}

    response = post_redact(redact_client, body)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("shape", ["text", "messages"])
def test_a_string_over_the_cap_is_422_before_any_store_write(
    redact_client: TestClient, redact_store: RecordingStore, shape: str
) -> None:
    """Catches a cap that is not enforced at all, and one enforced only on `text` while
    a 64-message batch walks past it: paired with the row above, only an inclusive
    per-string cap passes both."""
    over_cap = "a" * (MAX_REDACTION_INPUT_CHARS + 1)
    body = {"text": over_cap} if shape == "text" else {"messages": [message(over_cap)]}

    response = post_redact(redact_client, body)

    assert_error_body(response, status=422, code="VALIDATION")
    assert redact_store.writes == []


def test_the_cap_applies_to_a_later_message_too(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a cap checked on `messages[0]` alone: the over-length string is the
    SECOND turn and the first is short, so a loop that stops after one item accepts the
    body and writes the map."""
    response = post_redact(
        redact_client,
        {
            "messages": [
                message("short"),
                message("a" * (MAX_REDACTION_INPUT_CHARS + 1)),
            ]
        },
    )

    assert_error_body(response, status=422, code="VALIDATION")
    assert redact_store.writes == []


def test_two_messages_each_at_the_cap_are_accepted(redact_client: TestClient) -> None:
    """Catches a cap applied to the JOINED request rather than per string (decision F):
    32768 characters across two legal turns must pass, and a joined cap refuses exactly
    this body while passing every single-string row above."""
    at_cap = "a" * MAX_REDACTION_INPUT_CHARS

    response = post_redact(redact_client, {"messages": [message(at_cap), message(at_cap)]})

    assert response.status_code == 200, response.text


def test_an_unknown_field_in_the_body_is_422(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a request model built without `extra="forbid"`: a misspelled `texts`
    would otherwise be ignored, and the caller would get a 200 having redacted nothing
    they sent."""
    response = post_redact(redact_client, {"text": "hello", "texts": ["hello"]})

    assert_error_body(response, status=422, code="VALIDATION")
    assert redact_store.writes == []


def test_a_request_with_no_key_is_401_before_any_store_write(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a router registered without the `Context` dependency: an unauthenticated
    caller would otherwise get a map id and a redaction service for free, on the one
    endpoint whose input is by definition personal data."""
    response = redact_client.post(REDACT, json={"text": f"mail {EMAIL_A}"})

    assert_error_body(response, status=401, code="UNAUTHORIZED")
    assert redact_store.writes == []


def test_a_tenant_that_does_not_match_the_key_is_403(
    redact_client: TestClient, redact_store: RecordingStore
) -> None:
    """Catches a route that takes the tenant from the `X-Tenant` header instead of from
    the authenticated key: the map is keyed by tenant, so the mismatch has to be refused
    here rather than quietly filed under whichever name the client claimed."""
    response = post_redact(redact_client, {"text": "hello"}, key=TEST_KEYS["evals"], tenant="cuan")

    assert_error_body(response, status=403, code="TENANT_MISMATCH")
    assert redact_store.writes == []


# --------------------------------------------------------------------------
# redact_batch
# --------------------------------------------------------------------------


def test_the_batch_returns_one_text_per_input_in_order() -> None:
    """Catches a batch that drops the strings holding nothing to redact, or returns them
    in map order: the caller matches turns back by position and by nothing else."""
    batch = redact_batch([f"a {EMAIL_A}", "b", f"c {IPV4_A}", "d"])

    assert len(batch.texts) == 4
    assert batch.texts[1] == "b"
    assert batch.texts[3] == "d"


def test_one_numbering_spans_every_string_in_the_batch() -> None:
    """Catches a per-string engine call: the same address is in strings 0 and 3 with a
    different one in string 2, so per-string numbering hands `<EMAIL_1>` to two different
    values and the shared map stops inverting."""
    batch = redact_batch([f"from {EMAIL_A}", "nothing", f"cc {EMAIL_B}", f"still {EMAIL_A}"])

    assert batch.texts[0] == "from <EMAIL_1>"
    assert batch.texts[2] == "cc <EMAIL_2>"
    assert batch.texts[3] == "still <EMAIL_1>"
    assert dict(batch.mapping) == {"<EMAIL_1>": EMAIL_A, "<EMAIL_2>": EMAIL_B}


def test_the_batch_reports_redaction_only_when_a_value_was_replaced() -> None:
    """Catches a `redacted` flag wired to "the rule applied" rather than to the payload
    (decision C): the header a console sums comes from this property, so a constant True
    would count a redaction on every clean request."""
    replaced = redact_batch([f"mail {EMAIL_A}"])
    untouched = redact_batch(["nothing personal here", "nor here"])

    assert replaced.redacted is True
    assert untouched.redacted is False
    assert untouched.mapping == {}
    assert untouched.texts == ("nothing personal here", "nor here")


def test_a_placeholder_written_in_one_string_is_never_handed_out_for_another() -> None:
    """Catches a taken-token set computed per string: `<EMAIL_1>` is written literally in
    string 0 and the real address is in string 1, so a per-string scan hands `<EMAIL_1>`
    out again and rehydrating string 0 injects a value the caller never wrote."""
    batch = redact_batch(["a token <EMAIL_1> already", f"mail {EMAIL_A}"])

    assert batch.texts[0] == "a token <EMAIL_1> already"
    assert "<EMAIL_1>" not in placeholders(batch.texts[1])
    assert batch.mapping == {"<EMAIL_2>": EMAIL_A}
    assert rehydrate(batch.texts[0], batch.mapping) == "a token <EMAIL_1> already"
    assert rehydrate(batch.texts[1], batch.mapping) == f"mail {EMAIL_A}"


@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_string_over_the_cap_raises_wherever_it_sits(position: int) -> None:
    """Catches a cap checked on the first string only: the over-length string moves
    through every position, so a `texts[0]`-shaped check passes at 0 and reddens at 1
    and 2."""
    texts = ["short", "short", "short"]
    texts[position] = "a" * (MAX_REDACTION_INPUT_CHARS + 1)

    with pytest.raises(RedactionInputTooLong):
        redact_batch(texts)


def test_strings_at_the_cap_are_accepted_however_many_there_are() -> None:
    """Catches a joined cap (decision F) and an exclusive comparison at once: two strings
    of exactly 16384 characters are 32768 joined, so a joined cap refuses them, and a
    `>=` refuses either one alone."""
    at_cap = "a" * MAX_REDACTION_INPUT_CHARS

    batch = redact_batch([at_cap, at_cap])

    assert batch.texts == (at_cap, at_cap)
    assert batch.redacted is False


def test_the_batch_is_frozen() -> None:
    """Catches a mutable result object: the route writes `mapping` to the store and
    returns `texts` to the caller, so a later stage able to rewrite either would put a
    map and an answer out of step with no error anywhere."""
    batch = redact_batch([f"mail {EMAIL_A}"])

    assert isinstance(batch, RedactedBatch)
    with pytest.raises(AttributeError):
        batch.texts = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# redaction_applies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tenant", ["nordlicht", "cuan", "evals"])
@pytest.mark.parametrize(
    ("data_class", "expected"),
    [
        (DataClass.PUBLIC, False),
        (DataClass.INTERNAL, False),
        (DataClass.CONFIDENTIAL, True),
        (DataClass.PII, True),
    ],
)
def test_redaction_applies_reads_the_tenants_rule_for_the_class(
    policy: Policy, tenant: str, data_class: DataClass, expected: bool
) -> None:
    """Catches a predicate that answers from the class name alone, or from the tenant
    alone: every shipped tenant is swept, so a hardcoded True (redacting `public`
    traffic) and a hardcoded False (sending `pii` in the clear) both redden."""
    assert redaction_applies(policy, tenant, data_class) is expected


def test_redaction_applies_is_false_for_a_tenant_the_policy_does_not_define(
    policy: Policy,
) -> None:
    """Catches a lookup that raises `KeyError` on an unknown tenant, which would be a
    500 with no typed code; the resolver has already denied such a request, so the
    honest answer here is False rather than an exception."""
    assert redaction_applies(policy, "no-such-tenant", DataClass.PII) is False


# --- an unreachable map store --------------------------
#
# The store is an outbound dependency like a provider, so a Redis that cannot be
# reached is 502 PROVIDER_ERROR rather than a 500 with no code. Asserted through
# `TestClient` against the built app: the conversion the route
# performs is invisible to a test that reads the handler, because FastAPI has
# handlers nobody in this repo registered.

REDIS_FAILURE_TEXT = "Error 111 connecting to redis:6379. Connection refused."
"""The client's own words, planted as a marker: they describe internal topology
(a host name and a port) and must not travel to a caller."""


class UnavailableStore:
    """A `RedactionMapStore` whose `put` fails the way an unreachable Redis does."""

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        raise RedactionMapStoreUnavailable(REDIS_FAILURE_TEXT)

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        return None


@pytest.fixture
def unavailable_store_client(settings: Settings) -> Iterator[TestClient]:
    """A second app whose map store cannot be reached.

    Its own app rather than the module's: the store swap is what the case is
    about, and doing it on `redact_client` would leave every later test in this
    file talking to a store that refuses every write.
    """
    with build_chat_client(settings, NO_ADAPTERS) as client:
        app = cast(FastAPI, client.app)
        setattr(app.state, store_attribute(app), UnavailableStore())
        yield client


def test_a_store_that_cannot_be_reached_is_502_with_a_typed_code(
    unavailable_store_client: TestClient,
) -> None:
    """Catches a route that lets the store's failure escape: `RedactionMapStoreUnavailable`
    is not a `GatewayError`, so an uncaught one is a 500 whose body carries no `code` at
    all, and a client cannot tell a retryable dependency outage from a bug in its own
    request."""
    response = post_redact(unavailable_store_client, {"text": f"mail {EMAIL_A} today"})

    assert_error_body(response, status=502, code="PROVIDER_ERROR")


def test_a_store_failure_does_not_put_the_clients_words_in_front_of_a_caller(
    unavailable_store_client: TestClient,
) -> None:
    """Catches a conversion that pastes the Redis error into the client-facing message:
    that string names an internal host and port, and the same rule already holds on the
    provider path (`test_provider_message_never_reaches_the_client_message`)."""
    response = post_redact(unavailable_store_client, {"text": f"mail {EMAIL_A} today"})

    assert REDIS_FAILURE_TEXT not in response.text
    assert "6379" not in response.text
    assert "Connection refused" not in response.text
