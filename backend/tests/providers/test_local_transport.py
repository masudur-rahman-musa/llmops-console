"""The HTTP half of the OpenAI-compatible adapter: the endpoint, the auth and the retries.

The retry table, the backoff and the truncation live in `providers/transport.py`
and are already pinned for the neighbouring adapter. They are asserted again here
because this adapter has to REACH them: an adapter with its own `httpx.post` loop
would pass every mapping test in the suite and still hammer a busy local server,
and only a request count can tell the two apart.

Everything is scripted through `httpx.MockTransport`, so no socket opens; `sleep`
and `jitter` are injected, so the retry rows cost microseconds and the delays are
exact numbers rather than a range.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from app.core.errors import ProviderError
from app.providers.base import ChatMessage, ChatRequest, ResponseFormat
from app.providers.openai_compatible import (
    CHAT_COMPLETIONS_PATH,
    OpenAICompatibleAdapter,
)
from app.providers.transport import (
    CONNECT_TIMEOUT,
    MAX_RETRIES,
    PROVIDER_MESSAGE_LIMIT,
    READ_TIMEOUT,
    RETRY_BASE_DELAY,
)

from .conftest import (
    LOCAL_BASE_URL,
    LOCAL_PROVIDER_NAME,
    PROVIDER_DETAIL_MARKER,
    TEST_LOCAL_API_KEY,
    SleepRecorder,
    Step,
    load_local_fixture,
    local_fail,
    local_ok,
    make_local_adapter,
)

REQUEST = ChatRequest(
    model="qwen2.5-7b-instruct",
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)


def test_the_completions_path_is_appended_to_the_policy_base_url() -> None:
    """Catches a path carrying its own `/v1` (the shipped policy's base url already ends
    in one, so the request would go to `/v1/v1/chat/completions` and 404) and catches an
    adapter aimed at the Anthropic `/v1/messages` endpoint."""
    adapter, scripted = make_local_adapter([local_ok("plain")])

    response = adapter.chat(REQUEST)

    assert CHAT_COMPLETIONS_PATH == "/chat/completions"
    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://llama.test:8082/v1/chat/completions"
    assert sent.headers["content-type"].startswith("application/json")
    assert response.content == "Ship it."


def test_no_api_key_sends_no_authorization_header() -> None:
    """Catches an adapter that requires a key and sends `Authorization: Bearer None` (or
    an empty bearer) for the shipped `local` provider, which declares no `key_env` at all
    because it sits on a private network - decision B."""
    adapter, scripted = make_local_adapter([local_ok("plain")], api_key=None)

    adapter.chat(REQUEST)

    assert "authorization" not in scripted.requests[0].headers
    assert "x-api-key" not in scripted.requests[0].headers


def test_an_api_key_travels_only_as_a_bearer_header() -> None:
    """Catches a key sent the neighbouring provider's way (`x-api-key`, which an
    OpenAI-compatible server ignores, so a secured vLLM answers 401) and catches a key put
    in the query string, where every proxy on the path logs it."""
    adapter, scripted = make_local_adapter([local_ok("plain")], api_key=TEST_LOCAL_API_KEY)

    adapter.chat(REQUEST)

    sent = scripted.requests[0]
    assert sent.headers["authorization"] == f"Bearer {TEST_LOCAL_API_KEY}"
    assert TEST_LOCAL_API_KEY not in str(sent.url)


def test_client_timeouts_are_fixed_at_five_and_sixty() -> None:
    """Catches an adapter built with httpx's default timeout (5s for everything, so a
    normal local completion fails) or one that never sets `read` at all."""
    adapter, _ = make_local_adapter([local_ok("plain")])

    assert adapter.client.timeout.connect == CONNECT_TIMEOUT
    assert adapter.client.timeout.read == READ_TIMEOUT
    assert CONNECT_TIMEOUT == 5.0
    assert READ_TIMEOUT == 60.0


STATUS_CASES: list[tuple[str, int, str, int, bool]] = [
    ("400 bad request", 400, "error_400", 1, False),
    ("401 unauthorized", 401, "error_400", 1, False),
    ("404 not found", 404, "error_400", 1, False),
    ("422 unprocessable", 422, "error_400", 1, False),
    ("428 just under the retry row", 428, "error_400", 1, False),
    ("429 rate limited", 429, "error_429", 3, True),
    ("430 just over the retry row", 430, "error_400", 1, False),
    ("500 server error", 500, "error_500", 3, True),
    ("503 all slots busy", 503, "error_500", 3, True),
]


@pytest.mark.parametrize(
    ("case", "status", "fixture", "requests", "retryable"),
    STATUS_CASES,
    ids=[case for case, *_ in STATUS_CASES],
)
def test_the_retry_table_by_status(
    case: str,
    status: int,
    fixture: str,
    requests: int,
    retryable: bool,
    sleeps: SleepRecorder,
) -> None:
    """Catches an adapter that never reaches the shared transport at all (its own loop
    would retry everything or nothing), and a retry predicate written as `status >= 429`
    or `status != 200`, which the 428 and 430 rows either side of the single retryable 4xx
    are placed to expose."""
    adapter, scripted = make_local_adapter([local_fail(status, fixture)], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == requests, case
    assert raised.value.retryable is retryable, case
    assert len(sleeps.calls) == requests - 1, case
    assert raised.value.http_status == 502, case


def test_a_read_timeout_is_retried_twice_and_then_raises_retryable(sleeps: SleepRecorder) -> None:
    """Catches an adapter that only handles HTTP statuses: an unwrapped
    `httpx.ReadTimeout` reaching the route is a 500 with a traceback instead of the 502
    the routes' fallback loop can act on - and a local model that is merely slow is the single
    likeliest failure this provider has."""
    steps: list[Step] = [httpx.ReadTimeout]
    adapter, scripted = make_local_adapter(steps, sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == 3
    assert len(sleeps.calls) == 2
    assert MAX_RETRIES == 2, "2 retries means 3 attempts in total"
    assert raised.value.retryable is True


def test_the_local_server_being_unreachable_is_not_retried(sleeps: SleepRecorder) -> None:
    """Catches a transport that retries a `ConnectError`: the model container being down
    is not a condition half a second fixes, and three attempts only delay the 502 by the
    backoff. Catches the raw exception escaping too, which the route would answer as a 500
    with a traceback."""
    steps: list[Step] = [httpx.ConnectError]
    adapter, scripted = make_local_adapter(steps, sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == 1
    assert sleeps.calls == []
    assert raised.value.retryable is False
    assert raised.value.http_status == 502


def test_a_429_then_a_200_succeeds_after_exactly_two_requests(sleeps: SleepRecorder) -> None:
    """Catches an adapter that gives up on the first busy-slot answer, and one that
    retries the full three times regardless, by pinning the count at exactly 2."""
    adapter, scripted = make_local_adapter(
        [local_fail(429, "error_429"), local_ok("plain")], sleep=sleeps
    )

    response = adapter.chat(REQUEST)

    assert scripted.request_count == 2
    assert len(sleeps.calls) == 1, "one retry means one wait"
    assert response.usage.total_tokens == 48


JITTER_CASES: list[tuple[str, float, list[float]]] = [
    ("no jitter", 0.0, [RETRY_BASE_DELAY * 0.5, RETRY_BASE_DELAY * 2 * 0.5]),
    ("full jitter", 1.0, [RETRY_BASE_DELAY, RETRY_BASE_DELAY * 2]),
]


@pytest.mark.parametrize(
    ("case", "jitter", "expected"), JITTER_CASES, ids=[case for case, *_ in JITTER_CASES]
)
def test_backoff_doubles_and_is_jittered(
    case: str, jitter: float, expected: list[float], sleeps: SleepRecorder
) -> None:
    """Catches an adapter with a retry loop of its own rather than the shared transport's:
    a fixed sleep hits a busy server again at the same rate, and an unjittered one has
    every gateway replica retrying in lockstep."""
    adapter, _ = make_local_adapter([local_fail(500, "error_500")], sleep=sleeps, jitter=jitter)

    with pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    assert sleeps.calls == pytest.approx(expected), case
    assert RETRY_BASE_DELAY == 0.5


def test_a_retry_resends_the_same_body(sleeps: SleepRecorder) -> None:
    """Catches a retry that rebuilds or mutates the request between attempts, which would
    silently send the server something the caller never asked for."""
    adapter, scripted = make_local_adapter(
        [local_fail(429, "error_429"), local_ok("plain")], sleep=sleeps
    )

    adapter.chat(REQUEST)

    assert scripted.body(0) == scripted.body(1)


ERROR_FIXTURES: list[tuple[str, int, str]] = [
    ("a 400 body", 400, "error_400"),
    ("a 429 body", 429, "error_429"),
    ("a 500 body", 500, "error_500"),
]


@pytest.mark.parametrize(
    ("case", "status", "fixture"), ERROR_FIXTURES, ids=[case for case, *_ in ERROR_FIXTURES]
)
def test_the_provider_message_never_reaches_the_client_message(
    case: str, status: int, fixture: str, sleeps: SleepRecorder
) -> None:
    """Catches an adapter that raises `ProviderError(response.text)`, which would put the
    server's own wording - slot counts, file paths, model file names - into the body a
    tenant reads."""
    payload = load_local_fixture(fixture)
    assert PROVIDER_DETAIL_MARKER in payload["error"]["message"], (
        f"{fixture}: the fixture must carry the marker, or this test proves nothing"
    )
    adapter, _ = make_local_adapter([local_fail(status, fixture)], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    error = raised.value
    assert PROVIDER_DETAIL_MARKER not in error.message, f"{case}: provider text leaked"
    assert PROVIDER_DETAIL_MARKER not in str(error), f"{case}: provider text leaked via str()"
    assert error.provider_message is not None, f"{case}: the log loses the server's reason"
    assert PROVIDER_DETAIL_MARKER in error.provider_message, case


def test_the_client_message_names_the_policy_provider_and_the_status(
    sleeps: SleepRecorder,
) -> None:
    """Catches a message naming the vendor or the kind instead of the policy provider:
    `local` and `vllm` are the same kind on the same box, and an operator reading a
    failure has to be able to tell which one answered."""
    adapter, _ = make_local_adapter([local_fail(400, "error_400")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert LOCAL_PROVIDER_NAME in raised.value.message
    assert "400" in raised.value.message


def test_an_oversized_provider_body_is_truncated(sleeps: SleepRecorder) -> None:
    """Catches a dropped slice: a reverse proxy in front of the model answering 502 with a
    megabyte of HTML would then write a megabyte into the log on every failed attempt."""
    oversized = "x" * (PROVIDER_MESSAGE_LIMIT * 4)
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text=oversized))
    adapter = OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=transport,
        sleep=sleeps,
    )

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert raised.value.provider_message is not None
    assert len(raised.value.provider_message) == PROVIDER_MESSAGE_LIMIT
    assert PROVIDER_MESSAGE_LIMIT == 500


def test_a_non_json_success_body_is_a_provider_error(sleeps: SleepRecorder) -> None:
    """Catches the path a reverse proxy takes: a 200 carrying an HTML error page would
    otherwise raise `json.JSONDecodeError` out of the adapter, reaching the route as a 500
    with a traceback instead of the 502 the routes' fallback loop can move past."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>bad gateway</html>")
    )
    adapter = OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=transport,
        sleep=sleeps,
    )

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert raised.value.retryable is False


def test_price_key_is_the_policy_provider_name_slash_model() -> None:
    """Catches a price key built from the provider KIND: `local` and `vllm` are both
    `openai_compatible`, so a kind-keyed ledger would price two different boxes as one and
    make the cost of the local route unreadable."""
    adapter, _ = make_local_adapter([local_ok("plain")])

    assert adapter.price_key("qwen2.5-7b-instruct") == f"{LOCAL_PROVIDER_NAME}/qwen2.5-7b-instruct"
    assert adapter.name == LOCAL_PROVIDER_NAME


def test_close_releases_the_client() -> None:
    """Catches a `close()` that does not: `build_adapters` mints one client per adapter,
    so a policy reload that drops adapters without closing them leaks a connection pool."""
    adapter, _ = make_local_adapter([local_ok("plain")])

    adapter.close()

    assert adapter.client.is_closed


# --- a non-strict schema, and the 400 log ----------------------------

# The logger `ProviderTransport.status_error` writes the 400 line to.
TRANSPORT_LOGGER = "app.providers.transport"

# A schema OpenAI's strict mode refuses on two counts: `tags` is optional, and there
# is no `additionalProperties: false`. With `strict: true` added for the caller it is
# the 400 that broke the Helpdesk on every turn.
NON_STRICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "severity": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "severity"],
}

# Planted in the prompt and the key of the no-leak case, and nowhere else.
SENTINEL_PROMPT = "prompt-sentinel-9b41e7 tell nobody about the Kestrel account"
SENTINEL_KEY = "sk-key-sentinel-5d03aa-not-a-real-key"


def transport_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Every captured record from the transport's own logger."""
    return [record for record in caplog.records if record.name == TRANSPORT_LOGGER]


def raw_adapter(status: int, raw: bytes, sleeps: SleepRecorder) -> OpenAICompatibleAdapter:
    """A local adapter answering every request with exactly these bytes and this status."""
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=raw))
    return OpenAICompatibleAdapter(
        name=LOCAL_PROVIDER_NAME,
        base_url=LOCAL_BASE_URL,
        transport=transport,
        sleep=sleeps,
    )


def test_a_non_strict_schema_goes_through_to_openai() -> None:
    """Catches the bug this plan fixes: a mapper adding `strict: true` (or tightening the
    schema with `additionalProperties: false`) on the caller's behalf, which OpenAI answers
    with a 400 for this schema; and catches an answer whose JSON content is lost."""
    adapter, scripted = make_local_adapter([local_ok("json_schema_answer")])
    request = ChatRequest(
        model="qwen2.5-7b-instruct",
        messages=(ChatMessage(role="user", content="Summarise the ticket."),),
        response_format=ResponseFormat(name="ticket_summary", json_schema=NON_STRICT_SCHEMA),
    )

    response = adapter.chat(request)

    assert scripted.body()["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "ticket_summary", "schema": NON_STRICT_SCHEMA},
    }
    assert json.loads(response.content) == {"title": "Printer on fire", "severity": 3}


def test_a_400_logs_the_providers_message(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a 400 that reaches the gateway log as a bare 502 with no reason, which is
    how the strict-mode failure stayed invisible, and catches a line logged once per
    caller rather than once per failure."""
    provider_message = load_local_fixture("error_400")["error"]["message"]
    adapter, _ = make_local_adapter([local_fail(400, "error_400")], sleep=sleeps)

    with caplog.at_level(logging.WARNING, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    records = transport_records(caplog)
    assert len(records) == 1, [record.getMessage() for record in records]
    assert records[0].levelno == logging.WARNING
    logged = records[0].getMessage()
    assert LOCAL_PROVIDER_NAME in logged
    assert "400" in logged
    assert provider_message in logged
    assert PROVIDER_DETAIL_MARKER in logged


def test_the_400_log_carries_no_prompt_and_no_key(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a 400 log that writes the outbound request (its body or its headers) next
    to the provider's answer, which would put a tenant's prompt and the provider key in
    the gateway log."""
    adapter, _ = make_local_adapter(
        [local_fail(400, "error_400")], api_key=SENTINEL_KEY, sleep=sleeps
    )
    request = ChatRequest(
        model="qwen2.5-7b-instruct",
        messages=(ChatMessage(role="user", content=SENTINEL_PROMPT),),
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderError):
        adapter.chat(request)

    assert len(transport_records(caplog)) == 1, (
        "the 400 must be logged for this case to mean anything"
    )
    for record in caplog.records:
        logged = record.getMessage()
        assert SENTINEL_PROMPT not in logged, record.name
        assert SENTINEL_KEY not in logged, record.name


def test_the_400_log_is_cut_at_500(caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder) -> None:
    """Catches a log line with no cut (a provider echoing a large request writes all of it
    on every failure), a cut one character off, and a raw body slice logged in place of
    the parsed `error.message`, whose first 500 characters would then never appear."""
    message = ("upstream-cut-" + "a" * PROVIDER_MESSAGE_LIMIT)[:PROVIDER_MESSAGE_LIMIT] + "Z" * 100
    assert len(message) == 600
    assert message[PROVIDER_MESSAGE_LIMIT] != message[PROVIDER_MESSAGE_LIMIT - 1]
    adapter, _ = make_local_adapter([(400, {"error": {"message": message}})], sleep=sleeps)

    with caplog.at_level(logging.WARNING, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    records = transport_records(caplog)
    assert len(records) == 1
    logged = records[0].getMessage()
    assert message[:PROVIDER_MESSAGE_LIMIT] in logged
    assert message[: PROVIDER_MESSAGE_LIMIT + 1] not in logged
    assert PROVIDER_MESSAGE_LIMIT == 500


BODY_SENTINEL = "body-sentinel-71c0de"


def padded(prefix: str, suffix: str, length: int) -> bytes:
    """`prefix + filler + suffix` as exactly `length` bytes, so the logged length is a
    number no other part of the line can supply by accident."""
    filler = "x" * (length - len(prefix) - len(suffix))
    raw = (prefix + filler + suffix).encode("utf-8")
    assert len(raw) == length
    return raw


UNREADABLE_400_BODIES: list[tuple[str, bytes]] = [
    ("a plain-text body", padded(f"<html>{BODY_SENTINEL} ", "</html>", 1234)),
    ("JSON with no error key", padded(f'{{"detail": "{BODY_SENTINEL} ', '"}', 1357)),
    (
        "JSON whose error.message is not a string",
        padded(f'{{"error": {{"message": 42, "note": "{BODY_SENTINEL} ', '"}}', 1468),
    ),
]


@pytest.mark.parametrize(
    ("case", "raw"), UNREADABLE_400_BODIES, ids=[case for case, _ in UNREADABLE_400_BODIES]
)
def test_a_400_with_no_readable_message_logs_status_and_length_only(
    case: str, raw: bytes, caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a fallback that logs the raw body when no `error.message` can be read: a
    free-form body may echo the request, so only its length is safe to write."""
    adapter = raw_adapter(400, raw, sleeps)

    with caplog.at_level(logging.WARNING, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    records = transport_records(caplog)
    assert len(records) == 1, case
    logged = records[0].getMessage()
    assert LOCAL_PROVIDER_NAME in logged, case
    assert "400" in logged, case
    assert str(len(raw)) in logged, case
    assert BODY_SENTINEL not in logged, case


NON_400_STATUSES = [401, 429, 500]


@pytest.mark.parametrize("status", NON_400_STATUSES, ids=[str(s) for s in NON_400_STATUSES])
def test_a_500_logs_nothing_new(
    status: int, caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a log on every failure status rather than on 400 alone: a 401 body can
    quote part of the API key, and a retried 429 or 500 would write one line per attempt."""
    body = {"error": {"message": f"{PROVIDER_DETAIL_MARKER}: status {status}"}}
    adapter, _ = make_local_adapter([(status, body)], sleep=sleeps)

    with caplog.at_level(logging.DEBUG, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    assert transport_records(caplog) == []


def test_a_400_still_raises_the_same_client_message(sleeps: SleepRecorder) -> None:
    """Catches a 400 log that also changes what the caller is told: the provider's words
    moving into `message`, or the 400 turning retryable and being billed three times."""
    adapter, scripted = make_local_adapter([local_fail(400, "error_400")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert raised.value.message == f"{LOCAL_PROVIDER_NAME}: the provider answered 400"
    assert raised.value.retryable is False
    assert scripted.request_count == 1
