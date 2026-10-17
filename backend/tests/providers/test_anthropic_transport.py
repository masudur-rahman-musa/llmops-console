"""The HTTP half of the adapter: fixed timeouts, the retry table and the error boundary.

Every interaction is scripted through `httpx.MockTransport`, so no socket opens
and no key is read from the environment. `sleep` and `jitter` are injected, so
the retry rows cost microseconds and the backoff delays are exact numbers rather
than a range.

The counts are the point of this file. "Retries on a 429" is not a claim a test
can make by asserting a happy answer; only the exact number of requests made
separates one retry from three, and only the exact number of sleeps separates a
retry loop that waits from one that hammers the provider.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.errors import ProviderError
from app.providers.anthropic import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_VERSION,
    AnthropicAdapter,
)
from app.providers.base import ChatMessage, ChatRequest
from app.providers.transport import (
    CONNECT_TIMEOUT,
    MAX_RETRIES,
    PROVIDER_MESSAGE_LIMIT,
    READ_TIMEOUT,
    RETRY_BASE_DELAY,
)

from .conftest import (
    PROVIDER_DETAIL_MARKER,
    TEST_API_KEY,
    TEST_PROVIDER_NAME,
    SleepRecorder,
    Step,
    fail,
    load_fixture,
    make_adapter,
    ok,
)

REQUEST = ChatRequest(
    model="claude-haiku-4-5",
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)


def test_timeout_constants_are_exactly_the_spec_values() -> None:
    """Catches a constant drifting from the design (a 30s read is the httpx-tutorial value),
    which would cut off a long answer the caller paid for."""
    assert CONNECT_TIMEOUT == 5.0
    assert READ_TIMEOUT == 60.0
    assert MAX_RETRIES == 2, "2 retries means 3 attempts in total"


def test_client_timeouts_are_fixed_at_five_and_sixty() -> None:
    """Catches an adapter built with httpx's default timeout (5s for everything, so a
    normal 20s completion fails) or one that never sets `read` at all."""
    adapter, _ = make_adapter([ok("plain")])

    assert adapter.client.timeout.connect == 5.0
    assert adapter.client.timeout.read == 60.0


def test_a_successful_call_sends_one_request_to_the_messages_endpoint() -> None:
    """Catches a request aimed at the OpenAI path (`/v1/chat/completions`) or missing the
    `anthropic-version` header, both of which Anthropic answers with a 400."""
    adapter, scripted = make_adapter([ok("plain")])

    response = adapter.chat(REQUEST)

    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert sent.url.path == "/v1/messages"
    assert sent.headers["x-api-key"] == TEST_API_KEY
    assert sent.headers["anthropic-version"] == ANTHROPIC_VERSION
    assert sent.headers["content-type"].startswith("application/json")
    assert response.content == "Ship it."


def test_the_key_travels_only_in_the_x_api_key_header() -> None:
    """Catches an adapter sending the key as an OpenAI-style bearer token, which Anthropic
    ignores and answers 401 - and which would also put the key in a header proxies log."""
    adapter, scripted = make_adapter([ok("plain")])

    adapter.chat(REQUEST)

    assert "authorization" not in scripted.requests[0].headers
    assert TEST_API_KEY not in str(scripted.requests[0].url)


def test_429_then_200_succeeds_after_exactly_two_requests(sleeps: SleepRecorder) -> None:
    """Catches an adapter that gives up on the first 429 (no retry at all) and one that
    retries the full three times regardless, by pinning the count at exactly 2."""
    adapter, scripted = make_adapter([fail(429, "error_429"), ok("plain")], sleep=sleeps)

    response = adapter.chat(REQUEST)

    assert scripted.request_count == 2
    assert len(sleeps.calls) == 1, "one retry means one wait"
    assert response.content == "Ship it."
    assert response.usage.total_tokens == 48


def test_persistent_500_raises_retryable_after_exactly_three_requests(
    sleeps: SleepRecorder,
) -> None:
    """Catches an off-by-one retry loop: `range(MAX_RETRIES)` makes two attempts, not
    three, and the count is the only assertion that tells the two apart."""
    adapter, scripted = make_adapter([fail(500, "error_500")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == 3
    assert len(sleeps.calls) == 2, "two retries means two waits"
    assert raised.value.retryable is True
    assert raised.value.http_status == 502


def test_400_raises_non_retryable_after_exactly_one_request(sleeps: SleepRecorder) -> None:
    """Catches a retry loop that retries every failure: a 400 is our mapping's fault, and
    retrying it bills the tenant three times for a request that can never succeed."""
    adapter, scripted = make_adapter([fail(400, "error_400")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == 1
    assert sleeps.calls == [], "a non-retryable failure must not wait"
    assert raised.value.retryable is False


# status -> (requests expected, retryable). The 4xx rows either side of 429 are
# the boundary of the retry table.
STATUS_CASES: list[tuple[str, int, str, int, bool]] = [
    ("400 bad request", 400, "error_400", 1, False),
    ("401 unauthorized", 401, "error_400", 1, False),
    ("403 forbidden", 403, "error_400", 1, False),
    ("404 not found", 404, "error_400", 1, False),
    ("422 unprocessable", 422, "error_400", 1, False),
    ("428 just under the retry row", 428, "error_400", 1, False),
    ("429 rate limited", 429, "error_429", 3, True),
    ("430 just over the retry row", 430, "error_400", 1, False),
    ("500 server error", 500, "error_500", 3, True),
    ("503 unavailable", 503, "error_500", 3, True),
    ("529 overloaded", 529, "error_500", 3, True),
]


@pytest.mark.parametrize(
    ("case", "status", "fixture", "requests", "retryable"),
    STATUS_CASES,
    ids=[case for case, *_ in STATUS_CASES],
)
def test_retry_table_by_status(
    case: str,
    status: int,
    fixture: str,
    requests: int,
    retryable: bool,
    sleeps: SleepRecorder,
) -> None:
    """Catches a retry predicate written as `status >= 429` or `status != 200`, which the
    428 and 430 rows either side of the single retryable 4xx are placed to expose."""
    adapter, scripted = make_adapter([fail(status, fixture)], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == requests, case
    assert raised.value.retryable is retryable, case
    assert len(sleeps.calls) == requests - 1, case


TIMEOUT_CASES: list[tuple[str, type[httpx.RequestError]]] = [
    ("a read timeout", httpx.ReadTimeout),
    ("a connect timeout", httpx.ConnectTimeout),
]


@pytest.mark.parametrize(
    ("case", "exception"), TIMEOUT_CASES, ids=[case for case, _ in TIMEOUT_CASES]
)
def test_a_timeout_is_retried_and_raises_retryable(
    case: str, exception: type[httpx.RequestError], sleeps: SleepRecorder
) -> None:
    """Catches an adapter that only handles HTTP statuses: an unwrapped `httpx.ReadTimeout`
    escaping to the route is a 500 with a traceback instead of a 502 the routes' fallback
    loop can move past."""
    steps: list[Step] = [exception]
    adapter, scripted = make_adapter(steps, sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert scripted.request_count == 3, case
    assert len(sleeps.calls) == 2, case
    assert raised.value.retryable is True, case


def test_a_timeout_followed_by_success_returns_the_answer(sleeps: SleepRecorder) -> None:
    """Catches a loop that treats a timeout as fatal after recording it as retryable: the
    flag is worth nothing if the next attempt never happens."""
    steps: list[Step] = [httpx.ReadTimeout, ok("plain")]
    adapter, scripted = make_adapter(steps, sleep=sleeps)

    response = adapter.chat(REQUEST)

    assert scripted.request_count == 2
    assert response.content == "Ship it."


ERROR_FIXTURES: list[tuple[str, int, str]] = [
    ("a 400 body", 400, "error_400"),
    ("a 429 body", 429, "error_429"),
    ("a 500 body", 500, "error_500"),
]


@pytest.mark.parametrize(
    ("case", "status", "fixture"), ERROR_FIXTURES, ids=[case for case, *_ in ERROR_FIXTURES]
)
def test_provider_message_never_reaches_the_client_message(
    case: str, status: int, fixture: str, sleeps: SleepRecorder
) -> None:
    """Catches an adapter that raises `ProviderError(response.text)`, which would put the
    provider's internal request ids and quota wording into the body a tenant reads."""
    payload = load_fixture(fixture)
    assert PROVIDER_DETAIL_MARKER in payload["error"]["message"], (
        f"{fixture}: the fixture must carry the marker, or this test proves nothing"
    )
    adapter, _ = make_adapter([fail(status, fixture)], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    error = raised.value
    assert PROVIDER_DETAIL_MARKER not in error.message, f"{case}: provider text leaked"
    assert PROVIDER_DETAIL_MARKER not in str(error), f"{case}: provider text leaked via str()"
    assert error.provider_message is not None, f"{case}: the log loses the provider's reason"
    assert PROVIDER_DETAIL_MARKER in error.provider_message, case


def test_client_message_names_the_provider_and_the_status(sleeps: SleepRecorder) -> None:
    """Catches a generic "provider error" message: an operator reading the response has to
    be able to tell which policy provider failed and with what status."""
    adapter, _ = make_adapter([fail(400, "error_400")], sleep=sleeps)

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert TEST_PROVIDER_NAME in raised.value.message
    assert "400" in raised.value.message


# jitter value -> the two delays the formula must produce, for
# RETRY_BASE_DELAY * 2**attempt * (0.5 + 0.5 * jitter()).
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
    """Catches a fixed sleep (every retry waiting the same amount, so a rate limit is hit
    again at the same rate) and an unjittered one, where every gateway replica retries a
    provider outage in lockstep."""
    adapter, _ = make_adapter([fail(500, "error_500")], sleep=sleeps, jitter=jitter)

    with pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    assert sleeps.calls == pytest.approx(expected), case


def test_retry_resends_the_same_body(sleeps: SleepRecorder) -> None:
    """Catches a retry that rebuilds or mutates the request between attempts, which would
    silently send the provider something the caller never asked for."""
    adapter, scripted = make_adapter([fail(429, "error_429"), ok("plain")], sleep=sleeps)

    adapter.chat(REQUEST)

    assert scripted.body(0) == scripted.body(1)


def test_a_malformed_success_body_is_a_provider_error(sleeps: SleepRecorder) -> None:
    """Catches an adapter that lets a `KeyError` from an unexpected 200 shape escape as a
    500: every failure leaving the adapter is a `ProviderError` the fallback can read."""
    adapter, _ = make_adapter([(200, {"type": "message", "content": []})], sleep=sleeps)

    with pytest.raises(ProviderError):
        adapter.chat(REQUEST)


def test_provider_message_limit_is_exactly_500() -> None:
    """Catches the cap drifting: it is the only thing bounding how much of a provider's
    body a single failure writes to a log."""
    assert PROVIDER_MESSAGE_LIMIT == 500


def test_retry_base_delay_is_exactly_half_a_second() -> None:
    """Catches a base delay changed without the backoff rows changing: those rows are
    written in terms of the constant, so they follow it rather than pinning it."""
    assert RETRY_BASE_DELAY == 0.5


def test_anthropic_version_and_default_base_url_are_pinned() -> None:
    """Catches a silent bump of the API version (Anthropic's shapes are versioned by this
    header) or a default base url edited to a proxy nobody reviewed."""
    assert ANTHROPIC_VERSION == "2023-06-01"
    assert ANTHROPIC_BASE_URL == "https://api.anthropic.com"


def test_an_oversized_provider_body_is_truncated(sleeps: SleepRecorder) -> None:
    """Catches a dropped slice: a proxy answering 502 with a megabyte of HTML would then
    write a megabyte into the log on every failed attempt."""
    oversized = "x" * (PROVIDER_MESSAGE_LIMIT * 4)
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text=oversized))
    adapter = AnthropicAdapter(
        name=TEST_PROVIDER_NAME,
        api_key=TEST_API_KEY,
        base_url="https://api.anthropic.test",
        transport=transport,
        sleep=sleeps,
    )

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert raised.value.provider_message is not None
    assert len(raised.value.provider_message) == PROVIDER_MESSAGE_LIMIT


def test_a_non_json_success_body_is_a_provider_error(sleeps: SleepRecorder) -> None:
    """Catches the path a proxy takes: a 200 carrying an HTML error page would otherwise
    raise `json.JSONDecodeError` out of the adapter, reaching the route as a 500 with a
    traceback instead of the 502 the routes' fallback loop can move past."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>gateway timeout</html>")
    )
    adapter = AnthropicAdapter(
        name=TEST_PROVIDER_NAME,
        api_key=TEST_API_KEY,
        base_url="https://api.anthropic.test",
        transport=transport,
        sleep=sleeps,
    )

    with pytest.raises(ProviderError) as raised:
        adapter.chat(REQUEST)

    assert raised.value.retryable is False


def test_close_releases_the_client(sleeps: SleepRecorder) -> None:
    """Catches a `close()` that does not: `build_adapters` mints one client per adapter,
    so a policy reload that drops adapters without closing them leaks a connection pool."""
    adapter, _ = make_adapter([ok("plain")], sleep=sleeps)

    adapter.close()

    assert adapter.client.is_closed
