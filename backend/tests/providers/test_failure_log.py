"""The gateway log line for a provider call that failed for good.

A failed Anthropic call reaches the help desk as a bare 502. Without a log line the
reason (overloaded, a timeout, a server error) is lost, so the transport writes one
WARNING per call once its retries are spent, and the fallback loop writes one naming
the provider and model. Every answer is a recorded fixture or a scripted transport,
so nothing leaves the process.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.core.errors import ProviderError
from app.model_routing.resolver import Resolved
from app.providers.anthropic import AnthropicAdapter
from app.providers.base import ChatMessage, ChatRequest
from app.providers.breaker import CircuitBreaker
from app.services.chat_pipeline import walk_candidates

from .conftest import (
    PROVIDER_DETAIL_MARKER,
    TEST_API_KEY,
    TEST_PROVIDER_NAME,
    FixedJitter,
    SleepRecorder,
    fail,
    load_fixture,
    make_adapter,
    ok,
)

TRANSPORT_LOGGER = "app.providers.transport"
PIPELINE_LOGGER = "app.services.chat_pipeline"
SENTINEL_PROMPT = "prompt-sentinel-4c19ab tell nobody about the Kestrel account"
REQUEST = ChatRequest(
    model="claude-haiku-4-5",
    messages=(ChatMessage(role="user", content=SENTINEL_PROMPT),),
)


def lines(caplog: pytest.LogCaptureFixture, logger: str) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == logger]


def test_an_overload_that_never_clears_logs_one_line_with_the_reason(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches the last eval run's blind spot: Haiku calls ending in a 502 with nothing in
    the gateway log to say why. One line, after the retries, with the status, Anthropic's
    error type and message, and how many attempts were made."""
    adapter, scripted = make_adapter([fail(529, "error_529")], sleep=sleeps)

    with caplog.at_level(logging.DEBUG, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    logged = lines(caplog, TRANSPORT_LOGGER)
    assert scripted.request_count == 3
    assert len(logged) == 1, logged
    assert TEST_PROVIDER_NAME in logged[0]
    assert "529 overloaded_error" in logged[0]
    assert "Overloaded" in logged[0]
    assert "after 3 attempts" in logged[0]


def test_a_retry_that_then_succeeds_logs_nothing(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a line per failed attempt: a 429 the retry fixed is not a failure."""
    adapter, _ = make_adapter([fail(429, "error_429"), ok("plain")], sleep=sleeps)

    with caplog.at_level(logging.DEBUG, logger=TRANSPORT_LOGGER):
        adapter.chat(REQUEST)

    assert lines(caplog, TRANSPORT_LOGGER) == []


def test_a_server_error_logs_the_providers_message_and_request_id(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a line without the request id Anthropic support asks for, and one that
    writes the prompt or the key next to the reason."""
    body = load_fixture("error_500")

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=body, headers={"request-id": "req_011CQabc"})

    adapter = AnthropicAdapter(
        name=TEST_PROVIDER_NAME,
        api_key=TEST_API_KEY,
        base_url="https://api.anthropic.test",
        transport=httpx.MockTransport(answer),
        sleep=sleeps,
        jitter=FixedJitter(1.0),
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    logged = lines(caplog, TRANSPORT_LOGGER)
    assert len(logged) == 1, logged
    assert "500 api_error" in logged[0]
    assert PROVIDER_DETAIL_MARKER in logged[0]
    assert "req_011CQabc" in logged[0]
    for record in caplog.records:
        assert SENTINEL_PROMPT not in record.getMessage(), record.name
        assert TEST_API_KEY not in record.getMessage(), record.name


def test_a_refused_key_logs_its_type_but_not_its_message(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a 401 line that writes the provider's message, which can quote the key."""
    adapter, _ = make_adapter([fail(401, "error_401")], sleep=sleeps)

    with caplog.at_level(logging.DEBUG, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    logged = lines(caplog, TRANSPORT_LOGGER)
    assert len(logged) == 1, logged
    assert "401 authentication_error" in logged[0]
    assert TEST_API_KEY not in logged[0]
    assert PROVIDER_DETAIL_MARKER not in logged[0]


def test_a_call_that_keeps_timing_out_logs_the_timeout(
    caplog: pytest.LogCaptureFixture, sleeps: SleepRecorder
) -> None:
    """Catches a timeout that leaves no line: it reaches the caller as the same 502."""
    adapter, _ = make_adapter([httpx.ReadTimeout], sleep=sleeps)

    with caplog.at_level(logging.DEBUG, logger=TRANSPORT_LOGGER), pytest.raises(ProviderError):
        adapter.chat(REQUEST)

    logged = lines(caplog, TRANSPORT_LOGGER)
    assert len(logged) == 1, logged
    assert "ReadTimeout" in logged[0]
    assert "after 3 attempts" in logged[0]


def test_the_fallback_loop_names_the_provider_and_model_of_each_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches a failure with no HTTP status (an answer the adapter could not use) that
    leaves no line at all, and a line that does not say which model failed."""
    candidates = (
        Resolved(provider="anthropic-main", model="claude-haiku-4-5"),
        Resolved(provider="openai-backup", model="gpt-5-mini"),
    )
    errors = {
        "anthropic-main": ProviderError(
            "the provider answered with no readable content block",
            provider_message=SENTINEL_PROMPT,
        ),
    }

    def call(chosen: Resolved) -> str:
        raise errors.get(chosen.provider, ProviderError("x", retryable=True))

    with caplog.at_level(logging.DEBUG, logger=PIPELINE_LOGGER), pytest.raises(ProviderError):
        walk_candidates(candidates, CircuitBreaker(), call)

    logged = lines(caplog, PIPELINE_LOGGER)
    assert len(logged) == 1, logged
    assert "anthropic-main claude-haiku-4-5" in logged[0]
    assert "candidate 1 of 2" in logged[0]
    assert "no readable content block" in logged[0]
    assert SENTINEL_PROMPT not in logged[0]
