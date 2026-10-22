"""A provider refusing a request for content policy: how the shared transport spots it.

An earlier change. A refusal is a failure no other provider should be asked to overrule, so the
transport marks it (`ProviderError.refused`) and never calls it retryable. Everything
runs over `httpx.MockTransport` with recorded-shape bodies in `fixtures/openai/` and
`fixtures/azure/`: no socket, no key, no real wait (`sleep` is a recorder).

The transport is vendor-neutral, so these cases drive `ProviderTransport` directly
rather than one adapter; the OpenAI-shaped and Azure-shaped bodies are the two
spellings it has to read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.errors import ProviderError
from app.providers.transport import ProviderTransport

from .conftest import FixedJitter, ScriptedTransport, SleepRecorder

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROVIDER = "azure-eu"
DETAIL_MARKER = "upstream-detail-2f7c1a-do-not-leak"
PATH = "/chat/completions"
REFUSED_MESSAGE = f"{PROVIDER}: the provider refused the request for content policy"


def body_of(folder: str, name: str) -> dict[str, Any]:
    """Read one recorded body, e.g. `body_of("openai", "content_policy_400")`."""
    payload: Any = json.loads((FIXTURES / folder / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def transport_for(
    status: int, body: dict[str, Any]
) -> tuple[ProviderTransport, ScriptedTransport, SleepRecorder]:
    """A transport that answers every request with this one failure."""
    scripted = ScriptedTransport([(status, body)])
    sleeps = SleepRecorder()
    transport = ProviderTransport(
        name=PROVIDER,
        base_url="https://provider.test",
        headers={},
        transport=scripted.transport,
        sleep=sleeps,
        jitter=FixedJitter(1.0),
    )
    return transport, scripted, sleeps


def failure_of(status: int, body: dict[str, Any]) -> tuple[ProviderError, ScriptedTransport]:
    """Send one request and answer the error it raised with the requests the provider saw."""
    transport, scripted, _ = transport_for(status, body)
    with pytest.raises(ProviderError) as caught:
        transport.send(PATH, {"model": "m"})
    return caught.value, scripted


REFUSAL_CASES: list[tuple[str, int, str, str]] = [
    ("openai content_policy_violation", 400, "openai", "content_policy_400"),
    ("openai content_filter code alone", 400, "openai", "content_filter_code_400"),
    ("azure content_filter with the inner code", 400, "azure", "content_filter_400"),
    ("azure inner code alone, outer code BadRequest", 400, "azure", "inner_policy_only_400"),
]


@pytest.mark.parametrize(
    ("case", "status", "folder", "name"),
    REFUSAL_CASES,
    ids=[case for case, *_ in REFUSAL_CASES],
)
def test_a_content_policy_400_is_refused_and_not_retryable(
    case: str, status: int, folder: str, name: str
) -> None:
    """Catches a transport that reads only one of the three spellings (the outer
    `content_policy_violation`, the outer `content_filter`, the inner
    `ResponsibleAIPolicyViolation`), and one that calls a refusal retryable so the same
    request is billed three times. One request sent proves no same-provider retry."""
    error, scripted = failure_of(status, body_of(folder, name))

    assert error.refused is True
    assert error.retryable is False
    assert scripted.request_count == 1


def test_an_azure_content_filter_400_is_refused() -> None:
    """Catches a transport that knows the OpenAI `content_policy_violation` code but not
    Azure's `content_filter` with its `ResponsibleAIPolicyViolation` inner code, so Azure
    refusals stay a plain 400 and nothing downstream can tell them apart."""
    error, scripted = failure_of(400, body_of("azure", "content_filter_400"))

    assert error.refused is True
    assert error.retryable is False
    assert scripted.request_count == 1


def test_a_refusal_code_on_a_5xx_is_still_not_retried() -> None:
    """Catches a transport that only looks for a refusal on a 400 and lets the 5xx rule
    win: a 500 carrying `content_policy_violation` would then be retried twice and fall
    back, asking more providers to overrule the refusal. One request and no sleep prove the
    refusal check runs before the retry table."""
    transport, scripted, sleeps = transport_for(500, body_of("openai", "content_policy_400"))

    with pytest.raises(ProviderError) as caught:
        transport.send(PATH, {"model": "m"})

    assert caught.value.refused is True
    assert caught.value.retryable is False
    assert scripted.request_count == 1
    assert sleeps.calls == []


def test_a_refusal_on_a_streamed_call_is_refused_and_not_retried() -> None:
    """Catches a refusal check added to `send` only: the streaming path builds its error
    through the same `status_error`, and a stream opened against a refusing provider must
    not be retried either."""
    transport, scripted, _ = transport_for(400, body_of("azure", "content_filter_400"))

    with pytest.raises(ProviderError) as caught, transport.open_stream(PATH, {"model": "m"}):
        pass

    assert caught.value.refused is True
    assert caught.value.retryable is False
    assert scripted.request_count == 1


PLAIN_400_CASES: list[tuple[str, str, str]] = [
    ("a malformed request", "local", "error_400"),
    ("policy words in the free-text message only", "openai", "echoes_policy_words_400"),
    ("an Azure bad request with another inner code", "azure", "other_inner_code_400"),
]


@pytest.mark.parametrize(
    ("case", "folder", "name"),
    PLAIN_400_CASES,
    ids=[case for case, *_ in PLAIN_400_CASES],
)
def test_a_plain_400_is_not_marked_refused(case: str, folder: str, name: str) -> None:
    """Catches a transport that marks every 400 refused (a malformed request would then
    read as a policy block), and one that matches the policy words anywhere in the text
    instead of in `error.code`. The message stays the status message, unchanged."""
    error, scripted = failure_of(400, body_of(folder, name))

    assert error.refused is False
    assert error.retryable is False
    assert str(error) == f"{PROVIDER}: the provider answered 400"
    assert scripted.request_count == 1


def test_the_refusal_message_names_the_provider_and_not_its_words() -> None:
    """Catches a refusal message that repeats the provider's own text to the client (it can
    carry request ids or the filtered category), and one that words the refusal differently
    from the closed message. The provider's words must be kept, in `provider_message`, for
    the log."""
    error, _ = failure_of(400, body_of("azure", "content_filter_400"))

    assert str(error) == REFUSED_MESSAGE
    assert error.message == REFUSED_MESSAGE
    assert DETAIL_MARKER not in error.message
    assert error.provider_message is not None
    assert DETAIL_MARKER in error.provider_message


def test_a_provider_error_is_not_refused_unless_built_that_way() -> None:
    """Catches a `ProviderError` whose `refused` has no default, which would break every
    adapter that raises one without the flag."""
    assert ProviderError("x", retryable=False).refused is False
    assert ProviderError("x", retryable=False, refused=True).refused is True
