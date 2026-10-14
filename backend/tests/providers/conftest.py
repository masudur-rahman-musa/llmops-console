"""Fixtures for the provider adapter tests: recorded JSON and a scripted transport.

Everything here is in-process. The "recorded" Anthropic payloads are JSON files
read from disk and replayed through `httpx.MockTransport`, so no test opens a
socket and no test reads a provider key from the environment.
There is no backing service, so a project rule (session-scoped fixtures with a
per-test namespace) does not apply and function scope is the correct choice:
building a transport costs a dict lookup.

`PROVIDER_DETAIL_MARKER` is planted inside every error fixture's provider
message. It exists so `test_provider_message_never_reaches_the_client_message` can prove
the adapter keeps the provider's own text on `provider_message` and out of the
client-facing `message`, rather than asserting on a substring that might not
have been in the payload at all.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.providers.anthropic import AnthropicAdapter
from app.providers.openai_compatible import OpenAICompatibleAdapter

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "anthropic"

# Present in the `error.message` of every error fixture, and nowhere else.
PROVIDER_DETAIL_MARKER = "upstream-detail-2f7c1a-do-not-leak"

# The key every adapter under test is constructed with, passed explicitly.
TEST_API_KEY = "sk-ant-test-not-a-real-key"

# The policy provider name used throughout: deliberately NOT the string
# "anthropic", so a `price_key` or a factory key built from the provider kind
# instead of the policy name is visible.
TEST_PROVIDER_NAME = "anthropic-eu"


def load_fixture(name: str) -> dict[str, Any]:
    """Read one recorded Anthropic payload by file stem."""
    path = FIXTURE_DIR / f"{name}.json"
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path}: fixture must be a JSON object"
    return payload


# One scripted step: a (status, body) pair to answer with, or a transport-level
# exception class to raise instead of answering.
Step = tuple[int, Mapping[str, Any]] | type[httpx.RequestError]


class ScriptedTransport:
    """An `httpx.MockTransport` that plays a fixed script and records every request.

    The last step repeats once the script runs out, so a retry loop that makes
    more attempts than the plan allows gets a plausible answer rather than an
    IndexError; the exact-request-count assertions are what catch it.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        assert steps, "a scripted transport needs at least one step"
        self._steps = list(steps)
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._steps) - 1)
        step = self._steps[index]
        if isinstance(step, type):
            raise step(f"scripted {step.__name__}", request=request)
        status, body = step
        return httpx.Response(status, json=dict(body), request=request)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def body(self, index: int = 0) -> dict[str, Any]:
        """The JSON the adapter sent on the request at `index`."""
        sent: Any = json.loads(self.requests[index].content.decode("utf-8"))
        assert isinstance(sent, dict), "the adapter must send a JSON object"
        return sent


def ok(name: str) -> Step:
    """A 200 answering with the named recorded fixture."""
    return (200, load_fixture(name))


def fail(status: int, name: str) -> Step:
    """A failure status answering with the named recorded error fixture."""
    return (status, load_fixture(name))


class SleepRecorder:
    """The injected `sleep`: records the delays instead of spending them."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FixedJitter:
    """The injected `jitter`, so the backoff delays are exact numbers in a test."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def sleeps() -> SleepRecorder:
    return SleepRecorder()


def make_adapter(
    steps: Sequence[Step],
    *,
    name: str = TEST_PROVIDER_NAME,
    sleep: SleepRecorder | None = None,
    jitter: float = 1.0,
) -> tuple[AnthropicAdapter, ScriptedTransport]:
    """An adapter wired to a scripted transport, with an explicit key and no clock.

    The key is passed in rather than read from the environment, and `sleep` and
    `jitter` are injected, so the retry cases are instant and deterministic.
    """
    scripted = ScriptedTransport(steps)
    adapter = AnthropicAdapter(
        name=name,
        api_key=TEST_API_KEY,
        base_url="https://api.anthropic.test",
        transport=scripted.transport,
        sleep=sleep if sleep is not None else SleepRecorder(),
        jitter=FixedJitter(jitter),
    )
    return adapter, scripted


# --- the OpenAI-compatible local provider -------------------------
#
# The same in-process arrangement as above, against a second fixture directory.
# The `local` payloads are HAND-WRITTEN to the published OpenAI chat-completions
# wire shape rather than recorded from a llama.cpp server, because no such server
# runs in CI or in the environment these tests were written in; every `.sse` file
# says so in its own leading comment lines.

LOCAL_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "local"

# The policy provider name for the local adapter under test. Deliberately NOT
# "local" and not "openai_compatible", so a `price_key`, a header or a factory
# key built from the provider KIND rather than from the policy name is visible.
LOCAL_PROVIDER_NAME = "local-cax21"

# The base url the scripted local adapter is built with. It ends in `/v1` exactly
# as the shipped policy's does, so a path built with `/v1` inside the adapter
# would produce `/v1/v1/chat/completions` and fail the path assertions.
LOCAL_BASE_URL = "http://llama.test:8082/v1"

# Only for the auth cases: the shipped `local` provider declares no key at all.
TEST_LOCAL_API_KEY = "sk-local-test-not-a-real-key"


def load_local_fixture(name: str) -> dict[str, Any]:
    """Read one hand-written OpenAI-shaped payload by file stem."""
    path = LOCAL_FIXTURE_DIR / f"{name}.json"
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path}: fixture must be a JSON object"
    return payload


def local_ok(name: str) -> Step:
    """A 200 answering with the named local fixture."""
    return (200, load_local_fixture(name))


def local_fail(status: int, name: str) -> Step:
    """A failure status answering with the named local error fixture."""
    return (status, load_local_fixture(name))


def make_local_adapter(
    steps: Sequence[Step],
    *,
    name: str = LOCAL_PROVIDER_NAME,
    api_key: str | None = None,
    sleep: SleepRecorder | None = None,
    jitter: float = 1.0,
) -> tuple[OpenAICompatibleAdapter, ScriptedTransport]:
    """A local adapter wired to a scripted transport, with no clock and no key by default.

    `api_key=None` is the shipped `local` provider's configuration (an earlier change
    decision B): it sits on a private network and declares no `key_env`.
    """
    scripted = ScriptedTransport(steps)
    adapter = OpenAICompatibleAdapter(
        name=name,
        base_url=LOCAL_BASE_URL,
        api_key=api_key,
        transport=scripted.transport,
        sleep=sleep if sleep is not None else SleepRecorder(),
        jitter=FixedJitter(jitter),
    )
    return adapter, scripted


# --- embeddings ---------------------------------------------------
#
# No `make_local_embedding_adapter` here on purpose: `embed()` lives on the same
# `OpenAICompatibleAdapter` `make_local_adapter` already builds, so a second
# builder would be a second place for the base url, the key and the injected
# clock to drift from the chat one.
#
# The `embeddings_*.json` fixtures are HAND-WRITTEN to the published OpenAI
# embeddings shape, like every other file in `fixtures/local/`, and each says so
# in a `_comment` key. `load_local_fixture` reads whole objects, and every reader
# under test looks its fields up by name, so the key is inert.

# The provider model id the embedding fixtures answer with: the shipped policy's
# `embed` candidate, and deliberately not the `embed` alias, so a mapper that
# sends the alias is visible.
EMBEDDING_MODEL = "bge-small-en-v1.5"

# Every vector in the fixtures is exactly representable as a float32, so the
# base64 round trip in tests/gateway/test_embeddings.py compares equal rather
# than approximately equal, and a wrong byte order is the only thing that can
# break it.
SINGLE_VECTOR = (0.125, -0.375, 0.625, -0.875)
BATCH_VECTORS = (
    (0.5, -0.25, 0.75),
    (-0.0625, 0.1875, -0.3125),
    (0.4375, -0.5625, 0.6875),
)
