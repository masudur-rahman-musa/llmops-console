"""Tests for thin routes and one shared fallback loop.

Each test's docstring names what wrong implementation it would catch.
"""

from __future__ import annotations

import importlib
import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.types import Scope

from app.core.config import Settings

# These imports are from the NEW locations the implementation will create.
# They do NOT exist yet, so every import below will raise ImportError until
# the implementation is in place. That is the expected "failing for the right
# reason."
from app.core.context import RequestContext
from app.core.errors import ProviderError
from app.model_routing.errors import AllProvidersUnavailable
from app.model_routing.resolver import Resolved
from app.providers.breaker import CircuitBreaker, CircuitState
from app.services.chat_accounting import ChatResult, Clients
from app.services.chat_pipeline import complete_chat, walk_candidates
from app.services.embedding_pipeline import (
    EmbeddingsResult,
    embed_texts,
)
from app.services.streaming import StreamPlan

# ---------------------------------------------------------------------------
# Helpers shared across walk_candidates tests
# ---------------------------------------------------------------------------

_SENTINEL = object()
"""A unique object: any implementation that returns something other than the
value the passing candidate produced cannot make _SENTINEL come back."""


def _make_resolved(provider: str) -> Resolved:
    """Build a minimal Resolved with the given provider name."""
    return Resolved(provider=provider, model="m")


def _fake_scope(path: str) -> Scope:
    """A minimal ASGI scope dict for a POST to the given path."""
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
    }


# ---------------------------------------------------------------------------
# 1. Static-analysis: service modules import nothing from the API layer
# ---------------------------------------------------------------------------


_SERVICE_MODULES = [
    "app.services.chat_pipeline",
    "app.services.streaming",
    "app.services.embedding_pipeline",
    "app.services.wire_mapping",
    "app.services.redaction_flow",
]

_FORBIDDEN_STRINGS = [
    "app.api",
    "fastapi.Request",
    "StreamingResponse",
    "datastructures import State",
]


@pytest.mark.parametrize("module_name", _SERVICE_MODULES)
@pytest.mark.parametrize("forbidden", _FORBIDDEN_STRINGS)
def test_the_request_path_services_import_nothing_from_the_api_layer(
    module_name: str, forbidden: str
) -> None:
    """Catches an implementation that left an `app.api` or FastAPI import in a
    service module, which would couple the service layer to the HTTP layer."""
    mod = importlib.import_module(module_name)
    source = inspect.getsource(mod)
    assert forbidden not in source, f"{module_name} must not contain {forbidden!r} but does"


# ---------------------------------------------------------------------------
# 2. walk_candidates: retryable failure moves to the next candidate
# ---------------------------------------------------------------------------


def test_walk_candidates_serves_the_next_on_a_retryable_failure() -> None:
    """Catches an implementation that stops at the first error regardless of
    `retryable`, never trying the second candidate."""
    first = _make_resolved("first_provider")
    second = _make_resolved("second_provider")
    candidates = (first, second)

    breaker = CircuitBreaker()
    first_error = ProviderError("first failed", retryable=True)

    call_log: list[Resolved] = []

    def _call(candidate: Resolved) -> object:
        call_log.append(candidate)
        if candidate.provider == "first_provider":
            raise first_error
        return _SENTINEL

    result = walk_candidates(candidates, breaker, _call)

    assert result is _SENTINEL
    assert len(call_log) == 2
    # The first provider's breaker must have a failure record (one failure does
    # not open the circuit, but the record is proof absorb_failure was called).
    assert breaker._records.get("first_provider") is not None


# ---------------------------------------------------------------------------
# 3. walk_candidates: non-retryable failure stops immediately
# ---------------------------------------------------------------------------


def test_walk_candidates_stops_at_the_first_on_a_non_retryable_failure() -> None:
    """Catches an implementation that tries the next candidate even on a
    non-retryable error, or that swallows the error instead of re-raising it."""
    first = _make_resolved("first_provider")
    second = _make_resolved("second_provider")
    candidates = (first, second)

    breaker = CircuitBreaker()
    first_error = ProviderError("bad request", retryable=False)

    second_called = False

    def _call(candidate: Resolved) -> object:
        nonlocal second_called
        if candidate.provider == "first_provider":
            raise first_error
        second_called = True
        return _SENTINEL

    with pytest.raises(ProviderError) as exc_info:
        walk_candidates(candidates, breaker, _call)

    assert exc_info.value is first_error
    assert not second_called, "second candidate must not be called on a non-retryable error"
    # Non-retryable: the caller made a mistake, not the provider, so the
    # breaker records nothing.
    assert breaker.state("first_provider") is CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 4. walk_candidates: last error propagates when all candidates fail
# ---------------------------------------------------------------------------


def test_walk_candidates_raises_the_last_failure_when_every_candidate_fails() -> None:
    """Catches an implementation that re-raises the FIRST error instead of the
    last, or that raises a generic error that loses the provider's message."""
    first = _make_resolved("first_provider")
    second = _make_resolved("second_provider")
    candidates = (first, second)

    breaker = CircuitBreaker()
    first_error = ProviderError("first failed retryably", retryable=True)
    second_error = ProviderError("second failed retryably", retryable=True)

    def _call(candidate: Resolved) -> object:
        if candidate.provider == "first_provider":
            raise first_error
        raise second_error

    with pytest.raises(ProviderError) as exc_info:
        walk_candidates(candidates, breaker, _call)

    assert exc_info.value is second_error, "the LAST failure must propagate, not the first"


# ---------------------------------------------------------------------------
# 5. walk_candidates: empty tuple raises AllProvidersUnavailable
# ---------------------------------------------------------------------------


def test_walk_candidates_on_an_empty_tuple_is_all_providers_unavailable() -> None:
    """Catches an implementation that returns None or raises a different error
    when no candidates are given, rather than AllProvidersUnavailable."""
    breaker = CircuitBreaker()

    def _call(candidate: Resolved) -> object:  # pragma: no cover
        return _SENTINEL  # unreachable

    with pytest.raises(AllProvidersUnavailable):
        walk_candidates((), breaker, _call)


# ---------------------------------------------------------------------------
# 6. complete_chat returns the right result type for stream vs. non-stream
# ---------------------------------------------------------------------------


def test_complete_chat_returns_a_stream_plan_for_a_streamed_body_and_a_chat_result_otherwise(
    chat_client: TestClient,
    settings: Settings,
) -> None:
    """Catches an implementation where complete_chat always returns the same
    type regardless of the stream flag, or returns a raw FastAPI response
    instead of the plain result types."""
    from api.conftest import NO_PROVIDER_KEYS, StubAdapter
    from app.main import create_app
    from app.model_routing.policy import DataClass
    from app.schemas.chat import ChatCompletionRequest

    stub = StubAdapter(name="anthropic")
    app = create_app(settings, NO_PROVIDER_KEYS)
    app.state.adapters = {"anthropic": stub}

    with TestClient(app) as client:
        state = client.app.state  # type: ignore[attr-defined]
        clients = Clients(
            policy=state.policy.policy,
            rate_limit_store=state.rate_limit_store,
            budget_store=state.budget_store,
            settings=state.settings,
            cost_ledger=state.cost_ledger,
            prices=state.prices,
            adapters=state.adapters,
            breaker=state.breaker,
            response_cache=state.response_cache,
            semantic_cache=state.semantic_cache,
            redaction_map_store=state.redaction_map_store,
            directory_store=state.directory_store,
        )

        context = RequestContext(
            tenant="evals",
            data_class=DataClass.INTERNAL,
            session=None,
            trace_id="a" * 32,
        )

        non_stream_body = ChatCompletionRequest(
            model="agent-default",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        stream_body = ChatCompletionRequest(
            model="agent-default",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        scope = _fake_scope("/v1/chat/completions")

        non_stream_result = complete_chat(clients, context, non_stream_body, scope)
        stream_result = complete_chat(clients, context, stream_body, scope)

    assert isinstance(non_stream_result, ChatResult), (
        f"expected ChatResult for stream=False, got {type(non_stream_result)}"
    )
    assert isinstance(stream_result, StreamPlan), (
        f"expected StreamPlan for stream=True, got {type(stream_result)}"
    )


# ---------------------------------------------------------------------------
# 7. embed_texts returns an EmbeddingsResult the route only wraps
# ---------------------------------------------------------------------------


def test_embed_texts_returns_a_result_the_route_only_wraps(settings: Settings) -> None:
    """Catches an implementation where embed_texts returns a raw Response or
    StreamingResponse instead of a plain EmbeddingsResult with body and headers."""
    from api.conftest import EMBED_PROVIDER, NO_PROVIDER_KEYS, EmbeddingStubAdapter
    from app.main import create_app
    from app.model_routing.policy import DataClass
    from app.schemas.embeddings import EmbeddingsRequest

    embed_stub = EmbeddingStubAdapter(name=EMBED_PROVIDER)
    app = create_app(settings, NO_PROVIDER_KEYS)
    app.state.adapters = {EMBED_PROVIDER: embed_stub}

    with TestClient(app) as client:
        state = client.app.state  # type: ignore[attr-defined]
        clients = Clients(
            policy=state.policy.policy,
            rate_limit_store=state.rate_limit_store,
            budget_store=state.budget_store,
            settings=state.settings,
            cost_ledger=state.cost_ledger,
            prices=state.prices,
            adapters=state.adapters,
            breaker=state.breaker,
            response_cache=state.response_cache,
            semantic_cache=state.semantic_cache,
            redaction_map_store=state.redaction_map_store,
            directory_store=state.directory_store,
        )

        context = RequestContext(
            tenant="evals",
            data_class=DataClass.INTERNAL,
            session=None,
            trace_id="b" * 32,
        )

        body = EmbeddingsRequest(model="embed", input="the capital of Germany")

        result = embed_texts(clients, context, body, _fake_scope("/v1/embeddings"))

    assert isinstance(result, EmbeddingsResult), f"expected EmbeddingsResult, got {type(result)}"
    assert hasattr(result, "body"), "EmbeddingsResult must have a body field"
    assert hasattr(result, "headers"), "EmbeddingsResult must have a headers field"
    assert isinstance(result.headers, dict), "headers must be a plain dict"


# ---------------------------------------------------------------------------
# 8. Each route makes exactly one service call
# ---------------------------------------------------------------------------


def test_each_of_the_two_routes_makes_exactly_one_service_call(
    chat_client: TestClient,
    embeddings_client: TestClient,
) -> None:
    """Catches an implementation where a route calls the service function more
    than once per request, or calls it zero times and builds the pipeline
    inline instead."""
    from api.conftest import (
        CHAT,
        EMBEDDINGS,
        TEST_KEYS,
        auth_headers,
        chat_body,
        embeddings_body,
    )

    chat_spy = MagicMock(side_effect=complete_chat)
    embed_spy = MagicMock(side_effect=embed_texts)

    chat_patch = patch("app.api.routes.chat.complete_chat", chat_spy)
    embed_patch = patch("app.api.routes.embeddings.embed_texts", embed_spy)

    with chat_patch, embed_patch:
        chat_client.post(
            CHAT,
            json=chat_body(),
            headers=auth_headers(
                key=TEST_KEYS["evals"],
                tenant="evals",
                data_class="internal",
            ),
        )
        embeddings_client.post(
            EMBEDDINGS,
            json=embeddings_body(),
            headers=auth_headers(
                key=TEST_KEYS["evals"],
                tenant="evals",
                data_class="internal",
            ),
        )

    assert chat_spy.call_count == 1, (
        f"complete_chat must be called exactly once per request, "
        f"was called {chat_spy.call_count} times"
    )
    assert embed_spy.call_count == 1, (
        f"embed_texts must be called exactly once per request, "
        f"was called {embed_spy.call_count} times"
    )
