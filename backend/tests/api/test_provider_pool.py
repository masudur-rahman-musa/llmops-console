"""Provider pool: isolation, async shape, dispatch and stream teardown.

An earlier change moves the chat and embeddings routes to `async def` and hands provider
work to a dedicated bounded pool so one slow provider cannot hold the shared
worker pool. These tests assert the constant, the shape and the wiring before
the implementation exists.

No socket is opened, no provider key is read and no backing service is started.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.providers.base import (
    ChatRequest,
)

from .conftest import (
    TEST_KEYS,
    StubAdapter,
    auth_headers,
    build_chat_client,
    chat_body,
    stream_events,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def stream_body(**overrides: object) -> dict[str, Any]:
    """The baseline request body with `stream: true`."""
    return chat_body(stream=True, **overrides)


def data_frames(text: str) -> list[str]:
    """The payload of every `data:` line, `[DONE]` included."""
    return [
        line.removeprefix("data:").strip() for line in text.splitlines() if line.startswith("data:")
    ]


# ---------------------------------------------------------------------------
# 1. The pool size constant
# ---------------------------------------------------------------------------


def test_the_provider_worker_limit_is_exactly_32() -> None:
    """Catches an implementation that uses a different cap — 8, 16, or
    psycopg_pool's default — leaving the provider pool either too small
    to absorb a traffic spike or large enough to let a slow provider
    grab every worker on a small machine."""
    from app.core.concurrency import PROVIDER_POOL_SIZE

    assert PROVIDER_POOL_SIZE == 32


# ---------------------------------------------------------------------------
# 2. Both route handlers must be coroutine functions
# ---------------------------------------------------------------------------


def test_the_chat_and_embeddings_routes_are_async() -> None:
    """Catches an implementation that left either route as `def`, which puts
    every call on the one shared thread pool and lets a slow provider stall
    all other routes."""
    from app.api.routes.chat import create_chat_completion
    from app.api.routes.embeddings import create_embeddings

    assert asyncio.iscoroutinefunction(create_chat_completion)
    assert asyncio.iscoroutinefunction(create_embeddings)


# ---------------------------------------------------------------------------
# 3. Provider work is dispatched through asyncio.to_thread with the limiter
# ---------------------------------------------------------------------------


def _make_spy(
    calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]],
) -> Callable[..., object]:
    """Return an async spy that records every call and runs the function synchronously.

    Using a factory keeps the inner function fully typed without `Any` in its
    signature, which ruff's ANN401 rule forbids in non-test code and also flags
    in helpers defined inside test functions.
    """

    async def _spy(
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    return _spy


def test_provider_work_is_dispatched_on_the_dedicated_limiter(
    settings: Settings,
) -> None:
    """Catches an implementation that calls the pipeline directly in the async
    route instead of offloading it to the dedicated provider limiter, which
    would block the event loop for the duration of every provider call."""
    from app.api.deps import get_provider_limiter

    fake_limiter = MagicMock()
    to_thread_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []
    spy = _make_spy(to_thread_calls)

    stub = StubAdapter(name="anthropic")
    app = build_chat_client(settings, {"anthropic": stub}).app
    app.dependency_overrides[get_provider_limiter] = lambda: fake_limiter  # type: ignore[attr-defined]

    with (
        patch("app.api.routes.chat.asyncio.to_thread", side_effect=spy),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers=auth_headers(key=TEST_KEYS["evals"], tenant="evals", data_class="internal"),
        )

    assert response.status_code == 200, response.text
    assert to_thread_calls, "asyncio.to_thread was never called — pipeline ran directly"
    _, _, kwargs = to_thread_calls[0]
    assert "limiter" in kwargs, "asyncio.to_thread was called without the provider limiter kwarg"
    assert kwargs["limiter"] is fake_limiter


# ---------------------------------------------------------------------------
# 4. A streamed answer still ends with [DONE] after the async wrap
# ---------------------------------------------------------------------------


def test_a_streamed_answer_through_the_async_wrapper_still_ends_with_done(
    settings: Settings,
) -> None:
    """Catches an async wrapper that drops the final [DONE] sentinel or
    re-orders frames, so the client never knows the stream is finished."""
    stub = StubAdapter(name="anthropic", events=stream_events())

    with build_chat_client(settings, {"anthropic": stub}) as client:
        response = client.post(
            "/v1/chat/completions",
            json=stream_body(),
            headers=auth_headers(key=TEST_KEYS["evals"], tenant="evals", data_class="internal"),
        )

    assert response.status_code == 200, response.text
    frames = data_frames(response.text)
    assert frames, "no data frames in the response"
    assert frames[-1] == "[DONE]", f"last frame was {frames[-1]!r}, expected '[DONE]'"


# ---------------------------------------------------------------------------
# 5. Closing the async frame wrapper releases the underlying sync generator
# ---------------------------------------------------------------------------


def test_closing_the_async_frame_wrapper_releases_the_provider_stream() -> None:
    """Catches an async wrapper that does not call generator.close() on early
    exit, which leaves the provider's HTTP response checked out of the
    connection pool until the GC runs — exhausting the pool under load."""
    from app.providers.base import ChatMessage
    from app.services.streaming import sse_frames

    stub = StubAdapter(name="anthropic")
    events = stub.stream_chat(
        ChatRequest(
            model="claude-haiku-4-5",
            messages=(ChatMessage(role="user", content="Hi"),),
        )
    )
    first = next(events)

    frames = sse_frames(first, events, "agent-default", "chatcmpl-000000000000000000000000")

    assert next(frames).startswith("data: ")
    assert not stub.stream_closed, "provider stream must still be open mid-stream"

    frames.close()

    assert stub.stream_closed, (
        "provider stream was not released after the frame generator was closed"
    )
    # Keep a reference so refcounting cannot substitute for an explicit close.
    assert events is not None
