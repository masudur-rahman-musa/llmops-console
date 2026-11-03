"""A cut-off answer is never kept by either cache.

An answer that ran out of room is half an answer. Kept, the next caller with
the same question would get the same broken text, even with room to spare.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.providers.base import ChatResponse
from app.schemas.provider import Usage
from app.services.chat_cache import finished_whole, store_answer, store_probe


class _Recorder:
    """A cache that only records what it was asked to keep."""

    def __init__(self) -> None:
        self.puts: list[Any] = []

    def put(self, *args: object, **kwargs: object) -> None:
        self.puts.append((args, kwargs))


def _answer(finish_reason: str) -> ChatResponse:
    return ChatResponse(
        model="m",
        content="The email is <EMAI",
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _store_both(answer: ChatResponse) -> tuple[_Recorder, _Recorder]:
    exact, semantic = _Recorder(), _Recorder()
    store_answer(exact, "t", "key", answer)  # type: ignore[arg-type]
    store_probe(
        semantic,  # type: ignore[arg-type]
        "t",
        (0.1, 0.2),
        answer,
        alias="a",
        system_hash="s",
        tools_hash="h",
        provider="p",
        data_class="internal",
    )
    return exact, semantic


@pytest.mark.parametrize("reason", ["max_tokens", "length", "content_filter", "refusal"])
def test_an_answer_that_did_not_end_on_its_own_is_kept_by_neither_cache(reason: str) -> None:
    """Catches a cache that keeps a cut-off answer: the next caller would get the
    same half answer even when it allowed a longer one. Both the provider's own
    word (`max_tokens`) and the OpenAI one (`length`) count, and an unknown
    reason is not trusted."""
    exact, semantic = _store_both(_answer(reason))

    assert exact.puts == []
    assert semantic.puts == []
    assert finished_whole(_answer(reason)) is False


@pytest.mark.parametrize("reason", ["end_turn", "stop_sequence", "tool_use", "stop", "tool_calls"])
def test_an_answer_that_ended_on_its_own_is_kept_by_both_caches(reason: str) -> None:
    """Catches a guard so tight that a whole answer is never cached."""
    exact, semantic = _store_both(_answer(reason))

    assert len(exact.puts) == 1
    assert len(semantic.puts) == 1
