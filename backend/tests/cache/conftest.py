"""Shared data for the exact-cache tests: one clock, one answer, one key input set.

Everything here is in-process and immutable, so a project rule has nothing to
apply to beyond the session-scoped Redis client that lives beside the cases that
need it (`tests/cache/test_redis_cache.py`).

Two pieces carry weight:

* `FakeClock` is the injected clock every TTL boundary is asserted against, so no
  case sleeps and the "readable at exactly the TTL, gone one second past" pair is
  a pair of assertions rather than a day of waiting.
* `KEY_FIELDS` is the seven keyed fields of the design with a DIFFERENT value for
  each, which is what the per-field miss table in `test_key.py` is built from: a
  key that omits one field serves a wrong answer to a different request, so each
  field gets a row rather than a sample.
"""

from __future__ import annotations

from typing import Any

from app.providers.base import ChatResponse
from app.schemas.provider import ToolCall, Usage

TENANT = "evals"
OTHER_TENANT = "cuan"

START = 1_000.0
"""The clock's value at the moment of a `put`. Deliberately not 0, so a store that
compared against an absolute value rather than an elapsed one is visible."""


class FakeClock:
    """A monotonic clock a test moves by hand. `now` is seconds, like `time.monotonic`."""

    def __init__(self, now: float = START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# The numbers are distinct from each other and from every other fixture in the
# suite, so a zero-fill, a swap or a recomputed total is visible on the way
# through JSON and back.
CACHED_USAGE = Usage(prompt_tokens=211, completion_tokens=43, total_tokens=254)

CACHED_TOOL_CALL = ToolCall(
    id="toolu_01CachedWeatherCall00", name="get_weather", arguments='{"city": "Berlin"}'
)

CACHED_MODEL = "cached-model-neither-alias-nor-policy-id"
CACHED_CONTENT = "The cached answer."


def cached_answer(
    *,
    content: str = CACHED_CONTENT,
    model: str = CACHED_MODEL,
    finish_reason: str = "end_turn",
    with_tool_call: bool = True,
) -> ChatResponse:
    """One neutral `ChatResponse`, the value this cache stores (plan decision E).

    Carries a tool call by default: the wire body is not what is cached, so the
    round trip has to survive every field of the neutral answer and a store that
    kept only `content` would still pass a content-only fixture.
    """
    return ChatResponse(
        model=model,
        content=content,
        tool_calls=(CACHED_TOOL_CALL,) if with_tool_call else (),
        finish_reason=finish_reason,
        usage=CACHED_USAGE,
    )


# --- the seven keyed fields (the design) -----------------------------------------

BASE_MESSAGES: tuple[dict[str, Any], ...] = (
    {"role": "user", "content": "Which city is the capital of Germany?"},
)

BASE_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
)

BASE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "schema": {"type": "object"}},
}

BASE_ALIAS = "agent-default"
BASE_MODEL = "claude-haiku-4-5"
BASE_TEMPERATURE = 0.0
"""The baseline every miss row in `test_key.py` differs from by exactly one field.

`tools` and `response_format` have no baseline value because their rows differ by
their PRESENCE, which is the change a real caller makes; every other row changes a
value that is already there.
"""
