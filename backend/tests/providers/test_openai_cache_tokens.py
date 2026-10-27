"""Cached prompt tokens on the OpenAI-compatible wire.

OpenAI reports `prompt_tokens_details.cached_tokens` inside `prompt_tokens`. The
gateway's `Usage.prompt_tokens` means the part billed at the plain input rate (the
meaning Anthropic's has), so the adapter subtracts the cached count and books it as
`cache_read_tokens`. A cached token is then priced once, at the cached rate.

The two fixtures are HAND-WRITTEN to the published wire shape, not recorded.
"""

from __future__ import annotations

import pytest

from app.core.errors import ProviderError
from app.providers.base import UsageTotal
from app.providers.openai_compatible import from_openai, from_openai_events

from .conftest import load_local_fixture
from .test_local_streaming import sse_lines

REPORTED = 2006
CACHED = 1920
UNCACHED = 86
COMPLETION = 12
TOTAL = 2018


def test_cached_tokens_are_read_as_cache_read_tokens() -> None:
    """Catches an adapter that ignores `cached_tokens` (the ledger bills 1,920 cached tokens
    at the full input rate), one that leaves them inside `prompt_tokens` as well as in
    `cache_read_tokens` (billed twice), and one that recomputes the total: 2,006 reported
    and 1,920 cached give a `prompt_tokens` of 86, a `cache_read_tokens` of 1,920 and the
    total as the provider reported it."""
    usage = from_openai(load_local_fixture("cached_prompt")).usage

    assert usage.prompt_tokens == UNCACHED
    assert usage.cache_read_tokens == CACHED
    assert usage.cache_write_tokens == 0
    assert usage.completion_tokens == COMPLETION
    assert usage.total_tokens == TOTAL


def test_a_streamed_usage_chunk_reads_cached_tokens() -> None:
    """Catches a reader that applies the cached count to the whole reply but not to the
    stream's closing usage chunk (streamed fallbacks to OpenAI would be priced at full
    input): the same numbers must come out of the stream."""
    events = list(from_openai_events(sse_lines("stream_cached_prompt")))

    last = events[-1]
    assert isinstance(last, UsageTotal)
    assert last.usage.prompt_tokens == UNCACHED
    assert last.usage.cache_read_tokens == CACHED
    assert last.usage.total_tokens == TOTAL


ABSENT = "absent"
NULL = "null"


@pytest.mark.parametrize("shape", [ABSENT, NULL], ids=[ABSENT, NULL])
def test_no_details_or_null_details_reads_zero(shape: str) -> None:
    """Catches a reader that requires the details block (every reply from llama.cpp and
    older recordings carries none, so each would fail) or that treats a `null` block as an
    error or subtracts anything: both leave `prompt_tokens` as reported and cached at 0."""
    payload = load_local_fixture("cached_prompt")
    if shape == ABSENT:
        del payload["usage"]["prompt_tokens_details"]
    else:
        payload["usage"]["prompt_tokens_details"] = None

    usage = from_openai(payload).usage

    assert usage.prompt_tokens == REPORTED
    assert usage.cache_read_tokens == 0


UNREADABLE_ROWS: list[tuple[str, int]] = [
    ("cached one above the reported prompt count", REPORTED + 1),
    ("a negative cached count", -1),
]


@pytest.mark.parametrize(
    ("case", "cached"), UNREADABLE_ROWS, ids=[row[0] for row in UNREADABLE_ROWS]
)
def test_cached_above_prompt_or_negative_is_unreadable(case: str, cached: int) -> None:
    """Catches a reader that lets a cached count above the reported prompt count through
    (`prompt_tokens` would go negative, or be clamped to zero and book a real call as
    cheaper than it was) and one that passes a negative count on (a credit in the ledger):
    both raise the adapter's non-retryable unreadable `ProviderError`, not the schema's
    own error."""
    payload = load_local_fixture("cached_prompt")
    payload["usage"]["prompt_tokens_details"]["cached_tokens"] = cached

    with pytest.raises(ProviderError) as raised:
        from_openai(payload)

    assert raised.value.retryable is False, case
    assert raised.value.http_status == 502, case


def test_a_cached_count_equal_to_the_prompt_count_is_readable() -> None:
    """Catches an off-by-one that refuses `cached == prompt_tokens` (a fully cached prompt
    is legal): the uncached part is 0 and the whole count is a cache read."""
    payload = load_local_fixture("cached_prompt")
    payload["usage"]["prompt_tokens_details"]["cached_tokens"] = REPORTED

    usage = from_openai(payload).usage

    assert usage.prompt_tokens == 0
    assert usage.cache_read_tokens == REPORTED
