r"""Put the raw values back: the read half of what `redact` produced.

`redact` answers a text full of `<KIND_N>` placeholders and a mapping from each
placeholder to the value it stood in for. This module walks that mapping in the
other direction and does nothing else: no store, no tenant, no HTTP. Which map
to use, whose it is and whether the caller asked for it at all are the route's
questions, answered in `app.services.wire_mapping` and `app.services.streaming`.

Two shapes, one substitution. `rehydrate` is the pure function a whole answer
goes through; `StreamRehydrator` is the same substitution with a carry, for an
answer that arrives as fragments a placeholder can be split across.

The substitution is ONE `re.sub` over an alternation of the mapping's keys, not
a loop of `str.replace` per key. The difference is visible whenever a restored
value contains something placeholder-shaped: the loop rescans the text it has
already rewritten and substitutes inside its own output, while a single pass
cannot look at a span it has just emitted
(`tests/redaction/test_rehydrate.py::test_a_restored_value_containing_a_placeholder_is_not_substituted_again`).

The alternation is built from the mapping's OWN keys, `re.escape`d, so a
placeholder the map does not name is not matched at all and survives exactly as
the model wrote it - an implementation scanning for the generic `<KIND_N>` shape
and looking each hit up would have to decide what to do with a miss, and every
answer to that is worse than leaving it alone
(`test_a_placeholder_absent_from_the_map_is_left_alone`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping


def rehydrate(text: str, mapping: Mapping[str, str]) -> str:
    r"""`text` with every placeholder `mapping` names replaced by its raw value.

    One pass, one compiled alternation, and the replacement is a function rather
    than a template string so a raw value carrying a backslash or a `\g<1>` is
    inserted literally instead of being read as a group reference.

    An empty mapping is the identity and compiles nothing: an alternation built
    from no branches is the empty pattern, which matches at every position and
    would rewrite a string that has no placeholder in it at all
    (`test_an_empty_map_is_the_identity`).

    Branches are sorted longest-first. Python's alternation is first-match, not
    longest-match, so with `<EMAIL_1>` before `<EMAIL_11>` the shorter branch
    would win on the longer token and leave a stray `1>` behind. Every key ends
    in `>`, so the escaped branches cannot in fact overlap and the order is
    belt-and-braces rather than load-bearing - stated here because the test that
    covers the pair (`test_a_longer_placeholder_is_not_eaten_by_a_shorter_one`)
    stays green with the sort removed, and a reader is owed the reason.
    """
    if not mapping:
        return text
    alternation = "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
    return re.sub(alternation, lambda match: mapping[match.group(0)], text)


class StreamRehydrator:
    r"""`rehydrate` with a carry, for an answer that arrives in fragments.

    A streamed answer is a sequence of text deltas, and a placeholder can be cut
    anywhere: `<EMAIL_` in one frame and `1>` in the next. Substituting per frame
    would miss every such split, so `feed` holds back the tail that could still
    turn into a placeholder and answers the rest
    (`tests/redaction/test_rehydrate.py::test_a_placeholder_split_across_two_deltas_is_rehydrated`).

    What is held is the longest suffix of the buffer that is a PROPER PREFIX of
    one of the mapping's own keys (decision B). The hold is then bounded by the
    longest key rather than by the answer - a stray `<` the map can never
    complete (`a < b`, a snippet of HTML) is emitted at once instead of being
    kept until the stream ends
    (`test_the_buffer_holds_less_than_the_longest_key`,
    `test_text_that_cannot_complete_a_placeholder_is_not_held`).

    The buffer is RAW: it holds what the provider sent, never what this object
    has already emitted, and emitted text is never looked at again. That is what
    carries `rehydrate`'s one-pass guarantee onto the stream - a restored value
    ending in something placeholder-shaped cannot be completed by a later
    fragment and substituted a second time
    (`test_a_restored_value_is_not_rescanned_by_a_later_delta`).

    An empty mapping is the identity, exactly as `rehydrate` is: `feed` answers
    its argument whole and holds nothing
    (`test_an_empty_map_makes_every_feed_the_identity`).

    Not thread-safe and not reusable across two streams at once: one instance
    serves one answer, built where that answer's frames are rendered.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._mapping = mapping
        self._buffer = ""
        # One less than the longest key: a suffix as long as a whole key is that
        # key, which completes and is substituted rather than held.
        self._max_hold = max((len(key) for key in mapping), default=0) - 1

    def feed(self, text: str) -> str:
        """Everything `text` completes, rehydrated; the tail that might still complete is held.

        Answers `""` when the whole fragment could still be part of a placeholder,
        which is what makes a split invisible to a client rather than delivered
        twice (decision D: the route emits no frame for an empty answer).
        """
        self._buffer += text
        keep = len(self._buffer) - self._held_length()
        emitted = self._buffer[:keep]
        self._buffer = self._buffer[keep:]
        return rehydrate(emitted, self._mapping)

    def flush(self) -> str:
        """The remainder, rehydrated, and the buffer emptied.

        Called before every non-text event and at the end of the stream, so a
        partial placeholder the provider never completed is still delivered
        rather than dropped (decisions E and F). Emptying is what stops the same
        span being emitted a second time by the next flush
        (`test_flush_answers_the_remainder_and_empties_the_buffer`).
        """
        remainder, self._buffer = self._buffer, ""
        return rehydrate(remainder, self._mapping)

    def _held_length(self) -> int:
        """How many characters of the tail could still become a placeholder."""
        longest = min(self._max_hold, len(self._buffer))
        for length in range(longest, 0, -1):
            suffix = self._buffer[-length:]
            if any(len(suffix) < len(key) and key.startswith(suffix) for key in self._mapping):
                return length
        return 0
