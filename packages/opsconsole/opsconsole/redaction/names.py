r"""The tenth detector kind: a tenant's own display names, compiled into one pattern.

The nine kinds in `detectors.py` read nothing but the text. `PERSON` reads a list
an operator pushed through `POST /admin/directory`, so its input is the
directory and the text both, and that is the whole reason it lives in a module of
its own rather than as a tenth entry in `detect_all`'s list of pure functions.

One alternation, not one pattern per name.
`redact_batch` walks up to 64 strings under one numbering, and a matcher that
compiled per string would compile 2000 alternatives 64 times for one request; a
matcher that scanned once per NAME would walk the text 2000 times. So the
directory is compiled ONCE per request, by the route, and handed to every string.

Three things the alternation has to get right, each with its own group of rows in
`tests/redaction/test_names.py`.

`re.escape` on every name, individually. An operator's name may legitimately
contain a dot, a star or a bracket, and unescaped those change what the pattern
means: `A. B*` would read as "any character, then B repeated", matching `Ax BBB`
and not the person. A `|` or a `)` is worse than wrong - it ends the group early
and silently drops every name after it
(`test_a_metacharacter_name_does_not_change_what_a_neighbour_matches`).

Longest first. A Python alternation takes the FIRST branch that matches, so with
`Anna` before `Anna Schmidt` the text `Anna Schmidt` yields a four-character span
and leaves ` Schmidt` in the clear behind the placeholder. Sorting by length
descending is what makes the outcome independent of the order the store happened
to answer in (`test_the_longest_name_wins_over_its_own_first_token`).

Whole-word boundaries as lookarounds, `(?<!\w)` and `(?!\w)`, never `\b`. A name
may begin or end with a non-word character (`A. B*` ends in a star), and `\b` is
defined against the character on the other side, so it would refuse exactly those
names. The lookarounds say the same thing for a name that starts and ends in a
letter and the right thing for one that does not
(`test_a_name_with_regex_metacharacters_matches_itself_literally`,
`test_punctuation_around_a_name_is_not_part_of_the_span`).

An empty directory compiles to no pattern at all rather than to an empty
alternation. `(?:)` matches the empty string at every position, which would make
every request for a tenant that has pushed nothing one `<PERSON_N>` per character
(`test_an_empty_directory_yields_nothing_and_does_not_raise`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from opsconsole.redaction.detectors import Detection, Kind


class NameDirectory:
    """One tenant's names, compiled into a single case-insensitive alternation.

    Built by `compile` and never by the constructor directly, so there is one
    place where the escaping, the ordering and the boundaries are decided.
    Immutable once built: the route compiles one per request and hands the same
    object to every string in the batch, so a matcher that could be added to
    mid-request would redact the later turns of a conversation differently from
    the earlier ones.

    `detect` yields the same frozen `Detection` the nine pattern kinds yield,
    with `value` sliced out of the SCANNED text rather than taken from the
    directory: matching is case-insensitive, so the two spellings differ, and the
    placeholder map has to hold what the caller actually wrote or its inverse
    stops reproducing the input
    (`test_a_candidates_value_indexes_its_own_offsets`).
    """

    __slots__ = ("_pattern",)

    def __init__(self, pattern: re.Pattern[str] | None) -> None:
        self._pattern = pattern

    @classmethod
    def compile(cls, names: Iterable[str]) -> NameDirectory:
        """Build the matcher for one directory. An empty directory matches nothing.

        Names are de-duplicated on their case-folded form before compiling, so a
        directory holding two spellings of one name does not compile two
        identical branches; the store's hashed field already makes that rare.
        """
        unique = {name.strip().casefold(): name.strip() for name in names if name.strip()}
        ordered = sorted(unique.values(), key=lambda name: (-len(name), name))
        if not ordered:
            return cls(None)
        alternation = "|".join(re.escape(name) for name in ordered)
        return cls(re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE))

    def detect(self, text: str) -> Iterator[Detection]:
        """Every whole-word occurrence of a directory name in `text`, in start order.

        `finditer` and not `search`: a name mentioned twice must be replaced
        twice, and a second mention left in the clear behind a response that
        reports a redaction is the worst failure shape this control has
        (`test_every_occurrence_of_a_name_is_a_candidate`).
        """
        if self._pattern is None:
            return
        for match in self._pattern.finditer(text):
            yield Detection(
                kind=Kind.PERSON,
                start=match.start(),
                end=match.end(),
                value=match.group(0),
            )
