r"""`NameDirectory`: the compiled alternation, its boundaries and its escaping.

The matcher is the tenth kind's whole implementation, and it is the one detector
whose input is not the text. So the cases here are arranged around the four ways
a compiled alternation goes wrong.

* Boundaries. A name is matched as a whole word or not at all: `Annas` and
  `xAnnax` are the two directions, and each is checked with the SAME directory
  that matches the bare name in a neighbouring row, so "matched nothing at all"
  cannot pass a rejection row.
* Order. `Anna Schmidt` has to beat `Anna` whichever order the two were pushed
  in, because a Python alternation is first-alternative-wins and insertion order
  is what a store hands back.
* Escaping. `re.escape` is asserted twice: a metacharacter name matches itself
  literally, and it does not change what a neighbouring name in the same
  alternation matches - the second is the failure a single-name row cannot see.
* The invariant every other detector holds: `value == text[start:end]`, and the
  kind is `PERSON`.

The worst case (the `MAX_DIRECTORY_NAMES` compiled and run over a
`MAX_REDACTION_INPUT_CHARS` input) is asserted as a BOUND and never timed: a
wall-clock assertion is a flake on a loaded machine, and what the risk section
actually needs to know is that the largest legal directory over the largest legal
input still answers correctly.
"""

from __future__ import annotations

import re

import pytest

from app.redaction import Kind
from app.redaction.detectors import Detection
from app.redaction.names import NameDirectory
from app.redaction.names_store import MAX_DIRECTORY_NAMES
from app.redaction.redactor import MAX_REDACTION_INPUT_CHARS

#: The boundary `NameDirectory` compiles, as a guard a case can state: `(?<!\w)` refuses a
#: name glued to a word character, and `_` is one although `str.isalnum` says otherwise.
NON_WORD = re.compile(r"\W")

FULL_NAME = "Anna Schmidt"
FIRST_NAME = "Anna"
OTHER_NAME = "Bernd Kowalski"
META_NAME = "A. B*"


def found(directory: NameDirectory, text: str) -> list[tuple[Kind, int, int, str]]:
    """Every candidate as a comparable tuple, so a row asserts the span and the kind."""
    return [(d.kind, d.start, d.end, d.value) for d in directory.detect(text)]


def spans(directory: NameDirectory, text: str) -> set[tuple[int, int]]:
    """Just the offsets, for the rows that are about which run matched."""
    return {(d.start, d.end) for d in directory.detect(text)}


def test_the_matcher_is_exported_from_the_package() -> None:
    """Catches a matcher that lands in `names.py` and is never re-exported: the gateway
    routes import the engine's public names from `app.redaction`, so a missing
    export is an ImportError at process start rather than a failure in a test."""
    from app import redaction

    assert redaction.NameDirectory is NameDirectory


def test_a_stored_name_in_running_text_is_one_person_candidate() -> None:
    """Catches a matcher that never fires at all, which would make every rejection row
    below pass for the wrong reason, and one that labels its candidates with some other
    kind."""
    directory = NameDirectory.compile([FULL_NAME])

    assert found(directory, "Ansprechpartner: Anna Schmidt heute") == [
        (Kind.PERSON, 17, 29, FULL_NAME)
    ]


def test_a_candidates_value_indexes_its_own_offsets() -> None:
    """Catches a matcher reporting the DIRECTORY's spelling as the value while the offsets
    point at the text's: the redactor slices the text by the offsets and the map stores
    the value, so a mismatch files the wrong string under the placeholder. The case
    matches on a different casing, which is the only arrangement where the two can
    differ."""
    text = "ansprechpartner: ANNA SCHMIDT heute"
    directory = NameDirectory.compile([FULL_NAME])

    for detection in directory.detect(text):
        assert detection.value == text[detection.start : detection.end]
        assert detection.value == "ANNA SCHMIDT"


REJECTED_CONTEXTS: list[tuple[str, str]] = [
    ("a suffix glued to the name", "Annas Tasche steht dort"),
    ("a prefix glued to the name", "xAnna kam nicht"),
    ("glued on both sides", "xAnnax kam nicht"),
    ("inside a longer alphanumeric run", "the token xAnnaSchmidtx is not a person"),
    ("part of a longer word", "Anhang Annahme abgelehnt"),
]


@pytest.mark.parametrize(
    ("case", "text"), REJECTED_CONTEXTS, ids=[row[0] for row in REJECTED_CONTEXTS]
)
def test_a_name_inside_a_longer_word_is_not_a_candidate(case: str, text: str) -> None:
    """Catches an alternation compiled with no boundary lookarounds, which is the
    precision failure this guards against: a two-character minimum name would
    otherwise redact a substring of every ordinary word a tenant sends. The same
    directory matches the bare name in the row above, so "matched nothing" is not what
    makes these pass."""
    directory = NameDirectory.compile([FIRST_NAME, FULL_NAME])

    assert found(directory, text) == []


@pytest.mark.parametrize(
    ("case", "text"),
    [
        ("lowercase", "schreib an anna schmidt heute"),
        ("uppercase", "schreib an ANNA SCHMIDT heute"),
        ("mixed case", "schreib an aNnA sChMiDt heute"),
        ("the directory's own casing", "schreib an Anna Schmidt heute"),
    ],
)
def test_matching_is_case_insensitive(case: str, text: str) -> None:
    """Catches an alternation compiled without `re.IGNORECASE`, or one that lowercases
    the TEXT before matching and then reports offsets into a string the caller never
    sent: every row asserts the exact span, so a case-folded copy with a different length
    cannot pass."""
    directory = NameDirectory.compile([FULL_NAME])

    assert spans(directory, text) == {(11, 23)}


@pytest.mark.parametrize(
    ("case", "names"),
    [
        ("the short name pushed first", (FIRST_NAME, FULL_NAME)),
        ("the long name pushed first", (FULL_NAME, FIRST_NAME)),
    ],
)
def test_the_longest_name_wins_over_its_own_first_token(case: str, names: tuple[str, ...]) -> None:
    """Catches an alternation built in insertion order: a Python alternation takes the
    FIRST branch that matches, so `Anna|Anna Schmidt` leaves ` Schmidt` in the clear
    behind a `<PERSON_1>`. Both push orders are swept, because sorting is the only
    implementation that satisfies both."""
    directory = NameDirectory.compile(names)

    assert found(directory, "cc Anna Schmidt today") == [(Kind.PERSON, 3, 15, FULL_NAME)]


def test_the_shorter_name_still_matches_where_the_longer_one_does_not() -> None:
    """Catches a matcher that keeps only the longest name in the directory: sorting
    longest-first must decide which branch wins on one run of text, never which names are
    in the pattern at all."""
    directory = NameDirectory.compile([FIRST_NAME, FULL_NAME])

    assert found(directory, "cc Anna today") == [(Kind.PERSON, 3, 7, FIRST_NAME)]


def test_every_occurrence_of_a_name_is_a_candidate() -> None:
    """Catches a matcher using `re.search` rather than `finditer`: the second mention
    would stay in the clear while the response still reported a redaction, which is the
    worst failure shape this control has."""
    directory = NameDirectory.compile([FIRST_NAME])

    assert spans(directory, "Anna wrote, then Anna called") == {(0, 4), (17, 21)}


def test_two_different_names_are_two_candidates() -> None:
    """Catches an alternation that compiled only its first or its last entry, which a
    single-name directory can never show."""
    directory = NameDirectory.compile([FULL_NAME, OTHER_NAME])

    assert spans(directory, "Anna Schmidt met Bernd Kowalski") == {(0, 12), (17, 31)}


def test_a_name_with_regex_metacharacters_matches_itself_literally() -> None:
    """Catches a name interpolated into the pattern unescaped: `A. B*` would compile to
    "any character, then B repeated", so `Ax BBB` would be redacted and the operator's
    actual name would not. An unescaped `*` after an empty group is also a compile-time
    error, which turns every request for that tenant into a 500."""
    directory = NameDirectory.compile([META_NAME])

    assert found(directory, "ping A. B* about it") == [(Kind.PERSON, 5, 10, META_NAME)]


@pytest.mark.parametrize(
    ("case", "text"),
    [
        ("the dot as a wildcard", "ping Ax B* about it"),
        ("the star as a repetition", "ping A. BBB about it"),
        ("both metacharacters read as regex", "ping Ax BBBB about it"),
    ],
)
def test_a_metacharacter_name_does_not_match_what_the_regex_would(case: str, text: str) -> None:
    """The other half of the escaping pair: catches a name that compiles and matches its
    own spelling while ALSO eating unrelated text, which the literal row above cannot
    see."""
    directory = NameDirectory.compile([META_NAME])

    assert found(directory, text) == []


def test_a_metacharacter_name_does_not_change_what_a_neighbour_matches() -> None:
    """Catches an escape applied to the alternation as a whole rather than per name, and
    a name carrying `|` or `)` that would otherwise end the group early and silently drop
    every name after it: the neighbour is pushed AFTER the metacharacter name, so a
    broken group takes it out."""
    directory = NameDirectory.compile(["A. B*", "Ma|ke) Fischer", FULL_NAME])

    assert spans(directory, "A. B* and Anna Schmidt and Ma|ke) Fischer") == {
        (0, 5),
        (10, 22),
        (27, 41),
    }


def test_an_empty_directory_yields_nothing_and_does_not_raise() -> None:
    """Catches an empty alternation compiled as `()` or `(?:)`, which matches the empty
    string at every position: every request for a tenant with no names pushed yet would
    become one `<PERSON_N>` per character. The text carries a name-shaped run so a
    matcher that fired on words rather than on the directory would show."""
    directory = NameDirectory.compile([])

    assert found(directory, "Anna Schmidt is not in this directory") == []


def test_a_name_at_the_very_end_of_the_input_is_still_a_candidate() -> None:
    """Catches boundary lookarounds written with `\\b` against a following character that
    does not exist, or a pattern anchored to a trailing separator: the last turn of a
    chat message routinely ends on a name."""
    directory = NameDirectory.compile([FULL_NAME])

    assert spans(directory, "regards, Anna Schmidt") == {(9, 21)}


@pytest.mark.parametrize(
    ("case", "text"),
    [
        ("a trailing comma", "cc Anna Schmidt, please"),
        ("a trailing newline", "cc Anna Schmidt\nplease"),
        ("a trailing CRLF", "cc Anna Schmidt\r\nplease"),
        ("surrounded by quotes", 'cc "Anna Schmidt" please'),
    ],
)
def test_punctuation_around_a_name_is_not_part_of_the_span(case: str, text: str) -> None:
    """Catches a span that swallows the character after the name: the redactor slices by
    these offsets, so a comma inside the span disappears from the caller's text and
    rehydration puts it back in the wrong place. the project's analogue for a
    free-text detector, the same one `test_detectors.py` applies to the nine patterns."""
    start = text.index(FULL_NAME)

    directory = NameDirectory.compile([FULL_NAME])

    assert spans(directory, text) == {(start, start + len(FULL_NAME))}


def test_the_largest_legal_directory_over_the_largest_legal_input_still_answers() -> None:
    r"""Catches a matcher that compiles per name and scans the text once per entry, or one
    whose alternation exceeds a regex engine limit at the cap: 2000 names
    is a legal directory and 16384 characters is a legal redaction input, so this pair is
    reachable in production. Asserted as a bound and never timed.

    The filler is cut to length and then given a trailing SPACE, because truncating it
    at an arbitrary offset can end mid-word: the matcher's `(?<!\w)` boundary is right
    to refuse a name glued to the end of `quic`, so a planted name preceded by a word
    character would fail this case for the arrangement rather than for the bound.
    """
    names = [f"Persona{index:04d} Nachname" for index in range(MAX_DIRECTORY_NAMES)]
    filler = "the quick brown fox jumps over the lazy dog. "
    planted = names[MAX_DIRECTORY_NAMES - 1]
    before = (filler * (MAX_REDACTION_INPUT_CHARS // len(filler)))[
        : MAX_REDACTION_INPUT_CHARS - len(planted) - 1
    ] + " "
    text = before + planted

    assert len(names) == MAX_DIRECTORY_NAMES
    assert len(text) == MAX_REDACTION_INPUT_CHARS
    assert NON_WORD.match(text[-len(planted) - 1])

    directory = NameDirectory.compile(names)

    assert found(directory, text) == [(Kind.PERSON, len(text) - len(planted), len(text), planted)]


def test_a_candidate_is_a_detection_of_the_shape_every_other_detector_yields() -> None:
    """Catches a matcher yielding its own tuple or a bare `(start, end)` pair: `detect_all`
    puts these candidates into the same list as the nine pattern detectors and
    `resolve_overlaps` reads `.kind`, `.start`, `.end` and `.value` off each one."""
    directory = NameDirectory.compile([FULL_NAME])

    candidates = list(directory.detect("cc Anna Schmidt"))

    assert len(candidates) == 1
    assert isinstance(candidates[0], Detection)
