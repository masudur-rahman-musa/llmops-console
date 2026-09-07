r"""`rehydrate(text, mapping)`: the pure substitution an earlier change's route is built on.

Everything here is in-process string work - no store, no app, no backing service
- so a project rule has nothing to apply to and there is no fixture to share.
The map literals are the shape `redact` hands over: `<KIND_N>` -> the raw value.

Three cases carry the claims that are not obvious from the signature, and each is
arranged so that only the control it names can decide the outcome:

* ONE PASS. `test_a_restored_value_containing_a_placeholder_is_not_substituted_again`
  uses a map whose FIRST value contains the SECOND key, so the naive
  `for key, value in mapping.items(): text = text.replace(key, value)` loop -
  which is what `tests/gateway/test_redact_endpoint.py`'s local helper does, and
  what a first implementation reaches for - rewrites the injected token and
  reddens, while one `re.sub` over an alternation cannot.
* AN UNNAMED PLACEHOLDER. The map deliberately does not name `<CARD_9>`, so an
  implementation that scans for the generic `<KIND_N>` shape and looks each hit
  up (blanking, or raising `KeyError`, on a miss) is caught; an alternation built
  from the map's own keys cannot see that token at all.
* AN EMPTY MAP. The identity, and the one input that turns a carelessly built
  alternation into `re.compile("")`, which matches at every position.

`StreamRehydrator` is the same substitution with a carry, and its
cases are here rather than behind a `TestClient` because it needs no HTTP, no
tenant and no store. Its arrangements are chosen so that only the carry can
decide the outcome:

* THE SPLIT. Every row of `test_a_placeholder_split_across_two_deltas_is_rehydrated`
  cuts ONE placeholder in two at a different position, so no single feed ever
  contains a whole key and a per-delta `rehydrate` with no buffer cannot restore
  any of them.
* THE HOLD IS RAW. `TRAILING_PREFIX_MAPPING`'s first value ENDS in the second
  key's opening characters, which is the only arrangement in which "the hold is
  computed on the emitted text" and "the hold is computed on the raw buffer"
  differ visibly: the first re-scans a span it has already handed out and
  substitutes inside its own output.
* THE BOUND. The near-miss run in `test_the_buffer_holds_less_than_the_longest_key`
  presents `<PERSON`, `<EMAIL` and a bare `<` in turn and completes none of them,
  so the hold is exercised at every length without a single substitution making
  the emitted text differ from the raw one - which is what lets the held span be
  measured as "fed so far minus emitted so far".

`test_a_longer_placeholder_is_not_eaten_by_a_shorter_one` is about loose
matching, not about ordering on its own: every key ends in `>`, so a literal,
`re.escape`d alternation cannot match `<EMAIL_1>` inside `<EMAIL_11>` whatever
order its branches are in. What that row kills is an implementation that matches
the KIND and a bare number (`<EMAIL_1[0-9]*>`) or looks a placeholder up by
`startswith`, either of which restores `<EMAIL_11>` as `<EMAIL_1>`'s value with a
stray `1>` left behind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from opsconsole.redaction import StreamRehydrator, rehydrate

EMAIL_A = "alice@example.com"
EMAIL_B = "bob@example.net"
PERSON_A = "Alice Schmidt"

# One map exactly as `redact` produces it: placeholder -> the raw value it stood in for.
MAPPING: Mapping[str, str] = {
    "<EMAIL_1>": EMAIL_A,
    "<EMAIL_2>": EMAIL_B,
    "<PERSON_1>": PERSON_A,
}

# A map whose FIRST value carries the SECOND key inside it. Nothing a redaction
# run produces, and that is the point: it is the only arrangement in which a
# second substitution pass is visible in the output rather than merely possible.
NESTED_MAPPING: Mapping[str, str] = {
    "<EMAIL_1>": "bob+<EMAIL_2>@example.com",
    "<EMAIL_2>": "carol@example.org",
}

# The `<EMAIL_1>` / `<EMAIL_11>` pair from the plan's section 6.
DOUBLE_DIGIT_MAPPING: Mapping[str, str] = {
    "<EMAIL_1>": EMAIL_A,
    "<EMAIL_11>": "kim@example.org",
}

EMPTY_MAPPING: Mapping[str, str] = {}

# A map whose FIRST value ENDS in the opening characters of the SECOND key. Like
# `NESTED_MAPPING` it is nothing a redaction run produces, and the shape is
# load-bearing: an implementation that computes its hold on the text it has
# already rehydrated holds `<EMAIL_2` out of its own output, sees the `>` of the
# next feed complete it, and substitutes a second time.
TRAILING_PREFIX_MAPPING: Mapping[str, str] = {
    "<EMAIL_1>": "bob+<EMAIL_2",
    "<EMAIL_2>": "carol@example.org",
}

# The longest key of `MAPPING`, and its length. Both are re-derived from the map
# inside `test_the_buffer_holds_less_than_the_longest_key`, so adding a longer
# key to `MAPPING` fails that case rather than silently loosening its bound.
LONGEST_KEY = "<PERSON_1>"
LONGEST_KEY_LENGTH = 10

# One run of near misses, fed in pieces: `<PERSON`, `<EMAIL` and a bare `<` are
# each a proper prefix of a key, and none of `<PERSON_9>`, `<EMAIL_3>` or
# `<PERSON_4>` is IN the map, so nothing is substituted anywhere in the run and
# the emitted text is the raw text.
NEAR_MISS_CHUNKS: tuple[str, ...] = (
    "<PERSON",
    "_9> wrote ",
    "to <EMAIL",
    "_3",
    "> and to <",
    "PERSON_",
    "4>. ",
)

# A long answer with no angle bracket at all: the path almost every streamed
# request takes, and the one where holding text to the end of the stream would
# turn a live answer into a single block delivered at the end.
ORDINARY_CHUNKS: tuple[str, ...] = (
    "Berlin",
    " is the ",
    "capital of Germany, and the largest city ",
    "in the European Union by population.",
)


def test_one_placeholder_is_replaced_by_its_raw_value() -> None:
    """Catches a function that returns its input unchanged, or that deletes the
    placeholder instead of putting the value back: the raw address is asserted
    present AND the placeholder absent, so neither half alone passes."""
    restored = rehydrate("mail <EMAIL_1> today", MAPPING)

    assert restored == f"mail {EMAIL_A} today"
    assert "<EMAIL_1>" not in restored


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("<EMAIL_1>", EMAIL_A, id="the whole string is one placeholder"),
        pytest.param(
            "mail <EMAIL_1> and <EMAIL_2>",
            f"mail {EMAIL_A} and {EMAIL_B}",
            id="two placeholders of the same kind",
        ),
        pytest.param(
            "<EMAIL_1> wrote to <EMAIL_1>",
            f"{EMAIL_A} wrote to {EMAIL_A}",
            id="the same placeholder twice",
        ),
        pytest.param(
            "<PERSON_1><EMAIL_1>",
            f"{PERSON_A}{EMAIL_A}",
            id="two adjacent placeholders of different kinds",
        ),
        pytest.param(
            "<EMAIL_2>, cc <PERSON_1>.",
            f"{EMAIL_B}, cc {PERSON_A}.",
            id="a placeholder at the start and one before punctuation",
        ),
    ],
)
def test_every_placeholder_the_map_names_is_replaced(text: str, expected: str) -> None:
    """Catches a substitution that stops after the first hit (`re.sub(..., count=1)`,
    or a `return` inside the loop over the keys): the repeated-placeholder row leaves the
    second `<EMAIL_1>` standing, and the two-kind rows leave whichever key the
    implementation reached first as the only one restored."""
    assert rehydrate(text, MAPPING) == expected


def test_a_placeholder_absent_from_the_map_is_left_alone() -> None:
    """Catches an implementation that finds placeholders with a generic `<KIND_N>` scan
    and then looks each one up in the map: an unnamed token either becomes an empty
    string or raises `KeyError`, and both would corrupt an answer the model wrote."""
    restored = rehydrate("<EMAIL_1> and <CARD_9>", MAPPING)

    assert restored == f"{EMAIL_A} and <CARD_9>"


def test_a_restored_value_containing_a_placeholder_is_not_substituted_again() -> None:
    """Catches a per-key `str.replace` loop: it puts `bob+<EMAIL_2>@example.com` in, then
    the `<EMAIL_2>` pass rewrites the token it just injected into
    `bob+carol@example.org@example.com` - a value the caller never wrote, which is the
    same class of injection the redactor's own placeholder escaping exists to prevent."""
    restored = rehydrate("mail <EMAIL_1> and <EMAIL_2>", NESTED_MAPPING)

    assert restored == "mail bob+<EMAIL_2>@example.com and carol@example.org"


def test_an_empty_map_is_the_identity() -> None:
    """Catches an alternation built without a guard for the empty case: `re.compile("")`
    matches at every position, so an empty map would insert the replacement between every
    two characters rather than leaving the answer alone."""
    text = "nothing was redacted here, but <EMAIL_1> looks like it was"

    assert rehydrate(text, EMPTY_MAPPING) == text
    assert rehydrate("", EMPTY_MAPPING) == ""


def test_a_longer_placeholder_is_not_eaten_by_a_shorter_one() -> None:
    r"""Catches an implementation that matches a placeholder loosely - by its kind and a
    bare number (`<EMAIL_1[0-9]*>`), or by `startswith` over the keys - which restores
    `<EMAIL_11>` as `<EMAIL_1>`'s value with a stray `1>` left after it, handing one
    person's address out under another's placeholder."""
    restored = rehydrate("<EMAIL_11> not <EMAIL_1>", DOUBLE_DIGIT_MAPPING)

    assert restored == f"kim@example.org not {EMAIL_A}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("nobody redacted anything in this sentence", id="plain prose"),
        pytest.param("2 < 3 and 4 > 1", id="angle brackets that are not a placeholder"),
        pytest.param("<email_1> <EMAIL-1> <EMAIL_>", id="near-miss placeholder shapes"),
        pytest.param("", id="the empty string"),
    ],
)
def test_text_with_no_placeholder_is_returned_unchanged(text: str) -> None:
    """Catches an implementation that rewrites the text it did not match - stripping
    angle brackets, normalising whitespace, or answering an empty string when the
    alternation found nothing - on the path most requests take, since most answers
    carry no placeholder at all."""
    assert rehydrate(text, MAPPING) == text


# --------------------------------------------------------------------------
# StreamRehydrator: the same substitution with a carry
# --------------------------------------------------------------------------


def fed(rehydrator: StreamRehydrator, chunks: Sequence[str]) -> str:
    """Everything `rehydrator` emits for `chunks`, the final `flush` included.

    The assembled answer is what a client actually reads, and it is the only
    reading in which a lost carry, a duplicated carry and a re-scanned span are
    all visible at once; the per-feed answers are asserted separately by the
    cases that are about WHEN text leaves rather than about what it says.
    """
    return "".join(rehydrator.feed(chunk) for chunk in chunks) + rehydrator.flush()


# `<PERSON_1>` occupies positions 9 through 19 of this sentence, so a cut at 10
# through 18 splits the placeholder itself rather than the text around it.
SPLIT_TEXT = "write to <PERSON_1> now"
SPLIT_EXPECTED = f"write to {PERSON_A} now"


@pytest.mark.parametrize("cut", list(range(10, 19)), ids=[f"cut after {n}" for n in range(10, 19)])
def test_a_placeholder_split_across_two_deltas_is_rehydrated(cut: int) -> None:
    """Catches a route that calls the pure `rehydrate` once per delta with no carry: on
    every row here neither half contains a whole key, so nothing matches and the client
    reads `<PERSON_1>` in the answer it asked to have names put back into. Each cut is a
    different position inside the placeholder, so an implementation that carries only the
    final `>`, or only a leading `<`, fails the rows it does not cover."""
    rehydrator = StreamRehydrator(MAPPING)

    assert fed(rehydrator, (SPLIT_TEXT[:cut], SPLIT_TEXT[cut:])) == SPLIT_EXPECTED


def test_a_placeholder_split_one_character_at_a_time_is_rehydrated() -> None:
    """Catches a carry that holds at most one delta - a `previous` string replaced on
    every feed rather than a buffer appended to - which restores a placeholder split in
    two and loses one split across three or more. Two placeholders of different lengths
    are fed here, so an implementation that carries a fixed number of characters is
    caught on whichever of them is longer."""
    text = "mail <EMAIL_1> and <PERSON_1>"
    rehydrator = StreamRehydrator(MAPPING)

    assert fed(rehydrator, tuple(text)) == f"mail {EMAIL_A} and {PERSON_A}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("2 < 3 and 4 > 1", id="angle brackets that are not a placeholder"),
        pytest.param("wrap it in a <div> and ship", id="a complete tag the map cannot name"),
        pytest.param("<EMAIL-1> is not a placeholder", id="a near-miss placeholder shape"),
    ],
)
def test_text_that_cannot_complete_a_placeholder_is_not_held(text: str) -> None:
    """Catches decision B's rejected alternative - holding everything after the last
    unclosed `<` - which keeps `< 3 and 4 > 1` buffered until the stream ends and turns a
    live answer into one block delivered at the finish. `flush` is asserted empty as well
    as the feed being whole, so an implementation that emitted the text AND kept a copy
    (and would emit it twice) fails too."""
    rehydrator = StreamRehydrator(MAPPING)

    assert rehydrator.feed(text) == text
    assert rehydrator.flush() == ""


def test_an_empty_map_makes_every_feed_the_identity() -> None:
    """Catches a rehydrator that computes its hold from the generic `<KIND_N>` shape
    rather than from the map's own keys: with no keys there is nothing a suffix can be a
    prefix OF, so `<EMAIL_` must be handed straight out. A shape-based hold buffers it to
    the end of a stream that was never redacted at all, which is the majority of streams,
    and `re.compile("")` on an empty alternation would rewrite the text besides."""
    rehydrator = StreamRehydrator(EMPTY_MAPPING)
    chunks = ("nothing was redacted, ", "but <EMAIL_", "1> looks like it was")

    assert [rehydrator.feed(chunk) for chunk in chunks] == list(chunks)
    assert rehydrator.flush() == ""


def test_flush_answers_the_remainder_and_empties_the_buffer() -> None:
    """Catches a `flush` that reports the carry without clearing it, which emits the same
    span again on the next stream this object serves, and a `flush` that clears without
    reporting, which drops the last of an answer. The second call is what separates the
    two: it must be empty, and it cannot be unless the first both answered and emptied."""
    rehydrator = StreamRehydrator(MAPPING)

    assert rehydrator.feed("write to <PERSON_") == "write to "
    assert rehydrator.flush() == "<PERSON_"
    assert rehydrator.flush() == ""
    assert StreamRehydrator(MAPPING).flush() == ""


def test_a_restored_value_is_not_rescanned_by_a_later_delta() -> None:
    """Catches a rehydrator that buffers its OUTPUT instead of the raw text: this map's
    first value ends in `<EMAIL_2`, so an output-side hold keeps those characters back,
    the `>` of the next feed completes them, and the answer carries
    `bob+carol@example.org` - a value nobody wrote, which is the injection 12b's single
    pass exists to prevent. A raw buffer cannot see a span it has already emitted."""
    rehydrator = StreamRehydrator(TRAILING_PREFIX_MAPPING)

    assert fed(rehydrator, ("mail <EMAIL_1>", "> today")) == "mail bob+<EMAIL_2> today"


def test_the_buffer_holds_less_than_the_longest_key() -> None:
    """Catches an unbounded hold - "keep everything since the last `<`", or "keep
    everything until a key matches" - which grows with the answer rather than with the
    map and holds an entire streamed reply that happens to mention `<`. The inclusive
    boundary is asserted on the longest key itself: at its full length it is a key, not a
    proper prefix, so it completes and is emitted, and one character less is held."""
    assert max(MAPPING, key=len) == LONGEST_KEY
    assert len(LONGEST_KEY) == LONGEST_KEY_LENGTH

    boundary = StreamRehydrator(MAPPING)
    assert boundary.feed(LONGEST_KEY) == PERSON_A
    assert boundary.flush() == ""

    one_less = StreamRehydrator(MAPPING)
    assert one_less.feed(LONGEST_KEY[:-1]) == ""
    assert one_less.flush() == LONGEST_KEY[:-1]
    assert len(LONGEST_KEY[:-1]) == LONGEST_KEY_LENGTH - 1

    rehydrator = StreamRehydrator(MAPPING)
    fed_so_far = ""
    emitted_so_far = ""
    for chunk in NEAR_MISS_CHUNKS:
        fed_so_far += chunk
        emitted_so_far += rehydrator.feed(chunk)
        # No key in this run is ever completed, so nothing is substituted and the
        # difference in LENGTH is exactly the number of characters still held.
        assert len(fed_so_far) - len(emitted_so_far) < LONGEST_KEY_LENGTH

    assert emitted_so_far + rehydrator.flush() == fed_so_far


def test_a_run_of_ordinary_text_retains_nothing() -> None:
    """Catches a rehydrator that holds a fixed tail of every feed - the simplest way to
    make the split cases pass - which delays every fragment of the answer by one delta on
    the path almost every request takes, and is invisible to any assertion that only reads
    the assembled text at the end. Each feed is asserted to answer its own chunk WHOLE."""
    rehydrator = StreamRehydrator(MAPPING)

    assert [rehydrator.feed(chunk) for chunk in ORDINARY_CHUNKS] == list(ORDINARY_CHUNKS)
    assert rehydrator.flush() == ""
