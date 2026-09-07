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

`test_a_longer_placeholder_is_not_eaten_by_a_shorter_one` is about loose
matching, not about ordering on its own: every key ends in `>`, so a literal,
`re.escape`d alternation cannot match `<EMAIL_1>` inside `<EMAIL_11>` whatever
order its branches are in. What that row kills is an implementation that matches
the KIND and a bare number (`<EMAIL_1[0-9]*>`) or looks a placeholder up by
`startswith`, either of which restores `<EMAIL_11>` as `<EMAIL_1>`'s value with a
stray `1>` left behind.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from opsconsole.redaction import rehydrate

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
