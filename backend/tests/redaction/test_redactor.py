r"""Overlap resolution, placeholder assignment, the map and the input cap.

`resolve_overlaps` is exercised on hand-built `Detection` objects rather than on
text, because a text-driven overlap depends on which patterns happen to collide
today: the tie-break rows would then be passing for whichever detector fired,
not for the merge rule they claim to pin. The tie-break rows read the real
`KIND_PRECEDENCE` by index instead of hard-coding an order left to
the implementation, so they redden on a reversed or ignored precedence and stay
green on a re-ordering that is still honoured.

The cap's "before any scanning" half is asserted with a monkeypatched
`detect_all` recorder, paired with a positive control that asserts the SAME
recorder IS called for an accepted input: without the pair, "was not called"
would also pass if the monkeypatch missed its seam entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.redaction import KIND_PRECEDENCE as PRECEDENCE
from app.redaction import (
    MAX_REDACTION_INPUT_CHARS,
    Detection,
    Kind,
    NameDirectory,
    RedactionInputTooLong,
    redact,
)
from app.redaction.redactor import resolve_overlaps

from .conftest import PLACEHOLDER_PATTERN, placeholders_in_order

EMAIL_A = "alice@example.com"
EMAIL_B = "bob@example.net"
CARD_A = "4111111111111111"
IBAN_A = "DE12500105170648489890"
PHONE_A = "+4930123456789"
IPV4_A = "192.0.2.1"


def detection(kind: Kind, start: int, end: int, value: str = "x") -> Detection:
    """A synthetic candidate. `value` is irrelevant to the merge rules."""
    return Detection(kind=kind, start=start, end=end, value=value)


def as_tuples(detections: Sequence[Detection]) -> list[tuple[Kind, int, int]]:
    return [(d.kind, d.start, d.end) for d in detections]


# --------------------------------------------------------------------------
# overlap resolution
# --------------------------------------------------------------------------


def test_non_overlapping_candidates_all_survive_in_start_order() -> None:
    """Catches a merge that drops any candidate that merely touches another, and
    one that returns the input in the order the detectors happened to run."""
    candidates = [
        detection(Kind.IPV4, 20, 29),
        detection(Kind.EMAIL, 0, 10),
        detection(Kind.CARD, 10, 20),
    ]

    assert as_tuples(resolve_overlaps(candidates)) == [
        (Kind.EMAIL, 0, 10),
        (Kind.CARD, 10, 20),
        (Kind.IPV4, 20, 29),
    ]


def test_longer_span_beats_shorter_overlapping_span() -> None:
    """Catches a merge that keeps the first candidate it sees.

    The shorter span is given the earlier start AND the earlier position in the
    input, so first-wins and earliest-start-wins both produce the wrong answer;
    only the length comparison yields the long span.
    """
    candidates = [detection(Kind.CARD, 0, 10), detection(Kind.IBAN, 2, 24)]

    assert as_tuples(resolve_overlaps(candidates)) == [(Kind.IBAN, 2, 24)]


def test_longer_span_wins_when_it_is_seen_second_and_starts_later() -> None:
    """Catches a length comparison written `<` instead of `>` - a merge that
    prefers the shorter span still passes a single-orientation row."""
    candidates = [detection(Kind.IBAN, 2, 24), detection(Kind.CARD, 6, 16)]

    assert as_tuples(resolve_overlaps(candidates)) == [(Kind.IBAN, 2, 24)]


def test_equal_length_overlap_falls_to_the_precedence_order() -> None:
    """Catches a merge with no tie-break at all, and one that reverses the order.

    The LAST kind in `KIND_PRECEDENCE` is placed first in the input, so a merge
    that keeps its first candidate returns the loser.
    """
    winner, loser = PRECEDENCE[0], PRECEDENCE[-1]
    candidates = [detection(loser, 0, 12), detection(winner, 0, 12)]

    assert as_tuples(resolve_overlaps(candidates)) == [(winner, 0, 12)]


def test_equal_length_overlap_precedence_holds_in_the_other_input_order() -> None:
    """Catches a tie-break that is really just "keep the first seen": the same two
    candidates, swapped, must give the same winner."""
    winner, loser = PRECEDENCE[0], PRECEDENCE[-1]
    candidates = [detection(winner, 0, 12), detection(loser, 0, 12)]

    assert as_tuples(resolve_overlaps(candidates)) == [(winner, 0, 12)]


def test_same_kind_equal_length_overlap_falls_to_the_earlier_start() -> None:
    """Catches a tie-break that stops at precedence and then picks arbitrarily,
    which makes the resolved span list depend on dict or set iteration order."""
    candidates = [detection(Kind.PHONE, 4, 14), detection(Kind.PHONE, 0, 10)]

    assert as_tuples(resolve_overlaps(candidates)) == [(Kind.PHONE, 0, 10)]


def test_resolved_spans_are_disjoint_across_a_chain_of_overlaps() -> None:
    """Catches a merge that resolves pairs but leaves a third span interleaved with
    the winner, which is exactly the nested-placeholder failure the acceptance
    criterion names."""
    candidates = [
        detection(Kind.CARD, 0, 8),
        detection(Kind.IBAN, 4, 20),
        detection(Kind.PHONE, 16, 24),
    ]

    resolved = resolve_overlaps(candidates)

    assert as_tuples(resolved) == [(Kind.IBAN, 4, 20)]


def test_resolve_overlaps_on_no_candidates_is_empty() -> None:
    assert resolve_overlaps([]) == []


# --------------------------------------------------------------------------
# placeholders and the map
# --------------------------------------------------------------------------


def test_placeholders_are_numbered_per_kind_from_one_in_first_appearance_order() -> None:
    """Catches a counter shared across kinds (the card would be `<CARD_2>`) and one
    numbered by sort order rather than by first appearance."""
    text = f"Mail {EMAIL_A}, card {CARD_A}, mail {EMAIL_B}, mail {EMAIL_A}."

    result = redact(text)

    assert placeholders_in_order(text, result) == [
        "<EMAIL_1>",
        "<CARD_1>",
        "<EMAIL_2>",
        "<EMAIL_1>",
    ]


def test_repeated_value_reuses_its_placeholder_and_the_map_has_one_entry() -> None:
    """Catches an assignment keyed on the span offsets instead of on
    `(kind, value)`, which would emit `<EMAIL_1>` and `<EMAIL_2>` for the same
    address and store it twice."""
    text = f"{EMAIL_A} asked, so reply to {EMAIL_A}."

    result = redact(text)

    assert placeholders_in_order(text, result) == ["<EMAIL_1>", "<EMAIL_1>"]
    assert dict(result.mapping) == {"<EMAIL_1>": EMAIL_A}


def test_distinct_values_of_one_kind_never_share_a_placeholder() -> None:
    """Catches an assignment keyed on the kind alone, which would collapse every
    address in a request into one placeholder and make rehydration lossy."""
    text = f"From {EMAIL_A} to {EMAIL_B}."

    result = redact(text)

    assert dict(result.mapping) == {"<EMAIL_1>": EMAIL_A, "<EMAIL_2>": EMAIL_B}


def test_mapping_keys_are_exactly_the_placeholders_present_in_the_output() -> None:
    """Catches a map that carries entries for candidates the overlap merge dropped,
    which would be stored and then never be able to rehydrate."""
    text = f"Pay {IBAN_A} from {IPV4_A}, ring {PHONE_A}."

    result = redact(text)

    emitted = {match.group(0) for match in PLACEHOLDER_PATTERN.finditer(result.text)}
    assert set(result.mapping) == emitted


def test_mapping_round_trips_the_redacted_text_back_to_the_original() -> None:
    """Catches a map holding a normalised value (stripped spaces, upper-cased) - the
    text no longer reconstructs, which is what rehydration needs."""
    text = f"Mail {EMAIL_A}, pay {IBAN_A}, ring {PHONE_A}, mail {EMAIL_A}."

    result = redact(text)
    restored = result.text
    for placeholder, original in result.mapping.items():
        restored = restored.replace(placeholder, original)

    assert restored == text


def test_detection_offsets_index_the_original_text_not_the_redacted_text() -> None:
    """Catches offsets rebased onto the output as placeholders are substituted, which
    drift by the length difference from the second detection onwards."""
    text = f"Mail {EMAIL_A}, pay {IBAN_A}, ring {PHONE_A}."

    result = redact(text)

    assert len(result.detections) == 3
    for found in result.detections:
        assert text[found.start : found.end] == found.value


def test_text_with_no_detections_is_returned_unchanged_with_an_empty_map() -> None:
    """Catches a redactor that rebuilds the string through the substitution path
    and drops or duplicates a character when there is nothing to substitute."""
    text = "Order ORD-2024-88213 shipped yesterday."

    result = redact(text)

    assert result.text == text
    assert dict(result.mapping) == {}
    assert result.detections == ()


def test_empty_text_redacts_to_empty_text() -> None:
    result = redact("")

    assert result.text == ""
    assert dict(result.mapping) == {}


PROPERTY_POOL: tuple[str, ...] = (
    EMAIL_A,
    EMAIL_B,
    CARD_A,
    IBAN_A,
    PHONE_A,
    IPV4_A,
    "2001:db8::1",
    "1234567T",
)


@given(values=st.lists(st.sampled_from(PROPERTY_POOL), min_size=1, max_size=6))
def test_placeholder_assignment_is_a_bijection_over_distinct_values(values: list[str]) -> None:
    """Catches a counter that resets mid-pass or a map keyed on something lossy.

    Property: across any sequence of these values, one raw value has exactly one
    placeholder and one placeholder has exactly one raw value. DOB is out of the
    pool deliberately - it needs a cue in the surrounding text, so a generated
    sequence would not detect it and the property would be measuring the cue
    window rather than the assignment.
    """
    text = " and ".join(values) + "."

    result = redact(text)
    tokens = placeholders_in_order(text, result)

    by_value: dict[str, set[str]] = {}
    for found, token in zip(result.detections, tokens, strict=True):
        by_value.setdefault(found.value, set()).add(token)
    assert all(len(tokens_for_value) == 1 for tokens_for_value in by_value.values())
    assert len({next(iter(t)) for t in by_value.values()}) == len(by_value)


@given(values=st.lists(st.sampled_from(PROPERTY_POOL), min_size=1, max_size=6))
def test_every_placeholder_in_the_output_round_trips_through_the_map(values: list[str]) -> None:
    """Catches a map missing an entry for a placeholder that reached the output,
    which is unrehydratable and would surface only at rehydration."""
    text = " and ".join(values) + "."

    result = redact(text)

    for token in placeholders_in_order(text, result):
        assert token in result.mapping
        assert result.mapping[token] in text


# --------------------------------------------------------------------------
# the input cap
# --------------------------------------------------------------------------


@dataclass
class DetectRecorder:
    """A stand-in for `detect_all` that records whether it ran."""

    calls: list[str] = field(default_factory=list)

    def __call__(self, text: str, directory: NameDirectory | None = None) -> list[Detection]:
        """The arity `detect_all` has carried so far.

        `redactor` calls it as `detect_all(text, directory)`, so a stub taking `text`
        alone raises `TypeError` at the seam instead of recording the one call this
        case is here to see.
        """
        self.calls.append(text)
        return []


def test_max_redaction_input_chars_is_16384() -> None:
    """Catches a drift between the engine's cap and the request validator,
    which imports this same constant (exact literal)."""
    assert MAX_REDACTION_INPUT_CHARS == 16384


def test_input_of_exactly_the_cap_is_accepted_and_fully_scanned() -> None:
    """Catches an exclusive boundary (`>=` where `>` belongs) at 16384.

    The input is real prose repeated to the cap rather than one repeated
    character, so this row also runs the largest accepted input through every
    pattern: a detector that backtracks catastrophically hangs here rather than
    in production.
    """
    unit = f"Mail {EMAIL_A} or ring {PHONE_A} from {IPV4_A}. "
    text = (unit * (MAX_REDACTION_INPUT_CHARS // len(unit) + 1))[:MAX_REDACTION_INPUT_CHARS]
    assert len(text) == 16384

    result = redact(text)

    assert len(result.detections) > 0


def test_input_one_char_over_the_cap_raises_the_typed_error() -> None:
    """Catches a missing cap entirely, and one applied at the wrong number: 16385
    is the smallest rejected length."""
    text = "a" * (MAX_REDACTION_INPUT_CHARS + 1)

    with pytest.raises(RedactionInputTooLong) as excinfo:
        redact(text)

    assert excinfo.value.length == 16385
    assert excinfo.value.limit == 16384


def test_the_cap_is_checked_before_any_detector_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a cap checked after the scan, which is the ordering the acceptance
    criterion names: the CPU is spent before the request is refused, so an
    oversized body is still a way to burn gateway time."""
    from app.redaction import redactor

    recorder = DetectRecorder()
    monkeypatch.setattr(redactor, "detect_all", recorder)

    with pytest.raises(RedactionInputTooLong):
        redact("a" * (MAX_REDACTION_INPUT_CHARS + 1))

    assert recorder.calls == []


def test_the_detect_all_seam_the_ordering_test_patches_is_the_real_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the ordering test above going vacuous.

    If `redact` reached `detect_all` by any route the monkeypatch does not
    cover, "was not called" would pass for the wrong reason. Here the same
    patch, on an ACCEPTED input, must be called exactly once.
    """
    from app.redaction import redactor

    recorder = DetectRecorder()
    monkeypatch.setattr(redactor, "detect_all", recorder)

    result = redact(f"Mail {EMAIL_A}.")

    assert recorder.calls == [f"Mail {EMAIL_A}."]
    assert result.detections == ()


# --------------------------------------------------------------------------
# input that already contains a placeholder
# --------------------------------------------------------------------------

# This product hands redacted text back to people: the console shows it and a
# model can echo it. So a caller pasting `<EMAIL_1>` back in is an ordinary
# input, not an attack, and reusing that number would make the map stop being an
# inverse - rehydration would inject an address where nobody wrote one.


def test_a_placeholder_already_in_the_input_is_not_reused() -> None:
    """Catches a counter that starts at 1 regardless of what the input contains.

    Without the skip, the address becomes a second `<EMAIL_1>` and the two are
    indistinguishable in the output.
    """
    text = f"Contact <EMAIL_1> or {EMAIL_A} about it."

    result = redact(text)

    assert result.text == "Contact <EMAIL_1> or <EMAIL_2> about it."
    assert dict(result.mapping) == {"<EMAIL_2>": EMAIL_A}


def test_rehydration_is_exact_when_the_input_already_holds_a_placeholder() -> None:
    """Catches the collision at the level the claim is made at.

    `RedactionResult`'s docstring promises the map is a total inverse. Substituting
    every entry back has to reproduce the input character for character, with the
    caller's own `<EMAIL_1>` still sitting where they wrote it.
    """
    text = f"Was <EMAIL_1>, now {EMAIL_A}, cc {EMAIL_B}."

    result = redact(text)
    restored = result.text
    for placeholder, original in result.mapping.items():
        restored = restored.replace(placeholder, original)

    assert restored == text


def test_a_run_of_taken_numbers_is_skipped_not_merely_incremented_once() -> None:
    """Catches a skip written as a single `if` instead of a loop, which lands on
    the second taken number as soon as the input holds two."""
    text = f"Seen <EMAIL_1> and <EMAIL_2>; new one {EMAIL_A}."

    result = redact(text)

    assert dict(result.mapping) == {"<EMAIL_3>": EMAIL_A}


def test_a_taken_number_of_another_kind_does_not_shift_this_kind() -> None:
    """Catches a taken-token set collapsed to bare numbers, which would push every
    kind's counter forward because one kind's token was present."""
    text = f"Seen <CARD_1>; mail {EMAIL_A}."

    result = redact(text)

    assert dict(result.mapping) == {"<EMAIL_1>": EMAIL_A}


def test_a_placeholder_shaped_token_of_an_unknown_kind_is_ignored() -> None:
    """Catches a taken-token pattern written as `<[A-Z]+_[0-9]+>` rather than over
    the ten kinds: an unrelated template token would then shift our numbering."""
    text = f"Template <FIELD_1> here, mail {EMAIL_A}."

    result = redact(text)

    assert dict(result.mapping) == {"<EMAIL_1>": EMAIL_A}
