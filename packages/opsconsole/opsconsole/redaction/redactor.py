"""Overlap resolution, placeholder assignment and the input cap.

Three stages, no I/O: `detect_all` yields raw candidates, `resolve_overlaps`
turns them into one ordered disjoint list, and `redact` walks that list once
building both the output string and the placeholder map. Nothing here touches
HTTP, a store or the routing policy; an earlier change adds those around it.

The overlap rule is fixed and total, because a partial rule is how a nested or
truncated placeholder gets emitted: the longer span wins, an equal-length tie
falls to `KIND_PRECEDENCE`, and a tie there falls to the earlier start. Held by
the five rows from `test_longer_span_beats_shorter_overlapping_span` to
`test_same_kind_equal_length_overlap_falls_to_the_earlier_start`.

`detect_all` is imported into this module's namespace and called by that name,
which is the seam `test_the_cap_is_checked_before_any_detector_runs`
monkeypatches. `test_the_detect_all_seam_the_ordering_test_patches_is_the_real_one`
is the paired positive control: the same patch, on an accepted input, must be
called exactly once, so "was not called" cannot pass because the patch missed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from opsconsole.redaction.detectors import KIND_PRECEDENCE, Detection, Kind, detect_all
from opsconsole.redaction.errors import RedactionInputTooLong
from opsconsole.redaction.names import NameDirectory

# Twice an earlier change's per-input embedding cap. An earlier change's request model imports this
# constant rather than restating it, so a body and the engine cannot disagree
# about what is too long. Pinned exactly by `test_max_redaction_input_chars_is_16384`
# and bracketed by the 16384/16385 pair.
MAX_REDACTION_INPUT_CHARS: Final[int] = 16384

_PRECEDENCE_INDEX: Final[dict[Kind, int]] = {kind: i for i, kind in enumerate(KIND_PRECEDENCE)}

# `<KIND_N>` written literally in the INPUT. The input is client-supplied text and
# this product hands redacted text back to people, so a caller pasting our own
# output in is an ordinary case rather than an attack. Any number already spoken
# for is skipped when allocating, or the map would stop being an inverse: see
# `taken_placeholders` and `test_a_placeholder_already_in_the_input_is_not_reused`.
_LITERAL_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(
    r"<(?:" + "|".join(kind.value for kind in Kind) + r")_[0-9]+>"
)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """What one `redact` call produced.

    `detections` carries offsets into the ORIGINAL text, never into `text`: the
    output is shorter or longer at every substitution, so rebasing them would
    make everything after the first placeholder wrong
    (`test_detection_offsets_index_the_original_text_not_the_redacted_text`).

    `mapping` holds exactly the placeholders present in `text`, keyed to the raw
    original value, so an earlier change can store it without re-scanning and #20 can
    rehydrate from it by plain substitution. That inverse is total: a `<KIND_N>`
    token the INPUT already contained is never handed out again, so substituting
    every map entry back reproduces the original exactly, pre-existing tokens
    included (`test_rehydration_is_exact_when_the_input_already_holds_a_placeholder`).
    """

    text: str
    mapping: Mapping[str, str]
    detections: tuple[Detection, ...]


def resolve_overlaps(candidates: Iterable[Detection]) -> list[Detection]:
    """Reduce overlapping candidates to one disjoint list in start order.

    Greedy over the candidates sorted by the fixed rule, so the outcome does not
    depend on the order the detectors happened to run in. Spans that merely
    touch (`[0,10)` and `[10,20)`) do not overlap and both survive.
    """
    ranked = sorted(
        candidates,
        key=lambda d: (-(d.end - d.start), _PRECEDENCE_INDEX[d.kind], d.start),
    )
    accepted: list[Detection] = []
    for candidate in ranked:
        if any(candidate.start < kept.end and kept.start < candidate.end for kept in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda d: d.start)


def taken_placeholders(text: str) -> set[str]:
    """Every `<KIND_N>` token written literally in `text`.

    Scanned over the whole input, detections included: a value that happens to
    look like a placeholder is still a token a reader would confuse with ours.

    Public because `redact` is per-string and an earlier change redacts a whole request
    under ONE numbering: `gateway.routes.redact.redact_batch` has to skip a token
    literal in ANY of the call's strings, not only in the one being written, and
    a second copy of `_LITERAL_PLACEHOLDER` there is how the two definitions
    would drift. It reads `text` and nothing else, so exporting it changes no
    behaviour of the engine.
    """
    return {match.group(0) for match in _LITERAL_PLACEHOLDER.finditer(text)}


def redact(text: str, directory: NameDirectory | None = None) -> RedactionResult:
    """Replace every detected value in `text` with a stable `<KIND_N>` placeholder.

    Raises `RedactionInputTooLong` when `text` is longer than
    `MAX_REDACTION_INPUT_CHARS`, before any detector runs.

    `directory` is the tenant's compiled display names, passed straight through to
    `detect_all`. It defaults to `None`, which is no `PERSON` candidate at all, so
    a caller that passes none gets exactly what it got before an earlier change. The
    matcher is compiled by the ROUTE and handed in rather than built here
    (decision E): `redact` is per string and `redact_batch` calls it up to 64
    times for one request.

    Placeholders are keyed on `(kind, value)` and numbered per kind from 1 in
    first-appearance order, so the same raw value twice in one call gets one
    placeholder and two different values never share one. The key is the RAW
    matched text: two spellings of the same account number are two values, which
    is what keeps `mapping` able to reconstruct the input exactly.

    A number whose token already appears literally in `text` is skipped, so the
    first address in `"see <EMAIL_1> and alice@example.com"` becomes `<EMAIL_2>`
    and rehydration cannot inject a value where the caller never wrote one.
    """
    if len(text) > MAX_REDACTION_INPUT_CHARS:
        raise RedactionInputTooLong(len(text), MAX_REDACTION_INPUT_CHARS)

    detections = tuple(resolve_overlaps(detect_all(text, directory)))
    taken = taken_placeholders(text)
    mapping: dict[str, str] = {}
    assigned: dict[tuple[Kind, str], str] = {}
    counters: dict[Kind, int] = {}

    pieces: list[str] = []
    cursor = 0
    for found in detections:
        key = (found.kind, found.value)
        placeholder = assigned.get(key)
        if placeholder is None:
            number = counters.get(found.kind, 0) + 1
            while f"<{found.kind.value}_{number}>" in taken:
                number += 1
            counters[found.kind] = number
            placeholder = f"<{found.kind.value}_{number}>"
            assigned[key] = placeholder
            mapping[placeholder] = found.value
        pieces.append(text[cursor : found.start])
        pieces.append(placeholder)
        cursor = found.end
    pieces.append(text[cursor:])

    return RedactionResult(text="".join(pieces), mapping=mapping, detections=detections)
