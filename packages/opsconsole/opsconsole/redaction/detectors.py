r"""The nine detector kinds, their checksums and the DOB cue window.

Each kind is one compiled pattern run with `finditer` over free text, plus an
optional post-check the pattern itself cannot express: Luhn for `CARD`, mod-97
for `IBAN`, ISO 7064 MOD 11,10 for `STEUER_ID`, mod-23 for `PPSN`, the octet and
group ranges (via `ipaddress`) for `IPV4` and `IPV6`, and the cue window for
`DOB`. That split is an earlier change decision A: a single alternation regex cannot
carry a checksum, and a kind label is only truthful if the thing it labels
passed that kind's own check.

a project rule has two targets here. Run from
`packages/opsconsole/opsconsole/redaction/`, `grep -E '^    if not re\.'
detectors.py` prints exactly this and nothing else, and `grep 're\.fullmatch('
*.py` over the package prints nothing at all:

        if not re.match(r"\A[A-Z]{2}[0-9]{2}[0-9A-Z]+\Z", value):
        if not re.match(r"\A[0-9]{7}[A-W][A-W]?\Z", text):

No `-n`, and the paste is indented by eight spaces rather than four, so neither
command can match its own citation and neither goes stale when a line is added
above it. Both hits validate a whole client-supplied value.

Both validate a whole client-supplied value, and both are public: the fixture
integrity test calls them on whatever a case claims is an IBAN or a PPSN. So
both are anchored with `\Z`, never `$`, because in Python `$` also matches
before a trailing newline - `iban_ok("DE12...\n")` would otherwise be true, as
`str.replace` does not strip the newline and `int()` tolerates it.
`test_iban_rejects_trailing_whitespace` and `test_ppsn_rejects_trailing_whitespace`
carry the rule's required table (a trailing newline, a trailing CRLF, a trailing
space) for each.

The nine PATTERNS are the other half and have no `fullmatch` at all: every one
searches free text with explicit boundary lookarounds, so the rule's analogue is
what applies to them, pinned by
`test_value_at_end_of_input_is_detected_without_the_trailing_whitespace` - a
value at the very end of the input, with a trailing newline, CRLF or space, is
still detected and its span stops before the whitespace.

`detect_all` returns RAW candidates. They may overlap - the digits inside an
IBAN can look like a card number - and resolving that is
`opsconsole.redaction.redactor.resolve_overlaps`, not this module's job. The
boundary lookarounds are still written to keep raw overlaps rare, because
`test_precision_meets_the_floor` scores `detect_all` and every spurious
candidate is a false positive whether or not the merge would later drop it.

What a detector owes, for whoever adds the tenth kind in an earlier change: a span whose
`value` is `text[start:end]` exactly, boundaries that stop it firing inside a
longer alphanumeric run, and a fixture case in `tests/redaction/fixtures/cases.json`
for each of its accepted forms. `test_fixture_covers_every_detector_kind` fails
on a kind with no ground truth, but nothing forces the negative case that would
show the new pattern eating an ordinary number.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final


class Kind(StrEnum):
    """The nine detector labels. Each is the `KIND` half of a `<KIND_N>` placeholder.

    Wire-visible: an earlier change stores the map under these strings, so a rename is a
    contract change rather than an implementation detail. The member list is held
    against a literal by `test_kind_members_are_the_nine_named_in_acceptance`.
    """

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    IBAN = "IBAN"
    CARD = "CARD"
    STEUER_ID = "STEUER_ID"
    PPSN = "PPSN"
    IPV4 = "IPV4"
    IPV6 = "IPV6"
    DOB = "DOB"


# Tie-break for two overlapping candidates of the SAME length; longer always wins
# first. Ordered by how specific the kind's own check is: a value that satisfies
# both a checksum and a shape is far likelier to be the checksummed kind. Held
# complete by `test_precedence_lists_every_kind_exactly_once`; the tie-break
# behaviour is pinned by `test_equal_length_overlap_falls_to_the_precedence_order`.
KIND_PRECEDENCE: Final[tuple[Kind, ...]] = (
    Kind.EMAIL,
    Kind.IBAN,
    Kind.CARD,
    Kind.IPV6,
    Kind.IPV4,
    Kind.PHONE,
    Kind.STEUER_ID,
    Kind.PPSN,
    Kind.DOB,
)

# A date counts as a birth date only when one of these appears before it, ending
# within `DOB_CUE_DISTANCE` characters of where the date starts. Matched
# case-insensitively on word boundaries. Both values are pinned exactly by
# `test_dob_cues_are_the_three_named_in_acceptance` and
# `test_dob_cue_distance_is_thirty_two`.
DOB_CUES: Final[tuple[str, ...]] = ("born", "DOB", "date of birth")
DOB_CUE_DISTANCE: Final[int] = 32


@dataclass(frozen=True, slots=True)
class Detection:
    """One candidate span. `value` is always `text[start:end]` of the scanned text.

    Frozen so the merge step cannot rewrite a candidate in place and leave the
    raw list disagreeing with the resolved one
    (`test_detection_is_frozen`); the offset invariant is
    `test_detect_all_values_index_their_own_offsets`.
    """

    kind: Kind
    start: int
    end: int
    value: str


# --------------------------------------------------------------------------
# checksums
# --------------------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    """The Luhn check digit over a run of digits.

    `test_luhn_accepts_valid_numbers` and `test_luhn_rejects_invalid_numbers`
    hold both directions, which is what stops a `return True` stub passing every
    `CARD` row.
    """
    if not digits.isdigit():
        return False
    total = 0
    for position, character in enumerate(reversed(digits)):
        value = int(character)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_ok(compact: str) -> bool:
    r"""ISO 13616 mod-97: rotate the first four characters to the end, map letters
    to two-digit numbers from A=10, and require a remainder of 1.

    Length is checked here too (15 to 34 characters), because the pattern's
    bounds and the standard's bounds are two different statements. Pinned by
    `test_iban_mod97_accepts_valid` and `test_iban_mod97_rejects_bad_check_digits`,
    and at both ends by `test_iban_length_bounds_reach_both_ends_of_the_standard`.

    The argument is COMPACT: no separators of any kind are stripped here. An
    earlier version stripped spaces so a grouped IBAN would validate, which also
    made a trailing space invisible and defeated the `\Z` anchor below - the
    thing a project rule exists to stop. `IBAN_PATTERN` never yields a spaced
    value, so the flexibility bought nothing;
    `test_iban_rejects_trailing_whitespace` holds the corrected behaviour.
    """
    value = compact.upper()
    if not 15 <= len(value) <= 34:
        return False
    if not re.match(r"\A[A-Z]{2}[0-9]{2}[0-9A-Z]+\Z", value):
        return False
    rotated = value[4:] + value[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rotated)
    return int(digits) % 97 == 1


def steuer_id_ok(digits: str) -> bool:
    """The German tax identification number: 11 digits, ISO 7064 MOD 11,10.

    Two rules beyond the checksum, both of them real rejections rather than
    shape: the number is exactly 11 digits and its first digit is never 0.
    `test_steuer_id_rejects` carries a specimen with a CORRECT check digit and a
    leading zero, so only the leading-zero rule can reject that row.

    The standard's other rule - that exactly one digit repeats within the first
    ten - is NOT implemented here. It would tighten precision further; it is not
    claimed and no test asserts it.
    """
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    product = 10
    for character in digits[:10]:
        total = (int(character) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    check = (11 - product) % 10
    return check == int(digits[10])


# Index 0 is `W`, so a remainder of 0 yields `W`. Used for the check character and
# for the optional second letter's value.
PPSN_ALPHABET: Final[str] = "WABCDEFGHIJKLMNOPQRSTUV"


def ppsn_ok(value: str) -> bool:
    """The Irish PPS number: seven digits, a check character, an optional second letter.

    Weights 8..2 run over the seven digits; the second letter, when present,
    contributes its position in `PPSN_ALPHABET` times 9 (so a trailing `W`
    contributes nothing). The check character is `PPSN_ALPHABET[total % 23]`.
    Pinned by `test_ppsn_accepts_valid` and `test_ppsn_rejects`, whose rows
    include a correct shape with a wrong check character.
    """
    text = value.upper()
    if not re.match(r"\A[0-9]{7}[A-W][A-W]?\Z", text):
        return False
    weighted = zip(text[:7], range(8, 1, -1), strict=True)
    total = sum(int(character) * weight for character, weight in weighted)
    if len(text) == 9:
        total += PPSN_ALPHABET.index(text[8]) * 9
    return PPSN_ALPHABET[total % 23] == text[7]


# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------

# Boundaries are alphanumeric on both sides for every kind whose value is a bare
# run of digits or letters, so a candidate cannot start inside a longer token -
# the digits of `IE29AIBK93115212345678` are not a card number, and blocking them
# here is what keeps them out of the precision score.
EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z._%+\-])"
    r"[0-9A-Za-z._%+\-]+@[0-9A-Za-z\-]+(?:\.[0-9A-Za-z\-]+)*\.[A-Za-z]{2,}"
    r"(?![0-9A-Za-z\-])"
)

# E.164 and the spaced forms of it EU numbers are usually written in: a `+`, then
# 7 to 15 digits with at most one space or hyphen between any two.
PHONE_INTERNATIONAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z+])\+[0-9](?:[ \-]?[0-9]){6,14}(?![0-9])"
)
# The national trunk form: a leading 0, then 8 to 13 more digits. The upper bound
# is what stops a 20-digit batch reference being read as a phone number.
PHONE_NATIONAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z+])0[0-9](?:[ \-]?[0-9]){7,12}(?![0-9])"
)

# Single spaces between groups are allowed because that is how every bank prints
# an IBAN. Without them `DE12 5001 0517 0648 4898 90` is not an IBAN to this
# engine, and `CARD_PATTERN` - which does allow separators - picks up the digit
# tail instead: the country and check digits leak AND the value is stored under
# the wrong kind. `test_a_space_grouped_iban_is_one_detection_not_a_card` holds
# both halves. The separators are stripped before the checksum, never from the
# span, so `mapping` still reconstructs what the caller wrote.
IBAN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z])[A-Z]{2}[0-9]{2}(?:[ ]?[0-9A-Z]){11,30}(?![0-9A-Za-z])"
)

# `+` is excluded on the left so the digits of `+4930123456789` are never a card
# candidate: a 13-digit run that happens to satisfy Luhn would otherwise be a
# false positive on every international phone number.
CARD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z+\-])[0-9](?:[ \-]?[0-9]){12,18}(?![0-9A-Za-z])"
)

STEUER_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<![0-9A-Za-z])[0-9]{11}(?![0-9A-Za-z])")

# Lower case is accepted because `ppsn_ok` upper-cases before checking: an
# uppercase-only pattern would make that normalisation unreachable and a PPS
# number typed in lower case a silent miss. `test_a_lowercase_ppsn_is_detected`.
PPSN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z])[0-9]{7}[A-Wa-w][A-Wa-w]?(?![0-9A-Za-z])"
)

# The trailing `(?!\.[0-9])` is why a sentence-final `192.0.2.1.` is a detection
# while `1.2.3.4.5` is not: a dot only disqualifies the match when a digit
# follows it.
IPV4_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z.])[0-9]{1,3}(?:\.[0-9]{1,3}){3}(?![0-9A-Za-z])(?!\.[0-9])"
)

# At least two colon-separated groups after the first, so a time of day (`14:32`)
# cannot match. The boundaries are ALPHANUMERIC, not hex: `(?<![0-9A-Fa-f:])`
# would let `std::vector` start a match at the `d`, because `d` is a hex digit
# preceded by a non-hex letter. `ipaddress` accepts `d::`, so nothing downstream
# would have caught it. Pinned by `test_scope_resolution_operator_is_not_an_ipv6`.
# The trailing quad is all-or-nothing (exactly three dotted parts, or none), so
# `::ffff:192.0.2.1` is one detection rather than a hex prefix with three octets
# left in the clear, while `2001:db8::1.` at the end of a sentence still matches
# up to the `1`. Pinned by `test_an_ipv4_mapped_ipv6_is_detected_whole`.
IPV6_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-z:])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}"
    r"(?:(?:\.[0-9]{1,3}){3})?(?![0-9A-Za-z:])"
)
# The second half of the same guard: a candidate must carry at least this many hex
# characters. `a::b` parses as a valid IPv6 and is far likelier to be prose or code
# than an address. The cost is that `::1` and other very short forms are not
# detected; that is deliberate and is why the number is pinned by
# `test_ipv6_minimum_hex_characters_is_four`.
IPV6_MIN_HEX_CHARS: Final[int] = 4

# Years are four digits in all three forms. A two-digit year would match an
# invoice total or a version string far more often than it would match a birth
# date, and the cue window is not enough on its own to pay for that.
DOB_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?<![0-9])(?P<y>[0-9]{4})-(?P<m>[0-9]{1,2})-(?P<d>[0-9]{1,2})(?![0-9])"),
    re.compile(r"(?<![0-9])(?P<d>[0-9]{1,2})/(?P<m>[0-9]{1,2})/(?P<y>[0-9]{4})(?![0-9])"),
    re.compile(r"(?<![0-9])(?P<d>[0-9]{1,2})\.(?P<m>[0-9]{1,2})\.(?P<y>[0-9]{4})(?![0-9])"),
)

CUE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:" + "|".join(re.escape(cue) for cue in DOB_CUES) + r")\b", re.IGNORECASE
)


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------


def compact_separators(value: str) -> str:
    """Strip the spaces and hyphens a card number or an IBAN may be printed with.

    Public because the checksums take a compact value and the fixture integrity
    test has to normalise the same way the detectors do. Two definitions of
    "compact" would let a fixture row pass a check the detector never applies.
    """
    return value.replace(" ", "").replace("-", "")


def _is_a_date(year: str, month: str, day: str) -> bool:
    """True when the three parts name a real calendar day.

    Day and month are tried both ways round for the slashed and dotted forms,
    because `12/04/1985` is a birth date under either reading and a redaction
    engine that guessed the locale would lose half of them.
    """
    for first, second in ((day, month), (month, day)):
        try:
            date(int(year), int(second), int(first))
        except ValueError:
            continue
        return True
    return False


def _cue_ends_before(text: str, start: int) -> bool:
    """True when a cue ends within `DOB_CUE_DISTANCE` characters before `start`.

    Direction matters as much as distance: a cue that ends at or after the date
    does not qualify it, which is what
    `test_cue_after_the_date_does_not_qualify_it` pins.
    """
    window_start = max(0, start - DOB_CUE_DISTANCE - max(len(cue) for cue in DOB_CUES))
    for match in CUE_PATTERN.finditer(text, window_start, start):
        if match.end() <= start and start - match.end() <= DOB_CUE_DISTANCE:
            return True
    return False


def _simple(
    text: str, pattern: re.Pattern[str], kind: Kind
) -> Iterator[tuple[Kind, int, int, str]]:
    """Every match of `pattern`, with no post-check. Used where the pattern IS the rule."""
    for match in pattern.finditer(text):
        yield kind, match.start(), match.end(), match.group(0)


def _checked(
    text: str,
    pattern: re.Pattern[str],
    kind: Kind,
    check: Callable[[str], bool],
    *,
    compact: bool = False,
) -> Iterator[tuple[Kind, int, int, str]]:
    """Every match that also passes `check`.

    `compact=True` strips spaces and hyphens before the check but yields the RAW
    matched text, so the span and the placeholder map hold what the input
    actually contained rather than a normalised form
    (`test_mapping_round_trips_the_redacted_text_back_to_the_original`).
    """
    for match in pattern.finditer(text):
        value = match.group(0)
        candidate = compact_separators(value) if compact else value
        if check(candidate):
            yield kind, match.start(), match.end(), value


def _ipv6_ok(value: str) -> bool:
    """A candidate is an IPv6 address only if it parses AND carries enough hex.

    The parse alone is not a guard: `ipaddress.IPv6Address` accepts `d::`, which
    is what `std::vector` reduces to once a hex-only lookbehind lets the match
    start mid-word. See `IPV6_MIN_HEX_CHARS`.
    """
    if sum(1 for character in value if character not in ":.") < IPV6_MIN_HEX_CHARS:
        return False
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    return True


def _parses_as(factory: Callable[[str], object]) -> Callable[[str], bool]:
    """Turn an `ipaddress` constructor into a post-check.

    The pattern only fixes the shape; the octet range for `IPV4` and the group
    count and hex range for `IPV6` are the constructor's job, which is why
    `999.1.1.1` is not a detection.
    """

    def check(value: str) -> bool:
        try:
            factory(value)
        except ValueError:
            return False
        return True

    return check


def _dates(text: str) -> Iterator[tuple[Kind, int, int, str]]:
    for pattern in DOB_PATTERNS:
        for match in pattern.finditer(text):
            if not _is_a_date(match.group("y"), match.group("m"), match.group("d")):
                continue
            if not _cue_ends_before(text, match.start()):
                continue
            yield Kind.DOB, match.start(), match.end(), match.group(0)


def detect_all(text: str) -> list[Detection]:
    """Every candidate span in `text`, in start order, overlaps included.

    Deduplicated on `(kind, start, end)` so two date patterns matching the same
    run cannot yield the span twice. Ordering is pinned by
    `test_detect_all_returns_candidates_in_start_order`; resolving the overlaps
    is `resolve_overlaps` in `opsconsole.redaction.redactor`.
    """
    found: list[tuple[Kind, int, int, str]] = []
    found.extend(_simple(text, EMAIL_PATTERN, Kind.EMAIL))
    found.extend(_simple(text, PHONE_INTERNATIONAL_PATTERN, Kind.PHONE))
    found.extend(_simple(text, PHONE_NATIONAL_PATTERN, Kind.PHONE))
    found.extend(_checked(text, IBAN_PATTERN, Kind.IBAN, iban_ok, compact=True))
    found.extend(_checked(text, CARD_PATTERN, Kind.CARD, luhn_ok, compact=True))
    found.extend(_checked(text, STEUER_ID_PATTERN, Kind.STEUER_ID, steuer_id_ok))
    found.extend(_checked(text, PPSN_PATTERN, Kind.PPSN, ppsn_ok))
    found.extend(_checked(text, IPV4_PATTERN, Kind.IPV4, _parses_as(ipaddress.IPv4Address)))
    found.extend(_checked(text, IPV6_PATTERN, Kind.IPV6, _ipv6_ok))
    found.extend(_dates(text))

    unique = {(kind, start, end): value for kind, start, end, value in found}
    ordered = sorted(unique, key=lambda key: (key[1], key[2], KIND_PRECEDENCE.index(key[0])))
    return [
        Detection(kind=kind, start=start, end=end, value=unique[(kind, start, end)])
        for kind, start, end in ordered
    ]
