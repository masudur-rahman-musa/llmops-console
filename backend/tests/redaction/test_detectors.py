r"""The ten detector kinds, their checksums and the DOB cue window.

Every accepted row asserts the exact `(kind, start, end)` triple rather than
"something was found", because a detector that swallows the surrounding space or
mislabels the kind still satisfies a truthiness check while breaking both the
placeholder map and the precision score.

Rejection rows name one kind each (`no CARD covers this run`) instead of "no
detections at all". Asserting emptiness would let a neighbouring detector's
guard carry the row: a 16-digit run that fails Luhn is also not a PHONE, so an
implementation with no Luhn check at all would still pass an emptiness
assertion if PHONE happened to reject it first.

Strict whole-string validation has no direct target here - no pattern in this package validates
a whole client-supplied string with `fullmatch`, every one runs `finditer` over
free text - so the contract's stated analogue is asserted instead: a value
followed by a trailing newline, a trailing CRLF or a trailing space is still
detected, and the span stops before the whitespace.
"""

from __future__ import annotations

import pytest

from app.redaction import DOB_CUE_DISTANCE, KIND_PRECEDENCE, Kind, redact
from app.redaction.detectors import (
    DOB_CUES,
    IPV6_MIN_HEX_CHARS,
    Detection,
    detect_all,
    iban_ok,
    luhn_ok,
    ppsn_ok,
    steuer_id_ok,
)
from app.redaction.names import NameDirectory

# Synthetic specimens. Cards pass Luhn, IBANs mod-97, Steuer-IDs ISO 7064
# MOD 11,10 and PPSNs mod-23; every one was generated and verified rather than
# copied. Addresses use `example.*`, IPv4 the RFC 5737 documentation ranges and
# IPv6 the RFC 3849 `2001:db8::/32` block, so nothing here is a real identifier.
VALID_CARDS = ("4111111111111111", "4242424242424242", "5555555555554444", "378282246310005")
NON_LUHN_16 = "4111111111111112"
VALID_IBANS = (
    "DE12500105170648489890",
    "IE29AIBK93115212345678",
    "FR1420041010050500013M02606",
    "NL91ABNA0417164300",
)
BAD_MOD97_IBAN = "DE13500105170648489890"
VALID_STEUER_IDS = ("26869543017", "12345678903", "86523996324")
BAD_CHECK_STEUER_ID = "26869543018"
# Correct MOD 11,10 check digit, rejected only by the "first digit is not 0" rule.
LEADING_ZERO_STEUER_ID = "06869543015"
VALID_PPSNS = ("1234567T", "7654321PA", "2345678IW")
BAD_CHECK_PPSN = "1234567X"


def spans_of(text: str, kind: Kind) -> list[tuple[int, int]]:
    return [(d.start, d.end) for d in detect_all(text) if d.kind == kind]


def test_kind_members_are_the_ten_named_in_acceptance() -> None:
    """Catches a detector set that drops a kind or renames one.

    The ten labels are the wire-visible half of the placeholder `<KIND_N>`, so
    a rename is a silent contract break for the stored map. `PERSON` is
    the tenth and is asserted in its position, not merely as a member:
    the list is compared by equality, so a kind inserted in the middle fails here
    as well.
    """
    assert [kind.value for kind in Kind] == [
        "EMAIL",
        "PHONE",
        "IBAN",
        "CARD",
        "STEUER_ID",
        "PPSN",
        "IPV4",
        "IPV6",
        "DOB",
        "PERSON",
    ]


def test_precedence_lists_every_kind_exactly_once() -> None:
    """Catches a precedence tuple that omits a kind, so equal-length ties involving
    it fall through to an undefined branch."""
    assert len(KIND_PRECEDENCE) == 10
    assert set(KIND_PRECEDENCE) == set(Kind)


def test_person_is_last_in_the_precedence_order() -> None:
    """Catches `PERSON` inserted anywhere but the end.

    Precedence decides EQUAL-length ties only, and every other kind carries a
    check of its own - a Luhn, a mod-97, a parsed address - while a directory name
    is membership and nothing more. Put first, a tenant with a staff member called
    `Mai` would relabel a date span `PERSON` and the map would name the wrong
    class for a value that is not a name at all. `set(KIND_PRECEDENCE)` above
    cannot see the position, which is why this is its own row.
    """
    assert KIND_PRECEDENCE[-1] is Kind.PERSON


def test_dob_cue_distance_is_thirty_two() -> None:
    """Catches a widened or narrowed cue window (exact literal)."""
    assert DOB_CUE_DISTANCE == 32


def test_dob_cues_are_the_three_named_in_acceptance() -> None:
    """Catches a cue list that adds a cue (recall bought by inventing a trigger)
    or drops one, which silently makes a whole date form undetectable."""
    assert DOB_CUES == ("born", "DOB", "date of birth")


# --------------------------------------------------------------------------
# checksums
# --------------------------------------------------------------------------


@pytest.mark.parametrize("digits", VALID_CARDS)
def test_luhn_accepts_valid_numbers(digits: str) -> None:
    """Catches a Luhn implementation that doubles the wrong alternating position
    (it rejects every genuine card while still rejecting the invalid one)."""
    assert luhn_ok(digits) is True


@pytest.mark.parametrize(
    "digits",
    [NON_LUHN_16, "1234567890123456", "0000000000000001", "4111111111111121"],
)
def test_luhn_rejects_invalid_numbers(digits: str) -> None:
    """Catches a `luhn_ok` stubbed to `return True`, which is the shortest way to
    make every CARD row pass."""
    assert luhn_ok(digits) is False


@pytest.mark.parametrize("compact", VALID_IBANS)
def test_iban_mod97_accepts_valid(compact: str) -> None:
    """Catches a mod-97 that forgets to rotate the first four characters to the end."""
    assert iban_ok(compact) is True


@pytest.mark.parametrize("compact", [BAD_MOD97_IBAN, "DE00500105170648489890"])
def test_iban_mod97_rejects_bad_check_digits(compact: str) -> None:
    """Catches a check comparing the remainder to 0 instead of 1, or no check at all."""
    assert iban_ok(compact) is False


@pytest.mark.parametrize("digits", VALID_STEUER_IDS)
def test_steuer_id_accepts_valid(digits: str) -> None:
    """Catches an ISO 7064 MOD 11,10 loop that omits the `s == 0 -> 10` step."""
    assert steuer_id_ok(digits) is True


@pytest.mark.parametrize(
    ("digits", "why"),
    [
        (BAD_CHECK_STEUER_ID, "last digit is wrong"),
        (LEADING_ZERO_STEUER_ID, "check digit is correct but the first digit is 0"),
        ("2686954301", "ten digits"),
        ("268695430170", "twelve digits"),
    ],
)
def test_steuer_id_rejects(digits: str, why: str) -> None:
    """Catches a Steuer-ID check missing the leading-zero rule or the length rule.

    The leading-zero row carries a CORRECT MOD 11,10 check digit, so only the
    `first != "0"` rule can reject it.
    """
    assert steuer_id_ok(digits) is False, why


@pytest.mark.parametrize("value", VALID_PPSNS)
def test_ppsn_accepts_valid(value: str) -> None:
    """Catches a mod-23 that ignores the optional second letter's weight of 9."""
    assert ppsn_ok(value) is True


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (BAD_CHECK_PPSN, "check letter should be T"),
        ("123456T", "six digits"),
        ("12345678T", "eight digits"),
        ("1234567", "no check letter"),
    ],
)
def test_ppsn_rejects(value: str, why: str) -> None:
    """Catches a PPSN check that validates the shape but never the check character."""
    assert ppsn_ok(value) is False, why


# --------------------------------------------------------------------------
# accepted values, exact spans
# --------------------------------------------------------------------------

ACCEPTED: tuple[tuple[str, Kind, str, str], ...] = (
    ("email-plain", Kind.EMAIL, "Contact ", "alice@example.com"),
    ("email-plus-tag", Kind.EMAIL, "Write to ", "ops.team+alerts@example.org"),
    ("email-multi-label-domain", Kind.EMAIL, "Mail ", "a.b@mail.example.co.uk"),
    ("phone-e164-de", Kind.PHONE, "Call ", "+4930123456789"),
    ("phone-e164-ie-spaced", Kind.PHONE, "Ring ", "+353 1 234 5678"),
    ("phone-e164-fr-spaced", Kind.PHONE, "Dial ", "+33 1 23 45 67 89"),
    ("phone-de-national", Kind.PHONE, "Desk ", "030 12345678"),
    ("iban-de", Kind.IBAN, "Pay ", VALID_IBANS[0]),
    ("iban-ie", Kind.IBAN, "Pay ", VALID_IBANS[1]),
    ("iban-fr-alphanumeric", Kind.IBAN, "Pay ", VALID_IBANS[2]),
    ("iban-nl", Kind.IBAN, "Pay ", VALID_IBANS[3]),
    ("card-visa", Kind.CARD, "Card ", VALID_CARDS[0]),
    ("card-mastercard", Kind.CARD, "Card ", VALID_CARDS[2]),
    ("card-amex-15", Kind.CARD, "Card ", VALID_CARDS[3]),
    ("steuer-id", Kind.STEUER_ID, "Steuer-ID ", VALID_STEUER_IDS[0]),
    ("steuer-id-alt", Kind.STEUER_ID, "Tax number ", VALID_STEUER_IDS[2]),
    ("ppsn-one-letter", Kind.PPSN, "PPSN ", VALID_PPSNS[0]),
    ("ppsn-two-letters", Kind.PPSN, "PPSN ", VALID_PPSNS[1]),
    ("ppsn-w-suffix", Kind.PPSN, "PPSN ", VALID_PPSNS[2]),
    ("ipv4-low", Kind.IPV4, "Host ", "192.0.2.1"),
    ("ipv4-mid", Kind.IPV4, "Host ", "198.51.100.23"),
    ("ipv4-max-octet", Kind.IPV4, "Host ", "203.0.113.255"),
    ("ipv6-compressed", Kind.IPV6, "Peer ", "2001:db8::1"),
    ("ipv6-mixed", Kind.IPV6, "Peer ", "2001:db8:85a3::8a2e:370:7334"),
    ("ipv6-full", Kind.IPV6, "Peer ", "2001:db8:0:0:0:0:0:42"),
    ("dob-iso", Kind.DOB, "Patient born ", "1985-04-12"),
    ("dob-slashes", Kind.DOB, "DOB: ", "12/04/1985"),
    ("dob-dotted", Kind.DOB, "Born ", "01.07.1978"),
    ("dob-phrase-cue", Kind.DOB, "Her date of birth is ", "1990-03-03"),
)


@pytest.mark.parametrize(
    ("kind", "prefix", "value"),
    [pytest.param(k, p, v, id=case_id) for case_id, k, p, v in ACCEPTED],
)
def test_accepted_value_is_detected_with_exact_span(kind: Kind, prefix: str, value: str) -> None:
    """Catches a pattern that matches but over-reaches - a trailing space, the
    cue word, or a leading `+` left outside - since the span, not the fact of a
    match, is what the placeholder replaces and the map keys on."""
    text = f"{prefix}{value} and nothing else."

    assert (len(prefix), len(prefix) + len(value)) in spans_of(text, kind)


@pytest.mark.parametrize(
    ("kind", "prefix", "value"),
    [pytest.param(k, p, v, id=case_id) for case_id, k, p, v in ACCEPTED],
)
@pytest.mark.parametrize("suffix", ["\n", "\r\n", " ", ""], ids=["lf", "crlf", "space", "eof"])
def test_value_at_end_of_input_is_detected_without_the_trailing_whitespace(
    kind: Kind, prefix: str, value: str, suffix: str
) -> None:
    r"""Catches the standing-rule-4 family: a pattern anchored with `$` rather than
    `\Z`, or one whose right boundary is `\s` and so eats the newline into the span.

    A value at the very end of the input, with nothing after it, is the shape a
    trailing-newline bug shows up in: the span either stops at `len(value)` or it
    does not.
    """
    text = f"{prefix}{value}{suffix}"

    assert (len(prefix), len(prefix) + len(value)) in spans_of(text, kind)


# --------------------------------------------------------------------------
# rejected look-alikes
# --------------------------------------------------------------------------

REJECTED: tuple[tuple[str, Kind, str], ...] = (
    ("luhn-failure-is-not-a-card", Kind.CARD, f"Reference {NON_LUHN_16} was rejected."),
    ("thirteen-digit-luhn-failure", Kind.CARD, "Reference 1234567890123 rejected."),
    ("twenty-digit-run-is-not-a-card", Kind.CARD, "Batch 00112233445566778899 finished."),
    ("bad-mod97-is-not-an-iban", Kind.IBAN, f"Account {BAD_MOD97_IBAN} bounced."),
    ("bad-check-digit-is-not-a-steuer-id", Kind.STEUER_ID, f"Tax {BAD_CHECK_STEUER_ID} filed."),
    ("leading-zero-is-not-a-steuer-id", Kind.STEUER_ID, f"Tax {LEADING_ZERO_STEUER_ID} filed."),
    ("bad-check-letter-is-not-a-ppsn", Kind.PPSN, f"PPSN {BAD_CHECK_PPSN} recorded."),
    ("semver-is-not-an-ipv4", Kind.IPV4, "Upgrade to version 3.10.2 before Friday."),
    ("octet-over-255-is-not-an-ipv4", Kind.IPV4, "Host 999.1.1.1 refused."),
    ("four-part-date-is-not-an-ipv4", Kind.IPV4, "Born 01.07.1978, retired in 2043."),
    ("time-of-day-is-not-an-ipv6", Kind.IPV6, "Standup moves to 14:32 tomorrow."),
    ("date-without-a-cue-is-not-a-dob", Kind.DOB, "The audit is scheduled for 1985-04-12."),
    ("order-id-is-not-a-phone", Kind.PHONE, "Order ORD-2024-88213 shipped yesterday."),
    ("bare-number-is-not-a-phone", Kind.PHONE, "The queue held 42 items."),
    ("email-without-a-domain-dot", Kind.EMAIL, "Ping alice@localhost now."),
)


@pytest.mark.parametrize(
    ("kind", "text"),
    [pytest.param(k, t, id=case_id) for case_id, k, t in REJECTED],
)
def test_look_alike_is_not_detected_as_that_kind(kind: Kind, text: str) -> None:
    """Catches a detector whose own post-check is missing.

    Each row is asserted against ONE kind, never against "no detections", so a
    neighbouring detector rejecting the same string cannot carry the row: the
    only thing that can flip `luhn-failure-is-not-a-card` is removing the Luhn
    call, and the only thing that can flip `octet-over-255-is-not-an-ipv4` is
    widening the octet range.
    """
    assert spans_of(text, kind) == []


def test_ipv4_look_alike_row_would_fail_without_its_own_guard() -> None:
    """Catches the rejection table going vacuous.

    `999.1.1.1` differs from `192.0.2.1` in one octet only, and the second IS a
    detection, so the row above cannot be passing because IPV4 never fires on
    dotted quads.
    """
    assert spans_of("Host 192.0.2.1 refused.", Kind.IPV4) == [(5, 14)]


# --------------------------------------------------------------------------
# DOB cue window
# --------------------------------------------------------------------------

DOB_DATE = "1985-04-12"


def cue_text(cue: str, gap: int) -> str:
    """`cue`, then `gap` filler characters, then the date.

    The cue ENDS `gap` characters before the date starts, which is the distance
    `DOB_CUE_DISTANCE` bounds. Filler is `.` repeated so no second cue, digit or
    date fragment can appear inside the window by accident.
    """
    return f"{cue}{'.' * gap}{DOB_DATE}"


@pytest.mark.parametrize("cue", DOB_CUES)
def test_date_at_exactly_the_cue_distance_is_a_dob(cue: str) -> None:
    """Catches an off-by-one that makes the window exclusive at its far edge.

    Paired with the row below: at gap 32 the date IS a detection, at 33 it is
    not, so the boundary is pinned rather than bracketed.
    """
    text = cue_text(cue, DOB_CUE_DISTANCE)

    assert spans_of(text, Kind.DOB) == [(len(text) - len(DOB_DATE), len(text))]


@pytest.mark.parametrize("cue", DOB_CUES)
def test_date_one_char_past_the_cue_distance_is_not_a_dob(cue: str) -> None:
    """Catches a cue window that is unbounded, or bounded at the wrong number.

    Removing the distance comparison entirely leaves the cue present in the
    text, so this row goes red on exactly that mutation and nothing else.
    """
    assert spans_of(cue_text(cue, DOB_CUE_DISTANCE + 1), Kind.DOB) == []


def test_cue_matching_is_case_insensitive() -> None:
    """Catches a cue search that compares the raw cue tuple against raw text, so a
    sentence-initial `Born` never fires."""
    text = "BORN 1985-04-12 in Cork."

    assert spans_of(text, Kind.DOB) == [(5, 15)]


def test_cue_after_the_date_does_not_qualify_it() -> None:
    """Catches a window measured on absolute distance rather than direction: the
    cue must END BEFORE the date starts, not merely sit nearby."""
    assert spans_of("1985-04-12 was when she was born.", Kind.DOB) == []


# --------------------------------------------------------------------------
# detect_all's own contract
# --------------------------------------------------------------------------


def test_detect_all_returns_candidates_in_start_order() -> None:
    """Catches a `detect_all` that concatenates per-kind result lists without
    sorting, which leaves the overlap merge reading an unordered stream."""
    text = "Mail alice@example.com, card 4111111111111111, host 192.0.2.1."

    detections = detect_all(text)

    assert [d.start for d in detections] == sorted(d.start for d in detections)


def test_detect_all_values_index_their_own_offsets() -> None:
    """Catches a `Detection.value` copied from the pattern's group rather than
    sliced from the text, which would make the placeholder map hold something the
    original never contained."""
    text = "Mail alice@example.com or ring +4930123456789 today."

    for detection in detect_all(text):
        assert text[detection.start : detection.end] == detection.value


def test_detect_all_on_empty_text_finds_nothing() -> None:
    """Catches a detector whose pattern can match the empty string, which would
    emit a zero-width placeholder for every position."""
    assert detect_all("") == []


def test_detection_is_frozen() -> None:
    """Catches a mutable `Detection`, which the merge step could rewrite in place
    and leave the raw candidate list disagreeing with the resolved one."""
    detection = Detection(kind=Kind.EMAIL, start=0, end=5, value="a@b.c")

    with pytest.raises((AttributeError, TypeError)):
        detection.start = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# the checksum helpers' own guards
# --------------------------------------------------------------------------

# `luhn_ok`, `iban_ok`, `steuer_id_ok` and `ppsn_ok` are public: the fixture
# integrity test calls them on whatever a case claims is a card or an IBAN, not
# only on what a pattern already matched. So their shape guards are reachable
# from outside `detect_all` and are asserted here rather than left to coverage.


@pytest.mark.parametrize("value", ["41111111111111a1", "4111 1111 1111 1111", ""])
def test_luhn_rejects_anything_that_is_not_a_run_of_digits(value: str) -> None:
    """Catches a `luhn_ok` that calls `int()` on a separator and raises instead of
    answering False, which turns a malformed fixture row into a crash rather than
    a rejection."""
    assert luhn_ok(value) is False


@pytest.mark.parametrize(
    ("compact", "why"),
    [
        ("DE12", "four characters, far under the 15-character minimum"),
        ("DE12" + "5" * 31, "35 characters, one over the 34-character maximum"),
        ("1234500105170648489890", "the right length, but no country code"),
        ("DEXX500105170648489890", "letters where the two check digits belong"),
    ],
)
def test_iban_rejects_lengths_and_shapes_before_the_mod97(compact: str, why: str) -> None:
    """Catches an `iban_ok` that runs the mod-97 on anything it is given.

    The 15 and 34 bounds are ISO 13616's, not the pattern's, and the two are
    separate statements: a value reaching this function from a fixture row has
    passed no pattern at all.
    """
    assert iban_ok(compact) is False, why


# --------------------------------------------------------------------------
# calendar validity, before the cue window
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("Born 31.02.1985 in Cork.", "31 February is not a day in either reading"),
        ("Born 1985-13-45 in Cork.", "month 13 and day 45"),
        ("DOB: 32/13/1985 recorded.", "no reading of 32/13 is a date"),
    ],
)
def test_a_date_shaped_run_that_is_not_a_calendar_day_is_not_a_dob(text: str, why: str) -> None:
    """Catches a DOB detector that trusts the pattern.

    Each row carries a cue, so the cue window cannot be what rejects it: only the
    calendar check can. Without it every `dd.mm.yyyy`-shaped run near the word
    "born" becomes a placeholder.
    """
    assert spans_of(text, Kind.DOB) == [], why


def test_an_ambiguous_date_is_accepted_on_its_second_reading() -> None:
    """Catches a calendar check that tries one field order only.

    `12/25/1985` is not a date read as day/month, and is one read as month/day.
    A detector that stopped at the first reading would miss every US-ordered
    birth date, and one that stopped at the second would miss the row below.
    """
    assert spans_of("DOB: 12/25/1985 on the form.", Kind.DOB) == [(5, 15)]


def test_the_other_field_order_is_still_accepted() -> None:
    """Catches the row above going one-sided: `25/12/1985` is a date read as
    day/month and not as month/day, so both orders are proved to be tried."""
    assert spans_of("DOB: 25/12/1985 on the form.", Kind.DOB) == [(5, 15)]


# --------------------------------------------------------------------------
# The two whole-string validators
# --------------------------------------------------------------------------

# `grep -n 'fullmatch\|re\.match' backend/app/redaction/*.py`
# finds exactly two, both anchored `\A...\Z`: the shape guards inside `iban_ok`
# and `ppsn_ok`. Anchored with `$` instead, each would accept a value with a
# trailing newline - and both functions are public, so a caller can hand them
# one. The two rows below are rule 4's required table for those two.

TRAILING = ("\n", "\r\n", " ")


@pytest.mark.parametrize("suffix", TRAILING, ids=["lf", "crlf", "space"])
def test_iban_rejects_trailing_whitespace(suffix: str) -> None:
    r"""Catches `\Z` relaxed to `$` in `iban_ok`'s shape guard.

    `$` matches before a trailing newline, `str.replace(" ", "")` does not strip
    one, and `int()` tolerates one, so every layer downstream would accept
    `"DE12...\n"` as a valid IBAN.
    """
    assert iban_ok(VALID_IBANS[0] + suffix) is False


@pytest.mark.parametrize("suffix", TRAILING, ids=["lf", "crlf", "space"])
def test_ppsn_rejects_trailing_whitespace(suffix: str) -> None:
    r"""Catches `\Z` relaxed to `$` in `ppsn_ok`'s shape guard, the same failure as
    the row above on the other public validator."""
    assert ppsn_ok(VALID_PPSNS[0] + suffix) is False


# --------------------------------------------------------------------------
# IPv6: the parse is not the guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("Use std::vector here.", "d:: parses as a valid IPv6"),
        ("ERROR::abc failed", "::abc parses as a valid IPv6"),
        ("see foo::bar() below", "::ba parses as a valid IPv6"),
        ("call Repo::find on it", "::find would start at the second colon"),
        ("see Repo::abcd here", "::abcd is four hex characters and parses"),
        ("use Cache::beef now", "e::beef starts one character inside the word"),
        ("x = Grid::cafe;", "d::cafe starts one character inside the word"),
    ],
)
def test_scope_resolution_operator_is_not_an_ipv6(text: str, why: str) -> None:
    """Catches an IPv6 boundary written in hex characters rather than alphanumeric.

    `ipaddress.IPv6Address` accepts `d::`, so nothing downstream catches this: a
    hex-only lookbehind lets a match start at the `d` of `std::vector` and the
    redacted text comes back with a word cut in half. Ordinary source code and
    stack traces reach this engine, so these are real inputs, not adversarial
    ones.

    The last three rows are the ones ONLY the lookbehind can reject. The first
    three clear on `IPV6_MIN_HEX_CHARS` alone, so without these the boundary
    could be reverted to hex and the whole suite would stay green - the density
    guard would be silently carrying a row it was never meant to.
    """
    assert spans_of(text, Kind.IPV6) == [], why


def test_ipv6_minimum_hex_characters_is_four() -> None:
    """Catches the density guard being widened back to nothing.

    `a::b` is a valid IPv6 address and is almost always prose. Four is the
    number, and the cost of it is the row below.
    """
    assert IPV6_MIN_HEX_CHARS == 4


def test_a_short_valid_ipv6_is_deliberately_not_detected() -> None:
    """Records the density guard's cost rather than leaving it to be rediscovered.

    `::1` is a real address and is NOT redacted. That is the trade the guard
    buys; if loopback ever needs to be redacted, this row is the one to change
    and `test_scope_resolution_operator_is_not_an_ipv6` is what it must not break.
    """
    assert spans_of("Loopback ::1 responded.", Kind.IPV6) == []


def test_a_full_length_ipv6_is_still_detected() -> None:
    """Catches the density guard being tightened until it rejects real addresses,
    which would make the two rows above pass for the wrong reason."""
    assert spans_of("Peer 2001:db8::1 dropped.", Kind.IPV6) == [(5, 16)]


# --------------------------------------------------------------------------
# the quantifier bounds are bounds
# --------------------------------------------------------------------------

# Every specimen below was generated and checked against the shipped helper, not
# copied: the cards satisfy Luhn at 12, 13, 19 and 20 digits and the IBANs
# satisfy mod-97 at 15 and 34 characters, so ONLY the length bound can decide
# each row. Without these, narrowing `{12,18}` to `{13,17}` leaves the suite green.
CARD_12 = "412345678905"
CARD_13 = "4123456789011"
CARD_19 = "4123456789012345677"
CARD_20 = "41234567890123456787"
IBAN_15 = "NO9386011117947"
IBAN_34 = "SC7118SSCB11010000000000001497US33"


@pytest.mark.parametrize(
    ("value", "detected"),
    [
        pytest.param(CARD_12, False, id="twelve-digits-rejected"),
        pytest.param(CARD_13, True, id="thirteen-digits-accepted"),
        pytest.param(CARD_19, True, id="nineteen-digits-accepted"),
        pytest.param(CARD_20, False, id="twenty-digits-rejected"),
    ],
)
def test_card_length_bounds_are_thirteen_to_nineteen_digits(value: str, detected: bool) -> None:
    """Catches `CARD_PATTERN`'s `{12,18}` narrowed or widened.

    All four specimens pass Luhn, so the checksum cannot be what decides them.
    Narrowing to `{13,17}` would silently stop redacting 13-digit and 19-digit
    card numbers, which then travel to a provider in plaintext with every test
    still green.
    """
    text = f"Card {value} on file."

    assert (spans_of(text, Kind.CARD) == [(5, 5 + len(value))]) is detected


@pytest.mark.parametrize(
    ("value", "detected"),
    [
        pytest.param(IBAN_15, True, id="fifteen-characters-accepted"),
        pytest.param(IBAN_34, True, id="thirty-four-characters-accepted"),
    ],
)
def test_iban_length_bounds_reach_both_ends_of_the_standard(value: str, detected: bool) -> None:
    """Catches `IBAN_PATTERN`'s `{11,30}` narrowed off either end.

    Both specimens satisfy mod-97, so `iban_ok` cannot be what decides them, and
    15 and 34 are the shortest and longest an IBAN is allowed to be.
    """
    text = f"Pay {value} today."

    assert (spans_of(text, Kind.IBAN) == [(4, 4 + len(value))]) is detected


@pytest.mark.parametrize(
    ("value", "detected"),
    [
        pytest.param("+123456", False, id="six-digits-rejected"),
        pytest.param("+1234567", True, id="seven-digits-accepted"),
        pytest.param("+123456789012345", True, id="fifteen-digits-accepted"),
        pytest.param("+1234567890123456", False, id="sixteen-digits-rejected"),
    ],
)
def test_international_phone_bounds_are_seven_to_fifteen_digits(value: str, detected: bool) -> None:
    """Catches `PHONE_INTERNATIONAL_PATTERN`'s `{6,14}` moved.

    Seven to fifteen digits after the `+` is E.164's own range, so a narrowed
    bound drops real numbers and a widened one starts eating reference codes.
    """
    text = f"Call {value} back."

    assert (spans_of(text, Kind.PHONE) == [(5, 5 + len(value))]) is detected


@pytest.mark.parametrize(
    ("value", "detected"),
    [
        pytest.param("01234567", False, id="eight-digits-rejected"),
        pytest.param("012345678", True, id="nine-digits-accepted"),
        pytest.param("01234567890123", True, id="fourteen-digits-accepted"),
        pytest.param("012345678901234", False, id="fifteen-digits-rejected"),
    ],
)
def test_national_phone_bounds_are_nine_to_fourteen_digits(value: str, detected: bool) -> None:
    """Catches `PHONE_NATIONAL_PATTERN`'s `{7,12}` moved.

    The upper bound is what the 20-digit batch reference in the fixture relies
    on; the row below asserts that directly rather than leaving it implied.
    """
    text = f"Desk {value} please."

    assert (spans_of(text, Kind.PHONE) == [(5, 5 + len(value))]) is detected


def test_a_twenty_digit_run_starting_with_zero_is_not_a_phone() -> None:
    """Catches the national upper bound removed.

    The fixture's `neg-long-digit-run` asserts the same string is not a CARD; a
    20-digit run beginning with 0 also has to not be a PHONE, and nothing else in
    the suite says so.
    """
    assert spans_of("Batch 00112233445566778899 finished.", Kind.PHONE) == []


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("Peer 2001::db8::1 dropped.", "two `::` runs in one address"),
        ("Peer 2001:db8:::1 dropped.", "three consecutive colons"),
    ],
)
def test_a_colon_run_with_enough_hex_but_no_valid_parse_is_not_an_ipv6(text: str, why: str) -> None:
    """Catches the density guard being mistaken for the whole check.

    Both rows clear `IPV6_MIN_HEX_CHARS` and match the pattern, so only
    `ipaddress.IPv6Address` can reject them: dropping the parse would emit a
    placeholder for a string that is not an address.
    """
    assert spans_of(text, Kind.IPV6) == [], why


# --------------------------------------------------------------------------
# a value matched only in part leaks the rest
# --------------------------------------------------------------------------

# A later review found three of these at once. The boundaries were
# right at the token level, so nothing was ever truncated mid-placeholder - but
# the pattern stopped short of the value's ordinary PRINTED form and the leftover
# characters went to the provider in the clear. The rows below are the canonical
# forms, which is the shape the fixture had been missing.

SPACED_IBAN = "DE12 5001 0517 0648 4898 90"


def test_a_space_grouped_iban_is_one_detection_not_a_card() -> None:
    """Catches an IBAN pattern that admits no separators.

    Two failures in one, and the second is the worse: with no separators allowed,
    `DE12 5001 ...` is not an IBAN at all, and `CARD_PATTERN` - which does admit
    them - picks up the digit tail instead. The country and check digits stay in
    the clear AND the value is stored under the wrong kind, so the map would
    persist an IBAN in its map as a card number.
    """
    text = f"pay {SPACED_IBAN} today"

    result = redact(text)

    assert result.text == "pay <IBAN_1> today"
    assert dict(result.mapping) == {"<IBAN_1>": SPACED_IBAN}


def test_a_space_grouped_iban_keeps_its_spaces_in_the_map() -> None:
    """Catches a detector that stores the compacted value rather than the raw span.

    The separators are stripped for the checksum only. If the map held
    `DE12500105170648489890`, rehydration would silently reformat the caller's
    text.
    """
    text = f"pay {SPACED_IBAN} today"

    result = redact(text)
    restored = result.text
    for placeholder, original in result.mapping.items():
        restored = restored.replace(placeholder, original)

    assert restored == text


@pytest.mark.parametrize(
    "value",
    ["::ffff:192.0.2.1", "2001:db8::ffff:203.0.113.7"],
    ids=["mapped-compressed", "mapped-with-prefix"],
)
def test_an_ipv4_mapped_ipv6_is_detected_whole(value: str) -> None:
    """Catches an IPv6 pattern with no dotted-quad tail.

    Without it the match stops at the last hex group and three of the four octets
    survive into the redacted text - `ping <IPV6_1>.0.2.1 done`. IPv4-mapped
    addresses are a standard form, not an edge case: they turn up in every
    dual-stack log line.
    """
    text = f"ping {value} done"

    assert spans_of(text, Kind.IPV6) == [(5, 5 + len(value))]


@pytest.mark.parametrize(
    ("text", "end"),
    [
        pytest.param("host 2001:db8::1. ", 16, id="sentence-final-full-stop"),
        pytest.param("host 2001:db8::1.5 percent", 16, id="one-dotted-part-follows"),
        pytest.param("host 2001:db8::1.2.3 logged", 16, id="two-dotted-parts-follow"),
    ],
)
def test_the_dotted_tail_is_all_or_nothing(text: str, end: int) -> None:
    """Catches the IPv4-mapped tail written `{0,3}` instead of exactly three or none.

    The middle and last rows are the discriminating ones: with `{0,3}` the match
    greedily swallows `.5` or `.2.3`, the result no longer parses as an address,
    and a REAL IPv6 address stops being redacted. The first row alone cannot see
    that - a trailing full stop is followed by no digit, so both spellings stop at
    the same place.
    """
    assert spans_of(text, Kind.IPV6) == [(5, end)]


def test_a_lowercase_ppsn_is_detected() -> None:
    """Catches an uppercase-only PPSN pattern.

    `ppsn_ok` upper-cases its argument, so with an uppercase-only pattern that
    normalisation is unreachable from the detect path and a PPS number typed in
    lower case is a silent miss that no mutation can redden.
    """
    assert spans_of("PPSN 1234567t recorded.", Kind.PPSN) == [(5, 13)]


# --------------------------------------------------------------------------
# the tenth kind, whose input is not the text
# --------------------------------------------------------------------------

DIRECTORY_NAME = "Anna Schmidt"
NAME_IN_TEXT = "Ansprechpartner: Anna Schmidt, Abteilung 4"


def test_detect_all_without_a_directory_yields_no_person_candidate() -> None:
    """Catches a `detect_all` that reads a directory from anywhere but its own argument -
    a module-level default, a process-wide cache, a store import.

    This is what keeps the seam additive: every existing caller, the residency and
    accuracy suites included, calls `detect_all(text)` with one argument, and a PERSON
    candidate appearing there would be a name matched for a tenant nobody asked about.
    The text carries a name that IS in a directory in the row below, so an implementation
    that simply found nothing in this string cannot be what passes this.
    """
    assert [d.kind for d in detect_all(NAME_IN_TEXT) if d.kind is Kind.PERSON] == []


def test_detect_all_with_a_directory_yields_the_person_candidate() -> None:
    """The positive control for the row above: catches a `directory=` keyword that is
    accepted and then ignored, which would make the negative row pass for the wrong reason
    and leave the whole kind unreachable from the engine's own entry point."""
    directory = NameDirectory.compile([DIRECTORY_NAME])

    found = [d for d in detect_all(NAME_IN_TEXT, directory=directory) if d.kind is Kind.PERSON]

    assert [(d.start, d.end, d.value) for d in found] == [(17, 29, DIRECTORY_NAME)]


def test_a_directory_candidate_joins_the_others_in_start_order() -> None:
    """Catches PERSON candidates appended after the nine pattern kinds rather than merged
    into the sorted list: `resolve_overlaps` walks candidates in start order, so a name
    arriving out of order makes the merge compare spans that do not touch and lets an
    overlapping pair through as two placeholders."""
    text = "Anna Schmidt wrote from alice@example.com"
    directory = NameDirectory.compile([DIRECTORY_NAME])

    found = detect_all(text, directory=directory)

    assert [d.start for d in found] == sorted(d.start for d in found)
    assert [d.kind for d in found] == [Kind.PERSON, Kind.EMAIL]


def test_a_directory_name_inside_an_email_address_loses_to_the_longer_span() -> None:
    """Catches PERSON given a precedence that beats a longer overlapping span: the whole
    address is the value a caller must not leak, and a name matched inside it would
    replace `anna.schmidt` alone and leave `@example.com` - with the local part gone, the
    map's inverse no longer reproduces the input. The "unequal overlap is decided
    by length" reading, exercised through `redact` because that is where the merge runs."""
    directory = NameDirectory.compile(["anna.schmidt"])

    result = redact("write to anna.schmidt@example.com", directory=directory)

    assert [(d.kind, d.value) for d in result.detections] == [
        (Kind.EMAIL, "anna.schmidt@example.com")
    ]
