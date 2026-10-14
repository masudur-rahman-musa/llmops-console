r"""`limits/budget.py`: the month key, the pre-checks, the two boundaries and the override.

Library only. Nothing here builds an app, sends a request or touches Redis - the
route wiring is tests/gateway/test_budget_enforcement.py and the store's Redis
semantics are tests/limits/test_redis_budget.py. What this file pins is the
arithmetic and the validation that decide, for one spend and one cap, whether a
request is served, warned about or refused.

The contract these cases are written against, before it exists:

* `month_key(tenant, month)` - `budget:{tenant}:{YYYY-MM}`, raising
  `IdentifierInvalid` for a tenant or a month that may not shape a key. The
  checks live in the key function rather than in each store method, so "before
  any command is sent" is a property of the only thing a command needs.
* `current_month(now=None)` - the `YYYY-MM` a spend is booked under, in UTC.
* `to_micro_usd(usd)` - a `Decimal` cost as the integer counter's unit
  (decision C), `ROUND_HALF_UP` at the ledger's own quantum.
* `decide(spend_micro_usd=..., cap_micro_usd=...)` - `BudgetDecision.SERVE`,
  `WARN` or `REJECT`, both boundaries inclusive.
* `budget_ratio(spend_micro_usd=..., cap_micro_usd=...)` - what
  `gateway_budget_ratio{tenant}` reads.
* `verify_override(header, tenant=..., month=..., secret=...)` - `True` only for
  a token this gateway minted, for THIS tenant and THIS month.
* `BUDGET_KEY_TTL_SECONDS`, `WARN_RATIO`, `MAX_OVERRIDE_HEADER_CHARS`,
  `BUDGET_OVERRIDE_HEADER`, `MONTH_PATTERN`, `MICRO_USD_PER_USD` - the constants
  a project rule requires a test to pin exactly.

a project rule on both patterns: the month is matched with `re.fullmatch`, and
the rejection table below carries a trailing newline, a trailing CRLF and a
trailing space, because a check written `^...$` with `.match` accepts the first
of those and would put a caller-controlled line break into a Redis key.

The tenant is checked against the gateway's OWN `IDENTIFIER_PATTERN` and
`MAX_IDENTIFIER_LENGTH`, imported rather than restated, so the name a budget
counter accepts and the name `gateway/context.py` authenticated can never
disagree.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.limits.budget import (
    BUDGET_KEY_PREFIX,
    BUDGET_KEY_TTL_SECONDS,
    BUDGET_OVERRIDE_HEADER,
    MAX_OVERRIDE_HEADER_CHARS,
    MICRO_USD_PER_USD,
    MONTH_PATTERN,
    OVERRIDE_SIGNATURE_PATTERN,
    WARN_RATIO,
    BudgetDecision,
    budget_ratio,
    current_month,
    decide,
    month_key,
    to_micro_usd,
    verify_override,
)
from app.limits.cost import COST_QUANTUM
from app.model_routing.errors import IdentifierInvalid

TENANT = "evals"
OTHER_TENANT = "cuan"
MONTH = "2026-09"
LAST_MONTH = "2026-08"

SECRET = "test-budget-override-secret-belonging-to-nobody"
"""A worthless string, not anyone's key. It is the only secret this
file needs, and every token below is minted from it here rather than copied in as
a literal, so a case cannot pass against a signature nobody could reproduce."""

OTHER_SECRET = "test-budget-override-secret-that-is-not-the-one-configured"

MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)

CAP_USD = Decimal("10.00")
CAP = 10_000_000
"""One shipped cap, in dollars and in the counter's own unit. Both spellings are
written out so a case never recomputes the conversion it is testing."""

WARN_AT = 8_000_000
"""Exactly 80% of `CAP`, the inclusive warn boundary."""


def token(*, tenant: str, month: str, secret: str, stated_month: str | None = None) -> str:
    """Mint one override header the way the gateway signs it.

    `stated_month` is what the header SAYS, defaulting to the month that was
    signed. The two are separable so a row can carry a signature over one month
    and a month field naming another, which is the shape an implementation that
    trusts the header's own month field would accept.
    """
    digest = hmac.new(
        secret.encode("utf-8"), f"{tenant}:{month}".encode(), hashlib.sha256
    ).hexdigest()
    return f"{stated_month if stated_month is not None else month}:{digest}"


VALID = token(tenant=TENANT, month=MONTH, secret=SECRET)


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------


def test_the_month_key_is_the_specced_shape() -> None:
    """Catches a key built in any other shape - `budget:{month}:{tenant}` with the two
    segments swapped, a key with no month in it at all (which would make one counter
    serve every month for ever), or a prefix that collides with the response cache's.
    The prefix is asserted as its own segment, so a rename of the constant that did not
    reach the key is caught too."""
    assert month_key(TENANT, MONTH) == f"{BUDGET_KEY_PREFIX}:{TENANT}:{MONTH}"
    assert month_key(TENANT, MONTH).split(":") == [BUDGET_KEY_PREFIX, TENANT, MONTH]
    assert month_key(OTHER_TENANT, MONTH) != month_key(TENANT, MONTH)
    assert month_key(TENANT, LAST_MONTH) != month_key(TENANT, MONTH)


def test_the_key_prefix_is_the_published_one() -> None:
    """Catches a renamed prefix, which orphans every counter a running gateway has
    already accumulated this month and silently restarts every tenant's spend at zero."""
    assert BUDGET_KEY_PREFIX == "budget"


REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_TENANT),
    ("uppercase", "Evals"),
    ("a space", "ev als"),
    ("a colon, which would forge a second key segment", "ev:als"),
    ("a path separator", "ev/als"),
    ("a glob character, which would widen a later SCAN", "evals*"),
    ("a trailing newline", "evals\n"),
    ("a trailing CRLF", "evals\r\n"),
    ("a trailing space", "evals "),
    ("a leading separator", "-evals"),
]


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_a_malformed_tenant_is_refused_before_any_command(label: str, tenant: str) -> None:
    """Catches a key function that interpolates a client-supplied tenant straight into
    the key: `ev:als` would address the counter of a tenant named `als` under a month
    named `ev`, and `evals\\n` puts a caller-controlled line break into a Redis key. The
    refusal happens in the key function, which is what every command needs first, so
    there is no order in which a command could be sent with one of these (standing
    rule 1). The trailing-newline row is also a project rule's: a check written `^...$`
    with `.match` accepts it."""
    with pytest.raises(IdentifierInvalid):
        month_key(tenant, MONTH)


def test_the_tenant_length_boundary_is_inclusive() -> None:
    """Catches an off-by-one at the cap in either direction - a check written `>=` that
    refuses a tenant of exactly the permitted length (which `gateway/context.py` already
    authenticated, so the request would 500 rather than being served), and a check
    written on a different constant than the gateway's own, which would let a tenant one
    character over the gateway's cap mint a key."""
    assert month_key(MAX_TENANT, MONTH) == f"{BUDGET_KEY_PREFIX}:{MAX_TENANT}:{MONTH}"
    with pytest.raises(IdentifierInvalid):
        month_key(OVER_CAP_TENANT, MONTH)


REJECTED_MONTHS: list[tuple[str, str]] = [
    ("empty", ""),
    ("a one-digit month", "2026-9"),
    ("month zero", "2026-00"),
    ("month thirteen", "2026-13"),
    ("a three-digit year", "202-06"),
    ("a five-digit year", "20261-06"),
    ("an underscore separator", "2026_06"),
    ("no separator", "202606"),
    ("a day appended", "2026-06-01"),
    ("a colon, which would forge a fourth key segment", "2026-06:x"),
    ("non-digits", "aaaa-bb"),
    ("a trailing newline", "2026-06\n"),
    ("a trailing CRLF", "2026-06\r\n"),
    ("a trailing space", "2026-06 "),
    ("a leading space", " 2026-06"),
    ("a glob character", "2026-*"),
]


@pytest.mark.parametrize(
    ("label", "month"), REJECTED_MONTHS, ids=[row[0] for row in REJECTED_MONTHS]
)
def test_a_malformed_month_is_refused_before_any_command(label: str, month: str) -> None:
    """Catches a month that never passes through a pattern at all, and a pattern written
    `^\\d{4}-\\d{2}$` with `.match`: it accepts `2026-06\\n`, accepts
    month `00` and month `13` as periods no request will ever be booked under again, and
    accepts `2026-06:x`, which forges a key segment. The refusal is in the key function
    for the same reason the tenant's is."""
    with pytest.raises(IdentifierInvalid):
        month_key(TENANT, month)


@pytest.mark.parametrize("month", ["2026-01", "2026-09", "2026-12", "1970-01"])
def test_a_well_formed_month_is_accepted(month: str) -> None:
    """The other half of the rejection table, without which a pattern that refuses
    everything would pass every row above. Catches a pattern anchored so tightly that a
    real month - January and December especially, the two an `0[1-9]|1[0-2]` alternation
    gets wrong - cannot be booked."""
    assert month_key(TENANT, month) == f"{BUDGET_KEY_PREFIX}:{TENANT}:{month}"


def test_the_month_pattern_is_used_with_fullmatch_semantics() -> None:
    """Catches a pattern object that would behave differently under `.match`: asserted on
    the compiled pattern itself so the claim in the module docstring is checked against
    the object rather than only through the key function."""
    assert isinstance(MONTH_PATTERN, re.Pattern)
    assert MONTH_PATTERN.fullmatch(MONTH) is not None
    assert MONTH_PATTERN.fullmatch("2026-09\n") is None
    assert MONTH_PATTERN.fullmatch("2026-13") is None


# --------------------------------------------------------------------------
# the month a spend is booked under
# --------------------------------------------------------------------------


def test_the_current_month_is_the_utc_month_not_the_local_one() -> None:
    """Catches a month taken from the process's local clock: at 23:30 on the last day of
    a month in UTC+2 the local month is already the next one, so half an hour of spend
    would be booked against a counter with a different key and every cap would be
    briefly unenforceable. Also catches a `%Y-%m` written `%Y-%-m`, which produces
    `2026-9` and would then be refused by the month pattern."""
    late = datetime(2026, 9, 30, 23, 30, tzinfo=UTC)
    east = late.astimezone(timezone(timedelta(hours=2)))

    assert current_month(late) == "2026-09"
    assert current_month(east) == "2026-09"
    assert MONTH_PATTERN.fullmatch(current_month(late)) is not None


def test_the_current_month_defaults_to_now() -> None:
    """Catches a signature the route cannot call: `create_app` and the two routes take
    no clock, so a required `now` argument would leave no caller able to book anything.
    The answer is only checked for shape, because the value is the wall clock."""
    assert MONTH_PATTERN.fullmatch(current_month()) is not None


# --------------------------------------------------------------------------
# the counter's unit (decision C)
# --------------------------------------------------------------------------


def test_micro_usd_rounding_matches_the_ledger_quantum() -> None:
    """Catches a counter in a unit that cannot hold a ledger row exactly. The ledger
    quantizes to six places, so one micro-USD is one ledger quantum: a counter in
    millicents would lose the last two places of every row, and `int(usd * 1_000_000)`
    truncates - it books 0 for a row of 0.0000005 that `ROUND_HALF_UP` books as 1, and
    the counter drifts below the ledger by one quantum per request for ever. The
    half-up row is also what separates this from `ROUND_HALF_EVEN`, which answers 0."""
    assert MICRO_USD_PER_USD == 1_000_000
    assert Decimal("0.000001") == COST_QUANTUM
    assert Decimal(1) / Decimal(MICRO_USD_PER_USD) == COST_QUANTUM

    assert to_micro_usd(Decimal("0.000282")) == 282
    assert to_micro_usd(Decimal("10.00")) == CAP
    assert to_micro_usd(Decimal("0")) == 0
    assert to_micro_usd(Decimal("0.0000005")) == 1
    assert to_micro_usd(Decimal("0.0000004")) == 0
    assert isinstance(to_micro_usd(Decimal("0.000282")), int)


# --------------------------------------------------------------------------
# the two boundaries
# --------------------------------------------------------------------------

DECISION_ROWS: list[tuple[str, int, int, BudgetDecision]] = [
    ("nothing spent", 0, CAP, BudgetDecision.SERVE),
    ("one micro under the warn boundary", WARN_AT - 1, CAP, BudgetDecision.SERVE),
    ("exactly the warn boundary", WARN_AT, CAP, BudgetDecision.WARN),
    ("one micro over the warn boundary", WARN_AT + 1, CAP, BudgetDecision.WARN),
    ("one micro under the cap", CAP - 1, CAP, BudgetDecision.WARN),
    ("exactly the cap", CAP, CAP, BudgetDecision.REJECT),
    ("one micro over the cap", CAP + 1, CAP, BudgetDecision.REJECT),
    ("far over the cap", CAP * 3, CAP, BudgetDecision.REJECT),
    # A cap whose 80% is not a whole number of micro-USD. 80% of 9 is 7.2, so 7
    # serves and 8 warns; an implementation comparing `spend * 10 >= cap * 8`
    # agrees, and one comparing `spend >= cap * 8 // 10` (integer division to 7)
    # warns one micro early.
    ("under a fractional warn boundary", 7, 9, BudgetDecision.SERVE),
    ("over a fractional warn boundary", 8, 9, BudgetDecision.WARN),
    ("at a cap of one micro", 1, 1, BudgetDecision.REJECT),
]
"""The whole decision table, in the counter's own unit. The `SERVE` row one micro
under the cap is deliberately spelled through `BudgetDecision` rather than as a
literal so a renamed member fails here rather than silently comparing unequal."""


@pytest.mark.parametrize(
    ("label", "spend", "cap", "expected"), DECISION_ROWS, ids=[row[0] for row in DECISION_ROWS]
)
def test_the_decision_table(label: str, spend: int, cap: int, expected: BudgetDecision) -> None:
    """The two inclusive boundaries in one table (criteria 2 and 3).

    Catches `>` where the criteria say "at or above" - at exactly 80% the tenant would
    never be warned and at exactly the cap it would be served, which is the request the
    criterion says is the 429 - and catches the mirror error, a `>=` on the wrong side
    that refuses one micro under the cap. The two fractional rows catch a warn threshold
    computed by integer division, which warns a micro early on every cap that is not a
    multiple of five."""
    assert decide(spend_micro_usd=spend, cap_micro_usd=cap) is expected


def test_the_warn_boundary_is_inclusive_at_eighty_percent() -> None:
    """The warn boundary named alone, against the published ratio rather than against a
    number this test computed. Catches a threshold hardcoded at some other fraction (0.9
    is the other obvious choice) while the table above still passed, because the table's
    rows are all derived from the same `WARN_AT` constant."""
    assert Decimal("0.8") == WARN_RATIO
    assert decide(spend_micro_usd=int(CAP * WARN_RATIO), cap_micro_usd=CAP) is BudgetDecision.WARN
    assert (
        decide(spend_micro_usd=int(CAP * WARN_RATIO) - 1, cap_micro_usd=CAP) is BudgetDecision.SERVE
    )


def test_the_reject_boundary_is_inclusive_at_the_cap() -> None:
    """The reject boundary named alone. Catches an implementation that only refuses once
    the cap is EXCEEDED, which serves one more request than the criterion allows every
    month, on the request that is exactly at the cap."""
    assert decide(spend_micro_usd=CAP, cap_micro_usd=CAP) is BudgetDecision.REJECT
    assert decide(spend_micro_usd=CAP - 1, cap_micro_usd=CAP) is not BudgetDecision.REJECT
    assert decide(spend_micro_usd=CAP - 1, cap_micro_usd=CAP) is BudgetDecision.WARN


def test_the_ratio_is_spend_over_cap() -> None:
    """Catches a gauge fed with the raw spend (a dashboard panel scaled to 10,000,000
    instead of to 1, and an alert on `> 0.8` that never fires) and one fed with the
    inverted fraction. The three points are the two boundaries and one ordinary
    request."""
    assert budget_ratio(spend_micro_usd=0, cap_micro_usd=CAP) == 0.0
    assert budget_ratio(spend_micro_usd=WARN_AT, cap_micro_usd=CAP) == pytest.approx(0.8)
    assert budget_ratio(spend_micro_usd=CAP, cap_micro_usd=CAP) == pytest.approx(1.0)
    assert budget_ratio(spend_micro_usd=282, cap_micro_usd=CAP) == pytest.approx(0.0000282)


def test_the_ratio_keeps_reading_above_one_over_cap() -> None:
    """Catches a ratio clamped to 1.0: the gauge is what an operator reads to see how far
    past its cap a tenant with a standing override has gone, and a clamp makes 100% and
    400% the same picture."""
    assert budget_ratio(spend_micro_usd=CAP * 4, cap_micro_usd=CAP) == pytest.approx(4.0)


# --------------------------------------------------------------------------
# the key's expiry
# --------------------------------------------------------------------------


def test_the_budget_key_ttl_is_pinned() -> None:
    """Pins the exact constant, because both wrong values are silent.
    Too short and a counter expires mid-month, resetting a tenant's spend to zero and
    handing it a second budget; absent (or infinite) and one key per tenant per month
    accumulates in Redis for ever, which is the risk section 5 names. Sixty-two days is
    two of the longest months, so a key written on the first of a 31-day month is still
    readable through the whole of the next one."""
    assert BUDGET_KEY_TTL_SECONDS == 5_356_800
    assert BUDGET_KEY_TTL_SECONDS == 62 * 24 * 60 * 60
    assert BUDGET_KEY_TTL_SECONDS > 31 * 24 * 60 * 60


# --------------------------------------------------------------------------
# the override (decision A)
# --------------------------------------------------------------------------


def test_the_override_header_name_is_the_published_one() -> None:
    """Pins the wire name. Catches a rename that leaves every operator's minted token
    unusable while the gateway still answers 429 with no hint that the header was read
    under another name."""
    assert BUDGET_OVERRIDE_HEADER == "X-Gateway-Budget-Override"


def test_the_override_header_cap_is_exactly_a_well_formed_token() -> None:
    """Pins the length cap against the thing it has to admit. Catches a
    cap set below a real token, which refuses every legitimate override while looking
    like a signature failure, and a cap so wide that an unbounded header is hashed
    before anything checks its size. Seven characters of month, one colon and
    sixty-four of hex."""
    assert MAX_OVERRIDE_HEADER_CHARS == 72
    assert len("2026-09") + len(":") + 64 == MAX_OVERRIDE_HEADER_CHARS
    assert len(VALID) == MAX_OVERRIDE_HEADER_CHARS


def test_a_valid_token_for_this_tenant_and_month_is_accepted() -> None:
    """The accept row, without which every refusal below would pass against a function
    that returns `False` unconditionally - and a gateway that refused every override
    would only be discovered by an operator locked out during an incident."""
    assert verify_override(VALID, tenant=TENANT, month=MONTH, secret=SECRET) is True


OVERRIDE_ROWS: list[tuple[str, str | None, str]] = [
    ("absent", None, SECRET),
    ("empty", "", SECRET),
    ("the month field alone", MONTH, SECRET),
    ("the signature alone", VALID.split(":")[1], SECRET),
    ("no separator", VALID.replace(":", ""), SECRET),
    ("a second separator", f"{VALID}:extra", SECRET),
    ("the signature uppercased", VALID.upper(), SECRET),
    ("a signature that is not hex", f"{MONTH}:{'z' * 64}", SECRET),
    ("a signature one character short", VALID[:-1], SECRET),
    (
        "a signature minted for another tenant",
        token(tenant=OTHER_TENANT, month=MONTH, secret=SECRET),
        SECRET,
    ),
    (
        "a signature minted for last month",
        token(tenant=TENANT, month=LAST_MONTH, secret=SECRET),
        SECRET,
    ),
    (
        "last month's signature with the current month stated",
        token(tenant=TENANT, month=LAST_MONTH, secret=SECRET, stated_month=MONTH),
        SECRET,
    ),
    (
        "this month's signature with last month stated",
        token(tenant=TENANT, month=MONTH, secret=SECRET, stated_month=LAST_MONTH),
        SECRET,
    ),
    (
        "a signature minted with another secret",
        token(tenant=TENANT, month=MONTH, secret=OTHER_SECRET),
        SECRET,
    ),
    (
        "a token minted with the empty secret, against a gateway with no secret set",
        token(tenant=TENANT, month=MONTH, secret=""),
        "",
    ),
    ("a valid token against a gateway with no secret set", VALID, ""),
]
"""Every way a token can fail, and the secret in force for each row. The last two
are the reason the table carries a secret at all: an empty
`budget_override_secret` is the shipped default, and HMAC over an empty key is
still perfectly good HMAC, so an implementation that does not refuse outright
lets anyone who can read this repository mint an override for an unconfigured
gateway."""


@pytest.mark.parametrize(
    ("label", "header", "secret"), OVERRIDE_ROWS, ids=[row[0] for row in OVERRIDE_ROWS]
)
def test_the_override_table(label: str, header: str | None, secret: str) -> None:
    """Every refusal the criterion names, one row each.

    Catches, row by row: an implementation that treats a present header as sufficient;
    one that reads the TENANT out of the token instead of taking it from the
    authenticated context, which turns a captured token into a universal bypass (section
    5's risk); one that never compares the month field to the current month, so last
    month's token works for ever; one that signs the month the HEADER states rather than
    the month the gateway is in, which is the same bypass by a different route; and one
    that accepts an unconfigured gateway's empty secret, which is every deployment that
    has not set the variable."""
    assert verify_override(header, tenant=TENANT, month=MONTH, secret=secret) is False


def test_an_over_length_override_header_is_refused_before_it_is_parsed() -> None:
    """Catches a parser that slices a fixed prefix out of an unbounded header - `month =
    header[:7]`, `signature = header[8:72]` - which would accept this ten-thousand
    character value because its first seventy-two characters are a genuine token. The
    padded value is the only arrangement in which the cap is the thing that decides:
    with a cap, it is refused unread; without one, a prefix-slicing parser verifies it
    and a split-on-colon parser refuses it for a different reason. The one-character-over
    row makes the boundary inclusive rather than approximate."""
    padded = VALID + "x" * 9_928

    assert len(padded) > MAX_OVERRIDE_HEADER_CHARS
    assert verify_override(padded, tenant=TENANT, month=MONTH, secret=SECRET) is False
    assert verify_override(VALID + "x", tenant=TENANT, month=MONTH, secret=SECRET) is False
    assert verify_override(VALID, tenant=TENANT, month=MONTH, secret=SECRET) is True


def test_a_zero_cap_has_no_ratio_rather_than_a_division_error() -> None:
    """A branch the plan did not name, pinned where it is introduced.

    A tenant whose `budget_usd_month` is `0.00` is refused by `decide` on every request,
    so the ratio would never be computed - except on the one request that carries a
    valid override, which is served and books its cost. Catches the obvious
    implementation, `spend / cap`, which turns exactly that request into an untyped 500
    raised from the middle of `book_cost`, after the provider has already been paid."""
    assert decide(spend_micro_usd=0, cap_micro_usd=0) is BudgetDecision.REJECT
    assert budget_ratio(spend_micro_usd=0, cap_micro_usd=0) == 0.0
    assert budget_ratio(spend_micro_usd=282, cap_micro_usd=0) == 0.0


def test_the_override_signature_pattern_admits_a_minted_signature_and_nothing_else() -> None:
    """The other constant the implementation invented.

    The pattern is not decoration: `hmac.compare_digest` on `str` operands raises
    `TypeError` unless both are ASCII, so a header carrying any non-ASCII character
    would reach the comparison and crash the request rather than being refused. Catches
    that missing guard, and catches the pattern being used with `.match`, which would
    admit a signature with a trailing newline."""
    minted = VALID.split(":")[1]

    assert OVERRIDE_SIGNATURE_PATTERN.fullmatch(minted) is not None
    assert OVERRIDE_SIGNATURE_PATTERN.fullmatch(minted.upper()) is None
    assert OVERRIDE_SIGNATURE_PATTERN.fullmatch(minted[:-1]) is None
    assert OVERRIDE_SIGNATURE_PATTERN.fullmatch(f"{minted}\n") is None
    assert OVERRIDE_SIGNATURE_PATTERN.fullmatch("é" * 64) is None
    assert (
        verify_override(f"{MONTH}:{'é' * 64}", tenant=TENANT, month=MONTH, secret=SECRET) is False
    )
