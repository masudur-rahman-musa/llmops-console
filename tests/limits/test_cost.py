"""Pricing one request: the arithmetic, the estimator and the `unpriced` sentinel.

Every table here is built straight from the models, so nothing in this file
depends on YAML or on the shipped price file; `tests/limits/test_prices.py` owns
loading and selection. The rates are deliberately asymmetric, and so are the two
token counts, because a symmetric fixture cannot tell an implementation that
prices both directions at the input rate from one that prices them separately.

Money is `Decimal` throughout and the results are asserted as exact strings
rather than as bands: a band admits precisely the float implementation decision E
exists to forbid.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from opsconsole.limits.cost import _WARNED_KEYS, Cost, cost_for, estimate_tokens
from opsconsole.limits.prices import PriceEntry, PriceTable
from opsconsole.providers.base import Usage

KEY = "prov-a/model-a"
"""A made-up `provider/model` pair in `Adapter.price_key`'s format."""

OLD_ENTRY = PriceEntry(
    effective_from=date(2025, 1, 1),
    input_usd_per_1m=Decimal("1.00"),
    output_usd_per_1m=Decimal("2.00"),
)
CURRENT_ENTRY = PriceEntry(
    effective_from=date(2025, 6, 1),
    input_usd_per_1m=Decimal("3.00"),
    output_usd_per_1m=Decimal("15.00"),
)
CURRENT_VERSION = f"{KEY}@2025-06-01"
"""Decision A's version string for `CURRENT_ENTRY`: the key, `@`, and the entry's
OWN `effective_from` - never the instant the request was priced at, and never a
document-level version."""

AT = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
"""The instant every case prices at. A month after `CURRENT_ENTRY` took effect and
six months after `OLD_ENTRY` did, so a version stamped from the query date, from
the first entry in the table or from the newest one regardless of date each
produces a visibly different string."""

REPORTED = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
"""What the provider said the call consumed. 1000 and 500 are different numbers on
purpose: with 3.00 in and 15.00 out, no other pairing of the four values produces
the expected total."""


def table_with(*entries: PriceEntry, key: str = KEY) -> PriceTable:
    """One price key carrying the given entries."""
    return PriceTable(prices={key: tuple(entries)})


@pytest.fixture
def table() -> PriceTable:
    """Two entries under one key, so the version string has something to get wrong."""
    return table_with(OLD_ENTRY, CURRENT_ENTRY)


# --- reported usage -----------------------------------------------------------


def test_input_and_output_are_priced_per_1m(table: PriceTable) -> None:
    """Catches the three ways this arithmetic goes wrong and still returns a number:
    pricing both directions at one rate (1500 tokens at 3.00 is 0.004500), swapping the
    two rates (0.016500), and dividing by 1000 or by nothing rather than by 1,000,000
    (10.500000 and 10500.000000). 1000 in at 3.00 plus 500 out at 15.00 is 0.010500 and
    nothing else here is."""
    cost = cost_for(table, KEY, AT, usage=REPORTED)

    assert cost.usd == Decimal("0.010500")


def test_reported_usage_is_not_an_estimate(table: PriceTable) -> None:
    """Catches a `Cost` that flags every row `estimated=True` - which costs nothing at the
    moment it is written and makes the flag unreadable in the ledger forever, since a
    column that is always true separates nothing. The provider reported these tokens, so
    the honest answer is False."""
    cost = cost_for(table, KEY, AT, usage=REPORTED)

    assert cost.estimated is False


def test_the_price_version_names_the_entrys_own_effective_from(table: PriceTable) -> None:
    """Catches a version string built from anything but the SELECTED entry's own
    `effective_from`: the query instant (`@2025-07-01`), the first entry in the table
    (`@2025-01-01`), or a document-level version bumped by hand, which decision A rejects
    because editing a price without bumping it silently rewrites what every historical row
    was charged at. The table carries two entries dated six months apart and the instant
    matches neither, so each of those three mistakes produces a different string."""
    cost = cost_for(table, KEY, AT, usage=REPORTED)

    assert cost.price_version == CURRENT_VERSION


ROUNDING_ENTRY = PriceEntry(
    effective_from=date(2025, 1, 1),
    input_usd_per_1m=Decimal("2.50"),
    output_usd_per_1m=Decimal("0.00"),
)
"""One prompt token at 2.50 per 1M is 0.0000025 exactly - a value sitting on the
half at the seventh decimal place, which is the only kind that tells the rounding
modes apart."""


def test_the_cost_is_quantized_to_exactly_six_decimal_places() -> None:
    """Catches all three failures at once, because 0.0000025 renders differently under
    each: no quantization prints "0.0000025", truncation or `ROUND_DOWN` prints
    "0.000002", and `ROUND_HALF_EVEN` - Python's `Decimal` default, so the one an
    implementation gets by calling `quantize` without a rounding argument - also prints
    "0.000002". Only `ROUND_HALF_UP` prints "0.000003". Asserted as the exact string
    rather than a value or a band: `Decimal("0.000003")` compares equal
    to `Decimal("3E-6")`, so an equality assertion alone would not pin the six places."""
    usage = Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)

    cost = cost_for(table_with(ROUNDING_ENTRY), KEY, AT, usage=usage)

    assert str(cost.usd) == "0.000003"


def test_zero_reported_tokens_cost_exactly_zero_and_are_not_an_estimate(
    table: PriceTable,
) -> None:
    """Catches the truthiness trap: an implementation branching on `if not usage` or `if
    not usage.total_tokens` instead of `if usage is None` turns a genuinely empty call -
    a refusal, a cached answer, a stream that produced no text - into a fabricated
    character-count row, so the ledger reports an estimate where the provider actually
    reported zero. The version assertion proves the priced path ran at all rather than the
    `unpriced` short circuit, and the exact string proves a zero is still quantized to six
    places instead of arriving as `Decimal("0")`."""
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    cost = cost_for(table, KEY, AT, usage=usage)

    assert str(cost.usd) == "0.000000"
    assert cost.estimated is False
    assert cost.price_version == CURRENT_VERSION


# --- absent usage: the character-count estimator -------------------------------


def test_absent_usage_is_estimated_from_characters(table: PriceTable) -> None:
    """Catches an estimator applied to the total character count and booked as input (60
    chars / 4 = 15 tokens at 3.00 is 0.000045), one that estimates the prompt and
    zero-fills the completion (0.000030), and a divisor other than four (at 3 chars per
    token it is 0.000140). 40 and 20 characters are 10 and 5 tokens, which at 3.00 and
    15.00 is 0.000105. It also catches an estimated row reported as `estimated=False` -
    decision B calls that flag the honest part of the heuristic - and a version dropped to
    `unpriced` merely because usage was absent, when the entry was found and only the
    token counts were not."""
    cost = cost_for(table, KEY, AT, usage=None, prompt_chars=40, completion_chars=20)

    assert str(cost.usd) == "0.000105"
    assert cost.estimated is True
    assert cost.price_version == CURRENT_VERSION


ESTIMATOR_ROWS: list[tuple[str, int, int]] = [
    ("an exact multiple", 40, 10),
    ("a partial token rounds down", 39, 9),
    ("fewer characters than one token", 3, 0),
    ("no characters at all", 0, 0),
]


@pytest.mark.parametrize(
    ("case", "chars", "tokens"), ESTIMATOR_ROWS, ids=[row[0] for row in ESTIMATOR_ROWS]
)
def test_the_estimator_floors_at_four_characters_per_token(
    case: str, chars: int, tokens: int
) -> None:
    """Catches a divisor other than four and rounding that is not a floor. The two boundary
    rows are what the arithmetic case above cannot see: 3 characters must be 0 tokens and
    not 1, because an estimator that rounds up books a request with no text as having
    consumed something, and 39 must be 9 and not 10. Pins the divisor as a value (standing
    rules 2 and 37) rather than leaving it implied by one total."""
    assert estimate_tokens(chars) == tokens


# --- the unpriced sentinel -----------------------------------------------------

FUTURE_ENTRY = PriceEntry(
    effective_from=date(2030, 1, 1),
    input_usd_per_1m=Decimal("99.00"),
    output_usd_per_1m=Decimal("99.00"),
)
"""Priced far above every other rate in this file, so a sentinel that borrowed the
future entry's numbers instead of short-circuiting would be unmissable."""

UNPRICED_CASES: list[tuple[str, PriceTable, str]] = [
    ("the table does not carry the key", table_with(CURRENT_ENTRY), "prov-z/model-z"),
    ("every entry for the key is future dated", table_with(FUTURE_ENTRY), KEY),
]


@pytest.mark.parametrize(
    ("case", "unpriced_table", "key"), UNPRICED_CASES, ids=[row[0] for row in UNPRICED_CASES]
)
def test_an_unpriced_key_costs_zero_and_does_not_raise(
    case: str, unpriced_table: PriceTable, key: str
) -> None:
    """Catches decision C implemented as a raise - a bare `KeyError`, or a typed error the
    route would have to catch - which lets a gap in a config file change a served
    request's status code, the one outcome the decision forbids. It also catches a
    sentinel that borrows the future entry's 99.00, one reporting `estimated=False` (a
    fabricated zero then reads as a genuinely free call), and a version of `""` or `None`
    instead of the literal `"unpriced"` that makes these rows findable with one equality
    in SQL. Both rows carry REPORTED usage, so nothing about the estimator can be what
    produces the flag - only the missing entry can."""
    cost = cost_for(unpriced_table, key, AT, usage=REPORTED)

    assert isinstance(cost, Cost)
    assert cost.usd == Decimal("0")
    assert str(cost.usd) == "0.000000"
    assert cost.estimated is True
    assert cost.price_version == "unpriced"


def test_an_unpriced_key_is_warned_once_per_key(caplog: pytest.LogCaptureFixture) -> None:
    """Catches decision C's "logged once per key" implemented as either extreme: no log at
    all, so a model that quietly books every request unpriced is invisible until someone
    reads the ledger; or a log per CALL, which on a hot alias is the same gap buried under
    thousands of identical lines nobody reads. Both keys are asked for TWICE and in an
    interleaved order, so a dedupe that only remembers the most recent key - the cheap
    wrong implementation - emits four records and fails on the count while a per-key one
    emits two. `_WARNED_KEYS` is process state, so the case clears it first rather than
    inheriting whatever earlier cases in this file left in it."""
    _WARNED_KEYS.clear()
    table = table_with(CURRENT_ENTRY)
    first, second = "prov-z/model-z", "prov-y/model-y"

    with caplog.at_level(logging.WARNING, logger="opsconsole.limits.cost"):
        for key in (first, second, first, second):
            cost_for(table, key, AT, usage=REPORTED)

    warned = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warned) == 2
    assert [record.key for record in warned] == [first, second]  # type: ignore[attr-defined]
    assert set(_WARNED_KEYS) == {first, second}
