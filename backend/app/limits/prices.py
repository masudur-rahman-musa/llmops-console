"""The price table: its schema, the effective-from selection and the snapshot the
gateway reads.

Each provider file in `backend/config/providers/` lists its models with their
prices; `app.model_routing.provider_files` reads them and builds this table,
keyed by `provider/model` - the
key `Adapter.price_key` builds from the POLICY provider name, so two rows of the
same provider kind priced differently stay apart. This schema is the boundary
where an operator's YAML becomes typed data.

Loading is deliberately strict. Unknown keys are rejected, and a negative price, a
non-finite price and an `effective_from` that is not a date are refused before
any caller holds a table; the provider-file loader turns each rejection into a
`PolicyInvalid` naming the file and the field path.

Prices are Decimals, and the shipped files QUOTE every one of them. Not because
an unquoted number is corrupted - pydantic converts a parsed float to Decimal
through its repr, so `0.0000015` written bare does survive exactly - but because
a float carries about seventeen significant digits and silently truncates
anything longer, and drops trailing zeros with it: bare `0.12345678901234567890`
loads as `0.12345678901234568`, and bare `1.00` loads as `Decimal("1.0")`. The
quoted form is the digits as written, every time, which is the property a price
wants. The schema accepts either. What
would really break the arithmetic is a `float` field, or a `Decimal` built from
the parsed float rather than from its repr.

Selection is by date and never by file order. `price_for` answers the newest
entry whose `effective_from` is on or before the instant being priced, so
re-pricing is appending an entry and never editing one, and a request served
last month keeps the number it was actually charged. The boundary is INCLUSIVE:
an entry dated exactly on that day is selected.

Nothing here raises on a request. An unknown key and a key whose entries are all
future-dated both answer `None`, which `app.limits.cost` turns into the
`unpriced` sentinel; a missing price must never be able to change a status code.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

PriceAmount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
"""A price per 1,000,000 tokens. Non-negative and finite, refused at load
otherwise: a negative price would credit an account and a non-finite one would
poison every sum it reached."""


class _Frozen(BaseModel):
    """Attributes are frozen and unknown keys rejected: a typo in the price table
    is an error, not a silently ignored line."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PriceEntry(_Frozen):
    """One price, in force from `effective_from` until a later entry supersedes it.

    Input and output are priced separately because every provider prices them
    separately, and at USD per 1,000,000 tokens because that is the unit vendors
    publish - converting at the boundary would mean the file no longer reads
    like the price page it was copied from.

    The two cache rates are optional. A row without them prices cache
    tokens at its input rate, so a provider that reports none needs no made-up
    numbers.
    """

    effective_from: date
    input_usd_per_1m: PriceAmount
    output_usd_per_1m: PriceAmount
    cache_read_usd_per_1m: PriceAmount | None = None
    cache_write_usd_per_1m: PriceAmount | None = None


class PriceTable(_Frozen):
    """A whole price table, validated as one document.

    A key maps to a LIST of entries rather than to one price, which is what makes
    the table versioned: history is appended, never overwritten. Both the table
    and each key's list are non-empty, because a table that prices nothing is a
    file someone forgot to fill in rather than a deliberate configuration.
    """

    prices: Mapping[str, Annotated[tuple[PriceEntry, ...], Field(min_length=1)]] = Field(
        min_length=1
    )


def price_for(table: PriceTable, key: str, at: date | datetime) -> PriceEntry | None:
    """The entry in force for `key` at `at`, or `None` when there is not one.

    `None` covers two different absences on purpose - a key the table does not
    carry, and a key whose entries are all still in the future - because the
    caller does the same thing with both: it books the request `unpriced` rather
    than raising. `app.limits.cost.cost_for` is that caller.

    The comparison is by DATE. `at` may be an instant or a day; a `datetime` is
    narrowed to its own date, so an entry dated on that day is in force from the
    first moment of it. Selection is by `effective_from` and never by file
    order.
    """
    entries = table.prices.get(key)
    if entries is None:
        return None

    as_of = at.date() if isinstance(at, datetime) else at
    in_force = [entry for entry in entries if entry.effective_from <= as_of]
    if not in_force:
        return None
    return max(in_force, key=lambda entry: entry.effective_from)


def price_version(key: str, entry: PriceEntry) -> str:
    """What the ledger records as the price this cost came from.

    Per-entry rather than a document-level version bumped by hand: the row then
    names the exact price it was charged at, and a later, unrelated key added to
    the file cannot restate what an old row was priced at.
    """
    return f"{key}@{entry.effective_from.isoformat()}"


class PriceStore:
    """Holds the price table the gateway costs against, and swaps it on reload.

    Mirrors `app.model_routing.policy.PolicyStore`, and for the same reason:
    `reload` builds a new `PriceTable` before it rebinds the attribute, so a
    failed reload leaves the previous snapshot in place and a pricing already
    holding a snapshot keeps reading a consistent one.

    The provider-file loader is imported inside the methods because it imports
    this module for `PriceEntry`; the table itself never needs the files.
    """

    def __init__(self, providers_dir: Path) -> None:
        self._providers_dir = providers_dir
        self._table = self._build()

    def _build(self) -> PriceTable:
        from app.model_routing.provider_files import build_price_table, load_provider_files

        return build_price_table(load_provider_files(self._providers_dir))

    @property
    def table(self) -> PriceTable:
        """The current snapshot."""
        return self._table

    def reload(self) -> None:
        """Re-read the folder and swap the snapshot, or raise and keep the old one."""
        self._table = self._build()
