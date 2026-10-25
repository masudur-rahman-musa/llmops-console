"""Loading, validating and selecting from the versioned price table.

Prices live in the provider files. The crafted cases here are still
written as one `prices:` document keyed `provider/model`, because that is the
shortest way to state a price; `load_prices` below splits such a document into
one provider file per provider and loads the folder through the real loader.

A bad price is an operator error, never a client error: every rejection case
here expects `PolicyInvalid` and a message naming the file and the offending
field path, exactly as `tests/model_routing/test_policy_load.py` does for the policy. The
selection cases pin the effective-from rule, which is the part of this module a
wrong implementation gets wrong silently - the wrong price is still a plausible
number, and nothing downstream can tell.

Every crafted fixture key here is a made-up `prov-a/model-a`, so no vendor model
id lives in this file; the shipped-file cases derive the real keys from
`routing.yaml` at runtime rather than restating them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import pytest
import yaml

from app.limits.prices import PriceStore, PriceTable, price_for
from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import (
    Policy,
    check_policy_models_priced,
    load_policy,
    reachable_price_keys,
)
from app.model_routing.provider_files import build_price_table, load_provider_files

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_PROVIDERS_DIR = REPO_ROOT / "config" / "providers"
SHIPPED_POLICY_PATH = REPO_ROOT / "config" / "routing.yaml"

KEY = "prov-a/model-a"
"""The one price key every crafted document below carries."""

PROVIDER_FILE = "prov-a.yaml"
MODEL = "model-a"
"""Where KEY's price lives once the document is split into provider files."""


def folder_of(path: Path) -> Path:
    """Split the `prices:` document at `path` into provider files beside it.

    A document that does not parse, or has no `prices` mapping, is written as is
    into `prov-a.yaml`, so the real loader is what refuses it.
    """
    folder = path.parent / "providers"
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*.yaml"):
        old.unlink()
    text = path.read_text(encoding="utf-8")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        document = None
    prices = document.get("prices") if isinstance(document, dict) else None
    if not isinstance(prices, dict):
        (folder / PROVIDER_FILE).write_text(text, encoding="utf-8")
        return folder
    files: dict[str, dict[str, Any]] = {}
    for key, entries in prices.items():
        provider, model = str(key).split("/", 1)
        file = files.setdefault(
            provider,
            {
                "provider": provider,
                "kind": "openai_compatible",
                "base_url": "http://x",
                "models": {},
            },
        )
        file["models"][model] = {"prices": entries}
    if not files:
        files["prov-a"] = {"provider": "prov-a", "kind": "openai_compatible", "models": {}}
    for provider, file in files.items():
        (folder / f"{provider}.yaml").write_text(
            yaml.safe_dump(file, sort_keys=False), encoding="utf-8"
        )
    return folder


def load_prices(path: Path) -> PriceTable:
    """The table the gateway would build from the document at `path`."""
    return build_price_table(load_provider_files(folder_of(path)))


def shipped_table() -> PriceTable:
    return build_price_table(load_provider_files(SHIPPED_PROVIDERS_DIR))


class PriceWriter(Protocol):
    """Writes a price document to `tmp_path` and returns the path."""

    def __call__(self, document: str, name: str = ...) -> Path: ...


@pytest.fixture
def price_file(tmp_path: Path) -> PriceWriter:
    """Write a price file verbatim.

    Raw text rather than a dumped mapping, because a project rule's trailing
    whitespace rows and the `!!python/object/apply` row are properties of the
    BYTES, and a round trip through `yaml.safe_dump` would erase all four.
    """

    def write(document: str, name: str = "prices.yaml") -> Path:
        path = tmp_path / name
        path.write_text(document, encoding="utf-8")
        return path

    return write


def valid_document(input_price: str = "3.00") -> str:
    """A minimal price document that loads clean.

    `input_price` is the single value the reload cases change, so a swapped
    snapshot is visible without rewriting the shape of the file.
    """
    return f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: "{input_price}"
      output_usd_per_1m: "15.00"
"""


class DocFactory(Protocol):
    """Builds the baseline valid price document, varying one pinned value."""

    def __call__(self, input_price: str = ...) -> str: ...


@pytest.fixture
def valid_doc() -> DocFactory:
    return valid_document


def instant(value: str) -> datetime:
    """A UTC instant from `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`."""
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


# a project rule: every rejection input below is tried with each of these
# appended, so a loader that strips or normalises the text before validating
# cannot pass three rows and fail the fourth unnoticed.
TRAILING: list[tuple[str, str]] = [
    ("bare", ""),
    ("trailing newline", "\n"),
    ("trailing crlf", "\r\n"),
    ("trailing space", " "),
]


# --- the loader ---------------------------------------------------------------


def test_safe_load_only(price_file: PriceWriter, tmp_path: Path) -> None:
    """Catches a loader written with `yaml.load` and the default `Loader`, which
    constructs the tagged python object and runs the shell command inside it; only
    `yaml.safe_load` refuses the tag, and only the wrapping turns that refusal into
    `PolicyInvalid` rather than a bare `yaml.YAMLError` reaching the caller."""
    marker = tmp_path / "pwned.txt"
    document = f"""
prices: !!python/object/apply:os.system ["touch {marker}"]
"""
    path = price_file(document)

    with pytest.raises(PolicyInvalid):
        load_prices(path)

    assert not marker.exists()


def test_a_missing_folder_is_policy_invalid(tmp_path: Path) -> None:
    """Catches a store that lets the folder's absence escape as a `FileNotFoundError`
    traceback, instead of the one typed error the load path reports."""
    with pytest.raises(PolicyInvalid):
        PriceStore(tmp_path / "does-not-exist")


NEGATIVE_INPUT_PRICE = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: "-0.01"
      output_usd_per_1m: "15.00"
"""

NEGATIVE_OUTPUT_PRICE = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: "3.00"
      output_usd_per_1m: "-0.01"
"""

INFINITE_INPUT_PRICE = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: .inf
      output_usd_per_1m: "15.00"
"""

NOT_A_NUMBER_OUTPUT_PRICE = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: "3.00"
      output_usd_per_1m: .nan
"""

# (case id, document, the fragments the message must name)
PRICE_REJECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    ("negative input price", NEGATIVE_INPUT_PRICE, (PROVIDER_FILE, MODEL, "input_usd_per_1m")),
    ("negative output price", NEGATIVE_OUTPUT_PRICE, (PROVIDER_FILE, MODEL, "output_usd_per_1m")),
    ("non finite input price", INFINITE_INPUT_PRICE, (PROVIDER_FILE, MODEL, "input_usd_per_1m")),
    (
        "not a number output price",
        NOT_A_NUMBER_OUTPUT_PRICE,
        (PROVIDER_FILE, MODEL, "output_usd_per_1m"),
    ),
]


@pytest.mark.parametrize(
    ("case", "document", "fragments"), PRICE_REJECTIONS, ids=[row[0] for row in PRICE_REJECTIONS]
)
@pytest.mark.parametrize(("trailing_id", "trailing"), TRAILING, ids=[row[0] for row in TRAILING])
def test_price_rejection_table(
    case: str,
    document: str,
    fragments: tuple[str, ...],
    trailing_id: str,
    trailing: str,
    price_file: PriceWriter,
) -> None:
    """Catches an unconstrained `Decimal` price field - one written without `ge=0` or
    without `allow_inf_nan=False`. A negative rate credits an account on every request it
    prices, and an infinity or a NaN poisons every sum it reaches, including the monthly
    total a budget is checked against; both are refused at LOAD here, before any caller
    holds a table, which is the only place they can still be refused without changing a
    served request's status code. Each direction has its own row because the constraint
    can be applied to one field and forgotten on the other, and each row runs against all
    four trailing forms."""
    path = price_file(document + trailing)

    with pytest.raises(PolicyInvalid) as excinfo:
        load_prices(path)

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


UNKNOWN_KEY = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: "3.00"
      output_usd_per_1m: "15.00"
      surcharge_usd: "1.00"
"""

EFFECTIVE_FROM_IS_NOT_A_DATE = f"""
prices:
  "{KEY}":
    - effective_from: "sometime in june"
      input_usd_per_1m: "3.00"
      output_usd_per_1m: "15.00"
"""

EMPTY_KEY_SET = """
prices: {}
"""

NO_ENTRIES_FOR_A_KEY = f"""
prices:
  "{KEY}": []
"""

SHAPE_REJECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    ("unknown key", UNKNOWN_KEY, (PROVIDER_FILE, MODEL, "surcharge_usd")),
    (
        "effective_from is not a date",
        EFFECTIVE_FROM_IS_NOT_A_DATE,
        (PROVIDER_FILE, MODEL, "effective_from"),
    ),
    ("empty key set", EMPTY_KEY_SET, (PROVIDER_FILE, "models")),
    ("a key with no entries", NO_ENTRIES_FOR_A_KEY, (PROVIDER_FILE, MODEL)),
]


@pytest.mark.parametrize(
    ("case", "document", "fragments"), SHAPE_REJECTIONS, ids=[row[0] for row in SHAPE_REJECTIONS]
)
@pytest.mark.parametrize(("trailing_id", "trailing"), TRAILING, ids=[row[0] for row in TRAILING])
def test_invalid_price_table_names_the_field(
    case: str,
    document: str,
    fragments: tuple[str, ...],
    trailing_id: str,
    trailing: str,
    price_file: PriceWriter,
) -> None:
    """Catches a schema built with `extra="allow"`, where a mistyped price key is silently
    ignored and the row it was meant to price books at the old rate forever; a `str`
    `effective_from`, which compares lexically and would put "sometime in june" in force
    from the beginning of time; and the two emptiness constraints (`prices: {}` and a key
    with no entries at all) whose absence turns a half-written file into a table that
    prices nothing and reports nothing. It also catches a loader that refuses correctly
    but reports only "validation error", leaving an operator to bisect a price file by
    hand. Every row runs against all four trailing forms."""
    path = price_file(document + trailing)

    with pytest.raises(PolicyInvalid) as excinfo:
        load_prices(path)

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


BROKEN_CASES: list[tuple[str, str]] = [
    ("unparseable yaml", "prices: [oops\n  - :::\n"),
    ("document is a list", "- prices\n"),
    ("document is a scalar", "just-a-string\n"),
    ("document is empty", ""),
]


@pytest.mark.parametrize(("case", "document"), BROKEN_CASES, ids=[row[0] for row in BROKEN_CASES])
def test_non_mapping_or_broken_document_is_refused(
    case: str, document: str, price_file: PriceWriter
) -> None:
    """Catches a loader that hands a list, a string or `None` straight to `model_validate`,
    where an empty price file becomes a Pydantic `Input should be a valid dictionary` at
    best and an `AttributeError` on `None` at worst - neither of which names the file that
    is actually wrong."""
    path = price_file(document)

    with pytest.raises(PolicyInvalid):
        load_prices(path)


AWKWARD_PRICE = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: 0.0000015
      output_usd_per_1m: "0.00"
"""
"""The price written UNQUOTED, which is the form that goes through YAML's float
parser: the quoted form cannot exercise a binary round trip at all."""

FLOAT_ROUND_TRIP = Decimal(float("0.0000015"))
"""What a `float` field - or a `Decimal` built from the parsed float with
`Decimal(value)` rather than `Decimal(str(value))` - holds for that same written
value. Built through `float()` rather than from a literal so the arrangement says
out loud where the noise comes from."""


def test_an_awkward_price_survives_the_round_trip_exactly(price_file: PriceWriter) -> None:
    """Catches a schema whose price field is `float`, or a `Decimal` built from the parsed
    float, which stores 0.00000150000000000000003800... instead of 0.0000015 and makes a
    month of ledger rows sum to a different total on every re-run. The unquoted form is
    deliberate: a quoted price never touches a float, so a quoted fixture would pass on
    the very implementation this exists to kill. The first assertion is the vacuity guard
    - the noisy value really is a different `Decimal` - which is also why a band or a
    `pytest.approx` could not stand in for the equality here."""
    path = price_file(AWKWARD_PRICE)

    table = load_prices(path)
    entry = table.prices[KEY][0]

    assert Decimal("0.0000015") != FLOAT_ROUND_TRIP
    assert entry.input_usd_per_1m == Decimal("0.0000015")
    assert entry.input_usd_per_1m != FLOAT_ROUND_TRIP


# --- effective-from selection -------------------------------------------------

ASCENDING = f"""
prices:
  "{KEY}":
    - effective_from: 2025-01-01
      input_usd_per_1m: "1.00"
      output_usd_per_1m: "2.00"
    - effective_from: 2025-06-01
      input_usd_per_1m: "3.00"
      output_usd_per_1m: "4.00"
"""

DESCENDING = f"""
prices:
  "{KEY}":
    - effective_from: 2025-06-01
      input_usd_per_1m: "3.00"
      output_usd_per_1m: "4.00"
    - effective_from: 2025-01-01
      input_usd_per_1m: "1.00"
      output_usd_per_1m: "2.00"
"""

ORDERINGS: list[tuple[str, str]] = [("ascending", ASCENDING), ("descending", DESCENDING)]


@pytest.mark.parametrize(("case", "document"), ORDERINGS, ids=[row[0] for row in ORDERINGS])
def test_file_order_does_not_decide_the_selection(
    case: str, document: str, price_file: PriceWriter
) -> None:
    """Catches a selector that takes `entries[0]` or `entries[-1]` - one that trusts the
    file to be sorted - which prices at the January rate for one ordering and at the June
    rate for the other, while both documents describe exactly the same table. Both
    orderings are run for precisely that reason: a positional implementation is right on
    one row and wrong on the other, so removing the date comparison is the only change
    that can flip both."""
    path = price_file(document)
    table = load_prices(path)

    entry = price_for(table, KEY, instant("2025-07-01"))

    assert entry is not None
    assert entry.effective_from == date(2025, 6, 1)
    assert entry.input_usd_per_1m == Decimal("3.00")
    assert entry.output_usd_per_1m == Decimal("4.00")


BOUNDARY_INSTANTS: list[tuple[str, str]] = [
    ("midnight on the effective date", "2025-06-01T00:00"),
    ("last minute of the effective date", "2025-06-01T23:59"),
]


@pytest.mark.parametrize(
    ("case", "moment"), BOUNDARY_INSTANTS, ids=[row[0] for row in BOUNDARY_INSTANTS]
)
def test_an_entry_effective_on_the_instants_own_date_is_selected(
    case: str, moment: str, price_file: PriceWriter
) -> None:
    """Catches a strict `entry.effective_from < at` comparison, and equally a `date`
    compared against a `datetime` without narrowing it: both keep billing the previous
    rate for the whole first day a new price is in force, at midnight and at 23:59 alike.
    The table carries an OLDER entry too, so a broken boundary returns that one rather
    than `None`, which means the all-future case below cannot be what reddens this."""
    path = price_file(ASCENDING)
    table = load_prices(path)

    entry = price_for(table, KEY, instant(moment))

    assert entry is not None
    assert entry.effective_from == date(2025, 6, 1)


def test_an_entry_effective_the_day_after_the_instant_is_not_selected(
    price_file: PriceWriter,
) -> None:
    """Catches an inclusive comparison that has slipped a day the other way, and a selector
    that ignores `at` altogether and simply answers the newest entry in the table: both
    bill tomorrow's price today, and both leave a re-priced month unreconcilable against
    the vendor's invoice. The instant is the last minute before the June entry takes
    effect, so only the January entry is legal."""
    path = price_file(ASCENDING)
    table = load_prices(path)

    entry = price_for(table, KEY, instant("2025-05-31T23:59"))

    assert entry is not None
    assert entry.effective_from == date(2025, 1, 1)
    assert entry.input_usd_per_1m == Decimal("1.00")


def test_a_key_whose_entries_are_all_future_dated_selects_nothing(
    price_file: PriceWriter,
) -> None:
    """Catches a selector that falls back to the oldest entry when nothing qualifies, which
    invents a price for a period the operator deliberately did not price and hides the gap
    behind a plausible number instead of the `unpriced` ledger row that would be found."""
    path = price_file(ASCENDING)
    table = load_prices(path)

    assert price_for(table, KEY, instant("2024-12-31")) is None


def test_a_key_the_table_does_not_carry_selects_nothing(price_file: PriceWriter) -> None:
    """Catches a `price_for` that indexes `table.prices[key]` and raises `KeyError`, which
    is the one behaviour decision C forbids: this runs on every served request, so a
    missing price must never be able to change a status code."""
    path = price_file(ASCENDING)
    table = load_prices(path)

    assert price_for(table, "prov-z/model-z", instant("2025-07-01")) is None


# --- the snapshot store -------------------------------------------------------


def test_store_rejects_a_bad_file_at_construction(price_file: PriceWriter) -> None:
    """Catches a store that defers loading to the first read, so a price file broken at
    deploy time starts the process cleanly and fails on the first billed request rather
    than at boot, where a rollout can still see it and stop."""
    path = price_file(NEGATIVE_INPUT_PRICE)

    with pytest.raises(PolicyInvalid):
        PriceStore(folder_of(path))


def test_reload_swaps_the_price_snapshot(price_file: PriceWriter, valid_doc: DocFactory) -> None:
    """Catches a `reload` that mutates the table in place rather than rebinding it: a
    pricing already holding the snapshot would then charge its input at one rate and its
    output at the next, and the two halves of one ledger row would name different prices.
    The last assertion is what only in-place mutation can break - the snapshot captured
    before the reload must still read 3.00."""
    path = price_file(valid_doc(input_price="3.00"))
    store = PriceStore(folder_of(path))
    before = store.table
    assert before.prices[KEY][0].input_usd_per_1m == Decimal("3.00")

    path.write_text(valid_document(input_price="9.00"), encoding="utf-8")
    folder_of(path)
    # `reload` returns None: call it, never assert on its result.
    store.reload()

    assert store.table.prices[KEY][0].input_usd_per_1m == Decimal("9.00")
    assert store.table is not before
    assert before.prices[KEY][0].input_usd_per_1m == Decimal("3.00")


def test_a_failed_reload_keeps_the_previous_price_table(
    price_file: PriceWriter, valid_doc: DocFactory
) -> None:
    """Catches a `reload` that clears or assigns the attribute before it validates, which
    turns one bad edit to the provider files into an unpriced ledger - or an exception on every
    request - until someone notices and fixes the file."""
    path = price_file(valid_doc(input_price="3.00"))
    store = PriceStore(folder_of(path))
    before = store.table

    path.write_text("prices: [oops\n  - :::\n", encoding="utf-8")
    folder_of(path)

    with pytest.raises(PolicyInvalid):
        store.reload()

    assert store.table is before
    assert store.table.prices[KEY][0].input_usd_per_1m == Decimal("3.00")


# --- the shipped file ---------------------------------------------------------


def test_the_shipped_price_file_loads() -> None:
    """Catches a the provider files shipped in a shape the schema refuses - the one file no
    test would otherwise validate, because an operator never edits it in a unit test -
    which would take the process down at boot on the first deploy that reads it."""
    assert SHIPPED_PROVIDERS_DIR.is_dir()

    table = shipped_table()

    assert len(table.prices) >= 1
    for entries in table.prices.values():
        assert len(entries) >= 1


def test_the_shipped_prices_cover_every_pair_the_shipped_routing_can_resolve() -> None:
    """Catches a model added to `routing.yaml` with no price row - the drift risk section 5
    names - which otherwise surfaces as `unpriced` rows on real traffic rather than at
    `make verify`. The key set is DERIVED from the loaded policy, aliases and per-tenant
    overrides both, so no hand-written list in this file can go stale behind a routing
    change; the non-emptiness assertion is what stops a derivation that silently returns
    nothing from passing this vacuously."""
    policy = load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR)
    table = shipped_table()

    reachable = reachable_price_keys(policy)

    assert reachable, "the derivation found no pairs, so this case would prove nothing"
    assert reachable <= set(table.prices), (
        f"unpriced pairs reachable from routing.yaml: {sorted(reachable - set(table.prices))}"
    )


# --- load-time coverage check ---


def _minimal_policy_with_override() -> Policy:
    """One alias with one candidate, plus one tenant whose override introduces a
    second `provider/model` pair that the alias candidates do not carry.

    The override pair is the whole reason `reachable_price_keys` must walk
    tenant overrides and not just alias candidates.
    """
    return Policy.model_validate(
        {
            "providers": {
                "prov-a": {
                    "kind": "openai_compatible",
                    "base_url": "http://x",
                },
                "prov-b": {
                    "kind": "openai_compatible",
                    "base_url": "http://x",
                },
            },
            "aliases": {
                "my-alias": {
                    "candidates": [
                        {"provider": "prov-a", "model": "model-x"},
                    ],
                },
            },
            "tenants": {
                "t1": {
                    "data_class_rules": {
                        "public": {},
                    },
                    "budget_usd_month": "10.00",
                    "rate_limit": {"rpm": 60, "tpm": 60000},
                    "overrides": {
                        "my-alias": [
                            {"provider": "prov-b", "model": "override-model"},
                        ],
                    },
                },
            },
        }
    )


def test_reachable_price_keys_covers_alias_and_override_keys() -> None:
    """Catches an implementation of `reachable_price_keys` that walks alias
    candidates only and skips tenant overrides, so the pair that is reachable
    ONLY through a tenant override is missing from the returned set."""
    policy = _minimal_policy_with_override()
    result = reachable_price_keys(policy)

    assert result == {"prov-a/model-x", "prov-b/override-model"}


def test_check_policy_models_priced_passes_when_every_routed_model_is_priced() -> None:
    """Catches an implementation of `check_policy_models_priced` that raises even
    when every reachable pair is present in the price table, making the app
    unbootable on a correctly priced policy."""

    policy = load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR)
    table = shipped_table()

    # Must not raise.
    check_policy_models_priced(policy, table)


def test_check_policy_models_priced_names_every_unpriced_model() -> None:
    """Catches an implementation of `check_policy_models_priced` that raises with
    only the first missing key, or with a message that omits the key strings
    entirely, making it impossible for an operator to find which model is
    unpriced without bisecting the policy file by hand."""
    policy = Policy.model_validate(
        {
            "providers": {
                "prov-a": {
                    "kind": "openai_compatible",
                    "base_url": "http://x",
                },
                "prov-b": {
                    "kind": "openai_compatible",
                    "base_url": "http://x",
                },
            },
            "aliases": {
                "my-alias": {
                    "candidates": [
                        {"provider": "prov-a", "model": "model-x"},
                        {"provider": "prov-b", "model": "model-y"},
                    ],
                },
            },
            "tenants": {
                "t1": {
                    "data_class_rules": {
                        "public": {},
                    },
                    "budget_usd_month": "10.00",
                    "rate_limit": {"rpm": 60, "tpm": 60000},
                },
            },
        }
    )
    # A price table that covers neither model the policy routes to.
    table = PriceTable.model_validate(
        {
            "prices": {
                "prov-a/some-other-model": [
                    {
                        "effective_from": "2025-01-01",
                        "input_usd_per_1m": "1.00",
                        "output_usd_per_1m": "2.00",
                    }
                ]
            }
        }
    )

    with pytest.raises(PolicyInvalid) as excinfo:
        check_policy_models_priced(policy, table)

    message = str(excinfo.value)
    assert "prov-a/model-x" in message
    assert "prov-b/model-y" in message


SELF_HOSTED_PROVIDERS = ["local_embed"]


@pytest.mark.parametrize("provider", SELF_HOSTED_PROVIDERS)
def test_a_self_hosted_provider_is_priced_at_exactly_zero(provider: str) -> None:
    """Catches decision D being dropped or softened: the self-hosted embedding server
    left unpriced, so every embedding books `unpriced`, or a
    self-hosted rate written as a per-token guess rather than the 0.0 its marginal API
    spend actually is. Asserted against the SHIPPED file, because that is the one a
    deployment bills from, and the presence assertion is what makes the zero check
    non-vacuous on a file that simply omitted the provider."""
    table = shipped_table()

    matching = [key for key in table.prices if key.split("/", 1)[0] == provider]

    assert matching, f"the shipped price file carries no {provider} key at all"
    for key in matching:
        for entry in table.prices[key]:
            assert entry.input_usd_per_1m == Decimal("0")
            assert entry.output_usd_per_1m == Decimal("0")
