r"""`evals/tradeoff.py`: the manifest, the two cost formulas, the winner rule, the two files.

Pure: no backing service, no clock, no network. Every case writes small JSON
documents under `tmp_path` and reads two rendered strings back, so the whole file
runs well under a second.

The contract these cases are written against, before it exists:

* the constants a project rule requires pinned exactly - `TRADEOFF_MD_NAME`,
  `TRADEOFF_JSON_NAME`, `OPTION_ID_PATTERN`, `MIN_OPTIONS`, `COST_SCALE`,
  `ACCURACY_TOLERANCE`, `TURNS_PER_1000`, `KIND_API`, `KIND_SELF_HOSTED`, `NA`,
  `METHODOLOGY_HEADING`, `CONCLUSION_HEADING`, `NO_WINNER_LINE`,
  `SELF_HOSTING_LOST_CLAUSE`, `LEDGER_NOTE`, `ROW_METRIC_NAMES` and
  `COST_METRIC_NAME`;
* `TradeoffOption`, `TradeoffManifest`, `TradeoffRow` and `TradeoffTable`, all
  frozen and `extra="forbid"`;
* `load_manifest(path) -> TradeoffManifest` and
  `build_table(manifest, base_dir=...) -> TradeoffTable`, every refusal arriving
  as `TradeoffInputInvalid(reason, path=path)` whose `str` is `f"{path}: {reason}"`
  - `gate.py`'s `GateInputInvalid` shape;
* `tradeoff_json(table) -> str`, `tradeoff_markdown(table) -> str` and
  `write_tradeoff(out, table) -> None`.

Two things the rows are arranged against, so that no case passes because a
NEIGHBOUR refused it first. Every report this file builds is a valid `RunReport`
whose timestamps are 30 minutes apart, whose five read metrics are five DIFFERENT
numbers (90.0, 95.0, 98.0, 1840, 2310) and whose costs are strings, so the only
thing that can refuse a document is the clause the case broke. And the winner
cases put the cheapest option OUTSIDE the accuracy band or the most accurate one
at the highest price, so a table that answered "most accurate" or "first row"
cannot pass them.

Two reading conventions the module owes, stated here because the cases assert
them: the markdown table's FIRST cell in an option's row is `option.id` (the plan's
"that id names the markdown row"), and `## Methodology` and `## Conclusion` each
appear exactly once as whole lines.

The accuracy numbers are in POINTS (0-100), which is what `metrics.py` emits,
while `ACCURACY_TOLERANCE` is the literal `Decimal("0.02")` the plan pins. The
band cases are built directly on that literal and not on a rescaling of it. What
they pin is the INCLUSIVE edge - 0.02 below the best is in, 0.0201 below is out -
which is the only thing that separates a cheap option the table recommends from
one it does not.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.evals.scores import METRIC_NAMES
from app.evals.tradeoff import (
    ACCURACY_TOLERANCE,
    CONCLUSION_HEADING,
    COST_METRIC_NAME,
    COST_SCALE,
    KIND_API,
    KIND_SELF_HOSTED,
    LEDGER_NOTE,
    METHODOLOGY_HEADING,
    MIN_OPTIONS,
    NA,
    NO_WINNER_LINE,
    OPTION_ID_PATTERN,
    ROW_METRIC_NAMES,
    SELF_HOSTING_LOST_CLAUSE,
    TRADEOFF_JSON_NAME,
    TRADEOFF_MD_NAME,
    TURNS_PER_1000,
    TradeoffInputInvalid,
    TradeoffOption,
    TradeoffTable,
    build_table,
    load_manifest,
    tradeoff_json,
    tradeoff_markdown,
    write_tradeoff,
)

FIXTURES: Path = Path(__file__).parent / "fixtures" / "tradeoff"
BASE_REPORT: Mapping[str, Any] = json.loads(
    (FIXTURES / "base_report.json").read_text(encoding="utf-8")
)
"""One valid `RunReport` document, read once at module scope and never mutated.

A 200-case report as a checked-in file would be absurd, so every case builds the
document it needs from this one with `report_document` and `case_rows`."""

SET_NAME: str = BASE_REPORT["set_name"]
MANIFEST_NAME: str = "tradeoff-manifest.json"

BASE_ACCURACY: float = 90.0
"""The base report's `decision_accuracy`, in points, and the top of every band."""

READ_METRICS: dict[str, float | int] = {
    "decision_accuracy": 90.0,
    "refusal_recall": 95.0,
    "citation_accuracy": 98.0,
    "p50_latency_ms": 1840,
    "p95_latency_ms": 2310,
}
"""The five values the base report carries under the five names a row reads.

No two are equal, so a renderer that fills `p95_latency_ms` from `p50_latency_ms`
or reads `refusal_precision` where it meant `refusal_recall` is visible."""


def report_document(base: Mapping[str, Any], /, **overrides: object) -> dict[str, Any]:
    """A copy of `base` under `overrides`, with `metrics` MERGED rather than replaced.

    Positional-only first argument: every other name belongs to
    the report document, so a helper taking both a named keyword and `**overrides`
    would silently swallow a report key that collided with it.
    """
    document: dict[str, Any] = json.loads(json.dumps(base))
    metrics = overrides.pop("metrics", None)
    if isinstance(metrics, Mapping):
        document["metrics"] = {**document["metrics"], **metrics}
    document.update(overrides)
    return document


def case_rows(count: int) -> list[dict[str, Any]]:
    """`count` case rows, each the base report's one case under its own `case_id`."""
    template: Mapping[str, Any] = BASE_REPORT["cases"][0]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = json.loads(json.dumps(template))
        row["case_id"] = f"case-{index}"
        row["assertions"]["case_id"] = row["case_id"]
        row["residency"]["case_id"] = row["case_id"]
        rows.append(row)
    return rows


def report_id_for(option_id: str) -> str:
    """A distinct 32-character run id per option, so two rows cannot share one."""
    return option_id.replace("-", "")[:32].ljust(32, "0")


def option_document(**overrides: object) -> dict[str, Any]:
    """One manifest option. `report` defaults to `<id>.json` beside the manifest."""
    document: dict[str, Any] = {
        "id": "cheap-api",
        "label": "API cheap",
        "kind": KIND_API,
    }
    document.update(overrides)
    document.setdefault("report", f"{document['id']}.json")
    return document


def api_pair(
    option_id: str,
    *,
    cost_per_turn: str | None,
    accuracy: float | None = BASE_ACCURACY,
    label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """An `api` option and the report it names."""
    option = option_document(id=option_id, label=label or f"label of {option_id}")
    report = report_document(
        BASE_REPORT,
        id=report_id_for(option_id),
        metrics={COST_METRIC_NAME: cost_per_turn, "decision_accuracy": accuracy},
    )
    return option, report


def self_hosted_pair(
    option_id: str,
    *,
    hourly_usd: str,
    turns: int,
    accuracy: float | None = BASE_ACCURACY,
    label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A `self_hosted` option and a report of `turns` cases over the base 30 minutes."""
    option = option_document(
        id=option_id,
        label=label or f"label of {option_id}",
        kind=KIND_SELF_HOSTED,
        hourly_usd=hourly_usd,
    )
    report = report_document(
        BASE_REPORT,
        id=report_id_for(option_id),
        cases=case_rows(turns),
        metrics={"decision_accuracy": accuracy},
    )
    return option, report


def write_json(path: Path, document: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def write_inputs(
    root: Path,
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    set_name: str = SET_NAME,
    manifest_name: str = MANIFEST_NAME,
) -> Path:
    """Write every option's report and the manifest naming them; answer the manifest path.

    A pair whose report is `None` has no file written, which is how the missing-report
    cases are built.
    """
    for option, report in pairs:
        if report is not None:
            write_json(root / str(option["report"]), report)
    return write_json(
        root / manifest_name,
        {"set_name": set_name, "options": [option for option, _ in pairs]},
    )


def table_for(
    root: Path,
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    set_name: str = SET_NAME,
) -> TradeoffTable:
    """The table those inputs build, resolving reports against the manifest's directory."""
    path = write_inputs(root, pairs, set_name=set_name)
    return build_table(load_manifest(path), base_dir=path.parent)


def json_document(table: TradeoffTable) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(tradeoff_json(table))
    return document


def markdown_row(markdown: str, option_id: str) -> tuple[str, ...]:
    """The one table row whose FIRST cell is `option_id`, as stripped cells.

    Every table line is scanned and exactly one match is required (standing rule
    44): a row located as "the first line containing the id" would match the
    methodology block's mention of the same id.
    """
    rows = [cells for cells in _table_rows(markdown) if cells[0] == option_id]
    assert len(rows) == 1, f"expected exactly one table row for {option_id!r}, got {len(rows)}"
    return rows[0]


def markdown_order(markdown: str, option_ids: Sequence[str]) -> list[str]:
    """The ids of the table's rows, in the order they are rendered."""
    wanted = set(option_ids)
    return [cells[0] for cells in _table_rows(markdown) if cells[0] in wanted]


def _table_rows(markdown: str) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        rows.append(tuple(cell.strip() for cell in stripped.strip("|").split("|")))
    return rows


def section(markdown: str, heading: str) -> str:
    """The text under `heading`, up to the next `## ` line. The heading must be unique."""
    lines = markdown.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == heading]
    assert len(starts) == 1, f"expected exactly one {heading!r} line, got {len(starts)}"
    start = starts[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def carries_number(text: str, value: Decimal) -> bool:
    """True when some numeric token in `text` equals `value`, however it was formatted."""
    return any(Decimal(token) == value for token in re.findall(r"\d+(?:\.\d+)?", text))


def names_option(conclusion: str, option: Mapping[str, Any]) -> bool:
    """True when the conclusion names this option, by its id or by its label."""
    return str(option["id"]) in conclusion or str(option["label"]) in conclusion


# --------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------


def test_the_two_file_names_and_the_three_pinned_strings_are_the_published_ones() -> None:
    """Catches an implementation that timestamps an output name or lets a caller shape one,
    which would leave `make tradeoff` and the console reading a path that is never there,
    and catches a renamed heading, which would leave a reader sent to `## Methodology`
    looking at a document that has no such block."""
    assert TRADEOFF_MD_NAME == "tradeoff.md"
    assert TRADEOFF_JSON_NAME == "tradeoff.json"
    assert NA == "n/a"
    assert METHODOLOGY_HEADING == "## Methodology"
    assert CONCLUSION_HEADING == "## Conclusion"
    assert KIND_API == "api"
    assert KIND_SELF_HOSTED == "self_hosted"


def test_the_two_conclusion_literals_and_the_ledger_note_are_the_published_ones() -> None:
    """Catches a conclusion that quietly stops saying self-hosting lost, or that reports a
    no-winner table as a winner-less sentence nobody pinned: both are the claims a reader
    of the published table quotes, and an implementation free to reword them is free to
    flatter self-hosting."""
    assert NO_WINNER_LINE == (
        "No option carries both a decision accuracy and a cost, so this table names no winner."
    )
    assert SELF_HOSTING_LOST_CLAUSE == (
        "Self-hosting did not win here: the cheapest option inside the band is an API."
    )
    assert LEDGER_NOTE == (
        "API cost per turn comes from the ledger's recorded spend for the run, "
        "not from a list price."
    )


def test_the_bounds_and_the_scales_are_the_published_numbers() -> None:
    """Catches a tolerance widened from the plan's 0.02 - which would let a materially
    less accurate option win on price - and a cost quantised at 2 places, which rounds a
    sub-cent per-turn price to a number that is not the one the ledger recorded."""
    assert MIN_OPTIONS == 2
    assert COST_SCALE == 4
    assert Decimal("0.02") == ACCURACY_TOLERANCE
    assert isinstance(ACCURACY_TOLERANCE, Decimal)
    assert Decimal(1000) == TURNS_PER_1000
    assert isinstance(TURNS_PER_1000, Decimal)


def test_the_option_id_pattern_is_the_published_one() -> None:
    """Catches a widened or retyped pattern: this id keys the JSON document's option object
    and names a markdown row, so a pattern admitting a slash or a dot is the store-shaping
    input of this plan going unchecked."""
    assert OPTION_ID_PATTERN.pattern == r"[a-z][a-z0-9-]{0,31}"


def test_the_five_row_metrics_and_the_cost_metric_are_members_of_metric_names() -> None:
    """Catches the rename this module cannot otherwise see: a name dropped or spelled
    differently in `metrics.py` would leave the column silently empty in every published
    table, because a missing key reads exactly like a metric with no cases behind it."""
    assert ROW_METRIC_NAMES == (
        "decision_accuracy",
        "refusal_recall",
        "citation_accuracy",
        "p50_latency_ms",
        "p95_latency_ms",
    )
    for name in ROW_METRIC_NAMES:
        assert name in METRIC_NAMES
    assert COST_METRIC_NAME == "cost_usd_per_turn"
    assert COST_METRIC_NAME in METRIC_NAMES


# --------------------------------------------------------------------------
# option.id: the typed pre-check, before any file is read
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option_id",
    ["a", "a" + "b" * 31, "a1-b2", "a" * 32],
    ids=["one character", "exactly 32 characters", "digits and hyphens", "32 letters"],
)
def test_an_option_id_inside_the_pattern_is_accepted(option_id: str) -> None:
    """Catches a bound written exclusive - a 32-character id refused, or a 1-character id
    refused by a pattern requiring two - which would refuse a manifest the plan's
    acceptance criteria say is legal."""
    option = TradeoffOption.model_validate(option_document(id=option_id))

    assert option.id == option_id


@pytest.mark.parametrize(
    "option_id",
    [
        "a" + "b" * 32,
        "1abc",
        "Abc",
        "a_b",
        "",
        "-abc",
        "abc\n",
        "abc\r\n",
        "a b",
        "a/b",
    ],
    ids=[
        "33 characters",
        "a leading digit",
        "an uppercase letter",
        "an underscore",
        "empty",
        "a leading hyphen",
        "a trailing newline",
        "a CRLF",
        "a space",
        "a path separator",
    ],
)
def test_an_option_id_outside_the_pattern_is_refused(option_id: str) -> None:
    """Catches `^...$` with `.match` instead of `re.fullmatch`: in Python
    `$` also matches before a trailing newline, so `abc\\n` would pass a check that looks
    right and land a newline in the JSON document's key and in a markdown row."""
    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(option_document(id=option_id))


def test_an_option_id_is_refused_before_any_report_file_is_read(tmp_path: Path) -> None:
    """Catches an id checked after the reports are loaded: no report file exists here at
    all, so an implementation reading first would refuse the MISSING FILE and never reach
    the id. The message is asserted to name the id and not the report path, which is the
    only way to tell the two refusals apart."""
    first = option_document(id="Bad-Id", report="first.json")
    second = option_document(id="second-api", report="second.json")

    path = write_inputs(tmp_path, [(first, None), (second, None)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        load_manifest(path)

    assert "Bad-Id" in str(raised.value)
    assert "first.json" not in str(raised.value)
    assert str(path) in str(raised.value)


def test_two_options_sharing_an_id_are_refused_and_the_message_names_it(tmp_path: Path) -> None:
    """Catches a manifest read into a dict keyed by id, where the second option silently
    replaces the first and the published table quietly loses a row - and catches a refusal
    that says only "duplicate", leaving the operator to diff the file by eye."""
    first, first_report = api_pair("twin-api", cost_per_turn="0.0010")
    second, second_report = api_pair("twin-api", cost_per_turn="0.0020")
    second["report"] = "second.json"

    path = write_inputs(tmp_path, [(first, first_report), (second, second_report)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        load_manifest(path)

    assert "twin-api" in str(raised.value)


@pytest.mark.parametrize("count", [0, 1], ids=["no options", "one option"])
def test_a_manifest_of_fewer_than_two_options_is_refused(tmp_path: Path, count: int) -> None:
    """Catches a builder that renders a table of one: a comparison with nothing to compare
    against publishes a winner that beat no one, which is the claim the design exists to
    refuse."""
    pairs = [api_pair(f"only-{index}", cost_per_turn="0.0010") for index in range(count)]

    path = write_inputs(tmp_path, [(option, report) for option, report in pairs])
    with pytest.raises(TradeoffInputInvalid):
        load_manifest(path)


# --------------------------------------------------------------------------
# kind and hourly_usd: the pairing, and the Decimal (never a float)
# --------------------------------------------------------------------------


def test_a_self_hosted_option_without_an_hourly_price_is_refused() -> None:
    """Catches a self-hosted row whose cost is computed from a missing price as zero or
    null: a GPU that appears to cost nothing is exactly the flattering error risk 1 names."""
    document = option_document(id="gpu", kind=KIND_SELF_HOSTED)

    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(document)


def test_an_api_option_carrying_an_hourly_price_is_refused() -> None:
    """Catches the other direction, where an operator prices an API by the hour and the
    builder reads that price for nothing: the published number would then disagree with the
    ledger's own per-turn cost with nothing in the document saying which was used."""
    document = option_document(id="cheap-api", kind=KIND_API, hourly_usd="1.006")

    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(document)


@pytest.mark.parametrize(
    "hourly_usd",
    ["0", "0.0000", "-1.00", "-0.0001"],
    ids=["zero", "zero with places", "negative", "a negative fraction"],
)
def test_an_hourly_price_that_is_not_above_zero_is_refused(hourly_usd: object) -> None:
    """Catches a `>= 0` bound where the plan says above zero: a price of 0 renders a
    self-hosted cost of 0.0000 per 1,000 turns and wins every table forever."""
    document = option_document(id="gpu", kind=KIND_SELF_HOSTED, hourly_usd=hourly_usd)

    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(document)


@pytest.mark.parametrize(
    "hourly_usd", [1.006, 1, 0.1], ids=["a JSON float", "a JSON int", "a small float"]
)
def test_an_hourly_price_given_as_a_json_number_is_refused(hourly_usd: object) -> None:
    """Catches `Decimal(float)` entering through the manifest: 1.006 as a binary float is
    1.00599999999999978, and a per-1,000-turn cost built on it drifts in the fourth place
    the table publishes. Only a JSON STRING may price an option."""
    document = option_document(id="gpu", kind=KIND_SELF_HOSTED, hourly_usd=hourly_usd)

    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(document)


def test_an_hourly_price_string_arrives_as_an_exact_decimal() -> None:
    """Catches a price parsed through `float` and back: the accepted value has to compare
    equal to `Decimal("1.006")` exactly, which a round trip through a binary float does
    not."""
    option = TradeoffOption.model_validate(
        option_document(id="gpu", kind=KIND_SELF_HOSTED, hourly_usd="1.006")
    )

    assert option.hourly_usd == Decimal("1.006")
    assert isinstance(option.hourly_usd, Decimal)


@pytest.mark.parametrize(
    "kind",
    ["local", "gpu", "API", "self-hosted", ""],
    ids=["local", "gpu", "API", "hyphen", "empty"],
)
def test_a_kind_outside_the_two_words_is_refused(kind: str) -> None:
    """Catches a third kind reaching the cost branch, where neither formula applies and the
    row would carry a cost computed by whichever branch happened to be the `else`."""
    document = option_document(id="thing", kind=kind)

    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(document)


def test_an_unknown_key_on_an_option_is_refused() -> None:
    """Catches a manifest read past its schema: an operator who writes `hourly_cost` for
    `hourly_usd` would otherwise get a table priced from nothing, with the typo sitting in
    the file looking applied."""
    document = option_document(id="gpu", kind=KIND_SELF_HOSTED, hourly_usd="1.006", price="9")

    with pytest.raises(ValidationError):
        TradeoffOption.model_validate(document)


# --------------------------------------------------------------------------
# load_manifest: three shapes of unusable file, one typed refusal
# --------------------------------------------------------------------------


def test_a_manifest_path_that_does_not_exist_is_refused_and_the_message_names_it(
    tmp_path: Path,
) -> None:
    """Catches an `OSError` reaching the operator as a traceback instead of one line: the
    CLI turns this exception into stderr and exit 2, and an unhandled `FileNotFoundError`
    would exit 1, which a caller reads as a real verdict."""
    missing = tmp_path / "nowhere.json"

    with pytest.raises(TradeoffInputInvalid) as raised:
        load_manifest(missing)

    assert str(missing) in str(raised.value)


def test_a_manifest_that_is_not_json_is_refused_as_the_same_typed_error(tmp_path: Path) -> None:
    """Catches a `JSONDecodeError` escaping: a half-written manifest is the same operator
    error as a mistyped path and has to arrive at the CLI as one class, or the CLI needs a
    second except clause nobody wrote."""
    path = tmp_path / MANIFEST_NAME
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(TradeoffInputInvalid) as raised:
        load_manifest(path)

    assert str(path) in str(raised.value)


@pytest.mark.parametrize(
    "document",
    [[], {"options": []}, {"set_name": "s"}, {"set_name": "s", "options": {}}, "a string"],
    ids=["a list", "no set_name", "no options", "options is a mapping", "a bare string"],
)
def test_valid_json_that_is_not_a_manifest_is_refused(tmp_path: Path, document: object) -> None:
    """Catches a pydantic `ValidationError` escaping the loader, which would reach the
    operator as a multi-line traceback where the contract says one line on stderr."""
    path = write_json(tmp_path / MANIFEST_NAME, document)

    with pytest.raises(TradeoffInputInvalid) as raised:
        load_manifest(path)

    assert str(path) in str(raised.value)


# --------------------------------------------------------------------------
# build_table: the report refusals, each naming the report path
# --------------------------------------------------------------------------


def test_a_missing_report_is_refused_and_the_message_names_the_report_path(
    tmp_path: Path,
) -> None:
    """Catches a refusal that names the MANIFEST instead of the report: with four options in
    a manifest, the path the operator has to fix is the one that was not there, and the
    manifest's own path tells them nothing."""
    good, good_report = api_pair("good-api", cost_per_turn="0.0010")
    absent, _ = api_pair("absent-api", cost_per_turn="0.0020")

    path = write_inputs(tmp_path, [(good, good_report), (absent, None)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        build_table(load_manifest(path), base_dir=path.parent)

    assert str(tmp_path / "absent-api.json") in str(raised.value)


@pytest.mark.parametrize(
    "document",
    ["not json at all", None, {"id": "0123456789abcdef0123456789abcdef"}],
    ids=["unparseable", "a null document", "a document from another schema"],
)
def test_a_report_that_is_not_a_run_report_is_refused_naming_its_path(
    tmp_path: Path, document: object
) -> None:
    """Catches a loader that picks the five metric keys out of whatever it was handed: a
    hand-written file carrying `decision_accuracy` and nothing else would then be published
    as a measured run. The document is validated through `RunReport` itself."""
    good, good_report = api_pair("good-api", cost_per_turn="0.0010")
    broken, _ = api_pair("broken-api", cost_per_turn="0.0020")
    broken_path = tmp_path / str(broken["report"])
    if document == "not json at all":
        broken_path.write_text("{oops", encoding="utf-8")
    else:
        write_json(broken_path, document)

    path = write_inputs(tmp_path, [(good, good_report), (broken, None)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        build_table(load_manifest(path), base_dir=path.parent)

    assert str(broken_path) in str(raised.value)


def test_a_report_for_another_golden_set_is_refused_and_the_message_names_both_names(
    tmp_path: Path,
) -> None:
    """Catches the comparison this whole table exists to make being made across two
    different sets: an API scored on an easy set against a GPU scored on a hard one is not
    a tradeoff, it is two unrelated runs in one table. Both names are asserted because the
    operator has to see which file is the odd one."""
    good, good_report = api_pair("good-api", cost_per_turn="0.0010")
    other, other_report = api_pair("other-api", cost_per_turn="0.0020")
    other_report["set_name"] = "helpdesk-golden-v2.jsonl"

    path = write_inputs(tmp_path, [(good, good_report), (other, other_report)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        build_table(load_manifest(path), base_dir=path.parent)

    message = str(raised.value)
    assert SET_NAME in message
    assert "helpdesk-golden-v2.jsonl" in message
    assert str(tmp_path / "other-api.json") in message


@pytest.mark.parametrize(
    "finished_at",
    ["2026-05-01T10:00:00Z", "2026-05-01T09:59:59Z"],
    ids=["equal to started_at", "before started_at"],
)
def test_a_report_whose_run_did_not_move_forward_in_time_is_refused(
    tmp_path: Path, finished_at: str
) -> None:
    """Catches the ZeroDivisionError and the negative rate: elapsed hours divides the case
    count, so an equal pair of timestamps crashes the build and a reversed pair publishes a
    negative throughput and a negative self-hosted cost that wins every table."""
    good, good_report = api_pair("good-api", cost_per_turn="0.0010")
    still, still_report = self_hosted_pair("still-gpu", hourly_usd="1.006", turns=200)
    still_report["finished_at"] = finished_at

    path = write_inputs(tmp_path, [(good, good_report), (still, still_report)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        build_table(load_manifest(path), base_dir=path.parent)

    assert str(tmp_path / "still-gpu.json") in str(raised.value)


def test_a_self_hosted_option_whose_report_holds_no_cases_is_refused(tmp_path: Path) -> None:
    """Catches the division by zero the other way round: `hourly_usd / turns_per_hour` where
    the run answered nothing is a crash, and a builder that guarded it with a null would
    publish a GPU row with no cost and leave the reader to guess why."""
    good, good_report = api_pair("good-api", cost_per_turn="0.0010")
    empty, empty_report = self_hosted_pair("empty-gpu", hourly_usd="1.006", turns=0)

    path = write_inputs(tmp_path, [(good, good_report), (empty, empty_report)])
    with pytest.raises(TradeoffInputInvalid) as raised:
        build_table(load_manifest(path), base_dir=path.parent)

    assert str(tmp_path / "empty-gpu.json") in str(raised.value)


# --------------------------------------------------------------------------
# where a report path resolves from
# --------------------------------------------------------------------------


def test_a_relative_report_path_resolves_against_the_manifests_own_directory(
    tmp_path: Path,
) -> None:
    """Catches a path resolved against the process's current directory: the manifest is the
    artifact 27c commits beside its reports, so a relative entry has to mean "beside the
    manifest" or the same file builds on a laptop and fails in the image."""
    root = tmp_path / "evidence"
    first, first_report = api_pair("first-api", cost_per_turn="0.0010")
    second, second_report = api_pair("second-api", cost_per_turn="0.0020")
    first["report"] = "runs/first.json"
    second["report"] = "runs/second.json"

    table = table_for(root, [(first, first_report), (second, second_report)])

    assert [row.id for row in table.rows] == ["first-api", "second-api"]


def test_an_absolute_report_path_is_used_as_it_stands(tmp_path: Path) -> None:
    """Catches a builder that joins `base_dir` onto an absolute path anyway, which on a
    manifest naming `/evidence/run/report.json` produces a path under the manifest's
    directory that does not exist."""
    elsewhere = tmp_path / "elsewhere"
    root = tmp_path / "manifests"
    first, first_report = api_pair("first-api", cost_per_turn="0.0010")
    second, second_report = api_pair("second-api", cost_per_turn="0.0020")
    write_json(elsewhere / "second.json", second_report)
    second["report"] = str(elsewhere / "second.json")

    table = table_for(root, [(first, first_report), (second, None)])

    assert [row.id for row in table.rows] == ["first-api", "second-api"]


# --------------------------------------------------------------------------
# the arithmetic: Decimal end to end (risk 1, risk 2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cost_per_turn", "expected"),
    [
        ("0.0012345", "1.2345"),
        ("0.00012345", "0.1235"),
        ("0.00012344", "0.1234"),
        ("0.0100", "10.0000"),
    ],
    ids=["the plan's worked case", "half rounds up", "below half rounds down", "trailing zeros"],
)
def test_an_api_cost_per_1000_turns_is_the_ledgers_per_turn_cost_times_1000(
    tmp_path: Path, cost_per_turn: str, expected: str
) -> None:
    """Catches the two ways this number goes wrong: a float multiplication, which renders
    1.2344999999999999 for the plan's worked case, and ROUND_HALF_EVEN, which renders
    0.1234 where 0.00012345 must round up to 0.1235. The expected strings are hand-computed
    and never a recomputation of the formula (risk 1)."""
    priced, priced_report = api_pair("priced-api", cost_per_turn=cost_per_turn)
    other, other_report = api_pair("other-api", cost_per_turn="0.5000")

    table = table_for(tmp_path, [(priced, priced_report), (other, other_report)])
    document = json_document(table)

    assert document["options"]["priced-api"]["usd_per_1000_turns"] == expected
    assert expected in markdown_row(tradeoff_markdown(table), "priced-api")


def test_the_self_hosted_cost_is_the_hourly_price_over_the_measured_throughput(
    tmp_path: Path,
) -> None:
    """The plan's worked case: 200 cases over exactly 30 minutes is 400 turns an hour, and
    1.006 an hour over 400 turns an hour is 2.5150 per 1,000 turns. Catches the formula
    inverted (turns per hour over the price), the per-hour price published as the per-1,000
    figure, and elapsed time read in seconds or minutes rather than hours - each of which
    flatters or punishes the GPU by orders of magnitude."""
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200)
    other, other_report = api_pair("other-api", cost_per_turn="0.5000")

    table = table_for(tmp_path, [(gpu, gpu_report), (other, other_report)])
    document = json_document(table)

    assert document["options"]["gpu"]["usd_per_1000_turns"] == "2.5150"
    assert Decimal(document["options"]["gpu"]["turns_per_hour"]) == Decimal(400)
    assert document["options"]["gpu"]["turns"] == 200


def test_the_turn_count_is_the_reports_case_rows_and_never_a_number_in_its_metrics(
    tmp_path: Path,
) -> None:
    """Catches a builder reading `metrics["cases"]`: the report here carries 7 there and 200
    real case rows, so a throughput built on the metrics block reports 14 turns an hour
    instead of 400 and prices the GPU at roughly 71.86 per 1,000 turns."""
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200)
    gpu_report["metrics"]["cases"] = 7
    other, other_report = api_pair("other-api", cost_per_turn="0.5000")

    table = table_for(tmp_path, [(gpu, gpu_report), (other, other_report)])
    document = json_document(table)

    assert document["options"]["gpu"]["turns"] == 200
    assert document["options"]["gpu"]["usd_per_1000_turns"] == "2.5150"


def test_every_published_cost_is_a_json_string_and_never_a_number(tmp_path: Path) -> None:
    """Catches a Decimal serialised through `float`, which publishes 0.30000000000000004 for
    a cost the ledger computed exactly. The console reads this document, so the type is part
    of the contract and not a formatting preference (risk 2)."""
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200)
    api, api_report = api_pair("cheap-api", cost_per_turn="0.0012345")

    table = table_for(tmp_path, [(gpu, gpu_report), (api, api_report)])
    document = json_document(table)

    for option_id in ("gpu", "cheap-api"):
        published = document["options"][option_id]
        assert isinstance(published["usd_per_1000_turns"], str)
        assert isinstance(published["turns_per_hour"], str)
    assert isinstance(document["options"]["gpu"]["hourly_usd"], str)


# --------------------------------------------------------------------------
# the five metrics, and what a null one renders as (decision F)
# --------------------------------------------------------------------------


def test_each_row_carries_the_five_metrics_under_their_own_names(tmp_path: Path) -> None:
    """Catches a column filled from the neighbouring metric - `p95_latency_ms` read off
    `p50_latency_ms`, or `refusal_precision` read where `refusal_recall` was meant. The five
    values in the fixture are five different numbers, so any such swap changes a published
    cell."""
    first, first_report = api_pair("first-api", cost_per_turn="0.0010")
    second, second_report = api_pair("second-api", cost_per_turn="0.0020")

    table = table_for(tmp_path, [(first, first_report), (second, second_report)])
    document = json_document(table)

    assert dict(table.rows[0].metrics) == READ_METRICS
    assert document["options"]["first-api"]["metrics"] == READ_METRICS
    assert tuple(document["options"]["first-api"]["metrics"]) == ROW_METRIC_NAMES


def test_a_null_metric_renders_n_a_in_the_markdown_and_null_in_the_json(tmp_path: Path) -> None:
    """Catches a null coerced to 0: a judge that was down and a model that got every
    decision wrong are different facts, and publishing 0.0 for the first is a claim nobody
    measured. Also catches an empty cell, which reads as a rendering bug rather than as an
    unmeasured number."""
    unmeasured, unmeasured_report = api_pair(
        "unmeasured-api", cost_per_turn="0.0010", accuracy=None
    )
    other, other_report = api_pair("other-api", cost_per_turn="0.0020")

    table = table_for(tmp_path, [(unmeasured, unmeasured_report), (other, other_report)])
    document = json_document(table)

    assert document["options"]["unmeasured-api"]["metrics"]["decision_accuracy"] is None
    assert NA in markdown_row(tradeoff_markdown(table), "unmeasured-api")
    assert "0.0" not in markdown_row(tradeoff_markdown(table), "unmeasured-api")[:2]


def test_a_null_cost_per_turn_publishes_a_null_cost_rather_than_zero(tmp_path: Path) -> None:
    """Catches a missing ledger cost rendering as 0.0000 per 1,000 turns, which would make
    an unmeasured option the cheapest in the table and hand it the winner line."""
    unpriced, unpriced_report = api_pair("unpriced-api", cost_per_turn=None)
    other, other_report = api_pair("other-api", cost_per_turn="0.0020")

    table = table_for(tmp_path, [(unpriced, unpriced_report), (other, other_report)])
    document = json_document(table)

    assert document["options"]["unpriced-api"]["usd_per_1000_turns"] is None
    assert NA in markdown_row(tradeoff_markdown(table), "unpriced-api")


# --------------------------------------------------------------------------
# the winner rule (decision D, risk 4)
# --------------------------------------------------------------------------


def test_the_winner_is_the_cheapest_option_inside_the_band_and_not_the_most_accurate(
    tmp_path: Path,
) -> None:
    """Catches the obvious wrong rule, "the best decision_accuracy wins": the strong API
    here is the most accurate AND ten times the price of an option 0.01 points behind it,
    so a table that ranked on accuracy names the wrong winner and the published conclusion
    recommends the expensive model."""
    strong, strong_report = api_pair("strong-api", cost_per_turn="0.0100", accuracy=90.0)
    cheap, cheap_report = api_pair("cheap-api", cost_per_turn="0.0010", accuracy=89.99)

    table = table_for(tmp_path, [(strong, strong_report), (cheap, cheap_report)])

    assert table.winner_id == "cheap-api"
    assert names_option(table.conclusion, cheap)
    assert not names_option(table.conclusion, strong)
    assert json_document(table)["winner"] == "cheap-api"


@pytest.mark.parametrize(
    ("accuracy", "winner_id"),
    [(89.98, "cheap-api"), (89.9799, "strong-api")],
    ids=["exactly 0.02 below is inside", "0.0201 below is outside"],
)
def test_the_accuracy_band_is_inclusive_at_exactly_the_tolerance(
    tmp_path: Path, accuracy: float, winner_id: str
) -> None:
    """The boundary the whole rule turns on. Catches a `>` where the plan says `>=`, which
    drops the option sitting exactly on the edge and hands the table to the expensive row,
    and catches a tolerance applied to the wrong side of the comparison. Only the tolerance
    decides these two rows - the costs, the kinds and every other metric are identical, so
    no neighbouring rule can be what flips the winner."""
    strong, strong_report = api_pair("strong-api", cost_per_turn="0.0100", accuracy=90.0)
    cheap, cheap_report = api_pair("cheap-api", cost_per_turn="0.0010", accuracy=accuracy)

    table = table_for(tmp_path, [(strong, strong_report), (cheap, cheap_report)])

    assert table.winner_id == winner_id


def test_an_option_with_no_accuracy_stays_in_the_table_and_never_wins(tmp_path: Path) -> None:
    """Catches a null accuracy treated as 0 or as the best: the unmeasured option here is by
    far the cheapest, so a builder that let it into the candidate set would crown a row
    whose accuracy nobody knows. It still has to be IN the table - dropping the row is
    decision F's other failure."""
    unmeasured, unmeasured_report = api_pair(
        "unmeasured-api", cost_per_turn="0.0001", accuracy=None
    )
    measured, measured_report = api_pair("measured-api", cost_per_turn="0.0100", accuracy=90.0)

    table = table_for(tmp_path, [(unmeasured, unmeasured_report), (measured, measured_report)])

    assert table.winner_id == "measured-api"
    assert [row.id for row in table.rows] == ["unmeasured-api", "measured-api"]
    assert not names_option(table.conclusion, unmeasured)


def test_an_option_with_no_cost_stays_in_the_table_and_never_wins(tmp_path: Path) -> None:
    """Catches a null cost sorted as zero, which makes the unpriced option the cheapest in
    the band and hands it the conclusion. The unpriced row here is also the most accurate,
    so a builder falling back to accuracy picks it too."""
    unpriced, unpriced_report = api_pair("unpriced-api", cost_per_turn=None, accuracy=90.0)
    priced, priced_report = api_pair("priced-api", cost_per_turn="0.0100", accuracy=89.99)

    table = table_for(tmp_path, [(unpriced, unpriced_report), (priced, priced_report)])

    assert table.winner_id == "priced-api"
    assert [row.id for row in table.rows] == ["unpriced-api", "priced-api"]


def test_with_no_option_carrying_both_numbers_the_pinned_no_winner_line_is_rendered(
    tmp_path: Path,
) -> None:
    """Catches a builder that names a winner anyway - the first row, or the only row with a
    cost - when nothing in the table can be compared. Both rows are still published; what is
    refused is the CLAIM."""
    first, first_report = api_pair("first-api", cost_per_turn=None, accuracy=90.0)
    second, second_report = api_pair("second-api", cost_per_turn="0.0100", accuracy=None)

    table = table_for(tmp_path, [(first, first_report), (second, second_report)])

    assert table.winner_id is None
    assert table.conclusion == NO_WINNER_LINE
    assert NO_WINNER_LINE in section(tradeoff_markdown(table), CONCLUSION_HEADING)
    assert json_document(table)["winner"] is None
    assert len(table.rows) == 2


def test_an_api_winner_carries_the_pinned_self_hosting_lost_clause(tmp_path: Path) -> None:
    """Catches the conclusion that quietly omits the clause when the answer is inconvenient:
    the whole point of publishing this table is that it can say self-hosting lost. The GPU
    here is inside the band and more than twice the price."""
    cheap, cheap_report = api_pair("cheap-api", cost_per_turn="0.0010", accuracy=90.0)
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200, accuracy=89.99)

    table = table_for(tmp_path, [(cheap, cheap_report), (gpu, gpu_report)])

    assert table.winner_id == "cheap-api"
    assert SELF_HOSTING_LOST_CLAUSE in table.conclusion
    assert SELF_HOSTING_LOST_CLAUSE in section(tradeoff_markdown(table), CONCLUSION_HEADING)


def test_a_self_hosted_winner_carries_no_self_hosting_lost_clause(tmp_path: Path) -> None:
    """The other half of the same control. Catches a clause appended unconditionally, which
    would publish "self-hosting did not win here" under a table whose winner is the GPU -
    the one sentence a reader would quote, and it would be false."""
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200, accuracy=90.0)
    expensive, expensive_report = api_pair("expensive-api", cost_per_turn="0.0100", accuracy=89.99)

    table = table_for(tmp_path, [(gpu, gpu_report), (expensive, expensive_report)])

    assert table.winner_id == "gpu"
    assert SELF_HOSTING_LOST_CLAUSE not in table.conclusion
    assert SELF_HOSTING_LOST_CLAUSE not in tradeoff_markdown(table)


def test_a_tie_inside_the_band_is_won_by_the_earlier_manifest_entry(tmp_path: Path) -> None:
    """Catches a tie broken by sorting on the id, which would be a stable answer that is not
    the stated one: the earlier entry here is `zeta-api` and the later is `alpha-api`, so an
    id-sorted or a `max`-based tiebreak names `alpha-api`. Two identical runs must give one
    answer, and the plan says it is manifest order."""
    first, first_report = api_pair("zeta-api", cost_per_turn="0.0010", accuracy=90.0)
    second, second_report = api_pair("alpha-api", cost_per_turn="0.0010", accuracy=90.0)

    table = table_for(tmp_path, [(first, first_report), (second, second_report)])

    assert table.winner_id == "zeta-api"
    assert names_option(table.conclusion, first)
    assert not names_option(table.conclusion, second)


# --------------------------------------------------------------------------
# the two documents: order, methodology, and the files on disk
# --------------------------------------------------------------------------


def test_both_files_present_the_options_in_manifest_order_under_a_shuffled_cost_manifest(
    tmp_path: Path,
) -> None:
    """Catches a table sorted by price or by accuracy on its way to the renderer: the
    manifest here is ordered most expensive first, so any numeric sort reverses it. The
    order an operator wrote is the order the published table reads in, whichever option
    won."""
    dear, dear_report = api_pair("dear-api", cost_per_turn="0.0100", accuracy=90.0)
    middle, middle_report = api_pair("middle-api", cost_per_turn="0.0050", accuracy=89.99)
    cheap, cheap_report = api_pair("cheap-api", cost_per_turn="0.0010", accuracy=89.99)
    order = ["dear-api", "middle-api", "cheap-api"]

    table = table_for(
        tmp_path,
        [(dear, dear_report), (middle, middle_report), (cheap, cheap_report)],
    )
    document = json_document(table)

    assert [row.id for row in table.rows] == order
    assert document["order"] == order
    assert markdown_order(tradeoff_markdown(table), order) == order
    assert table.winner_id == "cheap-api"


def test_the_methodology_block_names_each_options_report_id_and_git_sha(tmp_path: Path) -> None:
    """Catches a methodology block written from the manifest alone: without the report id
    and the commit each number came from, nobody can reproduce a published row, and a table
    rebuilt from a stale report reads identically to a fresh one. Also catches a block that
    names one option and stops."""
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200)
    api, api_report = api_pair("cheap-api", cost_per_turn="0.0010")
    gpu_report["git_sha"] = "aaaaaaaabbbbbbbbccccccccdddddddd11111111"
    api_report["git_sha"] = "eeeeeeeeffffffff0000000011111111deadbeef"

    table = table_for(tmp_path, [(gpu, gpu_report), (api, api_report)])
    methodology = section(tradeoff_markdown(table), METHODOLOGY_HEADING)

    assert SET_NAME in methodology
    for option_id in ("gpu", "cheap-api"):
        assert option_id in methodology
        assert report_id_for(option_id) in methodology
    assert "aaaaaaaabbbbbbbbccccccccdddddddd11111111" in methodology
    assert "eeeeeeeeffffffff0000000011111111deadbeef" in methodology
    assert carries_number(methodology, Decimal("1.006"))
    assert LEDGER_NOTE in methodology


def test_a_report_with_no_commit_publishes_a_null_git_sha_rather_than_a_guess(
    tmp_path: Path,
) -> None:
    """Catches a `git_sha` defaulted to an empty string or to the other option's commit: a
    run made outside a checkout has no commit, and a table claiming one would send a reader
    to a tree that never produced the number."""
    first, first_report = api_pair("first-api", cost_per_turn="0.0010")
    second, second_report = api_pair("second-api", cost_per_turn="0.0020")
    first_report["git_sha"] = None

    table = table_for(tmp_path, [(first, first_report), (second, second_report)])
    document = json_document(table)

    assert document["options"]["first-api"]["git_sha"] is None
    assert document["options"]["second-api"]["git_sha"] == BASE_REPORT["git_sha"]


def test_the_json_document_publishes_the_set_the_tolerance_and_every_option_by_id(
    tmp_path: Path,
) -> None:
    """Catches a document the console cannot read: without `order` the rows have no order at
    all (a mapping keyed by id has none), and without the tolerance nobody reading the file
    can tell why the winner is not the most accurate row."""
    first, first_report = api_pair("first-api", cost_per_turn="0.0010")
    second, second_report = api_pair("second-api", cost_per_turn="0.0020")

    table = table_for(tmp_path, [(first, first_report), (second, second_report)])
    document = json_document(table)

    assert document["set_name"] == SET_NAME
    assert Decimal(str(document["accuracy_tolerance"])) == ACCURACY_TOLERANCE
    assert set(document["options"]) == {"first-api", "second-api"}
    assert document["order"] == ["first-api", "second-api"]
    assert document["conclusion"] == table.conclusion
    for option_id in ("first-api", "second-api"):
        published = document["options"][option_id]
        assert published["id"] == option_id
        assert published["label"] == f"label of {option_id}"
        assert published["kind"] == KIND_API
        assert published["report_id"] == report_id_for(option_id)


def test_write_tradeoff_writes_exactly_the_two_named_files(tmp_path: Path) -> None:
    """Catches a writer that names one file after the set or the date: `make tradeoff` and
    the console both read fixed paths, and a third file in the directory is a stale artifact
    a later run will not overwrite."""
    first, first_report = api_pair("first-api", cost_per_turn="0.0010")
    second, second_report = api_pair("second-api", cost_per_turn="0.0020")
    table = table_for(tmp_path / "inputs", [(first, first_report), (second, second_report)])
    out = tmp_path / "out"
    out.mkdir()

    write_tradeoff(out, table)

    assert sorted(path.name for path in out.iterdir()) == sorted(
        [TRADEOFF_JSON_NAME, TRADEOFF_MD_NAME]
    )
    assert (out / TRADEOFF_JSON_NAME).read_text(encoding="utf-8") == tradeoff_json(table)
    assert (out / TRADEOFF_MD_NAME).read_text(encoding="utf-8") == tradeoff_markdown(table)
