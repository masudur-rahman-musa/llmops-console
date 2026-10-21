r"""`ops-eval tradeoff`: N published runs as one comparison table.

Pure, in `gate.py`'s shape: one manifest and N `report.json` documents in, two
rendered files out, and no clock, no network and no environment variable in
between. The same inputs give the same table on a laptop and in the image the
eval CronJob runs, which is what makes the published table evidence rather than
a screenshot.

What the module refuses to do is the point of it:

* every cost is a `Decimal` from first read to published string, and both are
  serialised as JSON STRINGS, so a per-turn price the ledger computed exactly
  cannot reach the table as `0.30000000000000004`;
* a self-hosted option is priced from the throughput the box actually delivered
  over the run - `len(report.cases)` over the wall-clock hours between the two
  timestamps - and never from a case count the manifest or the metrics block
  supplies;
* a metric nobody measured is `null` in the JSON and `NA` in the markdown, never
  `0`, and its option is left out of the winner computation rather than crowned
  by it;
* the conclusion is COMPUTED under the one rule `_winner` states, so a table
  whose cheapest option is an API says so in `SELF_HOSTING_LOST_CLAUSE` and
  cannot quietly flatter the GPU.

The winner rule, in full. A CANDIDATE is a row carrying both a
`check_score` and a `usd_per_1000_turns`; `best` is the highest accuracy
among the candidates; a candidate QUALIFIES when its accuracy is at least
`best - ACCURACY_TOLERANCE`, inclusively; the winner is the qualifying candidate
with the lowest cost, ties going to the earlier manifest entry so two identical
runs give one stable answer. With no
candidate at all the table still publishes every row and names no winner.

`ACCURACY_TOLERANCE` is applied to the metric as `scores.py` emits it, which is
POINTS (0-100). The two published documents carry the tolerance beside the table
for that reason: a reader who sees a less accurate option win has the band in
front of them.

The two metric names a row reads are asserted against `scores.METRIC_NAMES` at
import, so a rename there is an `ImportError`-loud failure here rather than a
column that silently empties. Reports are read through the conversation `RunReport`,
so a report in the old single-reply shape is refused naming its path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.evals.report import RunReport
from app.evals.scores import METRIC_NAMES

TRADEOFF_MD_NAME: Final[str] = "tradeoff.md"
"""The markdown table's file name, and never anything a caller supplies: `make
tradeoff`, the README and the console all read a fixed path."""

TRADEOFF_JSON_NAME: Final[str] = "tradeoff.json"
"""The same table as data, for the console, which cannot read markdown."""

OPTION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9-]{0,31}")
"""The store-shaping input of this module: this id keys the JSON document's
option object and names a markdown row. Matched with `re.fullmatch` and never
`.match`, because `$` in Python also matches before a trailing newline."""

MIN_OPTIONS: Final[int] = 2
"""A table of one compares nothing, so a manifest of one is refused."""

COST_SCALE: Final[int] = 4
"""Decimal places on every published cost and throughput, `ROUND_HALF_UP`. Four
rather than the ledger's six because the figure is per 1,000 turns: six places
there is a hundredth of a cent nobody can act on."""

ACCURACY_TOLERANCE: Final[Decimal] = Decimal("0.02")
"""The band inside which two `check_score` values count as equal, in the
points `scores.py` emits. Widening it lets a materially less accurate option win
on price."""

TURNS_PER_1000: Final[Decimal] = Decimal(1000)
"""The unit both cost formulas publish in, as a `Decimal` so no multiplication in
this module can reach a float."""

SECONDS_PER_HOUR: Final[Decimal] = Decimal(3600)
"""Elapsed time is read in HOURS, because an hourly price divides it."""

KIND_API: Final[str] = "api"
"""A metered provider: cost comes from the ledger's own per-turn number."""

KIND_SELF_HOSTED: Final[str] = "self_hosted"
"""A box rented by the hour: cost comes from the price and the measured throughput."""

ROW_METRIC_NAMES: Final[tuple[str, ...]] = ("check_score", "judge_score")
"""The two metrics a row publishes, in the order the table renders them. The first is the
accuracy the winner rule reads."""

COST_METRIC_NAME: Final[str] = "cost_usd_per_turn"
"""The ledger's per-turn cost, which the `api` formula multiplies by 1,000."""

for _name in ROW_METRIC_NAMES:
    if _name not in METRIC_NAMES:  # pragma: no cover - an import-time guard, not a branch
        raise ImportError(f"{_name} is not a member of scores.METRIC_NAMES")

NA: Final[str] = "n/a"
"""What a null metric renders as in the markdown. A blank cell reads as a
rendering bug and a `0` reads as a measurement nobody made."""

METHODOLOGY_HEADING: Final[str] = "## Methodology"
"""Where a reader finds the report and the commit behind every published row."""

CONCLUSION_HEADING: Final[str] = "## Conclusion"
"""Where the one computed sentence lives."""

TABLE_HEADINGS: Final[tuple[str, ...]] = (
    "Option",
    "Kind",
    "check_score",
    "judge_score",
    "USD per 1,000 turns",
)
"""The markdown table's header cells. The first cell of every option row is the
option's id, which is what the tests locate a row by."""

NO_WINNER_LINE: Final[str] = (
    "No option carries both a check_score and a cost, so this table names no winner."
)
"""The whole conclusion when nothing can be compared. Every row is still
published; what is refused is the claim."""

SELF_HOSTING_LOST_CLAUSE: Final[str] = (
    "Self-hosting did not win here: the cheapest option inside the band is an API."
)
"""Appended when the winner's kind is `api`, and rendered nowhere else: the whole
point of publishing this table is that it is free to say self-hosting lost."""

LEDGER_NOTE: Final[str] = (
    "API cost per turn comes from the ledger's recorded spend for the run, not from a list price."
)
"""The methodology's one sentence on where an API's cost came from."""

WINNER_LINE: Final[str] = (
    "{label} ({option_id}) wins at {cost} USD per 1,000 turns, the cheapest option within "
    "{tolerance} check_score of the best row in the table."
)
"""The conclusion when a winner exists. It names the winner and no other option,
so a reader quoting one sentence quotes the whole answer."""


class TradeoffInputInvalid(Exception):
    """One of this command's input files could not be used, and the message names it.

    `gate.GateInputInvalid`'s shape, for its reason: every way a manifest or a
    report can be wrong arrives at the CLI as ONE class, so the refusal block
    prints one line and answers `EXIT_USAGE` rather than letting an `OSError`, a
    `JSONDecodeError` or a pydantic `ValidationError` reach an operator as a
    traceback.
    """

    def __init__(self, reason: str, *, path: Path) -> None:
        self.reason = reason
        self.path = path
        super().__init__(f"{path}: {reason}")


def _one_line(exc: ValidationError) -> str:
    """One pydantic `ValidationError` as ONE line: the first problem, and how many more.

    `str(exc)` runs to a line per field plus a documentation URL for each, and
    this message travels to a terminal as a single stderr line.
    """
    problems = exc.errors()
    if not problems:
        return "did not validate"
    first = problems[0]
    location = first.get("loc") or ()
    field = ".".join(str(part) for part in location) or "the document"
    message = str(first.get("msg", "is not valid")).replace("\n", " ")
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{field}: {message}{more}"


class TradeoffOption(BaseModel):
    """One option in the comparison, and the run that measured it.

    `extra="forbid"` so an operator who writes `hourly_cost` for `hourly_usd`
    gets a refusal rather than a table priced from nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    """The store-shaping input, checked against `OPTION_ID_PATTERN` here and so
    BEFORE any report file is read."""
    label: str
    kind: Literal["api", "self_hosted"]
    report: Path
    """That option's `report.json`. A relative path resolves against the
    manifest's own directory, never the process's working directory."""
    hourly_usd: Decimal | None = None
    """What the box costs per hour, required on `self_hosted` and forbidden on
    `api`. Read only from a JSON STRING: `1.006` as a binary float is
    1.00599999999999978, which drifts in the fourth place this module publishes."""
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _checked_id(cls, value: str) -> str:
        if OPTION_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{value!r} is not an option id matching {OPTION_ID_PATTERN.pattern}")
        return value

    @field_validator("hourly_usd", mode="before")
    @classmethod
    def _decimal_from_a_string(cls, value: object) -> object:
        if value is None or isinstance(value, str):
            return value
        raise ValueError("must be written as a JSON string, so no price is read through a float")

    @model_validator(mode="after")
    def _price_pairs_with_the_kind(self) -> TradeoffOption:
        if self.kind == KIND_SELF_HOSTED and self.hourly_usd is None:
            raise ValueError(f"a {KIND_SELF_HOSTED} option needs an hourly_usd")
        if self.kind == KIND_API and self.hourly_usd is not None:
            raise ValueError(f"an {KIND_API} option is priced from the ledger, not by the hour")
        if self.hourly_usd is not None and self.hourly_usd <= 0:
            raise ValueError("hourly_usd must be above zero")
        return self


class TradeoffManifest(BaseModel):
    """The one document `--manifest` names: the cases file, and the options over it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_name: str
    """Every option's report must carry this set, or the table compares two
    different exams."""
    options: Annotated[tuple[TradeoffOption, ...], Field(min_length=MIN_OPTIONS)]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> TradeoffManifest:
        seen: set[str] = set()
        for option in self.options:
            if option.id in seen:
                raise ValueError(f"{option.id!r} names two options")
            seen.add(option.id)
        return self


class TradeoffRow(BaseModel):
    """One option as the table publishes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: str
    kind: str
    report_id: str
    git_sha: str | None
    hourly_usd: Decimal | None
    turns: int
    """`len(report.cases)`, so the throughput is what the run really answered."""
    turns_per_hour: Decimal | None
    usd_per_1000_turns: Decimal | None
    metrics: Mapping[str, Any]
    """The two `ROW_METRIC_NAMES` as the report carried them, nulls included."""


class TradeoffTable(BaseModel):
    """Every row in manifest order, the computed winner, and the sentence it produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_name: str
    rows: tuple[TradeoffRow, ...]
    winner_id: str | None
    conclusion: str


def load_manifest(path: Path) -> TradeoffManifest:
    """Read the manifest, or raise `TradeoffInputInvalid` naming the path.

    A mistyped path, a half-written file and a document from another schema all
    become one typed refusal, which is what lets the CLI's pre-check block stay
    one `except`.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TradeoffInputInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TradeoffInputInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc

    try:
        return TradeoffManifest.model_validate(payload)
    except ValidationError as exc:
        raise TradeoffInputInvalid(
            f"is not a tradeoff manifest: {_one_line(exc)}", path=path
        ) from exc


def _load_report(path: Path) -> RunReport:
    """One option's `report.json`, validated through `RunReport` itself.

    Never by picking the two read metrics out of whatever the file holds: a
    hand-written document carrying an old metric and nothing else would
    otherwise be published as a measured run.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TradeoffInputInvalid(f"cannot be read: {exc.strerror}", path=path) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TradeoffInputInvalid(f"is not valid JSON: {exc.msg}", path=path) from exc

    try:
        return RunReport.model_validate(payload)
    except ValidationError as exc:
        raise TradeoffInputInvalid(f"is not a run report: {_one_line(exc)}", path=path) from exc


def _elapsed_hours(started_at: datetime, finished_at: datetime) -> Decimal:
    """The wall clock between the two timestamps, in hours, as an exact `Decimal`.

    `total_seconds()` answers a float, so the parts are read off the `timedelta`
    and assembled here: this number divides a case count and multiplies a price.
    """
    elapsed = finished_at - started_at
    seconds = Decimal(elapsed.days) * Decimal(86400) + Decimal(elapsed.seconds)
    return (seconds + Decimal(elapsed.microseconds) / Decimal(1_000_000)) / SECONDS_PER_HOUR


def _quantised(value: Decimal) -> Decimal:
    """One published number at `COST_SCALE` places, half rounding UP.

    Banker's rounding would render 0.00012345 per turn as 0.1234 per 1,000.
    """
    return value.quantize(Decimal(1).scaleb(-COST_SCALE), rounding=ROUND_HALF_UP)


def _api_cost(metrics: Mapping[str, Any]) -> Decimal | None:
    """The ledger's per-turn cost times 1,000, or `None` where it measured nothing."""
    recorded = metrics.get(COST_METRIC_NAME)
    if recorded is None:
        return None
    return _quantised(Decimal(str(recorded)) * TURNS_PER_1000)


def _row(option: TradeoffOption, report: RunReport, path: Path) -> TradeoffRow:
    """One option and its report as a published row, or the refusal it earns.

    Both refusals here are divisions this module will not make: equal or reversed
    timestamps give a zero or negative elapsed time, and a self-hosted run of no
    cases gives a throughput of zero.
    """
    if report.finished_at <= report.started_at:
        raise TradeoffInputInvalid(
            "finished_at is not after started_at, so the run has no elapsed time", path=path
        )

    turns = len(report.cases)
    if option.kind == KIND_SELF_HOSTED and turns == 0:
        raise TradeoffInputInvalid(
            f"a {KIND_SELF_HOSTED} option needs a run with cases to price its throughput",
            path=path,
        )

    turns_per_hour = Decimal(turns) / _elapsed_hours(report.started_at, report.finished_at)
    # Branching on the PRICE and not on the kind, because `_price_pairs_with_the_kind`
    # makes the two the same question - a priced option is a self-hosted one and an
    # unpriced one is an API - and this way no branch here is unreachable.
    price = option.hourly_usd
    cost: Decimal | None
    if price is not None:
        cost = _quantised(price / turns_per_hour * TURNS_PER_1000)
    else:
        cost = _api_cost(report.metrics)

    return TradeoffRow(
        id=option.id,
        label=option.label,
        kind=option.kind,
        report_id=report.id,
        git_sha=report.git_sha,
        hourly_usd=option.hourly_usd,
        turns=turns,
        turns_per_hour=_quantised(turns_per_hour),
        usd_per_1000_turns=cost,
        metrics={name: report.metrics.get(name) for name in ROW_METRIC_NAMES},
    )


def _accuracy(row: TradeoffRow) -> Decimal | None:
    """One row's check_score as a `Decimal`, or `None` where it is null."""
    value = row.metrics.get(ROW_METRIC_NAMES[0])
    return None if value is None else Decimal(str(value))


def _cost_key(row: TradeoffRow) -> Decimal:
    """One candidate's cost, for the `min` that picks the winner.

    Only ever called on a row whose cost `_winner` has already found non-null; the
    fallback is what keeps `mypy --strict` from widening the sort key to
    `Decimal | None`, which is not orderable.
    """
    return Decimal(0) if row.usd_per_1000_turns is None else row.usd_per_1000_turns


def _winner(rows: Sequence[TradeoffRow]) -> TradeoffRow | None:
    """The cheapest row inside the accuracy band, or `None` when nothing compares.

    The rule the module docstring states, in one place. `min` over the qualifying
    rows in manifest order is what breaks a tie by the earlier entry, because
    `min` keeps the first of equal keys.
    """
    candidates = [
        row for row in rows if _accuracy(row) is not None and row.usd_per_1000_turns is not None
    ]
    if not candidates:
        return None
    # `is not None` and never a truthiness test: a check_score of 0.0 and a
    # cost of 0.0000 are both measurements, and dropping them would quietly remove
    # the worst row from the band and the cheapest row from the comparison.
    best = max(
        accuracy for accuracy in (_accuracy(row) for row in candidates) if accuracy is not None
    )
    qualifying = [
        row
        for row in candidates
        if (accuracy := _accuracy(row)) is not None and accuracy >= best - ACCURACY_TOLERANCE
    ]
    return min(qualifying, key=_cost_key)


def _conclusion(winner: TradeoffRow | None) -> str:
    """The one computed sentence, and the clause an API winner earns."""
    if winner is None:
        return NO_WINNER_LINE
    line = WINNER_LINE.format(
        label=winner.label,
        option_id=winner.id,
        cost=winner.usd_per_1000_turns,
        tolerance=ACCURACY_TOLERANCE,
    )
    if winner.kind == KIND_API:
        return f"{line} {SELF_HOSTING_LOST_CLAUSE}"
    return line


def build_table(manifest: TradeoffManifest, *, base_dir: Path) -> TradeoffTable:
    """Every option's report read, priced and ordered as the manifest wrote them.

    Every report is loaded and checked before any row is built, and nothing is
    written from here at all, so a refusal on the last option cannot leave a
    half-rendered table behind.
    """
    rows: list[TradeoffRow] = []
    for option in manifest.options:
        path = base_dir / option.report
        report = _load_report(path)
        if report.set_name != manifest.set_name:
            raise TradeoffInputInvalid(
                f"was run against {report.set_name}, not the manifest's {manifest.set_name}",
                path=path,
            )
        rows.append(_row(option, report, path))

    winner = _winner(rows)
    return TradeoffTable(
        set_name=manifest.set_name,
        rows=tuple(rows),
        winner_id=None if winner is None else winner.id,
        conclusion=_conclusion(winner),
    )


def _published(value: Decimal | None) -> str | None:
    """One `Decimal` as the JSON document carries it: a STRING, or `null`."""
    return None if value is None else str(value)


def tradeoff_json(table: TradeoffTable) -> str:
    """The table as the console reads it.

    `order` is beside `options` because a mapping keyed by id has no order of its
    own, and the order an operator wrote is the order the table reads in.
    """
    document = {
        "set_name": table.set_name,
        "accuracy_tolerance": str(ACCURACY_TOLERANCE),
        "options": {
            row.id: {
                "id": row.id,
                "label": row.label,
                "kind": row.kind,
                "report_id": row.report_id,
                "git_sha": row.git_sha,
                "hourly_usd": _published(row.hourly_usd),
                "turns": row.turns,
                "turns_per_hour": _published(row.turns_per_hour),
                "usd_per_1000_turns": _published(row.usd_per_1000_turns),
                "metrics": dict(row.metrics),
            }
            for row in table.rows
        },
        "order": [row.id for row in table.rows],
        "winner": table.winner_id,
        "conclusion": table.conclusion,
    }
    return json.dumps(document, indent=2) + "\n"


def _cell(value: object) -> str:
    """One metric as the markdown renders it: the number, or `NA` where it is null."""
    return NA if value is None else str(value)


def _row_line(row: TradeoffRow) -> str:
    cells = [
        row.id,
        row.kind,
        *(_cell(row.metrics.get(name)) for name in ROW_METRIC_NAMES),
        _cell(row.usd_per_1000_turns),
    ]
    return "| " + " | ".join(cells) + " |"


def _methodology_lines(table: TradeoffTable) -> list[str]:
    """One bullet per option, carrying what a reader needs to reproduce its row.

    The report id and the commit are what separate a table rebuilt from a stale
    report from a fresh one.
    """
    lines = [
        METHODOLOGY_HEADING,
        "",
        f"Every row is one run of `{table.set_name}`, and every option was scored on that "
        "same set.",
        "",
    ]
    for row in table.rows:
        price = "priced by the hour at " + str(row.hourly_usd) if row.hourly_usd else "not hourly"
        lines.append(
            f"- `{row.id}` ({row.label}, {row.kind}): report `{row.report_id}`, "
            f"commit `{row.git_sha}`, {price}, {row.turns} turns at "
            f"{row.turns_per_hour} turns an hour."
        )
    lines.extend(["", LEDGER_NOTE, ""])
    return lines


def tradeoff_markdown(table: TradeoffTable) -> str:
    """The published table, its methodology and the computed conclusion."""
    lines = [
        f"# Tradeoff: {table.set_name}",
        "",
        "| " + " | ".join(TABLE_HEADINGS) + " |",
        "|" + "|".join(["---"] * len(TABLE_HEADINGS)) + "|",
    ]
    lines.extend(_row_line(row) for row in table.rows)
    lines.extend(["", *_methodology_lines(table)])
    lines.extend([CONCLUSION_HEADING, "", table.conclusion, ""])
    return "\n".join(lines)


def write_tradeoff(out: Path, table: TradeoffTable) -> None:
    """Write both files into `out`, overwriting whatever a previous run left.

    `write_text` and never an append: two tables under one heading would publish
    the previous run's winner beneath the new one.
    """
    (out / TRADEOFF_JSON_NAME).write_text(tradeoff_json(table), encoding="utf-8")
    (out / TRADEOFF_MD_NAME).write_text(tradeoff_markdown(table), encoding="utf-8")
