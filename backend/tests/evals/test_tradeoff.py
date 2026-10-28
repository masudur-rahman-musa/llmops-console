"""`evals/tradeoff.py`: the table over two or more conversation reports, ranked by check_score.

Pure: manifests and reports are small JSON documents under `tmp_path`.

Contract these tests are written against (everything not named here keeps its old value and
meaning: `TRADEOFF_MD_NAME`, `TRADEOFF_JSON_NAME`, `OPTION_ID_PATTERN`, `MIN_OPTIONS`,
`COST_SCALE`, `ACCURACY_TOLERANCE`, `TURNS_PER_1000`, `KIND_API`, `KIND_SELF_HOSTED`,
`NA`, the heading and line constants, `TradeoffInputInvalid`):

* `ROW_METRIC_NAMES == ("check_score", "judge_score")`, and the first of them is the accuracy
  the winner rule reads. `COST_METRIC_NAME == "cost_usd_per_turn"` stays: an `api` option's
  cost is that metric from its report, times `TURNS_PER_1000`.
* A report is read through `RunReport` (the conversation report). A report in the old
  single-reply shape is refused with `TradeoffInputInvalid` naming its path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.evals.report import build_report, report_json
from app.evals.tradeoff import (
    ACCURACY_TOLERANCE,
    COST_METRIC_NAME,
    KIND_API,
    KIND_SELF_HOSTED,
    NO_WINNER_LINE,
    ROW_METRIC_NAMES,
    TradeoffInputInvalid,
    TradeoffTable,
    build_table,
    load_manifest,
    tradeoff_json,
    tradeoff_markdown,
)

from .builders import case_line

START = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
SET_NAME = "cases.jsonl"


def write_report(
    directory: Path,
    option_id: str,
    *,
    check: float | None,
    judge: float | None = None,
    cost: str | None = None,
    cases: int = 4,
    set_name: str = SET_NAME,
) -> str:
    report = build_report(
        [case_line(f"conv-{n:03d}", passed=True) for n in range(cases)],
        started_at=START,
        finished_at=START + timedelta(minutes=30),
        set_name=set_name,
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="manual",
        git_sha="b" * 40,
        tenant="all",
        target="https://helpdesk.example",
    )
    metrics = {**report.metrics, "check_score": check, "judge_score": judge}
    if cost is not None:
        metrics[COST_METRIC_NAME] = cost
    report = report.model_copy(update={"metrics": metrics, "id": option_id.ljust(32, "0")[:32]})
    name = f"{option_id}.json"
    (directory / name).write_text(report_json(report), encoding="utf-8")
    return name


def write_manifest(directory: Path, *options: dict[str, Any], set_name: str = SET_NAME) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps({"set_name": set_name, "options": list(options)}), encoding="utf-8")
    return path


def api(option_id: str, report: str) -> dict[str, Any]:
    return {"id": option_id, "label": f"label {option_id}", "kind": KIND_API, "report": report}


def table_of(directory: Path, *options: dict[str, Any]) -> TradeoffTable:
    return build_table(load_manifest(write_manifest(directory, *options)), base_dir=directory)


def test_the_rows_read_check_score_and_judge_score() -> None:
    """Catches a table still reading the old five metrics: the console ranks by
    check_score, and a row without `judge_score` would hide the second number."""
    assert ROW_METRIC_NAMES == ("check_score", "judge_score")
    assert COST_METRIC_NAME == "cost_usd_per_turn"


def test_tradeoff_ranks_by_check_score(tmp_path: Path) -> None:
    """Catches a winner chosen by the first row, by the highest score, or by the lowest price
    outside the band. 'strong' scores highest and costs most; 'cheap' scores 5 points lower,
    outside the band, and costs least; 'mid' is the cheapest inside the band. 'mid' must win,
    and each row carries its own two scores."""
    directory = tmp_path
    options = [
        api("strong", write_report(directory, "strong", check=90.0, judge=70.0, cost="0.0300")),
        api("cheap", write_report(directory, "cheap", check=85.0, judge=60.0, cost="0.0010")),
        api("mid", write_report(directory, "mid", check=89.99, judge=65.0, cost="0.0100")),
    ]

    table = table_of(directory, *options)

    assert table.winner_id == "mid"
    by_id = {row.id: row for row in table.rows}
    assert by_id["strong"].metrics == {"check_score": 90.0, "judge_score": 70.0}
    assert by_id["cheap"].metrics == {"check_score": 85.0, "judge_score": 60.0}
    assert [row.id for row in table.rows] == ["strong", "cheap", "mid"]


def test_the_accuracy_band_is_inclusive_at_the_tolerance(tmp_path: Path) -> None:
    """Catches a `<` where `<=` belongs: a cheap option exactly 0.02 below the best is in the
    band and wins on price; 0.0201 below is out. Both rows have the same cost, so only the
    band decides."""
    assert Decimal("0.02") == ACCURACY_TOLERANCE
    inside = tmp_path / "in"
    outside = tmp_path / "out"
    inside.mkdir()
    outside.mkdir()

    in_table = table_of(
        inside,
        api("best", write_report(inside, "best", check=90.0, cost="0.0300")),
        api("cheap", write_report(inside, "cheap", check=89.98, cost="0.0100")),
    )
    out_table = table_of(
        outside,
        api("best", write_report(outside, "best", check=90.0, cost="0.0300")),
        api("cheap", write_report(outside, "cheap", check=89.9799, cost="0.0100")),
    )

    assert in_table.winner_id == "cheap"
    assert out_table.winner_id == "best"


def test_a_run_with_no_check_score_stays_in_the_table_and_never_wins(tmp_path: Path) -> None:
    """Catches a null score read as zero or as a pass: the unscored option is the cheapest
    and must not win, and with nothing else comparable there is no winner at all."""
    table = table_of(
        tmp_path,
        api("unscored", write_report(tmp_path, "unscored", check=None, cost="0.0010")),
        api("scored", write_report(tmp_path, "scored", check=80.0, cost="0.0200")),
    )

    assert table.winner_id == "scored"
    assert [row.id for row in table.rows] == ["unscored", "scored"]
    only = table_of(tmp_path, api("unscored", "unscored.json"), api("unscored2", "unscored.json"))
    assert only.winner_id is None
    assert only.conclusion == NO_WINNER_LINE


def test_an_api_cost_is_the_ledger_cost_per_turn_times_one_thousand(tmp_path: Path) -> None:
    """Catches a cost taken per turn instead of per thousand turns, or computed from the
    report's wall clock for an api option (which is priced from the ledger)."""
    table = table_of(
        tmp_path,
        api("a", write_report(tmp_path, "a", check=80.0, cost="0.0123")),
        api("b", write_report(tmp_path, "b", check=80.0, cost="0.0200")),
    )

    assert table.rows[0].usd_per_1000_turns == Decimal("12.3000")


def test_a_self_hosted_cost_is_the_hourly_price_over_the_measured_throughput(
    tmp_path: Path,
) -> None:
    """Catches a cost that ignores the measured run: 4 turns in 30 minutes is 8 an hour, so
    at 2.00 an hour one turn costs 0.25 and 1000 turns cost 250.0000."""
    own = {
        "id": "box",
        "label": "our box",
        "kind": KIND_SELF_HOSTED,
        "hourly_usd": "2.00",
        "report": write_report(tmp_path, "box", check=80.0),
    }

    table = table_of(tmp_path, own, api("a", write_report(tmp_path, "a", check=80.0, cost="1.0")))

    assert table.rows[0].usd_per_1000_turns == Decimal("250.0000")
    assert table.rows[0].turns == 4


def test_the_markdown_and_json_publish_the_scores(tmp_path: Path) -> None:
    """Catches output that still says decision accuracy: both files carry `check_score`, and
    the markdown row starts with the option id."""
    table = table_of(
        tmp_path,
        api("a", write_report(tmp_path, "a", check=80.0, judge=55.5, cost="0.01")),
        api("b", write_report(tmp_path, "b", check=70.0, cost="0.02")),
    )

    text = tradeoff_markdown(table)
    document = json.loads(tradeoff_json(table))

    assert "check_score" in text and "decision_accuracy" not in text
    assert any(line.startswith("| a ") for line in text.splitlines())
    assert document["options"]["a"]["metrics"]["check_score"] == 80.0


def test_a_report_in_the_old_format_is_refused_and_names_its_path(tmp_path: Path) -> None:
    """Catches a table that accepts an old single-reply report and publishes a winner from
    numbers measured on another exam."""
    (tmp_path / "old.json").write_text(
        json.dumps({"set_name": SET_NAME, "metrics": {"decision_accuracy": 90.0}}),
        encoding="utf-8",
    )
    other = write_report(tmp_path, "other", check=80.0)

    with pytest.raises(TradeoffInputInvalid) as raised:
        table_of(tmp_path, api("old", "old.json"), api("other", other))

    assert "old.json" in str(raised.value)


def test_a_report_for_another_case_file_is_refused(tmp_path: Path) -> None:
    """Catches a table comparing two different exams."""
    mine = write_report(tmp_path, "mine", check=80.0)
    theirs = write_report(tmp_path, "theirs", check=80.0, set_name="other.jsonl")

    with pytest.raises(TradeoffInputInvalid) as raised:
        table_of(tmp_path, api("mine", mine), api("theirs", theirs))

    assert "other.jsonl" in str(raised.value)


@pytest.mark.parametrize("option_id", ["a", "a" * 32, "cheap-api"])
def test_option_ids_inside_the_pattern_load(tmp_path: Path, option_id: str) -> None:
    """Catches a pattern that refuses the inclusive ends: 1 and 32 characters."""
    load_manifest(write_manifest(tmp_path, api(option_id, "x.json"), api("other", "y.json")))


@pytest.mark.parametrize(
    "option_id", ["a" * 33, "cheap-api\n", "cheap-api\r\n", "cheap-api ", "A", ""]
)
def test_option_ids_outside_the_pattern_are_refused(tmp_path: Path, option_id: str) -> None:
    """Catches a pattern matched with `$` (a trailing newline gets through) or without the
    32-character cap: the id names a markdown row and an evidence folder."""
    path = write_manifest(tmp_path, api(option_id, "x.json"), api("other", "y.json"))

    with pytest.raises(TradeoffInputInvalid):
        load_manifest(path)
