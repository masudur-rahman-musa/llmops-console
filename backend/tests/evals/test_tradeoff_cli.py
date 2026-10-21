"""`ops-eval tradeoff --manifest PATH --out DIR`: exit codes over conversation reports.

The shell over `evals/tradeoff.py`: every refusal is exit 2, one line on stderr, nothing on
stdout and an empty `--out`; the happy path writes exactly `tradeoff.md` and
`tradeoff.json`. The report and manifest builders are the ones in `test_tradeoff.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import evals as cli
from app.cli.evals import EXIT_OK, EXIT_USAGE
from app.evals.tradeoff import (
    CONCLUSION_HEADING,
    METHODOLOGY_HEADING,
    TRADEOFF_JSON_NAME,
    TRADEOFF_MD_NAME,
)

from .test_tradeoff import api, write_manifest, write_report


def good_manifest(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return write_manifest(
        directory,
        api("cheap", write_report(directory, "cheap", check=80.0, judge=50.0, cost="0.0100")),
        api("strong", write_report(directory, "strong", check=90.0, judge=70.0, cost="0.0300")),
    )


def run_tradeoff(manifest: Path, out: Path) -> int:
    return cli.main(["tradeoff", "--manifest", str(manifest), "--out", str(out)])


def test_the_happy_path_exits_zero_with_exactly_the_two_files(tmp_path: Path) -> None:
    """Catches a subcommand that falls through to a removed `run` handler, or an output named
    from the set or the date: the docs and the console read these two fixed names."""
    out = tmp_path / "out"
    out.mkdir()

    code = run_tradeoff(good_manifest(tmp_path / "in"), out)

    assert code == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == sorted([TRADEOFF_JSON_NAME, TRADEOFF_MD_NAME])
    assert "check_score" in (out / TRADEOFF_MD_NAME).read_text(encoding="utf-8")
    assert json.loads((out / TRADEOFF_JSON_NAME).read_text(encoding="utf-8"))["winner"]


def test_a_report_in_the_old_format_exits_two_with_one_line_and_an_empty_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches an unhandled refusal that exits 1 with a traceback (read as a real verdict), one
    printed on stdout, and one that renders a half table before it validates. The second
    option is valid, so the old-format report is the only thing that can refuse."""
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "old.json").write_text(
        json.dumps({"set_name": "cases.jsonl", "metrics": {"decision_accuracy": 90.0}}),
        encoding="utf-8",
    )
    manifest = write_manifest(
        inputs,
        api("old", "old.json"),
        api("new", write_report(inputs, "new", check=80.0, cost="0.0100")),
    )
    out = tmp_path / "out"
    out.mkdir()

    code = run_tradeoff(manifest, out)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert list(out.iterdir()) == []
    assert printed.out == ""
    assert printed.err.strip().count("\n") == 0 and printed.err.strip() != ""
    assert str(inputs / "old.json") in printed.err
    assert "Traceback" not in printed.err


def test_a_missing_manifest_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a mistyped path that surfaces as a traceback."""
    out = tmp_path / "out"
    out.mkdir()

    code = run_tradeoff(tmp_path / "nope.json", out)

    assert code == EXIT_USAGE
    assert "nope.json" in capsys.readouterr().err


def test_a_second_run_overwrites_both_files(tmp_path: Path) -> None:
    """Catches an append-mode write that leaves two tables and two conclusions in the
    published document."""
    out = tmp_path / "out"
    out.mkdir()
    manifest = good_manifest(tmp_path / "in")

    run_tradeoff(manifest, out)
    run_tradeoff(manifest, out)

    markdown = (out / TRADEOFF_MD_NAME).read_text(encoding="utf-8")
    assert markdown.count(METHODOLOGY_HEADING) == 1
    assert markdown.count(CONCLUSION_HEADING) == 1


@pytest.mark.parametrize(
    "argv",
    [["tradeoff"], ["tradeoff", "--manifest", "m.json"], ["tradeoff", "--out", "d"]],
    ids=["neither flag", "no --out", "no --manifest"],
)
def test_both_flags_are_required(argv: list[str]) -> None:
    """Catches a flag with a default: a mistyped command would rewrite a published table the
    operator never named."""
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    assert raised.value.code == 2
