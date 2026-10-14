r"""`ops-eval tradeoff --manifest PATH --out DIR`: the fifth subcommand's exit codes.

The shell over `evals/tradeoff.py`, in `gate`'s shape: every pre-check runs before
anything is written, and the two outputs are written last. Pure - no target, no
database, no clock - so every case here is `cli.main` over JSON files in
`tmp_path`.

The contract these cases are written against, before it exists:

* both flags are required and both are `type=Path`;
* the happy path exits `EXIT_OK` with exactly `tradeoff.md` and `tradeoff.json`
  in `--out`, under the module's own constants;
* EVERY refusal in `evals/tradeoff.py` - the manifest ones and the report ones
  alike - arrives as `TradeoffInputInvalid` and leaves `EXIT_USAGE`, ONE line on
  stderr, nothing on stdout and an EMPTY `--out`;
* a rerun overwrites both files rather than appending to either.

The refusal table is the one in `tests/evals/test_tradeoff.py` driven through the
command, and the `--out` assertion is what makes each row a test of the CLI rather
than a second test of the builder: a command that wrote a half-table and then
refused would pass on the exit code alone. `--out` is created by the case and
never by the assertion, so "the directory is empty" cannot be satisfied by a
command that never created it.

The helpers and the report fixture are imported from `test_tradeoff.py` rather
than copied, the way `tests/infra/test_aws_module.py` imports from
`tests/infra/test_hetzner_module.py`: one base report document, built once at
module scope, shapes both files' cases. The import names the module as pytest and
mypy both see it - `tests/evals/__init__.py` exists and `tests/__init__.py` does
not, so this package is `evals`, and spelling it `tests.evals` loads the same file
twice under two names.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app.evals import cli
from app.evals.cli import EXIT_OK, EXIT_USAGE
from app.evals.tradeoff import (
    CONCLUSION_HEADING,
    KIND_API,
    KIND_SELF_HOSTED,
    METHODOLOGY_HEADING,
    TRADEOFF_JSON_NAME,
    TRADEOFF_MD_NAME,
)
from evals.test_tradeoff import (
    MANIFEST_NAME,
    api_pair,
    self_hosted_pair,
    write_inputs,
    write_json,
)

Build = Callable[[Path], Path]
"""One case's inputs: write them under a root and answer the manifest path."""


def run_tradeoff(manifest: Path, out: Path) -> int:
    """The command as an operator types it."""
    return cli.main(["tradeoff", "--manifest", str(manifest), "--out", str(out)])


def two_good_pairs() -> list[tuple[dict[str, object], dict[str, object] | None]]:
    """Two priced, measured API options - a manifest with nothing wrong with it."""
    first, first_report = api_pair("cheap-api", cost_per_turn="0.0010", accuracy=90.0)
    second, second_report = api_pair("strong-api", cost_per_turn="0.0100", accuracy=89.99)
    return [(first, first_report), (second, second_report)]


def good_manifest(root: Path) -> Path:
    return write_inputs(root, two_good_pairs())


def _bad_option_id(root: Path) -> Path:
    first, first_report = api_pair("cheap-api", cost_per_turn="0.0010")
    second, second_report = api_pair("strong-api", cost_per_turn="0.0100")
    second["id"] = "Strong_API"
    return write_inputs(root, [(first, first_report), (second, second_report)])


def _duplicate_option_id(root: Path) -> Path:
    first, first_report = api_pair("twin-api", cost_per_turn="0.0010")
    second, second_report = api_pair("twin-api", cost_per_turn="0.0100")
    second["report"] = "second.json"
    return write_inputs(root, [(first, first_report), (second, second_report)])


def _one_option(root: Path) -> Path:
    return write_inputs(root, two_good_pairs()[:1])


def _self_hosted_without_a_price(root: Path) -> Path:
    pairs = two_good_pairs()
    option, _ = pairs[1]
    option["kind"] = KIND_SELF_HOSTED
    return write_inputs(root, pairs)


def _api_with_a_price(root: Path) -> Path:
    pairs = two_good_pairs()
    option, _ = pairs[1]
    option["kind"] = KIND_API
    option["hourly_usd"] = "1.006"
    return write_inputs(root, pairs)


def _hourly_price_of_zero(root: Path) -> Path:
    first, first_report = api_pair("cheap-api", cost_per_turn="0.0010")
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="0", turns=200)
    return write_inputs(root, [(first, first_report), (gpu, gpu_report)])


def _hourly_price_as_a_json_number(root: Path) -> Path:
    first, first_report = api_pair("cheap-api", cost_per_turn="0.0010")
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=200)
    gpu["hourly_usd"] = 1.006
    return write_inputs(root, [(first, first_report), (gpu, gpu_report)])


def _unknown_kind(root: Path) -> Path:
    pairs = two_good_pairs()
    option, _ = pairs[1]
    option["kind"] = "local"
    return write_inputs(root, pairs)


def _unknown_option_key(root: Path) -> Path:
    pairs = two_good_pairs()
    option, _ = pairs[1]
    option["hourly_cost"] = "1.006"
    return write_inputs(root, pairs)


def _another_set_name(root: Path) -> Path:
    pairs = two_good_pairs()
    _, report = pairs[1]
    assert report is not None
    report["set_name"] = "helpdesk-golden-v2.jsonl"
    return write_inputs(root, pairs)


def _a_run_that_did_not_move_forward(root: Path) -> Path:
    pairs = two_good_pairs()
    _, report = pairs[1]
    assert report is not None
    report["finished_at"] = report["started_at"]
    return write_inputs(root, pairs)


def _a_self_hosted_run_of_no_cases(root: Path) -> Path:
    first, first_report = api_pair("cheap-api", cost_per_turn="0.0010")
    gpu, gpu_report = self_hosted_pair("gpu", hourly_usd="1.006", turns=0)
    return write_inputs(root, [(first, first_report), (gpu, gpu_report)])


def _a_report_that_is_not_json(root: Path) -> Path:
    pairs = two_good_pairs()
    path = write_inputs(root, pairs)
    (root / "strong-api.json").write_text("{oops", encoding="utf-8")
    return path


def _a_manifest_that_is_not_json(root: Path) -> Path:
    path = write_inputs(root, two_good_pairs())
    path.write_text("{oops", encoding="utf-8")
    return path


def _a_manifest_that_is_not_there(root: Path) -> Path:
    write_inputs(root, two_good_pairs())
    return root / "nowhere.json"


def _a_manifest_of_another_schema(root: Path) -> Path:
    return write_json(root / MANIFEST_NAME, {"set_name": "helpdesk-golden-v1.jsonl"})


REFUSALS: list[tuple[str, Build]] = [
    ("an option id outside the pattern", _bad_option_id),
    ("two options sharing an id", _duplicate_option_id),
    ("a manifest of one option", _one_option),
    ("self_hosted with no hourly_usd", _self_hosted_without_a_price),
    ("api carrying an hourly_usd", _api_with_a_price),
    ("an hourly price of zero", _hourly_price_of_zero),
    ("an hourly price as a JSON number", _hourly_price_as_a_json_number),
    ("a kind outside the two words", _unknown_kind),
    ("an unknown key on an option", _unknown_option_key),
    ("a report for another golden set", _another_set_name),
    ("a run whose clock did not move", _a_run_that_did_not_move_forward),
    ("a self_hosted run of no cases", _a_self_hosted_run_of_no_cases),
    ("a report that is not JSON", _a_report_that_is_not_json),
    ("a manifest that is not JSON", _a_manifest_that_is_not_json),
    ("a manifest that is not there", _a_manifest_that_is_not_there),
    ("a manifest of another schema", _a_manifest_of_another_schema),
]
"""Every refusal `evals/tradeoff.py` can end in, driven through the command."""


def test_the_happy_path_exits_zero_with_both_files_under_their_pinned_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a subcommand wired to the wrong function or dispatched to `_run` by the
    `return _run(options)` fallthrough at the end of `main`, both of which answer a code
    that is not 0 with no tradeoff files written. Also catches an output named from the set
    or the date: `make tradeoff` and the console read these two fixed names."""
    out = tmp_path / "out"
    out.mkdir()
    manifest = good_manifest(tmp_path / "inputs")

    code = run_tradeoff(manifest, out)

    printed = capsys.readouterr()
    assert code == EXIT_OK
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [TRADEOFF_JSON_NAME, TRADEOFF_MD_NAME]
    )
    assert METHODOLOGY_HEADING in (out / TRADEOFF_MD_NAME).read_text(encoding="utf-8")
    assert json.loads((out / TRADEOFF_JSON_NAME).read_text(encoding="utf-8"))["winner"]
    assert "Traceback" not in printed.err


@pytest.mark.parametrize(
    "build", [build for _, build in REFUSALS], ids=[name for name, _ in REFUSALS]
)
def test_every_refusal_exits_two_with_one_line_on_stderr_and_an_empty_out_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], build: Build
) -> None:
    """Catches the three ways a refusal goes wrong at the shell. An unhandled
    `TradeoffInputInvalid` or `ValidationError` exits 1 with a traceback, which a caller
    reads as a real verdict; a refusal printed on stdout lands in the stream the table is
    read from; and a command that renders before it validates leaves a half-written
    `tradeoff.md` in `--out` for the next reader to quote. `--out` is created by this test,
    so the emptiness assertion cannot pass by the directory never existing."""
    out = tmp_path / "out"
    out.mkdir()
    root = tmp_path / "inputs"
    root.mkdir()
    manifest = build(root)

    code = run_tradeoff(manifest, out)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert list(out.iterdir()) == []
    assert printed.out.strip() == ""
    assert printed.err.strip() != ""
    assert printed.err.strip().count("\n") == 0
    assert "Traceback" not in printed.err


def test_a_missing_report_path_exits_two_and_names_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a message naming only the manifest: with four options in the file, the path
    the operator has to fix is the report that was not there, and a line saying "the
    manifest is invalid" sends them to the wrong file."""
    out = tmp_path / "out"
    out.mkdir()
    root = tmp_path / "inputs"
    first, first_report = api_pair("cheap-api", cost_per_turn="0.0010")
    absent, _ = api_pair("absent-api", cost_per_turn="0.0100")
    manifest = write_inputs(root, [(first, first_report), (absent, None)])

    code = run_tradeoff(manifest, out)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert str(root / "absent-api.json") in printed.err
    assert list(out.iterdir()) == []


def test_an_out_directory_that_cannot_be_written_exits_two_naming_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches the `PermissionError` from a read-only mount escaping as a traceback after
    every report has been read and every number computed: the operator needs the path and an
    exit code, not a stack trace. The inputs are all valid here, so only the write can
    produce the 2."""
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o500)
    manifest = good_manifest(tmp_path / "inputs")
    try:
        code = run_tradeoff(manifest, out)
    finally:
        out.chmod(0o700)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert str(out) in printed.err
    assert "Traceback" not in printed.err


def test_a_second_run_over_the_same_out_overwrites_both_files(tmp_path: Path) -> None:
    """Catches an append-mode write, which would leave `tradeoff.md` with two tables and two
    conclusions - the published document would then carry the previous run's winner under
    the new one - and would leave `tradeoff.json` as two concatenated documents that no
    reader can parse."""
    out = tmp_path / "out"
    out.mkdir()
    manifest = good_manifest(tmp_path / "inputs")

    first_code = run_tradeoff(manifest, out)
    second_code = run_tradeoff(manifest, out)

    markdown = (out / TRADEOFF_MD_NAME).read_text(encoding="utf-8")
    assert (first_code, second_code) == (EXIT_OK, EXIT_OK)
    assert markdown.count(METHODOLOGY_HEADING) == 1
    assert markdown.count(CONCLUSION_HEADING) == 1
    assert json.loads((out / TRADEOFF_JSON_NAME).read_text(encoding="utf-8"))["winner"]


@pytest.mark.parametrize(
    "argv",
    [["tradeoff"], ["tradeoff", "--manifest", "m.json"], ["tradeoff", "--out", "d"]],
    ids=["neither flag", "no --out", "no --manifest"],
)
def test_both_flags_are_required(argv: list[str]) -> None:
    """Catches a flag given a default - `--out` defaulting to `docs`, say - which would let
    a mistyped command rewrite a published table the operator never named. Argparse's own
    `required=True` is what refuses these, and it exits 2 like every other usage error."""
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    assert raised.value.code == EXIT_USAGE
