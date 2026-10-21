"""`cli/evals.py`: `ops-eval conversations`, `calibrate` and `gate`, and the end of `run`.

No network and no sleep. Every HTTP client the commands open comes from one seam,
`app.cli.evals_commands.new_http_client() -> httpx.Client`, which these tests replace with a
client over an `httpx.MockTransport` that routes by host: `helpdesk.example` is the stub
target, `gateway.example` the judge, `pushgateway.example` the metrics push.

Contract these tests are written against:

* `ops-eval conversations --target URL --cases PATH --out DIR [--run-id ID]
  [--run-kind ci|nightly|manual] [--judge-url URL] [--pushgateway-url URL]`.
  `--out` receives `<run-id>.jsonl`, `report.json` and `report.md`. The exit code is `0`
  whenever the run completed (failed cases are the gate's business, not the run's) and `2`
  for anything that stopped it: a bad argument, an unreadable cases file, a reset that
  failed. The one summary line goes to stdout; every refusal is one line on stderr. Nothing
  is sent to the target before the arguments and the cases file have been checked.
  The judge is on only when `--judge-url` and `OPSCONSOLE_EVAL_JUDGE_KEY` are both present.
* `ops-eval run` no longer exists: argparse refuses it, exit `2`.
* `ops-eval calibrate --labels PATH --judge-url URL --out DIR`: writes `calibration.json`
  and exits `0`; fewer than 40 labels is exit `2` with no judge request.
* `ops-eval gate --report PATH --baseline PATH [--calibration PATH] [--comment PATH]`:
  exit `0` passed, `1` failed, `2` for an unreadable input. stdout is one verdict line
  (`gate passed...` / `gate FAILED...`), stderr one line per reason.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.cli import evals as cli
from app.cli import evals_commands
from app.cli.evals import EXIT_FAILED, EXIT_OK, EXIT_USAGE, UNJUDGED_LINE
from app.evals.calibration import CALIBRATION_JSON_NAME
from app.evals.report import REPORT_JSON_NAME, REPORT_MD_NAME, RunReport, build_report, report_json

from .builders import StubTarget, answer, call, case_document, case_line, turn, write_cases

HELPDESK = "https://helpdesk.example"
GATEWAY = "https://gateway.example"
PUSHGATEWAY = "http://pushgateway.example:9091"
RUN_ID = "run-one"


class World:
    """The three hosts a command can reach, behind one transport, with every request kept."""

    def __init__(self, target: StubTarget) -> None:
        self.target = target
        self.judge_requests: list[httpx.Request] = []
        self.push_requests: list[httpx.Request] = []
        self.push_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "helpdesk.example":
            return self.target(request)
        if host == "gateway.example":
            self.judge_requests.append(request)
            text = json.dumps({"reason": "ok", "verdict": "pass"})
            return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})
        if host == "pushgateway.example":
            self.push_requests.append(request)
            return httpx.Response(self.push_status)
        raise AssertionError(f"unexpected host {host}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            evals_commands,
            "new_http_client",
            lambda: httpx.Client(transport=httpx.MockTransport(self)),
        )


def two_cases(directory: Path) -> Path:
    """conv-001 expects `search_policy`; conv-002 expects nothing."""
    return write_cases(
        directory / "cases.jsonl",
        case_document(
            "conv-001",
            turns=("what is the rule",),
            tool_calls=({"name": "search_policy", "args_subset": {}},),
            graded=((1, "cites a policy?"),),
        ),
        case_document("conv-002", turns=("hello",), graded=((1, "polite?"),)),
    )


def good_target() -> StubTarget:
    return StubTarget(
        {
            "conv-001": answer("conv-001", turn(1, "The rule says.", call("search_policy"))),
            "conv-002": answer("conv-002", turn(1, "Hello!")),
        }
    )


def conversations(
    out: Path,
    cases: Path,
    *extra: str,
    run_id: str = RUN_ID,
    target: str = HELPDESK,
) -> int:
    return cli.main(
        [
            "conversations",
            "--target",
            target,
            "--cases",
            str(cases),
            "--out",
            str(out),
            "--run-id",
            run_id,
            *extra,
        ]
    )


def read_report(out: Path) -> RunReport:
    return RunReport.model_validate_json((out / REPORT_JSON_NAME).read_text(encoding="utf-8"))


def test_the_exit_codes_are_the_published_ones() -> None:
    """Catches a code that moved: the workflow keys off these integers."""
    assert (EXIT_OK, EXIT_FAILED, EXIT_USAGE) == (0, 1, 2)


def test_ops_eval_conversations_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a command that plays the cases and writes nothing, writes only one of the three
    files, or prints more than the one summary line. Both cases pass, so the score is 100."""
    world = World(good_target())
    world.install(monkeypatch)
    out = tmp_path / "out"

    code = conversations(out, two_cases(tmp_path))

    printed = capsys.readouterr()
    assert code == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{RUN_ID}.jsonl", REPORT_JSON_NAME, REPORT_MD_NAME]
    )
    report = read_report(out)
    assert report.metrics["check_score"] == 100.0
    assert [c.case_id for c in report.cases] == ["conv-001", "conv-002"]
    assert len(printed.out.strip().splitlines()) == 1
    assert "check_score" in printed.out
    assert world.target.resets == 1


def test_failed_checks_do_not_change_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a run that exits 1 on a failed case: the baseline run itself has failures, and
    the gate (not the run) decides whether a score is acceptable. conv-001 never calls the
    tool it must, so check_score is 50."""
    target = StubTarget(
        {
            "conv-001": answer("conv-001", turn(1, "No idea.")),
            "conv-002": answer("conv-002", turn(1, "Hello!")),
        }
    )
    World(target).install(monkeypatch)
    out = tmp_path / "out"

    code = conversations(out, two_cases(tmp_path))

    assert code == EXIT_OK
    assert read_report(out).metrics["check_score"] == 50.0


def test_ops_eval_run_is_gone() -> None:
    """Catches the old single-reply command surviving as an alias: it would read a golden set
    that no longer exists and write a report in the old format."""
    with pytest.raises(SystemExit) as raised:
        cli.main(
            ["run", "--target", HELPDESK, "--set", "x.jsonl", "--tenant", "acme", "--out", "o"]
        )

    assert raised.value.code == 2


@pytest.mark.parametrize(
    ("target", "run_id"),
    [
        ("helpdesk.example", RUN_ID),
        ("file:///etc/passwd", RUN_ID),
        (HELPDESK, "Run One"),
        (HELPDESK, RUN_ID + "\n"),
        (HELPDESK, "../x"),
    ],
    ids=["no scheme", "file scheme", "space", "trailing newline", "path"],
)
def test_a_bad_argument_stops_the_run_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    run_id: str,
) -> None:
    """Catches arguments checked after the reset, when the seed is already spent: the target
    must be an absolute http(s) URL and the run id a slug (it names a file). Exit 2, one
    stderr line, no request at all, no file."""
    world = World(good_target())
    world.install(monkeypatch)
    out = tmp_path / "out"

    code = conversations(out, two_cases(tmp_path), run_id=run_id, target=target)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert world.target.requests == []
    assert printed.err.strip() != "" and "Traceback" not in printed.err
    assert not out.exists() or not list(out.glob("*.jsonl"))


def test_a_missing_or_malformed_cases_file_exits_two_before_the_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a cases file read after the reset, and a traceback over a typo: the path is
    named on stderr and the target is never touched."""
    world = World(good_target())
    world.install(monkeypatch)
    bad = write_cases(tmp_path / "bad.jsonl", {"id": "conv-001"})

    missing = conversations(tmp_path / "out", tmp_path / "nope.jsonl")
    malformed = conversations(tmp_path / "out", bad)

    printed = capsys.readouterr()
    assert (missing, malformed) == (EXIT_USAGE, EXIT_USAGE)
    assert "nope.jsonl" in printed.err and "bad.jsonl" in printed.err
    assert world.target.requests == []


def test_a_reset_that_fails_exits_two_and_names_the_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a run that goes on over an unreset seed: a number measured there cannot be
    compared with the baseline."""
    target = good_target()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/test/reset":
            return httpx.Response(404)
        return target(request)

    monkeypatch.setattr(
        evals_commands,
        "new_http_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    code = conversations(tmp_path / "out", two_cases(tmp_path))

    assert code == EXIT_USAGE
    assert "/admin/test/reset" in capsys.readouterr().err
    assert target.played == []


def test_a_second_invocation_with_the_same_run_id_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a command that resets or replays on a restart: the second invocation sends no
    request at all (both cases are saved), and the report it writes still covers both."""
    first = World(good_target())
    first.install(monkeypatch)
    cases = two_cases(tmp_path)
    out = tmp_path / "out"
    conversations(out, cases)
    second = World(StubTarget({}))
    second.install(monkeypatch)

    code = conversations(out, cases)

    assert code == EXIT_OK
    assert second.target.requests == []
    assert len(read_report(out).cases) == 2


def test_the_judge_is_on_only_with_a_url_and_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a judge that runs without a key (three 401s per reply) or silently does not
    run when asked: with a URL but no key the run says so once and judge_score is null; with
    both, each of the 2 graded replies costs 3 judge requests and judge_score is 100."""
    keyless = World(good_target())
    keyless.install(monkeypatch)
    monkeypatch.delenv("OPSCONSOLE_EVAL_JUDGE_KEY", raising=False)
    conversations(tmp_path / "a", two_cases(tmp_path), "--judge-url", GATEWAY)
    unjudged = capsys.readouterr().err
    keyed = World(good_target())
    keyed.install(monkeypatch)
    monkeypatch.setenv("OPSCONSOLE_EVAL_JUDGE_KEY", "sk-not-real")

    conversations(tmp_path / "b", two_cases(tmp_path), "--judge-url", GATEWAY)

    assert UNJUDGED_LINE in unjudged
    assert keyless.judge_requests == []
    assert read_report(tmp_path / "a").metrics["judge_score"] is None
    assert len(keyed.judge_requests) == 6
    assert read_report(tmp_path / "b").metrics["judge_score"] == 100.0


def test_a_push_sends_the_scores_and_a_failed_push_never_changes_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a push of the old names, a push that is never sent, and a metrics side channel
    that fails the run: the Pushgateway answering 500 is reported on stderr and the exit
    code stays 0."""
    world = World(good_target())
    world.install(monkeypatch)

    ok = conversations(tmp_path / "a", two_cases(tmp_path), "--pushgateway-url", PUSHGATEWAY)
    sent = world.push_requests[0].content.decode("utf-8")
    world.target.requests.clear()
    world.push_status = 500
    failed = conversations(
        tmp_path / "b", two_cases(tmp_path), "--pushgateway-url", PUSHGATEWAY, run_id="run-two"
    )

    assert (ok, failed) == (EXIT_OK, EXIT_OK)
    assert "gateway_eval_check_score 100.0" in sent
    assert "decision_accuracy" not in sent
    assert "pushed" in capsys.readouterr().err


# --- gate --------------------------------------------------------------------------


def write_gate_inputs(
    directory: Path,
    *,
    check: float,
    baseline: float,
    kappa: float | None = None,
    judge: float = 40.0,
) -> tuple[Path, Path, Path | None]:
    report = build_report(
        [case_line("conv-001", passed=True)],
        started_at=datetime(2026, 10, 22, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, 9, 20, tzinfo=UTC),
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target=HELPDESK,
    )
    report = report.model_copy(
        update={"metrics": {**report.metrics, "check_score": check, "judge_score": judge}}
    )
    report_path = directory / "report.json"
    report_path.write_text(report_json(report), encoding="utf-8")
    baseline_path = directory / "conversation-baseline.json"
    baseline_path.write_text(
        json.dumps({"check_score": baseline, "judge_score": 60.0}), encoding="utf-8"
    )
    calibration_path: Path | None = None
    if kappa is not None:
        calibration_path = directory / CALIBRATION_JSON_NAME
        calibration_path.write_text(
            json.dumps(
                {
                    "judge_prompt_version": "c" * 64,
                    "labels_name": "conversation-labels.jsonl",
                    "labels_sha": "d" * 64,
                    "pairs": 40,
                    "kappa": kappa,
                    "trusted": kappa >= 0.7,
                }
            ),
            encoding="utf-8",
        )
    return report_path, baseline_path, calibration_path


def gate(report: Path, baseline: Path, calibration: Path | None, comment: Path | None) -> int:
    argv = ["gate", "--report", str(report), "--baseline", str(baseline)]
    if calibration is not None:
        argv += ["--calibration", str(calibration)]
    if comment is not None:
        argv += ["--comment", str(comment)]
    return cli.main(argv)


def test_gate_exits_zero_on_the_baseline_and_one_below_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a gate command that always exits 0, or one that exits non-zero on an equal
    score. The comment file is written with the score in it. Failing prints one stdout line
    and the reason on stderr."""
    comment = tmp_path / "comment.md"
    (tmp_path / "eq").mkdir()
    (tmp_path / "low").mkdir()
    report, baseline, _ = write_gate_inputs(tmp_path / "eq", check=80.0, baseline=80.0)
    low_report, low_baseline, _ = write_gate_inputs(tmp_path / "low", check=79.0, baseline=80.0)

    passed = gate(report, baseline, None, comment)
    passed_out = capsys.readouterr()
    failed = gate(low_report, low_baseline, None, None)
    failed_out = capsys.readouterr()

    assert (passed, failed) == (EXIT_OK, EXIT_FAILED)
    assert passed_out.out.startswith("gate passed")
    assert "check_score" in comment.read_text(encoding="utf-8")
    assert failed_out.out.startswith("gate FAILED")
    assert "check_score" in failed_out.err


def test_gate_reads_the_calibration_file_to_decide_whether_the_judge_counts(
    tmp_path: Path,
) -> None:
    """Catches a gate that ignores `--calibration`: the same low judge_score (40 against 60)
    fails with kappa 0.7 and passes with 0.69. check_score equals its baseline in both."""
    (tmp_path / "t").mkdir()
    (tmp_path / "u").mkdir()
    trusted = write_gate_inputs(tmp_path / "t", check=80.0, baseline=80.0, kappa=0.7)
    untrusted = write_gate_inputs(tmp_path / "u", check=80.0, baseline=80.0, kappa=0.69)

    assert gate(*trusted, None) == EXIT_FAILED
    assert gate(*untrusted, None) == EXIT_OK


def test_gate_exits_two_on_an_unreadable_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a gate that treats a missing baseline or report as a pass, or a traceback: exit
    2 with the path on stderr, and no comment left behind."""
    comment = tmp_path / "comment.md"

    code = gate(tmp_path / "no-report.json", tmp_path / "no-baseline.json", None, comment)

    assert code == EXIT_USAGE
    assert "no-report.json" in capsys.readouterr().err
    assert not comment.exists()


# --- calibrate ---------------------------------------------------------------------


def write_labels(path: Path, count: int) -> Path:
    lines = [
        json.dumps(
            {
                "case_id": f"conv-{n % 33 + 1:03d}",
                "turn": 1,
                "rubric": "r",
                "reply": f"reply {n}",
                "verdict": "pass",
                "votes": ["pass", "pass", "pass"],
                "reason": "because",
                "source": "model-reference",
            }
        )
        for n in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def calibrate(labels: Path, out: Path) -> int:
    return cli.main(
        ["calibrate", "--labels", str(labels), "--judge-url", GATEWAY, "--out", str(out)]
    )


def test_calibrate_writes_the_artifact_and_refuses_too_few_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a calibration that spends on 39 labels, one that writes `calibration.json` it
    cannot stand behind, and one that writes nothing at 40. The judge answers pass to
    everything and the labels are all pass: kappa is undefined, so the file says not
    trusted, but it exists. The 39-label file is refused with no judge request."""
    world = World(good_target())
    world.install(monkeypatch)
    monkeypatch.setenv("OPSCONSOLE_EVAL_JUDGE_KEY", "sk-not-real")
    few = write_labels(tmp_path / "few.jsonl", 39)

    refused = calibrate(few, tmp_path / "few-out")
    spent = len(world.judge_requests)
    done = calibrate(write_labels(tmp_path / "ok.jsonl", 40), tmp_path / "ok-out")

    document = json.loads((tmp_path / "ok-out" / CALIBRATION_JSON_NAME).read_text())
    assert (refused, done) == (EXIT_USAGE, EXIT_OK)
    assert spent == 0
    assert not (tmp_path / "few-out" / CALIBRATION_JSON_NAME).exists()
    assert document["pairs"] == 40
    assert document["kappa"] is None
    assert document["trusted"] is False
    assert capsys.readouterr().err.strip() != ""
