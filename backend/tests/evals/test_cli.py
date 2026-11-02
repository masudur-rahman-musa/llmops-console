"""`cli/evals.py`: `ops-eval conversations` and `gate`.

No network and no sleep. Every HTTP client the commands open comes from one seam,
`app.cli.evals_commands.new_http_client() -> httpx.Client`, which these tests replace with a
client over an `httpx.MockTransport` that routes by host: `helpdesk.example` is the stub
help desk, `gateway.example` the gateway that serves the simulated person and the judge.

Contract these tests are written against:

* `ops-eval conversations --target URL --cases PATH --out DIR [--run-id ID]
  [--run-kind ci|nightly|manual] [--gateway-url URL] [--plays N] [--reset-command CMD]
  [--pushgateway-url URL]`. `--plays` is 1 to 10 (default 3), checked before anything is
  sent. `--out` receives `report.json`, `report.md`, `needs-a-look.md` and `comment.md`.
  Exit `0` whenever the run completed, `2` for anything that stopped it.
* `--judge-url`, `--repeats` and the `label` command no longer exist:
  argparse refuses them, exit `2`.
* `ops-eval gate --report PATH --baseline PATH [--comment PATH]`: exit `0` passed, `1`
  failed, `2` for an unreadable input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from app.cli import evals as cli
from app.cli import evals_args, evals_commands
from app.cli.evals import EXIT_FAILED, EXIT_OK, EXIT_USAGE
from app.evals.judge import prompt_version
from app.evals.report import REPORT_JSON_NAME, RunReport

from .builders import StubHelpdesk, answer, case_document, write_cases
from .label_builders import label_document, write_labels

HELPDESK = "https://helpdesk.example"
GATEWAY = "https://gateway.example"
RUN_ID = "run-one"
DONE = json.dumps({"message": "thanks [done]", "yes": False})
JUDGE_NOT_APPLICABLE = json.dumps({"answer": "does_not_apply", "quote": "", "reason": "r"})


class Hosts:
    """The two hosts a command can reach, behind one transport, with every request kept."""

    def __init__(self, helpdesk: StubHelpdesk) -> None:
        self.helpdesk = helpdesk
        self.gateway_requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "helpdesk.example":
            return self.helpdesk(request)
        if request.url.host == "gateway.example":
            self.gateway_requests.append(request)
            judged = json.loads(request.content)["model"] == "judge"
            content = JUDGE_NOT_APPLICABLE if judged else DONE
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        raise AssertionError(f"unexpected host {request.url.host}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            evals_commands,
            "new_http_client",
            lambda: httpx.Client(transport=httpx.MockTransport(self)),
        )
        monkeypatch.setenv("OPSCONSOLE_EVAL_JUDGE_KEY", "sk-not-real")


def one_case(directory: Path) -> Path:
    return write_cases(directory / "cases.jsonl", case_document("case-001"))


def hosts() -> Hosts:
    return Hosts(StubHelpdesk({"case-001": [answer("Hello.")]}))


def conversations(out: Path, cases: Path, *extra: str, target: str = HELPDESK) -> int:
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
            RUN_ID,
            "--gateway-url",
            GATEWAY,
            *extra,
        ]
    )


def test_the_exit_codes_are_the_published_ones() -> None:
    """Catches a code that moved: the workflow keys off these integers."""
    assert (EXIT_OK, EXIT_FAILED, EXIT_USAGE) == (0, 1, 2)


def test_ops_eval_conversations_writes_the_four_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a command that plays the cases and writes only some of the files, or prints
    more than the one summary line."""
    world = hosts()
    world.install(monkeypatch)
    out = tmp_path / "out"

    code = conversations(out, one_case(tmp_path), "--plays", "1")

    assert code == EXIT_OK
    assert {"report.json", "report.md", "needs-a-look.md", "comment.md"} <= {
        p.name for p in out.iterdir()
    }
    report = RunReport.model_validate_json((out / REPORT_JSON_NAME).read_text(encoding="utf-8"))
    assert "safety" in report.metrics
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


@pytest.mark.parametrize(
    ("plays", "code", "resets"),
    [("0", EXIT_USAGE, 0), ("11", EXIT_USAGE, 0), ("1", EXIT_OK, 1), ("10", EXIT_OK, 10)],
    ids=["zero", "one over", "lowest", "highest"],
)
def test_plays_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plays: str, code: int, resets: int
) -> None:
    """Catches a `--plays` range that admits 0 or 11, or one that stops at 9: the edges are
    exact. A refusal sends nothing to the help desk, and 10 plays of one case reset ten
    times."""
    world = hosts()
    world.install(monkeypatch)

    result = conversations(tmp_path / "out", one_case(tmp_path), "--plays", plays)

    assert result == code
    assert world.helpdesk.resets == resets


def test_plays_default_to_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches a default of one play: the safety score is meant over three."""
    world = hosts()
    world.install(monkeypatch)

    conversations(tmp_path / "out", one_case(tmp_path))

    assert world.helpdesk.resets == 3


@pytest.mark.parametrize(
    "argv",
    [
        ["conversations", "--judge-url", GATEWAY],
        ["conversations", "--repeats", "2"],
        ["label"],
    ],
    ids=["judge-url", "repeats", "label"],
)
def test_the_removed_flags_and_commands_are_refused(argv: list[str]) -> None:
    """Catches an old flag or command surviving beside the new ones: each would read a file
    or a score that no longer exists. Required flags are given so that only the removed
    name can cause the refusal."""
    required = ["--target", HELPDESK, "--cases", "c.jsonl", "--out", "o"]
    full = argv[:1] + required + argv[1:] if argv[0] == "conversations" else argv

    with pytest.raises(SystemExit) as raised:
        cli.main(full)

    assert raised.value.code == 2


def test_the_reset_command_and_gateway_url_are_read() -> None:
    """Catches a `--reset-command` or `--gateway-url` that is accepted but dropped: the
    parsed options carry both values."""
    options = evals_args._parser().parse_args(
        [
            "conversations",
            "--target",
            HELPDESK,
            "--cases",
            "c.jsonl",
            "--out",
            "o",
            "--gateway-url",
            GATEWAY,
            "--reset-command",
            "infra/evals/reset_helpdesk.sh",
        ]
    )

    assert options.gateway_url == GATEWAY
    assert options.reset_command == "infra/evals/reset_helpdesk.sh"


@pytest.mark.parametrize(
    "target", ["helpdesk.example", "file:///etc/passwd"], ids=["no scheme", "file scheme"]
)
def test_a_bad_target_stops_the_run_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    """Catches arguments checked after the reset, when the seed is already spent."""
    world = hosts()
    world.install(monkeypatch)

    code = conversations(tmp_path / "out", one_case(tmp_path), target=target)

    assert code == EXIT_USAGE
    assert world.helpdesk.events == []


def test_the_gate_exit_codes(tmp_path: Path) -> None:
    """Catches a gate command that exits 0 on a breach: 99 safety is exit 1, a full run on
    its baseline is 0, and a missing baseline file is 2."""
    from datetime import UTC, datetime

    from app.evals.report import build_report, report_json

    from .builders import play_record

    base = build_report(
        (play_record("case-001", 1, "pass"),),
        started_at=datetime(2026, 11, 2, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 11, 2, 9, 5, tzinfo=UTC),
        set_name="cases.jsonl",
        set_sha="a" * 40,
        set_sha_source="commit",
        run_kind="ci",
        git_sha=None,
        tenant="all",
        target=HELPDESK,
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"safety": 100, "outcome_every_play": 50}), encoding="utf-8")

    def gate(safety: float) -> int:
        report = base.model_copy(update={"metrics": {**base.metrics, "safety": safety}})
        path = tmp_path / "report.json"
        path.write_text(report_json(report), encoding="utf-8")
        return cli.main(["gate", "--report", str(path), "--baseline", str(baseline)])

    assert gate(100.0) == EXIT_OK
    assert gate(99.0) == EXIT_FAILED
    assert (
        cli.main(["gate", "--report", str(tmp_path / "report.json"), "--baseline", "nope.json"])
        == EXIT_USAGE
    )


# --- calibrate ----------------------------------------------------------------------------

JUDGE_YES = json.dumps({"answer": "yes", "quote": "q", "reason": "r"})


class JudgingHosts(Hosts):
    """The same two hosts, but the gateway answers a `judge` request with `yes`."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "gateway.example":
            self.gateway_requests.append(request)
            content = JUDGE_YES if json.loads(request.content)["model"] == "judge" else DONE
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        return super().__call__(request)


def calibrate_argv(labels: Path, out: Path) -> list[str]:
    return [
        "calibrate",
        "--labels",
        str(labels),
        "--gateway-url",
        GATEWAY,
        "--out",
        str(out),
    ]


def test_calibrate_writes_one_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches a calibrate that writes a directory of files or none, asks the judge about the
    model's own labels, or leaves out the hashes: `--out` is the one file, the judge is asked
    once per musa row (2 of the 3 rows), and the file names the judge prompt's hash and the
    labels file's hash."""
    world = JudgingHosts(StubHelpdesk({}))
    world.install(monkeypatch)
    labels = write_labels(
        tmp_path / "labels.jsonl",
        label_document("case-001", 1, "plain_words", answer="yes"),
        label_document("case-002", 1, "plain_words", answer="yes", labelled_by="model"),
        label_document("case-003", 1, "plain_words", answer="no"),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "calibration.json"

    code = cli.main(calibrate_argv(labels, out))

    assert code == EXIT_OK
    assert [p.name for p in out_dir.iterdir()] == ["calibration.json"]
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["judge_model"] == "anthropic/claude-sonnet-5-5"
    assert document["judge_prompt_sha256"] == prompt_version()
    assert document["labels_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert document["check_types"]["plain_words"]["labels"] == 2
    assert document["check_types"]["plain_words"]["answered"] == 2
    assert document["check_types"]["plain_words"]["agreed"] == 1
    assert len(world.gateway_requests) == 2


def test_calibrate_without_the_judge_key_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a calibrate that goes on without a key (every ask would be unanswered and the
    file would say nothing is trusted) or that writes a file anyway: exit 2, one line on
    stderr, no file, and nothing sent."""
    world = JudgingHosts(StubHelpdesk({}))
    world.install(monkeypatch)
    monkeypatch.delenv("OPSCONSOLE_EVAL_JUDGE_KEY")
    labels = write_labels(tmp_path / "labels.jsonl", label_document())
    out = tmp_path / "calibration.json"

    code = cli.main(calibrate_argv(labels, out))

    assert code == EXIT_USAGE
    assert not out.exists()
    assert len(capsys.readouterr().err.strip().splitlines()) == 1
    assert world.gateway_requests == []


def test_calibrate_with_no_musa_rows_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a calibrate that writes an empty file when only the model labelled: exit 2, no
    file, one line that says why, and the judge is never asked."""
    world = JudgingHosts(StubHelpdesk({}))
    world.install(monkeypatch)
    labels = write_labels(tmp_path / "labels.jsonl", label_document(labelled_by="model"))
    out = tmp_path / "calibration.json"

    code = cli.main(calibrate_argv(labels, out))

    assert code == EXIT_USAGE
    assert not out.exists()
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert "musa" in lines[0]
    assert world.gateway_requests == []


@pytest.mark.parametrize("content", ["not json\n", '{"case_id": "x"}\n'], ids=["junk", "short"])
def test_calibrate_with_a_bad_labels_file_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    """Catches a bad labels file that is skipped or crashes with a trace: exit 2, no file."""
    world = JudgingHosts(StubHelpdesk({}))
    world.install(monkeypatch)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(content, encoding="utf-8")
    out = tmp_path / "calibration.json"

    assert cli.main(calibrate_argv(labels, out)) == EXIT_USAGE
    assert not out.exists()
    assert cli.main(calibrate_argv(tmp_path / "missing.jsonl", out)) == EXIT_USAGE
    assert not out.exists()


def write_calibration_file(path: Path, prompt_hash: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "judge_model": "anthropic/claude-sonnet-5-5",
                "judge_prompt_sha256": prompt_hash,
                "labels_sha256": "l" * 64,
                "made_on": "2026-11-02",
                "check_types": {
                    "asks_before_change": {
                        "labels": 4,
                        "answered": 4,
                        "agreed": 4,
                        "kappa": 1.0,
                        "trusted": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("hash_of", "expected"),
    [("current", 100.0), ("stale", None)],
    ids=["same judge prompt", "older judge prompt"],
)
def test_conversations_reads_the_calibration_against_the_current_judge_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hash_of: str, expected: float | None
) -> None:
    """Catches a `--calibration` that is accepted and dropped, or trusted whatever judge prompt
    made it: with the current prompt's hash the trusted check type counts toward reply
    quality, with another hash reply quality stays empty."""
    world = JudgingHosts(StubHelpdesk({"case-001": [answer("Hello.")]}))
    world.install(monkeypatch)
    calibration = write_calibration_file(
        tmp_path / "calibration.json", prompt_version() if hash_of == "current" else "0" * 64
    )
    out = tmp_path / "out"

    code = conversations(out, one_case(tmp_path), "--plays", "1", "--calibration", str(calibration))

    assert code == EXIT_OK
    report = RunReport.model_validate_json((out / REPORT_JSON_NAME).read_text(encoding="utf-8"))
    assert report.metrics["reply_quality"] == expected


def test_conversations_without_a_calibration_has_no_reply_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a reply quality counted with no calibration given: the flag is optional and
    without it nothing is trusted."""
    world = JudgingHosts(StubHelpdesk({"case-001": [answer("Hello.")]}))
    world.install(monkeypatch)
    out = tmp_path / "out"

    assert conversations(out, one_case(tmp_path), "--plays", "1") == EXIT_OK

    report = RunReport.model_validate_json((out / REPORT_JSON_NAME).read_text(encoding="utf-8"))
    assert report.metrics["reply_quality"] is None


class SilentAliasHosts(JudgingHosts):
    """The judging hosts, but the gateway answers one alias with a 502 every time."""

    def __init__(self, helpdesk: StubHelpdesk, silent: str) -> None:
        super().__init__(helpdesk)
        self.silent = silent

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if (
            request.url.host == "gateway.example"
            and json.loads(request.content)["model"] == self.silent
        ):
            self.gateway_requests.append(request)
            return httpx.Response(502, json={"error": {"code": "provider_error"}})
        return super().__call__(request)


@pytest.mark.parametrize(
    ("alias", "named"),
    [("simulated-person", "simulated person"), ("judge", "judge")],
    ids=["person", "judge"],
)
def test_a_person_or_judge_with_no_answer_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    alias: str,
    named: str,
) -> None:
    """Catches a run that passes while the simulated person or the judge never answered:
    every play then ends after one reply, or every check is left open, and the report reads
    like a real result. The run stops with exit 1, says which one went silent, and writes no
    report for the gate to read."""
    world = SilentAliasHosts(
        StubHelpdesk({"case-001": [answer("Shall I file it?"), answer("Filed.")]}), alias
    )
    world.install(monkeypatch)
    out = tmp_path / "out"

    code = conversations(out, one_case(tmp_path), "--plays", "1")

    assert code == EXIT_FAILED
    assert named in capsys.readouterr().err
    assert not (out / REPORT_JSON_NAME).exists()
