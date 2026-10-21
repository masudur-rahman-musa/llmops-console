"""Command handlers for `ops-eval`.

The six functions that do the actual work of each subcommand: `_conversations`,
`_label`, `_calibrate`, `_gate`, `_tradeoff` and `_baseline_guard`. Each takes the parsed
`argparse.Namespace` and returns the process exit code. Every HTTP client a command opens
comes from one seam, `new_http_client()`, so a test replaces the network with one
`httpx.MockTransport`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

import app.cli.evals_args as _eargs
from app.cli.evals_args import UsageError
from app.core.config import Settings
from app.core.validation import IdentifierInvalid, checked_identifier
from app.evals.baseline_guard import GUARD_PASSED_LINE, check, offence_line
from app.evals.calibration import (
    CalibrationRefused,
    CalibrationReport,
    CalibrationSetInvalid,
    calibrate,
    load_labels,
    read_calibration,
    write_calibration,
)
from app.evals.conversations import ConversationSetInvalid, load_cases
from app.evals.gate import (
    GateInputInvalid,
    evaluate,
    gate_comment,
    load_baseline,
    load_report,
    verdict_line,
)
from app.evals.judge import JudgeClient, judge_prompt_version
from app.evals.labels import (
    LABEL_TARGET,
    GatewayLabelGrader,
    LabelCandidate,
    candidates_from,
    make_labels,
    write_labels,
)
from app.evals.push import push_metrics
from app.evals.report import ALL_TENANTS, REPORT_JSON_NAME, RunReport, build_report, write_reports
from app.evals.runner import CaseLine, load_run, run_conversations
from app.evals.target import TargetClient, TargetUnreachable
from app.evals.tradeoff import (
    TradeoffInputInvalid,
    build_table,
    load_manifest,
    write_tradeoff,
)
from app.storage.db import build_pool
from app.storage.eval_store import PostgresEvalStore, rows_from_report

EXIT_OK: int = 0
EXIT_FAILED: int = 1
EXIT_USAGE: int = 2

UNSTORED_LINE: str = (
    "no database URL configured: this run's rows were not stored, and its report files were"
)

UNJUDGED_LINE: str = (
    "no judge URL or no OPSCONSOLE_EVAL_JUDGE_KEY: this run judged nothing, and judge_score is null"
)


def new_http_client() -> httpx.Client:
    """The one place a command makes an HTTP client.

    A seam, not a convenience: the command tests replaces it with a client over a
    `MockTransport` that routes by host, so no command test opens a socket.
    """
    return httpx.Client()


def _summary(report: RunReport) -> str:
    """The one line `conversations` prints, so a CI log carries the real numbers."""
    metrics = report.metrics
    return (
        f"cases {metrics['cases']} · passed {metrics['passed']} · failed {metrics['failed']} · "
        f"check_score {metrics['check_score']} · judge_score {metrics['judge_score']}"
    )


def _store_rows(report: RunReport, out: Path) -> str | None:
    """Store the run and its cases, and answer what went wrong, or `None`.

    Called AFTER both report files are written: the files are what survives a failed
    insert, and the gate reads them off disk. The failure is answered rather than raised
    so no storage complaint becomes a traceback over a run already paid for. With no
    `OPSCONSOLE_DATABASE_URL` nothing is stored, `UNSTORED_LINE` is printed and the run
    keeps its own exit code.
    """
    database_url = Settings().database_url
    if database_url is None:
        print(UNSTORED_LINE, file=sys.stderr)
        return None
    try:
        run_row, case_rows = rows_from_report(report, report_path=out / REPORT_JSON_NAME)
        PostgresEvalStore(build_pool(database_url)).write(run_row, case_rows)
    except Exception as exc:
        return f"the eval store did not take this run: {exc}"
    return None


def _judge(judge_url: str | None, client: httpx.Client) -> JudgeClient | None:
    """The judge this run grades with, or `None` with one line on stderr.

    Both halves are required: a URL with no key would spend three 401s per graded
    reply, and a key with no URL has nowhere to send them.
    """
    api_key = Settings().eval_judge_key
    if judge_url is None or not api_key:
        print(UNJUDGED_LINE, file=sys.stderr)
        return None
    return JudgeClient(judge_url, client=client, api_key=api_key)


def _judged_prompt_version(lines: tuple[CaseLine, ...]) -> str | None:
    """The prompt's sha256 when a reply was actually graded, and `None` otherwise.

    Read off the RECORDS: a run whose every judge sample failed judged nothing, and a
    hash recorded there would tell a later comparison a prompt scored what it did not.
    """
    if any(reply.verdict is not None for line in lines for reply in line.judged):
        return judge_prompt_version()
    return None


def _conversations(options: argparse.Namespace) -> int:
    """Play the cases, save each one as it finishes, write the report, and answer the exit code.

    Every pre-check runs BEFORE the help desk is touched, because after the reset the
    seed is spent. The exit code is `EXIT_OK` whenever the run completed: failed cases are
    the gate's business, not the run's, because the baseline run itself has failures.
    Every refusal is one line on stderr and `EXIT_USAGE`.
    """
    try:
        target = _eargs._checked_target(options.target)
        run_id = checked_identifier(options.run_id or _eargs._new_run_id(), "run_id")
        judge_url = _eargs._checked_judge_url(options.judge_url)
        supplied_sha = _eargs._checked_set_sha(options.set_sha)
        cases_path = Path(options.cases)
        cases = load_cases(cases_path)
        set_sha, set_sha_source = _eargs._set_sha(supplied_sha, cases_path)
        out = _eargs._checked_out(options.out)
    except (IdentifierInvalid, UsageError, ConversationSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    started_at = datetime.now(UTC)
    with new_http_client() as http:
        try:
            lines = run_conversations(
                client=TargetClient(target, client=http),
                cases=cases,
                run_id=run_id,
                reports_dir=out,
                judge=_judge(judge_url, http),
            )
        except (TargetUnreachable, ConversationSetInvalid) as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE
        finished_at = datetime.now(UTC)

        report = build_report(
            lines,
            started_at=started_at,
            finished_at=finished_at,
            set_name=cases_path.name,
            set_sha=set_sha,
            set_sha_source=set_sha_source,
            run_kind=options.run_kind,
            git_sha=_eargs._git_sha(),
            tenant=ALL_TENANTS,
            target=target,
            judge_prompt_version=_judged_prompt_version(lines),
        )
        try:
            write_reports(out, report)
        except OSError as exc:
            print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
            return EXIT_USAGE

        stored = _store_rows(report, out)

        if options.pushgateway_url is not None:
            # The written form of the metrics: the document `eval_runs.metrics` and the
            # gate both read, so the gauges carry the numbers every other reader sees.
            written = report.model_dump(mode="json")["metrics"]
            pushed = push_metrics(written, options.pushgateway_url, client=http)
            if pushed is not None:
                print(pushed, file=sys.stderr)

    print(_summary(report))
    if stored is not None:
        print(stored, file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


def _label(options: argparse.Namespace) -> int:
    """Make the reference labels from a finished run, once, and answer the exit code.

    The labels come from the replies, tool calls and results the run already saved, so
    they cost no second run. The target file must not exist: it is refused BEFORE any
    request, because the labels are made once and every grade is three paid requests.
    Fewer than 40 unanimous replies still writes what there is and answers `EXIT_FAILED`.
    """
    try:
        judge_url = _eargs._checked_judge_url_value(options.judge_url)
        api_key = _eargs._checked_judge_key()
        cases_path = Path(options.cases)
        cases = {case.id: case for case in load_cases(cases_path)}
        lines = load_run(Path(options.run))
    except (UsageError, ConversationSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    out = Path(options.out)
    if out.exists():
        print(f"--out: {out} already exists, and labels are made once", file=sys.stderr)
        return EXIT_USAGE

    def candidates() -> list[LabelCandidate]:
        found: list[LabelCandidate] = []
        for line in lines:
            case = cases.get(line.case_id)
            if case is not None and line.answer is not None:
                found.extend(candidates_from(case, line.answer))
        return found

    with new_http_client() as http:
        grader = GatewayLabelGrader(judge_url, client=http, api_key=api_key)
        labels = make_labels(candidates(), grader)
    try:
        write_labels(out, labels)
    except (OSError, FileExistsError) as exc:
        print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(f"labels {len(labels)} · written to {out}")
    if len(labels) < LABEL_TARGET:
        print(
            f"only {len(labels)} replies were unanimous; a calibration needs {LABEL_TARGET}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    return EXIT_OK


def _calibration_summary(report: CalibrationReport) -> str:
    """The one line `calibrate` prints, with the trust flag beside the kappa."""
    kappa = "null" if report.kappa is None else report.kappa
    return f"pairs {report.pairs} · kappa {kappa} · trusted {str(report.trusted).lower()}"


def _calibrate(options: argparse.Namespace) -> int:
    """Calibrate the judge against a labels file and answer the process exit code.

    Every pre-check happens before a request is made: the spend is 40 labels times three
    samples. Both refusals a calibration can end in, an unloadable file and too few labels,
    are one line and `EXIT_USAGE`, never `EXIT_FAILED`: a gate reads 2 as "the harness never
    ran". The artifact is written LAST, so nothing on disk reads as trust unless the whole
    calibration succeeded.
    """
    try:
        judge_url = _eargs._checked_judge_url_value(options.judge_url)
        out = _eargs._checked_out(options.out)
        api_key = _eargs._checked_judge_key()
        labels_path = Path(options.labels)
        labels = load_labels(labels_path)
    except (UsageError, CalibrationSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    with new_http_client() as http:
        try:
            report = calibrate(
                labels,
                judge=JudgeClient(judge_url, client=http, api_key=api_key),
                labels_name=labels_path.name,
                labels_sha=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            )
        except CalibrationRefused as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE

    try:
        write_calibration(out, report)
    except OSError as exc:
        print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(_calibration_summary(report))
    return EXIT_OK


def _gate(options: argparse.Namespace) -> int:
    """Compare one published run against the baseline and answer the exit code.

    Every input is validated before anything is written, so a malformed report never
    leaves a comment behind claiming a run that never happened. stdout carries exactly the
    one verdict line the workflow reads; the reasons and notes go to stderr, one line each.
    """
    try:
        report = load_report(options.report)
        baseline = load_baseline(options.baseline)
        calibration = None if options.calibration is None else read_calibration(options.calibration)
    except (GateInputInvalid, CalibrationSetInvalid) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    verdict = evaluate(report, baseline, calibration=calibration)

    if options.comment is not None:
        try:
            options.comment.write_text(gate_comment(verdict, report, baseline), encoding="utf-8")
        except OSError as exc:
            print(
                f"--comment: {options.comment} could not be written: {exc.strerror}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    print(verdict_line(verdict, report, baseline))
    for line in (*verdict.reasons, *verdict.notes):
        print(line, file=sys.stderr)
    return EXIT_OK if verdict.passed else EXIT_FAILED


def _tradeoff(options: argparse.Namespace) -> int:
    """Build the tradeoff table from a manifest of published runs.

    Every input is read and checked before anything is written, so a refusal on the last
    option leaves `--out` empty rather than a half-written table. A relative `report` in
    the manifest resolves against the manifest's own directory.
    """
    try:
        manifest = load_manifest(options.manifest)
        table = build_table(manifest, base_dir=options.manifest.parent)
    except TradeoffInputInvalid as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    try:
        write_tradeoff(options.out, table)
    except OSError as exc:
        print(f"--out: {options.out} could not be written: {exc.strerror}", file=sys.stderr)
        return EXIT_USAGE

    winner = "none" if table.winner_id is None else table.winner_id
    print(f"options {len(table.rows)} · winner {winner}")
    return EXIT_OK


def _baseline_guard(options: argparse.Namespace) -> int:
    """Answer the baseline guard's verdict over one range of commits and the exit code.

    Every refusal is one line on stderr and `EXIT_USAGE`; stdout carries the one summary
    line; the offences go to stderr one per sha so an author sees every commit to fix in
    one CI round. "Could not read the range" and "the range breaks the rule" are kept
    apart because only the second is the author's to fix.
    """
    try:
        base = _eargs._checked_ref(options.base, "--base")
        head = _eargs._checked_ref(options.head, "--head")
        commits = _eargs._commits(base, head)
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    verdict = check(commits)
    if verdict.passed:
        print(GUARD_PASSED_LINE.format(checked=len(commits)))
        return EXIT_OK

    for offence in verdict.offences:
        print(offence_line(offence), file=sys.stderr)
    return EXIT_FAILED
