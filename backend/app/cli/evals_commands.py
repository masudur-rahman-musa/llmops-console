"""Command handlers for `ops-eval`.

The ten functions that do the actual work of each subcommand. Each one
takes the parsed `argparse.Namespace` and returns the process exit code.
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
from app.core.validation import IdentifierInvalid
from app.evals.baseline_guard import (
    GUARD_PASSED_LINE,
    check,
    offence_line,
)
from app.evals.calibration import (
    CalibrationRefused,
    CalibrationReport,
    CalibrationSetInvalid,
    calibrate,
    load_labels,
    write_calibration,
)
from app.evals.cases import EvalSetInvalid, load_golden_set
from app.evals.gate import (
    GateInputInvalid,
    evaluate,
    gate_comment,
    load_baseline,
    load_report,
    reason_line,
    verdict_line,
)
from app.evals.judge import JudgeClient, judge_prompt_version
from app.evals.push import push_metrics
from app.evals.report import (
    REPORT_JSON_NAME,
    RunReport,
    build_report,
    write_reports,
)
from app.evals.runner import RunResult, run_set
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
EXIT_RESIDENCY: int = 3

UNSTORED_LINE: str = (
    "no database URL configured: this run's rows were not stored, and its two report files were"
)

UNJUDGED_LINE: str = (
    "no judge URL or no OPSCONSOLE_EVAL_JUDGE_KEY: this run judged nothing, and judge_mean is null"
)

UNTRUSTED_JUDGE_LINE: str = (
    "the calibration supplied is not trusted: this run's judge metrics are marked "
    "untrusted in both reports"
)


def _summary(result: RunResult) -> str:
    """The one line this command prints, in the order `report.md`'s counts repeat.

    One line because `make eval`'s output is read in a CI log, and the counts are
    the run's real ones (`test_the_summary_is_one_line_naming_the_four_counts`).
    """
    return (
        f"cases {result.cases} · passed {result.passed_count} · "
        f"failed {result.failed_count} · residency violations {result.residency_violations}"
    )


def _exit_code(result: RunResult) -> int:
    """Turn one run into the integer the eval gate reads.

    Residency and a leak outrank the assertion counts: a boundary breach read as
    an accuracy dip is a breach somebody merges over.
    """
    if result.residency_violations or result.leaked:
        return EXIT_RESIDENCY
    if result.failed_count:
        return EXIT_FAILED
    return EXIT_OK


def _store_rows(report: RunReport, out: Path) -> str | None:
    """Store the run and its cases, and answer what went wrong, or `None`.

    Called AFTER both report files are written (decision E's ordering): the
    artefact is what survives a failed insert, and the eval gate reads it off
    disk. The failure is answered rather than raised so the caller can apply
    decision E's exit rule in one place, and a storage complaint never becomes a
    traceback over a run already paid for.

    With no `OPSCONSOLE_DATABASE_URL` configured nothing is stored, `UNSTORED_LINE`
    is printed and the run keeps its own exit code (decision B).
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
    """The judge this run scores with, or `None` with one line on stderr.

    Both halves are required (decision B): a URL with no key would spend three
    401s per rubric case, and a key with no URL has nowhere to send them. The
    client is the run's own `httpx.Client`, so the judge opens no second pool and
    the whole path is drivable through one `MockTransport`.
    """
    api_key = Settings().eval_judge_key
    if judge_url is None or not api_key:
        print(UNJUDGED_LINE, file=sys.stderr)
        return None
    return JudgeClient(judge_url, client=client, api_key=api_key)


def _judged_prompt_version(result: RunResult) -> str | None:
    """The prompt's sha256 when a case was actually judged, and `None` otherwise.

    Read off the RECORDS rather than off whether a judge was built (criterion 7):
    a run whose every judge sample failed, and a set with no rubric in it, both
    judged nothing, and a hash recorded there would tell a later comparison the
    two runs used a prompt neither of them scored anything with
    (`test_with_no_judge_url_the_run_judges_nothing_and_keeps_its_exit_code`).
    """
    if any(run.judge is not None for run in result.runs):
        return judge_prompt_version()
    return None


def _calibration_summary(report: CalibrationReport) -> str:
    """The one line `calibrate` prints, in the order a reader asks the questions.

    One line for `_summary`'s reason - this output is read in a CI log - and the
    trust flag is on it because that is the whole point of the command: a kappa
    with no verdict beside it makes every reader apply the threshold themselves
    (`test_calibrate_writes_the_artifact_into_out_and_exits_zero`).
    """
    kappa = "null" if report.kappa is None else report.kappa
    return (
        f"pairs {report.pairs} · dropped {report.dropped} · "
        f"kappa {kappa} · trusted {str(report.trusted).lower()}"
    )


def _run(options: argparse.Namespace) -> int:
    """Run one golden set and answer the process exit code.

    Every refusal goes to stderr and never to stdout, which is the stream the
    summary's counts are parsed from
    (`test_a_pre_check_failure_reports_on_stderr_rather_than_stdout`), and
    `EvalSetInvalid` and `TargetUnreachable` are reported as their own message
    rather than as a traceback: an operator who mistyped a path needs the path.
    """
    try:
        tenant = _eargs._checked_tenant(options.tenant)
        target = _eargs._checked_target(options.target)
        out = _eargs._checked_out(options.out)
        policy = _eargs._checked_policy(options.policy, tenant)
        supplied_sha = _eargs._checked_set_sha(options.set_sha)
        judge_url = _eargs._checked_judge_url(options.judge_url)
        calibration = _eargs._checked_calibration(options.calibration, judge_url)
        case_set = Path(options.case_set)
        cases = load_golden_set(case_set)
        set_sha, set_sha_source = _eargs._set_sha(supplied_sha, case_set)
    except (IdentifierInvalid, UsageError, EvalSetInvalid, CalibrationSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    if calibration is not None and not calibration.trusted:
        print(UNTRUSTED_JUDGE_LINE, file=sys.stderr)

    started_at = datetime.now(UTC)
    with httpx.Client() as http:
        client = TargetClient(target, client=http)
        try:
            result = run_set(
                client=client,
                cases=cases,
                policy=policy,
                tenant=tenant,
                other_tenant_markers=_eargs._other_tenant_markers(cases, tenant),
                judge=_judge(judge_url, http),
            )
        except TargetUnreachable as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE
    finished_at = datetime.now(UTC)

    report = build_report(
        result,
        started_at=started_at,
        finished_at=finished_at,
        set_name=case_set.name,
        set_sha=set_sha,
        set_sha_source=set_sha_source,
        run_kind=options.run_kind,
        git_sha=_eargs._git_sha(),
        tenant=tenant,
        target=target,
        judge_prompt_version=_judged_prompt_version(result),
        calibration=calibration,
    )
    try:
        write_reports(out, report)
    except OSError as exc:
        print(f"--out: {out} could not be written: {exc}", file=sys.stderr)
        return EXIT_USAGE

    stored = _store_rows(report, out)

    if options.pushgateway_url is not None:
        # `model_dump(mode="json")`, not `report.metrics`: a Decimal cost serialises to a
        # STRING on the written report (`report.py`'s `_serialise_cost`), and this is the
        # SAME document `eval_runs.metrics` and the eval gate both read - pushing the raw
        # in-memory `Decimal` would push a value no other reader of this run's metrics sees.
        written_metrics = report.model_dump(mode="json")["metrics"]
        with httpx.Client() as push_client:
            pushed = push_metrics(written_metrics, options.pushgateway_url, client=push_client)
        if pushed is not None:
            print(pushed, file=sys.stderr)

    print(_summary(result))
    code = _exit_code(result)
    if stored is not None:
        print(stored, file=sys.stderr)
        return EXIT_RESIDENCY if code == EXIT_RESIDENCY else EXIT_USAGE
    return code


def _gate(options: argparse.Namespace) -> int:
    """Compare one published run against the baseline and answer the exit code (18a).

    `_run`'s shape with no run in it: every refusal is one line on stderr answered
    with `EXIT_USAGE`, stdout carries exactly the one verdict line the workflow reads
    (decision E), and the breached rules go to stderr one line each so a reviewer
    sees all of them at once
    (`test_gate_reports_every_breached_rule_on_its_own_line_and_exits_one`).

    The order here is the contract. Both inputs are validated before anything is
    written, so a malformed report never leaves a comment file behind claiming a
    run that never happened; the comment is written before the verdict is printed,
    so a run whose comment could not be written does not look, from stdout and the
    code, exactly like one whose comment reached the PR
    (`test_a_comment_path_inside_a_directory_that_does_not_exist_is_a_usage_refusal`).

    `EXIT_RESIDENCY` outranks `EXIT_FAILED` exactly as `_exit_code` ranks them for
    `run` (decision D): a boundary breach read as an accuracy dip is a breach
    somebody merges over
    (`test_gate_exits_three_when_a_residency_violation_sits_beside_a_threshold_breach`).
    """
    try:
        report = load_report(options.report)
        baseline = load_baseline(options.baseline)
    except GateInputInvalid as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    verdict = evaluate(report, baseline)

    if options.comment is not None:
        try:
            options.comment.write_text(gate_comment(verdict, report), encoding="utf-8")
        except OSError as exc:
            print(
                f"--comment: {options.comment} could not be written: {exc.strerror}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    print(verdict_line(verdict))
    for breach in verdict.breaches:
        print(reason_line(breach), file=sys.stderr)

    if verdict.residency_breached:
        return EXIT_RESIDENCY
    return EXIT_OK if verdict.passed else EXIT_FAILED


def _tradeoff(options: argparse.Namespace) -> int:
    """Build the tradeoff table from a manifest of published runs (27b).

    `_gate`'s shape with no verdict in it: every input is read and checked before
    anything is written, so a refusal on the last option leaves `--out` empty
    rather than a half-written table for the next reader to quote
    (`test_every_refusal_exits_two_with_one_line_on_stderr_and_an_empty_out_directory`).
    A relative `report` in the manifest resolves against the manifest's own
    directory, which is what makes the committed evidence directory portable.
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


def _calibrate(options: argparse.Namespace) -> int:
    """Calibrate the judge against a labels file and answer the process exit code.

    Every pre-check happens before a request is made, for `run`'s own reason one
    level along: the spend here is `MIN_CALIBRATION_LABELS` times `JUDGE_SAMPLES`
    calls, and a URL refused afterwards has already cost all of it
    (`test_a_calibrate_judge_url_that_is_not_an_absolute_http_url_is_refused`
    asserts the request log is empty).

    Both refusals a calibration can end in - an unloadable file and a judge that
    scored too few labels - are reported as their own single line and answered
    with `EXIT_USAGE`, never as a traceback and never as `EXIT_FAILED`: a gate
    reads `2` as "the harness never ran", which is exactly what a judge outage is
    (`test_a_judge_that_answers_five_hundred_throughout_writes_nothing_and_exits_two`).
    The artifact is written LAST, so nothing on disk can be read as trust unless
    the whole calibration succeeded.
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

    with httpx.Client() as http:
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


def _baseline_guard(options: argparse.Namespace) -> int:
    """Answer #42's verdict over one range of commits and the process exit code (18b).

    `_gate`'s shape over git rather than over two files: every refusal is one line on
    stderr answered with `EXIT_USAGE`, stdout carries exactly the one summary line,
    and the offences go to stderr one per sha so an author sees every commit to fix
    in one CI round
    (`test_baseline_guard_prints_one_stderr_line_per_offence_and_exits_one`).

    `EXIT_USAGE` and `EXIT_FAILED` are kept apart here for the reason the rest of
    this module keeps them apart: "the guard could not read the range" and "the
    range breaks the rule" have different fixes, and only the second is the
    author's.
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
