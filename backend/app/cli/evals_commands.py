"""Command handlers for `ops-eval`.

The four functions that do the actual work of each subcommand: `_conversations`, `_gate`,
`_tradeoff` and `_baseline_guard`. Each takes the parsed `argparse.Namespace` and returns the
process exit code. Every HTTP client a command opens comes from one seam,
`new_http_client()`, so a test replaces the network with one `httpx.MockTransport`.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

import app.cli.evals_args as _eargs
from app.cli.evals_args import UsageError
from app.core.config import Settings
from app.core.validation import IdentifierInvalid, checked_identifier
from app.evals.baseline_guard import GUARD_PASSED_LINE, check, offence_line
from app.evals.cases import CaseSetInvalid, EvalCase, load_cases
from app.evals.gate import (
    GateInputInvalid,
    evaluate,
    gate_comment,
    load_baseline,
    load_report,
    verdict_line,
)
from app.evals.judge import JudgeClient, prompt_version
from app.evals.person import PersonModel, PersonTurn, PlayedTurn, SimulatedPerson
from app.evals.push import push_metrics
from app.evals.report import ALL_TENANTS, REPORT_JSON_NAME, RunReport, build_report, write_reports
from app.evals.results import PlayRecord
from app.evals.runner import run_eval
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
    "no gateway URL or no OPSCONSOLE_EVAL_JUDGE_KEY: the simulated person ends each play "
    "after the opening and nothing is judged"
)


def new_http_client() -> httpx.Client:
    """The one place a command makes an HTTP client.

    A seam, not a convenience: the command tests replaces it with a client over a
    `MockTransport` that routes by host, so no command test opens a socket.
    """
    return httpx.Client()


class _SilentPerson:
    """The person when there is no gateway: the play ends after the opening."""

    def answer(self, case: EvalCase, turns: Sequence[PlayedTurn]) -> PersonTurn | None:
        return None


def _summary(run_id: str, report: RunReport) -> str:
    """The one line `conversations` prints, so a CI log carries the real numbers."""
    metrics = report.metrics
    return (
        f"run {run_id} · cases {metrics['cases']} · plays {metrics['plays']} · "
        f"safety {metrics['safety']} · outcome_every_play {metrics['outcome_every_play']} · "
        f"needs a look {metrics['needs_a_look']} · setup wrong {metrics['setup_wrong']}"
    )


def _store_rows(report: RunReport, out: Path) -> str | None:
    """Store the run and its cases, and answer what went wrong, or `None`.

    Called AFTER the report files are written: the files are what survives a failed
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


def _model_clients(
    gateway_url: str | None, client: httpx.Client
) -> tuple[PersonModel, JudgeClient | None]:
    """The simulated person and the judge, or a silent person and no judge with one line.

    Both halves are required: a URL with no key would spend a 401 per request, and a key
    with no URL has nowhere to send them.
    """
    api_key = Settings().eval_judge_key
    if gateway_url is None or not api_key:
        print(UNJUDGED_LINE, file=sys.stderr)
        return _SilentPerson(), None
    return (
        SimulatedPerson(gateway_url, client=client, api_key=api_key),
        JudgeClient(gateway_url, client=client, api_key=api_key),
    )


def _reset_with(command: str) -> None:
    """Run the reset command, or raise `TargetUnreachable` when it fails."""
    try:
        finished = subprocess.run(shlex.split(command), check=False)
    except OSError as exc:
        raise TargetUnreachable(f"--reset-command: {command!r} could not run: {exc}") from exc
    if finished.returncode != 0:
        raise TargetUnreachable(
            f"--reset-command: {command!r} exited {finished.returncode}, so the data was not reset"
        )


def _conversations(options: argparse.Namespace) -> int:
    """Play the cases, write the report, and answer the exit code.

    Every pre-check runs BEFORE the help desk is touched, because after the reset the
    seed is spent. The exit code is `EXIT_OK` whenever the run completed: failed plays are
    the gate's business, not the run's. Every refusal is one line on stderr and `EXIT_USAGE`.
    """
    try:
        target = _eargs._checked_target(options.target)
        run_id = checked_identifier(options.run_id or _eargs._new_run_id(), "run_id")
        plays = _eargs._checked_plays(options.plays)
        gateway_url = _eargs._checked_gateway_url(options.gateway_url)
        supplied_sha = _eargs._checked_set_sha(options.set_sha)
        cases_path = Path(options.cases)
        cases = load_cases(cases_path)
        set_sha, set_sha_source = _eargs._set_sha(supplied_sha, cases_path)
        out = _eargs._checked_out(options.out)
    except (IdentifierInvalid, UsageError, CaseSetInvalid) as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    started_at = datetime.now(UTC)
    with new_http_client() as http:
        target_client = TargetClient(target, client=http)
        person, judge = _model_clients(gateway_url, http)
        reset_command = options.reset_command
        try:
            records: tuple[PlayRecord, ...] = run_eval(
                target=target_client,
                cases=cases,
                plays=plays,
                person=person,
                judge=judge,
                reset=target_client.reset
                if reset_command is None
                else lambda: _reset_with(reset_command),
                clock=time.monotonic,
                now=lambda: datetime.now(UTC),
            )
        except TargetUnreachable as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE
        finished_at = datetime.now(UTC)

        report = build_report(
            records,
            started_at=started_at,
            finished_at=finished_at,
            set_name=cases_path.name,
            set_sha=set_sha,
            set_sha_source=set_sha_source,
            run_kind=options.run_kind,
            git_sha=_eargs._git_sha(),
            tenant=ALL_TENANTS,
            target=target,
            judge_prompt_version=prompt_version() if judge is not None else None,
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

    print(_summary(run_id, report))
    if stored is not None:
        print(stored, file=sys.stderr)
        return EXIT_USAGE
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
    except GateInputInvalid as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    verdict = evaluate(report, baseline)

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
