"""`evals/cli.py`: `ops-eval run` and `ops-eval calibrate` - the pre-checks, the codes.

No network and no backing service: `run_set` is replaced by a recording stand-in,
so what is under test here is the SHELL - what the CLI refuses before anything
runs, what integer it answers, and what it prints. The loop itself is pinned
against `httpx.MockTransport` in `tests/evals/test_runner.py`.

The contract these cases are written against, before it exists:

* `main(argv=None) -> int`, wired as `[project.scripts] ops-eval`, with a `run`
  subcommand taking `--target --set --tenant --out` and `--policy`, which
  defaults to the shipped `routing.yaml`, plus `16c`'s `--set-sha` (optional)
  and `--run-kind` (an argparse `choices` of `ci`/`nightly`/`manual`,
  defaulting to `manual`).
* `EXIT_OK`, `EXIT_FAILED`, `EXIT_USAGE`, `EXIT_RESIDENCY` - the constants
  a project rule requires a test to pin exactly (decision C).
* `cli.run_set` resolved as a module attribute, which is the seam these cases
  replace.
* `report.write_reports` called with `--out` BEFORE `_exit_code` is chosen
  (decision D), so `report.json` and `report.md` are on disk whatever the run
  answered. The report cases below read the files rather than the call, so they
  cannot pass on a CLI that builds a report and drops it.
* `git_sha` from ONE `subprocess.run(["git", "rev-parse", "HEAD"], ...)`. Those
  cases double `subprocess.run` and nothing narrower: doubling a `_git_sha`
  helper would remove the control under test and the case would pass whether or
  not the failure is handled.
* `CaseRun` carrying `category` and `outcome` beside what `16b` shipped, which
  is where a `CaseReport`'s category and its four measured fields come from -
  `build_report` reads the `RunResult` and nothing else.
* `16d`'s two seams, both resolved as attributes of `opsconsole.evals.cli`:
  `cli.build_pool`, called with `Settings().database_url` only when that is set,
  and `cli.PostgresEvalStore`, whose `write(run_row, case_rows)` is called AFTER
  `write_reports` (decision E). The store cases below replace both, because a
  real `build_pool` would open TCP connections from a suite that has no
  PostgreSQL, and `OPSCONSOLE_DATABASE_URL` is cleared for the whole session by
  the root conftest, so "no database URL configured" is the
  arrangement every other case in this file already runs under.

* 17e's one seam, resolved the same way: `cli.JudgeClient` as an attribute of
  `opsconsole.evals.cli`. The `calibrate` subcommand opens its own
  `httpx.Client`, so replacing the client would mean doubling `httpx` itself;
  replacing the CLIENT CLASS with a factory over an `httpx.MockTransport` keeps
  a real `JudgeClient` - 17c's three samples, its one retry and its median all
  still run - and is the only seam the calibrate cases below use.
* `calibrate --labels PATH --judge-url URL --out DIR`, all three required, with
  `--judge-url` checked by `_checked_judge_url` and `--out` by `_checked_out`
  before any request, `OPSCONSOLE_EVAL_JUDGE_KEY` required rather than optional,
  and every refusal reported as ONE line on stderr answered with `EXIT_USAGE`.

The tenant table below is arranged against the neighbouring guard: the generated
policy DEFINES the exact tenant string each row passes, malformed ones included,
so the only control that can refuse a row is the `IDENTIFIER_PATTERN` pre-check
and not the "this policy has no such tenant" lookup that follows it. Standing
rule 4's rows (a trailing newline, a trailing CRLF, a trailing space) are in that
table for the same reason they are in 16a's: a check written `^...$` with
`.match` accepts the first of them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pytest

from opsconsole.evals import cli
from opsconsole.evals.assertions import AssertionResult, AssertionStatus, CaseAssertions
from opsconsole.evals.calibration import CALIBRATION_JSON_NAME, MIN_CALIBRATION_LABELS
from opsconsole.evals.cases import EvalCase, EvalCategory
from opsconsole.evals.cli import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_RESIDENCY,
    EXIT_USAGE,
    UNJUDGED_LINE,
    UNSTORED_LINE,
)
from opsconsole.evals.judge import DEFAULT_JUDGE_TIMEOUT_SECONDS, JudgeClient
from opsconsole.evals.outcome import CaseOutcome
from opsconsole.evals.report import REPORT_JSON_NAME, REPORT_MD_NAME
from opsconsole.evals.residency import ResidencyVerdict
from opsconsole.evals.runner import CaseRun, RunResult
from opsconsole.evals.target import TargetClient
from opsconsole.routing.policy import Policy
from opsconsole.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

VALID_SET = FIXTURE_DIR / "valid_set.jsonl"
"""16a's three-case set: `alpha`, `beta`, `gamma`, in file order."""

TARGET = "https://target.example"

SHIPPED_EVAL_TENANT = "evals"
"""The tenant the shipped `routing.yaml` defines for this harness's own traffic.

Used by the two cases that pass no `--policy` at all, which is the only way to
tell "the default points at the shipped file" from "the default points nowhere".
"""

MAX_TENANT = "t" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "t" * (MAX_IDENTIFIER_LENGTH + 1)


def write_policy(tmp_path: Path, *tenants: str) -> Path:
    """A minimal, valid policy file defining exactly the tenants named.

    Written as JSON, which `yaml.safe_load` reads: a tenant key carrying a
    newline or a trailing space survives the round trip literally, which is what
    lets a rejection row prove the PRE-CHECK refused it rather than the lookup.
    """
    document = {
        "providers": {},
        "eu_regions": ["eu", "self"],
        "aliases": {},
        "tenants": {
            tenant: {
                "residency": "eu",
                "data_class_rules": {
                    "public": {"allow_regions": ["us", "eu", "self"]},
                    "internal": {"allow_regions": ["eu", "self"]},
                },
                "budget_usd_month": "10.00",
                "rate_limit": {"rpm": 60, "tpm": 60000},
            }
            for tenant in tenants
        },
    }
    path = tmp_path / "policy.yaml"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def make_run(
    case_id: str,
    tenant: str,
    verdict: str,
    *,
    category: EvalCategory = EvalCategory.HELPDESK,
) -> CaseRun:
    """One per-case record of the named shape, as `run_set` would have built it.

    The measured outcome is filled with values distinct per case id, so a report
    that carries one case's latency against another's row is visible rather than
    plausible.
    """
    residency = ResidencyVerdict(
        case_id=case_id,
        tenant=tenant,
        data_class="internal",
        observed="us" if verdict == "residency" else "eu",
        allowed=("eu", "self"),
        violation=verdict == "residency",
    )
    results = [AssertionResult(name="intent", status=AssertionStatus.PASSED)]
    if verdict == "fail":
        results.append(
            AssertionResult(
                name="decision",
                status=AssertionStatus.FAILED,
                expected="auto_approved",
                actual="escalated",
            )
        )
    if verdict == "leak":
        results.append(
            AssertionResult(
                name="cross_tenant_leak",
                status=AssertionStatus.FAILED,
                expected="no marker of another tenant",
                actual="['globex']",
            )
        )
    return CaseRun(
        case_id=case_id,
        category=category,
        assertions=CaseAssertions(case_id=case_id, results=tuple(results)),
        residency=residency,
        outcome=CaseOutcome(
            reply="Done.",
            latency_ms=100 + len(case_id),
            cost_usd=Decimal("0.0010"),
            provider="anthropic_eu",
            region=residency.observed,
        ),
        error=None,
        passed=verdict == "pass",
    )


class RecordingRunner:
    """Stands in for `run_set`, answering a `RunResult` of the verdicts it was given.

    It also records every call, which is what the pre-check cases assert on: a
    CLI that validates its arguments AFTER starting the run would still exit 2
    and would still look correct without that record.
    """

    def __init__(self, *verdicts: str) -> None:
        self.verdicts = verdicts
        self.calls: list[str] = []
        self.judges: list[object] = []

    def __call__(
        self,
        *,
        client: TargetClient,
        cases: tuple[EvalCase, ...],
        policy: Policy,
        tenant: str,
        other_tenant_markers: tuple[str, ...] = (),
        judge: object | None = None,
    ) -> RunResult:
        self.calls.append(tenant)
        self.judges.append(judge)
        verdicts = self.verdicts if self.verdicts else ("pass",) * len(cases)
        return RunResult(
            runs=tuple(
                make_run(case.id, tenant, verdict, category=case.category)
                for case, verdict in zip(cases, verdicts, strict=True)
            )
        )


def install(monkeypatch: pytest.MonkeyPatch, runner: RecordingRunner) -> None:
    """Replace the runner the CLI calls, leaving every pre-check it owns in place."""
    monkeypatch.setattr(cli, "run_set", runner)


def argv(
    *,
    target: str = TARGET,
    case_set: Path = VALID_SET,
    tenant: str = "acme",
    out: Path | None = None,
    policy: Path | None = None,
    set_sha: str | None = None,
    run_kind: str | None = None,
    judge_url: str | None = None,
) -> list[str]:
    """One `ops-eval run` command line.

    `--set-sha`, `--run-kind` and `--judge-url` are omitted unless a case names
    one, which is what lets the default cases below tell "absent" from "passed as
    empty" - and, for the judge, "off because no URL" from "off because no key".
    """
    arguments = ["run", "--target", target, "--set", str(case_set), "--tenant", tenant]
    if out is not None:
        arguments += ["--out", str(out)]
    if policy is not None:
        arguments += ["--policy", str(policy)]
    if set_sha is not None:
        arguments += ["--set-sha", set_sha]
    if run_kind is not None:
        arguments += ["--run-kind", run_kind]
    if judge_url is not None:
        arguments += ["--judge-url", judge_url]
    return arguments


def run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runner: RecordingRunner | None = None,
    tenant: str = "acme",
    target: str = TARGET,
    case_set: Path = VALID_SET,
    out: Path | None = None,
    policy_tenants: Sequence[str] | None = None,
    set_sha: str | None = None,
    run_kind: str | None = None,
    judge_url: str | None = None,
    judge_key: str | None = None,
) -> int:
    """Run the CLI with a stand-in runner and a policy defining the tenant under test.

    `judge_key` is written to (or cleared from) the environment rather than
    passed, because that is where `Settings().eval_judge_key` reads it and a
    developer's own shell must not decide whether a case judges anything
    (a project rule's reason, applied to the one variable this plan adds).
    """
    installed = runner if runner is not None else RecordingRunner()
    install(monkeypatch, installed)
    if judge_key is None:
        monkeypatch.delenv("OPSCONSOLE_EVAL_JUDGE_KEY", raising=False)
    else:
        monkeypatch.setenv("OPSCONSOLE_EVAL_JUDGE_KEY", judge_key)
    tenants = tuple(policy_tenants) if policy_tenants is not None else (tenant,)
    return cli.main(
        argv(
            target=target,
            case_set=case_set,
            tenant=tenant,
            out=out if out is not None else tmp_path / "reports",
            policy=write_policy(tmp_path, *tenants),
            set_sha=set_sha,
            run_kind=run_kind,
            judge_url=judge_url,
        )
    )


def read_report(out: Path) -> dict[str, Any]:
    """The JSON report a run left in `--out`, parsed.

    Read off DISK rather than out of a doubled `write_reports`: a CLI that builds
    a correct report and never writes it would pass every call-recording check
    and leave row 18's gate with nothing to read.
    """
    document: dict[str, Any] = json.loads((out / REPORT_JSON_NAME).read_text(encoding="utf-8"))
    return document


# --------------------------------------------------------------------------
# the four exit codes
# --------------------------------------------------------------------------


def test_the_exit_codes_are_the_published_ones() -> None:
    """Catches the split of decision C collapsing back into one non-zero code. Row 18's
    gate has to tell an accuracy regression (1) from a residency breach (3) without
    parsing stdout, and a Makefile or a CI step already keys off these integers, so each
    is asserted exactly rather than as "non-zero"."""
    assert EXIT_OK == 0
    assert EXIT_FAILED == 1
    assert EXIT_USAGE == 2
    assert EXIT_RESIDENCY == 3


def test_a_run_whose_cases_all_passed_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a CLI that answers non-zero on a clean run - a gate that fails always is a
    gate that gets disabled - and one that never reaches the runner at all, which the
    recorded call proves it did."""
    runner = RecordingRunner("pass", "pass", "pass")

    code = run_cli(monkeypatch, tmp_path, runner=runner)

    assert code == EXIT_OK
    assert runner.calls == ["acme"]


def test_a_failed_assertion_exits_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches a CLI that reports success while cases failed, which is the failure mode
    that makes an eval gate worthless: CI stays green through an accuracy regression.
    No case here violates residency, so only the assertion verdict can decide the code."""
    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "fail", "pass"))

    assert code == EXIT_FAILED


def test_a_residency_violation_exits_three_and_outranks_a_failed_assertion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a residency breach reported as an ordinary failure (1), which row 18 would
    read as an accuracy dip and merge over. The run also carries a failed assertion, so a
    CLI that checks the assertion counts first and returns 1 fails here - which is the
    only arrangement that can tell the precedence apart from the code itself."""
    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("fail", "residency", "pass"))

    assert code == EXIT_RESIDENCY


def test_a_residency_violation_exits_three_even_when_every_assertion_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a CLI that derives its code from the pass/fail counts alone. Every assertion
    in this run passed, so a CLI ignoring `residency_violations` answers 0 and the whole
    of #38 is inert while the summary still prints the violation count."""
    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "residency", "pass"))

    assert code == EXIT_RESIDENCY


def test_a_cross_tenant_leak_exits_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches a leak exiting 1 beside the ordinary accuracy failures. A tenant boundary
    breach is not a score, and the plan puts it on the same exit as a residency violation
    precisely so a human looks at it; the run carries no residency violation, so only
    `leaked` can produce the 3."""
    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "leak", "pass"))

    assert code == EXIT_RESIDENCY


def test_a_missing_required_argument_is_a_usage_error() -> None:
    """Catches a required argument that is silently optional - a run with no `--set` that
    defaults to something, or a traceback instead of a usage message. argparse answers by
    raising `SystemExit`, and the integer it carries is the contract `make eval` reads."""
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "--target", TARGET, "--tenant", "acme"])

    assert raised.value.code == EXIT_USAGE


# --------------------------------------------------------------------------
# --tenant: the pre-check, inclusive at the boundary
# --------------------------------------------------------------------------


def test_the_tenant_length_cap_is_the_resolvers_own() -> None:
    """Catches a CLI restating the cap as its own literal, which then drifts from the
    gateway's: a tenant the running gateway serves would become one the harness refuses to
    evaluate. Pinned at 64 exactly, since that is the number both sides publish."""
    assert MAX_IDENTIFIER_LENGTH == 64
    assert IDENTIFIER_PATTERN.fullmatch(MAX_TENANT) is not None


def test_a_tenant_of_exactly_the_cap_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches an off-by-one that refuses a legitimate 64-character tenant, which is the
    half of the boundary a rejection table alone never sees. The generated policy defines
    this tenant, so reaching exit 0 proves the pre-check admitted it rather than something
    downstream tolerating it."""
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, tenant=MAX_TENANT)

    assert code == EXIT_OK
    assert runner.calls == [MAX_TENANT]


REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_TENANT),
    ("uppercase", "Acme"),
    ("a space", "ac me"),
    ("a path separator, which would escape the report directory", "../acme"),
    ("a trailing newline", "acme\n"),
    ("a trailing CRLF", "acme\r\n"),
    ("a trailing space", "acme "),
]


@pytest.mark.parametrize(
    ("label", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_a_malformed_tenant_is_refused_before_the_run(
    label: str, tenant: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a `--tenant` taken on trust: it selects a policy's rules, reaches `16c`'s
    `eval_cases` rows and names a report path, so `acme\\n` and `../acme` have to be
    refused before any effect runs. The generated policy DEFINES each of
    these strings as a tenant, so the "unknown tenant" guard cannot be what refuses them -
    removing the pattern pre-check is the only thing that flips this row."""
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, tenant=tenant)

    assert code == EXIT_USAGE
    assert runner.calls == []


def test_a_tenant_the_policy_does_not_define_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a well-formed tenant reaching the run unknown to the policy, where it would
    have no allow list at all - and deny-by-default would then report a residency violation
    per case instead of an operator's typo. The policy defines a different tenant, so the
    pattern pre-check passes and only the lookup can refuse."""
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, tenant="globex", policy_tenants=["acme"])

    assert code == EXIT_USAGE
    assert runner.calls == []


def test_the_policy_defaults_to_the_shipped_routing_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a `--policy` default pointing at a file that is not there (the harness would
    then be unusable without an argument `make eval` does not pass) or at an empty policy
    (every tenant unknown). Run with NO `--policy` at all, naming the tenant the shipped
    file defines for this harness's own traffic."""
    runner = RecordingRunner()
    install(monkeypatch, runner)

    code = cli.main(argv(tenant=SHIPPED_EVAL_TENANT, out=tmp_path / "reports"))

    assert code == EXIT_OK
    assert runner.calls == [SHIPPED_EVAL_TENANT]


def test_a_tenant_the_shipped_policy_does_not_define_is_refused_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other half of the default: catches a CLI that reads the default policy but never
    consults it, which would look identical to the case above. `acme` is well formed and is
    NOT a tenant of the shipped `routing.yaml`, so only the lookup against the default file
    can refuse it."""
    runner = RecordingRunner()
    install(monkeypatch, runner)

    code = cli.main(argv(tenant="acme", out=tmp_path / "reports"))

    assert code == EXIT_USAGE
    assert runner.calls == []


# --------------------------------------------------------------------------
# --target and --out
# --------------------------------------------------------------------------


REJECTED_TARGETS: list[tuple[str, str]] = [
    ("empty", ""),
    ("a relative host and path", "target.example/v1"),
    ("a bare host", "target.example"),
    ("a scheme that is not http(s)", "ftp://target.example"),
    ("a file URL", "file:///etc/passwd"),
    ("a scheme with no host", "http://"),
]


@pytest.mark.parametrize(
    ("label", "target"), REJECTED_TARGETS, ids=[row[0] for row in REJECTED_TARGETS]
)
def test_a_target_that_is_not_an_absolute_http_url_is_refused(
    label: str, target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a `--target` handed to httpx unchecked: a relative value produces an
    `httpx` error mid-run after the seed has been reset, and `file://` is a scheme no eval
    target speaks. Refusing here is what makes the message readable and keeps the reset
    from happening against something that is not a target at all."""
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, target=target)

    assert code == EXIT_USAGE
    assert runner.calls == []


@pytest.mark.parametrize("target", ["http://localhost:8080", "https://target.example/gateway"])
def test_an_absolute_http_target_is_accepted(
    target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a check so strict it refuses the two forms every real run uses: a local
    `http://localhost:PORT` and an `https://` host with a path prefix. A pre-check that
    admits nothing is the same outage as no CLI at all."""
    code = run_cli(monkeypatch, tmp_path, target=target)

    assert code == EXIT_OK


def test_an_out_directory_that_does_not_exist_is_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches `--out` accepted and never created, which `16c` would only discover at the
    end of a full run when it tries to write the report - after every case has been paid
    for. Created here, before the target is reset (decision D)."""
    out = tmp_path / "reports" / "nightly"

    code = run_cli(monkeypatch, tmp_path, out=out)

    assert code == EXIT_OK
    assert out.is_dir()


def test_an_out_path_that_is_an_existing_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a CLI that only checks "does it exist": pointed at a file, `16c`'s writes
    would fail after the run, and an implementation that "fixed" it by replacing the path
    would delete whatever an operator named. Refused before anything runs."""
    out = tmp_path / "reports"
    out.write_text("not a directory", encoding="utf-8")
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, out=out)

    assert code == EXIT_USAGE
    assert runner.calls == []
    assert out.read_text(encoding="utf-8") == "not a directory"


def test_a_run_writes_exactly_the_two_report_files_into_the_out_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a run that writes nothing (`16b` left this directory empty and this plan is
    the one that fills it) and, at the other end, a writer that puts a file per case, per
    tenant or a timestamped copy in there - the claim `16b`'s docstrings carry and this
    plan corrects. The names are the module constants, so a rename that keeps the gate's
    glob working still fails here (decision B)."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "fail", "pass"), out=out)

    assert code == EXIT_FAILED
    assert sorted(path.name for path in out.iterdir()) == [REPORT_JSON_NAME, REPORT_MD_NAME]


# --------------------------------------------------------------------------
# the set, and what the run prints
# --------------------------------------------------------------------------


def test_a_missing_set_file_exits_two_with_its_message_and_no_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches 16a's `EvalSetInvalid` escaping as an unhandled exception: an operator who
    mistyped a path would get a stack trace and a shell code of 1, which the gate would
    read as "cases failed" rather than "the set was never loaded". The path is asserted in
    the message because that is the one thing the operator has to fix."""
    missing = tmp_path / "nowhere.jsonl"
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, case_set=missing)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert runner.calls == []
    assert str(missing) in printed.err
    assert "Traceback" not in printed.err


def test_a_pre_check_failure_reports_on_stderr_rather_than_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a refusal printed to stdout, where it lands in the same stream the summary
    line is parsed from: a caller reading stdout for counts would then read an error
    sentence as a result. Also catches a silent exit 2, which leaves the operator with an
    integer and nothing to act on."""
    code = run_cli(monkeypatch, tmp_path, tenant="Acme")

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert printed.err.strip() != ""
    assert printed.out.strip() == ""


def test_the_summary_is_one_line_naming_the_four_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a summary that is a paragraph or a per-case dump - `make eval`'s output is
    read in a CI log, and one line is what makes it readable there - and a summary whose
    numbers are decoration: the counts asserted are the run's real ones (three cases, one
    passed, two failed, one of them a residency violation), so a CLI printing constants or
    the wrong count fails here."""
    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "fail", "residency"))

    printed = capsys.readouterr()
    line = printed.out.strip()
    assert code == EXIT_RESIDENCY
    assert "\n" not in line
    assert re.findall(r"\d+", line) == ["3", "1", "2", "1"]
    for word in ("cases", "passed", "failed", "residency"):
        assert word in line.lower()


# --------------------------------------------------------------------------
# the reports: written before the exit code is chosen (decision D)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("fail", EXIT_FAILED), ("residency", EXIT_RESIDENCY)],
    ids=["a failed assertion, exit 1", "a residency violation, exit 3"],
)
def test_both_reports_are_written_on_a_run_that_does_not_exit_zero(
    verdict: str, expected: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the reports written only on a passing run, or written after the exit code
    is chosen and returned - a failed run is exactly the one somebody opens the report for,
    and a gate that fails with no artefact is a gate people work around (decision D). Both
    non-zero codes are covered because a CLI that returns early on the residency branch
    alone would still pass the exit-1 row."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", verdict, "pass"), out=out)

    assert code == expected
    assert (out / REPORT_MD_NAME).read_text(encoding="utf-8") != ""
    document = read_report(out)
    assert document["passed"] is False
    assert document["metrics"]["failed"] == 1
    assert len(document["cases"]) == 3


def test_the_report_carries_the_run_identity_the_command_line_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a CLI that writes a report of defaults beside the run it actually made: the
    tenant, the target and the set are what tell two report directories apart, and `16d`
    stores all three as `eval_runs` columns. Each is a value no other argument on this
    command line carries."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, out=out, target="https://target.example/gateway")

    document = read_report(out)
    assert code == EXIT_OK
    assert document["tenant"] == "acme"
    assert document["target"] == "https://target.example/gateway"
    assert "valid_set" in document["set_name"]
    assert document["metrics"]["cases"] == 3


# --------------------------------------------------------------------------
# --set-sha: the pre-check, inclusive at the boundary
# --------------------------------------------------------------------------


SEVEN_HEX = "a1b2c3d"
SIXTY_FOUR_HEX = "0123456789abcdef" * 4


@pytest.mark.parametrize(
    ("label", "set_sha"),
    [
        ("the shortest accepted, seven", SEVEN_HEX),
        ("the longest accepted, sixty-four", SIXTY_FOUR_HEX),
    ],
    ids=["seven", "sixty-four"],
)
def test_a_set_sha_at_the_boundary_is_accepted_and_recorded_as_a_commit(
    label: str, set_sha: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches an off-by-one that refuses a legitimate short SHA or a full 64-character
    one, which is the half of the boundary a rejection table never sees, and a CLI that
    takes `--set-sha` and then hashes the file anyway - the caller's commit is the one
    answer to "which version of the set was this?" and a content hash silently substituted
    for it is decision A's whole failure mode."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, out=out, set_sha=set_sha)

    document = read_report(out)
    assert code == EXIT_OK
    assert document["set_sha"] == set_sha
    assert document["set_sha_source"] == "commit"


REJECTED_SET_SHAS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character under the floor, six", "a1b2c3"),
    ("one character over the cap, sixty-five", SIXTY_FOUR_HEX + "a"),
    ("uppercase hex", "A1B2C3D"),
    ("not hex at all", "zzzzzzz"),
    ("a path separator", "../a1b2c3d"),
    ("a trailing newline", SEVEN_HEX + "\n"),
    ("a trailing CRLF", SEVEN_HEX + "\r\n"),
    ("a trailing space", SEVEN_HEX + " "),
]


@pytest.mark.parametrize(
    ("label", "set_sha"), REJECTED_SET_SHAS, ids=[row[0] for row in REJECTED_SET_SHAS]
)
def test_a_malformed_set_sha_is_refused_before_the_target_is_touched(
    label: str, set_sha: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a `--set-sha` taken on trust: it is written into a report and into `16d`'s
    `eval_runs` row, so `a1b2c3d\\n` and `../a1b2c3d` have to be refused before the seed is
    reset and the run is paid for. Nothing else on this command line
    looks at the value, so removing the pre-check is the only thing that flips these rows -
    and the three trailing-whitespace rows are what separate `re.fullmatch` from an
    anchored `.match`, which accepts the newline."""
    out = tmp_path / "reports"
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, out=out, set_sha=set_sha)

    assert code == EXIT_USAGE
    assert runner.calls == []
    assert list(out.iterdir()) == []


def test_without_a_set_sha_the_field_is_the_sets_own_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a run with no `--set-sha` recording nothing, a placeholder, or a hash of the
    PATH rather than of the bytes - two machines running the same file from different
    directories would then disagree about whether the set changed. `set_sha_source` is
    asserted as `content` in the same breath, because the failure decision A exists to
    prevent is a reader being told a content hash is a commit."""
    out = tmp_path / "reports"
    expected = hashlib.sha256(VALID_SET.read_bytes()).hexdigest()

    code = run_cli(monkeypatch, tmp_path, out=out)

    document = read_report(out)
    assert code == EXIT_OK
    assert document["set_sha"] == expected
    assert document["set_sha_source"] == "content"


# --------------------------------------------------------------------------
# --run-kind
# --------------------------------------------------------------------------


@pytest.mark.parametrize("run_kind", ["ci", "nightly", "manual"])
def test_each_run_kind_reaches_the_report(
    run_kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a `--run-kind` parsed and dropped: row 18 filters `eval_runs` by it, so a
    field that always reads `manual` would make every CI run invisible as a CI run. All
    three are covered because a CLI recording only the value it was built around passes any
    single row."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, out=out, run_kind=run_kind)

    assert code == EXIT_OK
    assert read_report(out)["run_kind"] == run_kind


def test_the_run_kind_defaults_to_manual(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Catches a `--run-kind` that is required in practice (every operator running
    `ops-eval` by hand would have to name one) or that defaults to `ci`, which would label
    a laptop run as a gate run in `16d`'s rows. Run with the flag ABSENT, which is the only
    way to see the default at all."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, out=out)

    assert code == EXIT_OK
    assert read_report(out)["run_kind"] == "manual"


def test_a_run_kind_outside_the_three_is_a_usage_error(tmp_path: Path) -> None:
    """Catches a free-string `--run-kind`, which puts a fourth word nobody agreed into a
    report and into `16d`'s column - `nightly-2` and `CI` both read as a typo to a human
    and as a new kind to a query. argparse `choices` answers by raising `SystemExit`, and
    the integer it carries is the code `make eval` reads."""
    with pytest.raises(SystemExit) as raised:
        cli.main(argv(out=tmp_path / "reports", run_kind="smoke"))

    assert raised.value.code == EXIT_USAGE


# --------------------------------------------------------------------------
# git_sha
# --------------------------------------------------------------------------


HEAD_SHA = "b71f0c3a9d5e42188a6c0f37d21be9054c8a7f6d"


def install_git(monkeypatch: pytest.MonkeyPatch, answer: object) -> None:
    """Double `subprocess.run` itself, which is the only seam that keeps the control under
    test: doubling a `_git_sha` helper would replace the very code that decides what a
    failed `git rev-parse` becomes. `answer` is either an exception to raise or the
    `CompletedProcess` to hand back."""

    def fake_run(*args: object, **kwargs: object) -> object:
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_the_git_sha_is_this_repos_head_with_no_trailing_newline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the newline `git rev-parse` prints being carried into the report and into
    `16d`'s column, where it makes every equality query against a commit miss. The doubled
    answer ends in a newline for exactly that reason."""
    out = tmp_path / "reports"
    install_git(
        monkeypatch,
        subprocess.CompletedProcess(args=["git"], returncode=0, stdout=f"{HEAD_SHA}\n", stderr=""),
    )

    code = run_cli(monkeypatch, tmp_path, out=out)

    assert code == EXIT_OK
    assert read_report(out)["git_sha"] == HEAD_SHA


GIT_FAILURES: list[tuple[str, object]] = [
    ("git is not installed", FileNotFoundError("git")),
    ("git rev-parse exits non-zero", subprocess.CompletedProcess(["git"], 128, "", "not a repo")),
    (
        "git rev-parse raises",
        subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"], "", "not a repo"),
    ),
]


@pytest.mark.parametrize(("label", "answer"), GIT_FAILURES, ids=[row[0] for row in GIT_FAILURES])
def test_the_git_sha_is_null_when_git_rev_parse_does_not_answer(
    label: str, answer: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a `git_sha` guessed or defaulted when git cannot answer - an empty string, a
    literal `unknown`, or a traceback that loses a paid-for run. `null` is the only honest
    value, and the three rows are the three shapes the failure really takes (no git binary,
    a non-zero exit, a raising call), so an implementation handling one of them still
    fails the others. The run exits on its OWN counts, which proves the missing SHA did not
    become the exit code."""
    out = tmp_path / "reports"
    install_git(monkeypatch, answer)

    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "fail", "pass"), out=out)

    document = read_report(out)
    assert code == EXIT_FAILED
    assert document["git_sha"] is None


# --------------------------------------------------------------------------
# the write that fails after the run has been paid for
# --------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a mode 0o500 directory anyway")
def test_an_out_directory_that_cannot_be_written_exits_two_naming_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches the `PermissionError` from a read-only mount or a full disk escaping as a
    traceback at the END of a run whose counts are already spent: the operator needs the
    path and an exit code the gate can tell from "cases failed", not a stack trace and a 1.
    The directory passes `--out`'s parse-time check and only fails at write time - the
    runner IS called here - so the pre-check cannot be what produces the 2."""
    out = tmp_path / "reports"
    out.mkdir()
    out.chmod(0o500)
    runner = RecordingRunner()
    try:
        code = run_cli(monkeypatch, tmp_path, runner=runner, out=out)
    finally:
        out.chmod(0o700)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert runner.calls == ["acme"]
    assert str(out) in printed.err
    assert "Traceback" not in printed.err


# --------------------------------------------------------------------------
# the eval rows: written after both reports, or not at all (16d)
# --------------------------------------------------------------------------


DATABASE_URL = "postgresql://opsconsole@localhost:5432/opsconsole"
"""A URL that is never connected to: `cli.build_pool` is replaced in every case
that sets it, so what is under test is the branch the setting decides."""


class RecordingStore:
    """Stands in for `PostgresEvalStore`, recording what it was handed and when.

    `reports_on_disk` is the half that matters: "the store was called" would also
    be true of a CLI that stored the rows and then failed to write the files, and
    decision E's whole trade is that the artefact survives a failed insert.
    """

    def __init__(self, out: Path, *, failure: Exception | None = None) -> None:
        self.out = out
        self.failure = failure
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.reports_on_disk: list[bool] = []

    def write(self, run_row: Any, case_rows: Any) -> None:  # noqa: ANN401
        self.calls.append((run_row, tuple(case_rows)))
        self.reports_on_disk.append(
            (self.out / REPORT_JSON_NAME).is_file() and (self.out / REPORT_MD_NAME).is_file()
        )
        if self.failure is not None:
            raise self.failure


def install_store(monkeypatch: pytest.MonkeyPatch, store: RecordingStore) -> list[str]:
    """Configure a database URL and replace both seams, answering the URLs pooled.

    The URL list is what tells "a pool was built from the configured setting" from
    "a store was constructed out of nothing", which are two different bugs with
    the same symptom.
    """
    pooled: list[str] = []

    def fake_build_pool(database_url: str) -> object:
        pooled.append(database_url)
        return object()

    def fake_store(pool: object, **kwargs: object) -> RecordingStore:
        return store

    monkeypatch.setenv("OPSCONSOLE_DATABASE_URL", DATABASE_URL)
    monkeypatch.setattr(cli, "build_pool", fake_build_pool)
    monkeypatch.setattr(cli, "PostgresEvalStore", fake_store)
    return pooled


def forbid_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the database URL and fail the test if either seam is reached anyway."""

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("a pool or a store was built with no database URL configured")

    monkeypatch.delenv("OPSCONSOLE_DATABASE_URL", raising=False)
    monkeypatch.setattr(cli, "build_pool", forbidden)
    monkeypatch.setattr(cli, "PostgresEvalStore", forbidden)


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("pass", EXIT_OK), ("fail", EXIT_FAILED), ("residency", EXIT_RESIDENCY)],
    ids=["a clean run", "a failed assertion", "a residency violation"],
)
def test_with_no_database_url_nothing_is_stored_and_the_run_keeps_its_own_exit_code(
    verdict: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Decision B. Catches a run that refuses to start, or exits 2, on a machine with no
    PostgreSQL - which is every machine `make eval` is usually run on, and it would turn
    the whole harness off rather than turn storage off. All three codes are covered
    because a CLI that returned an unstored run's code correctly on the clean row could
    still swallow the residency 3. ONE `UNSTORED_LINE` on stderr is asserted exactly: silence
    leaves an operator believing rows were written, and a line per case is noise in a CI
    log. TWO lines in total since 17c, because a run with no `--judge-url` also prints
    `UNJUDGED_LINE` once - the count is asserted so a third line nobody decided on is a
    failure here rather than a surprise in a CI log."""
    forbid_store(monkeypatch)
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", verdict, "pass"), out=out)

    printed = capsys.readouterr()
    assert code == expected
    assert printed.err.count(UNSTORED_LINE) == 1
    assert printed.err.count(UNJUDGED_LINE) == 1
    assert len(printed.err.strip().splitlines()) == 2
    assert (out / REPORT_JSON_NAME).is_file()


def test_the_rows_are_stored_once_after_both_report_files_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Decision E's ordering. Catches a store called BEFORE `write_reports`, where an
    insert that failed would cost the run both its rows and its two files - the artefact is
    what survives a storage outage, and row 18's gate reads it off disk. Catches a store
    called once per case, which would need a transaction per case and is decision G's
    failure mode, and a store never called at all when a URL IS configured. The rows are
    asserted against the report just written, so a CLI storing a second, freshly built
    report fails here."""
    out = tmp_path / "reports"
    store = RecordingStore(out)
    pooled = install_store(monkeypatch, store)

    code = run_cli(monkeypatch, tmp_path, out=out)

    assert code == EXIT_OK
    assert pooled == [DATABASE_URL]
    assert len(store.calls) == 1
    assert store.reports_on_disk == [True]
    run_row, case_rows = store.calls[0]
    assert run_row.id == read_report(out)["id"]
    assert len(case_rows) == 3


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("pass", EXIT_USAGE), ("residency", EXIT_RESIDENCY)],
    ids=["a passing run becomes exit 2", "a residency violation stays exit 3"],
)
def test_a_store_that_fails_reports_two_unless_the_run_already_answered_three(
    verdict: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Decision E's exit rule, both halves in one shape. Catches a storage failure
    swallowed into exit 0, which would let a gate report a run whose rows nobody can find,
    and catches the opposite - a `2` written over a residency `3`, which downgrades a
    boundary breach to a database complaint and is exactly what a project rule's spirit
    forbids. The reports are asserted still on disk, because the run was paid for either
    way, and the message is asserted as a message rather than a traceback."""
    out = tmp_path / "reports"
    store = RecordingStore(out, failure=RuntimeError("the eval store is unreachable"))
    install_store(monkeypatch, store)

    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", verdict, "pass"), out=out)

    printed = capsys.readouterr()
    assert code == expected
    assert len(store.calls) == 1
    assert (out / REPORT_JSON_NAME).is_file()
    assert (out / REPORT_MD_NAME).is_file()
    assert "Traceback" not in printed.err


# --------------------------------------------------------------------------
# --judge-url and the key: off unless both are present (17c)
# --------------------------------------------------------------------------


JUDGE_URL = "https://gateway.example"

JUDGE_KEY = "sk-eval-judge-not-a-real-key"


@pytest.mark.parametrize(
    ("label", "judge_url"), REJECTED_TARGETS, ids=[row[0] for row in REJECTED_TARGETS]
)
def test_a_judge_url_that_is_not_an_absolute_http_url_is_refused(
    label: str, judge_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Criterion 10, against `_checked_target`'s own table so the two URLs this command
    takes cannot drift apart - a `--judge-url` of `target.example` or `file:///etc/passwd`
    is the same defect as a `--target` of one. Catches a judge URL handed to httpx
    unchecked, which surfaces as a transport error per rubric case after the seed is reset
    and the run is being paid for. The key IS configured on every row, so the judge is
    meant to be ON and "there is no key, so nothing was checked" cannot be what refuses
    them - removing the pre-check is the only thing that flips this table."""
    runner = RecordingRunner()

    code = run_cli(monkeypatch, tmp_path, runner=runner, judge_url=judge_url, judge_key=JUDGE_KEY)

    assert code == EXIT_USAGE
    assert runner.calls == []


def test_the_judge_url_is_refused_before_the_target_is_touched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ordering half of criterion 10, which the exit code alone cannot see: the check
    belongs in the pre-check block that runs BEFORE the reset, not beside the first
    judge call. Catches a CLI that resets the seed, runs two hundred cases and then
    refuses the judge URL at the end - the run is spent either way, and the report is not
    written. Everything else on this command line is valid, so only the judge URL can
    produce the 2."""
    out = tmp_path / "reports"
    runner = RecordingRunner()

    code = run_cli(
        monkeypatch,
        tmp_path,
        runner=runner,
        out=out,
        judge_url="not-a-url",
        judge_key=JUDGE_KEY,
    )

    assert code == EXIT_USAGE
    assert runner.calls == []
    assert not (out / REPORT_JSON_NAME).exists()


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("pass", EXIT_OK), ("fail", EXIT_FAILED), ("residency", EXIT_RESIDENCY)],
    ids=["a clean run", "a failed assertion", "a residency violation"],
)
def test_with_no_judge_url_the_run_judges_nothing_and_keeps_its_exit_code(
    verdict: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Decision B, the shape 16d took for the store. Catches a CLI that refuses to run
    without a judge - which would turn the whole harness off on every machine with no
    gateway up rather than turning the judge off - and catches a judge built anyway from
    a key alone, which would make paid calls nobody asked for: the key IS configured here
    and no URL is given, so `judge=None` is the only correct hand-off. All three codes are
    covered because a CLI that returned the clean code correctly could still swallow the
    residency 3 while reporting the judge was skipped."""
    out = tmp_path / "reports"

    runner = RecordingRunner("pass", verdict, "pass")

    code = run_cli(monkeypatch, tmp_path, runner=runner, out=out, judge_key=JUDGE_KEY)

    document = read_report(out)
    assert code == expected
    assert runner.judges == [None]
    assert document["metrics"]["judge_mean"] is None
    assert document["judge_prompt_version"] is None
    assert capsys.readouterr().err.count(UNJUDGED_LINE) == 1


def test_with_no_judge_key_the_run_judges_nothing_and_says_so_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of decision B, and the one an operator actually hits: the URL is on
    the command line, so they believe the judge ran. Catches silence, which leaves a run
    that scored nothing looking like a run that scored everything well; catches a line per
    CASE, which is noise in a CI log and the reason the count is asserted as exactly one;
    and catches a judge constructed with an empty key, which every rubric case would then
    spend three 401s on. The exit code is the run's own, so the missing key is not an
    error - it is a fact the report records as `null`."""
    runner = RecordingRunner("pass", "fail", "pass")
    out = tmp_path / "reports"

    code = run_cli(
        monkeypatch, tmp_path, runner=runner, out=out, judge_url=JUDGE_URL, judge_key=None
    )

    printed = capsys.readouterr()
    assert code == EXIT_FAILED
    assert runner.judges == [None]
    assert printed.err.count(UNJUDGED_LINE) == 1
    assert read_report(out)["metrics"]["judge_mean"] is None


# --------------------------------------------------------------------------
# ops-eval calibrate: the labels, the artifact, the refusals (17e)
# --------------------------------------------------------------------------


def calibration_labels(
    tmp_path: Path, *, count: int = MIN_CALIBRATION_LABELS, name: str = "labels.jsonl"
) -> Path:
    """A well-formed labels file of `count` lines, one JSON object each.

    The shape `evals/calibration.py` loads, written here rather than shipped as a
    fixture: a forty-line file of fabricated grading data is exactly what an earlier change
    decision G kept out of the repo.
    """
    path = tmp_path / name
    lines = [
        json.dumps(
            {
                "id": f"label-{index:03d}",
                "output": f"reply for label-{index:03d}",
                "human_score": 1 if index % 2 else 5,
                "judge_rubric": f"rubric for label-{index:03d}",
            }
        )
        for index in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class CalibrationJudge:
    """The gateway `calibrate` scores against, and the seam that puts it there.

    `install` replaces `cli.JudgeClient` - the module attribute, exactly as
    `cli.run_set`, `cli.build_pool` and `cli.PostgresEvalStore` are replaced above -
    with a factory that hands back a real `JudgeClient` over an
    `httpx.MockTransport`. The CLI's calibrate path opens its own `httpx.Client`,
    so this is the one seam that keeps the suite off the network without doubling
    the client itself; the `JudgeClient` is real, so 17c's sampling, retry and
    median all still run.

    `requests` is what the pre-check cases assert on: a CLI that validated its
    arguments AFTER the first request would still exit 2 and would still leave no
    file behind.
    """

    def __init__(self, *, status: int = 200, score: int = 4) -> None:
        self.status = status
        self.score = score
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "the judge is down"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"score": self.score, "reason": "It meets the rubric."}
                            ),
                        }
                    }
                ]
            },
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def build(
            base_url: str,
            *,
            client: httpx.Client,
            api_key: str,
            timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
        ) -> JudgeClient:
            return JudgeClient(
                base_url,
                client=httpx.Client(transport=httpx.MockTransport(self)),
                api_key=api_key,
                timeout=timeout,
            )

        monkeypatch.setattr(cli, "JudgeClient", build)


def run_calibrate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    judge: CalibrationJudge,
    labels: Path | None = None,
    judge_url: str = JUDGE_URL,
    out: Path | None = None,
    judge_key: str | None = JUDGE_KEY,
) -> tuple[int, Path]:
    """Run `ops-eval calibrate` against a doubled gateway, and answer the code and `--out`.

    The key is written to (or cleared from) the environment for the reason the run
    cases clear it: `Settings().eval_judge_key` reads it there, and a developer's
    own shell must not decide whether a calibration is refused.
    """
    judge.install(monkeypatch)
    if judge_key is None:
        monkeypatch.delenv("OPSCONSOLE_EVAL_JUDGE_KEY", raising=False)
    else:
        monkeypatch.setenv("OPSCONSOLE_EVAL_JUDGE_KEY", judge_key)
    directory = out if out is not None else tmp_path / "calibration"
    path = labels if labels is not None else calibration_labels(tmp_path)
    code = cli.main(
        [
            "calibrate",
            "--labels",
            str(path),
            "--judge-url",
            judge_url,
            "--out",
            str(directory),
        ]
    )
    return code, directory


def test_calibrate_writes_the_artifact_into_out_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The happy path of the second subcommand, read off DISK rather than off a doubled
    writer: catches a CLI that computes a calibration and never writes it, which leaves 17f
    and row 18 with nothing to read, and catches a file written somewhere other than
    `--out`. One line on stdout, because `make eval`'s output is read in a CI log and the
    exit code is the interface. The forty labels are scored through a real `JudgeClient`,
    so a subcommand that skipped the judge entirely could not produce forty rows."""
    judge = CalibrationJudge()

    code, out = run_calibrate(monkeypatch, tmp_path, judge=judge)

    printed = capsys.readouterr()
    document = json.loads((out / CALIBRATION_JSON_NAME).read_text(encoding="utf-8"))
    assert code == EXIT_OK
    assert len(printed.out.strip().splitlines()) == 1
    assert document["pairs"] == MIN_CALIBRATION_LABELS
    assert document["dropped"] == 0
    assert judge.requests != []


@pytest.mark.parametrize(
    ("label", "judge_url"), REJECTED_TARGETS, ids=[row[0] for row in REJECTED_TARGETS]
)
def test_a_calibrate_judge_url_that_is_not_an_absolute_http_url_is_refused(
    label: str,
    judge_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--judge-url` on this subcommand is checked by `_checked_judge_url`, the same rule the
    run path uses, BEFORE anything is sent. Catches a second copy of the URL rule that
    drifts, catches the flag not being named in the message - an operator who typed two
    URLs needs to know which one was refused - and catches a check placed after the first
    request, which the empty request log is the only thing that can see. Everything else on
    this command line is valid, so only the judge URL can produce the 2."""
    judge = CalibrationJudge()

    code, out = run_calibrate(monkeypatch, tmp_path, judge=judge, judge_url=judge_url)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert judge.requests == []
    assert "--judge-url" in printed.err
    assert not (out / CALIBRATION_JSON_NAME).exists()


def test_calibrate_with_no_judge_key_is_a_usage_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Where `run` turns the judge OFF with no key, `calibrate` has nothing left to do, so it
    refuses. Catches a calibrate that borrows the run path's decision B and prints
    `UNJUDGED_LINE` while writing an artifact with no kappa in it - a `calibration.json`
    that exists is read by 17f as trust - and catches a `JudgeClient` built with an empty
    key, which would spend forty labels' worth of 401s. One line, on stderr, and no file."""
    judge = CalibrationJudge()

    code, out = run_calibrate(monkeypatch, tmp_path, judge=judge, judge_key=None)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert judge.requests == []
    assert len(printed.err.strip().splitlines()) == 1
    assert not (out / CALIBRATION_JSON_NAME).exists()


def test_a_judge_that_answers_five_hundred_throughout_writes_nothing_and_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An earlier change's unhappy path. Every sample fails, so every label is dropped and no pairs
    survive the floor: catches an artifact written with `pairs: 0` and a null kappa, which
    is a file a later run trusts more than the warning nobody kept, and catches the
    `CalibrationRefused` reaching the operator as a traceback rather than as one line. The
    labels file and the URL are both valid here, so only the judge outage can produce the
    2 - the pre-check cases above cannot be what fails this row."""
    judge = CalibrationJudge(status=500)

    code, out = run_calibrate(monkeypatch, tmp_path, judge=judge)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert judge.requests != []
    assert len(printed.err.strip().splitlines()) == 1
    assert "Traceback" not in printed.err
    assert not (out / CALIBRATION_JSON_NAME).exists()


def test_a_labels_path_that_does_not_exist_is_a_usage_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mistyped path, which is the commonest way this command is run wrong. Catches an
    `OSError` from the loader escaping as a traceback, and catches the refusal being taken
    for a failed calibration: `EXIT_USAGE` is the code that tells a gate the harness never
    ran. The message is one line and names the path an operator typed."""
    missing = tmp_path / "labels" / "nothing-here.jsonl"
    judge = CalibrationJudge()

    code, out = run_calibrate(monkeypatch, tmp_path, judge=judge, labels=missing)

    printed = capsys.readouterr()
    assert code == EXIT_USAGE
    assert judge.requests == []
    assert len(printed.err.strip().splitlines()) == 1
    assert str(missing) in printed.err
    assert not (out / CALIBRATION_JSON_NAME).exists()
