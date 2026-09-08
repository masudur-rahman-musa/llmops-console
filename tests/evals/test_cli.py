"""`evals/cli.py`: `ops-eval run` - the pre-checks, the four exit codes, one line.

No network and no backing service: `run_set` is replaced by a recording stand-in,
so what is under test here is the SHELL - what the CLI refuses before anything
runs, what integer it answers, and what it prints. The loop itself is pinned
against `httpx.MockTransport` in `tests/evals/test_runner.py`.

The contract these cases are written against, before it exists:

* `main(argv=None) -> int`, wired as `[project.scripts] ops-eval`, with a `run`
  subcommand taking `--target --set --tenant --out` and `--policy`, which
  defaults to the shipped `routing.yaml`.
* `EXIT_OK`, `EXIT_FAILED`, `EXIT_USAGE`, `EXIT_RESIDENCY` - the constants
  a project rule requires a test to pin exactly (decision C).
* `cli.run_set` resolved as a module attribute, which is the seam these cases
  replace.

The tenant table below is arranged against the neighbouring guard: the generated
policy DEFINES the exact tenant string each row passes, malformed ones included,
so the only control that can refuse a row is the `IDENTIFIER_PATTERN` pre-check
and not the "this policy has no such tenant" lookup that follows it. Standing
rule 4's rows (a trailing newline, a trailing CRLF, a trailing space) are in that
table for the same reason they are in 16a's: a check written `^...$` with
`.match` accepts the first of them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from opsconsole.evals import cli
from opsconsole.evals.assertions import AssertionResult, AssertionStatus, CaseAssertions
from opsconsole.evals.cases import EvalCase
from opsconsole.evals.cli import EXIT_FAILED, EXIT_OK, EXIT_RESIDENCY, EXIT_USAGE
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


def make_run(case_id: str, tenant: str, verdict: str) -> CaseRun:
    """One per-case record of the named shape, as `run_set` would have built it."""
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
        assertions=CaseAssertions(case_id=case_id, results=tuple(results)),
        residency=residency,
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

    def __call__(
        self,
        *,
        client: TargetClient,
        cases: tuple[EvalCase, ...],
        policy: Policy,
        tenant: str,
        other_tenant_markers: tuple[str, ...] = (),
    ) -> RunResult:
        self.calls.append(tenant)
        verdicts = self.verdicts if self.verdicts else ("pass",) * len(cases)
        return RunResult(
            runs=tuple(
                make_run(case.id, tenant, verdict)
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
) -> list[str]:
    """One `ops-eval run` command line."""
    arguments = ["run", "--target", target, "--set", str(case_set), "--tenant", tenant]
    if out is not None:
        arguments += ["--out", str(out)]
    if policy is not None:
        arguments += ["--policy", str(policy)]
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
) -> int:
    """Run the CLI with a stand-in runner and a policy defining the tenant under test."""
    installed = runner if runner is not None else RecordingRunner()
    install(monkeypatch, installed)
    tenants = tuple(policy_tenants) if policy_tenants is not None else (tenant,)
    return cli.main(
        argv(
            target=target,
            case_set=case_set,
            tenant=tenant,
            out=out if out is not None else tmp_path / "reports",
            policy=write_policy(tmp_path, *tenants),
        )
    )


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


def test_nothing_is_written_into_the_out_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches this plan writing a report (decision D): the report file, its schema and the
    `eval_runs` rows are `16c`'s, and a half-written `report.json` shipped here is a file
    format nobody agreed and a consumer might start reading. The directory is validated and
    created, and it stays empty."""
    out = tmp_path / "reports"

    code = run_cli(monkeypatch, tmp_path, runner=RecordingRunner("pass", "fail", "pass"), out=out)

    assert code == EXIT_FAILED
    assert list(out.iterdir()) == []


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
