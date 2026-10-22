"""Argument checking and parser for `ops-eval`.

Everything that happens before a command runs: the class that signals a bad
argument, every pre-check function, and the `_parser` that wires them up.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import Settings
from app.evals.baseline_guard import (
    MAX_REF_LENGTH,
    REF_PATTERN,
    Commit,
)
from app.evals.report import SET_SHA_PATTERN
from app.evals.runner import MAX_REPEATS, RepeatsInvalid, checked_repeats

ALLOWED_TARGET_SCHEMES: frozenset[str] = frozenset({"http", "https"})

RUN_KINDS: tuple[str, ...] = ("ci", "nightly", "manual")

DEFAULT_RUN_KIND: str = "manual"

DEFAULT_LABELS_PATH: Path = Path("backend/evals/labels/conversation-labels.jsonl")


class UsageError(Exception):
    """One argument was refused. Reported on stderr, and answered with `EXIT_USAGE`."""


def _checked_target(value: str) -> str:
    """Refuse anything but an absolute `http` or `https` URL.

    An unchecked value produces an `httpx` error mid-run, after the seed has been
    reset and the first cases have been paid for; `file://` is a scheme no eval
    target speaks.
    """
    parsed = urlparse(value)
    if parsed.scheme not in ALLOWED_TARGET_SCHEMES or not parsed.netloc:
        raise UsageError(f"--target: {value!r} is not an absolute http(s) URL")
    return value


def _checked_repeats(value: int) -> int:
    """`--repeats`: 1 to `MAX_REPEATS`, checked by the runner's own rule before any reset."""
    try:
        return checked_repeats(value)
    except RepeatsInvalid:
        raise UsageError(f"--repeats: {value} is not between 1 and {MAX_REPEATS}") from None


def _checked_judge_url_value(value: str) -> str:
    """Validate a `--judge-url` that was given, by `_checked_target`'s own rule.

    The same function and not a second copy of the rule: the two URLs this command
    takes would otherwise drift, and a judge URL of `gateway.example` is the same
    defect as a target of one.
    The message still names the
    flag the operator typed.

    Split from `_checked_judge_url` because the flag is OPTIONAL on `conversations` and
    REQUIRED on `calibrate` and `label`: one function answering `str | None` would
    leave the required caller narrowing a value argparse already guaranteed.
    """
    try:
        return _checked_target(value)
    except UsageError:
        raise UsageError(f"--judge-url: {value!r} is not an absolute http(s) URL") from None


def _checked_judge_url(value: str | None) -> str | None:
    """`--judge-url` on `conversations`, where absent means the judge is simply off."""
    if value is None:
        return None
    return _checked_judge_url_value(value)


def _checked_judge_key() -> str:
    """The judge key `calibrate` and `label` cannot run without.

    Where `conversations` turns the judge OFF with no key and keeps its own exit code,
    a calibration with no judge has nothing left to do, so it
    refuses rather than writing an artifact with no kappa in it - a
    `calibration.json` that exists is read as trust.
    An empty string is
    the same as absent: a `JudgeClient` built on one would spend a 401 per sample.
    """
    api_key = Settings().eval_judge_key
    if not api_key:
        raise UsageError(
            "OPSCONSOLE_EVAL_JUDGE_KEY: not set, and a calibration cannot be made without the judge"
        )
    return api_key


def _checked_ref(value: str, flag: str) -> str:
    """Validate one `--base` or `--head` before git is asked anything .

    The cap and the pattern come from `baseline_guard`, and the order is the point:
    both values reach a subprocess argument list, so a value starting `-` is an
    option to `git log` and a value carrying a newline is a second argument. Neither
    is refusable once the command line is built, which is why this runs first.

    The length is measured before the pattern is applied, and the boundary is
    inclusive - 256 in, 257 out.
    """
    if len(value) > MAX_REF_LENGTH:
        raise UsageError(f"{flag}: longer than {MAX_REF_LENGTH} characters")
    if REF_PATTERN.fullmatch(value) is None:
        raise UsageError(
            f"{flag}: not a ref this command will pass to git "
            "(letters, digits, `.`, `_`, `/` and `-`, no leading `-`, no `..`)"
        )
    return value


def _commits(base: str, head: str) -> tuple[Commit, ...]:
    r"""Read `base..head` once and answer what each commit touched.

    `git log --format=%H%x00%s --name-status -M100%` prints a header line
    `<sha>\x00<subject>`, then a blank line, then one `<status>\t<path>` line per file
    until the next header. The NUL is what separates the sha from a subject that may
    contain anything, including the colon every well-formed baseline commit's subject
    carries.

    `-M100%` makes git report only a byte-identical move as `R100\told\tnew`; any edit
    turns it into a delete and an add, so it is judged like any change. Both sides of
    a move go into `files`, so moving the baseline away is still seen.

    A range git cannot resolve exits non-zero, and that is reported as a refusal
    rather than as an empty commit list: a shallow checkout would otherwise let this
    guard report clean having read nothing.
    git's own message runs to three lines, so what is printed is this command's own
    one line naming the range.
    """
    revisions = f"{base}..{head}"
    try:
        finished = subprocess.run(
            ["git", "log", "--format=%H%x00%s", "--name-status", "-M100%", revisions],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UsageError(f"the range {revisions} could not be read by git: {exc}") from exc
    if finished.returncode != 0:
        raise UsageError(f"the range {revisions} could not be read by git")

    commits: list[Commit] = []
    shas: list[str] = []
    subjects: list[str] = []
    touched: list[list[str]] = []
    moves: list[list[tuple[str, str]]] = []
    for line in finished.stdout.splitlines():
        if "\x00" in line:
            sha, _, subject = line.partition("\x00")
            shas.append(sha)
            subjects.append(subject)
            touched.append([])
            moves.append([])
        elif line and touched:
            status, *paths = line.split("\t")
            touched[-1].extend(paths)
            if status == "R100" and len(paths) == 2:
                moves[-1].append((paths[0], paths[1]))
    for sha, subject, files, exact in zip(shas, subjects, touched, moves, strict=True):
        commits.append(
            Commit(sha=sha, subject=subject, files=tuple(files), exact_moves=tuple(exact))
        )
    return tuple(commits)


def _checked_out(value: Path) -> Path:
    """Create the report directory, or refuse the path.

    A path that exists as a FILE is refused rather than replaced: the two report
    writes would fail at the end of a paid-for run, and an implementation that
    "fixed" it would delete whatever an operator named.
    """
    if value.exists() and not value.is_dir():
        raise UsageError(f"--out: {value} exists and is not a directory")
    value.mkdir(parents=True, exist_ok=True)
    return value


def _checked_set_sha(value: str | None) -> str | None:
    """Validate `--set-sha`, or answer `None` when the caller gave none.

    7 to 64 lowercase hex, inclusive at both ends, with `re.fullmatch` and never
    `.match`: `$` also matches before a trailing newline in Python, and
    this value is stored beside a run as the version of the set that
    produced it.
    """
    if value is None:
        return None
    if SET_SHA_PATTERN.fullmatch(value) is None:
        raise UsageError(f"--set-sha: {value!r} is not 7 to 64 lowercase hex characters")
    return value


def _set_sha(supplied: str | None, case_set: Path) -> tuple[str, str]:
    """Which version of the set this was, and how that answer was arrived at.

    The cases live in the Helpdesk repo, whose commit this process cannot
    see, so a caller who knows it passes it and everyone else gets the sha256 of
    the file's BYTES. The source travels with the value so no
    reader is ever told a content hash is a commit.
    """
    if supplied is not None:
        return supplied, "commit"
    return hashlib.sha256(case_set.read_bytes()).hexdigest(), "content"


def _git_sha() -> str | None:
    """This repo's `HEAD`, or `None` when git cannot answer.

    Never a guess and never a default string: `unknown` in a stored column is a
    value a query has to know about, while `null` is the absence it already
    handles. All three shapes of failure answer `None` - no git binary, a
    non-zero exit outside a repository, and a raising call - and the
    trailing newline `rev-parse` prints is stripped, or every equality query
    against a commit would miss.
    """
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip() or None


def _new_run_id() -> str:
    """A run id for a run that was not given one: `run-` and the UTC second it started.

    A lowercase slug, so it passes the same check a typed one does and names a file.
    """
    return "run-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def _parser() -> argparse.ArgumentParser:
    """`ops-eval` and its six subcommands: `conversations`, `label`, `calibrate`, `gate`,
    `baseline-guard`, `tradeoff`. The old `run` is gone: argparse refuses it with exit 2."""
    parser = argparse.ArgumentParser(
        prog="ops-eval", description="Play and score the help desk's conversation cases."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    conversations = subcommands.add_parser(
        "conversations", help="play the conversation cases against a help desk and score them"
    )
    conversations.add_argument(
        "--target", required=True, help="base URL of the help desk under evaluation"
    )
    conversations.add_argument(
        "--cases", required=True, type=Path, help="the help desk's conversation cases, JSONL"
    )
    conversations.add_argument(
        "--out",
        required=True,
        type=Path,
        help="where <run-id>.jsonl, report.json and report.md are written",
    )
    conversations.add_argument(
        "--run-id",
        dest="run_id",
        help="a short lowercase name; the same name on a restart skips saved cases "
        "(default: run- and the start time)",
    )
    conversations.add_argument(
        "--set-sha",
        dest="set_sha",
        help="the commit the cases came from; without it, the file's own sha256",
    )
    conversations.add_argument(
        "--run-kind",
        dest="run_kind",
        choices=RUN_KINDS,
        default=DEFAULT_RUN_KIND,
        help=f"why this run happened (default: {DEFAULT_RUN_KIND})",
    )
    conversations.add_argument(
        "--judge-url",
        dest="judge_url",
        help="base URL of the gateway that serves the judge alias; without it, no judging",
    )
    conversations.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=f"play every case this many times, 1 to {MAX_REPEATS}; a case that both "
        "passes and fails is unstable and left out of check_score (default: 1)",
    )
    conversations.add_argument(
        "--pushgateway-url",
        dest="pushgateway_url",
        help="base URL of a Pushgateway; without it, gateway_eval_* metrics are not pushed",
    )

    label = subcommands.add_parser(
        "label", help="make the reference labels from a finished run, once"
    )
    label.add_argument("--cases", required=True, type=Path, help="the conversation cases, JSONL")
    label.add_argument("--run", required=True, type=Path, help="the finished run's <run-id>.jsonl")
    label.add_argument(
        "--judge-url",
        dest="judge_url",
        required=True,
        help="base URL of the gateway that serves the agent-strong alias",
    )
    label.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"the labels file; it must not exist yet (default: {DEFAULT_LABELS_PATH})",
    )

    calibrate_command = subcommands.add_parser(
        "calibrate", help="grade the labels through the judge and report the agreement"
    )
    calibrate_command.add_argument(
        "--labels", required=True, type=Path, help="the labelled replies, JSONL"
    )
    calibrate_command.add_argument(
        "--judge-url",
        dest="judge_url",
        required=True,
        help="base URL of the gateway that serves the judge alias",
    )
    calibrate_command.add_argument(
        "--out", required=True, type=Path, help="where calibration.json is written"
    )

    gate_command = subcommands.add_parser(
        "gate", help="compare a published run against the conversation baseline"
    )
    gate_command.add_argument("--report", required=True, type=Path, help="the run's report.json")
    gate_command.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="the checked-in backend/evals/conversation-baseline.json",
    )
    gate_command.add_argument(
        "--calibration",
        type=Path,
        help="a calibration.json; the judge score gates only while its kappa is 0.7 or better",
    )
    gate_command.add_argument(
        "--comment", type=Path, help="where the PR comment markdown is written; without it, none"
    )

    guard_command = subcommands.add_parser(
        "baseline-guard",
        help="refuse a baseline change arriving without its prefix and its ADR",
    )
    guard_command.add_argument(
        "--base", required=True, help="the ref the range starts after (exclusive)"
    )
    guard_command.add_argument("--head", required=True, help="the ref the range ends at")

    tradeoff_command = subcommands.add_parser(
        "tradeoff", help="build the tradeoff table from N published runs"
    )
    tradeoff_command.add_argument(
        "--manifest", required=True, type=Path, help="the manifest naming the options and runs"
    )
    tradeoff_command.add_argument(
        "--out", required=True, type=Path, help="where tradeoff.md and tradeoff.json are written"
    )
    return parser
