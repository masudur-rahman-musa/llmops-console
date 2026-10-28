"""`evals/baseline_guard.py` and `ops-eval baseline-guard`: both baseline files are guarded.

The pure rule is a table of commits; the git read is driven against a throwaway repository.

Contract these tests are written against:

* `BASELINE_PATHS == ("backend/evals/baseline.json", "backend/evals/conversation-baseline.json")`
  replaces `BASELINE_PATH`. `BASELINE_PREFIX`, `ADR_PREFIX`, `MAX_REF_LENGTH`, `REF_PATTERN`,
  `MISSING_PREFIX_REASON`, `MISSING_ADR_REASON` and `GUARD_PASSED_LINE` keep their values.
* A commit is examined only when one of its `files` is exactly one of `BASELINE_PATHS`
  (a deletion counts: the old file's path is in `files`). It is clean when its subject starts
  with `BASELINE_PREFIX` and some file starts with `ADR_PREFIX`. A byte-identical move
  (`exact_moves`) onto a watched path from a file of the same name is not a change.
* `Offence(sha, reasons, paths)`: `paths` are the watched paths the commit touched, in
  `BASELINE_PATHS` order. `offence_line` reads
  `f"{sha}: touches {' and '.join(paths)} but {' and '.join(reasons)}"`.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.cli import evals as cli
from app.cli.evals import EXIT_FAILED, EXIT_OK
from app.evals.baseline_guard import (
    ADR_PREFIX,
    BASELINE_PATHS,
    BASELINE_PREFIX,
    GUARD_PASSED_LINE,
    MAX_REF_LENGTH,
    MISSING_ADR_REASON,
    MISSING_PREFIX_REASON,
    REF_PATTERN,
    Commit,
    Offence,
    check,
    offence_line,
)

OLD = "backend/evals/baseline.json"
NEW = "backend/evals/conversation-baseline.json"
ADR = "docs/adr/0040-conversation-baseline.md"
GOOD_SUBJECT = "baseline: set the conversation baseline"


def test_the_published_values_are_pinned() -> None:
    """Catches a guard watching only the new file (the old one could be re-added quietly) or
    only the old one (the new baseline could move without an ADR), a changed prefix, or a
    ref cap of 255."""
    assert BASELINE_PATHS == (OLD, NEW)
    assert BASELINE_PREFIX == "baseline:"
    assert ADR_PREFIX == "docs/adr/"
    assert MAX_REF_LENGTH == 256
    assert MISSING_PREFIX_REASON == 'the subject does not start "baseline:"'
    assert MISSING_ADR_REASON == "the commit adds no file under docs/adr/"
    assert GUARD_PASSED_LINE.format(checked=2) == "baseline guard passed: 2 commits checked"


def test_the_ref_pattern_matches_whole_values_only() -> None:
    r"""Catches a pattern applied with `.match` that lets `HEAD\n` through (a second argument
    smuggled into `git log`), one that admits `..`, a leading dash or an empty ref, and one
    that refuses the longest allowed ref."""
    assert REF_PATTERN.fullmatch("a" * MAX_REF_LENGTH) is not None
    for bad in ("HEAD\n", "HEAD\r\n", "HEAD ", "a..b", "-upload-pack=evil", ""):
        assert REF_PATTERN.fullmatch(bad) is None, repr(bad)


@pytest.mark.parametrize("path", BASELINE_PATHS)
def test_a_baseline_change_needs_the_prefix_and_an_adr(path: str) -> None:
    """Catches a guard that checks one half only, or only one of the two files. Each row is
    arranged so exactly one half is missing: with the prefix and no ADR the ADR reason is the
    only one; with an ADR and no prefix the prefix reason is the only one."""
    no_adr = check([Commit(sha="a1", subject=GOOD_SUBJECT, files=(path,))])
    no_prefix = check([Commit(sha="b2", subject="update numbers", files=(path, ADR))])
    clean = check([Commit(sha="c3", subject=GOOD_SUBJECT, files=(path, ADR))])

    assert [o.reasons for o in no_adr.offences] == [(MISSING_ADR_REASON,)]
    assert [o.reasons for o in no_prefix.offences] == [(MISSING_PREFIX_REASON,)]
    assert no_adr.offences[0].paths == (path,)
    assert clean.passed is True


def test_a_commit_breaking_both_halves_is_one_offence_with_both_reasons() -> None:
    """Catches two offences for one sha, or one that drops a reason. The order is fixed:
    prefix first, then ADR."""
    verdict = check([Commit(sha="a1", subject="tweak", files=(NEW,))])

    assert verdict.offences == (
        Offence(sha="a1", reasons=(MISSING_PREFIX_REASON, MISSING_ADR_REASON), paths=(NEW,)),
    )


def test_the_commit_that_swaps_the_baselines_is_clean_with_prefix_and_adr() -> None:
    """Catches a guard that fails its own commit: it deletes `baseline.json` and adds
    `conversation-baseline.json`, with the prefix and its ADR."""
    swap = Commit(sha="a1", subject=GOOD_SUBJECT, files=(OLD, NEW, ADR))

    assert check([swap]).passed is True


def test_deleting_the_old_baseline_alone_still_needs_the_prefix_and_an_adr() -> None:
    """Catches a guard that watches only files that exist after the commit: a deletion of
    `baseline.json` is a baseline change like any other."""
    verdict = check([Commit(sha="a1", subject="tidy", files=(OLD,))])

    assert [o.paths for o in verdict.offences] == [(OLD,)]


def test_a_commit_touching_both_files_names_both_paths_in_order() -> None:
    """Catches an offence that names one of the two files: the author must see both."""
    verdict = check([Commit(sha="a1", subject="tidy", files=(NEW, OLD))])

    assert verdict.offences[0].paths == (OLD, NEW)


@pytest.mark.parametrize(
    "path",
    [
        NEW + ".bak",
        "x/" + NEW,
        "backend/evals/conversation-baseline.json/old",
        "backend/evals/baseline.jsonl",
        "backend/evals/conversation-baselines.json",
        "conversation-baseline.json",
        "backend/evals/Conversation-baseline.json",
    ],
)
def test_a_path_that_is_not_exactly_a_baseline_is_not_examined(path: str) -> None:
    """Catches a substring or suffix match, which would fail every pull request touching a
    near-miss name. The subject and files would be an offence if the commit were examined."""
    assert check([Commit(sha="a1", subject="tweak", files=(path,))]).passed is True


def test_only_a_file_under_the_adr_directory_counts() -> None:
    """Catches an ADR check that accepts a near-miss folder or a file merely containing the
    directory name."""
    for elsewhere in ("adr/0040.md", "x/docs/adr/0040.md", "docs/adrs/0040.md"):
        verdict = check([Commit(sha="a1", subject=GOOD_SUBJECT, files=(NEW, elsewhere))])
        assert [o.reasons for o in verdict.offences] == [(MISSING_ADR_REASON,)], elsewhere


def test_every_commit_is_examined_not_just_the_first_bad_one() -> None:
    """Catches an early return: a documented commit must not hide an undocumented one after
    it, and the offences keep commit order."""
    verdict = check(
        [
            Commit(sha="a1", subject=GOOD_SUBJECT, files=(NEW, ADR)),
            Commit(sha="b2", subject="tweak", files=(NEW,)),
            Commit(sha="c3", subject="tweak again", files=(NEW,)),
        ]
    )

    assert [o.sha for o in verdict.offences] == ["b2", "c3"]


def test_an_exact_move_onto_a_baseline_is_not_a_change() -> None:
    """Catches a guard that fails a pure `git mv`: nothing in the file changed, so there is
    nothing for an ADR to justify."""
    move = Commit(
        sha="a1",
        subject="move the files",
        files=("old/conversation-baseline.json", NEW),
        exact_moves=(("old/conversation-baseline.json", NEW),),
    )

    assert check([move]).passed is True


def test_a_different_file_moved_onto_a_baseline_is_still_judged() -> None:
    """Catches an exemption that accepts any exact move: renaming `scratch.json` over the
    baseline replaces its content unjudged. The old name is not the baseline's name."""
    move = Commit(
        sha="a1", subject="move", files=("scratch.json", NEW), exact_moves=(("scratch.json", NEW),)
    )

    assert [o.sha for o in check([move]).offences] == ["a1"]


def test_moving_a_baseline_away_is_still_judged() -> None:
    """Catches an exemption keyed on the old side: the file leaves the guarded path, so the
    move must be judged."""
    away = Commit(
        sha="a1",
        subject="move",
        files=(NEW, "old/conversation-baseline.json"),
        exact_moves=((NEW, "old/conversation-baseline.json"),),
    )

    assert [o.sha for o in check([away]).offences] == ["a1"]


def test_an_offence_line_names_the_sha_the_files_and_every_reason() -> None:
    """Catches a line the author cannot act on: the sha first, then the touched files, then
    both reasons joined by 'and'."""
    line = offence_line(
        Offence(sha="a1", reasons=(MISSING_PREFIX_REASON, MISSING_ADR_REASON), paths=(OLD, NEW))
    )

    assert line == (
        f"a1: touches {OLD} and {NEW} but {MISSING_PREFIX_REASON} and {MISSING_ADR_REASON}"
    )


# --- the git read, through the command ---------------------------------------------


@dataclass(frozen=True)
class GuardRepo:
    path: Path
    shas: dict[str, str]


def git(repo: Path, *arguments: str) -> str:
    """One git command, with the developer's own config unable to reach it."""
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repo,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def commit(repo: Path, subject: str, files: Sequence[str], *, remove: Sequence[str] = ()) -> str:
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{subject}\n", encoding="utf-8")
        git(repo, "add", "--", name)
    for name in remove:
        git(repo, "rm", "-q", "--", name)
    git(repo, "commit", "--no-gpg-sign", "-m", subject)
    return git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture(scope="module")
def guard_repo(tmp_path_factory: pytest.TempPathFactory) -> GuardRepo:
    """One small history: a root with the old baseline, the swap commit done the right
    way, a change to the new file with no prefix, and a deletion of the old file with none."""
    repo = tmp_path_factory.mktemp("guard-repo")
    git(repo, "init", "--initial-branch=main", ".")
    root = commit(repo, "add the readme and the old baseline", ["README.md", OLD])
    swap = commit(repo, GOOD_SUBJECT, [NEW, ADR], remove=[OLD])
    no_prefix = commit(repo, "update the numbers", [NEW, "docs/adr/0041-x.md"])
    readd = commit(repo, "bring the old one back", [OLD])
    return GuardRepo(repo, {"root": root, "swap": swap, "no_prefix": no_prefix, "readd": readd})


def run_guard(base: str, head: str) -> int:
    return cli.main(["baseline-guard", f"--base={base}", f"--head={head}"])


def test_the_swap_commit_passes_through_git(
    guard_repo: GuardRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a git read that loses the deleted file's path or the new file's path: the swap
    deletes `baseline.json` and adds `conversation-baseline.json`, well formed, and must pass
    with the one commit counted."""
    monkeypatch.chdir(guard_repo.path)

    code = run_guard(guard_repo.shas["root"], guard_repo.shas["swap"])

    printed = capsys.readouterr()
    assert code == EXIT_OK
    assert printed.out.strip() == GUARD_PASSED_LINE.format(checked=1)
    assert printed.err == ""


def test_the_guard_watches_both_baseline_files(
    guard_repo: GuardRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a guard that still watches only `baseline.json`: a commit changing
    `conversation-baseline.json` without the prefix is an offence (the ADR file is present,
    so the prefix is the only half missing), and bringing `baseline.json` back without
    either half is another. One stderr line per sha, nothing on stdout."""
    monkeypatch.chdir(guard_repo.path)

    code = run_guard(guard_repo.shas["swap"], guard_repo.shas["readd"])

    printed = capsys.readouterr()
    assert code == EXIT_FAILED
    assert printed.out == ""
    assert sorted(printed.err.strip().splitlines()) == sorted(
        [
            f"{guard_repo.shas['no_prefix']}: touches {NEW} but {MISSING_PREFIX_REASON}",
            f"{guard_repo.shas['readd']}: touches {OLD} but "
            f"{MISSING_PREFIX_REASON} and {MISSING_ADR_REASON}",
        ]
    )
