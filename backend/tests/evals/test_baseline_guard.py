"""`evals/baseline_guard.py`: the pure half of #42's guard over `backend/evals/baseline.json`.

No git, no I/O and no clock here - `check()` takes the commits it is told about and
answers a verdict, so every bypass of the rule is a row rather than a pull request.
The git read itself is pinned in `tests/evals/test_cli.py` against a real throwaway
repository.

The contract these cases are written against, before it exists:

* `BASELINE_PATH`, `BASELINE_PREFIX`, `ADR_PREFIX`, `MAX_REF_LENGTH` and
  `REF_PATTERN`, the constants a project rule requires pinned by value.
* frozen `Commit(sha, subject, files)`, `Offence(sha, reasons)` and
  `GuardVerdict(offences)` with a `passed` property, all pydantic v2.
* `check(commits) -> GuardVerdict`: a commit is EXAMINED only when `BASELINE_PATH`
  is one of its files as an exact string, and is clean when its subject starts
  with `BASELINE_PREFIX` AND at least one file starts with `ADR_PREFIX`. One
  failing commit yields exactly one `Offence` carrying one or both reasons, in the
  fixed order (prefix, then ADR).
* `offence_line(offence)` and `GUARD_PASSED_LINE`, the two strings the CLI prints.

Every table row below is arranged so that the control it names is the only thing
that can decide it: the ADR rows all carry a well-formed `baseline:` subject, and
the path rows all carry a subject and a file set that would be an offence if the
commit were examined at all - so a substring path match turns the row red rather
than a neighbouring guard.
"""

from __future__ import annotations

import pytest

from app.evals.baseline_guard import (
    ADR_PREFIX,
    BASELINE_PATH,
    BASELINE_PREFIX,
    GUARD_PASSED_LINE,
    MAX_REF_LENGTH,
    MISSING_ADR_REASON,
    MISSING_PREFIX_REASON,
    REF_PATTERN,
    Commit,
    GuardVerdict,
    Offence,
    check,
    offence_line,
)

SHAPE_SHA = "a87abf2"
"""The sha of the commit this repo already made the right way.

Its real shape is the happy-path row below: subject `baseline: write the thresholds
in points`, files `docs/adr/0008-the-baseline-is-written-in-points.md` and
`backend/evals/baseline.json`.
"""

SHAPE_SUBJECT = "baseline: write the thresholds in points"
SHAPE_ADR = "docs/adr/0008-the-baseline-is-written-in-points.md"

OTHER_SHA = "deadbee"


def test_the_four_constants_are_exactly_the_published_values() -> None:
    """Catches an implementation that restates any of the four as its own near-miss - a
    guard watching `backend/evals/baseline.json ` or `docs/adrs/`, or a ref cap of 255 -
    which would pass every case below while guarding a path that does not exist and
    refusing a ref git resolves. Pinned by value because the workflow, the docs and the
    CLI all quote these strings."""
    assert BASELINE_PATH == "backend/evals/baseline.json"
    assert BASELINE_PREFIX == "baseline:"
    assert ADR_PREFIX == "docs/adr/"
    assert MAX_REF_LENGTH == 256


def test_the_three_published_strings_are_exactly_what_the_cli_prints() -> None:
    """The two reasons and the pass line are the whole operator-visible vocabulary of this
    guard, and `tests/evals/test_cli.py` asserts the rendered output against them. Catches a
    module that publishes the constants and prints something else, which would leave those
    CLI cases passing over a stderr line no reviewer can act on, and catches a pass line
    whose placeholder is renamed so `.format(checked=n)` raises at the happy path."""
    assert MISSING_PREFIX_REASON == 'the subject does not start "baseline:"'
    assert MISSING_ADR_REASON == "the commit adds no file under docs/adr/"
    assert GUARD_PASSED_LINE == "baseline guard passed: {checked} commits checked"
    assert GUARD_PASSED_LINE.format(checked=2) == "baseline guard passed: 2 commits checked"


def test_the_ref_pattern_matches_whole_values_only() -> None:
    r"""a project rule at its source: catches a pattern written `^...$` and applied with
    `.match`, which accepts `HEAD\n` because `$` also matches before a trailing newline -
    and a ref carrying a newline is a second argument smuggled into the `git log` command
    line. Catches a pattern that admits `..`, a leading `-` or an empty value too."""
    assert REF_PATTERN.fullmatch("main") is not None
    assert REF_PATTERN.fullmatch("a" * MAX_REF_LENGTH) is not None
    assert REF_PATTERN.fullmatch("HEAD\n") is None
    assert REF_PATTERN.fullmatch("HEAD\r\n") is None
    assert REF_PATTERN.fullmatch("HEAD ") is None
    assert REF_PATTERN.fullmatch("a..b") is None
    assert REF_PATTERN.fullmatch("-upload-pack=evil") is None
    assert REF_PATTERN.fullmatch("") is None


def test_a_range_touching_no_baseline_file_is_clean() -> None:
    """The PR that must never be blocked: catches a guard that examines every commit rather
    than the ones touching the baseline, which would fail every pull request in the repo
    whose subject is not `baseline:` - the guard a team disables in a week. Neither commit
    here has the prefix or an ADR, so only the path test can keep the verdict clean."""
    verdict = check(
        [
            Commit(sha=SHAPE_SHA, subject="add the routing overlay", files=("routing.yaml",)),
            Commit(sha=OTHER_SHA, subject="fix a typo", files=("docs/eval-gate.md",)),
        ]
    )

    assert verdict.passed is True
    assert verdict.offences == ()


def test_the_shape_commit_a87abf2_already_has_passes() -> None:
    """Criterion 2 read off the commit this repo actually made (decision F). Catches a guard
    that demands the ADR and the baseline in separate commits, or that compares the subject
    with equality rather than a prefix - either would refuse the one commit in history that
    is the published example of doing it right."""
    verdict = check(
        [Commit(sha=SHAPE_SHA, subject=SHAPE_SUBJECT, files=(SHAPE_ADR, BASELINE_PATH))]
    )

    assert verdict.passed is True
    assert verdict.offences == ()


def test_a_prefixed_commit_with_no_adr_is_one_offence_naming_the_sha() -> None:
    """Criterion 3's first half: catches a guard that checks the subject and stops, which is
    the cheap half of the rule and the one that lets a threshold move with no justification
    on record. The subject here is well formed, so the ADR test is the only control that
    can produce the offence, and the sha must be in it or CI names no commit to fix."""
    verdict = check([Commit(sha=SHAPE_SHA, subject=SHAPE_SUBJECT, files=(BASELINE_PATH,))])

    assert verdict.passed is False
    assert verdict.offences == (Offence(sha=SHAPE_SHA, reasons=(MISSING_ADR_REASON,)),)


def test_an_adr_commit_with_no_prefix_is_one_offence() -> None:
    """The mirror row: catches a guard that accepts any commit carrying an ADR file. The
    subject is `a87abf2`'s with the prefix stripped, so the commit is otherwise the
    published happy path and only the prefix test can redden it."""
    verdict = check(
        [
            Commit(
                sha=SHAPE_SHA,
                subject="write the thresholds in points",
                files=(SHAPE_ADR, BASELINE_PATH),
            )
        ]
    )

    assert verdict.passed is False
    assert verdict.offences == (Offence(sha=SHAPE_SHA, reasons=(MISSING_PREFIX_REASON,)),)


def test_only_the_second_of_two_baseline_commits_being_malformed_still_fails() -> None:
    """Criterion 3's second half. Catches a guard that returns on the first clean commit it
    finds - the shape a two-commit PR uses to slip a threshold change in behind a properly
    documented one - and catches an offence reported against the wrong sha. The first
    commit is the full happy path, so a guard reading only one commit answers clean."""
    verdict = check(
        [
            Commit(sha=SHAPE_SHA, subject=SHAPE_SUBJECT, files=(SHAPE_ADR, BASELINE_PATH)),
            Commit(sha=OTHER_SHA, subject="bump the floor", files=(BASELINE_PATH,)),
        ]
    )

    assert verdict.passed is False
    assert verdict.offences == (
        Offence(sha=OTHER_SHA, reasons=(MISSING_PREFIX_REASON, MISSING_ADR_REASON)),
    )


def test_a_commit_breaking_both_halves_is_one_offence_carrying_both_reasons() -> None:
    """Criterion 3 says each offending sha is ONE line. Catches a guard that appends an
    offence per broken rule, which prints the same sha twice and makes a reviewer think two
    commits are wrong; and catches one that reports only the first reason it finds, which
    sends the author back for a second CI round over the half it did not mention."""
    verdict = check([Commit(sha=OTHER_SHA, subject="bump the floor", files=(BASELINE_PATH,))])

    assert len(verdict.offences) == 1
    assert verdict.offences[0].sha == OTHER_SHA
    assert verdict.offences[0].reasons == (MISSING_PREFIX_REASON, MISSING_ADR_REASON)


NON_BASELINE_PATHS: tuple[tuple[str, str], ...] = (
    ("a backup beside it", f"{BASELINE_PATH}.bak"),
    ("the same name under another root", f"other/{BASELINE_PATH}"),
    ("the directory alone", "project-evals/"),
    ("a sibling in the directory", "project-evals/README.md"),
)
r"""Criterion 8's path rows: every one of these would be caught by a substring or a
prefix test (`BASELINE_PATH in path`, `path.startswith(BASELINE_PATH)`,
`path.endswith("baseline.json")`) and none of them is the file.

Each row's commit is malformed in BOTH halves, so a guard that examines it at all
produces an offence: the exact-path test is the only thing that can keep it clean.
"""


@pytest.mark.parametrize(
    ("label", "path"), NON_BASELINE_PATHS, ids=[row[0] for row in NON_BASELINE_PATHS]
)
def test_a_path_that_is_not_exactly_the_baseline_is_not_examined(label: str, path: str) -> None:
    """Catches the substring match in criterion 8: a guard comparing loosely blocks a PR that
    renames a backup file and never touched the baseline, and - with `endswith` - one that
    edits a baseline belonging to somebody else's directory. Nothing but the exact-string
    comparison decides these rows, since the commit is otherwise a full offence."""
    verdict = check([Commit(sha=OTHER_SHA, subject="tidy the eval directory", files=(path,))])

    assert verdict.passed is True
    assert verdict.offences == ()


ADR_CANDIDATES: tuple[tuple[str, str, bool], ...] = (
    ("a real adr", "docs/adr/0009-x.md", True),
    ("the plural directory", "docs/adrs/0009-x.md", False),
    ("the bare directory name", "docs/adr", False),
    ("adr somewhere else", "other/docs/adr/0009-x.md", False),
)
"""Criterion 8's ADR rows, against `ADR_PREFIX == "docs/adr/"`.

`docs/adrs/0009-x.md` and the bare `docs/adr` both start with `docs/adr`, so a
prefix test missing the trailing slash accepts both; the last row is the one an
`in` test accepts. Every row carries a well-formed `baseline:` subject, so the
prefix guard cannot be what decides it.
"""


@pytest.mark.parametrize(
    ("label", "path", "counts"), ADR_CANDIDATES, ids=[row[0] for row in ADR_CANDIDATES]
)
def test_only_a_file_under_the_adr_directory_counts_as_the_justification(
    label: str, path: str, counts: bool
) -> None:
    """Catches an ADR test written without the trailing slash, which would accept a file in
    `docs/adrs/` that no ADR index lists, and catches one written as `in` rather than
    `startswith`, which accepts an ADR vendored under any directory. The subject is the
    published happy path's, so removing the ADR test is the only thing that flips a
    rejecting row to clean."""
    verdict = check([Commit(sha=SHAPE_SHA, subject=SHAPE_SUBJECT, files=(path, BASELINE_PATH))])

    assert verdict.passed is counts
    if not counts:
        assert verdict.offences == (Offence(sha=SHAPE_SHA, reasons=(MISSING_ADR_REASON,)),)


def test_check_is_pure_over_the_same_commits() -> None:
    """The purity claim in section 2: catches an implementation that accumulates offences in
    module state or mutates the commits it was handed, where a second run inside one
    process reports the first run's shas again - which in a CI job that checks two ranges
    blocks a PR for a commit that is not in it."""
    commits = [
        Commit(sha=SHAPE_SHA, subject=SHAPE_SUBJECT, files=(SHAPE_ADR, BASELINE_PATH)),
        Commit(sha=OTHER_SHA, subject="bump the floor", files=(BASELINE_PATH,)),
    ]

    first = check(commits)
    second = check(commits)

    assert first == second
    assert isinstance(first, GuardVerdict)


def test_an_offence_line_names_the_sha_the_file_and_every_reason() -> None:
    """The one line criterion 3 promises per offending sha. Catches a line that prints the
    reasons without the sha (a reviewer cannot find the commit), that drops the second
    reason when both are broken, and that renders the tuple's repr - `('the subject...',)`
    in a CI log is how a guard gets ignored."""
    line = offence_line(Offence(sha=OTHER_SHA, reasons=(MISSING_PREFIX_REASON, MISSING_ADR_REASON)))

    assert line == (
        f"{OTHER_SHA}: touches {BASELINE_PATH} but {MISSING_PREFIX_REASON} and {MISSING_ADR_REASON}"
    )
    assert "\n" not in line


def test_the_models_are_frozen() -> None:
    """The three records travel from the git read into the verdict and out to stderr. Catches
    a mutable `Commit`, where the check could rewrite the subject it is judging, and a
    mutable `Offence`, where a caller holding the verdict can edit what CI reports."""
    commit = Commit(sha=SHAPE_SHA, subject=SHAPE_SUBJECT, files=(BASELINE_PATH,))
    offence = Offence(sha=SHAPE_SHA, reasons=(MISSING_ADR_REASON,))
    verdict = GuardVerdict(offences=(offence,))

    with pytest.raises(ValueError):
        commit.subject = "rewritten"  # type: ignore[misc]
    with pytest.raises(ValueError):
        offence.sha = "rewritten"  # type: ignore[misc]
    with pytest.raises(ValueError):
        verdict.offences = ()  # type: ignore[misc]
