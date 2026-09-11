r"""`ops-eval baseline-guard`: #42's rule over `project-evals/baseline.json`.

Pure, for `gate.py`'s reason one step further along: the same commits give the
same verdict on a laptop and in `.github/workflows/eval-gate.yml`, so every way
of bypassing the rule is a test row rather than a pull request somebody has to
open. The git read and the exit code live in `evals/cli.py`; what lives here is
the comparison.

The rule, from the design and #42: a commit that touches the baseline is clean
only when its subject starts `baseline:` AND the same commit carries a file under
`docs/adr/`. Same commit, not same pull request - a merge
and a revert can split a baseline change from its justification across a PR, and
`a87abf2` is the shape this repo already has
(`test_the_shape_commit_a87abf2_already_has_passes`).

Both halves are checked, and a commit breaking both is ONE offence carrying two
reasons (`test_a_commit_breaking_both_halves_is_one_offence_carrying_both_reasons`):
the same sha printed twice reads as two bad commits.

`BASELINE_PATH` is compared as an exact string and `ADR_PREFIX` carries its
trailing slash, because the loose forms of both are how a guard stops being one -
`baseline.json.bak` is not the baseline and `docs/adrs/` is not the ADR directory
(`test_a_path_that_is_not_exactly_the_baseline_is_not_examined`,
`test_only_a_file_under_the_adr_directory_counts_as_the_justification`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

BASELINE_PATH: Final[str] = "project-evals/baseline.json"
"""The one file this guard watches, matched as an EXACT path and never as a
prefix or a substring (`test_the_four_constants_are_exactly_the_published_values`)."""

BASELINE_PREFIX: Final[str] = "baseline:"
"""What the subject of a commit touching that file must start with.

The one commit subject in this repository that carries a colon: the house style
has no `type: subject` prefixes, and this rule names the literal string. Published
in `docs/eval-gate.md` since an earlier change."""

ADR_PREFIX: Final[str] = "docs/adr/"
"""The directory the justification must land in, trailing slash included: without
it `docs/adrs/0009-x.md` and the bare name `docs/adr` both count."""

MAX_REF_LENGTH: Final[int] = 256
"""The longest `--base` or `--head` the CLI accepts, inclusive.

A ref reaches a subprocess argument list, so it is pre-checked against a cap and a
pattern before git is asked anything - 256 in, 257 out
(`test_a_ref_of_exactly_the_cap_gets_past_the_pre_check`,
`test_a_ref_pre_check_refusal_is_one_line_and_exit_two`)."""

REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?!.*\.\.)[A-Za-z0-9_][A-Za-z0-9._/-]*")
r"""What a `--base` or `--head` may be, applied with `re.fullmatch`.

Three refusals matter and each is structural rather than a later check: the first
character cannot be `-`, so no value can be read by git as an option
(`-upload-pack=evil`); the lookahead refuses `..`, so no value can smuggle a
second range in; and `fullmatch` refuses a trailing newline, CRLF or space, which
`^...$` with `.match` would accept
(`test_the_ref_pattern_matches_whole_values_only`)."""

MISSING_PREFIX_REASON: Final[str] = 'the subject does not start "baseline:"'
"""Half of what a reviewer is told, pinned exactly because the CLI prints it
(`test_the_three_published_strings_are_exactly_what_the_cli_prints`)."""

MISSING_ADR_REASON: Final[str] = "the commit adds no file under docs/adr/"
"""The other half. Named separately so a commit that got the subject right and
forgot the ADR is not sent back twice."""

GUARD_PASSED_LINE: Final[str] = "baseline guard passed: {checked} commits checked"
"""The ONE line a clean range prints on stdout, rendered with `.format(checked=n)`.

`{checked}` is how many commits the range HELD, not how many touched the baseline:
a shallow checkout resolves an empty range and would otherwise report "passed"
having read nothing, which is risk 2
(`test_a_clean_range_exits_zero_with_one_line_on_stdout`)."""


class Commit(BaseModel):
    """One commit as the git read hands it over: `sha`, `subject`, `files`.

    Frozen, so the check cannot rewrite what it is judging
    (`test_the_models_are_frozen`).
    """

    model_config = ConfigDict(frozen=True)

    sha: str
    subject: str
    files: tuple[str, ...]


class Offence(BaseModel):
    """One commit that touched the baseline and broke the rule, with every reason.

    One record per SHA, never one per broken half: `reasons` carries both when both
    are broken, in the fixed order the prefix is checked before the ADR.
    """

    model_config = ConfigDict(frozen=True)

    sha: str
    reasons: tuple[str, ...]


class GuardVerdict(BaseModel):
    """What `check` answers: the offences, in commit order, and nothing else."""

    model_config = ConfigDict(frozen=True)

    offences: tuple[Offence, ...]

    @property
    def passed(self) -> bool:
        """No offence at all. A range touching no baseline file passes here."""
        return not self.offences


def check(commits: Sequence[Commit]) -> GuardVerdict:
    """Answer the verdict for one range of commits.

    Pure: no git, no clock, no module state, so two calls over the same commits are
    equal (`test_check_is_pure_over_the_same_commits`). Every commit is examined -
    there is no early return on the first clean one, which is the shape a two-commit
    PR would otherwise use to carry an undocumented change behind a documented one
    (`test_only_the_second_of_two_baseline_commits_being_malformed_still_fails`).
    """
    offences: list[Offence] = []
    for commit in commits:
        if BASELINE_PATH not in commit.files:
            continue
        reasons: list[str] = []
        if not commit.subject.startswith(BASELINE_PREFIX):
            reasons.append(MISSING_PREFIX_REASON)
        if not any(path.startswith(ADR_PREFIX) for path in commit.files):
            reasons.append(MISSING_ADR_REASON)
        if reasons:
            offences.append(Offence(sha=commit.sha, reasons=tuple(reasons)))
    return GuardVerdict(offences=tuple(offences))


def offence_line(offence: Offence) -> str:
    """The one line the CLI prints on stderr for one offending sha.

    The sha is first because it is what the author has to act on, and the reasons
    are joined rather than rendered as a tuple: a repr in a CI log is how a guard
    gets ignored (`test_an_offence_line_names_the_sha_the_file_and_every_reason`).
    """
    return f"{offence.sha}: touches {BASELINE_PATH} but {' and '.join(offence.reasons)}"
