r"""`ops-eval baseline-guard`: the rule over `backend/evals/baseline.json`.

Pure, for `gate.py`'s reason one step further along: the same commits give the
same verdict on a laptop and in `.github/workflows/eval-gate.yml`, so every way
of bypassing the rule is a test row rather than a pull request somebody has to
open. The git read and the exit code live in `cli/evals.py`; what lives here is
the comparison.

The rule: a commit that touches the baseline is clean only when its subject
starts `baseline:` AND the same commit carries a file under the decision records
folder. Same commit, not same pull request - a merge and a revert can split a baseline change
from its justification across a PR
.

Both halves are checked, and a commit breaking both is ONE offence carrying two
reasons:
the same sha printed twice reads as two bad commits.

`BASELINE_PATH` is compared as an exact string and `ADR_PREFIX` carries its
trailing slash, because the loose forms of both are how a guard stops being one -
`baseline.json.bak` is not the baseline and `docs/adrs/` is not the ADR directory
.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

BASELINE_PATH: Final[str] = "backend/evals/baseline.json"
"""The one file this guard watches, matched as an EXACT path and never as a
prefix or a substring."""

BASELINE_PREFIX: Final[str] = "baseline:"
"""What the subject of a commit touching that file must start with.

The one commit subject in this repository that carries a colon: the house style
has no `type: subject` prefixes, and this rule names the literal string. Published
in `docs/eval-gate.md`."""

ADR_PREFIX: Final[str] = "dev/adr/"
"""The directory the justification must land in, trailing slash included: without
it `docs/adrs/0009-x.md` and the bare name `docs/adr` both count."""

MAX_REF_LENGTH: Final[int] = 256
"""The longest `--base` or `--head` the CLI accepts, inclusive.

A ref reaches a subprocess argument list, so it is pre-checked against a cap and a
pattern before git is asked anything - 256 in, 257 out."""

REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?!.*\.\.)[A-Za-z0-9_][A-Za-z0-9._/-]*")
r"""What a `--base` or `--head` may be, applied with `re.fullmatch`.

Three refusals matter and each is structural rather than a later check: the first
character cannot be `-`, so no value can be read by git as an option
(`-upload-pack=evil`); the lookahead refuses `..`, so no value can smuggle a
second range in; and `fullmatch` refuses a trailing newline, CRLF or space, which
`^...$` with `.match` would accept."""

MISSING_PREFIX_REASON: Final[str] = 'the subject does not start "baseline:"'
"""Half of what a reviewer is told, pinned exactly because the CLI prints it."""

MISSING_ADR_REASON: Final[str] = "the commit adds no file under dev/adr/"
"""The other half. Named separately so a commit that got the subject right and
forgot the ADR is not sent back twice."""

GUARD_PASSED_LINE: Final[str] = "baseline guard passed: {checked} commits checked"
"""The ONE line a clean range prints on stdout, rendered with `.format(checked=n)`.

`{checked}` is how many commits the range HELD, not how many touched the baseline:
a shallow checkout resolves an empty range and would otherwise report "passed"
having read nothing."""


class Commit(BaseModel):
    """One commit as the git read hands it over: `sha`, `subject`, `files`.

    Frozen, so the check cannot rewrite what it is judging.
    """

    model_config = ConfigDict(frozen=True)

    sha: str
    subject: str
    files: tuple[str, ...]
    exact_moves: tuple[tuple[str, str], ...] = ()
    """(old, new) pairs git reported as byte-identical moves (`R100`). Both sides are
    also in `files`, so moving the baseline away is still judged. Empty by default, so a caller
    that knows nothing of moves keeps working."""


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
    equal. Every commit is examined -
    there is no early return on the first clean one, which is the shape a two-commit
    PR would otherwise use to carry an undocumented change behind a documented one.
    """
    offences: list[Offence] = []
    for commit in commits:
        if BASELINE_PATH not in commit.files or _only_moves_the_baseline_in(commit):
            continue
        reasons: list[str] = []
        if not commit.subject.startswith(BASELINE_PREFIX):
            reasons.append(MISSING_PREFIX_REASON)
        if not any(path.startswith(ADR_PREFIX) for path in commit.files):
            reasons.append(MISSING_ADR_REASON)
        if reasons:
            offences.append(Offence(sha=commit.sha, reasons=tuple(reasons)))
    return GuardVerdict(offences=tuple(offences))


def _only_moves_the_baseline_in(commit: Commit) -> bool:
    """True when the commit lands a byte-identical `baseline.json` on `BASELINE_PATH`.

    The content did not change, so there is nothing for an ADR to justify. The old
    name must be a `baseline.json` too, or any file moved onto the baseline would
    replace it unjudged;
    the new name must be the baseline, or moving it away would pass.
    """
    return any(
        new == BASELINE_PATH and old.rsplit("/", 1)[-1] == "baseline.json"
        for old, new in commit.exact_moves
    )


def offence_line(offence: Offence) -> str:
    """The one line the CLI prints on stderr for one offending sha.

    The sha is first because it is what the author has to act on, and the reasons
    are joined rather than rendered as a tuple: a repr in a CI log is how a guard
    gets ignored.
    """
    return f"{offence.sha}: touches {BASELINE_PATH} but {' and '.join(offence.reasons)}"
