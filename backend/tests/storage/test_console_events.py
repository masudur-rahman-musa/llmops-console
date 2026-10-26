r"""Releases without a database: the pre-checks, the rows, the in-memory writer.

Nothing here opens a connection. Every case reaches either a module-level
checker or `InMemoryConsoleEventWriter`, which is what a gateway with no
`database_url` builds. The cases needing a real server -
the status CHECK constraint, the key and the nullable foreign key - live
in tests/storage/test_console_events_postgres.py and are skipped without
`OPSCONSOLE_TEST_DATABASE_URL`.

Every rejection row asserts BOTH directions of the contract: that
`IdentifierInvalid` was raised, and that it names the field it was raised about,
in `.param` and at the start of its message.

a project rule on the patterned field `git_sha`: its rejection table carries a
trailing newline, a trailing CRLF and a trailing space.

An earlier change removed the incident half (the incident rows, writer calls and checks);
the release tests below are unchanged.

The contract these cases are written against:

* `ReleaseRow(git_sha, deployed_at, eval_run_id, pr_url, status)`, a frozen
  Pydantic model;
* module-level `checked_release_status`, `checked_url(value, field)` and
  `checked_git_sha`, each raising `IdentifierInvalid` with `param` set to the
  field name;
* `InMemoryConsoleEventWriter` with `write_release`, `eval_run_exists` and the
  read-back snapshot `.releases`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.validation import IdentifierInvalid
from app.storage.console_events import (
    MAX_GIT_SHA_LENGTH,
    MAX_URL_LENGTH,
    MIN_GIT_SHA_LENGTH,
    RELEASE_STATUSES,
    URL_SCHEME,
    InMemoryConsoleEventWriter,
    ReleaseRow,
    checked_git_sha,
    checked_release_status,
    checked_url,
)

DEPLOYED = datetime(2026, 9, 7, 9, 15, 0, tzinfo=UTC)

RUN_ID = "0af7651916cd43dd8448eb211c80319c"
OTHER_RUN_ID = "1b2c3d4e5f60718293a4b5c6d7e8f901"

GIT_SHA = "abc1234"

# An `https://` URL of exactly MAX_URL_LENGTH characters, built from the scheme
# rather than from a literal so the two can never disagree about the length.
URL_PREFIX = "https://x.example/"
AT_CAP_URL = URL_PREFIX + "a" * (MAX_URL_LENGTH - len(URL_PREFIX))
OVER_CAP_URL = AT_CAP_URL + "a"

URL_FIELDS = ("pr_url",)


def refused(field: str, call: Callable[[], object]) -> None:
    """Assert `call` raises `IdentifierInvalid` NAMING `field`, both ways it names it.

    `.param` is what the route copies into the wire body's `param`, and the
    message prefix is what an operator reads; a checker that raised the right
    exception about the wrong field would pass an assertion on the exception
    type alone.
    """
    with pytest.raises(IdentifierInvalid) as caught:
        call()
    assert caught.value.param == field
    assert str(caught.value).startswith(field)


def a_release(**overrides: Any) -> ReleaseRow:  # noqa: ANN401
    """One well-formed release row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "git_sha": GIT_SHA,
        "deployed_at": DEPLOYED,
        "eval_run_id": None,
        "pr_url": None,
        "status": "deployed",
    }
    fields.update(overrides)
    return ReleaseRow(**fields)


# --------------------------------------------------------------------------
# the closed status vocabulary
# --------------------------------------------------------------------------

RELEASE_STATUS_ROWS: list[tuple[str, str, bool]] = [
    ("deployed", "deployed", True),
    ("rolled back", "rolled_back", True),
    ("blocked", "blocked", True),
    ("the word without its underscore", "rolledback", False),
    ("empty", "", False),
    ("uppercase", "DEPLOYED", False),
    ("trailing newline", "deployed\n", False),
    ("trailing CRLF", "deployed\r\n", False),
    ("trailing space", "deployed ", False),
]


@pytest.mark.parametrize(
    ("case", "value", "accepted"),
    RELEASE_STATUS_ROWS,
    ids=[row[0] for row in RELEASE_STATUS_ROWS],
)
def test_the_status_table(case: str, value: str, accepted: bool) -> None:
    """Catches the same gap on the releases table, plus the spelling drift the console
    reads by: `rolledback` is one character away from a legal value, so an
    implementation checking only "is it a non-empty string" accepts it and the deploys
    page then shows a status no filter it offers will ever match.
    """
    assert frozenset({"deployed", "rolled_back", "blocked"}) == RELEASE_STATUSES
    if accepted:
        checked_release_status(value)
    else:
        refused("status", lambda: checked_release_status(value))


# --------------------------------------------------------------------------
# the url length bound (a project rule: inclusive, both sides)
# --------------------------------------------------------------------------


# Every rejected URL here is UNDER the length cap, so the scheme rule is the only
# thing that can refuse it; the over-cap row is the only one the length rule
# owns. Without that split a single over-long `javascript:` value would pass the
# case whichever of the two checks was missing.
REJECTED_URLS: list[tuple[str, str]] = [
    ("plain http", "http://x.example/postmortem"),
    ("a file url", "file:///etc/passwd"),
    ("a bare host with no scheme", "x.example/a"),
    ("the scheme uppercased", "HTTPS://x.example/a"),
    ("empty", ""),
]


@pytest.mark.parametrize("field", URL_FIELDS)
@pytest.mark.parametrize(("case", "value"), REJECTED_URLS, ids=[row[0] for row in REJECTED_URLS])
def test_the_url_length_boundary_is_inclusive(field: str, case: str, value: str) -> None:
    """Catches a URL field written through unchecked, and catches the check being
    hard-coded to a field other than `pr_url`: the refusal must name `pr_url` in `.param`,
    so a checker that reports another field fails on the release path where the console
    shows the wrong input as the one at fault. `file:///etc/passwd` and the bare host are
    the rows that matter for the scheme rule - the console renders these as links, so a
    non-`https://` value is a link an operator clicks.
    """
    assert MAX_URL_LENGTH == 500
    assert URL_SCHEME == "https://"

    checked_url(AT_CAP_URL, field)
    assert len(AT_CAP_URL) == MAX_URL_LENGTH

    refused(field, lambda: checked_url(OVER_CAP_URL, field))
    refused(field, lambda: checked_url(value, field))


# --------------------------------------------------------------------------
# the patterned git sha
# --------------------------------------------------------------------------

REJECTED_GIT_SHAS: list[tuple[str, str]] = [
    ("one short of the minimum", "a" * (MIN_GIT_SHA_LENGTH - 1)),
    ("one past the maximum", "a" * (MAX_GIT_SHA_LENGTH + 1)),
    ("empty", ""),
    ("uppercase hex", "A" * MIN_GIT_SHA_LENGTH),
    ("a non-hex letter", "abcdefg"),
    ("trailing newline", "a" * MIN_GIT_SHA_LENGTH + "\n"),
    ("trailing CRLF", "a" * MIN_GIT_SHA_LENGTH + "\r\n"),
    ("trailing space", "a" * MIN_GIT_SHA_LENGTH + " "),
]


@pytest.mark.parametrize(
    ("case", "value"), REJECTED_GIT_SHAS, ids=[row[0] for row in REJECTED_GIT_SHAS]
)
def test_the_git_sha_boundaries_are_inclusive(case: str, value: str) -> None:
    r"""Catches `git_sha` validated with `re.match` and a trailing `$` rather than
    `re.fullmatch`: in Python `$` also matches before a trailing
    newline, so `"a" * 7 + "\n"` passes that spelling and lands a line break in a column
    the releases page renders and CI posts from a shell. The short and long rows are the
    other guard - an abbreviated sha of six characters is ambiguous in a real repository
    and a 41-character value is not a sha at all.
    """
    assert MIN_GIT_SHA_LENGTH == 7
    assert MAX_GIT_SHA_LENGTH == 40
    checked_git_sha("a" * MIN_GIT_SHA_LENGTH)
    checked_git_sha("a" * MAX_GIT_SHA_LENGTH)

    refused("git_sha", lambda: checked_git_sha(value))


# --------------------------------------------------------------------------
# the in-memory writer (decision C)
# --------------------------------------------------------------------------


def test_the_in_memory_writer_files_and_reads_back_a_release() -> None:
    """Catches a writer that accepts a release and drops it - `write_release` returns
    nothing, so an empty body would raise nothing either and every route case in
    tests/api/test_admin.py would still answer 201.
    """
    writer = InMemoryConsoleEventWriter()
    release = a_release()

    writer.write_release(release)

    assert writer.releases == (release,)


def test_the_in_memory_writer_answers_that_every_eval_run_exists() -> None:
    """Catches `eval_run_exists` left off the in-memory writer or answering False there:
    the route runs this pre-check before the insert on every writer, so a False here
    would turn every `POST /admin/releases` naming a run into a 422 on a laptop with no
    `database_url` - the one configuration decision C exists to keep answering. The
    writer has no `eval_runs` table to consult, so True for a well-formed id IS the
    documented behaviour, and this case is what makes that a decision rather than an
    accident.
    """
    writer = InMemoryConsoleEventWriter()

    assert writer.eval_run_exists(RUN_ID) is True
    assert writer.eval_run_exists(OTHER_RUN_ID) is True
