r"""Incidents and releases without a database: the pre-checks, the rows, the in-memory writer.

Nothing here opens a connection. Every case reaches either a module-level
checker or `InMemoryConsoleEventWriter`, which is what a gateway with no
`database_url` builds. The cases needing a real server -
the two CHECK constraints, the composite key and the nullable foreign key - live
in tests/ledger/test_console_events_postgres.py and are skipped without
`OPSCONSOLE_TEST_DATABASE_URL`.

Every rejection row asserts BOTH directions of the contract: that
`IdentifierInvalid` was raised, and that it names the field it was raised about,
in `.param` and at the start of its message. Asserting only the exception would
pass against an implementation that refused every value for the wrong reason, or
that named `title` when `summary` was the field at fault - and the route turns
`.param` straight into the `param` a 422 body carries, so a wrong name there is
a wrong answer on the wire.

a project rule on the two patterned fields, `git_sha` and the incident `id`:
each rejection table carries a trailing newline, a trailing CRLF and a trailing
space. The `id` rows are built so the WHOLE value is still exactly
`INCIDENT_ID_LENGTH` characters long (31 hex plus a newline, 30 hex plus a
CRLF), because a 33-character value would be refused by the length rule and the
case would then pass without the pattern being involved at all.

a project rule on the seven bounds: the longest legal value is asserted
ACCEPTED beside every "one more is refused" row, so an implementation that
refuses everything - an off-by-one, or a checker that raises unconditionally -
cannot make the rejection half pass on its own.

The contract these cases are written against, before it exists:

* `IncidentRow(id, started_at, resolved_at, severity, title, summary,
  postmortem_url)` and `ReleaseRow(git_sha, deployed_at, eval_run_id, pr_url,
  status)`, both frozen Pydantic models;
* module-level `checked_severity`, `checked_release_status`, `checked_title`,
  `checked_summary`, `checked_url(value, field)`, `checked_git_sha` and
  `checked_incident_id`, each raising `IdentifierInvalid` with `param` set to
  the field name;
* `InMemoryConsoleEventWriter` with `write_incident`, `resolve_incident`,
  `write_release`, `eval_run_exists`, and the read-back snapshots `.incidents`
  and `.releases`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from opsconsole.ledger.console_events import (
    INCIDENT_ID_LENGTH,
    MAX_GIT_SHA_LENGTH,
    MAX_SUMMARY_LENGTH,
    MAX_TITLE_LENGTH,
    MAX_URL_LENGTH,
    MIN_GIT_SHA_LENGTH,
    RELEASE_STATUSES,
    SEVERITIES,
    URL_SCHEME,
    IncidentRow,
    InMemoryConsoleEventWriter,
    ReleaseRow,
    checked_git_sha,
    checked_incident_id,
    checked_release_status,
    checked_severity,
    checked_summary,
    checked_title,
    checked_url,
)
from opsconsole.routing.errors import IdentifierInvalid

STARTED = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
RESOLVED = datetime(2026, 9, 8, 1, 30, 0, tzinfo=UTC)
DEPLOYED = datetime(2026, 9, 7, 9, 15, 0, tzinfo=UTC)

INCIDENT_ID = "0af7651916cd43dd8448eb211c80319c"
"""32 lowercase hex, the shape `uuid4().hex` produces."""

OTHER_INCIDENT_ID = "1b2c3d4e5f60718293a4b5c6d7e8f901"

GIT_SHA = "abc1234"

# An `https://` URL of exactly MAX_URL_LENGTH characters, built from the scheme
# rather than from a literal so the two can never disagree about the length.
URL_PREFIX = "https://x.example/"
AT_CAP_URL = URL_PREFIX + "a" * (MAX_URL_LENGTH - len(URL_PREFIX))
OVER_CAP_URL = AT_CAP_URL + "a"

URL_FIELDS = ("postmortem_url", "pr_url")


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


def an_incident(**overrides: Any) -> IncidentRow:  # noqa: ANN401
    """One well-formed incident row, with the named fields replaced."""
    fields: dict[str, Any] = {
        "id": INCIDENT_ID,
        "started_at": STARTED,
        "resolved_at": None,
        "severity": "sev2",
        "title": "gateway 5xx on the eu pool",
        "summary": "the eu pool answered 503 for eleven minutes",
        "postmortem_url": None,
    }
    fields.update(overrides)
    return IncidentRow(**fields)


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
# the two closed vocabularies
# --------------------------------------------------------------------------

# (case, value, accepted). The three accepted rows are what keeps a checker that
# raises unconditionally from passing the rejection rows.
SEVERITY_ROWS: list[tuple[str, str, bool]] = [
    ("sev1", "sev1", True),
    ("sev2", "sev2", True),
    ("sev3", "sev3", True),
    ("a fourth word", "sev4", False),
    ("uppercase", "SEV1", False),
    ("empty", "", False),
    ("trailing newline", "sev1\n", False),
    ("trailing CRLF", "sev1\r\n", False),
    ("trailing space", "sev1 ", False),
]


@pytest.mark.parametrize(
    ("case", "value", "accepted"), SEVERITY_ROWS, ids=[row[0] for row in SEVERITY_ROWS]
)
def test_the_severity_table(case: str, value: str, accepted: bool) -> None:
    """Catches an implementation that stores whatever severity a caller sent and leaves
    the CHECK constraint to refuse it - which is a 500 from an integrity error rather
    than the 422 VALIDATION naming `severity` the contract fixes, and is no refusal at
    all on the in-memory writer, where there is no constraint behind it. `SEV1` is the
    sharp row: a checker that lowercased before comparing would widen the closed
    vocabulary the CHECK spells in lowercase only.
    """
    assert frozenset({"sev1", "sev2", "sev3"}) == SEVERITIES
    if accepted:
        checked_severity(value)
    else:
        refused("severity", lambda: checked_severity(value))


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
# the five length bounds (a project rule: inclusive, both sides)
# --------------------------------------------------------------------------


def test_the_title_length_boundary_is_inclusive() -> None:
    """Catches an off-by-one at the cap in either direction: a checker refusing at
    exactly 200 loses a legal title, and one refusing only above 201 lets an unbounded
    `text` column take whatever an admin paste carries. The empty row is the other end -
    a title is 1 to 200 characters, so "" is a refusal and not a default.
    """
    assert MAX_TITLE_LENGTH == 200
    checked_title("a" * MAX_TITLE_LENGTH)

    refused("title", lambda: checked_title("a" * (MAX_TITLE_LENGTH + 1)))
    refused("title", lambda: checked_title(""))


def test_the_summary_length_boundary_is_inclusive() -> None:
    """Catches the cap being copied from `title` onto `summary`, which is the failure a
    single shared checker produces: a 4000-character summary is legal and would be
    refused, and the 4001 row is what proves a cap exists at all rather than the field
    being written through unchecked.
    """
    assert MAX_SUMMARY_LENGTH == 4000
    checked_summary("a" * MAX_SUMMARY_LENGTH)

    refused("summary", lambda: checked_summary("a" * (MAX_SUMMARY_LENGTH + 1)))
    refused("summary", lambda: checked_summary(""))


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
    hard-coded to one field name: the same rejection is run for `postmortem_url` and for
    `pr_url` and each must be named in `.param`, so a checker that always reports
    `postmortem_url` fails on the release path where the console shows the wrong input
    as the one at fault. `file:///etc/passwd` and the bare host are the rows that matter
    for the scheme rule - the console renders these as links, so a non-`https://` value
    is a link an operator clicks.
    """
    assert MAX_URL_LENGTH == 500
    assert URL_SCHEME == "https://"

    checked_url(AT_CAP_URL, field)
    assert len(AT_CAP_URL) == MAX_URL_LENGTH

    refused(field, lambda: checked_url(OVER_CAP_URL, field))
    refused(field, lambda: checked_url(value, field))


# --------------------------------------------------------------------------
# the two patterned identifiers
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


# Each of these is exactly INCIDENT_ID_LENGTH characters long except the two rows
# that name a length, so the length rule cannot be what refuses them - only the
# pattern can.
REJECTED_INCIDENT_IDS: list[tuple[str, str]] = [
    ("one hex short", "0" * (INCIDENT_ID_LENGTH - 1)),
    ("one hex long", "0" * (INCIDENT_ID_LENGTH + 1)),
    ("empty", ""),
    ("uppercase hex", "A" * INCIDENT_ID_LENGTH),
    ("a non-hex letter at full length", "z" + "0" * (INCIDENT_ID_LENGTH - 1)),
    ("trailing newline at full length", "0" * (INCIDENT_ID_LENGTH - 1) + "\n"),
    ("trailing CRLF at full length", "0" * (INCIDENT_ID_LENGTH - 2) + "\r\n"),
    ("trailing space at full length", "0" * (INCIDENT_ID_LENGTH - 1) + " "),
]


@pytest.mark.parametrize(
    ("case", "value"), REJECTED_INCIDENT_IDS, ids=[row[0] for row in REJECTED_INCIDENT_IDS]
)
def test_the_incident_id_is_exactly_thirty_two_hex(case: str, value: str) -> None:
    r"""Catches the PATCH path parameter reaching a lookup unchecked. Decision A makes
    the id server-generated, so the path parameter is the ONLY place a client string
    shapes this value and the only place it can be checked; the three whitespace rows
    are each exactly 32 characters, so a length check alone accepts them and only
    `re.fullmatch` against 32 hex refuses - the difference between a refused patch and
    an id carrying a line break in a lookup.
    """
    assert INCIDENT_ID_LENGTH == 32
    checked_incident_id("0" * INCIDENT_ID_LENGTH)
    checked_incident_id(INCIDENT_ID)

    refused("id", lambda: checked_incident_id(value))


# --------------------------------------------------------------------------
# the in-memory writer (decision C)
# --------------------------------------------------------------------------


def test_the_in_memory_writer_files_and_reads_back_both_rows() -> None:
    """Catches a writer that accepts a row and drops it - `write_incident` returns
    nothing, so an empty body would raise nothing either and every route case in
    tests/gateway/test_admin_incidents.py would still answer 201. The two tables are
    asserted apart: an implementation keeping one list for both would put the release in
    `.incidents` and the timeline would render a deploy as an outage.
    """
    writer = InMemoryConsoleEventWriter()
    incident = an_incident()
    release = a_release()

    writer.write_incident(incident)
    writer.write_release(release)

    assert writer.incidents == (incident,)
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

    assert writer.eval_run_exists(INCIDENT_ID) is True
    assert writer.eval_run_exists(OTHER_INCIDENT_ID) is True


def test_resolving_an_unknown_id_and_an_already_resolved_incident() -> None:
    """Catches `resolve_incident` implemented as an idempotent no-op - an UPDATE that
    matched nothing and returned quietly. Both failures then answer 200, so a mistyped
    id silently resolves nothing and a second PATCH silently overwrites the first
    resolution time. The two messages are asserted DISTINCT because they are the only
    thing telling the two failures apart once both became 422 VALIDATION naming `id`
    (deviation 1): identical text would make "no such incident" and "already resolved"
    indistinguishable on the wire.
    """
    writer = InMemoryConsoleEventWriter()
    writer.write_incident(an_incident())

    with pytest.raises(IdentifierInvalid) as unknown:
        writer.resolve_incident(OTHER_INCIDENT_ID, RESOLVED)

    # The stored row is frozen, so resolving REPLACES it rather than mutating it.
    writer.resolve_incident(INCIDENT_ID, RESOLVED)
    assert [row.resolved_at for row in writer.incidents] == [RESOLVED]
    assert [row.id for row in writer.incidents] == [INCIDENT_ID]

    with pytest.raises(IdentifierInvalid) as already:
        writer.resolve_incident(INCIDENT_ID, RESOLVED)

    assert unknown.value.param == "id"
    assert already.value.param == "id"
    assert str(unknown.value).startswith("id")
    assert str(already.value).startswith("id")
    assert str(unknown.value) != str(already.value)
