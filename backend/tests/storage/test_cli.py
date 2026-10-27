"""`make migrate`: the entry point.

Skipped without `OPSCONSOLE_TEST_DATABASE_URL`, except
the cases about an UNCONFIGURED database, which are the whole point of the
non-zero exit code and need no server at all. Those set
`OPSCONSOLE_DATABASE_URL` through `monkeypatch` where they need it: the root
tests/conftest.py clears it for the session, so a developer with a local stack up
runs the same suite as one without.

The success case runs against the DEFAULT schema of the test database rather
than a throwaway one, because the entry points take their configuration from
`Settings` and nothing on the command line points them elsewhere. It is
idempotent, which is what makes that safe to repeat.
"""

from __future__ import annotations

import pytest

from app.cli.ledger import migrate

from .conftest import DATABASE_URL, requires_postgres

DATABASE_URL_ENV = "OPSCONSOLE_DATABASE_URL"
"""What the entry points read through `Settings`. Cleared by tests/conftest.py."""


def test_migrate_refuses_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches an entry point that reports success having done nothing, which in a deploy
    pipeline is the worst possible answer: the next step starts the gateway against a
    database with no table in it. Catches a bare traceback too - the exit code is the
    contract, and a message an operator can read is the rest of it."""
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    code = migrate([])

    assert code != 0
    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""


@requires_postgres
def test_migrate_applies_the_migrations_and_answers_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches the entry point never reaching the runner - a `Makefile` target that exits
    0 having imported a module - and an entry point that raises instead of answering a
    code, which is the difference between a failed deploy step and a stack trace. Run
    twice, because `make migrate` on an already-migrated database is the ordinary case and
    it must also answer 0."""
    assert DATABASE_URL is not None
    monkeypatch.setenv(DATABASE_URL_ENV, DATABASE_URL)

    assert migrate([]) == 0
    assert migrate([]) == 0

    printed = capsys.readouterr()
    assert (printed.out + printed.err).strip() != ""
