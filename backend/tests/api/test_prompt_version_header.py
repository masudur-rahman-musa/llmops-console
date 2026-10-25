"""`X-Prompt-Version`: an optional header the gateway books in the cost ledger row.

No backing service. The shared `chat_client` app answers from a stub adapter, and each
case puts a fresh in-memory ledger on the app, then puts the old one back, so the rows
read are exactly the ones this case's request booked (a project rule: never a new app).

The header is a lowercase slug of at most `MAX_IDENTIFIER_LENGTH` (64) characters, so a
SHA-256 hex digest fits exactly. It is checked the same way as `X-Session`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.storage import CostLedgerRow, InMemoryCostLedgerWriter

from .conftest import CHAT, TEST_KEYS, app_of, assert_error_body, auth_headers, chat_body

HEADER = "X-Prompt-Version"
SHA256_HEX = hashlib.sha256(b"prompt-v1").hexdigest()


@pytest.fixture
def ledger(chat_client: TestClient) -> Iterator[InMemoryCostLedgerWriter]:
    """A fresh in-memory ledger on the shared chat app for one case, then the old one back."""
    app = app_of(chat_client)
    previous = app.state.cost_ledger
    writer = InMemoryCostLedgerWriter()
    app.state.cost_ledger = writer
    try:
        yield writer
    finally:
        app.state.cost_ledger = previous


def post(client: TestClient, prompt_version: str | None) -> Response:
    """One chat as `evals` under `internal`, with the header only when a value is given."""
    headers = auth_headers(key=TEST_KEYS["evals"], tenant="evals", data_class="internal")
    if prompt_version is not None:
        headers[HEADER] = prompt_version
    response: Response = client.post(CHAT, json=chat_body(), headers=headers)
    return response


def only_row(writer: InMemoryCostLedgerWriter) -> CostLedgerRow:
    """The one row the request booked."""
    assert len(writer.rows) == 1
    return writer.rows[0]


def test_prompt_version_reaches_the_ledger_row(
    chat_client: TestClient, ledger: InMemoryCostLedgerWriter
) -> None:
    """Catches an implementation that reads the header and drops it before `record_cost`
    (the row stays None), or one that books a truncated or rewritten value: the value is a
    full 64-character digest, the inclusive limit, and must come back unchanged."""
    assert len(SHA256_HEX) == MAX_IDENTIFIER_LENGTH

    response = post(chat_client, SHA256_HEX)

    assert response.status_code == 200, response.text
    assert only_row(ledger).prompt_version == SHA256_HEX


def test_no_header_books_none(chat_client: TestClient, ledger: InMemoryCostLedgerWriter) -> None:
    """Catches an implementation that books an empty string, "none" or a default version
    when the header is absent, which the console would then show as a real prompt version."""
    response = post(chat_client, None)

    assert response.status_code == 200, response.text
    assert only_row(ledger).prompt_version is None


def test_65_characters_are_refused(
    chat_client: TestClient, ledger: InMemoryCostLedgerWriter
) -> None:
    """Catches a check with no length cap: 65 lowercase hex characters are a valid slug in
    every other way, so only the cap refuses them. The refusal must name the header and
    book nothing."""
    over = "a" * (MAX_IDENTIFIER_LENGTH + 1)

    response = post(chat_client, over)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == HEADER
    assert HEADER in response.json()["error"]["message"]
    assert ledger.rows == ()


REFUSED: list[tuple[str, str]] = [
    ("uppercase", "ABC"),
    ("mixed case digest", "9F2c4be0"),
    ("trailing newline", "prompt-v1\n"),
    ("trailing CRLF", "prompt-v1\r\n"),
    ("trailing space", "prompt-v1 "),
    ("a slash", "prompt/v1"),
    ("a dot", "prompt.v1"),
]


@pytest.mark.parametrize(("case", "value"), REFUSED, ids=[row[0] for row in REFUSED])
def test_a_value_that_is_not_a_lowercase_slug_is_refused(
    case: str, value: str, chat_client: TestClient, ledger: InMemoryCostLedgerWriter
) -> None:
    """Catches a check that lowercases instead of refusing (`ABC` would book as `abc`), a
    `.match` with `$` that lets a trailing newline through, or a pattern that is looser than
    the slug rule: each of these would otherwise book a row with the odd value."""
    response = post(chat_client, value)

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == HEADER
    assert ledger.rows == ()
