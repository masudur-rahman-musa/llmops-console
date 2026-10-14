"""Per-tenant API keys: the store, the digest comparison and the two failures.

Every value this file pins is a value the plan fixes: 401 for a
missing or unknown key, 403 for a key whose tenant does not match `X-Tenant`, and
`MAX_KEY_LENGTH` for the bearer pre-check, asserted against the imported constant
rather than a retyped number.
"""

from __future__ import annotations

import logging
import re

import pytest
from fastapi.testclient import TestClient

from app.gateway.auth import MAX_KEY_LENGTH, SettingsKeyStore

from .conftest import (
    API_KEYS,
    AT_CAP_KEY,
    OVER_CAP_KEY,
    TEST_KEYS,
    UNKNOWN_KEY,
    assert_error_body,
    auth_headers,
    sha256_hex,
)

NORDLICHT_KEY = TEST_KEYS["nordlicht"]
CUAN_KEY = TEST_KEYS["cuan"]
EVALS_KEY = TEST_KEYS["evals"]

# Both routes under test go through the same dependency; the harness one is used
# where the response body has to show what the request became.
MODELS = "/v1/models"
CONTEXT = "/_harness/context"


# --- 401: no usable key ------------------------------------------------------

# Decision table. A key that cannot be read is the same answer as a key that can
# be read and is not in the store: 401, with no hint about which it was.
MISSING_KEY_CASES: list[tuple[str, str | None]] = [
    ("no Authorization header at all", None),
    ("empty Authorization header", ""),
    ("a scheme that is not Bearer", "Basic dGVzdDp0ZXN0"),
    ("Bearer with no value", "Bearer"),
    ("Bearer with an empty value", "Bearer "),
]


@pytest.mark.parametrize(
    ("case", "authorization"), MISSING_KEY_CASES, ids=[c[0] for c in MISSING_KEY_CASES]
)
def test_missing_key_is_unauthorized(
    case: str, authorization: str | None, gateway_client: TestClient
) -> None:
    response = gateway_client.get(
        MODELS,
        headers=auth_headers(key=None, tenant="nordlicht", authorization=authorization),
    )

    assert_error_body(response, status=401, code="UNAUTHORIZED")


# A well-formed key the store does not hold. The whitespace rows are standing
# rule 4 applied to the bearer value: a key with a trailing newline, CRLF or
# space is a different string and must not authenticate as the key it wraps.
UNKNOWN_KEY_CASES: list[tuple[str, str]] = [
    ("a key belonging to nobody", UNKNOWN_KEY),
    ("a sha256 digest presented as the key", sha256_hex(NORDLICHT_KEY)),
    ("a valid key with a trailing newline", NORDLICHT_KEY + "\n"),
    ("a valid key with a trailing CRLF", NORDLICHT_KEY + "\r\n"),
    ("a valid key with a trailing space", NORDLICHT_KEY + " "),
    ("a key one character over the cap", OVER_CAP_KEY),
]


@pytest.mark.parametrize(("case", "key"), UNKNOWN_KEY_CASES, ids=[c[0] for c in UNKNOWN_KEY_CASES])
def test_unknown_key_is_unauthorized(case: str, key: str, gateway_client: TestClient) -> None:
    """Anything that is not exactly a stored key is 401, the over-cap row included.

    The over-cap key IS registered in the store (see `conftest.TEST_KEYS`), so it
    is rejected by the length pre-check and not by being unknown; the answer is
    the same 401 either way, because the reason a key failed is never disclosed.
    """
    response = gateway_client.get(MODELS, headers=auth_headers(key=key, tenant="nordlicht"))

    assert_error_body(response, status=401, code="UNAUTHORIZED")


def test_max_key_length_is_pinned() -> None:
    """The value itself, not just the boundary behaviour.

    Every other assertion in this file is relative to the imported constant, so
    without this one a cap trimmed below the length of an issued key would turn
    every tenant key into a silent 401 with nothing going red.
    """
    assert MAX_KEY_LENGTH == 256


@pytest.mark.parametrize(
    ("case", "scheme"),
    [("canonical", "Bearer"), ("lowercase", "bearer"), ("shouted", "BEARER")],
    ids=["canonical", "lowercase", "shouted"],
)
def test_bearer_scheme_is_case_insensitive(
    case: str, scheme: str, gateway_client: TestClient
) -> None:
    """RFC 7235 makes the scheme token case-insensitive; the key after it is not."""
    response = gateway_client.get(
        MODELS,
        headers=auth_headers(
            key=None, tenant="nordlicht", authorization=f"{scheme} {NORDLICHT_KEY}"
        ),
    )

    assert response.status_code == 200


def test_bearer_key_length_boundary_is_inclusive(harness_client: TestClient) -> None:
    """`MAX_KEY_LENGTH` characters is accepted; one more is not."""
    assert len(AT_CAP_KEY) == MAX_KEY_LENGTH
    assert len(OVER_CAP_KEY) == MAX_KEY_LENGTH + 1

    accepted = harness_client.get(CONTEXT, headers=auth_headers(key=AT_CAP_KEY, tenant="cap-key"))
    rejected = harness_client.get(
        CONTEXT, headers=auth_headers(key=OVER_CAP_KEY, tenant="over-cap-key")
    )

    assert accepted.status_code == 200
    assert accepted.json()["tenant"] == "cap-key"
    assert_error_body(rejected, status=401, code="UNAUTHORIZED")


# --- 403: the key is valid, the tenant header is not -------------------------

MISMATCH_CASES: list[tuple[str, str, str]] = [
    ("cuan's key claiming nordlicht", CUAN_KEY, "nordlicht"),
    ("nordlicht's key claiming cuan", NORDLICHT_KEY, "cuan"),
    ("evals' key claiming a tenant that does not exist", EVALS_KEY, "no-such-tenant"),
]


@pytest.mark.parametrize(
    ("case", "key", "tenant"), MISMATCH_CASES, ids=[c[0] for c in MISMATCH_CASES]
)
def test_key_tenant_mismatch_is_403(
    case: str, key: str, tenant: str, gateway_client: TestClient
) -> None:
    response = gateway_client.get(MODELS, headers=auth_headers(key=key, tenant=tenant))

    assert_error_body(response, status=403, code="TENANT_MISMATCH")


def test_matching_key_and_tenant_is_accepted(gateway_client: TestClient) -> None:
    """The positive row of the same table: the mismatch test is not vacuous."""
    response = gateway_client.get(MODELS, headers=auth_headers(key=CUAN_KEY, tenant="cuan"))

    assert response.status_code == 200


# --- the plaintext key never escapes -----------------------------------------


@pytest.mark.parametrize(
    ("case", "key", "tenant"),
    [
        ("an accepted request", NORDLICHT_KEY, "nordlicht"),
        ("a rejected request", UNKNOWN_KEY, "nordlicht"),
        ("a mismatched request", CUAN_KEY, "nordlicht"),
    ],
    ids=["accepted", "rejected", "mismatched"],
)
def test_key_never_appears_in_body_or_log(
    case: str,
    key: str,
    tenant: str,
    gateway_client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No response body and no log record carries the presented key.

    DEBUG for the whole request, so a debug-level "authenticating <key>" is
    caught. Both the rendered message and the raw `args` are checked: `%s`
    formatting hides the key from `record.message` until something renders it.
    """
    with caplog.at_level(logging.DEBUG):
        response = gateway_client.get(MODELS, headers=auth_headers(key=key, tenant=tenant))

    assert key not in response.text
    # Zero captured records is a pass: nothing was logged, so nothing leaked.
    for record in caplog.records:
        assert key not in record.getMessage()
        assert key not in str(record.msg)
        assert key not in str(record.args)


# --- the store itself --------------------------------------------------------

VALID_DIGEST = sha256_hex("some-test-key")

# Decision table for construction. Each malformed row names the 1-based position
# of the entry that broke it, so an operator fixes the right pair.
MALFORMED_ENTRY_CASES: list[tuple[str, str, int]] = [
    ("no colon", "nordlicht", 1),
    ("empty entry", ",", 1),
    ("empty tenant", f":{VALID_DIGEST}", 1),
    ("tenant that is not a lowercase slug", f"Nordlicht:{VALID_DIGEST}", 1),
    ("tenant with a trailing newline", f"nordlicht\n:{VALID_DIGEST}", 1),
    ("tenant with a trailing space", f"nordlicht :{VALID_DIGEST}", 1),
    ("digest that is not hex", f"nordlicht:{'z' * 64}", 1),
    ("digest one character short", f"nordlicht:{VALID_DIGEST[:63]}", 1),
    ("digest one character long", f"nordlicht:{VALID_DIGEST}a", 1),
    ("empty digest", "nordlicht:", 1),
    ("second entry empty", f"nordlicht:{VALID_DIGEST},", 2),
    ("second entry has no colon", f"nordlicht:{VALID_DIGEST},cuan", 2),
    ("third entry malformed", f"nordlicht:{VALID_DIGEST},cuan:{VALID_DIGEST},evals", 3),
]


@pytest.mark.parametrize(
    ("case", "entries", "position"),
    MALFORMED_ENTRY_CASES,
    ids=[c[0] for c in MALFORMED_ENTRY_CASES],
)
def test_key_store_rejects_malformed_entry_at_construction(
    case: str, entries: str, position: int
) -> None:
    """A bad entry fails when the store is built, never at request time."""
    with pytest.raises(ValueError) as excinfo:
        SettingsKeyStore(entries)

    message = str(excinfo.value)
    assert re.search(rf"\b{position}\b", message), message


def test_key_store_looks_up_by_digest() -> None:
    """`tenant_for` maps a plaintext key to its tenant, and anything else to None."""
    store = SettingsKeyStore(API_KEYS)

    assert store.tenant_for(NORDLICHT_KEY) == "nordlicht"
    assert store.tenant_for(CUAN_KEY) == "cuan"
    assert store.tenant_for(UNKNOWN_KEY) is None
    assert store.tenant_for(sha256_hex(NORDLICHT_KEY)) is None
    assert store.tenant_for("") is None


@pytest.mark.parametrize(
    ("case", "suffix"),
    [
        ("trailing newline", "\n"),
        ("trailing CRLF", "\r\n"),
        ("trailing space", " "),
    ],
    ids=["newline", "crlf", "space"],
)
def test_key_store_rejects_whitespace_padded_keys(case: str, suffix: str) -> None:
    """a project rule at the store: a padded key is a different key, never trimmed."""
    store = SettingsKeyStore(API_KEYS)

    assert store.tenant_for(NORDLICHT_KEY + suffix) is None


def test_empty_key_store_is_valid_and_authorises_nobody() -> None:
    """The shipped default (`api_keys=""`) builds; it just holds no key."""
    store = SettingsKeyStore("")

    assert store.tenant_for(NORDLICHT_KEY) is None
