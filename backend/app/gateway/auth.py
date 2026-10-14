"""Per-tenant API keys: the store, and the bearer key a request presents.

Keys are held as sha256 digests and never as plaintext. A presented key is
hashed and compared with `hmac.compare_digest`, so the comparison does not leak
the position of the first differing byte, and nothing here logs the key: the
plaintext appears in no response body and no log record, pinned by
`test_key_never_appears_in_body_or_log`.

The store is a seam. `SettingsKeyStore` reads `tenant:sha256hex` pairs from one
setting, which is the implementation that costs nothing to delete once Postgres
owns the table. Its entries are parsed at construction, so
a malformed pair fails when the process starts and names the pair's 1-based
position, rather than surfacing as a puzzling 401 at request time
(`test_key_store_rejects_malformed_entry_at_construction`).
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Protocol

from app.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

MAX_KEY_LENGTH = 256
"""Longest bearer key the gateway will hash. The value is pinned by
`test_max_key_length_is_pinned` and the inclusive boundary by
`test_bearer_key_length_boundary_is_inclusive`; a key one character over it is
`UNAUTHORIZED`, like any other unusable key."""

BEARER_SCHEME = "bearer"
"""The `Authorization` scheme, matched case-insensitively: RFC 7235 makes the
scheme token case-insensitive, and a client that sends `bearer` would otherwise
get a 401 whose reason is deliberately never disclosed. Pinned by
`test_bearer_scheme_is_case_insensitive`."""

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
"""A lowercase sha256 digest, matched with `re.fullmatch`: `.match` with `$`
would accept a trailing newline, and the digest is a dictionary key here."""


class KeyStore(Protocol):
    """Where the gateway looks a presented key up. Postgres fills this seam later."""

    def tenant_for(self, key: str) -> str | None:
        """The tenant the key belongs to, or `None` if no stored key matches."""
        ...


def _digest(key: str) -> str:
    """The stored form of a key. No plaintext is kept by the store or logged."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SettingsKeyStore:
    """`tenant:sha256hex` pairs from a single setting, parsed once.

    An empty setting is a valid store that authorises nobody, which is what the
    shipped default has to be (`test_empty_key_store_is_valid_and_authorises_nobody`).
    """

    def __init__(self, entries: str) -> None:
        self._by_digest: dict[str, str] = {}
        if not entries:
            return
        for position, entry in enumerate(entries.split(","), start=1):
            tenant, digest = self._parsed(entry, position)
            self._by_digest[digest] = tenant

    @staticmethod
    def _parsed(entry: str, position: int) -> tuple[str, str]:
        """Split and validate one pair, naming its position in every rejection."""
        tenant, separator, digest = entry.partition(":")
        if not separator:
            raise ValueError(
                f"api key entry {position}: expected 'tenant:sha256hex', got {entry!r}"
            )
        if len(tenant) > MAX_IDENTIFIER_LENGTH or IDENTIFIER_PATTERN.fullmatch(tenant) is None:
            raise ValueError(f"api key entry {position}: {tenant!r} is not a tenant name")
        if _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"api key entry {position}: the digest is not a lowercase sha256")
        return tenant, digest

    def tenant_for(self, key: str) -> str | None:
        """Hash the presented key and return its tenant, or `None`.

        Every stored digest is compared with `hmac.compare_digest` rather than
        the presented one being looked up in the dictionary. A dictionary lookup
        is itself a comparison, on a value derived from what the caller sent, so
        the loop is the version with nothing timing-sensitive left in it. Do not
        "optimise" this back to `self._by_digest[presented]`: the cost is one
        hash comparison per configured tenant, and no test can catch the
        regression, because constant-time behaviour is not something an
        assertion observes.
        """
        presented = _digest(key)
        for digest, tenant in self._by_digest.items():
            if hmac.compare_digest(digest, presented):
                return tenant
        return None


def bearer_key(authorization: str | None) -> str | None:
    """The key inside an `Authorization` header, or `None` if there is not one.

    Absent, empty, a scheme other than `Bearer`, an empty value and a value over
    `MAX_KEY_LENGTH` all give `None`: the caller turns every one of them into the
    same `UNAUTHORIZED` (`test_missing_key_is_unauthorized`). The key itself is
    taken verbatim and never stripped, so a trailing newline or space makes it a
    different key (`test_unknown_key_is_unauthorized`).
    """
    if authorization is None:
        return None
    scheme, separator, key = authorization.partition(" ")
    if not separator or scheme.lower() != BEARER_SCHEME:
        return None
    if not key or len(key) > MAX_KEY_LENGTH:
        return None
    return key
