"""`POST /admin/directory`: the third admin route, its refusals, and what it never says.

The claims are the same three `/admin/routing` makes, plus one this route has of
its own.

* The admin key space. A WORKING tenant key is a 401 here and an unset admin key
  authorises nobody, including the caller presenting the key a configured app
  would accept. Both rows reuse `tests/gateway/conftest.py`'s `admin_client` and
  `no_admin_key_client`, so this file is asserting the same dependency the other
  two routes are behind rather than a second one that could drift.
* Every refusal happens before any store write. The app's directory store is
  wrapped once in a recorder, and each refused row asserts `writes == []` -
  "the directory is unchanged afterwards" would also pass for a route that wrote
  and then rolled back, which is not what a project rule asks for.
* Nothing carries a name back out. The response body, the response text and the
  log records of a SUCCESSFUL push are all checked for the name that was pushed,
  because a leak on the success path is the one a 4xx-only test never reaches.

a project rule: one app, the session-scoped `admin_client`, and per-test
isolation from a tenant name of this test's own. A directory is per tenant and
merges rather than replaces, so two cases that share a tenant would see each
other's names; every case here pushes under a tenant name of its own, and the
two that must NOT share one - the count-cap rows - derive theirs from the push
they send.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app.api.routes.admin.directory import ADMIN_DIRECTORY
from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.redaction.names_store import (
    MAX_DIRECTORY_NAME_CHARS,
    MAX_DIRECTORY_NAMES,
    MIN_DIRECTORY_NAME_CHARS,
    DirectoryStore,
)

from .conftest import (
    ADMIN_KEY,
    TEST_KEYS,
    assert_error_body,
    auth_headers,
)

MODELS = "/v1/models"
NORDLICHT_KEY = TEST_KEYS["nordlicht"]
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}

DIRECTORY_PATH = "/admin/directory"
"""The path spelled out here so the requests below do not read their target from
the module under test; `test_the_directory_path_is_the_published_one` compares
this literal with the module's own constant and with the mounted route."""

NAME_A = "Anna Schmidt"
NAME_B = "Bernd Kowalski"

AT_MIN_NAME = "Li"
UNDER_MIN_NAME = "L"
AT_MAX_NAME = "a" * MAX_DIRECTORY_NAME_CHARS
OVER_MAX_NAME = "a" * (MAX_DIRECTORY_NAME_CHARS + 1)


def bulk(count: int) -> list[str]:
    """`count` distinct legal names, for the rows about the count cap."""
    return [f"Persona{index:05d}" for index in range(count)]


class RecordingDirectoryStore:
    """The real store with every `put` recorded, so "no write" is observable.

    Wraps rather than replaces, because the read half has to keep working: the
    round-trip rows ask the same object for the names the route just wrote.
    """

    def __init__(self, inner: DirectoryStore) -> None:
        self.inner = inner
        self.writes: list[tuple[str, tuple[str, ...]]] = []

    def put(self, tenant: str, names: Sequence[str]) -> None:
        self.writes.append((tenant, tuple(names)))
        self.inner.put(tenant, names)

    def get(self, tenant: str) -> tuple[str, ...]:
        return self.inner.get(tenant)


@pytest.fixture(scope="module")
def directory_store(admin_client: TestClient) -> RecordingDirectoryStore:
    """The app's own directory store, wrapped once so every write here is recorded.

    `app.state.directory_store` by name, not by shape: `app.state` now carries two
    objects with a `put` and a `get` - the placeholder map store and this one - so
    a search by shape could not tell them apart, and picking the wrong one would
    make every "no write" row below pass for the wrong reason.
    """
    app = cast(FastAPI, admin_client.app)
    recorder = RecordingDirectoryStore(inner=cast(DirectoryStore, app.state.directory_store))
    app.state.directory_store = recorder
    return recorder


@pytest.fixture(scope="module")
def no_key_directory_store(no_admin_key_client: TestClient) -> RecordingDirectoryStore:
    """The same wrap on the app that has no admin key, for the row that asserts it wrote
    nothing: two apps mean two stores, and only this one can see that app's writes."""
    app = cast(FastAPI, no_admin_key_client.app)
    recorder = RecordingDirectoryStore(inner=cast(DirectoryStore, app.state.directory_store))
    app.state.directory_store = recorder
    return recorder


@pytest.fixture(autouse=True)
def _fresh_writes(directory_store: RecordingDirectoryStore) -> Iterator[None]:
    """Per-test isolation by clearing the record, never by rebuilding the app."""
    directory_store.writes.clear()
    yield


def push(
    client: TestClient,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> Response:
    """POST one directory push with the admin key unless the case sends its own."""
    response: Response = client.post(
        DIRECTORY_PATH, json=body, headers=ADMIN_HEADERS if headers is None else headers
    )
    return response


# --------------------------------------------------------------------------
# the path and the key space
# --------------------------------------------------------------------------


def test_the_directory_path_is_the_published_one(admin_client: TestClient) -> None:
    """Catches a route mounted at a near-miss path (`/admin/directories`, or under
    `/v1/`), which would leave the published one answering Starlette's own 404 with no
    typed code at all. Asserted through the built app rather than by reading the router: the framework has handlers nobody here registered."""
    assert ADMIN_DIRECTORY == DIRECTORY_PATH == "/admin/directory"

    response = push(admin_client, {"tenant": "tpath", "names": [NAME_A]})

    assert response.status_code == 200, response.text


def test_the_route_writes_to_the_store_the_app_built(
    admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """Catches a route that builds a store of its own per request: the names would be
    written into an object thrown away at the end of the call, so the push would answer
    200 and the redaction path would go on seeing an empty directory forever."""
    push(admin_client, {"tenant": "tstore", "names": [NAME_A]})

    assert directory_store.writes == [("tstore", (NAME_A,))]
    assert directory_store.get("tstore") == (NAME_A,)


def test_the_directory_route_rejects_a_tenant_api_key(admin_client: TestClient) -> None:
    """Catches a route registered on a router without the admin dependency, or one
    authenticating against the tenant key store: the key sent here is a WORKING tenant key
    on the same app, so a 200 means the two key spaces were merged and any tenant could
    rewrite any other tenant's directory."""
    response = push(
        admin_client,
        {"tenant": "nordlicht", "names": [NAME_A]},
        headers={"Authorization": f"Bearer {NORDLICHT_KEY}"},
    )

    assert_error_body(response, status=401, code="UNAUTHORIZED")

    # The same key on the tenant route still works, so the 401 above is the admin
    # comparison refusing it and not a broken key or a broken app.
    assert (
        admin_client.get(
            MODELS, headers=auth_headers(key=NORDLICHT_KEY, tenant="nordlicht")
        ).status_code
        == 200
    )


NO_ADMIN_KEY_CASES: list[tuple[str, dict[str, str]]] = [
    ("the key a configured app would accept", {"Authorization": f"Bearer {ADMIN_KEY}"}),
    ("no Authorization header at all", {}),
    ("an empty Authorization header", {"Authorization": ""}),
    ("a Bearer scheme with an empty value", {"Authorization": "Bearer "}),
]


@pytest.mark.parametrize(
    ("case", "headers"), NO_ADMIN_KEY_CASES, ids=[row[0] for row in NO_ADMIN_KEY_CASES]
)
def test_the_directory_route_rejects_everyone_when_no_admin_key_is_configured(
    case: str, headers: dict[str, str], no_admin_key_client: TestClient
) -> None:
    """Catches an unset `admin_key_sha256` read as "no admin key means no check": the first
    row presents the very key the configured app accepts, so a permissive-when-unset
    dependency answers 200 to it and an unconfigured deployment lets anyone seed the
    directory that decides what gets redacted."""
    response = push(no_admin_key_client, {"tenant": "tnokey", "names": [NAME_A]}, headers=headers)

    assert_error_body(response, status=401, code="UNAUTHORIZED")


def test_an_unauthorised_push_writes_nothing(
    no_admin_key_client: TestClient, no_key_directory_store: RecordingDirectoryStore
) -> None:
    """Catches a dependency evaluated after the body has been handled: the 401 status
    alone cannot tell the difference between a request refused and a request served and
    then reported as refused, and the recorder is what can.

    The recorder is the one wrapping THIS app's store, not the configured app's - a
    request to one app can never move the other's recorder, and asserting the wrong one
    would make this row pass whatever the route did.
    """
    push(no_admin_key_client, {"tenant": "tnowrite", "names": [NAME_A]}, headers={})

    assert no_key_directory_store.writes == []


# --------------------------------------------------------------------------
# what a push does
# --------------------------------------------------------------------------


def test_a_push_stores_every_name_for_that_tenant(
    admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """Catches a route that stores only the first name of the list, which every
    single-name row would miss."""
    response = push(admin_client, {"tenant": "tstores", "names": [NAME_A, NAME_B]})

    assert response.status_code == 200, response.text
    assert sorted(directory_store.get("tstores")) == sorted([NAME_A, NAME_B])


def test_a_re_push_leaves_the_directory_unchanged(
    admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """Catches a route appending blindly: the second push spells the name differently, so
    a directory keyed on the raw string grows to two entries for one person, burns a slot
    of the 2000 and compiles a duplicate branch into the alternation."""
    push(admin_client, {"tenant": "trepush", "names": [NAME_A]})

    response = push(admin_client, {"tenant": "trepush", "names": ["  anna schmidt  "]})

    assert response.status_code == 200, response.text
    assert len(directory_store.get("trepush")) == 1


def test_a_second_push_merges_rather_than_replacing(
    admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """Catches replace-whole semantics (decision C): a caller paging a large directory
    would silently delete every name it had already pushed, and nothing in the response
    would say so."""
    push(admin_client, {"tenant": "tmerge", "names": [NAME_A]})

    push(admin_client, {"tenant": "tmerge", "names": [NAME_B]})

    assert sorted(directory_store.get("tmerge")) == sorted([NAME_A, NAME_B])


def test_a_push_names_the_tenant_from_the_body(
    admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """Catches a route filing the names under some other tenant - the admin caller, a
    default, an `X-Tenant` header: the whole point of decision A is that an operator seeds
    a tenant it is not authenticating as, so a route reading the tenant from anywhere but
    the body writes into the wrong directory and the names are never matched."""
    push(admin_client, {"tenant": "tbody", "names": [NAME_A]})

    assert directory_store.get("tbody") == (NAME_A,)
    assert directory_store.get("nordlicht") == ()


# --------------------------------------------------------------------------
# the refusals, each before any store write
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "names"),
    [
        ("a name at exactly the minimum length", [AT_MIN_NAME]),
        ("a name at exactly the maximum length", [AT_MAX_NAME]),
        ("exactly the maximum number of names", bulk(MAX_DIRECTORY_NAMES)),
    ],
)
def test_a_push_at_a_boundary_is_accepted(
    case: str, names: list[str], admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """The inclusive half of all three boundary pairs: catches a comparison written `>=`
    where `>` belongs, which would 422 a push an operator is entitled to make and which
    the rejected halves below cannot see. Each row uses its own tenant, so the 2000-name
    row does not fill a directory a later row pushes into."""
    tenant = f"tok{len(names)}{len(names[0])}"

    response = push(admin_client, {"tenant": tenant, "names": names})

    assert response.status_code == 200, response.text
    assert len(directory_store.get(tenant)) == len(names)


@pytest.mark.parametrize(
    ("case", "names", "limit"),
    [
        ("one character under the minimum", [UNDER_MIN_NAME], MIN_DIRECTORY_NAME_CHARS),
        ("an empty name", [""], MIN_DIRECTORY_NAME_CHARS),
        ("one character over the maximum", [OVER_MAX_NAME], MAX_DIRECTORY_NAME_CHARS),
        ("one name over the count cap", bulk(MAX_DIRECTORY_NAMES + 1), MAX_DIRECTORY_NAMES),
        ("a bad name last in an otherwise legal push", [NAME_A, UNDER_MIN_NAME], 2),
    ],
)
def test_a_push_outside_a_cap_is_422_before_any_store_write(
    case: str,
    names: list[str],
    limit: int,
    admin_client: TestClient,
    directory_store: RecordingDirectoryStore,
) -> None:
    """Catches a cap enforced after the write - which stores the names and then reports a
    failure, leaving a directory neither side believes in - and a cap not enforced at the
    route at all, which would surface as a 500 from the store. The last row is the one
    that fails a per-item write loop: the good name comes first."""
    response = push(admin_client, {"tenant": "tcap", "names": names})

    assert_error_body(response, status=422, code="VALIDATION")
    assert str(limit) in response.json()["error"]["message"]
    assert directory_store.writes == []


REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the identifier cap", "a" * (MAX_IDENTIFIER_LENGTH + 1)),
    ("uppercase", "Nordlicht"),
    ("a space", "nord licht"),
    ("a colon, which would forge a second key segment", "nord:licht"),
    ("a glob character, which would widen a later SCAN", "nord*"),
    ("a trailing newline", "nordlicht\n"),
    ("a trailing CRLF", "nordlicht\r\n"),
    ("a trailing space", "nordlicht "),
]


@pytest.mark.parametrize(
    ("case", "tenant"), REJECTED_TENANTS, ids=[row[0] for row in REJECTED_TENANTS]
)
def test_a_push_for_a_malformed_tenant_is_422_before_any_store_write(
    case: str, tenant: str, admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """Catches a `tenant` that reaches the store unchecked: it shapes a Redis key, so the
    colon row would write into another tenant's namespace, and the three trailing-
    whitespace rows are what separate `fullmatch` from `match`. The
    route's own refusal is a 422 rather than the 500 an unchecked value would become."""
    response = push(admin_client, {"tenant": tenant, "names": [NAME_A]})

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "tenant"
    assert directory_store.writes == []


def test_a_tenant_at_the_identifier_cap_is_accepted(
    admin_client: TestClient, directory_store: RecordingDirectoryStore
) -> None:
    """The inclusive half of the tenant boundary pair: catches a cap written `>=`, which
    would refuse a tenant name the gateway authenticates everywhere else and leave that
    tenant unable to have a directory at all."""
    tenant = "b" * MAX_IDENTIFIER_LENGTH

    response = push(admin_client, {"tenant": tenant, "names": [NAME_A]})

    assert response.status_code == 200, response.text
    assert directory_store.get(tenant) == (NAME_A,)


@pytest.mark.parametrize(
    ("case", "body"),
    [
        ("no tenant at all", {"names": [NAME_A]}),
        ("no names field", {"tenant": "tshape"}),
        ("names as a bare string", {"tenant": "tshape", "names": NAME_A}),
        ("a misspelled field", {"tenant": "tshape", "name": [NAME_A]}),
        ("an unknown extra field", {"tenant": "tshape", "names": [NAME_A], "replace": True}),
    ],
)
def test_a_body_of_the_wrong_shape_is_422_before_any_store_write(
    case: str,
    body: dict[str, Any],
    admin_client: TestClient,
    directory_store: RecordingDirectoryStore,
) -> None:
    """Catches a model that ignores unknown fields rather than forbidding them, as the
    other wire models here do: a caller sending `replace: true` would otherwise get a 200
    for a request whose meaning the gateway silently dropped, and a misspelled `names`
    would answer 200 having stored nothing."""
    response = push(admin_client, body)

    assert_error_body(response, status=422, code="VALIDATION")
    assert directory_store.writes == []


# --------------------------------------------------------------------------
# what the route never says
# --------------------------------------------------------------------------


def test_the_response_carries_no_name_back(admin_client: TestClient) -> None:
    """Catches a response echoing the stored directory: an admin endpoint is a lower bar
    than a tenant one, but the criterion says no name reaches a body, and an echo is how a
    directory ends up in a proxy log or a CI transcript that nobody classified as personal
    data."""
    response = push(admin_client, {"tenant": "tquiet", "names": [NAME_A, NAME_B]})

    assert response.status_code == 200, response.text
    assert NAME_A not in response.text
    assert NAME_B not in response.text
    assert response.json()["count"] == 2


def test_a_rejected_name_never_appears_in_the_error_body(admin_client: TestClient) -> None:
    """Catches a refusal built as `f"{name} is too long"`, and a route that lets pydantic
    render the offending input: the value is a person's name and the message a caller
    reads is the likeliest place for it to escape. The cap has to be there for the message
    to be actionable, so both halves are asserted together."""
    rejected = "Wilhelmina" * 20

    response = push(admin_client, {"tenant": "tmsg", "names": [rejected]})

    assert_error_body(response, status=422, code="VALIDATION")
    assert rejected not in response.text
    assert str(MAX_DIRECTORY_NAME_CHARS) in response.json()["error"]["message"]


def test_a_pushed_name_never_reaches_a_log_line(
    admin_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a route logging what it stored on the SUCCESS path, which a refusal-only
    leak test never reaches. DEBUG for the whole request, and both the rendered message
    and the raw `args` are checked, because `%s` formatting hides a value from
    `record.message`."""
    with caplog.at_level(logging.DEBUG):
        response = push(admin_client, {"tenant": "tlog", "names": [NAME_A]})

    assert response.status_code == 200, response.text
    for record in caplog.records:
        assert NAME_A not in record.getMessage()
        assert NAME_A not in str(record.args)
