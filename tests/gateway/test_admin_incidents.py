"""`/admin/incidents`, `/admin/incidents/{incident_id}` and `/admin/releases`: the write surface.

Three claims are pinned here, and each case says which wrong implementation it
would catch rather than merely which behaviour it likes.

* The three routes live in the admin key space the router already carries. A
  working TENANT key is a 401 on all three, and an app with no admin key
  configured authorises nobody - so a route added to the wrong router, or
  registered without the `admin_key` dependency, fails here rather than in
  production.
* An incident id is server-generated (decision A) and an incident is born OPEN.
  `POST` returns 32 hex and can never set `resolved_at`; only `PATCH` sets it,
  and both of its failures - an unknown id, an already-resolved one - are 422
  `VALIDATION` naming `id`, told apart by their messages (deviation 1).
* `eval_run_id` is checked for EXISTENCE before the insert, so the foreign key
  never surfaces as a 500.

Every case uses the session-scoped `admin_client` and `no_admin_key_client`. There is ONE documented exception, in the shape
tests/gateway/test_admin.py's `tmp_path` apps take: the unstored-run case needs
a writer whose `eval_run_exists` answers False, and no configuration of the
shipped app produces one - the in-memory writer answers True by design, and a
Postgres app would put this file behind a backing service. That case owns
`unstored_run_client`, whose `app.state.console_events` is replaced after
`create_app` with the double defined below, exactly as the provider stubs in
tests/gateway/conftest.py are swapped in after the factory has run.

`admin_client` is built from a `Settings` with no `database_url`, so it already
carries `InMemoryConsoleEventWriter` - which is what makes decision C's unhappy
path (`test_the_routes_answer_with_no_database_url_configured`) a read of the
same app every other case here uses, rather than a fifth app.

The contract these cases are written against, before it exists:

* `admin.ADMIN_INCIDENTS`, `admin.ADMIN_INCIDENT` and `admin.ADMIN_RELEASES`,
  mounted POST/PATCH/POST and answering 201/200/201;
* `app.state.console_events`, a `ConsoleEventWriter`, built alongside
  `app.state.db_pool` and `InMemoryConsoleEventWriter` when there is no pool.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsconsole.core.settings import Settings
from opsconsole.gateway.app import create_app
from opsconsole.gateway.routes import admin
from opsconsole.ledger.console_events import (
    IncidentRow,
    InMemoryConsoleEventWriter,
    ReleaseRow,
)

from .conftest import (
    ADMIN_KEY,
    API_KEYS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    TEST_KEYS,
    assert_error_body,
    sha256_hex,
)

# The three paths, spelled out here so the requests below do not read their
# target from the module under test. `test_the_admin_console_event_paths_and_
# statuses_are_pinned` compares these literals with the module's own constants
# and with the routes the factory mounted.
ADMIN_INCIDENTS = "/admin/incidents"
ADMIN_INCIDENT = "/admin/incidents/{incident_id}"
ADMIN_RELEASES = "/admin/releases"

ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}
TENANT_KEY = TEST_KEYS["nordlicht"]
TENANT_HEADERS = {"Authorization": f"Bearer {TENANT_KEY}"}

STARTED = datetime(2026, 9, 7, 22, 45, 13, tzinfo=UTC)
RESOLVED = STARTED + timedelta(hours=3)
DEPLOYED = datetime(2026, 9, 7, 9, 15, 0, tzinfo=UTC)

HEX_32 = re.compile(r"[0-9a-f]{32}")
"""What the body's `id` must be: 32 lowercase hex, `re.fullmatch`."""

UNKNOWN_INCIDENT_ID = "ffffffffffffffffffffffffffffffff"
"""Well-formed and stored by nothing, so the SHAPE pre-check cannot be what refuses it."""

UNSTORED_RUN_ID = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
"""Likewise for `eval_run_id`: 32 hex, so only an existence check can refuse it."""

GIT_SHA = "abc1234"


def an_incident_body(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """The `POST /admin/incidents` body, with the named fields replaced."""
    body: dict[str, Any] = {
        "started_at": STARTED.isoformat(),
        "severity": "sev2",
        "title": "gateway 5xx on the eu pool",
        "summary": "the eu pool answered 503 for eleven minutes",
        "postmortem_url": None,
    }
    body.update(overrides)
    return body


def a_release_body(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """The `POST /admin/releases` body, with the named fields replaced."""
    body: dict[str, Any] = {
        "git_sha": GIT_SHA,
        "deployed_at": DEPLOYED.isoformat(),
        "eval_run_id": None,
        "pr_url": None,
        "status": "deployed",
    }
    body.update(overrides)
    return body


def a_patch_body() -> dict[str, Any]:
    """The `PATCH /admin/incidents/{incident_id}` body."""
    return {"resolved_at": RESOLVED.isoformat()}


def incident_path(incident_id: str) -> str:
    """The PATCH path for one id, from the template literal above."""
    return ADMIN_INCIDENT.format(incident_id=incident_id)


def post_incident(client: TestClient, **overrides: Any) -> httpx.Response:  # noqa: ANN401
    """One authorised `POST /admin/incidents`."""
    response: httpx.Response = client.post(
        ADMIN_INCIDENTS, json=an_incident_body(**overrides), headers=ADMIN_HEADERS
    )
    return response


def created_incident_id(client: TestClient, **overrides: Any) -> str:  # noqa: ANN401
    """POST one incident and answer the id the body carried."""
    response = post_incident(client, **overrides)
    assert response.status_code == 201
    return str(response.json()["id"])


def console_events_of(client: TestClient) -> Any:  # noqa: ANN401
    """The writer the app under this client was built with."""
    return cast(FastAPI, client.app).state.console_events


# --- the one extra app (see the module docstring) ----------------------------


class NoRunsConsoleEventWriter:
    """A `ConsoleEventWriter` that knows of no eval run at all.

    The in-memory writer answers True for every run id by design, so it cannot
    express the one state `test_a_release_naming_an_unstored_run_is_422_
    validation_naming_eval_run_id` is about. `releases` is recorded so that case
    can assert the insert never ran, which is the difference between a pre-check
    and a caught integrity error.
    """

    def __init__(self) -> None:
        self.incidents: list[IncidentRow] = []
        self.releases: list[ReleaseRow] = []

    def write_incident(self, row: IncidentRow) -> None:
        self.incidents.append(row)

    def resolve_incident(self, incident_id: str, resolved_at: datetime) -> None:
        raise AssertionError(f"this double never resolves {incident_id} at {resolved_at}")

    def write_release(self, row: ReleaseRow) -> None:
        self.releases.append(row)

    def eval_run_exists(self, eval_run_id: str) -> bool:
        return False


@pytest.fixture(scope="session")
def unstored_run_client() -> Iterator[TestClient]:
    """The admin app with a writer that reports every eval run missing."""
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        admin_key_sha256=sha256_hex(ADMIN_KEY),
    )
    app = create_app(settings, NO_PROVIDER_KEYS)
    app.state.console_events = NoRunsConsoleEventWriter()
    with TestClient(app) as client:
        yield client


# --- the paths and the statuses ----------------------------------------------


def test_the_admin_console_event_paths_and_statuses_are_pinned(admin_client: TestClient) -> None:
    """Catches an implementation that mounts these under some other path or some other
    method - `POST /admin/incident`, a PUT where the contract says PATCH - and catches a
    create answering 200 instead of 201. Both halves are asserted: the literals against
    the module's constants, so a renamed constant is caught, and the literals against
    the routes the FACTORY mounted, so a constant that exists while the route was never
    registered is caught too. 201/200/201 are values this plan introduces, so standing
    rule 2 has them asserted exactly rather than as "not an error".
    """
    assert admin.ADMIN_INCIDENTS == ADMIN_INCIDENTS == "/admin/incidents"
    assert admin.ADMIN_INCIDENT == ADMIN_INCIDENT == "/admin/incidents/{incident_id}"
    assert admin.ADMIN_RELEASES == ADMIN_RELEASES == "/admin/releases"

    # The mounted paths, read from the app's own OpenAPI document rather than by
    # walking `app.routes`: this FastAPI wraps every `include_router` call in an
    # internal router object, so `app.routes` carries no flat `path` to compare.
    mounted = set(cast(FastAPI, admin_client.app).openapi()["paths"])
    assert {ADMIN_INCIDENTS, ADMIN_INCIDENT, ADMIN_RELEASES} <= mounted

    created = post_incident(admin_client)
    assert created.status_code == 201

    patched = admin_client.patch(
        incident_path(str(created.json()["id"])), json=a_patch_body(), headers=ADMIN_HEADERS
    )
    assert patched.status_code == 200

    released = admin_client.post(ADMIN_RELEASES, json=a_release_body(), headers=ADMIN_HEADERS)
    assert released.status_code == 201


# --- the admin key space -----------------------------------------------------

# (case, method, path, body). The PATCH row targets a well-formed but unstored
# id on purpose: an unauthorised call must be refused BEFORE the lookup, so a
# route ordering its key check after its id check would answer 422 here.
ROUTE_ROWS: list[tuple[str, str, str, dict[str, Any] | None]] = [
    ("post an incident", "POST", ADMIN_INCIDENTS, an_incident_body()),
    ("patch an incident", "PATCH", incident_path(UNKNOWN_INCIDENT_ID), a_patch_body()),
    ("post a release", "POST", ADMIN_RELEASES, a_release_body()),
]


@pytest.mark.parametrize(
    ("case", "method", "path", "body"), ROUTE_ROWS, ids=[row[0] for row in ROUTE_ROWS]
)
def test_every_console_event_route_is_401_without_an_admin_key(
    case: str,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    admin_client: TestClient,
    no_admin_key_client: TestClient,
) -> None:
    """Catches a route registered on the admin router WITHOUT the `admin_key`
    dependency, and catches it authenticating against the tenant key store instead: the
    key presented to `admin_client` here is a WORKING tenant key on the same app, so a
    201 or a 200 means the two key spaces were merged and any tenant can write the
    incident timeline. The no-key client is the other direction - an unset admin key
    must authorise nobody rather than everybody. The last assertion is that neither the
    presented key nor the configured one appears in the response text, because the
    cheapest way to write an unhelpful 401 is to echo what was sent.
    """
    unauthorised = no_admin_key_client.request(method, path, json=body, headers=ADMIN_HEADERS)
    assert_error_body(unauthorised, status=401, code="UNAUTHORIZED")

    tenant = admin_client.request(method, path, json=body, headers=TENANT_HEADERS)
    assert_error_body(tenant, status=401, code="UNAUTHORIZED")

    assert TENANT_KEY not in tenant.text
    assert ADMIN_KEY not in unauthorised.text


# --- the incident write path -------------------------------------------------


def test_the_incident_post_returns_a_thirty_two_hex_id_and_never_sets_resolved_at(
    admin_client: TestClient,
) -> None:
    """Catches the two ways decision A can be implemented wrongly. First, a route that
    stores an incident and answers an empty body: the caller then has no id to PATCH
    with and the resolve path is unreachable, which no other case here would notice
    because they all read the id from this response. Second, a request model carrying
    `resolved_at`, which lets a caller file an incident that was never open - the
    timeline's whole definition of "open" is a NULL there. The second POST sends
    `resolved_at` anyway, and the assertion is over every row this case ADDED rather
    than over a status code, so it holds whether the route refuses the field or ignores
    it. The before-snapshot is what keeps the claim about this case's own writes on the
    session-scoped app, which earlier cases have already resolved incidents on.
    """
    writer = console_events_of(admin_client)
    before = {row.id for row in writer.incidents}

    response = post_incident(admin_client)
    assert response.status_code == 201

    body = response.json()
    assert HEX_32.fullmatch(str(body["id"])) is not None

    admin_client.post(
        ADMIN_INCIDENTS,
        json=an_incident_body(resolved_at=RESOLVED.isoformat()),
        headers=ADMIN_HEADERS,
    )

    added = [row for row in writer.incidents if row.id not in before]
    assert body["id"] in {row.id for row in added}
    assert [row.id for row in added if row.resolved_at is not None] == []


def test_patching_an_unknown_incident_is_422_validation_naming_id(
    admin_client: TestClient,
) -> None:
    """Catches `PATCH` implemented as an UPDATE whose zero-row result is discarded,
    which answers 200 for an id that never existed - so an operator who mistyped one
    character believes the incident is closed while it is still open on the timeline.
    The id sent is well-formed 32 hex, so the shape pre-check cannot be what refuses it
    and only a lookup can; `param` is asserted because that is the field the console
    puts the message beside.
    """
    response = admin_client.patch(
        incident_path(UNKNOWN_INCIDENT_ID), json=a_patch_body(), headers=ADMIN_HEADERS
    )

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "id"


def test_patching_a_resolved_incident_is_422_validation_with_its_own_message(
    admin_client: TestClient,
) -> None:
    """Catches a resolve implemented as an idempotent write, which quietly OVERWRITES
    the first resolution time on a second PATCH - the one number a postmortem's duration
    is computed from. It also catches both failures being rendered by one branch: the
    unknown-id message is fetched in this same test and asserted DIFFERENT, because once
    deviation 1 made both 422 VALIDATION naming `id`, the message is the only thing left
    telling "no such incident" from "already resolved" on the wire.
    """
    incident_id = created_incident_id(admin_client)

    first = admin_client.patch(
        incident_path(incident_id), json=a_patch_body(), headers=ADMIN_HEADERS
    )
    assert first.status_code == 200

    second = admin_client.patch(
        incident_path(incident_id), json=a_patch_body(), headers=ADMIN_HEADERS
    )
    assert_error_body(second, status=422, code="VALIDATION")
    assert second.json()["error"]["param"] == "id"

    unknown = admin_client.patch(
        incident_path(UNKNOWN_INCIDENT_ID), json=a_patch_body(), headers=ADMIN_HEADERS
    )
    assert second.json()["error"]["message"] != unknown.json()["error"]["message"]


# --- the release write path --------------------------------------------------


def test_a_release_naming_an_unstored_run_is_422_validation_naming_eval_run_id(
    unstored_run_client: TestClient,
) -> None:
    """Catches the release INSERT running first and the foreign key being what refuses
    the row - which reaches the caller as a 500 from an integrity error, so CI's deploy
    step fails with a stack trace instead of a typed message naming the input at fault.
    The recorded-releases assertion is what makes this a test of the PRE-check rather
    than of a caught exception: a route that inserted and then translated the error
    would answer 422 too, but the writer would have seen the row.
    """
    response = unstored_run_client.post(
        ADMIN_RELEASES,
        json=a_release_body(eval_run_id=UNSTORED_RUN_ID),
        headers=ADMIN_HEADERS,
    )

    assert_error_body(response, status=422, code="VALIDATION")
    assert response.json()["error"]["param"] == "eval_run_id"
    assert console_events_of(unstored_run_client).releases == []


# --- decision C: no database_url -------------------------------------------


def test_the_routes_answer_with_no_database_url_configured(admin_client: TestClient) -> None:
    """Catches `app.state.console_events` wired as a Postgres writer unconditionally, or
    wired to a pool the factory opens just for it. Either turns `docker compose`-less
    local work into a 500 on every one of these routes, and the second opens a second
    pool the app never closes. `db_pool is None` and the writer's TYPE are asserted
    together with live 201/200/201 answers, because a correct `state` with a route that
    still reaches for a connection would pass the first half alone.
    """
    app = cast(FastAPI, admin_client.app)

    assert app.state.db_pool is None
    assert isinstance(app.state.console_events, InMemoryConsoleEventWriter)

    incident_id = created_incident_id(admin_client)
    patched = admin_client.patch(
        incident_path(incident_id), json=a_patch_body(), headers=ADMIN_HEADERS
    )
    assert patched.status_code == 200

    released = admin_client.post(ADMIN_RELEASES, json=a_release_body(), headers=ADMIN_HEADERS)
    assert released.status_code == 201
