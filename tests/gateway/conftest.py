"""Fixtures for the request-envelope tests: keys, headers, errors, /v1/models.

a project rule: the app, its key store and the policy snapshot are built once,
in session-scoped fixtures. Per-test isolation comes from the tenant, the key and
the headers a case sends, never from rebuilding the app.

Three clients, each with its own app from `create_app`, so all three carry the
real handlers and none shares state with another:

* `gateway_client` talks to the shipped app exactly as it is served.
* `harness_client` talks to a second app plus the five routes `build_harness`
  registers under `/_harness/`: `context`, `residency`, `validated`, `boom` and
  `http-detail`. They exist because this plan adds no route that can be observed
  from the outside: nothing yet returns the `RequestContext` a header builds, and
  nothing yet raises a `GatewayError`, an `HTTPException` or a bug at request
  time (`/v1/models` filters the unresolvable aliases out rather than failing).
  Registering them on an app the factory built keeps the handlers, the
  dependencies and the state under test real; only those five endpoints are
  test-owned.
* `crashing_client` is a third app with the same five routes and
  `raise_server_exceptions` off, so the 500 handler answers instead of the
  exception reaching the test. Keeping it separate means an accidental crash in
  any other test still surfaces as the real exception rather than a masked 500.

The plaintext keys below are test data, not secrets: they are random-looking only
by shape, are worth nothing, and match no provider's key format.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import Response

from opsconsole.core.settings import Settings
from opsconsole.gateway.app import create_app
from opsconsole.gateway.auth import MAX_KEY_LENGTH
from opsconsole.gateway.context import RequestContext, request_context
from opsconsole.routing.errors import ResidencyUnavailable, RoutingError
from opsconsole.routing.policy import DataClass, Policy, load_policy
from opsconsole.routing.resolver import MAX_IDENTIFIER_LENGTH, resolve

# Resolved from this file, so the tests do not depend on pytest's working
# directory. The same path the shipped `Settings.routing_policy_path` defaults to.
SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "routing.yaml"

# A tenant name of exactly `MAX_IDENTIFIER_LENGTH` characters, used for the
# inclusive boundary row of the `X-Tenant` pre-check. It has a key of its own so
# the boundary case reaches the header check instead of stopping at auth.
MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH

# A key of exactly `MAX_KEY_LENGTH` characters and one character over it. Both
# are registered in the store, so the over-cap row is rejected by the length
# pre-check rather than by simply being unknown.
AT_CAP_KEY = "k" * MAX_KEY_LENGTH
OVER_CAP_KEY = "k" * (MAX_KEY_LENGTH + 1)

UNKNOWN_KEY = "test-key-belonging-to-nobody"

# tenant -> plaintext key. The store is given the sha256 digests of these.
TEST_KEYS: Mapping[str, str] = {
    "nordlicht": "test-key-nordlicht",
    "cuan": "test-key-cuan",
    "evals": "test-key-evals",
    MAX_TENANT: "test-key-max-tenant",
    "cap-key": AT_CAP_KEY,
    "over-cap-key": OVER_CAP_KEY,
}


def sha256_hex(key: str) -> str:
    """The digest form a key is stored as. Never the plaintext."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_entries(keys: Mapping[str, str]) -> str:
    """Render `tenant:sha256hex` pairs the way `OPSCONSOLE_API_KEYS` carries them."""
    return ",".join(f"{tenant}:{sha256_hex(key)}" for tenant, key in keys.items())


API_KEYS = key_entries(TEST_KEYS)


def auth_headers(
    *,
    key: str | None,
    tenant: str | None,
    data_class: str | None = None,
    session: str | None = None,
    traceparent: str | None = None,
    authorization: str | None = None,
) -> dict[str, str]:
    """Build the request headers, omitting anything left as `None`.

    `authorization` overrides the whole header, for the rows that send something
    other than a well-formed `Bearer` value.
    """
    headers: dict[str, str] = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    elif key is not None:
        headers["Authorization"] = f"Bearer {key}"
    if tenant is not None:
        headers["X-Tenant"] = tenant
    if data_class is not None:
        headers["X-Data-Class"] = data_class
    if session is not None:
        headers["X-Session"] = session
    if traceparent is not None:
        headers["traceparent"] = traceparent
    return headers


def assert_error_body(response: Response, *, status: int, code: str | None) -> None:
    """Assert the OpenAI-shaped failure body exactly: shape, status and code.

    The key set is compared as a whole, so an extra field (a stack trace, a
    `detail` list leaking from FastAPI's default handler) fails here. `code` is
    `None` for the failures Starlette raises before any of our code runs, which
    carry the shape but no typed code.
    """
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"message", "type", "code", "param"}
    assert error["code"] == code
    assert isinstance(error["message"], str)
    assert error["message"] != ""
    assert isinstance(error["type"], str)
    assert error["type"] != ""
    assert error["param"] is None or isinstance(error["param"], str)


def resolvable_aliases(policy: Policy, tenant: str, data_class: DataClass) -> list[str]:
    """The aliases `resolve()` does not raise for, in policy order.

    `/v1/models` is specified as exactly this list, so the
    expectation is computed from the resolver rather than copied from the policy
    file. Each test that uses it also asserts one membership literally, so a
    resolver that returned nothing could not make the comparison pass.
    """
    listed: list[str] = []
    for alias in policy.aliases:
        try:
            resolve(policy, tenant, alias, data_class.value)
        except RoutingError:
            continue
        listed.append(alias)
    return listed


def build_harness(app: FastAPI) -> FastAPI:
    """Add the five observation routes described in this module's docstring."""

    @app.get("/_harness/context")
    def read_context(
        context: Annotated[RequestContext, Depends(request_context)],
    ) -> RequestContext:
        """Echo the context the header pre-checks built."""
        return context

    @app.get("/_harness/residency")
    def raise_residency() -> None:
        """Raise a `GatewayError` at request time, for the handler test."""
        raise ResidencyUnavailable("harness: nothing may serve this request")

    @app.get("/_harness/validated")
    def validated(count: int) -> dict[str, int]:
        """A required, typed query parameter, for the FastAPI validation test."""
        return {"count": count}

    @app.get("/_harness/boom")
    def boom() -> None:
        """Raise something untyped, for the last-resort handler test."""
        raise RuntimeError("harness: a bug in a route")

    @app.get("/_harness/http-detail")
    def http_detail() -> None:
        """Raise an `HTTPException` whose detail is not a string."""
        # FastAPI's own `HTTPException` types `detail` as `Any`, which is how a
        # non-string one reaches the handler at all; Starlette's types it `str`.
        raise HTTPException(status_code=409, detail={"reason": "already registered"})

    return app


@pytest.fixture(scope="session")
def settings() -> Settings:
    """One settings object: the test key store and the shipped policy."""
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH)


@pytest.fixture(scope="session")
def gateway_client(settings: Settings) -> Iterator[TestClient]:
    """The shipped app, built once."""
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture(scope="session")
def harness_client(settings: Settings) -> Iterator[TestClient]:
    """The shipped app plus the `/_harness/` observation routes, built once."""
    with TestClient(build_harness(create_app(settings))) as client:
        yield client


@pytest.fixture(scope="session")
def crashing_client(settings: Settings) -> Iterator[TestClient]:
    """The harness app with `raise_server_exceptions` off.

    `TestClient` re-raises an unhandled exception by default, which is right for
    every other test and would hide the very handler under test here: the client
    would see the `RuntimeError` instead of the 500 a real client gets.
    """
    with TestClient(build_harness(create_app(settings)), raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(scope="session")
def policy() -> Policy:
    """The shipped routing policy, loaded once, for computing expectations."""
    return load_policy(SHIPPED_POLICY_PATH)
