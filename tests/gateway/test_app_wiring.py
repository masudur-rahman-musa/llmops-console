"""What `create_app` builds, refuses to build, and takes apart again.

The key store and the policy snapshot are constructed by the factory, not on the
first request that needs them, so bad configuration takes the process down at
start rather than turning into a 401 nobody can explain or a 451 that is really
a typo. These are the tests that make that a property rather than an intention:
deferring either construction, or wrapping it to make a boot failure go away,
turns one of them red.

Every `create_app` call in this file passes an explicit environment, the three
refusal cases included: they raise on the key store or the policy before the
adapters are built, but that ordering is nothing this file pins, and a later plan
that builds adapters first would silently construct a live-keyed client from a
developer's exported key.

An earlier change adds the adapter set to the same list. It is built here rather than per
request (decision D), it is built from an injected environment rather than the
process's own (#66), and it is closed on shutdown - the half of decision D that
an earlier change's policy reload reuses, and the half nothing would otherwise notice: a
leaked `httpx` connection pool costs file descriptors, not a failing assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opsconsole.core.settings import Settings
from opsconsole.gateway.app import close_adapters, create_app
from opsconsole.providers.base import Adapter
from opsconsole.routing.errors import PolicyInvalid

from .conftest import API_KEYS, NO_PROVIDER_KEYS, SHIPPED_POLICY_PATH, StubAdapter

# The variable the shipped `routing.yaml` names for the enabled US Anthropic
# provider, and a value that is not a key: nothing here ever sends it anywhere.
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
FAKE_KEY = "not-a-key-and-never-sent"


def test_create_app_refuses_a_malformed_key_setting() -> None:
    """A pair with no digest fails the factory, naming the entry's position."""
    with pytest.raises(ValueError, match="api key entry 1"):
        create_app(
            Settings(api_keys="nordlicht", routing_policy_path=SHIPPED_POLICY_PATH),
            NO_PROVIDER_KEYS,
        )


def test_create_app_refuses_an_invalid_policy_file(tmp_path: Path) -> None:
    """A policy the schema rejects fails the factory, not the first request."""
    broken = tmp_path / "routing.yaml"
    broken.write_text("providers: {}\nnot_a_key: 1\n", encoding="utf-8")

    with pytest.raises(PolicyInvalid):
        create_app(Settings(api_keys=API_KEYS, routing_policy_path=broken), NO_PROVIDER_KEYS)


def test_create_app_refuses_a_missing_policy_file(tmp_path: Path) -> None:
    """The path pointing at nothing is the same class of failure."""
    with pytest.raises(PolicyInvalid):
        create_app(
            Settings(api_keys=API_KEYS, routing_policy_path=tmp_path / "absent.yaml"),
            NO_PROVIDER_KEYS,
        )


def wiring_settings() -> Settings:
    """The test key store against the shipped policy.

    Not named `settings`: that is the session-scoped fixture in `conftest.py`, and
    a module-level function of the same name shadows it for this whole file.
    """
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH)


def test_create_app_builds_the_adapter_set() -> None:
    """Catches an app that leaves `app.state.adapters` unset or builds it per request:
    the chat route reads it for the resolver's health seam, so an unset one is a 500 on
    the first completion, and a per-request one mints a connection pool per call."""
    app = create_app(wiring_settings(), {ANTHROPIC_KEY_ENV: FAKE_KEY})

    adapters = app.state.adapters
    # `local` joins `anthropic` from an earlier change on: the shipped policy's
    # `openai_compatible` provider declares no `key_env`, so it is built from the
    # policy alone and needs nothing in the environment (06a decision B).
    assert set(adapters) == {"anthropic", "local"}
    assert isinstance(adapters["anthropic"], Adapter)
    assert isinstance(adapters["local"], Adapter)
    close_adapters(adapters)


def test_create_app_reads_keys_only_from_the_given_environ() -> None:
    """Catches a factory that reads `os.environ` behind the injected mapping (#66): with a
    real key exported, every app the suite builds would hold a live client against the
    provider, and a test that made a real billed call would pass locally and fail in CI."""
    app = create_app(wiring_settings(), NO_PROVIDER_KEYS)

    # `local` survives an empty environment on purpose - it holds no key to read.
    # `anthropic` is the provider this test is about, and its absence is what
    # proves nothing was read from `os.environ` behind the injected mapping.
    assert set(app.state.adapters) == {"local"}
    close_adapters(app.state.adapters)


class ClosingStub(StubAdapter):
    """A stub that records having been closed, which the protocol does not require."""

    def __init__(self) -> None:
        super().__init__(name="anthropic")
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_adapters_are_closed_on_shutdown() -> None:
    """Catches a `create_app` built without its lifespan, or a lifespan that closes nothing:
    every dropped adapter leaks an `httpx` connection pool, and nothing else in the suite
    would go red - an earlier change's reload churns the same path once per policy change."""
    app = create_app(wiring_settings(), NO_PROVIDER_KEYS)
    stub = ClosingStub()
    app.state.adapters = {stub.name: stub}

    with TestClient(app):
        assert stub.closed == 0

    assert stub.closed == 1


def test_close_adapters_skips_an_adapter_with_nothing_to_close() -> None:
    """Catches a close loop that assumes `close()` is on the `Adapter` protocol: it is not,
    because an adapter with no connection pool has nothing to close, and an
    `AttributeError` here would take a shutdown down with it."""
    plain = StubAdapter(name="anthropic")

    close_adapters({plain.name: plain})
