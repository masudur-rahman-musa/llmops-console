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

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.gateway.app import close_adapters, create_app
from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import ResidencyMode
from app.providers.base import Adapter
from app.redaction import InProcessRedactionMapStore, RedisRedactionMapStore

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
    # `local` and `local_embed` join `anthropic` from an earlier change on: the shipped
    # policy's `openai_compatible` providers declare no `key_env`, so they are built
    # from the policy alone and need nothing in the environment (06a decision B).
    # There are two of them because one llama.cpp process serves one model.
    assert set(adapters) == {"anthropic", "local", "local_embed"}
    assert isinstance(adapters["anthropic"], Adapter)
    assert isinstance(adapters["local"], Adapter)
    assert isinstance(adapters["local_embed"], Adapter)
    close_adapters(adapters)


def test_create_app_reads_keys_only_from_the_given_environ() -> None:
    """Catches a factory that reads `os.environ` behind the injected mapping (#66): with a
    real key exported, every app the suite builds would hold a live client against the
    provider, and a test that made a real billed call would pass locally and fail in CI."""
    app = create_app(wiring_settings(), NO_PROVIDER_KEYS)

    # The two self-hosted providers survive an empty environment on purpose - neither
    # holds a key to read. `anthropic` is the provider this test is about, and its
    # absence is what proves nothing was read from `os.environ` behind the injected
    # mapping.
    assert set(app.state.adapters) == {"local", "local_embed"}
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


# --- the redaction map store ---------------------------
#
# The store `create_app` picks is settings-driven: no `OPSCONSOLE_REDIS_URL`
# keeps the in-process default, so the suite and a bare local run need no Redis;
# one set switches to Redis for the whole process. Both rows are asserted,
# because a factory that always built one of them would satisfy the other row's
# test alone.

# A Redis nothing listens on. `redis.Redis` connects lazily, so building the
# store costs no connection and this URL is never dialled by these two cases.
UNCONNECTED_REDIS_URL = "redis://127.0.0.1:1/0"


def store_settings(*, redis_url: str | None) -> Settings:
    """`wiring_settings()` with the Redis URL stated either way.

    Stated rather than left to default: `Settings` reads `OPSCONSOLE_REDIS_URL`
    from the process environment, which the Redis suite asks a developer to
    export, and the unconfigured row would then assert the opposite of what it
    says.
    """
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, redis_url=redis_url)


def test_create_app_keeps_the_in_process_store_without_a_redis_url() -> None:
    """Catches a factory that requires Redis to start: the whole existing suite, `make
    verify` and a bare `uvicorn` run configure no `redis_url`, so a Redis-only factory
    turns every one of them into a connection error at import time rather than a 503."""
    app = create_app(store_settings(redis_url=None), NO_PROVIDER_KEYS)

    assert isinstance(app.state.redaction_map_store, InProcessRedactionMapStore)
    close_adapters(app.state.adapters)


def test_create_app_uses_the_redis_store_when_a_redis_url_is_configured() -> None:
    """Catches a factory that reads the setting for the probe and still files maps in
    process memory: two gateway replicas would then each hold half the maps, and a caller
    would get "no such map" for an id the other replica issued - a failure no test with
    one replica can see except at the wiring."""
    app = create_app(store_settings(redis_url=UNCONNECTED_REDIS_URL), NO_PROVIDER_KEYS)

    assert isinstance(app.state.redaction_map_store, RedisRedactionMapStore)
    close_adapters(app.state.adapters)


# --- the residency mode --------------------------------------------
#
# A relaxed process is a process that may send a tenant's traffic to a US
# provider, so the one thing it may never be is indistinguishable from a strict
# one. The startup warning is half of that (the `/readyz` field is the other
# half, pinned in `tests/test_health.py`), and the refusal below is what stops a
# mistyped value starting at all.

APP_LOGGER = "app.gateway.app"


def residency_settings(mode: ResidencyMode) -> Settings:
    """`wiring_settings()` with the mode stated rather than left to the environment.

    Stated for the strict row too: the root conftest clears the variable for the
    suite, and a case that relied on that would assert the developer's shell.
    """
    return Settings(api_keys=API_KEYS, routing_policy_path=SHIPPED_POLICY_PATH, residency_mode=mode)


def residency_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The factory's own WARNING records about residency, and nothing else's.

    Filtered by logger and by subject, so an unrelated warning from the same
    module (an unconfigured store, a future deprecation) neither satisfies the
    relaxed case nor fails the strict one.
    """
    return [
        record
        for record in caplog.records
        if record.name == APP_LOGGER
        and record.levelno == logging.WARNING
        and "residency" in record.getMessage().lower()
    ]


def test_create_app_warns_once_under_relaxed_residency(caplog: pytest.LogCaptureFixture) -> None:
    """Catches a relaxed app that starts silently: the mode is the one setting that weakens a
    locked guarantee, and a log nobody can grep for is how a relaxed gateway ends up serving
    production traffic.

    Once rather than at least once, because a warning logged per call site - there are five
    `resolve()` callers - becomes per request and buries itself. The message must name the
    mode, so the line answers which mode without a second lookup."""
    with caplog.at_level(logging.WARNING, logger=APP_LOGGER):
        app = create_app(residency_settings(ResidencyMode.RELAXED), NO_PROVIDER_KEYS)

    warned = residency_warnings(caplog)
    assert len(warned) == 1, [record.getMessage() for record in warned]
    assert ResidencyMode.RELAXED.value in warned[0].getMessage()
    close_adapters(app.state.adapters)


def test_create_app_logs_no_residency_warning_under_strict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches a warning logged unconditionally beside the field it is about: that is what
    would make the case above pass whether or not the branch exists, and an alert that fires
    on every strict start is an alert an operator mutes."""
    with caplog.at_level(logging.WARNING, logger=APP_LOGGER):
        app = create_app(residency_settings(ResidencyMode.STRICT), NO_PROVIDER_KEYS)

    assert residency_warnings(caplog) == []
    close_adapters(app.state.adapters)


def test_create_app_refuses_an_unknown_residency_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a factory that builds its own `Settings` leniently - reading the variable with
    `os.environ.get` and a default - so a mistyped mode serves requests under whichever
    regime that default names. No settings argument is passed, because the environment is the
    only way this value ever arrives in a deployment."""
    monkeypatch.setenv("OPSCONSOLE_RESIDENCY_MODE", "loose")

    with pytest.raises(ValidationError, match="residency_mode"):
        create_app(environ={})
