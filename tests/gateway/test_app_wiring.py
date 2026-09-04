"""What `create_app` refuses to build.

The key store and the policy snapshot are constructed by the factory, not on the
first request that needs them, so bad configuration takes the process down at
start rather than turning into a 401 nobody can explain or a 451 that is really
a typo. These are the tests that make that a property rather than an intention:
deferring either construction, or wrapping it to make a boot failure go away,
turns one of them red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opsconsole.core.settings import Settings
from opsconsole.gateway.app import create_app
from opsconsole.routing.errors import PolicyInvalid

from .conftest import API_KEYS, SHIPPED_POLICY_PATH


def test_create_app_refuses_a_malformed_key_setting() -> None:
    """A pair with no digest fails the factory, naming the entry's position."""
    with pytest.raises(ValueError, match="api key entry 1"):
        create_app(Settings(api_keys="nordlicht", routing_policy_path=SHIPPED_POLICY_PATH))


def test_create_app_refuses_an_invalid_policy_file(tmp_path: Path) -> None:
    """A policy the schema rejects fails the factory, not the first request."""
    broken = tmp_path / "routing.yaml"
    broken.write_text("providers: {}\nnot_a_key: 1\n", encoding="utf-8")

    with pytest.raises(PolicyInvalid):
        create_app(Settings(api_keys=API_KEYS, routing_policy_path=broken))


def test_create_app_refuses_a_missing_policy_file(tmp_path: Path) -> None:
    """The path pointing at nothing is the same class of failure."""
    with pytest.raises(PolicyInvalid):
        create_app(Settings(api_keys=API_KEYS, routing_policy_path=tmp_path / "absent.yaml"))
