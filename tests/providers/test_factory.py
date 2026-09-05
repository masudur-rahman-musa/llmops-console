"""`build_adapters`: which policy providers become adapters, and which are dropped loudly.

The environment is passed in as an explicit mapping rather than patched onto
`os.environ`, so no test reads a provider key from the real environment (#66) and
two cases cannot interfere through process state.

Every provider in `POLICY` below differs from the built one in exactly one
attribute - the key, the `enabled` flag, or the kind - so a row going green can
only be the control it names. The disabled provider deliberately has its key
present in `ENVIRON`: without that, "absent" would be explained by the missing
key and the `enabled` check could be deleted with the test still passing.
"""

from __future__ import annotations

import logging

import pytest

from opsconsole.providers.base import Adapter
from opsconsole.providers.factory import build_adapters
from opsconsole.routing.policy import Policy, Provider, ProviderKind

KEY_ENV = "ANTHROPIC_API_KEY_EU"
UNSET_ENV = "ANTHROPIC_API_KEY_NOWHERE"
EMPTY_ENV = "ANTHROPIC_API_KEY_EMPTY"
OPENAI_ENV = "OPENAI_API_KEY"

POLICY = Policy(
    providers={
        "anthropic-eu": Provider(kind=ProviderKind.ANTHROPIC, region="eu", key_env=KEY_ENV),
        "anthropic-unset": Provider(kind=ProviderKind.ANTHROPIC, region="eu", key_env=UNSET_ENV),
        "anthropic-empty": Provider(kind=ProviderKind.ANTHROPIC, region="eu", key_env=EMPTY_ENV),
        "anthropic-nokeyenv": Provider(kind=ProviderKind.ANTHROPIC, region="eu"),
        # Enabled=False is the ONLY difference from `anthropic-eu`: its key is
        # set in ENVIRON, so nothing else can explain its absence.
        "anthropic-off": Provider(
            kind=ProviderKind.ANTHROPIC, region="eu", key_env=KEY_ENV, enabled=False
        ),
        # Kind is the only difference: enabled, and its key is set too.
        "openai-eu": Provider(kind=ProviderKind.OPENAI, region="eu", key_env=OPENAI_ENV),
    },
    eu_regions=("eu", "self"),
    aliases={},
    tenants={},
)

ENVIRON = {
    KEY_ENV: "sk-ant-eu-not-a-real-key",
    EMPTY_ENV: "",
    OPENAI_ENV: "sk-openai-not-a-real-key",
}


def test_enabled_provider_with_its_key_set_is_built() -> None:
    """Catches a factory keyed by the provider kind or by the vendor name: the ledger, the
    metric label and the resolver all address an adapter by its policy provider name."""
    adapters = build_adapters(POLICY, ENVIRON)

    assert "anthropic-eu" in adapters
    built = adapters["anthropic-eu"]
    assert isinstance(built, Adapter)
    assert built.name == "anthropic-eu"
    assert built.price_key("claude-haiku-4-5") == "anthropic-eu/claude-haiku-4-5"


def test_only_the_usable_anthropic_provider_is_built() -> None:
    """Catches a factory that builds an adapter with an empty key (or with `None`), which
    would look healthy to the resolver and 401 on the first real request."""
    adapters = build_adapters(POLICY, ENVIRON)

    assert set(adapters) == {"anthropic-eu"}


OMITTED_CASES: list[tuple[str, str, str | None]] = [
    ("the variable is not in the environment", "anthropic-unset", UNSET_ENV),
    ("the variable is set but empty", "anthropic-empty", EMPTY_ENV),
    ("no key_env is configured at all", "anthropic-nokeyenv", None),
]


@pytest.mark.parametrize(
    ("case", "provider", "variable"), OMITTED_CASES, ids=[case for case, *_ in OMITTED_CASES]
)
def test_enabled_provider_without_a_key_is_omitted_with_a_warning(
    case: str, provider: str, variable: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches the silent drop: a provider that vanishes without a log line looks to an
    operator like a routing bug, and the empty-string row catches a truthiness check
    written as `variable in environ`."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(POLICY, ENVIRON)

    assert provider not in adapters, case
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and provider in record.getMessage()
    ]
    assert len(warnings) == 1, f"{case}: expected exactly one warning naming {provider}"
    if variable is not None:
        assert variable in warnings[0], f"{case}: the warning must name the variable to set"


def test_disabled_provider_is_omitted_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches a factory that ignores `enabled` (an operator switching a provider off in
    the policy would keep serving traffic from it) and one that warns about a deliberate
    decision, which trains operators to ignore the warning that matters."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(POLICY, ENVIRON)

    assert "anthropic-off" not in adapters
    assert [
        record.getMessage() for record in caplog.records if "anthropic-off" in record.getMessage()
    ] == []


def test_non_anthropic_kind_is_absent_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Catches a factory that builds an `AnthropicAdapter` for every enabled provider with
    a key, which would send OpenAI traffic to Anthropic's endpoint with an OpenAI key."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(POLICY, ENVIRON)

    assert "openai-eu" not in adapters
    assert [
        record.getMessage() for record in caplog.records if "openai-eu" in record.getMessage()
    ] == []


def test_an_empty_policy_builds_nothing_and_says_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Catches a factory with a hardcoded default provider: an empty policy must produce an
    empty mapping, not a real Anthropic adapter nobody configured."""
    empty = Policy(providers={}, eu_regions=("eu",), aliases={}, tenants={})

    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(empty, ENVIRON)

    assert adapters == {}
    assert caplog.records == []
