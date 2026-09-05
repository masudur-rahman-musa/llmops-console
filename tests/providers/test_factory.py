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

from opsconsole.providers.anthropic import AnthropicAdapter
from opsconsole.providers.base import Adapter
from opsconsole.providers.factory import build_adapters
from opsconsole.providers.openai_compatible import OpenAICompatibleAdapter
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


# --- the openai_compatible kind -----------------------------------
#
# A policy of its own rather than more rows in `POLICY` above: the cases there
# assert the whole built set (`set(adapters) == {"anthropic-eu"}`), so adding a
# provider to it would edit an existing test's meaning. Every provider below
# differs from the built one in exactly one attribute - the `enabled` flag, the
# `key_env`, the `base_url` or the kind - so a row going green can only be the
# control it names.

LOCAL_KEY_ENV = "VLLM_API_KEY_NOWHERE"

LOCAL_POLICY = Policy(
    providers={
        # The shipped `local`: enabled, a base url, and no `key_env` at all.
        "local": Provider(
            kind=ProviderKind.OPENAI_COMPATIBLE, region="self", base_url="http://llama:8082/v1"
        ),
        # The shipped `vllm`, enabled here: the same kind on a second endpoint.
        "vllm": Provider(
            kind=ProviderKind.OPENAI_COMPATIBLE, region="self", base_url="http://vllm:8000/v1"
        ),
        # `enabled=False` is the ONLY difference from `local`.
        "local-off": Provider(
            kind=ProviderKind.OPENAI_COMPATIBLE,
            region="self",
            base_url="http://llama:8082/v1",
            enabled=False,
        ),
        # A declared key that is not in the environment is the only difference.
        "local-keyed": Provider(
            kind=ProviderKind.OPENAI_COMPATIBLE,
            region="self",
            base_url="http://llama:8082/v1",
            key_env=LOCAL_KEY_ENV,
        ),
        # No base url is the only difference: there is no endpoint to call.
        "local-nourl": Provider(kind=ProviderKind.OPENAI_COMPATIBLE, region="self"),
        # A different kind on the same policy, with its key set.
        "anthropic-eu": Provider(kind=ProviderKind.ANTHROPIC, region="eu", key_env=KEY_ENV),
    },
    eu_regions=("eu", "self"),
    aliases={},
    tenants={},
)


def test_an_openai_compatible_provider_with_no_key_env_is_built() -> None:
    """Catches a factory that demands a key of every provider (decision B): `local` and
    `vllm` sit on a private network and declare none, so a key check applied to them drops
    the residency-only route and every EU `pii` request stays a 503."""
    adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    assert "local" in adapters
    built = adapters["local"]
    assert isinstance(built, OpenAICompatibleAdapter)
    assert built.name == "local"
    assert built.price_key("qwen2.5-7b-instruct") == "local/qwen2.5-7b-instruct"


def test_one_class_serves_both_openai_compatible_providers() -> None:
    """Catches a factory that special-cases the provider NAME `local`: `vllm` is the same
    kind on another endpoint, and a name-keyed factory would silently drop it or
    build it with the wrong base url."""
    adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    local = adapters["local"]
    vllm = adapters["vllm"]
    assert isinstance(local, OpenAICompatibleAdapter)
    assert isinstance(vllm, OpenAICompatibleAdapter)
    assert type(local) is type(vllm)
    assert vllm.name == "vllm"
    # `rstrip`: httpx normalises a base url to a trailing slash, which is the one
    # difference between the policy string and the client's own.
    assert str(local.client.base_url).rstrip("/") == "http://llama:8082/v1"
    assert str(vllm.client.base_url).rstrip("/") == "http://vllm:8000/v1"


def test_the_anthropic_kind_still_gets_its_own_adapter() -> None:
    """Catches a factory that builds the new class for every enabled provider, which would
    send Anthropic traffic to the OpenAI chat-completions path with a bearer token - a
    401 on every request the EU tenants do not make locally."""
    adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    assert isinstance(adapters["anthropic-eu"], AnthropicAdapter)
    assert not isinstance(adapters["anthropic-eu"], OpenAICompatibleAdapter)


def test_a_disabled_openai_compatible_provider_is_omitted_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches a factory that ignores `enabled` for the new kind (an operator switching
    the model box off would keep having traffic routed to it) and one that warns about a
    deliberate decision, which trains operators to ignore the warning that matters."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    assert "local-off" not in adapters
    assert [
        record.getMessage() for record in caplog.records if "local-off" in record.getMessage()
    ] == []


def test_an_openai_compatible_provider_declaring_an_unset_key_is_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches decision B applied too widely: an absent `key_env` is legal, but one that
    is DECLARED and unset is a half-configured deployment, and building it anyway would
    put an adapter behind the resolver that 401s on the first real request."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    assert "local-keyed" not in adapters
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "local-keyed" in record.getMessage()
    ]
    assert len(warnings) == 1, "expected exactly one warning naming local-keyed"
    assert LOCAL_KEY_ENV in warnings[0], "the warning must name the variable to set"


def test_an_openai_compatible_provider_without_a_base_url_is_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches a factory that falls back to a default endpoint (the Anthropic one, or
    `localhost`) when the policy names none: an adapter with a guessed base url sends a
    tenant's `pii` traffic wherever that url happens to point, and the whole point of this
    provider is that its region is the box we run. A silent drop is caught too - it looks
    to an operator like a routing bug."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    assert "local-nourl" not in adapters
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "local-nourl" in record.getMessage()
    ]
    assert len(warnings) == 1, "expected exactly one warning naming local-nourl"


def test_exactly_the_usable_providers_of_both_kinds_are_built() -> None:
    """Catches every drop above being explained by something other than its own control:
    the built set is asserted whole, so a factory that dropped `local` for want of a key
    or kept `local-nourl` with a guessed endpoint fails here as well as in its own row."""
    adapters = build_adapters(LOCAL_POLICY, ENVIRON)

    assert set(adapters) == {"local", "vllm", "anthropic-eu"}
