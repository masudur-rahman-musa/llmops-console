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
import shutil
from pathlib import Path

import httpx
import pytest

from app.model_routing.policy import Policy, Provider, ProviderKind, load_policy
from app.providers.anthropic import AnthropicAdapter
from app.providers.azure_openai import AZURE_API_VERSION, AzureOpenAIAdapter
from app.providers.base import Adapter, ChatMessage, ChatRequest
from app.providers.factory import build_adapters
from app.providers.openai_compatible import OPENAI_BASE_URL, OpenAICompatibleAdapter

from .conftest import ScriptedTransport, local_ok

KEY_ENV = "ANTHROPIC_API_KEY_EU"
UNSET_ENV = "ANTHROPIC_API_KEY_NOWHERE"
EMPTY_ENV = "ANTHROPIC_API_KEY_EMPTY"
OPENAI_ENV = "OPENAI_API_KEY"
OPENAI_UNSET_ENV = "OPENAI_API_KEY_NOWHERE"
OPENAI_EMPTY_ENV = "OPENAI_API_KEY_EMPTY"

POLICY = Policy(
    providers={
        "anthropic-eu": Provider(kind=ProviderKind.ANTHROPIC, key_env=KEY_ENV),
        "anthropic-unset": Provider(kind=ProviderKind.ANTHROPIC, key_env=UNSET_ENV),
        "anthropic-empty": Provider(kind=ProviderKind.ANTHROPIC, key_env=EMPTY_ENV),
        "anthropic-nokeyenv": Provider(kind=ProviderKind.ANTHROPIC),
        # Enabled=False is the ONLY difference from `anthropic-eu`: its key is
        # set in ENVIRON, so nothing else can explain its absence.
        "anthropic-off": Provider(kind=ProviderKind.ANTHROPIC, key_env=KEY_ENV, enabled=False),
        # Kind is the only difference: enabled, and its key is set too.
        "openai-eu": Provider(kind=ProviderKind.OPENAI, key_env=OPENAI_ENV),
    },
    aliases={},
    tenants={},
)

ENVIRON = {
    KEY_ENV: "sk-ant-eu-not-a-real-key",
    EMPTY_ENV: "",
    OPENAI_ENV: "sk-openai-not-a-real-key",
    OPENAI_EMPTY_ENV: "",
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


def test_only_the_usable_providers_of_the_base_policy_are_built() -> None:
    """Catches a factory that builds an adapter with an empty key (or with `None`), which
    would look healthy to the resolver and 401 on the first real request.

    `openai-eu` joins `anthropic-eu` from an earlier change on: it is enabled and its key is set,
    so from that plan on the ONLY reason a provider is missing here is the control its own
    row names."""
    adapters = build_adapters(POLICY, ENVIRON)

    assert set(adapters) == {"anthropic-eu", "openai-eu"}


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


def test_an_openai_provider_is_not_built_as_anthropic() -> None:
    """Catches a factory that builds an `AnthropicAdapter` for every enabled provider with
    a key, which would send OpenAI traffic to Anthropic's endpoint with an OpenAI key."""
    adapters = build_adapters(POLICY, ENVIRON)

    built = adapters["openai-eu"]
    assert not isinstance(built, AnthropicAdapter)
    assert built.name == "openai-eu"


def test_an_empty_policy_builds_nothing_and_says_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Catches a factory with a hardcoded default provider: an empty policy must produce an
    empty mapping, not a real Anthropic adapter nobody configured."""
    empty = Policy(providers={}, aliases={}, tenants={})

    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(empty, ENVIRON)

    assert adapters == {}
    assert caplog.records == []


# --- the openai_compatible kind -----------------------------------
#
# A policy of its own rather than more rows in `POLICY` above: the cases there
# assert the whole built set (`set(adapters) == {"anthropic-eu", "openai-eu"}`),
# so adding a provider to it would edit an existing test's meaning. Every provider
# below differs from the built one in exactly one attribute - the `enabled` flag,
# the `key_env`, the `base_url` or the kind - so a row going green can only be the
# control it names.

LOCAL_KEY_ENV = "VLLM_API_KEY_NOWHERE"

LOCAL_POLICY = Policy(
    providers={
        # The shipped `local`: enabled, a base url, and no `key_env` at all.
        "local": Provider(kind=ProviderKind.OPENAI_COMPATIBLE, base_url="http://llama:8082/v1"),
        # The shipped `vllm`, enabled here: the same kind on a second endpoint.
        "vllm": Provider(kind=ProviderKind.OPENAI_COMPATIBLE, base_url="http://vllm:8000/v1"),
        # `enabled=False` is the ONLY difference from `local`.
        "local-off": Provider(
            kind=ProviderKind.OPENAI_COMPATIBLE,
            base_url="http://llama:8082/v1",
            enabled=False,
        ),
        # A declared key that is not in the environment is the only difference.
        "local-keyed": Provider(
            kind=ProviderKind.OPENAI_COMPATIBLE,
            base_url="http://llama:8082/v1",
            key_env=LOCAL_KEY_ENV,
        ),
        # No base url is the only difference: there is no endpoint to call.
        "local-nourl": Provider(kind=ProviderKind.OPENAI_COMPATIBLE),
        # A different kind on the same policy, with its key set.
        "anthropic-eu": Provider(kind=ProviderKind.ANTHROPIC, key_env=KEY_ENV),
    },
    aliases={},
    tenants={},
)


def test_an_openai_compatible_provider_with_no_key_env_is_built() -> None:
    """Catches a factory that demands a key of every provider (decision B): `local` and
    `vllm` sit on a private network and declare none, so a key check applied to them drops
    the local route and every `pii` request stays a 503."""
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
    provider is that it is the box we run. A silent drop is caught too - it looks
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


# --- the openai kind ----------------------------------------------
#
# A third policy, for the same reason `LOCAL_POLICY` is a second one: the cases
# above assert whole built sets. Every provider here differs from the built one
# in exactly one attribute - the `key_env`, the value that variable holds, or the
# `base_url` - so a row going green can only be the control it names.
#
# The provider names are deliberately neither `openai` (the kind, and the shipped
# policy's own name for this provider) nor a vendor name, so an adapter keyed or
# priced by the kind rather than by the policy name is visible.

OPENAI_PROVIDER = "openai-us"
OPENAI_PROVIDER_BASE_URL = "https://api.openai.test/v1"

OPENAI_POLICY = Policy(
    providers={
        # Enabled, a key in ENVIRON, and a base url of its own.
        OPENAI_PROVIDER: Provider(
            kind=ProviderKind.OPENAI,
            base_url=OPENAI_PROVIDER_BASE_URL,
            key_env=OPENAI_ENV,
        ),
        # No base url is the ONLY difference: the vendor default is what it gets.
        "openai-default-url": Provider(kind=ProviderKind.OPENAI, key_env=OPENAI_ENV),
        # A declared key that is not in the environment is the only difference.
        "openai-unset": Provider(
            kind=ProviderKind.OPENAI,
            base_url=OPENAI_PROVIDER_BASE_URL,
            key_env=OPENAI_UNSET_ENV,
        ),
        # A declared key set to the empty string is the only difference.
        "openai-empty": Provider(
            kind=ProviderKind.OPENAI,
            base_url=OPENAI_PROVIDER_BASE_URL,
            key_env=OPENAI_EMPTY_ENV,
        ),
        # No `key_env` at all is the only difference: legal for the
        # `openai_compatible` kind, a drop for this one.
        "openai-nokeyenv": Provider(kind=ProviderKind.OPENAI, base_url=OPENAI_PROVIDER_BASE_URL),
    },
    aliases={},
    tenants={},
)

OPENAI_REQUEST = ChatRequest(
    model="gpt-4o",
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)


def test_an_openai_provider_with_its_key_set_is_built() -> None:
    """Catches a factory with no `openai` branch at all (the provider is silently absent,
    and every alias naming it loses a candidate) and one that keys or names the adapter by
    the KIND: the ledger, the metric label and the resolver all address it as `openai-us`,
    the policy's own name for it."""
    adapters = build_adapters(OPENAI_POLICY, ENVIRON)

    assert OPENAI_PROVIDER in adapters
    built = adapters[OPENAI_PROVIDER]
    assert isinstance(built, Adapter)
    assert built.name == "openai-us"


def test_an_openai_adapter_sends_the_key_only_as_a_bearer_header() -> None:
    """Catches a branch that hands the vendor its key any way other than one bearer header
    - the neighbouring `x-api-key`, which OpenAI answers 401 to, or a base url or query
    parameter carrying the key, which writes it into every proxy log on the path.

    The adapter is the one `build_adapters` built, with only its transport swapped, so the
    headers under assertion are the factory's own rather than a second set written here."""
    adapters = build_adapters(OPENAI_POLICY, ENVIRON)
    built = adapters[OPENAI_PROVIDER]
    assert isinstance(built, OpenAICompatibleAdapter)

    scripted = ScriptedTransport([local_ok("plain")])
    factory_client = built.client
    built.http.client = httpx.Client(
        base_url=factory_client.base_url,
        headers=factory_client.headers,
        timeout=factory_client.timeout,
        transport=scripted.transport,
    )
    factory_client.close()

    built.chat(OPENAI_REQUEST)

    key = ENVIRON[OPENAI_ENV]
    sent = scripted.requests[0]
    assert sent.headers.get_list("authorization") == [f"Bearer {key}"]
    assert [name for name, value in sent.headers.multi_items() if key in value] == ["authorization"]
    assert key not in str(sent.url)
    assert key not in sent.content.decode("utf-8")


OPENAI_DROPPED_CASES: list[tuple[str, str, str | None]] = [
    ("the variable is not in the environment", "openai-unset", OPENAI_UNSET_ENV),
    ("the variable is set but empty", "openai-empty", OPENAI_EMPTY_ENV),
    ("no key_env is configured at all", "openai-nokeyenv", None),
]


@pytest.mark.parametrize(
    ("case", "provider", "variable"),
    OPENAI_DROPPED_CASES,
    ids=[case for case, *_ in OPENAI_DROPPED_CASES],
)
def test_an_openai_provider_without_a_usable_key_is_dropped_with_a_warning(
    case: str, provider: str, variable: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches an `openai` branch copied from the `openai_compatible` one, where a missing
    key is legal: this kind has no private network to sit on, so an adapter built without a
    key is a 401 the tenant paid for, and the no-`key_env` row is the one that separates
    the two kinds. The empty-string row catches a check written as `variable in environ`,
    and each row asserts a warning, so a branch that drops the provider in silence - which
    is what an absent branch does - fails here too."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(OPENAI_POLICY, ENVIRON)

    assert provider not in adapters, case
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and provider in record.getMessage()
    ]
    assert len(warnings) == 1, f"{case}: expected exactly one warning naming {provider}"
    if variable is not None:
        assert variable in warnings[0], f"{case}: the warning must name the variable to set"


def test_an_openai_provider_without_a_base_url_uses_the_vendor_default() -> None:
    """Catches a branch that drops an `openai` provider naming no base url (the vendor has
    exactly one public endpoint, so there is nothing to guess and no reason to lose the
    candidate) and one that falls back to some other host - `localhost`, or the Anthropic
    base url the branch next door uses - which would send a real key off to it."""
    adapters = build_adapters(OPENAI_POLICY, ENVIRON)

    assert "openai-default-url" in adapters
    built = adapters["openai-default-url"]
    assert isinstance(built, OpenAICompatibleAdapter)
    assert OPENAI_BASE_URL == "https://api.openai.com/v1"
    # `rstrip`: httpx normalises a base url to a trailing slash, which is the one
    # difference between the constant and the client's own.
    assert str(built.client.base_url).rstrip("/") == "https://api.openai.com/v1"
    # The policy's own base url is still what a provider naming one gets.
    policy_built = adapters[OPENAI_PROVIDER]
    assert isinstance(policy_built, OpenAICompatibleAdapter)
    assert str(policy_built.client.base_url).rstrip("/") == OPENAI_PROVIDER_BASE_URL


def test_openai_price_key_is_the_policy_provider_name_slash_model() -> None:
    """Catches an adapter priced by the vendor or by the kind (`openai/gpt-4o` for every
    account): the ledger prices against this string, so two accounts on one vendor - or a
    US and an EU deployment - would be billed into one row and one metric label."""
    adapters = build_adapters(OPENAI_POLICY, ENVIRON)

    assert adapters[OPENAI_PROVIDER].price_key("gpt-4o") == "openai-us/gpt-4o"


# --- the azure_openai kind ----------------------------------------
#
# A fourth policy, for the same reason `OPENAI_POLICY` is a third one: the cases
# above assert whole built sets. Every provider here differs from the built one in
# exactly one attribute - the `key_env`, the `endpoint_env`, the value one of those
# variables holds, or the presence of a second endpoint source - so a row going
# green can only be the control it names.
#
# The provider name is neither the kind nor a vendor name, so an adapter keyed,
# named or priced by the kind rather than by the policy name is visible.

AZURE_PROVIDER = "azure-eu-sweden"
AZURE_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
AZURE_ENDPOINT_UNSET_ENV = "AZURE_OPENAI_ENDPOINT_NOWHERE"
AZURE_ENDPOINT_EMPTY_ENV = "AZURE_OPENAI_ENDPOINT_EMPTY"
AZURE_KEY_ENV = "AZURE_OPENAI_KEY"
AZURE_KEY_UNSET_ENV = "AZURE_OPENAI_KEY_NOWHERE"
AZURE_KEY_EMPTY_ENV = "AZURE_OPENAI_KEY_EMPTY"
AZURE_API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"

AZURE_ENDPOINT = "https://ops-console-eu.openai.azure.test"
AZURE_DEPLOYMENT = "gpt-4o-eu"

AZURE_POLICY = Policy(
    providers={
        # Enabled, an endpoint variable and a key variable, both set in AZURE_ENVIRON.
        AZURE_PROVIDER: Provider(
            kind=ProviderKind.AZURE_OPENAI,
            endpoint_env=AZURE_ENDPOINT_ENV,
            key_env=AZURE_KEY_ENV,
        ),
        # A declared key variable that is not in the environment is the only difference.
        "azure-key-unset": Provider(
            kind=ProviderKind.AZURE_OPENAI,
            endpoint_env=AZURE_ENDPOINT_ENV,
            key_env=AZURE_KEY_UNSET_ENV,
        ),
        # A declared key variable set to the empty string is the only difference.
        "azure-key-empty": Provider(
            kind=ProviderKind.AZURE_OPENAI,
            endpoint_env=AZURE_ENDPOINT_ENV,
            key_env=AZURE_KEY_EMPTY_ENV,
        ),
        # No `key_env` at all is the only difference: legal for the
        # `openai_compatible` kind on a private network, a drop for this one.
        "azure-nokeyenv": Provider(kind=ProviderKind.AZURE_OPENAI, endpoint_env=AZURE_ENDPOINT_ENV),
        # A declared endpoint variable that is not in the environment is the only
        # difference.
        "azure-endpoint-unset": Provider(
            kind=ProviderKind.AZURE_OPENAI,
            endpoint_env=AZURE_ENDPOINT_UNSET_ENV,
            key_env=AZURE_KEY_ENV,
        ),
        # A declared endpoint variable set to the empty string is the only difference.
        "azure-endpoint-empty": Provider(
            kind=ProviderKind.AZURE_OPENAI,
            endpoint_env=AZURE_ENDPOINT_EMPTY_ENV,
            key_env=AZURE_KEY_ENV,
        ),
        # No `endpoint_env` at all is the only difference: there is no vendor
        # default for a per-resource host.
        "azure-noendpointenv": Provider(kind=ProviderKind.AZURE_OPENAI, key_env=AZURE_KEY_ENV),
        # A second endpoint source is the only difference: everything else is the
        # built row, and both `endpoint_env` and `base_url` name a usable host.
        "azure-two-sources": Provider(
            kind=ProviderKind.AZURE_OPENAI,
            endpoint_env=AZURE_ENDPOINT_ENV,
            base_url="https://ops-console-us.openai.azure.test",
            key_env=AZURE_KEY_ENV,
        ),
    },
    aliases={},
    tenants={},
)

AZURE_ENVIRON = {
    AZURE_ENDPOINT_ENV: AZURE_ENDPOINT,
    AZURE_ENDPOINT_EMPTY_ENV: "",
    AZURE_KEY_ENV: "azure-not-a-real-key",
    AZURE_KEY_EMPTY_ENV: "",
}

AZURE_REQUEST = ChatRequest(
    model=AZURE_DEPLOYMENT,
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)


def scripted_chat_url(built: OpenAICompatibleAdapter) -> str:
    """Run one chat call on a FACTORY-built adapter over a scripted transport, and return
    the whole URL it addressed.

    Only the transport is swapped; the base url, the headers and the timeouts are the
    ones `build_adapters` chose, so the URL under assertion is the factory's own rather
    than a second adapter constructed here.
    """
    scripted = ScriptedTransport([local_ok("plain")])
    factory_client = built.client
    built.http.client = httpx.Client(
        base_url=factory_client.base_url,
        headers=factory_client.headers,
        timeout=factory_client.timeout,
        transport=scripted.transport,
    )
    factory_client.close()

    built.chat(AZURE_REQUEST)

    assert scripted.request_count == 1
    return str(scripted.requests[0].url)


def test_an_azure_provider_with_its_endpoint_and_key_is_built() -> None:
    """Catches a factory with no `azure_openai` branch at all (the provider is silently
    absent, so enabling the shipped row would still build nothing and the promised
    config-only switch-on would need a code change), one that builds the parent class
    instead - which would send the key as a bearer to a path Azure 404s - and one that
    keys or names the adapter by the KIND: the ledger, the metric label and the resolver
    all address it as `azure-eu-sweden`, the policy's own name for it."""
    adapters = build_adapters(AZURE_POLICY, AZURE_ENVIRON)

    assert AZURE_PROVIDER in adapters
    built = adapters[AZURE_PROVIDER]
    assert isinstance(built, Adapter)
    assert isinstance(built, AzureOpenAIAdapter)
    assert built.name == "azure-eu-sweden"
    assert built.price_key(AZURE_DEPLOYMENT) == "azure-eu-sweden/gpt-4o-eu"


AZURE_KEY_CASES: list[tuple[str, str, str | None]] = [
    ("the variable is not in the environment", "azure-key-unset", AZURE_KEY_UNSET_ENV),
    ("the variable is set but empty", "azure-key-empty", AZURE_KEY_EMPTY_ENV),
    ("no key_env is configured at all", "azure-nokeyenv", None),
]


@pytest.mark.parametrize(
    ("case", "provider", "variable"),
    AZURE_KEY_CASES,
    ids=[case for case, *_ in AZURE_KEY_CASES],
)
def test_an_azure_provider_without_a_usable_key_is_dropped_with_a_warning(
    case: str, provider: str, variable: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches an azure branch copied from the `openai_compatible` one, where a missing key
    is legal: an Azure resource is reached over the internet, so a keyless adapter is a 401
    the tenant paid for, and the no-`key_env` row is what separates the two kinds. The
    empty-string row catches a check written as `variable in environ`, and each row asserts
    exactly one warning naming the provider, so a branch that drops it in silence - which
    is what an absent branch does - fails here too.

    Every row has a usable ENDPOINT, so the endpoint control cannot explain the drop."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(AZURE_POLICY, AZURE_ENVIRON)

    assert provider not in adapters, case
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and provider in record.getMessage()
    ]
    assert len(warnings) == 1, f"{case}: expected exactly one warning naming {provider}"
    if variable is not None:
        assert variable in warnings[0], f"{case}: the warning must name the variable to set"


AZURE_ENDPOINT_CASES: list[tuple[str, str, str | None]] = [
    (
        "the variable is not in the environment",
        "azure-endpoint-unset",
        AZURE_ENDPOINT_UNSET_ENV,
    ),
    ("the variable is set but empty", "azure-endpoint-empty", AZURE_ENDPOINT_EMPTY_ENV),
    ("no endpoint_env is configured at all", "azure-noendpointenv", None),
]


@pytest.mark.parametrize(
    ("case", "provider", "variable"),
    AZURE_ENDPOINT_CASES,
    ids=[case for case, *_ in AZURE_ENDPOINT_CASES],
)
def test_an_azure_provider_without_a_usable_endpoint_is_dropped_with_a_warning(
    case: str, provider: str, variable: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches a branch that falls back to a vendor default when no endpoint resolves -
    there is none to fall back to, because the host is per-resource, and guessing one
    would send a real Azure key to a host we do not own. Catches a branch that builds the
    adapter with an empty base url, which httpx turns into a relative request nobody can
    read, and a silent drop, which looks to an operator like a routing bug.

    Every row here has a usable KEY, so the key control cannot explain the drop."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(AZURE_POLICY, AZURE_ENVIRON)

    assert provider not in adapters, case
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and provider in record.getMessage()
    ]
    assert len(warnings) == 1, f"{case}: expected exactly one warning naming {provider}"
    if variable is not None:
        assert variable in warnings[0], f"{case}: the warning must name the variable to set"


def test_an_azure_provider_naming_both_an_endpoint_env_and_a_base_url_is_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Catches decision C not implemented: with both sources present one of them quietly
    wins, and which one is undocumented - that is how a vendor key reaches the wrong host.
    Both of this row's sources are USABLE (the variable is set and the base url is a real
    string), so neither the key control nor the endpoint control can explain the drop: only
    the both-sources branch can."""
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(AZURE_POLICY, AZURE_ENVIRON)

    assert "azure-two-sources" not in adapters
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "azure-two-sources" in record.getMessage()
    ]
    assert len(warnings) == 1, "expected exactly one warning naming azure-two-sources"


def test_a_configured_api_version_overrides_the_default() -> None:
    """Catches an adapter that hardcodes `AZURE_API_VERSION` past its own parameter, and a
    factory branch that never reads `AZURE_OPENAI_API_VERSION`: a resource pinned to
    another api-version would then 400 or 404 on every call, and the only fix would be a
    code change - which is exactly what the shipped-disabled row promises it is not. The
    default is asserted in the same case, so an override that silently applies everywhere
    is caught as well."""
    override = "2025-01-01-preview"
    assert override != AZURE_API_VERSION

    default_built = build_adapters(AZURE_POLICY, AZURE_ENVIRON)[AZURE_PROVIDER]
    assert isinstance(default_built, OpenAICompatibleAdapter)
    assert scripted_chat_url(default_built) == (
        "https://ops-console-eu.openai.azure.test"
        "/openai/deployments/gpt-4o-eu/chat/completions?api-version=2024-10-21"
    )

    configured = {**AZURE_ENVIRON, AZURE_API_VERSION_ENV: override}
    overridden = build_adapters(AZURE_POLICY, configured)[AZURE_PROVIDER]
    assert isinstance(overridden, OpenAICompatibleAdapter)
    assert scripted_chat_url(overridden) == (
        "https://ops-console-eu.openai.azure.test"
        "/openai/deployments/gpt-4o-eu/chat/completions?api-version=2025-01-01-preview"
    )


def test_exactly_the_usable_azure_providers_are_built() -> None:
    """Catches every drop above being explained by something other than its own control:
    the built set is asserted whole, so a factory that dropped the usable row for want of
    a base url, or kept `azure-noendpointenv` with a guessed host, fails here as well as
    in its own row."""
    adapters = build_adapters(AZURE_POLICY, AZURE_ENVIRON)

    assert set(adapters) == {AZURE_PROVIDER}


# The policy file the product ships with, resolved from this file so the test does
# not depend on the working directory pytest was started from.
SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"
SHIPPED_PROVIDERS_DIR = SHIPPED_POLICY_PATH.parent / "providers"


AZURE_PROVIDER_FILE = """\
provider: azure
kind: azure_openai
endpoint_env: AZURE_OPENAI_ENDPOINT
key_env: AZURE_OPENAI_KEY
enabled: false
models:
  gpt-5-mini:
    prices:
      - effective_from: 2025-08-07
        input_usd_per_1m: "0.25"
        output_usd_per_1m: "2.00"
"""
"""The file `docs/azure.md` tells an operator to drop in, switched off. The shipped
folder carries no Azure file, so the case brings its own."""


def test_a_disabled_azure_provider_file_builds_nothing(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Catches an azure branch that ignores `enabled` - a dropped-in `azure.yaml` with
    `enabled: false` would then become a live adapter the moment an operator exported
    the two variables for something else, against a resource nobody has reviewed - and
    catches a branch that warns about a deliberate `enabled: false`, which trains
    operators to ignore the warning that matters.

    Both variables are set in the injected environment on purpose: without them "absent"
    would be explained by a missing credential and the `enabled` check could be deleted
    with this test still green."""
    folder = tmp_path / "providers"
    shutil.copytree(SHIPPED_PROVIDERS_DIR, folder)
    (folder / "azure.yaml").write_text(AZURE_PROVIDER_FILE, encoding="utf-8")
    policy = load_policy(SHIPPED_POLICY_PATH, folder)
    shipped = policy.providers["azure"]
    assert shipped.kind is ProviderKind.AZURE_OPENAI
    assert shipped.enabled is False

    environ = {
        AZURE_ENDPOINT_ENV: AZURE_ENDPOINT,
        AZURE_KEY_ENV: "azure-not-a-real-key",
    }
    with caplog.at_level(logging.WARNING):
        adapters = build_adapters(policy, environ)

    assert "azure" not in adapters
    assert [
        record.getMessage() for record in caplog.records if "azure" in record.getMessage().lower()
    ] == []
