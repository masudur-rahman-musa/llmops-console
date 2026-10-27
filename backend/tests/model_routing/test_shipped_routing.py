"""The shipped `routing.yaml` read together with the shipped provider folder.

The file is a test case: every tenant, data class and alias is swept, so a later
edit that breaks the order of providers fails here and not in production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import DataClass, Policy, load_policy, reachable_price_keys
from app.model_routing.provider_files import build_price_table, load_provider_files
from app.model_routing.resolver import resolve

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
SHIPPED_PROVIDERS_DIR = CONFIG_DIR / "providers"
SHIPPED_POLICY_PATH = CONFIG_DIR / "routing.yaml"

CHAT_ALIASES = ("agent-default", "agent-strong", "judge")


@pytest.fixture(scope="module")
def policy() -> Policy:
    return load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR)


@pytest.mark.parametrize("alias", CHAT_ALIASES)
def test_chat_aliases_list_anthropic_then_openai(policy: Policy, alias: str) -> None:
    """Catches an alias that lists a third provider, the wrong order, or a tenant
    override that puts openai first: for every tenant and data class the answer
    is anthropic, then openai, so openai is the fallback when anthropic is down."""
    assert policy.tenants, "the sweep needs tenants to be a test"
    for tenant in policy.tenants:
        for data_class in DataClass:
            resolved = resolve(policy, tenant, alias, data_class.value)
            assert tuple(item.provider for item in resolved) == ("anthropic", "openai"), (
                tenant,
                alias,
                data_class,
            )


def test_each_chat_alias_names_its_models(policy: Policy) -> None:
    """Catches a swapped model: the cheap pair serves `agent-default`, the judge runs on
    Sonnet with the cheap OpenAI model behind it, the strong pair serves `agent-strong`."""
    pairs = {
        alias: tuple((c.provider, c.model) for c in policy.aliases[alias].candidates)
        for alias in CHAT_ALIASES
    }

    cheap = (("anthropic", "claude-haiku-5-5"), ("openai", "gpt-5-mini"))
    assert pairs["agent-default"] == cheap
    assert pairs["judge"] == (
        ("anthropic", "claude-sonnet-5-5"),
        ("openai", "gpt-5-mini"),
    )
    assert pairs["agent-strong"] == (
        ("anthropic", "claude-sonnet-5-5"),
        ("openai", "gpt-5"),
    )


def test_embed_lists_only_the_local_embedding_server(policy: Policy) -> None:
    """Catches a second embedding model on `embed`: a vector from another model sits
    in a different space and would quietly break search over stored vectors."""
    resolved = resolve(policy, "cuan", "embed", DataClass.INTERNAL.value)

    assert [(item.provider, item.model) for item in resolved] == [
        ("local_embed", "bge-small-en-v1.5")
    ]
    assert tuple((c.provider, c.model) for c in policy.aliases["embed"].candidates) == (
        ("local_embed", "bge-small-en-v1.5"),
    )


def test_every_routed_model_has_a_price_in_its_file(policy: Policy) -> None:
    """Catches a model added to an alias with no price in its provider file, which
    would fill the ledger with `unpriced` rows."""
    table = build_price_table(load_provider_files(SHIPPED_PROVIDERS_DIR))
    reachable = reachable_price_keys(policy)

    assert {"anthropic/claude-haiku-5-5", "openai/gpt-5-mini", "openai/gpt-5"} <= reachable
    assert reachable <= set(table.prices)


DEMO_TENANTS = ("nordlicht", "cuan", "evals")


def policy_with_trace_content(directory: Path, value: object, *, omit: bool = False) -> Policy:
    """Load a copy of the shipped policy where `evals` has the given `trace_content`.

    `omit` leaves the line out of the copy altogether.
    """
    document: dict[str, Any] = yaml.safe_load(SHIPPED_POLICY_PATH.read_text(encoding="utf-8"))
    if omit:
        document["tenants"]["evals"].pop("trace_content", None)
    else:
        document["tenants"]["evals"]["trace_content"] = value
    copy = directory / "routing-copy.yaml"
    copy.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_policy(copy, SHIPPED_PROVIDERS_DIR)


@pytest.mark.parametrize("tenant", DEMO_TENANTS)
def test_the_demo_tenants_send_trace_content(policy: Policy, tenant: str) -> None:
    """Catches a demo tenant left off in the shipped file (the Langfuse conversation view
    would stay empty for it), and a default of on, which would not show here."""
    assert policy.tenants[tenant].trace_content is True


def test_a_tenant_without_the_switch_is_off(tmp_path: Path) -> None:
    """Catches a default of on: a new tenant would send its conversations to Langfuse
    without anyone choosing it. A written `false` is loaded as false too."""
    assert (
        policy_with_trace_content(tmp_path, None, omit=True).tenants["evals"].trace_content is False
    )
    assert policy_with_trace_content(tmp_path, False).tenants["evals"].trace_content is False
    assert policy_with_trace_content(tmp_path, True).tenants["evals"].trace_content is True


@pytest.mark.parametrize("value", ["maybe", "yes", "true", 1, None], ids=repr)
def test_a_switch_that_is_not_a_bool_is_refused(tmp_path: Path, value: object) -> None:
    """Catches a switch that coerces text or numbers (`"yes"` or `1` turning a tenant on by
    typo), or a null read as off. A real bool on the same file loads first, so the refusal
    is the switch's own and not some other part of the copy being broken."""
    assert policy_with_trace_content(tmp_path, False).tenants["evals"].trace_content is False

    with pytest.raises(PolicyInvalid):
        policy_with_trace_content(tmp_path, value)


@pytest.mark.parametrize("tenant", DEMO_TENANTS)
def test_the_demo_tenants_follow_the_switch(policy: Policy, tenant: str) -> None:
    """Catches a demo tenant with `trace_public` written in the shipped file, which would
    pin it on or off whatever the gateway setting says."""
    assert policy.tenants[tenant].trace_public is None


def policy_with_trace_public(directory: Path, value: object) -> Policy:
    """Load a copy of the shipped policy where `evals` has the given `trace_public`."""
    document: dict[str, Any] = yaml.safe_load(SHIPPED_POLICY_PATH.read_text(encoding="utf-8"))
    document["tenants"]["evals"]["trace_public"] = value
    copy = directory / "routing-public-copy.yaml"
    copy.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_policy(copy, SHIPPED_PROVIDERS_DIR)


@pytest.mark.parametrize("value", ["yes", "true", 1], ids=repr)
def test_a_trace_public_that_is_not_a_bool_is_refused(tmp_path: Path, value: object) -> None:
    """Catches a field that coerces text or numbers (`"yes"` or `1` making a tenant's chats
    public by typo). Real bools load first on the same copy, so the refusal is this field's."""
    assert policy_with_trace_public(tmp_path, False).tenants["evals"].trace_public is False
    assert policy_with_trace_public(tmp_path, True).tenants["evals"].trace_public is True

    with pytest.raises(PolicyInvalid):
        policy_with_trace_public(tmp_path, value)
