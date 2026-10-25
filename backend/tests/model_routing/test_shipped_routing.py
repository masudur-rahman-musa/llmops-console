"""The shipped `routing.yaml` read together with the shipped provider folder.

The file is a test case: every tenant, data class and alias is swept, so a later
edit that breaks the order of providers fails here and not in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    """Catches a swapped model: the cheap pair serves `agent-default` and `judge`,
    the strong pair serves `agent-strong`."""
    pairs = {
        alias: tuple((c.provider, c.model) for c in policy.aliases[alias].candidates)
        for alias in CHAT_ALIASES
    }

    cheap = (("anthropic", "claude-haiku-5-5"), ("openai", "gpt-5-mini"))
    assert pairs["agent-default"] == cheap
    assert pairs["judge"] == cheap
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
