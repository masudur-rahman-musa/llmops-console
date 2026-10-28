"""`infra/load/routing.load.yaml`: the whole routing policy the gateway
loads during the k6 run.

Loaded through the real `opsconsole.routing.policy.load_policy`, so the file
is validated the same way the gateway validates it at start. No backing service.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.model_routing.policy import Policy, load_policy

REPO_ROOT = Path(__file__).resolve().parents[3]
LOAD_POLICY = REPO_ROOT / "infra" / "load" / "routing.load.yaml"
LOAD_PROVIDERS = REPO_ROOT / "infra" / "load" / "providers"

MOCK_BASE_URL = "http://mock-provider:8090/v1"


@pytest.fixture(scope="module")
def policy() -> Policy:
    return load_policy(LOAD_POLICY, LOAD_PROVIDERS)


def test_the_load_policy_is_a_whole_document_that_loads() -> None:
    """Catches a partial overlay in the style of `routing.ci.yaml`,
    which `load_policy` refuses because `providers`, `aliases` or `tenants` is
    missing."""
    load_policy(LOAD_POLICY, LOAD_PROVIDERS)


def test_only_the_two_mock_backed_providers_are_enabled(policy: Policy) -> None:
    """Catches a real provider (anthropic, openai) left enabled, which
    would put paid or unreachable traffic in the load run."""
    enabled = {name for name, provider in policy.providers.items() if provider.enabled}
    assert enabled == {"mock_chat", "local_embed"}


@pytest.mark.parametrize("provider", ["mock_chat", "local_embed"])
def test_both_enabled_providers_point_at_the_mock(policy: Policy, provider: str) -> None:
    """Catches a provider still pointing at `llama` or `llama-embed`,
    which does not exist in the load stack."""
    assert policy.providers[provider].base_url == MOCK_BASE_URL


def test_the_load_tenant_has_the_load_sized_limits_and_budget(
    policy: Policy,
) -> None:
    """Catches the load tenant inheriting a real tenant's 60 rpm or
    15.00 budget, which would 429 or 402 the run long before 50 RPS."""
    tenant = policy.tenants["load"]
    assert tenant.rate_limit.rpm == 6000
    assert tenant.rate_limit.tpm == 6000000
    assert tenant.budget_usd_month == Decimal("100000.00")


@pytest.mark.parametrize("alias", ["agent-default", "embed"])
def test_the_aliases_k6_sends_exist(policy: Policy, alias: str) -> None:
    """Catches a load policy without the alias `gateway.js` names,
    which would answer every iteration with an unknown-alias error."""
    assert alias in policy.aliases
