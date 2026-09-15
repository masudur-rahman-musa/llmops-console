"""`project-evals/load/routing.load.yaml`: the whole routing policy the gateway
loads during the k6 run.

Loaded through the real `opsconsole.routing.policy.load_policy`, so the file
is validated the same way the gateway validates it at start. No backing service.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from opsconsole.routing.policy import Policy, Residency, load_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
LOAD_POLICY = REPO_ROOT / "project-evals" / "load" / "routing.load.yaml"

MOCK_BASE_URL = "http://mock-provider:8090/v1"


@pytest.fixture(scope="module")
def policy() -> Policy:
    return load_policy(LOAD_POLICY)


def test_the_load_policy_is_a_whole_document_that_loads() -> None:
    """Criterion 5: catches a partial overlay in the style of `routing.ci.yaml`,
    which `load_policy` refuses because `eu_regions`, `aliases` or `tenants` is
    missing."""
    load_policy(LOAD_POLICY)


def test_only_the_two_mock_backed_providers_are_enabled(policy: Policy) -> None:
    """Criterion 6: catches a real provider (anthropic, openai) left enabled, which
    would put paid or unreachable traffic in the load run."""
    enabled = {name for name, provider in policy.providers.items() if provider.enabled}
    assert enabled == {"local", "local_embed"}


@pytest.mark.parametrize("provider", ["local", "local_embed"])
def test_both_enabled_providers_point_at_the_mock(policy: Policy, provider: str) -> None:
    """Criterion 6: catches a provider still pointing at `llama` or `llama-embed`,
    which does not exist in the load stack."""
    assert policy.providers[provider].base_url == MOCK_BASE_URL


def test_the_load_tenant_is_eu_with_the_load_sized_limits_and_budget(policy: Policy) -> None:
    """Criterion 7: catches the load tenant inheriting a real tenant's 60 rpm or
    15.00 budget, which would 429 or 402 the run long before 50 RPS."""
    tenant = policy.tenants["load"]
    assert tenant.residency == Residency.EU
    assert tenant.rate_limit.rpm == 6000
    assert tenant.rate_limit.tpm == 6000000
    assert tenant.budget_usd_month == Decimal("100000.00")


@pytest.mark.parametrize("alias", ["agent-default", "embed"])
def test_the_aliases_k6_sends_exist(policy: Policy, alias: str) -> None:
    """Criterion 12: catches a load policy without the alias `gateway.js` names,
    which would answer every iteration with an unknown-alias error."""
    assert alias in policy.aliases
