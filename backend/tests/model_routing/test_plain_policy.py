"""The routing policy after residency is removed.

Providers have no region, tenants have no residency setting, a data class rule
holds `redact` and nothing else, and the resolver never answers 451. Every case
here is in-process: the shipped `routing.yaml`, a YAML file under `tmp_path`, or a
policy built straight from the models.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.core.errors import TenantMismatch
from app.model_routing.errors import AllProvidersUnavailable, PolicyInvalid
from app.model_routing.policy import (
    Alias,
    Candidate,
    DataClass,
    DataClassRule,
    Policy,
    Provider,
    ProviderKind,
    RateLimit,
    Tenant,
    load_policy,
)
from app.model_routing.resolver import resolve

from .conftest import SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR

REMOVED_KEYS = ("region", "residency", "eu_regions", "allow_regions")
REDACTED_CLASSES = {DataClass.CONFIDENTIAL, DataClass.PII}
PLAIN_TENANTS = ("nordlicht", "cuan", "evals")


def test_the_shipped_policy_names_no_region_or_residency() -> None:
    """Catches a shipped `routing.yaml` that still carries a `region:`, a `residency:`, an
    `eu_regions` list or an `allow_regions` rule, or a comment explaining one: the file is
    what a reader opens first, and a leftover word there says the rule still exists."""
    text = SHIPPED_POLICY_PATH.read_text(encoding="utf-8").lower()

    for key in REMOVED_KEYS:
        assert key not in text, f"routing.yaml still names {key!r}"


def test_every_tenant_alias_and_data_class_resolves() -> None:
    """Catches a policy rewritten plain but left with a candidate list that is empty or all
    disabled for some triple: the old file made demo tenants fail with a 451, and the point
    of removing the rule is that no triple in the shipped file is refused any more."""
    policy = load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR)
    checked = 0

    for tenant in policy.tenants:
        for alias in policy.aliases:
            for data_class in DataClass:
                found = resolve(policy, tenant, alias, data_class.value)
                assert len(found) >= 1, (tenant, alias, data_class)
                checked += 1

    assert checked == len(policy.tenants) * len(policy.aliases) * len(DataClass)
    assert checked > 0


# Where each removed key lives, so the key is added back in the one place the old
# schema read it. A provider's `region` now goes in that provider's own file.
def _add_region(document: dict[str, Any], folder: Path) -> None:
    path = folder / "anthropic.yaml"
    provider = yaml.safe_load(path.read_text(encoding="utf-8"))
    provider["region"] = "eu"
    path.write_text(yaml.safe_dump(provider, sort_keys=False), encoding="utf-8")


def _add_residency(document: dict[str, Any], folder: Path) -> None:
    next(iter(document["tenants"].values()))["residency"] = "eu"


def _add_eu_regions(document: dict[str, Any], folder: Path) -> None:
    document["eu_regions"] = ["eu", "self"]


def _add_allow_regions(document: dict[str, Any], folder: Path) -> None:
    tenant = next(iter(document["tenants"].values()))
    next(iter(tenant["data_class_rules"].values()))["allow_regions"] = ["eu"]


ADDERS: dict[str, Callable[[dict[str, Any], Path], None]] = {
    "region": _add_region,
    "residency": _add_residency,
    "eu_regions": _add_eu_regions,
    "allow_regions": _add_allow_regions,
}


@pytest.mark.parametrize("key", REMOVED_KEYS)
def test_a_removed_key_is_refused_by_name(key: str, tmp_path: Path) -> None:
    """Catches a schema that quietly ignores the old keys: an old policy file would then
    half-load, with its region rules read as nothing, and look as if it still protected
    something. The control loads the SAME document without the key, so the key alone is
    what flips the result and the error must name it."""
    document = yaml.safe_load(SHIPPED_POLICY_PATH.read_text(encoding="utf-8"))
    for stale in REMOVED_KEYS:  # strip whatever the file still carries, then add back one
        _strip(document, stale)
    folder = tmp_path / "providers"
    shutil.copytree(SHIPPED_PROVIDERS_DIR, folder)
    clean = tmp_path / "clean.yaml"
    clean.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    load_policy(clean, folder)

    ADDERS[key](document, folder)
    dirty = tmp_path / "dirty.yaml"
    dirty.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(PolicyInvalid, match=re.escape(key)):
        load_policy(dirty, folder)


def _strip(node: object, key: str) -> None:
    """Remove `key` from every mapping under `node`."""
    if isinstance(node, dict):
        node.pop(key, None)
        for child in node.values():
            _strip(child, key)
    elif isinstance(node, list):
        for child in node:
            _strip(child, key)


def _plain_policy(*, enabled: bool) -> Policy:
    return Policy(
        providers={"only": Provider(kind=ProviderKind.OPENAI, enabled=enabled)},
        aliases={"chat": Alias(candidates=(Candidate(provider="only", model="m-only"),))},
        tenants={
            "acme": Tenant(
                data_class_rules={DataClass.PUBLIC: DataClassRule()},
                budget_usd_month=Decimal("100.00"),
                rate_limit=RateLimit(rpm=60, tpm=100_000),
            )
        },
    )


def test_an_unknown_tenant_is_tenant_mismatch() -> None:
    """Catches an unknown tenant that still raises the old 451, or that silently falls
    back to the alias default: the policy does not carry `ghost`, so it is a 403
    `TENANT_MISMATCH`. The control resolves the known tenant on the same policy."""
    policy = _plain_policy(enabled=True)
    assert [r.provider for r in resolve(policy, "acme", "chat", "public")] == ["only"]

    with pytest.raises(TenantMismatch) as excinfo:
        resolve(policy, "ghost", "chat", "public")

    assert excinfo.value.code == "TENANT_MISMATCH"
    assert excinfo.value.http_status == 403


def test_no_enabled_candidate_is_all_providers_failed() -> None:
    """Catches an empty candidate list that still raises a residency refusal, or one that
    returns an empty tuple: with every provider turned off the answer is the 503
    `ALL_PROVIDERS_FAILED`. The control enables the provider on an otherwise identical policy."""
    assert len(resolve(_plain_policy(enabled=True), "acme", "chat", "public")) == 1

    with pytest.raises(AllProvidersUnavailable) as excinfo:
        resolve(_plain_policy(enabled=False), "acme", "chat", "public")

    assert excinfo.value.code == "ALL_PROVIDERS_FAILED"
    assert excinfo.value.http_status == 503


@pytest.mark.parametrize("tenant", PLAIN_TENANTS)
def test_a_rule_holds_only_redact(tenant: str) -> None:
    """Catches a rule that keeps a leftover field, or a redact flag on the wrong class:
    data classes decide redaction and nothing else, so `confidential` and `pii` redact,
    `public` and `internal` do not, for every shipped tenant."""
    assert set(DataClassRule.model_fields) == {"redact"}

    rules = load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR).tenants[tenant].data_class_rules

    assert set(rules) == set(DataClass)
    for data_class, rule in rules.items():
        assert rule.redact is (data_class in REDACTED_CLASSES), (tenant, data_class)
