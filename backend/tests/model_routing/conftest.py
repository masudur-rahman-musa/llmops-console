"""Fixtures for the routing policy and residency resolver tests.

Everything here is in-process: a Pydantic model tree and a YAML file under
`tmp_path`. There is no backing service, so a project rule (session-scoped
fixtures, per-test isolation by namespace) does not apply and function scope is
the correct choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import pytest
import yaml

from app.core.validation import MAX_IDENTIFIER_LENGTH
from app.model_routing.policy import (
    Alias,
    Candidate,
    DataClass,
    DataClassRule,
    Policy,
    Provider,
    ProviderKind,
    RateLimit,
    Residency,
    Tenant,
)

# The policy file the product ships with, resolved from this file so the tests
# do not depend on the working directory pytest was started from.
SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"


class PolicyWriter(Protocol):
    """Writes a policy document to `tmp_path` and returns the path."""

    def __call__(self, document: str | Mapping[str, object], name: str = ...) -> Path: ...


@pytest.fixture
def policy_file(tmp_path: Path) -> PolicyWriter:
    """Write a policy file, either as raw text or as a document to be dumped.

    Raw text is needed for the documents that must never reach the schema at
    all: broken YAML and a python object tag.
    """

    def write(document: str | Mapping[str, object], name: str = "routing.yaml") -> Path:
        if isinstance(document, str):
            text = document
        else:
            text = yaml.safe_dump(dict(document), sort_keys=False)
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return write


def valid_document(rpm: int = 60) -> dict[str, object]:
    """A minimal policy document that loads clean.

    `rpm` is the single value the reload test changes, so a swapped snapshot is
    visible without rewriting the shape of the file.
    """
    return {
        "providers": {
            "anthropic-eu": {"kind": "anthropic", "region": "eu", "key_env": "ANTHROPIC_API_KEY"},
            "openai-us": {"kind": "openai", "region": "us", "key_env": "OPENAI_API_KEY"},
            "local-self": {
                "kind": "openai_compatible",
                "region": "self",
                "base_url": "http://localhost:8081/v1",
            },
        },
        "eu_regions": ["eu", "self"],
        "aliases": {
            "chat": {
                "candidates": [
                    {"provider": "anthropic-eu", "model": "claude-sonnet-4-5"},
                    {"provider": "openai-us", "model": "gpt-4o-mini"},
                ]
            },
            "judge": {"candidates": [{"provider": "local-self", "model": "qwen2.5-7b-instruct"}]},
        },
        "tenants": {
            "acme": {
                "residency": "eu",
                "data_class_rules": {
                    "public": {"allow_regions": ["us", "eu", "self"]},
                    "internal": {"allow_regions": ["eu", "self"]},
                    "confidential": {"allow_regions": ["self"]},
                    "pii": {"allow_regions": ["self"], "redact": True},
                },
                "budget_usd_month": "100.00",
                "rate_limit": {"rpm": rpm, "tpm": 100000},
                "overrides": {"chat": [{"provider": "local-self", "model": "qwen2.5-7b-instruct"}]},
            }
        },
    }


def make_policy(
    *,
    providers: Mapping[str, Provider] | None = None,
    eu_regions: tuple[str, ...] = ("eu", "self"),
    aliases: Mapping[str, Alias] | None = None,
    tenants: Mapping[str, Tenant] | None = None,
) -> Policy:
    """Build a `Policy` straight from the models, skipping YAML entirely."""
    return Policy(
        providers=providers if providers is not None else {},
        eu_regions=eu_regions,
        aliases=aliases if aliases is not None else {},
        tenants=tenants if tenants is not None else {},
    )


def make_tenant(
    *,
    residency: Residency,
    rules: Mapping[DataClass, tuple[str, ...]],
    overrides: Mapping[str, tuple[Candidate, ...]] | None = None,
) -> Tenant:
    return Tenant(
        residency=residency,
        data_class_rules={
            data_class: DataClassRule(allow_regions=regions)
            for data_class, regions in rules.items()
        },
        budget_usd_month=Decimal("100.00"),
        rate_limit=RateLimit(rpm=60, tpm=100_000),
        overrides=overrides if overrides is not None else {},
    )


PROVIDERS: Mapping[str, Provider] = {
    "eu-a": Provider(kind=ProviderKind.ANTHROPIC, region="eu"),
    "eu-b": Provider(kind=ProviderKind.OPENAI, region="eu"),
    "eu-off": Provider(kind=ProviderKind.OPENAI, region="eu", enabled=False),
    "us-a": Provider(kind=ProviderKind.OPENAI, region="us"),
    "self-a": Provider(kind=ProviderKind.OPENAI_COMPATIBLE, region="self"),
}

ALIASES: Mapping[str, Alias] = {
    # Overridden by the `acme` tenant, so the default is never what resolves.
    "chat": Alias(candidates=(Candidate(provider="eu-a", model="m-eu-a"),)),
    # A us candidate first, to prove the eu clamp drops it rather than reorders.
    "mixed": Alias(
        candidates=(
            Candidate(provider="us-a", model="m-us-a"),
            Candidate(provider="eu-a", model="m-eu-a"),
        )
    ),
    "ordered": Alias(
        candidates=(
            Candidate(provider="eu-a", model="m-eu-a"),
            Candidate(provider="eu-b", model="m-eu-b"),
            Candidate(provider="self-a", model="m-self-a"),
        )
    ),
    "disabled-first": Alias(
        candidates=(
            Candidate(provider="eu-off", model="m-eu-off"),
            Candidate(provider="eu-a", model="m-eu-a"),
        )
    ),
    "us-only": Alias(candidates=(Candidate(provider="us-a", model="m-us-a"),)),
}


@pytest.fixture
def policy() -> Policy:
    """The in-memory policy every resolver case resolves against."""
    return make_policy(
        providers=PROVIDERS,
        aliases=ALIASES,
        tenants={
            "acme": make_tenant(
                residency=Residency.EU,
                rules={
                    DataClass.PUBLIC: ("us", "eu", "self"),
                    DataClass.INTERNAL: ("eu", "self"),
                    DataClass.CONFIDENTIAL: ("self",),
                    # No `pii` rule at all: deny by default must cover the gap.
                },
                overrides={"chat": (Candidate(provider="eu-b", model="m-override"),)},
            ),
            "globex": make_tenant(
                residency=Residency.ANY,
                rules={
                    DataClass.PUBLIC: ("us", "eu", "self"),
                    DataClass.INTERNAL: ("us", "eu"),
                },
            ),
            "narrow": make_tenant(
                residency=Residency.EU,
                # An eu tenant allowed only `us`: the clamp empties this.
                rules={DataClass.PUBLIC: ("us",)},
            ),
        },
    )


class DocFactory(Protocol):
    """Builds the baseline valid policy document, varying one pinned value."""

    def __call__(self, rpm: int = ...) -> dict[str, object]: ...


@pytest.fixture
def valid_doc() -> DocFactory:
    return valid_document


@pytest.fixture
def boundary_policy() -> Policy:
    """A policy whose tenant and alias keys are exactly the maximum length.

    Used for the inclusive boundary row: an identifier of exactly
    `MAX_IDENTIFIER_LENGTH` characters must resolve, not be rejected.
    """
    name = "a" * MAX_IDENTIFIER_LENGTH
    return make_policy(
        providers={"eu-a": Provider(kind=ProviderKind.ANTHROPIC, region="eu")},
        aliases={name: Alias(candidates=(Candidate(provider="eu-a", model="m-eu-a"),))},
        tenants={
            name: make_tenant(residency=Residency.EU, rules={DataClass.PUBLIC: ("eu", "self")})
        },
    )


@pytest.fixture
def shipped_policy_path() -> Path:
    return SHIPPED_POLICY_PATH
