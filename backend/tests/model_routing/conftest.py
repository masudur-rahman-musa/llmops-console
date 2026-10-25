"""Fixtures for the routing policy and resolver tests.

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
    Tenant,
)

# The policy file the product ships with, resolved from this file so the tests
# do not depend on the working directory pytest was started from.
SHIPPED_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"
SHIPPED_PROVIDERS_DIR = SHIPPED_POLICY_PATH.parent / "providers"


class PolicyWriter(Protocol):
    """Writes a policy document to `tmp_path` and returns the path."""

    def __call__(self, document: str | Mapping[str, object], name: str = ...) -> Path: ...


PLACEHOLDER_MODEL: dict[str, object] = {
    "placeholder-model": {
        "prices": [
            {"effective_from": "2025-01-01", "input_usd_per_1m": "0", "output_usd_per_1m": "0"}
        ]
    }
}
"""The one model each split-out provider file carries: a file needs at least one."""


def providers_dir_of(path: Path) -> Path:
    """The provider folder `policy_file` writes beside a routing file."""
    return path.parent / "providers"


def _split_providers(document: object, folder: Path) -> object:
    """Move a `providers:` block out into one file per provider, as the gateway wants.

    The cases here were written with providers inline; the folder is where they
    live now. A document with no such block leaves the folder holding one plain
    provider, so the folder is never what fails.
    """
    folder.mkdir(exist_ok=True)
    providers: object = None
    if isinstance(document, dict) and isinstance(document.get("providers"), dict):
        document = dict(document)
        providers = document.pop("providers")
    if not isinstance(providers, dict):
        providers = {"a": {"kind": "anthropic"}}
    for name, body in providers.items():
        file = {"provider": name, **(body if isinstance(body, dict) else {})}
        file.setdefault("models", PLACEHOLDER_MODEL)
        (folder / f"{name}.yaml").write_text(
            yaml.safe_dump(file, sort_keys=False), encoding="utf-8"
        )
    return document


@pytest.fixture
def policy_file(tmp_path: Path) -> PolicyWriter:
    """Write a policy file, either as raw text or as a document to be dumped.

    Raw text is needed for the documents that must never reach the schema at
    all: broken YAML and a python object tag. A `providers:` block is split out
    into `providers_dir_of(path)`; text that does not parse is written as is.
    """

    def write(document: str | Mapping[str, object], name: str = "routing.yaml") -> Path:
        path = tmp_path / name
        folder = providers_dir_of(path)
        parsed: object
        if isinstance(document, str):
            try:
                parsed = yaml.safe_load(document)
            except yaml.YAMLError:
                _split_providers(None, folder)
                path.write_text(document, encoding="utf-8")
                return path
            if not (isinstance(parsed, dict) and "providers" in parsed):
                _split_providers(None, folder)
                path.write_text(document, encoding="utf-8")
                return path
        else:
            parsed = dict(document)
        routing = _split_providers(parsed, folder)
        path.write_text(yaml.safe_dump(routing, sort_keys=False), encoding="utf-8")
        return path

    return write


def valid_document(rpm: int = 60) -> dict[str, object]:
    """A minimal policy document that loads clean.

    `rpm` is the single value the reload test changes, so a swapped snapshot is
    visible without rewriting the shape of the file.
    """
    return {
        "providers": {
            "anthropic-a": {"kind": "anthropic", "key_env": "ANTHROPIC_API_KEY"},
            "openai-a": {"kind": "openai", "key_env": "OPENAI_API_KEY"},
            "local-self": {
                "kind": "openai_compatible",
                "base_url": "http://localhost:8081/v1",
            },
        },
        "aliases": {
            "chat": {
                "candidates": [
                    {"provider": "anthropic-a", "model": "claude-sonnet-4-5"},
                    {"provider": "openai-a", "model": "gpt-4o-mini"},
                ]
            },
            "judge": {"candidates": [{"provider": "local-self", "model": "qwen2.5-7b-instruct"}]},
        },
        "tenants": {
            "acme": {
                "data_class_rules": {
                    "public": {},
                    "internal": {},
                    "confidential": {"redact": True},
                    "pii": {"redact": True},
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
    aliases: Mapping[str, Alias] | None = None,
    tenants: Mapping[str, Tenant] | None = None,
) -> Policy:
    """Build a `Policy` straight from the models, skipping YAML entirely."""
    return Policy(
        providers=providers if providers is not None else {},
        aliases=aliases if aliases is not None else {},
        tenants=tenants if tenants is not None else {},
    )


def make_tenant(
    *,
    rules: Mapping[DataClass, bool] | None = None,
    overrides: Mapping[str, tuple[Candidate, ...]] | None = None,
) -> Tenant:
    return Tenant(
        data_class_rules={
            data_class: DataClassRule(redact=redact)
            for data_class, redact in (rules if rules is not None else {}).items()
        },
        budget_usd_month=Decimal("100.00"),
        rate_limit=RateLimit(rpm=60, tpm=100_000),
        overrides=overrides if overrides is not None else {},
    )


PROVIDERS: Mapping[str, Provider] = {
    "a": Provider(kind=ProviderKind.ANTHROPIC),
    "b": Provider(kind=ProviderKind.OPENAI),
    "off": Provider(kind=ProviderKind.OPENAI, enabled=False),
    "c": Provider(kind=ProviderKind.OPENAI),
    "self-a": Provider(kind=ProviderKind.OPENAI_COMPATIBLE),
}

ALIASES: Mapping[str, Alias] = {
    # Overridden by the `acme` tenant, so the default is never what resolves.
    "chat": Alias(candidates=(Candidate(provider="a", model="m-a"),)),
    "ordered": Alias(
        candidates=(
            Candidate(provider="a", model="m-a"),
            Candidate(provider="b", model="m-b"),
            Candidate(provider="self-a", model="m-self-a"),
        )
    ),
    "disabled-first": Alias(
        candidates=(
            Candidate(provider="off", model="m-off"),
            Candidate(provider="a", model="m-a"),
        )
    ),
    "all-off": Alias(candidates=(Candidate(provider="off", model="m-off"),)),
}


@pytest.fixture
def policy() -> Policy:
    """The in-memory policy every resolver case resolves against."""
    return make_policy(
        providers=PROVIDERS,
        aliases=ALIASES,
        tenants={
            "acme": make_tenant(
                rules={
                    DataClass.PUBLIC: False,
                    DataClass.INTERNAL: False,
                    DataClass.CONFIDENTIAL: True,
                    # No `pii` rule at all: a missing rule is not redacted.
                },
                overrides={"chat": (Candidate(provider="b", model="m-override"),)},
            ),
            "globex": make_tenant(rules={DataClass.PUBLIC: False}),
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
        providers={"a": Provider(kind=ProviderKind.ANTHROPIC)},
        aliases={name: Alias(candidates=(Candidate(provider="a", model="m-a"),))},
        tenants={name: make_tenant(rules={DataClass.PUBLIC: False})},
    )


@pytest.fixture
def shipped_policy_path() -> Path:
    return SHIPPED_POLICY_PATH
