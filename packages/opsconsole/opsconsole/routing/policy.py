"""The routing policy: its schema, its loader and the snapshot the gateway reads.

`routing.yaml` is the only shipped file that names a provider model id; every
other module names a model by alias. This schema is the boundary where an
operator's YAML becomes typed data.

Loading is deliberately strict. `yaml.safe_load` is the only deserialiser used
(pinned by `test_safe_load_only`), unknown keys are rejected, and the two
cross-model references YAML cannot express - a candidate naming a provider, an
override naming an alias - are checked here so a broken file fails on reload
instead of at the first request that happens to touch it. Every rejection is a
`PolicyInvalid` whose message names the field path, pinned by
`test_invalid_policy_names_the_field`.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from opsconsole.routing.errors import PolicyInvalid


class DataClass(StrEnum):
    """The sensitivity classes a request can declare. A closed set on purpose."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"


class Residency(StrEnum):
    """A tenant's residency regime. `EU` is the clamp; `ANY` applies none."""

    EU = "eu"
    ANY = "any"


class ResidencyMode(StrEnum):
    """How strictly the PROCESS enforces residency, as opposed to a tenant's regime.

    `STRICT` is the shipped behaviour and the default everywhere: the resolver only
    ever narrows, which is what makes the residency property a proof. `RELAXED` is
    an opt-in local-demo mode that widens `resolve()`'s allowed region set by
    `resolver.RELAXED_REGIONS` for a data class that has a rule and is not `PII`. It is a policy concept rather than a settings one, which is why it
    lives beside `Residency` and `core.settings` imports it from here.

    Read from `OPSCONSOLE_RESIDENCY_MODE` by `Settings.residency_mode`, where an
    unknown value refuses construction
    (`test_an_unknown_residency_mode_is_refused_at_construction`). The two members
    are pinned by `test_the_residency_mode_defaults_to_strict` and
    `test_the_residency_mode_accepts_relaxed`.
    """

    STRICT = "strict"
    RELAXED = "relaxed"


class ProviderKind(StrEnum):
    """Which adapter drives a provider. The adapters themselves arrive later."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class _Frozen(BaseModel):
    """Attributes are frozen and unknown keys rejected: a typo in the policy is an
    error. The mapping fields are ordinary dicts, so what a snapshot guarantees is
    that `PolicyStore` replaces it rather than editing it, not deep immutability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Provider(_Frozen):
    """One reachable endpoint. `region` is required: residency depends on it."""

    kind: ProviderKind
    region: str
    base_url: str | None = None
    endpoint_env: str | None = None
    key_env: str | None = None
    enabled: bool = True


class Candidate(_Frozen):
    """A provider and the model id to call on it. The id is an opaque string."""

    provider: str
    model: str


class Alias(_Frozen):
    """The ordered candidates an alias resolves to when a tenant has no override.

    `semantic_cache` is #23's per-alias switch, and it defaults to OFF: the
    semantic tier serves an answer to a question that is merely SIMILAR, which is
    right for a chat assistant and wrong for a judge or an extraction, so an
    alias reaches it only by saying so. An alias that omits the key is off
    (`test_an_alias_that_omits_the_semantic_switch_is_off`), and what the shipped
    file turns on is pinned by
    `test_only_agent_default_carries_the_semantic_cache_switch`.
    """

    candidates: tuple[Candidate, ...]
    semantic_cache: bool = False


class DataClassRule(_Frozen):
    """What a data class permits. `redact` is read by the redaction plan, not here."""

    allow_regions: tuple[str, ...]
    redact: bool = False


class RateLimit(_Frozen):
    """Per-tenant request and token ceilings. Enforced by the limits package.

    Both are `ge=1`, so a ceiling of zero or below is refused at LOAD with its
    field path named (an earlier change decision G,
    `test_a_zero_rate_limit_is_refused_at_load`). Before enforcement existed a
    zero parsed and was inert; enforced, it refuses every request that tenant
    ever sends, with a typed 429 and a `Retry-After` that never comes true, and
    nothing distinguishes that from the limiter working. A negative is a bucket
    whose capacity is below its own empty level. One is admitted - a smoke-test
    tenant is a real setting (`test_a_rate_limit_of_exactly_one_is_admitted`).
    """

    rpm: int = Field(ge=1)
    tpm: int = Field(ge=1)


class Tenant(_Frozen):
    """One tenant's routing regime.

    A data class with no rule is not a permissive default: the resolver treats a
    missing rule as an empty allow list, so an unlisted class denies.
    """

    residency: Residency
    data_class_rules: Mapping[DataClass, DataClassRule]
    budget_usd_month: Decimal
    rate_limit: RateLimit
    overrides: Mapping[str, tuple[Candidate, ...]] = {}


class Policy(_Frozen):
    """A whole routing policy, validated as one document."""

    providers: Mapping[str, Provider]
    eu_regions: tuple[str, ...]
    aliases: Mapping[str, Alias]
    tenants: Mapping[str, Tenant]

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        """Every candidate names a real provider and every override a real alias."""
        problems: list[str] = []

        for alias_name, alias in self.aliases.items():
            for index, candidate in enumerate(alias.candidates):
                if candidate.provider not in self.providers:
                    problems.append(
                        f"aliases.{alias_name}.candidates[{index}].provider: "
                        f"unknown provider {candidate.provider!r}"
                    )

        for tenant_name, tenant in self.tenants.items():
            for alias_name, candidates in tenant.overrides.items():
                if alias_name not in self.aliases:
                    problems.append(
                        f"tenants.{tenant_name}.overrides.{alias_name}: "
                        f"unknown alias {alias_name!r}"
                    )
                    continue
                for index, candidate in enumerate(candidates):
                    if candidate.provider not in self.providers:
                        problems.append(
                            f"tenants.{tenant_name}.overrides.{alias_name}[{index}].provider: "
                            f"unknown provider {candidate.provider!r}"
                        )

        if problems:
            raise ValueError("; ".join(problems))
        return self


def load_policy(path: Path) -> Policy:
    """Read, parse and validate one policy file.

    Every failure is a `PolicyInvalid` carrying the path and the underlying
    message, so an operator reading a failed reload sees the field, not a
    traceback in three different exception types.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyInvalid(f"{path}: cannot be read: {exc}") from exc

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyInvalid(f"{path}: not valid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise PolicyInvalid(
            f"{path}: the document must be a mapping, got {type(document).__name__}"
        )

    try:
        return Policy.model_validate(document)
    except ValidationError as exc:
        raise PolicyInvalid(f"{path}: {exc}") from exc


class PolicyStore:
    """Holds the policy the gateway resolves against, and swaps it on reload.

    `reload` parses into a new `Policy` before it rebinds the attribute, so a
    failed reload leaves the previous snapshot in place and a resolution already
    holding a snapshot keeps reading a consistent one. Pinned by
    `test_reload_swaps_snapshot` and `test_failed_reload_keeps_previous_policy`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._policy = load_policy(path)

    @property
    def policy(self) -> Policy:
        """The current snapshot."""
        return self._policy

    def reload(self) -> None:
        """Re-read the file and swap the snapshot, or raise and keep the old one."""
        self._policy = load_policy(self._path)
