"""The routing policy: its schema, its loader and the snapshot the gateway reads.

`routing.yaml` is the only shipped file that names a provider model id; every
other module names a model by alias. This schema is the boundary where an
operator's YAML becomes typed data.

Loading is deliberately strict. `yaml.safe_load` is the only deserialiser used,
unknown keys are rejected, and the two
cross-model references YAML cannot express - a candidate naming a provider, an
override naming an alias - are checked here so a broken file fails on reload
instead of at the first request that happens to touch it. Every rejection is a
`PolicyInvalid` whose message names the field path.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# `ResidencyMode` now lives in `core/config.py` (the cycle break); re-exported
# here with the explicit `as` form so the call sites that import it from the
# policy keep resolving and `routing.yaml` stays the home of the policy types.
from app.core.config import ResidencyMode as ResidencyMode
from app.core.context import DataClass as DataClass
from app.model_routing.errors import PolicyInvalid

if TYPE_CHECKING:
    from app.limits.prices import PriceTable


class Residency(StrEnum):
    """A tenant's residency regime. `EU` is the clamp; `ANY` applies none."""

    EU = "eu"
    ANY = "any"


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

    `semantic_cache` is the per-alias switch, and it defaults to OFF: the
    semantic tier serves an answer to a question that is merely SIMILAR, which is
    right for a chat assistant and wrong for a judge or an extraction, so an
    alias reaches it only by saying so. An alias that omits the key is off, and what the shipped
    file turns on is fixed.
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
    field path named. Before enforcement existed a
    zero parsed and was inert; enforced, it refuses every request that tenant
    ever sends, with a typed 429 and a `Retry-After` that never comes true, and
    nothing distinguishes that from the limiter working. A negative is a bucket
    whose capacity is below its own empty level. One is admitted - a smoke-test
    tenant is a real setting.
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

    def allowed_regions(self, data_class: DataClass) -> tuple[str, ...]:
        """The raw allow list for one data class, or `()` when no rule exists.

        Deny by default: a missing rule returns an empty tuple, never a widened
        set. Both the resolver and the eval residency check call this instead of
        reading `data_class_rules` directly, so the reading lives once.

        The resolver applies its EU clamp and relaxed-mode union on top of this
        value; the eval check uses it raw. Neither caller changes the return value
        itself.
        """
        rule = self.data_class_rules.get(data_class)
        return rule.allow_regions if rule is not None else ()


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
    holding a snapshot keeps reading a consistent one.
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


def reachable_price_keys(policy: Policy) -> set[str]:
    """Every `provider/model` pair the policy can resolve to.

    Both sources, because a tenant override introduces a pair that need appear in
    no alias: the resolver uses an override in place of the alias's own candidate
    list, so a pair reachable only that way is just as billable.
    """
    keys: set[str] = set()
    for alias in policy.aliases.values():
        for candidate in alias.candidates:
            keys.add(f"{candidate.provider}/{candidate.model}")
    for tenant in policy.tenants.values():
        for candidates in tenant.overrides.values():
            for candidate in candidates:
                keys.add(f"{candidate.provider}/{candidate.model}")
    return keys


def check_policy_models_priced(policy: Policy, table: PriceTable) -> None:
    """Raise `PolicyInvalid` if any pair the policy can reach has no price row.

    Both sources are checked: alias candidates and tenant overrides. A pair with
    no price row would be booked `unpriced` on every request that reaches it,
    silently filling the ledger with rows that cost nothing. Called at load time
    (in `create_app`) and at reload time (in `RoutingRuntimeStore.reload`), so
    the case can only be reached through a deliberate policy change after boot.

    The unhappy path names all missing pairs in one message, sorted, so an
    operator can fix all of them in one edit rather than learning about them one
    reload at a time.
    """
    missing = sorted(reachable_price_keys(policy) - set(table.prices))
    if missing:
        raise PolicyInvalid(f"unpriced models reachable from the policy: {', '.join(missing)}")
