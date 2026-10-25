"""The routing policy: its schema, its loader and the snapshot the gateway reads.

`routing.yaml` and the provider files in `backend/config/providers/` are the
only shipped files that name a provider model id; every other module names a
model by alias. The providers themselves come from that folder, never from
`routing.yaml`. This schema is the boundary where an
operator's YAML becomes typed data.

Loading is deliberately strict. `yaml.safe_load` is the only deserialiser used,
unknown keys are rejected, and the two
cross-model references YAML cannot express - a candidate naming a provider, an
override naming an alias - are checked here so a broken file fails on reload
instead of at the first request that happens to touch it. Every rejection is a
`PolicyInvalid` whose message names the field path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.context import DataClass as DataClass
from app.limits.prices import PriceTable
from app.model_routing.errors import PolicyInvalid
from app.model_routing.provider_files import Provider as Provider
from app.model_routing.provider_files import ProviderKind as ProviderKind
from app.model_routing.provider_files import (
    build_price_table,
    build_providers,
    load_provider_files,
)

PROVIDERS_FOLDER_HINT = "backend/config/providers/"
"""Where a provider is declared now, named in the refusal of an old `providers:` block."""


class _Frozen(BaseModel):
    """Attributes are frozen and unknown keys rejected: a typo in the policy is an
    error. The mapping fields are ordinary dicts, so what a snapshot guarantees is
    that `PolicyStore` replaces it rather than editing it, not deep immutability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


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

    `purpose` is the one line the console's Models page shows beside the alias.
    It is optional, and an alias without one reads "no purpose given" there.
    Pinned by `tests/model_routing/test_alias_purpose.py`.
    """

    candidates: tuple[Candidate, ...]
    semantic_cache: bool = False
    purpose: str | None = Field(default=None, min_length=1, max_length=200)


class DataClassRule(_Frozen):
    """What a data class does to a request: it redacts, or it does not."""

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

    A data class with no rule is not redacted.
    """

    data_class_rules: Mapping[DataClass, DataClassRule]
    budget_usd_month: Decimal
    rate_limit: RateLimit
    overrides: Mapping[str, tuple[Candidate, ...]] = {}


class Policy(_Frozen):
    """A whole routing policy, validated as one document."""

    providers: Mapping[str, Provider]
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


def _read_routing(path: Path) -> dict[str, object]:
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
    if "providers" in document:
        raise PolicyInvalid(
            f"{path}: providers: not allowed here; each provider is one file in "
            f"{PROVIDERS_FOLDER_HINT}"
        )
    return document


def load_routing(path: Path, providers_dir: Path) -> tuple[Policy, PriceTable]:
    """Read `routing.yaml` and the provider folder together, as one set.

    The folder is read once and gives both the policy's providers and the price
    table, so a reload can never pair a new policy with old prices. Every failure
    is a `PolicyInvalid` naming the file and the field.
    """
    document = _read_routing(path)
    files = load_provider_files(providers_dir)
    try:
        policy = Policy.model_validate({**document, "providers": build_providers(files)})
    except ValidationError as exc:
        raise PolicyInvalid(f"{path}: {exc}") from exc
    return policy, build_price_table(files)


def load_policy(path: Path, providers_dir: Path) -> Policy:
    """The policy alone, for callers that do not cost anything."""
    policy, _ = load_routing(path, providers_dir)
    return policy


class PolicyStore:
    """Holds the policy the gateway resolves against, and swaps it on reload.

    `reload` parses into a new `Policy` before it rebinds the attribute, so a
    failed reload leaves the previous snapshot in place and a resolution already
    holding a snapshot keeps reading a consistent one.

    `loaded_at` is when the live snapshot went in, read from `clock` (UTC by
    default). It moves only on a swap, so a failed reload keeps the old time.
    Pinned by `tests/model_routing/test_policy_loaded_at.py`.
    """

    def __init__(
        self,
        path: Path,
        providers_dir: Path,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._path = path
        self._providers_dir = providers_dir
        self._clock = clock
        self._policy = load_policy(path, providers_dir)
        self._loaded_at = clock()

    @property
    def path(self) -> Path:
        """The `routing.yaml` this store reads."""
        return self._path

    @property
    def providers_dir(self) -> Path:
        """The provider folder this store reads."""
        return self._providers_dir

    @property
    def policy(self) -> Policy:
        """The current snapshot."""
        return self._policy

    @property
    def loaded_at(self) -> datetime:
        """When the current snapshot was loaded or swapped in."""
        return self._loaded_at

    def replace(self, policy: Policy) -> None:
        """Swap in a policy already parsed elsewhere, and stamp the time."""
        self._policy = policy
        self._loaded_at = self._clock()

    def reload(self) -> None:
        """Re-read the files and swap the snapshot, or raise and keep the old one."""
        self.replace(load_policy(self._path, self._providers_dir))


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
