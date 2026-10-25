"""Provider files: one YAML file per provider, in one folder.

A file holds everything about one provider: its kind, where it lives, which
environment variable holds its key, and each model it serves with that model's
prices. Adding a model is dropping a file in the folder and adding an alias line
to `routing.yaml`; no code changes.

Loading is strict, in the shape `app.model_routing.policy` set. `yaml.safe_load`
is the only parser, unknown keys are refused, a file must be named
`<provider>.yaml`, two files may not claim one provider, and an empty or missing
folder is refused. Every refusal is a `PolicyInvalid` naming the file and the
field, so a bad file stops startup (or a reload) with a message an operator can
act on.

Prices keep the rules `app.limits.prices` set: a list per model, append-only,
chosen by `effective_from`. The table the gateway costs against is built from
these files by `build_price_table`, keyed `provider/model` as before.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from app.limits.prices import PriceEntry, PriceTable
from app.model_routing.errors import PolicyInvalid

PROVIDER_NAME_MAX = 64
"""Longest provider name. It is a metric label, a log field and a price-key prefix."""

MODEL_ID_MAX = 128
"""Longest model id. It is half a price key and travels in a response header."""

_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
_MODEL_ID = re.compile(r"\S+")


def _provider_name(value: str) -> str:
    """`fullmatch`, never `match` with `$`: a trailing newline must not pass."""
    if len(value) > PROVIDER_NAME_MAX or _PROVIDER_NAME.fullmatch(value) is None:
        raise ValueError(
            f"must be lower-case letters, digits, - and _ (at most {PROVIDER_NAME_MAX})"
        )
    return value


def _model_id(value: str) -> str:
    if len(value) > MODEL_ID_MAX or _MODEL_ID.fullmatch(value) is None:
        raise ValueError(f"must be 1 to {MODEL_ID_MAX} characters with no spaces")
    return value


ProviderName = Annotated[str, AfterValidator(_provider_name)]
ModelId = Annotated[str, AfterValidator(_model_id)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderKind(StrEnum):
    """Which adapter drives a provider."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class Provider(_Frozen):
    """One reachable endpoint: a provider file without its models."""

    kind: ProviderKind
    base_url: str | None = None
    endpoint_env: str | None = None
    key_env: str | None = None
    enabled: bool = True


class ModelEntry(_Frozen):
    """One model a provider serves, and its prices over time. At least one price."""

    prices: Annotated[tuple[PriceEntry, ...], Field(min_length=1)]


class ProviderFile(_Frozen):
    """One provider file, validated as one document."""

    provider: ProviderName
    kind: ProviderKind
    base_url: str | None = None
    endpoint_env: str | None = None
    key_env: str | None = None
    enabled: bool = True
    models: Annotated[Mapping[ModelId, ModelEntry], Field(min_length=1)]


def _load_one(path: Path) -> ProviderFile:
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
        return ProviderFile.model_validate(document)
    except ValidationError as exc:
        raise PolicyInvalid(f"{path}: {exc}") from exc


def load_provider_files(directory: Path) -> Mapping[str, ProviderFile]:
    """Read every `*.yaml` in `directory`, keyed by provider name.

    All files are parsed first, then two files claiming one provider are refused
    (naming both), then each file name is checked against its `provider:` line.
    Files are read in name order, so the same folder always gives the same error.
    """
    if not directory.is_dir():
        raise PolicyInvalid(f"{directory}: the provider folder does not exist")
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise PolicyInvalid(f"{directory}: the provider folder holds no *.yaml file")

    loaded = [(path, _load_one(path)) for path in paths]

    claimed: dict[str, list[Path]] = {}
    for path, file in loaded:
        claimed.setdefault(file.provider, []).append(path)
    for provider, owners in claimed.items():
        if len(owners) > 1:
            names = ", ".join(owner.name for owner in owners)
            raise PolicyInvalid(f"{directory}: provider {provider!r} is claimed by {names}")

    for path, file in loaded:
        if path.stem != file.provider:
            raise PolicyInvalid(
                f"{path}: provider: {file.provider!r} must match the file name "
                f"(expected {path.stem!r})"
            )

    return {file.provider: file for _, file in loaded}


def build_providers(files: Mapping[str, ProviderFile]) -> dict[str, Provider]:
    """The policy's provider rows: each file without its models."""
    return {
        name: Provider(
            kind=file.kind,
            base_url=file.base_url,
            endpoint_env=file.endpoint_env,
            key_env=file.key_env,
            enabled=file.enabled,
        )
        for name, file in files.items()
    }


def build_price_table(files: Mapping[str, ProviderFile]) -> PriceTable:
    """The price table, keyed `provider/model` - the key `Adapter.price_key` builds."""
    return PriceTable(
        prices={
            f"{name}/{model_id}": model.prices
            for name, file in files.items()
            for model_id, model in file.models.items()
        }
    )
