"""Provider files: one YAML file per provider in a folder, loaded at startup and on reload.

The folder is the whole way to add a model: no code is touched. These cases write
files into `tmp_path` (or copy the shipped folder there) and load them through
the real loader. No backing service and no socket.

Every refused case changes ONE thing in an otherwise good file, so the refusal
can only come from the check the case names and never from a neighbouring one.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.core.config import Settings
from app.main import create_app
from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import load_policy
from app.model_routing.provider_files import (
    build_price_table,
    build_providers,
    load_provider_files,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
SHIPPED_PROVIDERS_DIR = CONFIG_DIR / "providers"
SHIPPED_POLICY_PATH = CONFIG_DIR / "routing.yaml"

BadMaker = Callable[[Path], tuple[str, str]]

PROVIDER_SLUG_MAX = 64
MODEL_ID_MAX = 128


def provider_document(provider: str, **changes: object) -> dict[str, Any]:
    """A provider file that loads clean, with the given top-level keys replaced."""
    document: dict[str, Any] = {
        "provider": provider,
        "kind": "openai_compatible",
        "base_url": "http://example.invalid/v1",
        "key_env": "EXAMPLE_API_KEY",
        "enabled": True,
        "models": {
            "model-one": {
                "prices": [
                    {
                        "effective_from": "2026-10-01",
                        "input_usd_per_1m": "1.00",
                        "output_usd_per_1m": "2.00",
                    }
                ]
            }
        },
    }
    document.update(changes)
    return document


def write_provider(folder: Path, document: Mapping[str, Any], file_name: str | None = None) -> Path:
    """Write one provider file. The file name defaults to `<provider>.yaml`."""
    folder.mkdir(parents=True, exist_ok=True)
    name = file_name if file_name is not None else f"{document['provider']}.yaml"
    path = folder / name
    path.write_text(yaml.safe_dump(dict(document), sort_keys=False), encoding="utf-8")
    return path


def model_with(model_id: str) -> dict[str, Any]:
    return {
        model_id: {
            "prices": [
                {
                    "effective_from": "2026-10-01",
                    "input_usd_per_1m": "1.00",
                    "output_usd_per_1m": "2.00",
                }
            ]
        }
    }


# --- the shipped folder -------------------------------------------------------


def test_the_shipped_folder_loads_three_providers() -> None:
    """Catches a shipped folder that carries a stray provider (a leftover
    `anthropic_eu`, `local` or `vllm`) or misses one of the three the gateway needs."""
    files = load_provider_files(SHIPPED_PROVIDERS_DIR)

    assert set(files) == {"anthropic", "openai", "local_embed"}


def test_a_new_file_adds_a_provider_and_its_prices(tmp_path: Path) -> None:
    """Catches a loader that only knows the three shipped names: a `mistral.yaml`
    dropped into a copy of the folder must yield an adapter entry and a price row
    with no code touched."""
    folder = tmp_path / "providers"
    shutil.copytree(SHIPPED_PROVIDERS_DIR, folder)
    write_provider(folder, provider_document("mistral"))

    files = load_provider_files(folder)

    assert set(files) == {"anthropic", "openai", "local_embed", "mistral"}
    assert "mistral" in build_providers(files)
    assert "mistral/model-one" in build_price_table(files).prices
    assert "anthropic/claude-haiku-5-5" in build_price_table(files).prices


# --- a bad file is refused with its name and its field ------------------------


def bad_unknown_key(folder: Path) -> tuple[str, str]:
    write_provider(folder, provider_document("badone", bogus_key=1))
    return "badone.yaml", "bogus_key"


def bad_yaml(folder: Path) -> tuple[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "badone.yaml").write_text("provider: [unclosed\n", encoding="utf-8")
    return "badone.yaml", ""


def bad_no_prices(folder: Path) -> tuple[str, str]:
    document = provider_document("badone")
    document["models"] = {"model-one": {"prices": []}}
    write_provider(folder, document)
    return "badone.yaml", "models.model-one.prices"


def bad_prices_not_a_list(folder: Path) -> tuple[str, str]:
    document = provider_document("badone")
    document["models"] = {"model-one": {"prices": "one dollar"}}
    write_provider(folder, document)
    return "badone.yaml", "models.model-one.prices"


@pytest.mark.parametrize(
    "make_bad",
    [bad_unknown_key, bad_yaml, bad_no_prices, bad_prices_not_a_list],
    ids=["unknown-key", "bad-yaml", "no-prices", "prices-not-a-list"],
)
def test_a_bad_file_names_the_file_and_the_field(
    tmp_path: Path, make_bad: Callable[[Path], tuple[str, str]]
) -> None:
    """Catches a loader that raises a bare pydantic error or a message without the
    file name: an operator reading a failed startup must see which file and which
    field to fix. A good neighbour sits beside the bad file so the name in the
    message is a choice, not the only file there was."""
    folder = tmp_path / "providers"
    write_provider(folder, provider_document("goodone"))
    file_name, field = make_bad(folder)

    with pytest.raises(PolicyInvalid) as raised:
        load_provider_files(folder)

    assert file_name in str(raised.value)
    assert field in str(raised.value)
    assert "goodone.yaml" not in str(raised.value)


def test_the_file_name_must_match_the_provider(tmp_path: Path) -> None:
    """Catches a loader that trusts the `provider:` line: `openai.yaml` saying
    `provider: anthropic` would silently replace the real anthropic file."""
    folder = tmp_path / "providers"
    write_provider(folder, provider_document("anthropic"), file_name="openai.yaml")

    with pytest.raises(PolicyInvalid) as raised:
        load_provider_files(folder)

    assert "openai.yaml" in str(raised.value)


def test_two_files_cannot_claim_one_provider(tmp_path: Path) -> None:
    """Catches a loader that lets the last file win a provider name. The extra
    file is named `extra.yaml` and says `provider: shared`, and `shared.yaml`
    says the same, so the refusal must name BOTH files."""
    folder = tmp_path / "providers"
    write_provider(folder, provider_document("shared"))
    write_provider(folder, provider_document("shared"), file_name="extra.yaml")

    with pytest.raises(PolicyInvalid) as raised:
        load_provider_files(folder)

    assert "shared.yaml" in str(raised.value)
    assert "extra.yaml" in str(raised.value)


# --- the inclusive limits -----------------------------------------------------


def test_a_provider_name_at_the_limit(tmp_path: Path) -> None:
    """Catches a provider-slug limit that is off by one: 64 characters loads, 65 is
    refused. The file is named after the provider each time, so the name-match
    check cannot be what refuses the 65."""
    ok = tmp_path / "ok"
    slug = "a" * PROVIDER_SLUG_MAX
    write_provider(ok, provider_document(slug))
    assert set(load_provider_files(ok)) == {slug}

    over = tmp_path / "over"
    write_provider(over, provider_document("a" * (PROVIDER_SLUG_MAX + 1)))
    with pytest.raises(PolicyInvalid):
        load_provider_files(over)


@pytest.mark.parametrize("tail", ["\n", "\r\n", " "], ids=["lf", "crlf", "space"])
def test_a_provider_name_with_a_trailing_character_is_refused(tmp_path: Path, tail: str) -> None:
    """Catches a slug check built on `$` or `.match`, which lets a trailing newline
    through (and a space-padded name). The file is NAMED with the same tail, so the
    file-name check agrees with the `provider:` line and only the slug check can refuse."""
    folder = tmp_path / "providers"
    name = f"mistral{tail}"
    write_provider(folder, provider_document(name), file_name=f"{name}.yaml")

    with pytest.raises(PolicyInvalid):
        load_provider_files(folder)


def test_a_model_id_at_the_limit(tmp_path: Path) -> None:
    """Catches a model-id limit that is off by one: 128 characters loads, 129 is
    refused."""
    ok = tmp_path / "ok"
    write_provider(ok, provider_document("good", models=model_with("m" * MODEL_ID_MAX)))
    assert "good" in load_provider_files(ok)

    over = tmp_path / "over"
    write_provider(over, provider_document("good", models=model_with("m" * (MODEL_ID_MAX + 1))))
    with pytest.raises(PolicyInvalid):
        load_provider_files(over)


@pytest.mark.parametrize(
    "model_id",
    ["two words", "model-one ", "model-one\n", "model-one\r\n", "\tmodel"],
    ids=["inner-space", "trailing-space", "trailing-lf", "trailing-crlf", "leading-tab"],
)
def test_a_model_id_with_whitespace_is_refused(tmp_path: Path, model_id: str) -> None:
    """Catches a model-id check that allows whitespace, or one built on `$` that
    lets a trailing newline through: the id becomes a price key and a header value."""
    folder = tmp_path / "providers"
    write_provider(folder, provider_document("good", models=model_with(model_id)))

    with pytest.raises(PolicyInvalid):
        load_provider_files(folder)


# --- the folder itself --------------------------------------------------------


def test_an_empty_or_missing_folder_is_refused(tmp_path: Path) -> None:
    """Catches a loader that starts a gateway with no providers at all, and one that
    leaks a FileNotFoundError: both an empty and a missing folder are a
    `PolicyInvalid` that names the folder."""
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "missing"

    for folder in (empty, missing):
        with pytest.raises(PolicyInvalid) as raised:
            load_provider_files(folder)
        assert str(folder) in str(raised.value)


def test_a_providers_block_in_routing_yaml_is_refused(tmp_path: Path) -> None:
    """Catches a policy loader that still accepts the old `providers:` block (two
    sources of truth for one provider), or refuses it without saying where
    providers live now."""
    folder = tmp_path / "providers"
    shutil.copytree(SHIPPED_PROVIDERS_DIR, folder)
    document = yaml.safe_load(SHIPPED_POLICY_PATH.read_text(encoding="utf-8"))
    document["providers"] = {"anthropic": {"kind": "anthropic", "key_env": "ANTHROPIC_API_KEY"}}
    routing = tmp_path / "routing.yaml"
    routing.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(PolicyInvalid) as raised:
        load_policy(routing, folder)

    assert "backend/config/providers/" in str(raised.value)


def test_the_policy_gets_its_providers_from_the_folder(tmp_path: Path) -> None:
    """Catches a `load_policy` that ignores the folder argument and reads the
    providers from somewhere else: a provider that exists only in the folder copy
    is in the policy."""
    folder = tmp_path / "providers"
    shutil.copytree(SHIPPED_PROVIDERS_DIR, folder)
    write_provider(folder, provider_document("mistral"))

    policy = load_policy(SHIPPED_POLICY_PATH, folder)

    assert set(policy.providers) == {"anthropic", "openai", "local_embed", "mistral"}


def test_create_app_will_not_start_on_a_bad_file(tmp_path: Path) -> None:
    """Catches a startup that skips a bad provider file and serves on the rest: the
    process must stop, and the message must name the file."""
    folder = tmp_path / "providers"
    shutil.copytree(SHIPPED_PROVIDERS_DIR, folder)
    broken = yaml.safe_load((folder / "openai.yaml").read_text(encoding="utf-8"))
    broken["bogus_key"] = 1
    (folder / "openai.yaml").write_text(yaml.safe_dump(broken), encoding="utf-8")
    settings = Settings(routing_policy_path=SHIPPED_POLICY_PATH, providers_dir=folder)

    with pytest.raises(PolicyInvalid) as raised:
        create_app(settings, {})

    assert "openai.yaml" in str(raised.value)
