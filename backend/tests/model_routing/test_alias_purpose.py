"""The optional `purpose:` line on an alias: its limits and the shipped file.

Each case loads real YAML through `load_policy`, so the schema is what is under test.
The Models page shows this line to a visitor, so a bound that drifts changes what they read.
"""

from __future__ import annotations

import pytest

from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import load_policy

from .conftest import (
    SHIPPED_POLICY_PATH,
    SHIPPED_PROVIDERS_DIR,
    DocFactory,
    PolicyWriter,
    providers_dir_of,
)

MAX_PURPOSE = 200


def document_with_purpose(valid_doc: DocFactory, purpose: object) -> dict[str, object]:
    """The baseline document with `purpose` written on the `chat` alias only."""
    document = valid_doc()
    aliases = document["aliases"]
    assert isinstance(aliases, dict)
    aliases["chat"] = {**aliases["chat"], "purpose": purpose}
    return document


def test_every_shipped_alias_has_a_purpose() -> None:
    """Catches a shipped `routing.yaml` where one alias was left without its purpose line,
    which the page would show as "no purpose given" for a live alias."""
    policy = load_policy(SHIPPED_POLICY_PATH, SHIPPED_PROVIDERS_DIR)

    assert list(policy.aliases) == [
        "agent-default",
        "agent-strong",
        "judge",
        "simulated-person",
        "embed",
    ]
    for name, alias in policy.aliases.items():
        assert alias.purpose, name
        assert alias.purpose.strip() == alias.purpose, name


def test_purpose_at_200_characters_loads(policy_file: PolicyWriter, valid_doc: DocFactory) -> None:
    """Catches an upper bound set below 200 (a `max_length=199` off-by-one) that refuses a
    purpose of exactly the limit."""
    purpose = "p" * MAX_PURPOSE
    path = policy_file(document_with_purpose(valid_doc, purpose))

    policy = load_policy(path, providers_dir_of(path))

    assert policy.aliases["chat"].purpose == purpose


def test_purpose_of_201_characters_is_refused(
    policy_file: PolicyWriter, valid_doc: DocFactory
) -> None:
    """Catches a purpose with no upper bound, or one set above 200, which lets a long line
    into the page and the response body."""
    path = policy_file(document_with_purpose(valid_doc, "p" * (MAX_PURPOSE + 1)))

    with pytest.raises(PolicyInvalid, match=r"purpose"):
        load_policy(path, providers_dir_of(path))


def test_empty_purpose_is_refused(policy_file: PolicyWriter, valid_doc: DocFactory) -> None:
    """Catches a lower bound of 0 that accepts `purpose: ""`, which the page would print as
    a blank line instead of "no purpose given"."""
    path = policy_file(document_with_purpose(valid_doc, ""))

    with pytest.raises(PolicyInvalid, match=r"purpose"):
        load_policy(path, providers_dir_of(path))


def test_alias_without_purpose_loads_as_none(
    policy_file: PolicyWriter, valid_doc: DocFactory
) -> None:
    """Catches a purpose made required (which refuses every older policy file) or defaulted
    to an empty string (which the page cannot tell from a written one)."""
    path = policy_file(valid_doc())

    policy = load_policy(path, providers_dir_of(path))

    assert policy.aliases["chat"].purpose is None
