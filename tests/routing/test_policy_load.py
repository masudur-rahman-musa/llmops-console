"""Loading, validating and hot-reloading the routing policy file.

A bad policy file is an operator error, never a client error: every case here
expects `PolicyInvalid` and a message that names the offending field path, so
the operator can find it without reading the loader. The reload cases pin the
snapshot contract: a resolution in flight holds one immutable `Policy` and a
failed reload leaves the previous one in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opsconsole.routing.errors import PolicyInvalid
from opsconsole.routing.policy import PolicyStore, load_policy

from .conftest import SHIPPED_POLICY_PATH, DocFactory, PolicyWriter


def test_policy_loads_shipped_file() -> None:
    """The file the product ships with is itself a test case."""
    assert SHIPPED_POLICY_PATH.exists()

    policy = load_policy(SHIPPED_POLICY_PATH)

    assert len(policy.providers) >= 1
    assert len(policy.aliases) >= 1
    assert len(policy.tenants) >= 1
    assert len(policy.eu_regions) >= 1
    # Every alias candidate names a provider that exists: the cross-field rule
    # holds on the real file, not only on the crafted bad ones below.
    for alias in policy.aliases.values():
        for candidate in alias.candidates:
            assert candidate.provider in policy.providers


ALIAS_WITHOUT_THE_SWITCH = """
providers:
  eu-a: {kind: anthropic, region: eu}
eu_regions: [eu]
aliases:
  chat:
    candidates:
      - {provider: eu-a, model: m1}
tenants: {}
"""
"""One alias written the way every alias in the repository was written before #23:
no `semantic_cache` key at all."""


def test_only_agent_default_carries_the_semantic_cache_switch() -> None:
    """Catches the semantic tier turned on for an alias whose callers need the answer to
    THEIR input - `judge`, `embed`, `agent-strong` - which serves a near-neighbour's answer
    to an evaluation and makes a scoring run irreproducible without a single failing
    assertion anywhere. Asserted against the SHIPPED file rather than a fixture, because the
    file is what a deployment runs, and stated as an exhaustive pair (the one that is on,
    every other one off) so that a later alias added with the switch on cannot slip in
    behind a check that only names today's aliases (#23, "off by default per alias")."""
    policy = load_policy(SHIPPED_POLICY_PATH)

    enabled = {name for name, alias in policy.aliases.items() if alias.semantic_cache}

    assert enabled == {"agent-default"}


def test_an_alias_that_omits_the_semantic_switch_is_off(policy_file: PolicyWriter) -> None:
    """Catches a switch whose default is `True`, or one Pydantic makes required: every alias
    written before #23 omits the key, so a required field would refuse every policy file in
    existence and a `True` default would silently opt all of them into approximate reuse -
    the opposite of what "off by default per alias" means."""
    path = policy_file(ALIAS_WITHOUT_THE_SWITCH)

    policy = load_policy(path)

    assert policy.aliases["chat"].semantic_cache is False


UNKNOWN_PROVIDER_IN_ALIAS = """
providers:
  eu-a: {kind: anthropic, region: eu}
eu_regions: [eu]
aliases:
  chat:
    candidates:
      - {provider: eu-a, model: m1}
      - {provider: ghost, model: m2}
tenants: {}
"""

UNKNOWN_ALIAS_IN_OVERRIDE = """
providers:
  eu-a: {kind: anthropic, region: eu}
eu_regions: [eu]
aliases:
  chat:
    candidates:
      - {provider: eu-a, model: m1}
tenants:
  acme:
    residency: eu
    data_class_rules:
      public: {allow_regions: [eu]}
    budget_usd_month: "10.00"
    rate_limit: {rpm: 60, tpm: 1000}
    overrides:
      nope:
        - {provider: eu-a, model: m1}
"""

UNKNOWN_PROVIDER_IN_OVERRIDE = """
providers:
  eu-a: {kind: anthropic, region: eu}
eu_regions: [eu]
aliases:
  chat:
    candidates:
      - {provider: eu-a, model: m1}
tenants:
  acme:
    residency: eu
    data_class_rules:
      public: {allow_regions: [eu]}
    budget_usd_month: "10.00"
    rate_limit: {rpm: 60, tpm: 1000}
    overrides:
      chat:
        - {provider: ghost, model: m1}
"""

PROVIDER_WITHOUT_REGION = """
providers:
  eu-a: {kind: anthropic}
eu_regions: [eu]
aliases:
  chat:
    candidates:
      - {provider: eu-a, model: m1}
tenants: {}
"""

MISSING_TOP_LEVEL_KEY = """
providers:
  eu-a: {kind: anthropic, region: eu}
aliases:
  chat:
    candidates:
      - {provider: eu-a, model: m1}
tenants: {}
"""

# (case id, document, the field path the message must name)
INVALID_CASES: list[tuple[str, str, str]] = [
    (
        "candidate names an unknown provider",
        UNKNOWN_PROVIDER_IN_ALIAS,
        "aliases.chat.candidates[1].provider",
    ),
    (
        "override key names an unknown alias",
        UNKNOWN_ALIAS_IN_OVERRIDE,
        "tenants.acme.overrides.nope",
    ),
    (
        "override candidate names an unknown provider",
        UNKNOWN_PROVIDER_IN_OVERRIDE,
        "tenants.acme.overrides.chat[0].provider",
    ),
    ("provider has no region", PROVIDER_WITHOUT_REGION, "providers.eu-a.region"),
    ("top level key missing", MISSING_TOP_LEVEL_KEY, "eu_regions"),
]


@pytest.mark.parametrize(
    ("case", "document", "field"), INVALID_CASES, ids=[c[0] for c in INVALID_CASES]
)
def test_invalid_policy_names_the_field(
    case: str, document: str, field: str, policy_file: PolicyWriter
) -> None:
    path = policy_file(document)

    with pytest.raises(PolicyInvalid) as excinfo:
        load_policy(path)

    assert field in str(excinfo.value)


BROKEN_CASES: list[tuple[str, str]] = [
    ("unparseable yaml", "providers: [oops\n  - :::\n"),
    ("document is a list", "- providers\n- aliases\n"),
    ("document is a scalar", "just-a-string\n"),
    ("document is empty", ""),
]


@pytest.mark.parametrize(("case", "document"), BROKEN_CASES, ids=[c[0] for c in BROKEN_CASES])
def test_non_mapping_or_broken_document_is_rejected(
    case: str, document: str, policy_file: PolicyWriter
) -> None:
    path = policy_file(document)

    with pytest.raises(PolicyInvalid):
        load_policy(path)


def test_safe_load_only(policy_file: PolicyWriter, tmp_path: Path) -> None:
    """A python object tag must be a load error, never a constructed object.

    `yaml.load` with the default loader would execute this; `yaml.safe_load`
    refuses the tag, and the loader turns that refusal into `PolicyInvalid`.
    """
    marker = tmp_path / "pwned.txt"
    document = f"""
providers: !!python/object/apply:os.system ["touch {marker}"]
eu_regions: [eu]
aliases: {{}}
tenants: {{}}
"""
    path = policy_file(document)

    with pytest.raises(PolicyInvalid):
        load_policy(path)

    assert not marker.exists()


def test_missing_file_is_policy_invalid(tmp_path: Path) -> None:
    with pytest.raises(PolicyInvalid):
        load_policy(tmp_path / "does-not-exist.yaml")


def test_store_rejects_a_bad_file_at_construction(policy_file: PolicyWriter) -> None:
    path = policy_file(PROVIDER_WITHOUT_REGION)

    with pytest.raises(PolicyInvalid):
        PolicyStore(path)


def test_reload_swaps_snapshot(policy_file: PolicyWriter, valid_doc: DocFactory) -> None:
    """The swap is atomic: the snapshot a caller already holds never mutates."""
    path = policy_file(valid_doc(rpm=60))
    store = PolicyStore(path)
    before = store.policy
    assert before.tenants["acme"].rate_limit.rpm == 60

    path.write_text(
        policy_file(valid_doc(rpm=120), name="next.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # `reload` returns None: call it, never assert on its result.
    store.reload()

    assert store.policy.tenants["acme"].rate_limit.rpm == 120
    assert store.policy is not before
    # The snapshot captured before the reload is untouched, which is what "a
    # resolution in flight sees one consistent policy" means.
    assert before.tenants["acme"].rate_limit.rpm == 60


def test_failed_reload_keeps_previous_policy(
    policy_file: PolicyWriter, valid_doc: DocFactory
) -> None:
    path = policy_file(valid_doc(rpm=60))
    store = PolicyStore(path)
    before = store.policy

    path.write_text("providers: [oops\n  - :::\n", encoding="utf-8")

    with pytest.raises(PolicyInvalid):
        store.reload()

    assert store.policy is before
    assert store.policy.tenants["acme"].rate_limit.rpm == 60
