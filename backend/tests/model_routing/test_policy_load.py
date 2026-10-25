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

from app.model_routing.errors import PolicyInvalid
from app.model_routing.policy import PolicyStore, RateLimit, load_policy

from .conftest import SHIPPED_POLICY_PATH, DocFactory, PolicyWriter


def test_policy_loads_shipped_file() -> None:
    """The file the product ships with is itself a test case."""
    assert SHIPPED_POLICY_PATH.exists()

    policy = load_policy(SHIPPED_POLICY_PATH)

    assert len(policy.providers) >= 1
    assert len(policy.aliases) >= 1
    assert len(policy.tenants) >= 1
    # Every alias candidate names a provider that exists: the cross-field rule
    # holds on the real file, not only on the crafted bad ones below.
    for alias in policy.aliases.values():
        for candidate in alias.candidates:
            assert candidate.provider in policy.providers


ALIAS_WITHOUT_THE_SWITCH = """
providers:
  a: {kind: anthropic}
aliases:
  chat:
    candidates:
      - {provider: a, model: m1}
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
  a: {kind: anthropic}
aliases:
  chat:
    candidates:
      - {provider: a, model: m1}
      - {provider: ghost, model: m2}
tenants: {}
"""

UNKNOWN_ALIAS_IN_OVERRIDE = """
providers:
  a: {kind: anthropic}
aliases:
  chat:
    candidates:
      - {provider: a, model: m1}
tenants:
  acme:
    data_class_rules:
      public: {}
    budget_usd_month: "10.00"
    rate_limit: {rpm: 60, tpm: 1000}
    overrides:
      nope:
        - {provider: a, model: m1}
"""

UNKNOWN_PROVIDER_IN_OVERRIDE = """
providers:
  a: {kind: anthropic}
aliases:
  chat:
    candidates:
      - {provider: a, model: m1}
tenants:
  acme:
    data_class_rules:
      public: {}
    budget_usd_month: "10.00"
    rate_limit: {rpm: 60, tpm: 1000}
    overrides:
      chat:
        - {provider: ghost, model: m1}
"""

PROVIDER_WITHOUT_KIND = """
providers:
  a: {key_env: SOME_KEY}
aliases:
  chat:
    candidates:
      - {provider: a, model: m1}
tenants: {}
"""

MISSING_TOP_LEVEL_KEY = """
providers:
  a: {kind: anthropic}
aliases:
  chat:
    candidates:
      - {provider: a, model: m1}
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
    ("provider has no kind", PROVIDER_WITHOUT_KIND, "providers.a.kind"),
    ("top level key missing", MISSING_TOP_LEVEL_KEY, "tenants"),
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
    path = policy_file(PROVIDER_WITHOUT_KIND)

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


# --------------------------------------------------------------------------
# a rate limit that refuses everything
# --------------------------------------------------------------------------


def rate_limited_policy(*, rpm: str, tpm: str) -> str:
    """One well-formed policy document whose only variable is the tenant's rate limit."""
    return f"""
providers:
  a: {{kind: anthropic}}
aliases:
  chat:
    candidates:
      - {{provider: a, model: m1}}
tenants:
  acme:
    data_class_rules:
      public: {{}}
    budget_usd_month: "10.00"
    rate_limit: {{rpm: {rpm}, tpm: {tpm}}}
"""


# (case id, rpm, tpm, the field path the message must name)
BAD_RATE_LIMITS: list[tuple[str, str, str, str]] = [
    ("rpm is zero", "0", "1000", "tenants.acme.rate_limit.rpm"),
    ("rpm is negative", "-1", "1000", "tenants.acme.rate_limit.rpm"),
    ("tpm is zero", "60", "0", "tenants.acme.rate_limit.tpm"),
    ("tpm is negative", "60", "-1000", "tenants.acme.rate_limit.tpm"),
]


@pytest.mark.parametrize(
    ("case", "rpm", "tpm", "field"), BAD_RATE_LIMITS, ids=[c[0] for c in BAD_RATE_LIMITS]
)
def test_a_zero_rate_limit_is_refused_at_load(
    case: str, rpm: str, tpm: str, field: str, policy_file: PolicyWriter
) -> None:
    """A ceiling of zero is an operator error, not a policy.

    Today `rpm: 0` parses and is inert because nothing enforces it. Once it IS enforced,
    a zero refuses every request that tenant ever sends, with a typed 429 and a
    `Retry-After` that never comes true - and nothing anywhere distinguishes that from
    the limiter working. A negative is worse: it is a bucket whose capacity is below its
    empty level, so it can never be refilled into a servable state at all.

    Catches `rpm: int` left without `ge=1`, which is exactly the shape the field has
    before this plan. The FIELD PATH is asserted rather than only the raise, because a
    `PolicyInvalid` that says "validation failed" leaves an operator grepping a YAML file
    during an outage - which is why every other row of this module's invalid table
    asserts a path too. The four rows are two fields times two ways of being unservable,
    so a `ge=1` applied to `rpm` alone fails the last two."""
    path = policy_file(rate_limited_policy(rpm=rpm, tpm=tpm))

    with pytest.raises(PolicyInvalid) as excinfo:
        load_policy(path)

    assert field in str(excinfo.value)


def test_a_rate_limit_of_exactly_one_is_admitted() -> None:
    """The inclusive boundary, and the row that stops the table above passing against a
    bound written `gt=1` or `ge=2`.

    One request a minute is a real setting - it is what a smoke-test tenant or a
    throttled trial account looks like - so a bound that refuses it turns a legitimate
    policy into a gateway that will not start. Asserted through `RateLimit` directly
    rather than through a file, because the claim is about the field's bound and not
    about the loader's error path."""
    limit = RateLimit(rpm=1, tpm=1)

    assert limit.rpm == 1
    assert limit.tpm == 1
