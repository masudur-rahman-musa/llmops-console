"""Resolving (tenant, alias, data class) to an ordered candidate list.

Every unhappy path here asserts the exact typed code and HTTP status, because an earlier change maps those values into the wire body and a
drift would be invisible until then.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.core.errors import TenantMismatch
from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.model_routing.errors import AliasUnknown, AllProvidersUnavailable
from app.model_routing.policy import DataClass, Policy, load_policy
from app.model_routing.resolver import Resolved, resolve
from app.providers.factory import build_adapters

from .conftest import SHIPPED_POLICY_PATH


def as_tuples(resolved: tuple[Resolved, ...]) -> list[tuple[str, str]]:
    return [(r.provider, r.model) for r in resolved]


# The `judge` alias against the SHIPPED policy, resolved through the health seam
# the routes fill: a provider is healthy when `build_adapters` built an adapter
# for it, so the reachable set is decided by the environment alone.

JUDGE_TENANT = "evals"
JUDGE_ALIAS = "judge"
JUDGE_DATA_CLASS = "public"

ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
FAKE_ANTHROPIC_KEY = "sk-ant-not-a-real-key"
FAKE_OPENAI_KEY = "sk-openai-not-a-real-key"


def judge_candidates(environ: dict[str, str]) -> list[tuple[str, str]]:
    """What `judge` resolves to when only `environ` is set, adapters as the health seam.

    The keys are injected rather than read from `os.environ`, so no case here
    depends on a provider key being present in the environment (#66).
    """
    policy = load_policy(SHIPPED_POLICY_PATH)
    adapters = build_adapters(policy, environ)
    return as_tuples(
        resolve(
            policy,
            JUDGE_TENANT,
            JUDGE_ALIAS,
            JUDGE_DATA_CLASS,
            is_healthy=lambda provider: provider in adapters,
        )
    )


def test_override_beats_alias_default(policy: Policy) -> None:
    """`acme` overrides `chat`, so the alias default never appears."""
    result = resolve(policy, "acme", "chat", "public")

    assert as_tuples(result) == [("b", "m-override")]


def test_disabled_provider_filtered(policy: Policy) -> None:
    result = resolve(policy, "acme", "disabled-first", "public")

    assert as_tuples(result) == [("a", "m-a")]


def test_order_preserved(policy: Policy) -> None:
    """Three candidates, returned in the policy's order, not sorted."""
    result = resolve(policy, "acme", "ordered", "public")

    assert as_tuples(result) == [("a", "m-a"), ("b", "m-b"), ("self-a", "m-self-a")]


def test_every_data_class_gets_the_same_candidates(policy: Policy) -> None:
    """The data class decides redaction, not providers: all four resolve alike."""
    results = {
        data_class: resolve(policy, "acme", "ordered", data_class.value) for data_class in DataClass
    }

    assert len(set(results.values())) == 1


def test_a_tenant_with_no_rule_for_a_data_class_still_resolves(policy: Policy) -> None:
    """`acme` has no `pii` rule, and a missing rule is not a refusal."""
    assert resolve(policy, "acme", "ordered", "pii")


def test_all_candidates_disabled_raises_all_providers_unavailable(policy: Policy) -> None:
    with pytest.raises(AllProvidersUnavailable) as excinfo:
        resolve(policy, "acme", "all-off", "public")

    assert excinfo.value.code == "ALL_PROVIDERS_FAILED"
    assert excinfo.value.http_status == 503


def test_unknown_tenant_raises_tenant_mismatch(policy: Policy) -> None:
    """A tenant the policy does not carry is a 403, never a widened default."""
    with pytest.raises(TenantMismatch) as excinfo:
        resolve(policy, "not-a-tenant", "ordered", "public")

    assert excinfo.value.code == "TENANT_MISMATCH"
    assert excinfo.value.http_status == 403


def test_unknown_alias_raises(policy: Policy) -> None:
    with pytest.raises(AliasUnknown) as excinfo:
        resolve(policy, "acme", "not-an-alias", "public")

    assert excinfo.value.code == "ALIAS_UNKNOWN"
    assert excinfo.value.http_status == 404


def always_unhealthy(provider: str) -> bool:
    return False


def only_b_healthy(provider: str) -> bool:
    return provider == "b"


def always_healthy(provider: str) -> bool:
    return True


def test_is_healthy_filtering_everything_raises_all_providers_unavailable(
    policy: Policy,
) -> None:
    """A health outage is 503."""
    with pytest.raises(AllProvidersUnavailable) as excinfo:
        resolve(policy, "acme", "ordered", "public", is_healthy=always_unhealthy)

    assert excinfo.value.code == "ALL_PROVIDERS_FAILED"
    assert excinfo.value.http_status == 503


def test_is_healthy_filters_a_subset_and_keeps_order(policy: Policy) -> None:
    result = resolve(policy, "acme", "ordered", "public", is_healthy=only_b_healthy)

    assert as_tuples(result) == [("b", "m-b")]


def test_default_is_healthy_keeps_every_candidate(policy: Policy) -> None:
    """The seam the routes fill with the breaker defaults to always-true for callers without one."""
    healthy: Callable[[str], bool] = always_healthy

    assert resolve(policy, "acme", "ordered", "public") == resolve(
        policy, "acme", "ordered", "public", is_healthy=healthy
    )


def test_max_identifier_length_is_pinned() -> None:
    """The cap is a bound the pre-check enforces, so it is pinned exactly."""
    assert MAX_IDENTIFIER_LENGTH == 64


# Rejection table. The trailing-newline row is the one a
# `^...$` pattern with `.match` would wrongly accept.
BAD_IDENTIFIERS: list[tuple[str, str]] = [
    ("trailing newline", "acme\n"),
    ("trailing crlf", "acme\r\n"),
    ("trailing space", "acme "),
    ("leading space", " acme"),
    ("uppercase", "ACME"),
    ("empty string", ""),
    ("one over the maximum length", "a" * (MAX_IDENTIFIER_LENGTH + 1)),
    ("path separator", "acme/../root"),
    ("null byte", "acme\x00"),
]

FIELDS: list[str] = ["tenant", "alias"]


@pytest.mark.parametrize("field", FIELDS, ids=FIELDS)
@pytest.mark.parametrize(("case", "value"), BAD_IDENTIFIERS, ids=[c[0] for c in BAD_IDENTIFIERS])
def test_identifier_rejection_table(case: str, value: str, field: str, policy: Policy) -> None:
    tenant = value if field == "tenant" else "acme"
    alias = value if field == "alias" else "ordered"

    with pytest.raises(IdentifierInvalid) as excinfo:
        resolve(policy, tenant, alias, "public")

    assert excinfo.value.code == "VALIDATION"
    assert excinfo.value.http_status == 422
    # The message names which argument was rejected, so a caller can fix it.
    assert field in str(excinfo.value)


def test_identifier_at_the_maximum_length_is_accepted(boundary_policy: Policy) -> None:
    """The inclusive boundary: exactly `MAX_IDENTIFIER_LENGTH` resolves."""
    name = "a" * MAX_IDENTIFIER_LENGTH

    result = resolve(boundary_policy, name, name, "public")

    assert as_tuples(result) == [("a", "m-a")]


BAD_DATA_CLASSES: list[tuple[str, str]] = [
    ("not a member", "secret"),
    ("uppercase member", "PUBLIC"),
    ("trailing newline", "public\n"),
    ("trailing space", "public "),
    ("empty string", ""),
]


@pytest.mark.parametrize(
    ("case", "data_class"), BAD_DATA_CLASSES, ids=[c[0] for c in BAD_DATA_CLASSES]
)
def test_unknown_data_class_raises(case: str, data_class: str, policy: Policy) -> None:
    with pytest.raises(IdentifierInvalid) as excinfo:
        resolve(policy, "acme", "ordered", data_class)

    assert excinfo.value.code == "VALIDATION"
    assert "data_class" in str(excinfo.value)


def test_identifier_check_runs_before_any_lookup(policy: Policy) -> None:
    """A bad identifier is `IdentifierInvalid`, not `AliasUnknown`.

    Order matters: the pre-check must reject before the string is used as a key.
    """
    with pytest.raises(IdentifierInvalid):
        resolve(policy, "acme", "no such alias", "public")


def test_shipped_policy_exhaustive() -> None:
    """Every triple in the shipped file resolves to enabled providers only."""
    policy = load_policy(SHIPPED_POLICY_PATH)
    checked = 0

    for tenant_name in policy.tenants:
        for alias_name in policy.aliases:
            for data_class in DataClass:
                checked += 1
                result = resolve(policy, tenant_name, alias_name, data_class.value)
                assert result, (tenant_name, alias_name, data_class)
                for candidate in result:
                    assert policy.providers[candidate.provider].enabled

    assert checked == len(policy.tenants) * len(policy.aliases) * len(DataClass)
    assert checked > 0


def test_judge_stays_on_anthropic_when_the_openai_key_is_set() -> None:
    """Catches a judge that reaches OpenAI while the `openai` provider is turned off: a set
    `OPENAI_API_KEY` must not bring back a second candidate the policy has switched off."""
    candidates = judge_candidates(
        {ANTHROPIC_KEY_ENV: FAKE_ANTHROPIC_KEY, OPENAI_KEY_ENV: FAKE_OPENAI_KEY}
    )

    assert candidates == [("anthropic", "claude-haiku-5-5")]


def test_judge_has_one_candidate_when_it_is_not() -> None:
    """Catches a judge pointed anywhere but the cheaper Anthropic model."""
    candidates = judge_candidates({ANTHROPIC_KEY_ENV: FAKE_ANTHROPIC_KEY})

    assert candidates == [("anthropic", "claude-haiku-5-5")]


EVAL_FIRST_CHOICE = (
    ("agent-default", "claude-haiku-5-5"),
    ("judge", "claude-haiku-5-5"),
    ("agent-strong", "claude-sonnet-5-5"),
)


@pytest.mark.parametrize(
    ("alias", "model"), EVAL_FIRST_CHOICE, ids=[a for a, _ in EVAL_FIRST_CHOICE]
)
def test_eval_aliases_run_on_haiku_and_sonnet_5_5(alias: str, model: str) -> None:
    """Catches a shipped policy that still sends the cheap alias or the judge to Haiku 4.5,
    or the strong alias to anything but Sonnet 5.5: the eval baseline is set on these
    exact models, so a different first choice makes every later run incomparable."""
    policy = load_policy(SHIPPED_POLICY_PATH)

    resolved = resolve(
        policy,
        JUDGE_TENANT,
        alias,
        JUDGE_DATA_CLASS,
        is_healthy=lambda provider: provider.startswith("anthropic"),
    )

    assert resolved
    assert (resolved[0].provider, resolved[0].model) == ("anthropic", model)


def test_no_candidate_in_the_shipped_policy_names_openai() -> None:
    """Catches a leftover `openai` or `gpt-4o` candidate on any alias or any tenant override:
    OpenAI is not used for evals, and a fallback to it would spend on a
    provider the owner turned off. Checked on the policy file, not on what resolves, so a
    candidate hidden behind a disabled provider is still found."""
    policy = load_policy(SHIPPED_POLICY_PATH)
    candidates = [c for alias in policy.aliases.values() for c in alias.candidates]
    for tenant in policy.tenants.values():
        for override in tenant.overrides.values():
            candidates.extend(override)

    assert candidates
    assert [c for c in candidates if c.provider == "openai" or c.model.startswith("gpt")] == []
