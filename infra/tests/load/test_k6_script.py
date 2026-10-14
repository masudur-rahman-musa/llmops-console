r"""`infra/load/k6/gateway.js`: the k6 scenario, thresholds and traffic mix.

Pinned strings, as written in the agreed contract, plus one re-derivation of
the mix from the boundaries the script actually carries. Nothing here runs k6.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "infra" / "load" / "k6" / "gateway.js"


@pytest.fixture(scope="module")
def script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "pinned",
    [
        "executor: 'constant-arrival-rate'",
        "rate: 50",
        "timeUnit: '1s'",
        "duration: '10m'",
        "preAllocatedVUs: 100",
        "maxVUs: 300",
    ],
)
def test_the_scenario_offers_fifty_requests_a_second_for_ten_minutes(
    script: str, pinned: str
) -> None:
    """Criterion 11: catches a closed-model executor (which slows its offered rate
    when the gateway slows), a shorter run, or a VU pool too small for 50 RPS."""
    assert pinned in script


@pytest.mark.parametrize(
    "pinned",
    [
        "'http_req_duration{kind:chat}': ['p(95)<6000']",
        "http_req_failed: ['rate<0.005']",
        "dropped_iterations: ['count==0']",
    ],
    ids=["chat-p95", "availability", "dropped-iterations"],
)
def test_the_thresholds_are_the_slo_values(script: str, pinned: str) -> None:
    """Criteria 14-16 and the unhappy path: catches a loosened SLO bound, a p95
    measured over embeddings too, or a missing dropped-iterations threshold that
    lets a run under 50 RPS pass."""
    assert pinned in script


def test_the_mix_is_driven_by_the_global_iteration_counter(script: str) -> None:
    """Criteria 12-13: catches a mix from `Math.random()` (not reproducible run to
    run) or from a per-VU counter (which repeats across VUs)."""
    assert re.search(r"import\s+exec\s+from\s+'k6/execution'", script)
    assert "exec.scenario.iterationInTest" in script
    assert "const bucket = iteration % 100;" in script


def test_the_bucket_boundaries_are_checked_chat_first_then_stream(script: str) -> None:
    """Criterion 12: every boundary match is scanned. Catches the two
    checks in reverse order, which sends 90% to streaming and none to plain chat."""
    eighty = [match.start() for match in re.finditer(r"bucket < 80\b", script)]
    ninety = [match.start() for match in re.finditer(r"bucket < 90\b", script)]
    assert len(eighty) == 1 and len(ninety) == 1
    assert eighty[0] < ninety[0]


def test_the_boundaries_in_the_script_produce_80_10_10_and_a_20_percent_repeat(
    script: str,
) -> None:
    """Criteria 12-13: re-derives the mix from the numbers the script carries.
    Catches any modulus or boundary set (e.g. `bucket < 85`) whose buckets are
    not 80/10/10 with 18 of the 90 chat buckets repeating."""
    moduli = {int(value) for value in re.findall(r"iteration % (\d+);", script)}
    boundaries = sorted(int(value) for value in re.findall(r"bucket < (\d+)\b", script))
    repeats = {int(value) for value in re.findall(r"bucket % (\d+) === 0", script)}
    assert len(moduli) == 1 and len(boundaries) == 2 and len(repeats) == 1
    (modulus,) = moduli
    chat_below, stream_below = boundaries
    (repeat_every,) = repeats

    counts = {"chat": 0, "stream": 0, "embed": 0}
    repeated = 0
    for bucket in range(modulus):
        if bucket < chat_below:
            kind = "chat"
        elif bucket < stream_below:
            kind = "stream"
        else:
            kind = "embed"
        counts[kind] += 1
        if kind != "embed" and bucket % repeat_every == 0:
            repeated += 1
    assert modulus == 100
    assert counts == {"chat": 80, "stream": 10, "embed": 10}
    assert repeated == 18
    assert repeated / (counts["chat"] + counts["stream"]) == 0.2


@pytest.mark.parametrize(
    "pinned",
    [
        "'agent-default'",
        "'embed'",
        "'X-Tenant': 'load'",
        "Authorization: `Bearer ${__ENV.LOAD_API_KEY}`",
        "__ENV.GATEWAY_URL",
    ],
)
def test_requests_use_the_load_tenant_aliases_and_environment_key(script: str, pinned: str) -> None:
    """Criteria 12 and 18: catches a different tenant or alias (budget and cache
    booked elsewhere), or a key or address written into the script."""
    assert pinned in script


@pytest.mark.parametrize("pinned", ["kind: 'chat'", "kind: 'embed'"])
def test_requests_are_tagged_so_the_chat_threshold_excludes_embeddings(
    script: str, pinned: str
) -> None:
    """Criterion 14: catches an untagged request, for which the `{kind:chat}`
    threshold has no samples and passes vacuously."""
    assert pinned in script


def test_unique_prompts_carry_the_iteration_counter(script: str) -> None:
    """Criterion 13 and risk 3: catches unique prompts that do not interpolate the
    counter, which would make every chat an exact-cache hit."""
    assert "${iteration}" in script


def test_the_summary_is_written_to_the_mounted_results_directory(script: str) -> None:
    """Criterion 17: catches no `handleSummary`, or a summary path outside the
    `/results` mount, which is lost when the container is removed."""
    assert "export function handleSummary(data)" in script
    assert "'/results/summary.json'" in script


def test_the_script_carries_no_key_literal(script: str) -> None:
    """Criterion 18: catches a generated key or its digest pasted into the script."""
    assert re.search(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])", script) is None
