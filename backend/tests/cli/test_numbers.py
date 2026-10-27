"""The two commands behind `make numbers`: send real traffic, then check every console tab.

Nothing here touches a network or waits. The gateway and the console API are an
`httpx.MockTransport`, and time is a small fake clock whose `sleep` moves `monotonic`.
The stub pages follow the models in `app.schemas.dashboard`.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.cli.numbers import PAGES as CHECKED_PAGES
from app.cli.numbers import check_console, send_traffic

TENANT_KEY = "tenant-key-for-tests"
ADMIN_KEY = "admin-key-for-tests"
GATEWAY = "http://gateway.test"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

PAGES = ("home", "requests", "cost", "quality")

NOW = "2026-10-26T10:00:00Z"
SOURCES = {"postgres": "ok", "prometheus": "ok"}


# --------------------------------------------------------------------------
# traffic
# --------------------------------------------------------------------------


def run_traffic(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    environ: dict[str, str] | None = None,
    argv: list[str] | None = None,
) -> int:
    env = {"NUMBERS_TENANT_KEY": TENANT_KEY} if environ is None else environ
    return send_traffic(argv or [], environ=env, transport=httpx.MockTransport(handler))


def ok_chat(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": []})


def test_traffic_sends_six_requests_on_agent_default_with_one_email_and_one_repeat() -> None:
    """Catches a sender that posts the wrong count, the wrong alias, no redaction case or no
    repeat: six posts, alias `agent-default`, one body with an email, one body sent twice."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_chat(request)

    assert run_traffic(handler, argv=["--gateway-url", GATEWAY]) == 0

    assert len(seen) == 6
    assert {r.method for r in seen} == {"POST"}
    assert {str(r.url) for r in seen} == {f"{GATEWAY}/v1/chat/completions"}
    assert {json.loads(r.content)["model"] for r in seen} == {"agent-default"}
    bodies = [r.content.decode() for r in seen]
    assert sum(1 for body in bodies if EMAIL.search(body)) == 1
    counts = Counter(bodies)
    assert sorted(counts.values()) == [1, 1, 1, 1, 2]


def test_traffic_opts_every_request_into_the_cache() -> None:
    """Catches a sender whose six requests never reach the cache, so the repeated question is
    not a hit and the hit rate stays empty: each must carry `X-Gateway-Cache: allow`. Catches
    one that sends `temperature` instead, which the cheap model refuses with a 400 (`make
    numbers` then fails on a 502), so no body may carry it."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_chat(request)

    assert run_traffic(handler) == 0

    assert len(seen) == 6
    for request in seen:
        assert request.headers["X-Gateway-Cache"] == "allow"
        assert "temperature" not in json.loads(request.content)


def test_traffic_sends_the_tenant_headers_and_defaults_to_the_local_gateway() -> None:
    """Catches a sender that drops the bearer key, the tenant or the session header, or that
    defaults to some address other than http://localhost:8080."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_chat(request)

    assert run_traffic(handler) == 0

    assert {str(r.url) for r in seen} == {"http://localhost:8080/v1/chat/completions"}
    for request in seen:
        assert request.headers["Authorization"] == f"Bearer {TENANT_KEY}"
        assert request.headers["X-Tenant"] == "nordlicht"
        assert request.headers["X-Session"] != ""


def test_traffic_fails_on_a_non_200(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a sender that ignores a refused request and reports success: a 401 must
    exit 1 and print a line naming the status on stderr."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "no"})

    assert run_traffic(handler) == 1
    assert "401" in capsys.readouterr().err


def test_traffic_without_a_key_exits_2() -> None:
    """Catches a sender that runs with no tenant key and sends unauthenticated traffic:
    a missing NUMBERS_TENANT_KEY exits 2 before any request."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_chat(request)

    assert run_traffic(handler, environ={}) == 2
    assert seen == []


# --------------------------------------------------------------------------
# the console check
# --------------------------------------------------------------------------


class FakeClock:
    """`monotonic` and `sleep` that agree: sleeping moves the clock, nothing waits."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        assert self.now < 1000.0 + 600, "the check never gave up"


def full_pages() -> dict[str, dict[str, Any]]:
    """Four filled pages that follow the models in `app.schemas.dashboard`."""
    return {
        "home": {
            "sources": dict(SOURCES),
            "requests_today": 6,
            "cost_usd_today": 0.02,
            "p95_seconds": 1.2,
            "error_rate": 0.0,
            "last_eval": {
                "id": "run-1",
                "finished_at": NOW,
                "passed": True,
                "check_score": 0.9,
            },
            "routing": {
                "loaded_at": NOW,
                "aliases": [
                    {
                        "name": "agent-default",
                        "purpose": "Everyday chat replies.",
                        "candidates": [{"provider": "openai", "model": "m"}],
                    }
                ],
            },
            # Unset is a legal answer: the Overview just hides the link.
            "grafana_url": None,
        },
        "cost": {
            "sources": dict(SOURCES),
            "days": [
                {
                    "day": "2026-10-26",
                    "tenant": "nordlicht",
                    "provider": "openai",
                    "requests": 6,
                    "cached_requests": 1,
                    "cost_usd": 0.02,
                }
            ],
            "savings": [{"day": "2026-10-26", "tenant": "nordlicht", "saved_usd": 0.001}],
            "savings_usd_total": 0.001,
            "estimated": False,
            "budgets": [
                {"tenant": "nordlicht", "cap_usd": 10.0, "spent_usd": 0.02, "ratio": 0.002}
            ],
            "cost_per_request": [{"alias": "agent-default", "cost_usd": 0.003}],
        },
        "quality": {
            "sources": dict(SOURCES),
            "gate": [
                {
                    "name": "check_score",
                    "label": "Check score",
                    "value": 0.9,
                    "threshold": 0.8,
                    "passed": True,
                    "note": None,
                }
            ],
            "runs": [
                {
                    "id": "run-1",
                    "started_at": NOW,
                    "run_kind": "local",
                    "git_sha": "abc",
                    "passed": True,
                    "metrics": {},
                }
            ],
            "release_markers": [{"git_sha": "abc", "deployed_at": NOW, "status": "deployed"}],
            "categories": [{"category": "refunds", "passed": 3, "total": 3}],
            "taxonomy": [{"tag": "wrong_tone", "count": 1}],
            "blocked": [
                {
                    "git_sha": "abc",
                    "deployed_at": NOW,
                    "pr_url": None,
                    "eval_run_id": None,
                    "regressed": [],
                }
            ],
        },
        "requests": {
            "sources": dict(SOURCES),
            "redactions": [{"kind": "email", "count": 1.0}],
            "refusals": [{"reason": "budget", "count": 1.0}],
            "guardrail_blocks": 0.0,
            "tokens": [{"model": "m", "input": 10.0, "output": 5.0}],
            "cache_hit_rate": 0.16,
            "requests": [
                {
                    "ts": NOW,
                    "tenant": "nordlicht",
                    "alias": "agent-default",
                    "provider": "openai",
                    "model": "m",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cached": False,
                    "cost_usd": 0.003,
                    "prompt_version": None,
                    "trace_id": "t1",
                    "trace_url": None,
                }
            ],
        },
    }


def pages_with(**changes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`full_pages()` with the named fields of the named pages replaced."""
    pages = full_pages()
    for page, fields in changes.items():
        pages[page].update(copy.deepcopy(fields))
    return pages


class Console:
    """A stub console API: serves one set of pages per pass, records every request."""

    def __init__(
        self, passes: list[dict[str, dict[str, Any]]], status: dict[str, int] | None = None
    ) -> None:
        self.passes = passes
        self.status = status or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        page = request.url.path.removeprefix("/console/")
        if page in self.status:
            return httpx.Response(self.status[page], json={"detail": "boom"})
        # Each page is read once per pass, so its own call count is the pass number.
        index = min(self.requests_for(page) - 1, len(self.passes) - 1)
        return httpx.Response(200, json=self.passes[index][page])

    def requests_for(self, page: str) -> int:
        return sum(1 for r in self.requests if r.url.path == f"/console/{page}")

    def home_calls(self) -> int:
        return self.requests_for("home")


def run_check(
    console: Console,
    clock: FakeClock | None = None,
    environ: dict[str, str] | None = None,
) -> int:
    env = {"NUMBERS_ADMIN_KEY": ADMIN_KEY} if environ is None else environ
    clock = clock or FakeClock()
    return check_console(
        [],
        environ=env,
        transport=httpx.MockTransport(console),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_check_passes_when_every_page_is_full() -> None:
    """Catches a check that fails on a good stack, skips a page, or sends no admin key:
    all four pages are read with the bearer key and the exit is 0 with no waiting."""
    console = Console([full_pages()])
    clock = FakeClock()

    assert run_check(console, clock) == 0

    assert {r.url.path for r in console.requests} == {f"/console/{p}" for p in PAGES}
    assert {r.headers["Authorization"] for r in console.requests} == {f"Bearer {ADMIN_KEY}"}
    assert clock.sleeps == []


def test_check_without_a_key_exits_2() -> None:
    """Catches a check that runs with no admin key and reports 401 pages as a stack
    problem: a missing NUMBERS_ADMIN_KEY exits 2 before any request."""
    console = Console([full_pages()])

    assert run_check(console, environ={}) == 2
    assert console.requests == []


def test_check_fails_on_a_source_in_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a check that reads only the page fields and ignores `sources`: Cost with
    `prometheus: error` must exit 1 and name `cost sources.prometheus`."""
    cost = {"sources": {"postgres": "ok", "prometheus": "error"}}

    assert run_check(Console([pages_with(cost=cost)])) == 1

    assert "cost sources.prometheus: error" in capsys.readouterr().out


def test_check_fails_on_a_source_unconfigured(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a check that treats `unconfigured` as fine because it is not `error`: Home
    with `postgres: unconfigured` must exit 1 and name it."""
    home = {"sources": {"postgres": "unconfigured", "prometheus": "ok"}}

    assert run_check(Console([pages_with(home=home)])) == 1

    assert "home sources.postgres: unconfigured" in capsys.readouterr().out


def test_check_skips_a_null_source(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a check that reads a null `prometheus` (Cost and Quality have no Prometheus
    source) as a failing state, or that skips every non-`ok` source: Cost with
    `prometheus: null` and full fields must pass, while Home with `prometheus: unconfigured`
    must still fail and name it."""
    cost = {"sources": {"postgres": "ok", "prometheus": None}}
    assert run_check(Console([pages_with(cost=cost)])) == 0

    home = {"sources": {"postgres": "ok", "prometheus": "unconfigured"}}
    assert run_check(Console([pages_with(home=home)])) == 1
    assert "home sources.prometheus: unconfigured" in capsys.readouterr().out


EMPTY_CASES: list[tuple[str, str, object]] = [
    ("home", "requests_today", 0),
    ("home", "requests_today", None),
    ("home", "cost_usd_today", None),
    ("home", "p95_seconds", None),
    ("home", "error_rate", None),
    ("home", "last_eval", None),
    ("cost", "days", []),
    ("cost", "savings", []),
    ("cost", "budgets", []),
    ("cost", "cost_per_request", []),
    ("quality", "gate", []),
    ("quality", "runs", []),
    ("quality", "release_markers", []),
    ("quality", "categories", []),
    ("requests", "requests", []),
    ("requests", "tokens", []),
    ("requests", "redactions", []),
    ("requests", "cache_hit_rate", None),
]


@pytest.mark.parametrize(
    ("page", "field", "empty"),
    EMPTY_CASES,
    ids=[f"{page}-{field}-{empty!r}" for page, field, empty in EMPTY_CASES],
)
def test_check_fails_on_each_required_field_empty(
    page: str, field: str, empty: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a check that forgets one required field: that one field alone empty or null
    must exit 1 and be named as `<page> <field>`."""
    assert run_check(Console([pages_with(**{page: {field: empty}})])) == 1

    assert f"{page} {field}" in capsys.readouterr().out


def test_check_allows_the_listed_fields_to_be_empty() -> None:
    """Catches a check that demands data only a real failure creates: `taxonomy`, `blocked`
    and `refusals` empty, and an unset `grafana_url`, must still exit 0."""
    pages = pages_with(
        quality={"taxonomy": [], "blocked": []},
        requests={"refusals": []},
    )
    clock = FakeClock()

    assert run_check(Console([pages]), clock) == 0
    assert clock.sleeps == []


def test_check_retries_until_full_then_passes() -> None:
    """Catches a check that fails on the first pass before the first scrape lands: an empty
    first answer then a full one must exit 0, sleeping 5 s on the fake clock between them."""
    empty_cost = pages_with(cost={"days": []})
    console = Console([empty_cost, full_pages()])
    clock = FakeClock()

    assert run_check(console, clock) == 0

    assert clock.sleeps == [5]
    assert console.home_calls() == 2


def test_check_gives_up_after_60_seconds(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a check that retries forever, or gives up early: an always-empty page must
    exit 1 after sleeping in 5 s steps for about 60 s, and print the failing field."""
    console = Console([pages_with(cost={"days": []})])
    clock = FakeClock()

    assert run_check(console, clock) == 1

    assert set(clock.sleeps) == {5}
    assert 55 <= sum(clock.sleeps) <= 60
    assert "cost days" in capsys.readouterr().out


def test_check_fails_on_a_non_200_page(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a check that reads an error body as an empty page, or skips a failing page:
    a 500 on /console/quality must exit 1 with the line `quality: HTTP 500`."""
    console = Console([full_pages()], status={"quality": 500})

    assert run_check(console) == 1

    assert "quality: HTTP 500" in capsys.readouterr().out


def test_the_check_covers_the_four_tabs(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches a check still reading a removed tab (it would 404 on the real stack), one that
    drops a kept tab, and an Overview check that never looks at the routing: a home body with
    an empty `routing.aliases` is the only thing wrong here, so only a check that requires the
    routing can turn it into exit 1."""
    assert CHECKED_PAGES == ("home", "requests", "cost", "quality")

    routing = {"loaded_at": NOW, "aliases": []}

    assert run_check(Console([pages_with(home={"routing": routing})])) == 1

    out = capsys.readouterr().out
    assert "home" in out
    assert "routing" in out
