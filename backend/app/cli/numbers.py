r"""The two commands behind `make numbers`: real traffic in, then a check of every tab.

`ops-send-traffic` sends six fixed chat requests through the gateway on the cheap
alias, so the ledger, the request list and the Prometheus series all fill through
the real app. One request carries an email address, so the redaction count moves,
and one is sent twice, so the cache hit rate is not null.

`ops-check-console` then reads all four `/console/*` pages with the admin key and
fails when a tab would show nothing. "Nothing" is spelled out per page in
`REQUIRED`: a source in `error` or `unconfigured`, a list that is empty, a number
that is null. The fields in `ALLOWED_EMPTY` stay empty on a healthy local stack,
because only a real failure fills them and `make numbers` stages none. The check
retries for up to `CHECK_DEADLINE_SECONDS`, since Prometheus scrapes every five
seconds and the first pass can land before the first scrape
(`test_check_retries_until_full_then_passes`).

No row is written here. Both commands only talk to the gateway's HTTP API, and the
keys come from the environment `make numbers` makes for one run. A missing key is
exit 2, the same code `ops-migrate` gives for a missing database, so it reads as
"not configured" rather than "the app is broken".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import httpx

from app.cache import CACHE_ALLOW, CACHE_HEADER

NO_KEY = 2
"""The exit code for a missing key, distinct from 1 (a real failure)."""

DEFAULT_GATEWAY_URL: Final[str] = "http://localhost:8080"
TENANT: Final[str] = "nordlicht"
ALIAS: Final[str] = "agent-default"
TENANT_KEY_VAR: Final[str] = "NUMBERS_TENANT_KEY"
ADMIN_KEY_VAR: Final[str] = "NUMBERS_ADMIN_KEY"
REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0
"""A real model answers in seconds; a minute is the point where waiting longer
would only hide a stuck gateway."""

REPEATED: Final[str] = "In one sentence, what does a help desk do?"
PROMPTS: Final[tuple[str, ...]] = (
    REPEATED,
    REPEATED,
    "Please reply to jane.doe@example.com and say the ticket is closed.",
    "Name one thing a status page should show.",
    "In five words, why do budgets matter?",
    "Say hello in German.",
)
"""Six requests: the first two are equal (a cache hit on the second), the third
carries an email address (a redaction). Short on purpose: a few cents in all.
Each carries `X-Gateway-Cache: allow`, the cache's opt-in, and no `temperature`:
the cheap model refuses that field with a 400. Pinned by
`test_traffic_opts_every_request_into_the_cache`."""

PAGES: Final[tuple[str, ...]] = ("home", "requests", "cost", "quality")
"""The four console tabs, in nav order."""

REQUIRED: Final[Mapping[str, tuple[str, ...]]] = {
    "home": (
        "requests_today",
        "cost_usd_today",
        "p95_seconds",
        "error_rate",
        "last_eval",
        "routing",
    ),
    "requests": ("requests", "tokens", "redactions", "cache_hit_rate"),
    "cost": ("days", "savings", "budgets", "cost_per_request"),
    "quality": ("gate", "runs", "release_markers", "categories"),
}
"""Per page, the fields that must be neither null nor an empty list. `routing` is
full only when it names at least one alias."""

ALLOWED_EMPTY: Final[Mapping[str, tuple[str, ...]]] = {
    "home": ("grafana_url",),
    "quality": ("taxonomy", "blocked"),
    "requests": ("refusals",),
}
"""Not read by the check; written down so the list of what may stay empty has one home."""

SOURCE_OK: Final[str] = "ok"
CHECK_DEADLINE_SECONDS: Final[float] = 60.0
CHECK_INTERVAL_SECONDS: Final[float] = 5.0
"""The Prometheus scrape interval in `infra/local/prometheus.yml`: a retry sooner
than that would read the same empty answer again."""


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--gateway-url",
        default=DEFAULT_GATEWAY_URL,
        help=f"where the gateway listens (default: {DEFAULT_GATEWAY_URL})",
    )
    return parser


def _client(base_url: str, key: str, transport: httpx.BaseTransport | None) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
        transport=transport,
    )


def send_traffic(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Send the six chat requests. `ops-send-traffic`; exit 1 on the first non-200."""
    options = _parser("Send six real chat requests through the gateway.").parse_args(argv)
    key = (os.environ if environ is None else environ).get(TENANT_KEY_VAR)
    if not key:
        print(f"{TENANT_KEY_VAR} is not set; nothing was sent", file=sys.stderr)
        return NO_KEY

    session = f"numbers-{uuid.uuid4().hex[:12]}"
    with _client(options.gateway_url, key, transport) as client:
        for prompt in PROMPTS:
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "X-Tenant": TENANT,
                    "X-Session": session,
                    CACHE_HEADER: CACHE_ALLOW,
                },
                json={"model": ALIAS, "messages": [{"role": "user", "content": prompt}]},
            )
            if response.status_code != httpx.codes.OK:
                print(f"chat request answered HTTP {response.status_code}", file=sys.stderr)
                return 1
    print(f"sent {len(PROMPTS)} chat requests on {ALIAS} for {TENANT}")
    return 0


def page_failures(page: str, body: Mapping[str, Any]) -> list[str]:
    """One line per reason this page would show nothing; empty when it is full."""
    failures: list[str] = []
    for source, state in (body.get("sources") or {}).items():
        # Null means the page reads no such source (Cost and Quality never ask
        # Prometheus), so there is nothing to be down.
        if state is not None and state != SOURCE_OK:
            failures.append(f"{page} sources.{source}: {state}")
    for field in REQUIRED[page]:
        value = body.get(field)
        if value is None or value in ([], {}) or (field == "requests_today" and value <= 0):
            failures.append(f"{page} {field}: empty")
        elif field == "routing" and not value.get("aliases"):
            failures.append(f"{page} routing: no aliases")
    return failures


def _one_pass(client: httpx.Client) -> list[str]:
    failures: list[str] = []
    for page in PAGES:
        response = client.get(f"/console/{page}")
        if response.status_code != httpx.codes.OK:
            failures.append(f"{page}: HTTP {response.status_code}")
            continue
        failures.extend(page_failures(page, response.json()))
    return failures


def check_console(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Read every console page until all are full, or give up. `ops-check-console`."""
    options = _parser("Check that every console tab shows numbers.").parse_args(argv)
    key = (os.environ if environ is None else environ).get(ADMIN_KEY_VAR)
    if not key:
        print(f"{ADMIN_KEY_VAR} is not set; nothing was checked", file=sys.stderr)
        return NO_KEY

    started = monotonic()
    with _client(options.gateway_url, key, transport) as client:
        while True:
            failures = _one_pass(client)
            if not failures:
                print(f"all {len(PAGES)} console pages show numbers")
                return 0
            if monotonic() - started + CHECK_INTERVAL_SECONDS > CHECK_DEADLINE_SECONDS:
                for line in failures:
                    print(line)
                return 1
            sleep(CHECK_INTERVAL_SECONDS)
