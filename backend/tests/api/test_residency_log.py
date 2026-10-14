"""The residency log through the route: one row per served request, and nothing else.

Everything here goes through `TestClient` against an app `create_app` built, so
every claim about a row, a status or a header is settled by a real request rather
than by reading the call site. No backing service: the writer
under `app.state.residency_log` is the in-process one, and the fixtures are the
`fallback_client` / `eu_client` pair from tests/gateway/conftest.py, whose stubs
and breaker are already reset per test.

Installing the writer per test is the one piece of arrangement this file adds,
and it puts the PREVIOUS writer back afterwards: the app is session-scoped and
shared with tests/gateway/test_fallback.py and tests/gateway/test_metrics.py,
which must keep seeing the writer the factory built.

The contract these cases are written against, before it exists:

* `create_app` hangs a `ResidencyLogWriter` on `app.state.residency_log`, the
  in-process one when `settings.database_url` is unset;
* `record_served` in `services/chat_pipeline.py` calls `record_residency` for the
  candidate that SERVED, on all three paths, with `allowed` computed from the
  policy rather than passed;
* an `allowed=false` row is the only thing that moves
  `gateway_residency_violations_total` (decision E). A 451 no longer does, which
  is the changed case in tests/gateway/test_metrics.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.ledger import (
    InMemoryResidencyLogWriter,
    ResidencyLogRow,
    ResidencyLogWriter,
    record_residency,
    region_allowed,
)
from app.main import create_app
from app.model_routing.policy import DataClass, Policy, load_policy
from app.model_routing.resolver import Resolved

from .conftest import (
    EU_ALLOWED_REGIONS,
    EU_CLASS,
    EU_EXCLUDED_PROVIDER,
    EU_FIRST_PROVIDER,
    EU_TENANT,
    FALLBACK_ALIAS,
    FALLBACK_CLASS,
    FALLBACK_TENANT,
    FIRST_PROVIDER,
    FIRST_REGION,
    NO_PROVIDER_KEYS,
    SECOND_PROVIDER,
    SECOND_REGION,
    TEST_KEYS,
    FallbackStubAdapter,
    MetricDelta,
    app_of,
    chat_body,
    embeddings_body,
    post_chat,
    post_embeddings,
    retryable_error,
)

RESIDENCY_VIOLATIONS = "gateway_residency_violations_total"
REQUESTS = "gateway_requests_total"
"""Two of the family names tests/gateway/test_metrics.py declares, spelled here so
this file reads the exposition's name rather than importing another test module."""

GATEWAY_HEADERS = (
    "x-gateway-provider",
    "x-gateway-region",
    "x-gateway-model",
    "x-gateway-cache",
    "x-gateway-redacted",
)
"""The response headers whose VALUES a failed log write must not disturb. The
trace header and the body's `id` are excluded because they differ between any two
requests, which is why the comparison below is against a baseline request rather
than against a literal."""


class RaisingLogWriter:
    """A writer that always fails, and counts the attempts.

    The count is what stops the "nothing changed" cases from being vacuous: a
    route that stopped calling the writer at all would leave every status, header
    and body identical too.
    """

    def __init__(self) -> None:
        self.calls = 0

    def write(self, row: ResidencyLogRow) -> None:
        self.calls += 1
        raise RuntimeError("the residency ledger is unreachable")


def install(client: TestClient, writer: ResidencyLogWriter) -> ResidencyLogWriter:
    """Put a writer on the app behind this client, and hand back the one replaced."""
    app = app_of(client)
    previous: ResidencyLogWriter = app.state.residency_log
    app.state.residency_log = writer
    return previous


def installed(client: TestClient) -> Iterator[InMemoryResidencyLogWriter]:
    """An in-process writer on this client's app for one test, then the old one back."""
    writer = InMemoryResidencyLogWriter()
    previous = install(client, writer)
    try:
        yield writer
    finally:
        install(client, previous)


@pytest.fixture
def log(fallback_client: TestClient) -> Iterator[InMemoryResidencyLogWriter]:
    """The rows the fallback app wrote during this test."""
    yield from installed(fallback_client)


@pytest.fixture
def eu_log(eu_client: TestClient) -> Iterator[InMemoryResidencyLogWriter]:
    """The rows the EU-policy app wrote during this test."""
    yield from installed(eu_client)


@pytest.fixture(scope="session")
def eu_policy(eu_policy_path: Path) -> Policy:
    """The policy the EU app resolves against, for computing the expected answer."""
    return load_policy(eu_policy_path)


def send_chat(client: TestClient) -> Response:
    """One non-streamed completion as the tenant whose alias has two candidates."""
    return post_chat(
        client,
        chat_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )


def send_stream(client: TestClient) -> Response:
    """The same request, streamed."""
    return post_chat(
        client,
        chat_body(model=FALLBACK_ALIAS, stream=True),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )


def send_embeddings(client: TestClient) -> Response:
    """One embeddings request against the same alias and tenant."""
    return post_embeddings(
        client,
        embeddings_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[FALLBACK_TENANT],
        tenant=FALLBACK_TENANT,
        data_class=FALLBACK_CLASS,
    )


def streamed_text(response: Response) -> str:
    """The text deltas of a streamed answer, joined, so the ANSWER can be compared."""
    text = ""
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        for choice in json.loads(payload)["choices"]:
            text += str(choice["delta"].get("content") or "")
    return text


def chat_answer(response: Response) -> object:
    """The one part of a completion body that is the same on two identical requests."""
    return response.json()["choices"][0]["message"]["content"]


def embedding_answer(response: Response) -> object:
    """The vectors an embeddings response carries."""
    return [item["embedding"] for item in response.json()["data"]]


Send = Callable[[TestClient], Response]
Answer = Callable[[Response], object]

PATHS: list[tuple[str, Send, Answer]] = [
    ("the non-streaming chat path", send_chat, chat_answer),
    ("the streaming chat path", send_stream, streamed_text),
    ("the embeddings path", send_embeddings, embedding_answer),
]

PATH_IDS = [case for case, _, _ in PATHS]


# --------------------------------------------------------------------------
# one row per served request, on each of the three paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "send", "answer"), PATHS, ids=PATH_IDS)
def test_a_served_request_writes_exactly_one_row(
    fallback_client: TestClient,
    log: InMemoryResidencyLogWriter,
    case: str,
    send: Send,
    answer: Answer,
) -> None:
    """The criterion, on all three paths. Catches a write placed on the non-streaming
    return path alone - the streaming row is the one that matters most, because a stream
    is where a provider holds the data longest - and one placed on the chat route only,
    which leaves every embedding the gateway sends abroad unrecorded. Catches a row
    written per candidate or per frame, since the count is exact. The answer is read off
    the response so that a path which returned an error body cannot pass by writing one
    row on the way out."""
    response = send(fallback_client)

    assert response.status_code == 200, response.text
    assert answer(response) not in ("", [], None)
    assert len(log.rows) == 1


@pytest.mark.parametrize(("case", "send", "answer"), PATHS, ids=PATH_IDS)
def test_a_served_row_names_the_request_and_the_candidate(
    fallback_client: TestClient,
    log: InMemoryResidencyLogWriter,
    case: str,
    send: Send,
    answer: Answer,
) -> None:
    """Catches a row filled from the request body rather than from the resolver's answer:
    `provider` would read `agent-default` and the one question this table exists to answer
    - which provider held this tenant's data, in which region - would be unanswerable.
    The provider and region are compared against the headers the SAME response carried, so
    the row and the wire contract cannot drift apart. `data_class` is checked because the
    header is what decides which rule `allowed` was computed under."""
    response = send(fallback_client)
    assert response.status_code == 200, response.text

    row = log.rows[0]
    assert row.tenant == FALLBACK_TENANT
    assert row.alias == FALLBACK_ALIAS
    assert row.data_class is DataClass(FALLBACK_CLASS)
    assert row.provider == response.headers["x-gateway-provider"]
    assert row.region == response.headers["x-gateway-region"]
    assert row.provider == FIRST_PROVIDER
    assert row.region == FIRST_REGION
    assert row.allowed is True
    assert row.ts.tzinfo is not None


# --------------------------------------------------------------------------
# a fallback logs the candidate that served
# --------------------------------------------------------------------------


def test_a_chat_request_that_fell_back_logs_the_serving_provider(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    log: InMemoryResidencyLogWriter,
) -> None:
    """The criterion's own sentence: the row names the provider that SERVED. Catches a row
    written where the first candidate is chosen rather than where it answered, which
    records `anthropic`/`us` for data that never left `self` - a residency log that is
    wrong in the direction that invents a breach - and catches a row per ATTEMPT, which
    would file the failed candidate as having held the data. Neither the header assertion
    in tests/gateway/test_fallback.py nor the breaker touches this row."""
    first = fallback_adapters[FIRST_PROVIDER]
    first.chat_error = retryable_error(FIRST_PROVIDER)

    response = send_chat(fallback_client)

    assert response.status_code == 200, response.text
    assert len(log.rows) == 1
    assert log.rows[0].provider == SECOND_PROVIDER
    assert log.rows[0].region == SECOND_REGION


def test_a_streamed_request_that_fell_back_logs_the_serving_provider(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    log: InMemoryResidencyLogWriter,
) -> None:
    """The same claim on the path where the choice is made at a different moment: the
    stream's candidate is settled when the first event is pulled, so a row written from a
    variable captured before the loop names the first candidate here even where the
    non-streaming path is right."""
    first = fallback_adapters[FIRST_PROVIDER]
    first.stream_error = retryable_error(FIRST_PROVIDER)
    first.stream_error_at = 0

    response = send_stream(fallback_client)

    assert response.status_code == 200, response.text
    assert "[DONE]" in response.text
    assert len(log.rows) == 1
    assert log.rows[0].provider == SECOND_PROVIDER


def test_an_embeddings_request_that_fell_back_logs_the_serving_provider(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    log: InMemoryResidencyLogWriter,
) -> None:
    """Catches the embeddings route passing a candidate its own loop has already moved
    past - the route reaches `record_served` from a different place than the chat route
    does - which would log every fallen-back embedding against a provider that refused
    it."""
    first = fallback_adapters[FIRST_PROVIDER]
    first.embed_error = retryable_error(FIRST_PROVIDER)

    response = send_embeddings(fallback_client)

    assert response.status_code == 200, response.text
    assert len(log.rows) == 1
    assert log.rows[0].provider == SECOND_PROVIDER


# --------------------------------------------------------------------------
# a failing writer changes nothing (decision D, a design note)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "send", "answer"), PATHS, ids=PATH_IDS)
def test_a_failing_writer_changes_no_status_body_or_header(
    fallback_client: TestClient, case: str, send: Send, answer: Answer
) -> None:
    """Decision D through the route. Catches an unguarded write: a Postgres blip would
    take out a gateway whose residency behaviour is unchanged, and on the streaming path
    it would do so with a 200 already on the wire, truncating an answer the client had
    begun reading. The baseline request is what makes this a comparison rather than a
    restatement of the happy path, and the attempt count is what stops a route that
    silently stopped writing rows from passing."""
    original = install(fallback_client, InMemoryResidencyLogWriter())
    failing = RaisingLogWriter()
    try:
        good = send(fallback_client)
        assert good.status_code == 200, good.text

        install(fallback_client, failing)
        bad = send(fallback_client)
    finally:
        install(fallback_client, original)

    assert failing.calls == 1
    assert bad.status_code == good.status_code
    assert set(bad.headers) == set(good.headers)
    for name in GATEWAY_HEADERS:
        assert bad.headers[name] == good.headers[name], name
    assert answer(bad) == answer(good)


# --------------------------------------------------------------------------
# an EU tenant, and the counter the resolver cannot move
# --------------------------------------------------------------------------


def test_an_eu_tenants_served_row_is_allowed_inside_the_eu_regions(
    eu_client: TestClient, eu_log: InMemoryResidencyLogWriter, eu_policy: Policy
) -> None:
    """The criterion "the resolver cannot produce a false one", driven through the route
    for a real EU tenant. Catches an `allowed` computed against the wrong tenant, the
    wrong data class or a region taken from somewhere other than the served candidate:
    the EU policy's alias lists a US provider between two EU ones, so anything that
    resolves or recomputes loosely lands on a region outside `eu_regions` and this case
    sees it as either a false row or a region it does not recognise. `region_allowed` is
    asked the same question independently, so the row agreeing with itself is not enough."""
    response = post_chat(
        eu_client,
        chat_body(model=FALLBACK_ALIAS),
        key=TEST_KEYS[EU_TENANT],
        tenant=EU_TENANT,
        data_class=EU_CLASS,
    )

    assert response.status_code == 200, response.text
    row = eu_log.rows[0]
    assert len(eu_log.rows) == 1
    assert row.provider == EU_FIRST_PROVIDER
    assert row.provider != EU_EXCLUDED_PROVIDER
    assert row.region in EU_ALLOWED_REGIONS
    assert row.allowed is True
    assert region_allowed(eu_policy, EU_TENANT, DataClass(EU_CLASS), row.region) is True


def test_no_served_request_moves_the_violation_counter(
    fallback_client: TestClient,
    fallback_adapters: dict[str, FallbackStubAdapter],
    log: InMemoryResidencyLogWriter,
    metrics: MetricDelta,
) -> None:
    """Decision E's "and on nothing else". Catches a counter moved once per ROW rather
    than once per `allowed=false` row, which is the easy mistake at the call site and
    would fire the design's alert - the one alert that exists for a residency breach - on
    every request the gateway serves. The four rows are what make the zero mean something:
    logging was happening, on all three paths and through a fallback, while the counter
    stayed flat."""
    fallback_adapters[FIRST_PROVIDER].chat_error = retryable_error(FIRST_PROVIDER)
    assert send_chat(fallback_client).status_code == 200
    fallback_adapters[FIRST_PROVIDER].chat_error = None
    assert send_chat(fallback_client).status_code == 200
    assert send_stream(fallback_client).status_code == 200
    assert send_embeddings(fallback_client).status_code == 200

    assert len(log.rows) == 4
    assert all(row.allowed for row in log.rows)
    assert metrics.total(RESIDENCY_VIOLATIONS) == 0
    assert metrics.total(REQUESTS) == 4


def test_a_forced_allowed_false_row_moves_the_violation_counter_once(
    policy: Policy, metrics: MetricDelta
) -> None:
    """The other half of decision E, and the only way to reach it: the resolver cannot
    produce this row, so it is driven at `record_residency` directly. Catches a counter
    that was declared and never fed - which is what "the metric exists and alerts above
    zero" would otherwise mean in practice - and one labelled by provider or region
    instead of tenant, since the delta is read off an exactly-labelled series. The total
    is asserted as well, so a counter moved under two label sets at once fails."""
    log = InMemoryResidencyLogWriter()

    record_residency(
        log,
        policy=policy,
        tenant="nordlicht",
        data_class=DataClass.PUBLIC,
        alias="agent-default",
        chosen=Resolved(provider="anthropic", model="claude-haiku-4-5", region="us"),
    )

    assert log.rows[0].allowed is False
    assert metrics.delta(RESIDENCY_VIOLATIONS, tenant="nordlicht") == 1
    assert metrics.total(RESIDENCY_VIOLATIONS) == 1
    assert metrics.total(REQUESTS) == 0


# --------------------------------------------------------------------------
# the wiring (the unhappy path: no database configured)
# --------------------------------------------------------------------------


def test_the_app_builds_the_in_process_writer_without_a_database_url(
    settings: Settings,
) -> None:
    """The unhappy-path criterion, and the reason the whole suite still needs no backing
    service. Catches a factory that builds a pool unconditionally: `build_pool` opens
    connections, so every app in this session - and every local run without Postgres -
    would fail at construction or hold a pool retrying in the background. The settings
    fixture carries no `database_url`, which the root tests/conftest.py also guarantees by
    clearing the variable."""
    assert settings.database_url is None

    app = create_app(settings, NO_PROVIDER_KEYS)

    assert isinstance(app.state.residency_log, InMemoryResidencyLogWriter)
