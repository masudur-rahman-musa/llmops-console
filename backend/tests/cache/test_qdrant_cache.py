r"""`QdrantSemanticCache`: the collection it creates, the neighbours it finds, and its silence.

Two halves, the same split `tests/cache/test_redis_cache.py` makes and for the
same reasons.

The half that needs a REAL Qdrant is the half whose claims are Qdrant semantics:
the round trip through a collection, the lazy creation and the vector size it is
created with, the second write that creates nothing, the three payload filters
and the score the threshold is applied to. Qdrant is required by an explicit
conditional on `OPSCONSOLE_TEST_QDRANT_URL`, and the fixture
skips with that variable's name in the reason. Deliberately NOT
`OPSCONSOLE_QDRANT_URL`, which `Settings` itself reads: exporting that one to run
this file would hand every other test in the suite a `qdrant` readiness probe it
never asked for, and the root `tests/conftest.py` clears it for exactly that
reason.

The half that needs NOTHING is the failure half, over an `httpx.MockTransport`
that opens no socket. Those four rows are where this class earns its contract: a
Qdrant that is unreachable, one that answers a non-2xx to a search, one that
answers a non-2xx to an upsert, and a stored payload this build cannot read are
each a MISS or a swallow. Nothing in this package raises for a store reason
(plan decision G), because turning a Qdrant incident into 5xx on requests a
provider would have served is the failure this tier is arranged to make
impossible.

a project rule: one session-scoped client, built once. Per-test isolation is a
unique `collection_prefix` handed to the cache, never a fresh client and never a
sweep over collections this suite does not own; the session's own collections
are removed at the end by name, and they all begin with `SESSION_PREFIX`.

Contract this file assumes, beyond the one recorded in
`tests/cache/test_vectors.py`: an entry's Qdrant payload carries `alias`,
`system_hash`, `tools_hash` and `answer`, where `answer` is
`ChatResponse.model_dump_json()` - the same neutral value the exact tier stores - and a search reads the top hit's `score` and `payload`
from Qdrant's `{"result": [...]}` body.

The exact-0.97 boundary is asserted in `tests/cache/test_vectors.py` and not
here: Qdrant normalises stored vectors in float32, so a value engineered to sit
exactly on the threshold in float64 can land a few ulps either side of it after
that conversion, and a case that flips on a rounding mode would be red for a
reason that has nothing to do with the comparison. The pair asserted here - one
hundredth over is a hit, one hundredth under is a miss - is what catches a query
sent with no threshold at all.
"""

from __future__ import annotations

import httpx
import pytest

from app.cache.store import RESPONSE_CACHE_TTL_SECONDS
from app.cache.vectors import QdrantSemanticCache
from app.providers.base import ChatResponse

from .conftest import OTHER_TENANT, START, TENANT, FakeClock, cached_answer
from .test_vectors import (
    ABOVE_THRESHOLD,
    ALIAS,
    BELOW_THRESHOLD,
    DATA_CLASS,
    FILTER_ROWS,
    PROBE,
    PROVIDER,
    REGION,
    SYSTEM_HASH,
    TOOLS_HASH,
)

UNREACHABLE_URL = "http://qdrant.invalid:6333"


# --------------------------------------------------------------------------
# the failure half: no socket, no service
# --------------------------------------------------------------------------


def cache_over(handler: httpx.MockTransport) -> QdrantSemanticCache:
    """A cache over a transport that answers however the row needs it to."""
    return QdrantSemanticCache(httpx.Client(transport=handler, base_url=UNREACHABLE_URL))


def raising(error: Exception) -> httpx.MockTransport:
    """A transport whose every request raises, which is what an unreachable Qdrant is."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise error

    return httpx.MockTransport(handle)


def answering(status: int, body: object) -> httpx.MockTransport:
    """A transport answering one status and one JSON body to every request."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, request=request)

    return httpx.MockTransport(handle)


TRANSPORT_FAILURES: list[tuple[str, Exception]] = [
    ("a refused connection", httpx.ConnectError("connection refused")),
    ("a connect timeout", httpx.ConnectTimeout("timed out connecting")),
    ("a hung server hitting the read timeout", httpx.ReadTimeout("timed out reading")),
    ("a connection dropped mid-request", httpx.RemoteProtocolError("server disconnected")),
]


@pytest.mark.parametrize(
    ("case", "error"), TRANSPORT_FAILURES, ids=[row[0] for row in TRANSPORT_FAILURES]
)
def test_a_search_against_an_unreachable_qdrant_is_a_miss(case: str, error: Exception) -> None:
    """Catches a search that lets `httpx.HTTPError` escape to the route: a cache is not a
    control, so a Qdrant outage has to look exactly like a miss - anything else turns an
    incident into 5xx on requests the provider would have served. The four rows are the
    errors the client actually raises, not one: a bare `except httpx.ConnectError` passes
    the first row and drops the other three."""
    assert (
        cache_over(raising(error)).search(
            TENANT,
            PROBE,
            alias=ALIAS,
            system_hash=SYSTEM_HASH,
            tools_hash=TOOLS_HASH,
            provider=PROVIDER,
            region=REGION,
            data_class=DATA_CLASS,
        )
        is None
    )


@pytest.mark.parametrize(
    ("case", "error"), TRANSPORT_FAILURES, ids=[row[0] for row in TRANSPORT_FAILURES]
)
def test_a_put_against_an_unreachable_qdrant_is_swallowed(case: str, error: Exception) -> None:
    """Catches the swallow being applied to the read path alone: the write happens AFTER a
    provider has already answered and been paid, so an escaping error there would fail a
    request that was fully served - the most expensive way to lose a response there is."""
    cache_over(raising(error)).put(
        TENANT,
        PROBE,
        cached_answer(),
        alias=ALIAS,
        system_hash=SYSTEM_HASH,
        tools_hash=TOOLS_HASH,
        provider=PROVIDER,
        region=REGION,
        data_class=DATA_CLASS,
    )


def hit_body(payload: object) -> dict[str, object]:
    """A Qdrant search answer carrying one hit well over the threshold, and that payload."""
    return {"result": [{"id": 1, "version": 0, "score": 1.0, "payload": payload}], "status": "ok"}


def readable_payload() -> dict[str, object]:
    """The payload a healthy entry carries: the six filters and a neutral answer as JSON."""
    return {
        "alias": ALIAS,
        "system_hash": SYSTEM_HASH,
        "tools_hash": TOOLS_HASH,
        "provider": PROVIDER,
        "region": REGION,
        "data_class": DATA_CLASS,
        "answer": cached_answer().model_dump_json(),
    }


NON_2XX_ROWS: list[tuple[str, int]] = [
    ("a collection that does not exist", 404),
    ("a request Qdrant refused", 400),
    ("an unauthorised request", 403),
    ("a server error", 500),
    ("a Qdrant that is still loading", 503),
]


@pytest.mark.parametrize(("case", "status"), NON_2XX_ROWS, ids=[row[0] for row in NON_2XX_ROWS])
def test_a_non_2xx_answer_to_a_search_is_a_miss(case: str, status: int) -> None:
    """Catches a search that calls `raise_for_status()` and lets the `HTTPStatusError` out, and
    one that reads the body of a non-2xx as if it were results. The body handed over is a
    WELL-FORMED hit at score 1.0 with a readable answer in it, so nothing but the status can
    refuse it: against an error-shaped body the payload guard would refuse it too, and this
    case would be green for a search that never looks at the status at all. The 404 row is
    the ordinary one and the reason this cannot be an error - a tenant that has never had an
    answer cached has no collection, so the FIRST semantic request every tenant makes takes
    this path."""
    assert (
        cache_over(answering(status, hit_body(readable_payload()))).search(
            TENANT,
            PROBE,
            alias=ALIAS,
            system_hash=SYSTEM_HASH,
            tools_hash=TOOLS_HASH,
            provider=PROVIDER,
            region=REGION,
            data_class=DATA_CLASS,
        )
        is None
    )


@pytest.mark.parametrize(("case", "status"), NON_2XX_ROWS, ids=[row[0] for row in NON_2XX_ROWS])
def test_a_non_2xx_answer_to_a_put_is_swallowed(case: str, status: int) -> None:
    """Catches a write path that raises on a refused create or a refused upsert: a
    collection created with the wrong vector size rejects every later write for that tenant
    (plan §5), and if that rejection escaped, the tenant's every request would 5xx after the
    provider had already answered - a cache failure taking down a working gateway."""
    cache_over(answering(status, {"status": {"error": "no"}})).put(
        TENANT,
        PROBE,
        cached_answer(),
        alias=ALIAS,
        system_hash=SYSTEM_HASH,
        tools_hash=TOOLS_HASH,
        provider=PROVIDER,
        region=REGION,
        data_class=DATA_CLASS,
    )


UNREADABLE_PAYLOADS: list[tuple[str, object]] = [
    (
        "an answer that is not JSON",
        {
            "alias": ALIAS,
            "system_hash": SYSTEM_HASH,
            "tools_hash": TOOLS_HASH,
            "answer": "not json at all",
        },
    ),
    (
        "an answer this build cannot validate",
        {
            "alias": ALIAS,
            "system_hash": SYSTEM_HASH,
            "tools_hash": TOOLS_HASH,
            "answer": '{"model": "m", "content": "hi"}',
        },
    ),
    (
        "a payload written by a build that had no answer field",
        {"alias": ALIAS, "system_hash": SYSTEM_HASH, "tools_hash": TOOLS_HASH},
    ),
    ("no payload at all", None),
]


@pytest.mark.parametrize(
    ("case", "payload"), UNREADABLE_PAYLOADS, ids=[row[0] for row in UNREADABLE_PAYLOADS]
)
def test_a_stored_payload_this_build_cannot_read_is_a_miss(case: str, payload: object) -> None:
    """Catches a search that trusts whatever is in the payload: an entry written by an older
    build, one whose `ChatResponse` has since gained a required field, or one written by
    another process would raise a `ValidationError`, a `KeyError` or a `TypeError` out of a
    lookup and 500 a request that only wanted a cache probe. Every row is a hit at score
    1.0, so the threshold cannot be what makes any of them absent."""
    assert (
        cache_over(answering(200, hit_body(payload))).search(
            TENANT,
            PROBE,
            alias=ALIAS,
            system_hash=SYSTEM_HASH,
            tools_hash=TOOLS_HASH,
            provider=PROVIDER,
            region=REGION,
            data_class=DATA_CLASS,
        )
        is None
    )


# --------------------------------------------------------------------------
# against a real Qdrant
# --------------------------------------------------------------------------


@pytest.mark.db
def test_the_qdrant_client_is_the_shared_session_client(
    cache: QdrantSemanticCache, qdrant_client: httpx.Client, request: pytest.FixtureRequest
) -> None:
    """Catches a file that builds its own Qdrant client instead of taking the shared one:
    the cache under test must hold the session client from `conftest.py`, and that client
    is built once per run. Asking for `qdrant_client` a second time hands back the identical
    object, which is what "built once" means - a per-test rebuild would answer a new one."""
    assert cache._client is qdrant_client
    assert request.getfixturevalue("qdrant_client") is qdrant_client


@pytest.fixture
def cache(qdrant_client: httpx.Client, collection_prefix: str) -> QdrantSemanticCache:
    """The cache under test, namespaced to this test and sharing the session's client.

    `qdrant_client` and `collection_prefix` come from the top `tests/conftest.py`, built
    once per run; this file builds no client of its own.
    """
    return QdrantSemanticCache(qdrant_client, collection_prefix=collection_prefix)


def put(
    cache: QdrantSemanticCache,
    *,
    tenant: str = TENANT,
    vector: tuple[float, ...] = PROBE,
    answer: ChatResponse | None = None,
    alias: str = ALIAS,
    system_hash: str = SYSTEM_HASH,
    tools_hash: str = TOOLS_HASH,
    provider: str = PROVIDER,
    region: str = REGION,
    data_class: str = DATA_CLASS,
) -> ChatResponse:
    """Store one entry with the base arrangement and hand back what was stored."""
    stored = cached_answer() if answer is None else answer
    cache.put(
        tenant,
        vector,
        stored,
        alias=alias,
        system_hash=system_hash,
        tools_hash=tools_hash,
        provider=provider,
        region=region,
        data_class=data_class,
    )
    return stored


def find(
    cache: QdrantSemanticCache,
    *,
    tenant: str = TENANT,
    vector: tuple[float, ...] = PROBE,
    alias: str = ALIAS,
    system_hash: str = SYSTEM_HASH,
    tools_hash: str = TOOLS_HASH,
    provider: str = PROVIDER,
    region: str = REGION,
    data_class: str = DATA_CLASS,
) -> ChatResponse | None:
    """One search with the base arrangement, so a case changes exactly what it names."""
    return cache.search(
        tenant,
        vector,
        alias=alias,
        system_hash=system_hash,
        tools_hash=tools_hash,
        provider=provider,
        region=region,
        data_class=data_class,
    )


def collection_config(client: httpx.Client, name: str) -> httpx.Response:
    """Qdrant's own description of a collection, or its 404 when there is not one."""
    return client.get(f"/collections/{name}")


@pytest.mark.db
def test_a_stored_answer_round_trips_through_qdrant(cache: QdrantSemanticCache) -> None:
    """Catches an answer written with `str(answer)` or read back as raw bytes: the entry
    travels as JSON in the point payload and comes back as a `ChatResponse`, so a round trip
    that lost the tool calls or the usage would serve a hit the output path renders
    differently from the answer that was cached."""
    stored = put(cache)

    read = find(cache)

    assert isinstance(read, ChatResponse)
    assert read == stored


def test_a_search_before_any_write_is_a_miss(cache: QdrantSemanticCache) -> None:
    """Catches a search that treats a missing collection as an error rather than a miss:
    every tenant's first semantic request takes this path, so it is the hot path and not an
    edge case, and it is asserted against a real Qdrant because the 404 that makes it is
    Qdrant's, not the double's."""
    assert find(cache) is None


def test_a_search_before_any_write_creates_no_collection(
    cache: QdrantSemanticCache, qdrant_client: httpx.Client
) -> None:
    """Catches a search that creates the collection it is about to query - the obvious way to
    make the case above pass: every ineligible tenant would then own an empty collection
    forever, and the lazy-creation claim below would be untestable because the collection
    would already exist with a vector size nobody wrote."""
    find(cache)

    assert collection_config(qdrant_client, cache.collection(TENANT)).status_code == 404


def test_the_collection_is_created_on_the_first_write_with_the_written_vector_size(
    cache: QdrantSemanticCache, qdrant_client: httpx.Client
) -> None:
    """Catches a collection created with a hardcoded size - 384, the fastembed default the
    plan explicitly refused (decision F) - or with the wrong distance. A collection created
    with the wrong size rejects every later write for that tenant, silently, forever, and a
    Euclidean distance makes every score in this file meaningless while the round trip still
    passes. The size asserted is the length of the vector that was actually written."""
    name = cache.collection(TENANT)
    assert collection_config(qdrant_client, name).status_code == 404

    put(cache)

    described = collection_config(qdrant_client, name)
    assert described.status_code == 200
    vectors = described.json()["result"]["config"]["params"]["vectors"]
    assert vectors["size"] == len(PROBE)
    assert vectors["distance"] == "Cosine"


def test_a_second_write_to_the_same_tenant_creates_nothing(cache: QdrantSemanticCache) -> None:
    """Catches a write path that recreates the collection every time - the shape a `PUT
    /collections` issued unconditionally takes once someone adds a delete to make it
    idempotent: the first entry would be gone after the second write and the tier would hold
    exactly one answer per tenant. The first entry surviving the second write is what says
    the collection was left alone, and both entries are read back so a cache that lost
    everything cannot pass by symmetry."""
    first = put(cache, answer=cached_answer(content="the first answer"))
    second = put(
        cache, vector=(0.0, 1.0, 0.0, 0.0, 0.0), answer=cached_answer(content="the second answer")
    )

    assert find(cache) == first
    assert find(cache, vector=(0.0, 1.0, 0.0, 0.0, 0.0)) == second


def test_a_neighbour_above_the_threshold_is_a_hit(cache: QdrantSemanticCache) -> None:
    """Catches a query whose vector is sent unnormalised or whose score is read from the
    wrong field: a similar - not identical - probe is the entire point of this tier, so a
    round trip that only ever matches the exact vector it stored is an expensive second copy
    of the exact cache."""
    stored = put(cache, vector=ABOVE_THRESHOLD)

    assert find(cache) == stored


def test_a_neighbour_below_the_threshold_is_a_miss(cache: QdrantSemanticCache) -> None:
    """Catches a search sent with no threshold at all: Qdrant returns its nearest neighbour
    whatever the score, so without a threshold every tenant's first stored answer would be
    served to every later question, however unrelated. The pair with the case above is the
    boundary this file asserts; the inclusive-at-exactly-0.97 half is in
    `tests/cache/test_vectors.py`, where the arithmetic is float64 and exact."""
    put(cache, vector=BELOW_THRESHOLD)

    assert find(cache) is None


@pytest.mark.parametrize(
    ("case", "alias", "system_hash", "tools_hash", "provider", "region", "data_class"),
    FILTER_ROWS,
    ids=[row[0] for row in FILTER_ROWS],
)
def test_a_query_differing_in_one_filter_field_misses_at_cosine_one(
    cache: QdrantSemanticCache,
    case: str,
    alias: str,
    system_hash: str,
    tools_hash: str,
    provider: str,
    region: str,
    data_class: str,
) -> None:
    """Catches a Qdrant filter built over fewer than all six payload fields, one row per
    field - the in-process implementation can hold all six and this one could still send a
    `must` clause short. The query vector is IDENTICAL to the stored one, so the score
    cannot be what refuses it, and a dropped clause means the nearest neighbour is another
    conversation's answer (plan §5)."""
    put(cache)

    assert (
        find(
            cache,
            alias=alias,
            system_hash=system_hash,
            tools_hash=tools_hash,
            provider=provider,
            region=region,
            data_class=data_class,
        )
        is None
    )


def test_a_query_matching_all_three_filters_hits_at_cosine_one(
    cache: QdrantSemanticCache,
) -> None:
    """Catches a filter so strict nothing matches - a `must` clause on a payload key the
    write never sets, which is the easy half of getting the three rows above green. Every
    filter row asserts a miss, so a search that always returns absent passes all of them."""
    stored = put(cache)

    assert find(cache) == stored


def test_one_tenant_cannot_read_another_tenants_live_entry(cache: QdrantSemanticCache) -> None:
    """Catches a collection name that lost its tenant, or a search that queries a collection
    the tenant does not own: the vector handed over is a REAL, live entry for the other
    tenant at cosine 1.0, so "nothing stored" cannot be what refuses it, and the closing
    reads prove the refusal is not a cache that lost both entries. One tenant reading
    another's answer is the failure a per-tenant collection exists to make impossible.

    Nothing but the tenant differs between the two entries - same vector, same alias, same
    hashes - so no payload filter can be what separates them and only the collection name
    can."""
    mine = put(cache, tenant=TENANT, answer=cached_answer(content="the answer for evals"))
    theirs = put(cache, tenant=OTHER_TENANT, answer=cached_answer(content="the answer for cuan"))

    assert find(cache, tenant=TENANT) == mine
    assert find(cache, tenant=OTHER_TENANT) == theirs


def test_a_tenant_with_nothing_stored_cannot_read_a_live_entry(
    cache: QdrantSemanticCache,
) -> None:
    """Catches the same failure in the arrangement the case above cannot reach: with only ONE
    tenant holding an entry, a cache that queried one shared collection would return that
    entry to everybody, while the two-entry case above could still pass because each tenant
    happened to have its own answer at the identical distance."""
    put(cache, tenant=OTHER_TENANT, answer=cached_answer(content="the answer for cuan"))

    assert find(cache, tenant=TENANT) is None


# --------------------------------------------------------------------------
# new filter fields: provider, region, data_class
# --------------------------------------------------------------------------


def test_the_search_filters_on_provider_region_and_data_class() -> None:
    """Catches a search request that omits one of the three new `must` clauses: a filter
    without provider allows an answer made via a different provider to be served; without
    region it crosses a data-residency boundary; without data_class it hands a PII answer
    to a public caller. All three must appear in the Qdrant `filter.must` list."""
    from .test_vectors import RecordingTransport

    transport = RecordingTransport()
    import json

    import httpx

    client = httpx.Client(transport=transport, base_url="http://qdrant.invalid:6333")
    cache = QdrantSemanticCache(client)

    cache.search(
        TENANT,
        PROBE,
        alias=ALIAS,
        system_hash=SYSTEM_HASH,
        tools_hash=TOOLS_HASH,
        provider=PROVIDER,
        region=REGION,
        data_class=DATA_CLASS,
    )

    assert len(transport.requests) == 1
    sent = json.loads(transport.requests[0].content)
    must_keys = {clause["key"] for clause in sent["filter"]["must"]}
    assert "provider" in must_keys
    assert "region" in must_keys
    assert "data_class" in must_keys


def test_a_put_payload_carries_provider_region_and_data_class() -> None:
    """Catches a write that omits provider, region or data_class from the Qdrant payload:
    if the payload is missing one of them, the search filter for that field can never match
    and the tier stores answers it can never serve -- filling Qdrant with unreachable entries
    while every request misses."""
    from .test_vectors import RecordingTransport

    transport = RecordingTransport()
    import json

    import httpx

    client = httpx.Client(transport=transport, base_url="http://qdrant.invalid:6333")
    cache = QdrantSemanticCache(client)

    cache.put(
        TENANT,
        PROBE,
        cached_answer(),
        alias=ALIAS,
        system_hash=SYSTEM_HASH,
        tools_hash=TOOLS_HASH,
        provider=PROVIDER,
        region=REGION,
        data_class=DATA_CLASS,
    )

    # Two requests: the collection existence check and then the upsert.
    upsert_request = next(
        r for r in transport.requests if r.method == "PUT" and "points" in r.url.path
    )
    body = json.loads(upsert_request.content)
    payload = body["points"][0]["payload"]
    assert payload.get("provider") == PROVIDER
    assert payload.get("region") == REGION
    assert payload.get("data_class") == DATA_CLASS


@pytest.mark.db
def test_an_entry_past_the_ttl_is_not_served_by_a_real_qdrant(
    qdrant_client: httpx.Client, collection_prefix: str
) -> None:
    """Catches an implementation that passes the boundary test in Python only: the Qdrant
    `range` filter on `stored_at` must actually drop an over-age point server-side, not just
    the Python-side check in the in-process tier."""
    clocked_cache = QdrantSemanticCache(
        qdrant_client, collection_prefix=collection_prefix, clock=FakeClock(START)
    )

    # Write at t0, read at t0 — entry is 0 seconds old, must hit.
    stored = put(clocked_cache)
    assert find(clocked_cache) == stored

    # Advance to TTL+1: entry is now over-age, Qdrant range filter must drop it.
    clocked_cache._clock.now = START + RESPONSE_CACHE_TTL_SECONDS + 1  # type: ignore[attr-defined]
    assert find(clocked_cache) is None

    # Write a fresh entry at the advanced clock moment; it is 0 seconds old and must hit.
    fresh = put(clocked_cache, answer=cached_answer(content="fresh answer"))
    assert find(clocked_cache) == fresh
