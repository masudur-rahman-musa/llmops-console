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

import os
import uuid
from collections.abc import Iterator

import httpx
import pytest

from opsconsole.cache.vectors import QdrantSemanticCache
from opsconsole.providers.base import ChatResponse

from .conftest import OTHER_TENANT, TENANT, cached_answer
from .test_vectors import (
    ABOVE_THRESHOLD,
    ALIAS,
    BELOW_THRESHOLD,
    FILTER_ROWS,
    PROBE,
    SYSTEM_HASH,
    TOOLS_HASH,
)

QDRANT_URL_VAR = "OPSCONSOLE_TEST_QDRANT_URL"
QDRANT_URL = os.environ.get(QDRANT_URL_VAR)

SESSION_PREFIX = f"opsconsole-test-{uuid.uuid4().hex[:8]}-"
"""Every collection this run creates begins with it, so the teardown can name its
own collections and touch nothing else."""

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
            TENANT, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
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
    )


def hit_body(payload: object) -> dict[str, object]:
    """A Qdrant search answer carrying one hit well over the threshold, and that payload."""
    return {"result": [{"id": 1, "version": 0, "score": 1.0, "payload": payload}], "status": "ok"}


def readable_payload() -> dict[str, object]:
    """The payload a healthy entry carries: the three filters and a neutral answer as JSON."""
    return {
        "alias": ALIAS,
        "system_hash": SYSTEM_HASH,
        "tools_hash": TOOLS_HASH,
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
            TENANT, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
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
            TENANT, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
        )
        is None
    )


# --------------------------------------------------------------------------
# against a real Qdrant
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qdrant_client() -> Iterator[httpx.Client]:
    """One client for the whole session, built once.

    The skip is an explicit conditional on `OPSCONSOLE_TEST_QDRANT_URL` (standing
    rule 9), never a swallowed connection error: this cache answers absent on
    every failure, so a suite that could not reach Qdrant would otherwise report
    success having checked nothing at all - the trap this class's own semantics
    set for its tests.

    The teardown removes the collections THIS run created, found by
    `SESSION_PREFIX`. It is not a sweep: a shared Qdrant may hold collections
    this suite does not own, and a project rule's namespace is what keeps them
    out of reach in the first place.
    """
    if QDRANT_URL is None:
        pytest.skip(f"{QDRANT_URL_VAR} is not set: start Qdrant and export it to run these")
    client = httpx.Client(base_url=QDRANT_URL, timeout=httpx.Timeout(5.0))
    yield client
    listed = client.get("/collections")
    if listed.status_code < 300:
        for entry in listed.json()["result"]["collections"]:
            if str(entry["name"]).startswith(SESSION_PREFIX):
                client.delete(f"/collections/{entry['name']}")
    client.close()


@pytest.fixture
def collection_prefix() -> str:
    """A collection namespace unique to one test, which is the whole of its isolation."""
    return f"{SESSION_PREFIX}{uuid.uuid4().hex[:8]}-cache_"


@pytest.fixture
def cache(qdrant_client: httpx.Client, collection_prefix: str) -> QdrantSemanticCache:
    """The cache under test, namespaced to this test and sharing the session's client."""
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
) -> ChatResponse:
    """Store one entry with the base arrangement and hand back what was stored."""
    stored = cached_answer() if answer is None else answer
    cache.put(tenant, vector, stored, alias=alias, system_hash=system_hash, tools_hash=tools_hash)
    return stored


def find(
    cache: QdrantSemanticCache,
    *,
    tenant: str = TENANT,
    vector: tuple[float, ...] = PROBE,
    alias: str = ALIAS,
    system_hash: str = SYSTEM_HASH,
    tools_hash: str = TOOLS_HASH,
) -> ChatResponse | None:
    """One search with the base arrangement, so a case changes exactly what it names."""
    return cache.search(tenant, vector, alias=alias, system_hash=system_hash, tools_hash=tools_hash)


def collection_config(client: httpx.Client, name: str) -> httpx.Response:
    """Qdrant's own description of a collection, or its 404 when there is not one."""
    return client.get(f"/collections/{name}")


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
    ("case", "alias", "system_hash", "tools_hash"),
    FILTER_ROWS,
    ids=[row[0] for row in FILTER_ROWS],
)
def test_a_query_differing_in_one_filter_field_misses_at_cosine_one(
    cache: QdrantSemanticCache, case: str, alias: str, system_hash: str, tools_hash: str
) -> None:
    """Catches a Qdrant filter built over two of the three payload fields, one row per
    field - the in-process implementation can hold all three and this one still send a
    `must` clause short. The query vector is IDENTICAL to the stored one, so the score
    cannot be what refuses it, and a dropped clause means the nearest neighbour is another
    conversation's answer (plan §5)."""
    put(cache)

    assert find(cache, alias=alias, system_hash=system_hash, tools_hash=tools_hash) is None


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
