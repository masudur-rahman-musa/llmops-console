r"""The semantic tier's seam: the cosine rule, the three filters, and the collection name.

Everything here runs in one process. The cosine arithmetic is asserted against
`InProcessSemanticCache`, which is the implementation a bare local run and this
suite get (plan decision D); the Qdrant class appears only where the claim is
about a name or a pre-check, over an `httpx` transport that records requests and
never opens a socket. The cases needing a real Qdrant live in
`tests/cache/test_qdrant_cache.py`.

Four claims, each arranged so that only the control it names can decide it:

* THE THRESHOLD is inclusive at exactly `SEMANTIC_THRESHOLD`. The vectors are
  built so their cosine is an EXACT float, not an approximation that happens to
  round the right way: every neighbour has length exactly 100 and the query
  vector length exactly 1, so the cosine is an integer divided by 100 in every
  order the arithmetic can be written. A `>` written where a `>=` belongs
  reddens on the boundary row and on nothing else.
* THE THREE FILTERS get one miss case each at cosine 1.0 - the SAME vector, so
  distance cannot be what refuses it - and one control row where all three match.
  A query that dropped a filter serves another conversation's answer, and the
  nearest neighbour will happily be one (plan §5).
* TENANT SCOPING is asserted with a vector that is genuinely LIVE for the other
  tenant, so "nothing stored at all" cannot be what makes it pass.
* THE COLLECTION NAME is `cache_{tenant}`, and the tenant is checked against the
  gateway's own identifier rules BEFORE that name is built.
  Qdrant is an unsized store, so that check has no backstop behind it.

Contract this file assumes (the implementation follows it):

    SEMANTIC_COLLECTION_PREFIX: Final[str]  # "cache_"

    class SemanticCache(Protocol):
        def search(self, tenant: str, vector: Sequence[float], *, alias: str,
                   system_hash: str, tools_hash: str) -> ChatResponse | None: ...
        def put(self, tenant: str, vector: Sequence[float], answer: ChatResponse, *,
                alias: str, system_hash: str, tools_hash: str) -> None: ...

    class InProcessSemanticCache:
        def __init__(self) -> None: ...

    class QdrantSemanticCache:
        def __init__(self, client: httpx.Client, *,
                     collection_prefix: str = SEMANTIC_COLLECTION_PREFIX) -> None: ...
        def collection(self, tenant: str) -> str: ...

The `httpx.Client` carries the Qdrant `base_url`; the class sends paths relative
to it. Neither method raises for a store reason - a failure is a miss and a
failed write is swallowed (plan decision G) - but a malformed tenant raises
`IdentifierInvalid` before any effect, exactly as `cache/store.py` does.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import httpx
import pytest

from app.cache.probe import SEMANTIC_THRESHOLD
from app.cache.vectors import (
    SEMANTIC_COLLECTION_PREFIX,
    InProcessSemanticCache,
    QdrantSemanticCache,
    SemanticCache,
)
from app.core.validation import MAX_IDENTIFIER_LENGTH, IdentifierInvalid
from app.providers.base import ChatResponse

from .conftest import OTHER_TENANT, TENANT, cached_answer

ALIAS = "agent-default"
OTHER_ALIAS = "agent-strong"
SYSTEM_HASH = "0" * 64
OTHER_SYSTEM_HASH = "1" * 64
TOOLS_HASH = "2" * 64
OTHER_TOOLS_HASH = "3" * 64

# --------------------------------------------------------------------------
# vectors whose cosine against PROBE is an exact float
# --------------------------------------------------------------------------

PROBE: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0)
"""The query vector. Length exactly 1, and its only non-zero component is the
first, so the dot product with any neighbour below is that neighbour's first
component and nothing is added to it."""

AT_THRESHOLD: tuple[float, ...] = (97.0, 21.0, 10.0, 5.0, 5.0)
"""Cosine exactly 0.97 against `PROBE`: 97² + 21² + 10² + 5² + 5² = 10000, so the
length is exactly 100 and the cosine is 97/100 - the same double the literal 0.97
parses to, in float64, whichever order the division is written in."""

BELOW_THRESHOLD: tuple[float, ...] = (96.0, 28.0, 0.0, 0.0, 0.0)
"""Cosine exactly 0.96: 96² + 28² = 10000."""

ABOVE_THRESHOLD: tuple[float, ...] = (98.0, 14.0, 14.0, 2.0, 0.0)
"""Cosine exactly 0.98: 98² + 14² + 14² + 2² = 10000."""

COSINE_ROWS: list[tuple[str, tuple[float, ...], float]] = [
    ("the neighbour at exactly the threshold", AT_THRESHOLD, 0.97),
    ("the neighbour one hundredth under it", BELOW_THRESHOLD, 0.96),
    ("the neighbour one hundredth over it", ABOVE_THRESHOLD, 0.98),
    ("the probe itself", PROBE, 1.0),
]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, computed here so the fixtures can be checked independently."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm)


@pytest.mark.parametrize(
    ("case", "vector", "expected"), COSINE_ROWS, ids=[row[0] for row in COSINE_ROWS]
)
def test_the_fixture_vectors_have_exactly_the_cosines_the_table_claims(
    case: str, vector: tuple[float, ...], expected: float
) -> None:
    """Catches a boundary pair whose cosine is not the number this file names it by - the
    ordinary way of writing one, `(0.97, sqrt(1 - 0.97**2))`, lands at 0.9699999999999998
    in float64 and would make the inclusive boundary row below red against a correct `>=`.
    Every row here is an integer vector of length exactly 100 against a query of length
    exactly 1, so the arithmetic is exact and the threshold rows fail only when the
    threshold is wrong."""
    assert cosine(PROBE, vector) == expected


# --------------------------------------------------------------------------
# the in-process implementation: round trip and threshold
# --------------------------------------------------------------------------


def stored(
    *,
    tenant: str = TENANT,
    vector: Sequence[float] = PROBE,
    answer: ChatResponse | None = None,
    alias: str = ALIAS,
    system_hash: str = SYSTEM_HASH,
    tools_hash: str = TOOLS_HASH,
) -> InProcessSemanticCache:
    """A cache holding one entry, defaulting to the probe vector and the base filters."""
    cache = InProcessSemanticCache()
    cache.put(
        tenant,
        vector,
        cached_answer() if answer is None else answer,
        alias=alias,
        system_hash=system_hash,
        tools_hash=tools_hash,
    )
    return cache


def search(
    cache: SemanticCache,
    *,
    tenant: str = TENANT,
    vector: Sequence[float] = PROBE,
    alias: str = ALIAS,
    system_hash: str = SYSTEM_HASH,
    tools_hash: str = TOOLS_HASH,
) -> ChatResponse | None:
    """One search with the base arrangement, so a row changes exactly what it names.

    Annotated against the Protocol rather than the class, so a signature that
    drifts from `SemanticCache` is a `mypy --strict` error in every case below.
    """
    return cache.search(tenant, vector, alias=alias, system_hash=system_hash, tools_hash=tools_hash)


def test_a_stored_answer_round_trips_on_an_identical_vector() -> None:
    """Catches a store that keeps the vector and drops the answer, or keeps only the
    content: the value is the NEUTRAL `ChatResponse` the exact tier stores, so the tool
    calls, the finish reason and the usage are what the output path re-renders from, and a
    hit that lost the tool calls answers text where the caller expected a call."""
    answer = cached_answer()

    assert search(stored(answer=answer)) == answer


def test_a_search_against_an_empty_cache_is_absent() -> None:
    """Catches a search that raises, or that returns a default answer, when nothing has been
    written: a miss is the ordinary outcome for the first request of every process, and
    every threshold row below would be unreadable if an empty cache did anything else."""
    assert search(InProcessSemanticCache()) is None


def test_a_neighbour_at_exactly_the_threshold_is_a_hit() -> None:
    """Catches the comparison written `>` where the plan says `>=` (decision E): this is the
    only row in the suite that separates the published 0.97 from a shipped 0.971, and the
    vector is built so its cosine is exactly the double `SEMANTIC_THRESHOLD` holds rather
    than a value that happens to round above it."""
    assert search(stored(vector=AT_THRESHOLD)) == cached_answer()


def test_a_neighbour_one_hundredth_under_the_threshold_is_a_miss() -> None:
    """Catches a search with no threshold at all - Qdrant's nearest neighbour is returned
    whatever its score, and an in-process one that returns the closest entry it has would
    serve a different question's answer to every request that misses. The other half of the
    boundary pair: without it the inclusive row above passes for a cache that never
    compares anything."""
    assert search(stored(vector=BELOW_THRESHOLD)) is None


def test_a_neighbour_above_the_threshold_is_a_hit() -> None:
    """Catches a threshold applied in the wrong direction (`score <= SEMANTIC_THRESHOLD`, or
    a distance compared as if it were a similarity): the 0.96 row alone passes for a cache
    that refuses everything, and the exact boundary row would pass for one that accepts
    only the boundary."""
    assert search(stored(vector=ABOVE_THRESHOLD)) == cached_answer()


@pytest.mark.parametrize("nearest_written_first", [True, False])
def test_the_nearest_of_two_eligible_neighbours_is_the_one_returned(
    nearest_written_first: bool,
) -> None:
    """Catches a search that returns the FIRST entry over the threshold rather than the
    nearest one: both entries below are eligible, so a threshold check alone cannot tell
    them apart, and the two insertion orders are what stop "returns the last written" and
    "returns the first written" from passing. The plan's own words are "the nearest
    neighbour among the entries that are exactly comparable"."""
    near = cached_answer(content="the answer for the nearer probe")
    far = cached_answer(content="the answer for the further probe")
    cache = InProcessSemanticCache()
    order = [(ABOVE_THRESHOLD, near), (AT_THRESHOLD, far)]
    if not nearest_written_first:
        order.reverse()
    for vector, answer in order:
        cache.put(
            TENANT, vector, answer, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
        )

    assert search(cache) == near


# --------------------------------------------------------------------------
# the three filter fields, one miss each at cosine 1.0
# --------------------------------------------------------------------------

FILTER_ROWS: list[tuple[str, str, str, str]] = [
    ("a different alias", OTHER_ALIAS, SYSTEM_HASH, TOOLS_HASH),
    ("a different system hash", ALIAS, OTHER_SYSTEM_HASH, TOOLS_HASH),
    ("a different tools hash", ALIAS, SYSTEM_HASH, OTHER_TOOLS_HASH),
]


@pytest.mark.parametrize(
    ("case", "alias", "system_hash", "tools_hash"),
    FILTER_ROWS,
    ids=[row[0] for row in FILTER_ROWS],
)
def test_a_query_differing_in_one_filter_field_misses_at_cosine_one(
    case: str, alias: str, system_hash: str, tools_hash: str
) -> None:
    """Catches a query built over two of the three payload fields, one row per field: the
    vector is IDENTICAL to the stored one, so similarity cannot be what refuses it and the
    row for the dropped field is the only thing that reddens. A dropped filter serves
    another conversation's answer - a different system prompt's, a different alias's, a
    different tool set's - and the nearest neighbour will happily be one (plan §5)."""
    assert search(stored(), alias=alias, system_hash=system_hash, tools_hash=tools_hash) is None


def test_a_query_matching_all_three_filters_hits_at_cosine_one() -> None:
    """Catches filters so strict that nothing ever matches - a query that ANDs a field
    against itself twice, or matches on a field the write never sets: every row above
    asserts a miss, so a search returning `None` unconditionally passes all three, and this
    is the row that separates the two readings."""
    assert search(stored()) == cached_answer()


# --------------------------------------------------------------------------
# tenant scoping
# --------------------------------------------------------------------------


def test_one_tenant_cannot_read_another_tenants_live_entry() -> None:
    """Catches a store keyed on the vector and the filters alone: the vector handed over is
    a REAL, live entry for the other tenant at cosine 1.0, so "nothing stored" cannot be
    what refuses it, and the two closing reads prove the refusals are not a cache that lost
    both entries. This is the failure the tier must not have - one tenant's question served
    another tenant's answer."""
    mine = cached_answer(content="the answer for evals")
    theirs = cached_answer(content="the answer for cuan")
    cache = InProcessSemanticCache()
    cache.put(TENANT, PROBE, mine, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH)
    cache.put(
        OTHER_TENANT, PROBE, theirs, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
    )

    assert search(cache, tenant=TENANT) == mine
    assert search(cache, tenant=OTHER_TENANT) == theirs


def test_a_tenant_with_nothing_stored_cannot_read_a_live_entry() -> None:
    """Catches the same failure in the arrangement the case above cannot reach: with only
    ONE tenant holding an entry, a cache that ignores the tenant returns that entry to
    everybody, while the two-entry case above would still pass because each tenant happened
    to have its own answer at the same distance."""
    cache = stored(tenant=OTHER_TENANT, answer=cached_answer(content="the answer for cuan"))

    assert search(cache, tenant=TENANT) is None


# --------------------------------------------------------------------------
# the collection name and the tenant pre-check (no service, no socket)
# --------------------------------------------------------------------------


class RecordingTransport(httpx.BaseTransport):
    """An `httpx` transport that records every request and answers a stub search.

    It is what makes "the pre-check ran before any effect" observable: an empty
    `requests` list is the only evidence that no collection name was built and no
    call was sent, which is what a project rule actually claims.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"result": []}, request=request)


def qdrant_cache() -> tuple[QdrantSemanticCache, RecordingTransport]:
    """A Qdrant cache over a recording transport, and the transport, so calls can be read."""
    transport = RecordingTransport()
    client = httpx.Client(transport=transport, base_url="http://qdrant.invalid:6333")
    return QdrantSemanticCache(client), transport


def test_the_collection_prefix_is_the_published_one() -> None:
    """Catches a prefix renamed after a cluster holds collections under the old one: a
    running gateway's entries live in collections addressed by this string, so a rename
    orphans every one of them and turns a warm tier into a day of misses nobody sees as a
    failure."""
    assert SEMANTIC_COLLECTION_PREFIX == "cache_"


def test_a_tenants_collection_is_the_prefix_and_the_tenant() -> None:
    """Catches a collection name built in another shape - one collection for everybody with
    the tenant as a payload filter, or the segments swapped: the plan says `cache_{tenant}`,
    an operator's tooling and every later migration depend on it, and a shared collection
    would make tenant scoping a filter that a dropped `must` clause silently removes."""
    cache, _ = qdrant_cache()

    assert cache.collection(TENANT) == f"cache_{TENANT}"
    assert cache.collection(OTHER_TENANT) == f"cache_{OTHER_TENANT}"


def test_the_collection_prefix_is_overridable_for_a_namespaced_run() -> None:
    """Catches a prefix hardcoded into the name rather than read from the constructor: the
    Qdrant suite's whole isolation is a per-test prefix, so without this
    seam those cases would have to drop collections they do not own."""
    transport = RecordingTransport()
    client = httpx.Client(transport=transport, base_url="http://qdrant.invalid:6333")
    cache = QdrantSemanticCache(client, collection_prefix="run7_cache_")

    assert cache.collection(TENANT) == f"run7_cache_{TENANT}"


MAX_TENANT = "a" * MAX_IDENTIFIER_LENGTH
OVER_CAP_TENANT = "a" * (MAX_IDENTIFIER_LENGTH + 1)

REJECTED_TENANTS: list[tuple[str, str]] = [
    ("empty", ""),
    ("one character over the cap", OVER_CAP_TENANT),
    ("uppercase", "Evals"),
    ("a space", "ev als"),
    ("a colon", "ev:als"),
    ("a path separator, which would forge a URL segment", "ev/als"),
    ("a parent segment, which would address another collection", "../evals"),
    ("a query separator", "evals?wait=true"),
    ("a trailing newline", "evals\n"),
    ("a trailing CRLF", "evals\r\n"),
    ("a trailing space", "evals "),
    ("a leading separator", "-evals"),
]


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_search_refuses_a_malformed_tenant_before_touching_qdrant(case: str, tenant: str) -> None:
    r"""Catches a collection name built from an unchecked tenant: the path rows would address
    another collection or another endpoint entirely, and the three trailing-whitespace rows
    are what separate `fullmatch` from `match` (a project rule, where `$` also matches
    before a final newline). Asserting the transport saw NO request is what pins "sends
    nothing" rather than "sends something and fails afterwards". Qdrant is an unsized store,
    so this check has no backstop behind it."""
    cache, transport = qdrant_cache()

    with pytest.raises(IdentifierInvalid):
        cache.search(tenant, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH)

    assert transport.requests == []


@pytest.mark.parametrize(("case", "tenant"), REJECTED_TENANTS, ids=[c[0] for c in REJECTED_TENANTS])
def test_put_refuses_a_malformed_tenant_before_touching_qdrant(case: str, tenant: str) -> None:
    """Catches a pre-check applied to the read path alone: the write path is where a
    malformed tenant actually CREATES a collection, so a `put` that skips the check is the
    half that does the damage the read half only reads back."""
    cache, transport = qdrant_cache()

    with pytest.raises(IdentifierInvalid):
        cache.put(
            tenant,
            PROBE,
            cached_answer(),
            alias=ALIAS,
            system_hash=SYSTEM_HASH,
            tools_hash=TOOLS_HASH,
        )

    assert transport.requests == []


def test_a_tenant_at_the_length_cap_is_accepted() -> None:
    """Catches an exclusive comparison (`>=` where `>` belongs) on the length cap: exactly
    `MAX_IDENTIFIER_LENGTH` characters is a legal tenant name everywhere else in the
    gateway, so refusing it here would make a tenant that authenticates unable to reach the
    tier. The cap is imported rather than restated, so the two cannot drift, and a request
    reaching the transport is what separates "accepted" from "refused quietly"."""
    cache, transport = qdrant_cache()

    result = cache.search(
        MAX_TENANT, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
    )

    assert result is None
    assert transport.requests != []


def test_a_tenant_one_character_over_the_cap_is_rejected() -> None:
    """The other half of the boundary pair: catches a cap that was never applied at all,
    which the accepted row alone cannot see."""
    cache, transport = qdrant_cache()

    with pytest.raises(IdentifierInvalid):
        cache.search(
            OVER_CAP_TENANT, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH
        )

    assert transport.requests == []


def test_both_implementations_are_held_by_the_protocol() -> None:
    """Catches a signature that drifts between the two implementations - a positional
    `alias`, a `put` taking the answer first - which would make the pair impossible to hold
    behind the one name `create_app` resolves at startup.

    Untestable at runtime by design: `SemanticCache` is a Protocol with no
    `runtime_checkable`, so there is no `isinstance` to assert and the kill is
    `mypy --strict` refusing the annotated tuple below. The call keeps the
    annotation on a line the test suite executes, so a class that stops existing
    is still a red rather than a silent skip."""
    transport = RecordingTransport()
    client = httpx.Client(transport=transport, base_url="http://qdrant.invalid:6333")
    holders: tuple[SemanticCache, SemanticCache] = (
        InProcessSemanticCache(),
        QdrantSemanticCache(client),
    )

    assert len(holders) == 2


# --------------------------------------------------------------------------
# controls the implementation introduced, pinned here
# --------------------------------------------------------------------------


def test_vectors_of_different_lengths_are_not_comparable() -> None:
    """Catches a cosine that zips two vectors of different lengths - `strict=True` raises a
    `ValueError` out of a lookup, and a non-strict zip silently scores a five-dimensional
    query against a three-dimensional entry on its first three components and can hand back
    a hit. Different lengths mean a different embedding model produced them, which Qdrant
    refuses at the collection boundary and this implementation, having no schema, has to
    refuse itself. Absent, never an exception: nothing in this package raises."""
    cache = stored(vector=PROBE)

    assert search(cache, vector=(1.0, 0.0, 0.0)) is None


def test_a_zero_vector_is_never_a_hit() -> None:
    """Catches a cosine whose divisor is an unguarded norm: a zero vector has no angle, so
    the quotient is undefined and the division raises `ZeroDivisionError` out of a cache
    lookup, 500ing a request that only wanted a probe. Both directions are asserted - a zero
    query against a live entry and a live query against a zero entry - because a guard on
    one norm alone passes whichever half it covers."""
    assert search(stored(vector=PROBE), vector=(0.0, 0.0, 0.0, 0.0, 0.0)) is None
    assert search(stored(vector=(0.0, 0.0, 0.0, 0.0, 0.0)), vector=PROBE) is None


def test_a_search_asks_for_exactly_one_neighbour() -> None:
    """Catches a search sent with no `limit` at all, or with a limit large enough that
    Qdrant's default page size decides it: this class returns the top hit of the answer, so
    a query asking for many neighbours pays for a page of vectors and payloads on every
    request and throws all but one away - and one asking for none would make every request a
    miss. The threshold travels with it, so a server-side comparison is what filters the
    index and the Python-side one is what pins the published number."""
    cache, transport = qdrant_cache()

    cache.search(TENANT, PROBE, alias=ALIAS, system_hash=SYSTEM_HASH, tools_hash=TOOLS_HASH)

    assert len(transport.requests) == 1
    sent = json.loads(transport.requests[0].content)
    assert sent["limit"] == 1
    assert sent["score_threshold"] == SEMANTIC_THRESHOLD
