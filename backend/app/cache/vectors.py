"""Where a semantic answer waits, and what counts as near enough to serve it.

The shape mirrors `cache/store.py` on purpose - a Protocol, an in-process
implementation and a real one, picked in `create_app` by `OPSCONSOLE_QDRANT_URL` - so
someone who has read the exact tier can read this one. Three things about it
follow from what a VECTOR cache is, and each is what its tests are arranged
around.

**A neighbour is comparable before it is near.** Every query carries three
payload filters beside the vector - the alias, the system hash and the tools
hash - so a search is "the nearest neighbour among the entries that are exactly
comparable" and never "the nearest neighbour, hopefully comparable". A query
short one filter serves another conversation's answer, and the nearest neighbour
will happily be one, which is why each field has its own miss case at cosine 1.0
(`test_a_query_differing_in_one_filter_field_misses_at_cosine_one`, in
tests/cache/test_vectors.py and again against a real Qdrant in
tests/cache/test_qdrant_cache.py) rather than a sample.

**Near enough is `>= SEMANTIC_THRESHOLD`, inclusive**
(`test_a_neighbour_at_exactly_the_threshold_is_a_hit`,
`test_a_neighbour_one_hundredth_under_the_threshold_is_a_miss`). The comparison
is made here in Python as well as sent to Qdrant as a `score_threshold`, so the
published number is the shipped one whichever way the server rounds its own
comparison.

**A tenant is a COLLECTION, not a filter.** `cache_{tenant}`, created on the
first write with the size of the vector being written (plan decision F: swapping
the model behind the `embed` alias is then a `routing.yaml` change and not a code
change). One tenant reading another's answer is the failure this tier must not
have, and a separate collection is the one arrangement a dropped `must` clause
cannot undo (`test_one_tenant_cannot_read_another_tenants_live_entry`).

Nothing here raises for a store reason (plan decision G), exactly as
`cache/store.py` does not: an unreachable Qdrant, a non-2xx answer and a stored
payload this build cannot read are each a MISS, and a failed write is swallowed
after the caller has already been served
(`test_a_search_against_an_unreachable_qdrant_is_a_miss`,
`test_a_non_2xx_answer_to_a_search_is_a_miss`,
`test_a_stored_payload_this_build_cannot_read_is_a_miss`, and their `put` twins).
A malformed tenant is the one exception and is not a store failure: it raises
`IdentifierInvalid` before any request is sent, because Qdrant
is an unsized store and there is no backstop behind that check.

Nothing here embeds anything. The vector arrives as an argument, because an
embedding is a provider call with a residency regime and a failure mode of its
own, and those belong to the route, where `embedded_probe` in
`services/chat_pipeline.py` makes the call and swallows its every failure into a
miss.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Sequence
from typing import Final, Protocol

import httpx
from pydantic import ValidationError

from app.cache.probe import SEMANTIC_THRESHOLD
from app.core.validation import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH
from app.model_routing.errors import IdentifierInvalid
from app.providers.base import ChatResponse

SEMANTIC_COLLECTION_PREFIX: Final[str] = "cache_"
"""The first segment of every collection a tenant's semantic entries live in.

A running gateway's entries are addressed by this string, so renaming it orphans
every collection already created and turns a warm tier into a day of misses
nobody sees as a failure; pinned by
`test_the_collection_prefix_is_the_published_one`."""

_QDRANT_DISTANCE: Final[str] = "Cosine"
"""The distance every collection this class creates is configured with.

Qdrant's own spelling, and not a choice: `SEMANTIC_THRESHOLD` is a COSINE
similarity, so a collection created with `Dot` or `Euclid` would return scores on
a different scale and the threshold would silently mean something else while
every round trip still passed. Pinned against a real server by
`test_the_collection_is_created_on_the_first_write_with_the_written_vector_size`."""

_SEARCH_LIMIT: Final[int] = 1
"""How many neighbours a search asks for.

One, because a search wants the nearest comparable entry and nothing else does
anything with the rest; asking for more would only move the choice of "nearest"
from the server into this module. Pinned by
`test_a_search_asks_for_exactly_one_neighbour`."""

_ANSWER_FIELD: Final[str] = "answer"
"""The payload key a stored `ChatResponse` travels under, as JSON.

Named rather than repeated, because the write and the read have to agree on it or
every entry is written and then found unreadable - a tier that stores everything
and serves nothing, with no failure anywhere to see. The read side is pinned by
`test_a_stored_payload_this_build_cannot_read_is_a_miss`'s "a payload written by
a build that had no answer field" row and the pair is closed by
`test_a_stored_answer_round_trips_through_qdrant`."""


class SemanticCache(Protocol):
    """Put an answer away under a tenant and a probe vector; find it again from a near one.

    Two methods and one rule, and the rule is `ResponseCache`'s: NEITHER method
    raises for any reason a store can have. `search` answers `None` for a tenant
    with nothing stored, for a neighbour under the threshold, for an entry whose
    filters do not match, for a payload it cannot read and for a store that is
    unreachable; `put` returns whether or not the write landed. An implementation
    that raised would turn a Qdrant incident into 5xx on requests a provider
    would happily have served.

    The tenant still has to be well formed - a malformed one raises
    `IdentifierInvalid` BEFORE any effect. That is not a store
    failure: it is a programming error at the call site, and the route never
    reaches it with anything but the tenant `gateway/context.py` authenticated.

    `alias`, `system_hash` and `tools_hash` are keyword-only on both methods, for
    the reason the exact tier's `allow_header` is: three same-typed strings in a
    row are two mistakes waiting, and a search that swapped two of them would
    filter on values no entry carries and simply never hit.
    """

    def search(
        self,
        tenant: str,
        vector: Sequence[float],
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> ChatResponse | None:
        """The nearest comparable answer at or above the threshold, or `None`."""
        ...

    def put(
        self,
        tenant: str,
        vector: Sequence[float],
        answer: ChatResponse,
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> None:
        """File one answer under this tenant, this probe vector and these filters."""
        ...


class InProcessSemanticCache:
    """A `SemanticCache` holding every vector in one process's memory.

    What a bare local run and the suite get when no Qdrant URL is configured
    (plan decision D, the same arrangement 13a decision C chose for the exact
    tier), so neither needs a backing service to exercise the cosine rule.

    Two replicas do not share it, and nothing expires: for a cache that costs a
    hit rate rather than correctness, and a process that has been up long enough
    for the memory to matter is a process the semantic tier was not sized for.
    Which implementation a deployment gets is decided by `create_app`
    (`test_create_app_keeps_the_in_process_semantic_cache_without_a_qdrant_url`).

    The arithmetic is deliberately plain Python and not a library: the whole
    contents of one tenant's entries are scanned per search, which is the wrong
    algorithm at any real scale and exactly right for a fallback whose purpose is
    to let the tests and a laptop run the same code path as the cluster.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[tuple[tuple[float, ...], str, str, str, ChatResponse]]] = {}

    def search(
        self,
        tenant: str,
        vector: Sequence[float],
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> ChatResponse | None:
        """The nearest comparable entry at or above `SEMANTIC_THRESHOLD`, or `None`.

        Comparable means all three filters match exactly; near means
        `>= SEMANTIC_THRESHOLD`, inclusive. The best score is tracked rather than
        the first one over the line, because "the first entry over the threshold"
        depends on insertion order and would serve a worse answer than the one
        held - the pair of orders in
        `test_the_nearest_of_two_eligible_neighbours_is_the_one_returned`.
        """
        best: tuple[float, ChatResponse] | None = None
        for stored_vector, stored_alias, stored_system, stored_tools, answer in self._entries.get(
            tenant, []
        ):
            if (stored_alias, stored_system, stored_tools) != (alias, system_hash, tools_hash):
                continue
            score = _cosine(vector, stored_vector)
            if score < SEMANTIC_THRESHOLD:
                continue
            if best is None or score > best[0]:
                best = (score, answer)
        return None if best is None else best[1]

    def put(
        self,
        tenant: str,
        vector: Sequence[float],
        answer: ChatResponse,
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> None:
        """Append one entry to this tenant's list.

        Appended rather than keyed: two probes are the same question by DISTANCE
        and not by equality, so there is no key under which one could replace the
        other, and a near-duplicate simply becomes a second neighbour that the
        search will rank. A `ChatResponse` is frozen, so it is filed as it stands.
        """
        self._entries.setdefault(tenant, []).append(
            (
                tuple(float(component) for component in vector),
                alias,
                system_hash,
                tools_hash,
                answer,
            )
        )


class QdrantSemanticCache:
    """A `SemanticCache` keeping vectors in Qdrant, where every replica reads the same ones.

    What the gateway ships with once a Qdrant URL is configured. The client
    carries the base URL, so this class sends relative paths and holds no
    endpoint of its own; it is the same `httpx` the readiness probe an earlier change
    shipped already talks to Qdrant with, rather than a new dependency (plan
    decision A, a project rule).

    One tenant is one collection, created LAZILY on the first write with the size
    of the vector being written and cosine distance (plan decision F). A search
    creates nothing: every tenant's first semantic request arrives before there
    is a collection to query, so the 404 that comes back is the hot path and not
    an edge case (`test_a_search_before_any_write_creates_no_collection`,
    `test_a_search_before_any_write_is_a_miss`).

    `collection_prefix` exists for the suite: per-test isolation is a namespace
    rather than a sweep that would delete collections this suite does not own. The default is the shipped one, exercised by
    `test_a_tenants_collection_is_the_prefix_and_the_tenant`.
    """

    def __init__(
        self, client: httpx.Client, *, collection_prefix: str = SEMANTIC_COLLECTION_PREFIX
    ) -> None:
        self._client = client
        self._collection_prefix = collection_prefix

    def collection(self, tenant: str) -> str:
        """The collection one tenant's entries live in.

        Public because an operator's tooling, the lazy-creation test and every
        later migration all address entries by this name
        (`test_a_tenants_collection_is_the_prefix_and_the_tenant`).
        """
        return f"{self._collection_prefix}{tenant}"

    def search(
        self,
        tenant: str,
        vector: Sequence[float],
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> ChatResponse | None:
        """The nearest comparable answer at or above the threshold, or `None`.

        `None` covers a tenant with no collection yet, a collection with no
        comparable entry, a neighbour under the threshold, a payload this build
        cannot read and a Qdrant that is not answering at all. A caller cannot
        tell those apart, and for a cache there is nothing it could do
        differently if it could.

        The threshold is sent to the server AND applied here: Qdrant's
        `score_threshold` does the filtering in the index, and the comparison
        below is what makes the published `SEMANTIC_THRESHOLD` the shipped one
        whichever way the server rounds its own.
        """
        _checked_tenant(tenant)
        found = self._post(
            f"/collections/{self.collection(tenant)}/points/search",
            {
                "vector": [float(component) for component in vector],
                "limit": _SEARCH_LIMIT,
                "with_payload": True,
                "score_threshold": SEMANTIC_THRESHOLD,
                "filter": {
                    "must": [
                        {"key": "alias", "match": {"value": alias}},
                        {"key": "system_hash", "match": {"value": system_hash}},
                        {"key": "tools_hash", "match": {"value": tools_hash}},
                    ]
                },
            },
        )
        if found is None:
            return None
        hits = found.get("result")
        if not isinstance(hits, list) or not hits:
            return None
        return _answer_of(hits[0])

    def put(
        self,
        tenant: str,
        vector: Sequence[float],
        answer: ChatResponse,
        *,
        alias: str,
        system_hash: str,
        tools_hash: str,
    ) -> None:
        """File one answer in this tenant's collection, creating it if this is the first.

        Swallows every store failure the search does, and for a sharper reason:
        the write happens after a provider has already answered and been paid, so
        an escaping error would fail a request that was fully served.

        The point id is a fresh UUID rather than a digest of the probe: two
        probes are the same question by distance, so there is no id under which
        one could replace another, and an id derived from the vector would make
        near-duplicates collide at exact equality only - the worst of both.
        """
        _checked_tenant(tenant)
        components = [float(component) for component in vector]
        self._ensure_collection(tenant, len(components))
        self._put(
            f"/collections/{self.collection(tenant)}/points",
            {
                "points": [
                    {
                        "id": str(uuid.uuid4()),
                        "vector": components,
                        "payload": {
                            "alias": alias,
                            "system_hash": system_hash,
                            "tools_hash": tools_hash,
                            _ANSWER_FIELD: answer.model_dump_json(),
                        },
                    }
                ]
            },
        )

    def _ensure_collection(self, tenant: str, size: int) -> None:
        """Create this tenant's collection if Qdrant does not already hold it.

        The existence check is what keeps a second write from recreating the
        collection and dropping everything in it
        (`test_a_second_write_to_the_same_tenant_creates_nothing`). A create that
        loses a race with another replica's create answers a non-2xx, which is
        swallowed like every other store failure: the upsert that follows lands
        in the collection the other replica made.
        """
        name = self.collection(tenant)
        try:
            described = self._client.get(f"/collections/{name}")
        except httpx.HTTPError:
            return
        if described.is_success:
            return
        self._put(
            f"/collections/{name}",
            {"vectors": {"size": size, "distance": _QDRANT_DISTANCE}},
        )

    def _post(self, path: str, body: object) -> dict[str, object] | None:
        """One POST, or `None` for anything that is not a readable 2xx JSON body."""
        try:
            answered = self._client.post(path, json=body)
        except httpx.HTTPError:
            return None
        return _readable_body(answered)

    def _put(self, path: str, body: object) -> None:
        """One PUT whose outcome nobody reads: both write paths swallow alike."""
        try:
            self._client.put(path, json=body)
        except httpx.HTTPError:
            return


def _readable_body(answered: httpx.Response) -> dict[str, object] | None:
    """A 2xx answer's JSON object, or `None` for a non-2xx or an unreadable body.

    `is_success` rather than `raise_for_status()`: a 404 is what Qdrant returns
    for a tenant that has never had an answer cached, so the FIRST semantic
    request every tenant makes takes this path and an exception there would be an
    incident report per new tenant
    (`test_a_non_2xx_answer_to_a_search_is_a_miss`, whose 404 row is that one).
    """
    if not answered.is_success:
        return None
    try:
        decoded = answered.json()
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _answer_of(hit: object) -> ChatResponse | None:
    """The `ChatResponse` inside one Qdrant hit, or `None` when it cannot be read.

    Every shape an older or a foreign writer can leave behind is absent rather
    than an exception: no payload at all, a payload with no answer field, an
    answer that is not JSON, and one this build's model cannot validate
    (`test_a_stored_payload_this_build_cannot_read_is_a_miss`, four rows). An
    entry this build cannot use is a miss, not a 500.
    """
    if not isinstance(hit, dict):
        return None
    payload = hit.get("payload")
    if not isinstance(payload, dict):
        return None
    stored = payload.get(_ANSWER_FIELD)
    if not isinstance(stored, str):
        return None
    try:
        return ChatResponse.model_validate_json(stored)
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or `0.0` when it is not defined for this pair.

    Two pairs have no angle between them and are scored as far apart rather than
    raising: vectors of different lengths, which is what a query built by a
    different embedding model than the stored entry is
    (`test_vectors_of_different_lengths_are_not_comparable`), and a zero vector,
    whose norm makes the quotient undefined
    (`test_a_zero_vector_is_never_a_hit`). Qdrant refuses both at the collection
    boundary; this implementation has no schema, so it refuses them here.
    """
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _checked_tenant(tenant: str) -> None:
    """Refuse a tenant that may not shape a collection name, before any request is sent.

    The gateway's own identifier rules, imported rather than restated, so the
    name this cache accepts and the name `gateway/context.py` authenticated can
    never disagree. Inclusive at the cap: exactly `MAX_IDENTIFIER_LENGTH`
    characters is accepted (`test_a_tenant_at_the_length_cap_is_accepted`,
    `test_a_tenant_one_character_over_the_cap_is_rejected`). A collection name is
    a URL segment, so an unchecked tenant would address another collection or
    another endpoint entirely, and Qdrant has no backstop behind this check.
    """
    if len(tenant) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"tenant: longer than {MAX_IDENTIFIER_LENGTH} characters", param="tenant"
        )
    if IDENTIFIER_PATTERN.fullmatch(tenant) is None:
        raise IdentifierInvalid("tenant: not a valid identifier", param="tenant")
