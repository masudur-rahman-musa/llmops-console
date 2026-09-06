"""Where a placeholder map lives between the call that made it and the one that reads it.

Redaction is only half a control. `redact` turns values into `<KIND_N>`
placeholders and hands back the map that inverts them; something has to hold that
map for the hour in which a caller may want its text back (#20, an earlier change). This
module is that seam and nothing else - it does not redact, and it never looks
inside a mapping it is given.

Two things are deliberate here rather than incidental.

The key is `(tenant, map_id)`, not the id alone. A map holds the raw personal
data a request carried, so a tenant reading another tenant's map is the one
failure this store must not have; pinned by
`test_one_tenant_cannot_read_another_tenants_map`, which reads with an id that is
genuinely valid for the other tenant so "unknown id" cannot be what makes it
pass.

The clock is injected. The TTL boundary is a criterion with an exact number on
it, and a test that pinned it by sleeping for an hour is not a test - `FakeClock`
in `tests/redaction/test_store.py` moves time by hand, and the pair at
`REDACTION_MAP_TTL_SECONDS` and one second past it is what pins `>` where `>=`
would be wrong.

`RedactionMapStore` is a Protocol because this implementation is temporary and
known to be. It keeps maps in the process, so two gateway replicas do not share
them. Nothing reads a map before an earlier change, and an earlier change
brings Redis, at which point a second implementation satisfies this protocol
rather than replacing a class the routes import by name.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping
from typing import Final, Protocol

REDACTION_MAP_TTL_SECONDS: Final[int] = 3600
"""How long a placeholder map stays readable, in seconds (the design's one hour).

Pinned exactly by `test_the_ttl_is_exactly_one_hour` and bracketed by
`test_a_map_is_readable_up_to_and_including_the_ttl` and
`test_a_map_is_gone_past_the_ttl`. The boundary is inclusive: a map is still
there at exactly this many seconds and gone one second later."""

REDACTION_MAP_ID_BYTES: Final[int] = 16
"""How much randomness a map id carries, before hex rendering doubles its length.

A map id is the only thing standing between a caller and the raw values behind
someone else's placeholders, so it is drawn from `secrets` and never from a
counter or a hash of the mapping. Pinned exactly by
`test_the_id_is_exactly_sixteen_random_bytes`, with the rendered length pinned
against both this constant and the literal 32 by
`test_an_id_is_thirty_two_lowercase_hex_characters`."""


class RedactionMapStore(Protocol):
    """Put a placeholder map away under a tenant; read it back by id until it expires.

    The contract is two methods and one rule: `get` answers `None` for an id this
    tenant did not write or whose map has expired, and raises for neither
    (`test_an_unknown_id_answers_absent_rather_than_raising`). A store that raised
    would make "expired" and "wrong tenant" distinguishable from the outside,
    which is exactly the difference a caller must not be able to measure.
    """

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        """Store one map for `tenant` and answer the fresh id it was stored under."""
        ...

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        """The map `tenant` stored under `map_id`, or `None` if there is not one."""
        ...


class InProcessRedactionMapStore:
    """A `RedactionMapStore` holding everything in one process's memory.

    Expiry is lazy: a map is dropped when it is read past its TTL rather than by
    a sweeper, so there is no background task and no lock. That leaves expired
    entries occupying memory until someone asks for them, which is the trade a
    single G0 node can afford and a reason this implementation is not the one
    that ships past an earlier change.

    `clock` defaults to `time.monotonic` rather than `time.time`: a wall clock
    stepped backwards by NTP would resurrect an expired map, and monotonic time
    cannot be stepped. The default exists so `create_app` can build one with no
    arguments (`test_the_clock_parameter_has_a_default`).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._maps: dict[tuple[str, str], tuple[float, Mapping[str, str]]] = {}

    def put(self, tenant: str, mapping: Mapping[str, str]) -> str:
        """Store `mapping` under a fresh id and answer it.

        The mapping is copied, so a caller that goes on to mutate what it handed
        over cannot change what a later `get` answers. The id is drawn fresh
        every call, so two identical mappings stored at the same instant get
        different ids (`test_two_maps_never_share_an_id`).
        """
        map_id = secrets.token_hex(REDACTION_MAP_ID_BYTES)
        self._maps[(tenant, map_id)] = (self._clock(), dict(mapping))
        return map_id

    def get(self, tenant: str, map_id: str) -> Mapping[str, str] | None:
        """The stored map, or `None` when this tenant has no live map under that id.

        `>` and not `>=`: the map is still readable at exactly
        `REDACTION_MAP_TTL_SECONDS` and gone the second after.
        """
        entry = self._maps.get((tenant, map_id))
        if entry is None:
            return None
        stored_at, mapping = entry
        if self._clock() - stored_at > REDACTION_MAP_TTL_SECONDS:
            del self._maps[(tenant, map_id)]
            return None
        return mapping
