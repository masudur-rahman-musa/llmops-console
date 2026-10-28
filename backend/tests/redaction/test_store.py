"""The redaction map store: its two constants, the TTL boundary, and tenant scoping.

Everything here is in-process and the clock is injected, so no test sleeps and no
backing service is started, so there is nothing to scope or isolate.

Three claims carry the criteria and each is arranged so that only the
control it names can decide the outcome:

* the TTL is a boundary PAIR against one clock (present at exactly
  `REDACTION_MAP_TTL_SECONDS`, gone one second later), so a `>=` written where a
  `>` belongs reddens on the first row and a missing expiry reddens on the second;
* the id is asserted against `REDACTION_MAP_ID_BYTES * 2` AND against the literal
  32, so an implementation that shrank the byte count in step with the rendering
  still fails;
* tenant scoping is asserted with a map id that IS valid - for the other tenant -
  so "unknown id" cannot be what makes the case pass.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest

from app.redaction import (
    REDACTION_MAP_ID_BYTES,
    REDACTION_MAP_TTL_SECONDS,
    InProcessRedactionMapStore,
    RedactionMapStore,
)

TENANT = "nordlicht"
OTHER_TENANT = "cuan"

# A map exactly as `redact` hands one over: placeholder -> the raw value.
MAPPING: Mapping[str, str] = {
    "<EMAIL_1>": "alice@example.com",
    "<EMAIL_2>": "bob@example.net",
    "<IPV4_1>": "192.0.2.1",
}
OTHER_MAPPING: Mapping[str, str] = {"<EMAIL_1>": "carol@example.org"}

HEX_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

START = 1_000.0
"""The clock's value at the moment of the `put`. Deliberately not 0, so a store
that compared against an absolute value rather than an elapsed one is visible."""


class FakeClock:
    """A monotonic clock a test moves by hand. `now` is seconds, like `time.monotonic`."""

    def __init__(self, now: float = START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_the_ttl_is_exactly_one_hour() -> None:
    """Catches a store whose TTL was written in minutes, or set to a plausible
    neighbour like 3000 or 86400: the value is the published one-hour rehydration
    window, and a band would admit every one of those."""
    assert REDACTION_MAP_TTL_SECONDS == 3600


def test_the_id_is_exactly_sixteen_random_bytes() -> None:
    """Catches an implementation that trimmed the entropy behind a map id (8 bytes,
    or a counter): a guessable id is another tenant's PII map, and only the exact
    value pins it."""
    assert REDACTION_MAP_ID_BYTES == 16


@pytest.mark.parametrize("elapsed", [0.0, 1.0, 3599.0, 3600.0])
def test_a_map_is_readable_up_to_and_including_the_ttl(elapsed: float) -> None:
    """Catches an expiry written `now - stored_at >= TTL` (or an off-by-one on the
    clock): the 3600.0 row is exactly the last moment the map is still
    there, and the earlier rows keep a store that expires everything immediately
    from passing the pair by accident."""
    clock = FakeClock()
    store = InProcessRedactionMapStore(clock=clock)
    map_id = store.put(TENANT, MAPPING)

    clock.now = START + elapsed

    assert store.get(TENANT, map_id) == MAPPING


@pytest.mark.parametrize("elapsed", [3601.0, 7200.0])
def test_a_map_is_gone_past_the_ttl(elapsed: float) -> None:
    """Catches a store that never expires anything - the map id is valid, the tenant
    is right, and the ONLY thing that can make this answer None is the TTL, so a
    dict with no clock reading fails here and nowhere else."""
    clock = FakeClock()
    store = InProcessRedactionMapStore(clock=clock)
    map_id = store.put(TENANT, MAPPING)

    clock.now = START + elapsed

    assert store.get(TENANT, map_id) is None


def test_an_unknown_id_answers_absent_rather_than_raising() -> None:
    """Catches a store implementing `get` as a bare `self._maps[key]`: rehydration
    asks about ids a caller made up, and a KeyError there is a 500 with
    no typed code instead of the "no such map" the endpoint owes."""
    store = InProcessRedactionMapStore(clock=FakeClock())

    assert store.get(TENANT, "0" * 32) is None


def test_an_id_is_thirty_two_lowercase_hex_characters() -> None:
    """Catches an id rendered as base64, as a UUID with dashes, or from fewer bytes:
    the id travels in a JSON body and becomes a store key, so the rendering is part
    of the contract and not an implementation detail."""
    store = InProcessRedactionMapStore(clock=FakeClock())

    map_id = store.put(TENANT, MAPPING)

    assert len(map_id) == REDACTION_MAP_ID_BYTES * 2
    # The literal anchor: an implementation that halved both the constant and the
    # rendering would satisfy the line above and fail this one.
    assert len(map_id) == 32
    assert HEX_ID_PATTERN.fullmatch(map_id) is not None


def test_two_maps_never_share_an_id() -> None:
    """Catches an id derived from the clock, from a counter reset per tenant, or from
    the mapping's contents: two puts of the SAME mapping at the SAME instant are the
    arrangement that separates real entropy from a hash of the inputs."""
    clock = FakeClock()
    store = InProcessRedactionMapStore(clock=clock)

    first = store.put(TENANT, MAPPING)
    second = store.put(TENANT, MAPPING)

    assert first != second
    assert store.get(TENANT, first) == MAPPING
    assert store.get(TENANT, second) == MAPPING


def test_one_tenant_cannot_read_another_tenants_map() -> None:
    """Catches a store keyed on the map id alone: the id handed to `get` is a REAL,
    unexpired id, so the tenant half of the key is the only thing that can refuse
    it, and a global dict would hand `cuan` `nordlicht`'s PII."""
    clock = FakeClock()
    store = InProcessRedactionMapStore(clock=clock)
    mine = store.put(TENANT, MAPPING)
    theirs = store.put(OTHER_TENANT, OTHER_MAPPING)

    assert store.get(OTHER_TENANT, mine) is None
    assert store.get(TENANT, theirs) is None
    # Both ids still work for their own tenant, so the two refusals above are not
    # a store that lost the maps.
    assert store.get(TENANT, mine) == MAPPING
    assert store.get(OTHER_TENANT, theirs) == OTHER_MAPPING


def test_the_stored_map_round_trips_every_entry() -> None:
    """Catches a store that keeps only the placeholders (or only the last entry):
    rehydration substitutes every pair back, so a map that came out short would
    leave a placeholder in text a caller believes was restored."""
    store = InProcessRedactionMapStore(clock=FakeClock())

    map_id = store.put(TENANT, MAPPING)
    read = store.get(TENANT, map_id)

    assert read is not None
    assert dict(read) == dict(MAPPING)


def test_the_in_process_store_satisfies_the_protocol() -> None:
    """Catches an implementation whose method names or signatures drift from
    `RedactionMapStore` - the annotation below is what `mypy --strict` checks, and
    the round trip through the protocol-typed name is what proves the two methods
    are callable in the shape the Redis store will have to match."""
    store: RedactionMapStore = InProcessRedactionMapStore()

    map_id = store.put(TENANT, MAPPING)

    assert store.get(TENANT, map_id) == MAPPING


def test_the_clock_parameter_has_a_default() -> None:
    """Catches a store whose `clock` parameter is required: every test above injects
    one, so a missing default would go unnoticed here and turn `app.py`'s
    `InProcessRedactionMapStore()` into a TypeError at process start."""
    store = InProcessRedactionMapStore()

    map_id = store.put(TENANT, MAPPING)

    assert store.get(TENANT, map_id) == MAPPING
