"""Where a tenant's display names live, so the tenth detector kind has an input.

The nine pattern kinds read nothing but the text. `PERSON` reads a list of names
an operator pushed, so it needs a store, and this module is that store and
nothing else - it does not match, and it never sees a request. The matcher that
compiles what `get` answers is `app.redaction.names`.

Shaped deliberately like `store.py`, because the two answer the same questions.
`DirectoryStore` is a Protocol with an in-process implementation for a bare local
run and the suite, and a Redis one `create_app` picks when
`OPSCONSOLE_REDIS_URL` is set. The differences from the map
store are the two the design asks for: a directory has no TTL, because it is
configuration rather than the residue of one request, and it is keyed
`(tenant,)` alone rather than `(tenant, id)`.

Three things are deliberate here rather than incidental.

The entry is `sha256(normalised name) -> plaintext name`, which is the design's
"hashed + plaintext" read literally. The hash is what makes
a re-push idempotent: an operator who types `  ANNA SCHMIDT ` after having
pushed `Anna Schmidt` writes the same field rather than a second entry that
burns one of the 2000 slots and compiles a duplicate branch into the alternation
(`test_re_pushing_a_name_the_directory_already_holds_changes_nothing`).

A push MERGES (decision C). A caller that pages a large directory would
otherwise silently delete everything it had already sent, and nothing in the
answer would say so (`test_a_second_push_merges_rather_than_replacing`). There is
no delete in this plan.

Every cap is checked over the WHOLE push before anything is written, and the
count cap over the resulting TOTAL rather than over one push.
Validating inside the write loop would leave a tenant's directory half applied
with the caller holding an error naming a cap
(`test_one_bad_name_in_a_push_stores_none_of_the_good_ones`,
`test_the_count_cap_is_over_the_directory_total_and_not_over_one_push`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, Protocol

import redis
from redis.typing import EncodableT, FieldT

from app.redaction.errors import DirectoryNameInvalid, DirectoryStoreUnavailable
from app.routing.errors import IdentifierInvalid
from app.routing.resolver import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH

MIN_DIRECTORY_NAME_CHARS: Final[int] = 2
"""The shortest name an operator may push, in normalised characters.

The cap that matters. A one-character name redacts every
`a` in every request that tenant sends, so a single typo in a push is a precision
failure across the tenant's whole traffic - and precision is a number this plan
has to keep above a floor. Pinned exactly by
`test_the_three_caps_are_the_published_numbers` and bracketed by
`test_a_name_at_a_length_boundary_is_accepted` and
`test_a_name_outside_the_length_bounds_is_refused_and_nothing_is_stored`."""

MAX_DIRECTORY_NAME_CHARS: Final[int] = 128
"""The longest name an operator may push, in normalised characters.

Long enough for a full name with titles and a double-barrelled surname; short
enough that 2000 of them cannot make the compiled alternation enormous. Pinned
and bracketed by the same three tests as the minimum."""

MAX_DIRECTORY_NAMES: Final[int] = 2000
"""How many names one tenant's directory may hold in total.

The compiled alternation is the only unbounded thing this plan introduces, and
this is what bounds it. Over the TOTAL and not over one push: a per-push reading
lets a paged caller build a directory of any size at all
(`test_the_count_cap_is_over_the_directory_total_and_not_over_one_push`).
Pinned by `test_the_three_caps_are_the_published_numbers` and bracketed by
`test_a_push_at_the_count_cap_is_accepted` and
`test_a_push_one_name_over_the_count_cap_is_refused`."""

DIRECTORY_KEY_PREFIX: Final[str] = "directory"
"""The first segment of every Redis key a directory is stored under.

A live gateway's directories are addressed by this string, so renaming it empties
every directory an operator has pushed; pinned by
`test_the_key_prefix_is_the_published_one`."""

NAMES_PARAM: Final[str] = "names"
"""The field every cap rejection names, and the only thing about the push besides
the cap that a rejection is allowed to say
(`test_a_rejection_names_the_field_and_the_cap_and_never_the_value`)."""


def normalise_name(name: str) -> str:
    """The form a name is hashed and length-checked in: case-folded and stripped.

    Both halves matter and neither is enough alone. Stripping is what makes a
    trailing newline from an operator's file the same name; case-folding is what
    makes `ANNA SCHMIDT` the same name as `anna schmidt`, which the matcher
    treats as one anyway because it matches case-insensitively. The length check
    runs on this rather than on the raw string, or `"  L  "` would be five
    characters and slip a one-character name past the minimum
    (`test_a_name_whose_length_is_only_whitespace_is_refused`,
    `test_normalisation_folds_case_and_strips_surrounding_whitespace`).
    """
    return name.strip().casefold()


def name_field(name: str) -> str:
    """The hash one name is stored under: sha256 of its normalised form, as hex.

    the design's hashed half. It is a stable id for an entry, which is what makes a
    re-push idempotent and what a later plan's delete would address
    (`test_the_field_is_the_sha256_of_the_normalised_name`).
    """
    return hashlib.sha256(normalise_name(name).encode("utf-8")).hexdigest()


def checked_entries(names: Sequence[str], held: Sequence[str]) -> dict[str, str]:
    """Validate a whole push and answer the entries it should write, or raise.

    Nothing is written by a caller of this function until it has returned, which
    is what a project rule asks of the three caps: a name outside the length
    bounds, or a push that would take the directory past
    `MAX_DIRECTORY_NAMES`, raises `DirectoryNameInvalid` with none of the push
    applied (`test_one_bad_name_in_a_push_stores_none_of_the_good_ones`).

    `held` is what the tenant's directory already contains, so the count is over
    the union of the two rather than over either alone: a re-push of names the
    directory already holds is not growth
    (`test_the_count_cap_counts_a_re_push_once`).

    The stored value is the name STRIPPED but not case-folded, so an operator
    reads back what they pushed; only the field is folded.
    """
    entries: dict[str, str] = {}
    for name in names:
        normalised = normalise_name(name)
        if len(normalised) < MIN_DIRECTORY_NAME_CHARS:
            raise DirectoryNameInvalid(
                f"{NAMES_PARAM}: a name shorter than {MIN_DIRECTORY_NAME_CHARS} "
                "characters cannot be stored",
                limit=MIN_DIRECTORY_NAME_CHARS,
            )
        if len(normalised) > MAX_DIRECTORY_NAME_CHARS:
            raise DirectoryNameInvalid(
                f"{NAMES_PARAM}: a name longer than {MAX_DIRECTORY_NAME_CHARS} "
                "characters cannot be stored",
                limit=MAX_DIRECTORY_NAME_CHARS,
            )
        entries[name_field(name)] = name.strip()

    total = len({name_field(name) for name in held} | entries.keys())
    if total > MAX_DIRECTORY_NAMES:
        raise DirectoryNameInvalid(
            f"{NAMES_PARAM}: a directory cannot hold more than {MAX_DIRECTORY_NAMES} names",
            limit=MAX_DIRECTORY_NAMES,
        )
    return entries


class DirectoryStore(Protocol):
    """Push a tenant's display names; read them all back.

    Two methods and one rule: `get` answers the empty tuple for a tenant nobody
    has pushed for, and raises for neither that nor an expired anything - there is
    no expiry (`test_an_empty_directory_answers_an_empty_tuple`). Every redacted
    request makes this read, so a `KeyError` or a `None` here is a 500 on the
    first request of every tenant with no names, which is most of them.

    What `get` may NOT do is answer an empty tuple because the store is
    unreachable. That is an earlier change decision H, and it is why `RedisDirectoryStore`
    raises `DirectoryStoreUnavailable`: a request redacted without the tenant's
    names and served anyway sends exactly the values this control exists to
    remove (`test_a_get_against_an_unreachable_redis_raises_the_typed_failure`).
    """

    def put(self, tenant: str, names: Sequence[str]) -> None:
        """Merge `names` into `tenant`'s directory, or raise before writing anything."""
        ...

    def get(self, tenant: str) -> tuple[str, ...]:
        """Every name `tenant` has pushed, in no particular order."""
        ...


class InProcessDirectoryStore:
    """A `DirectoryStore` holding every tenant's names in one process's memory.

    What a bare `uv run` and the whole suite use, so neither needs Redis. Two
    gateway replicas do not share a directory built this way, which is why it is
    not the implementation that ships once `OPSCONSOLE_REDIS_URL` is set.

    There is no expiry to be lazy about, unlike the map store: a directory is
    configuration and is meant to outlive every request that reads it.
    """

    def __init__(self) -> None:
        self._directories: dict[str, dict[str, str]] = {}

    def put(self, tenant: str, names: Sequence[str]) -> None:
        """Merge `names` into this tenant's directory, or raise having written nothing.

        The caps are checked over the whole push first, against what this tenant
        already holds, so a refusal leaves the directory exactly as it was
        (`test_one_bad_name_in_a_push_stores_none_of_the_good_ones`). Merging and
        not replacing is decision C.
        """
        held = self._directories.get(tenant, {})
        entries = checked_entries(names, tuple(held.values()))
        self._directories[tenant] = {**held, **entries}

    def get(self, tenant: str) -> tuple[str, ...]:
        """Every name this tenant has pushed, or the empty tuple."""
        return tuple(self._directories.get(tenant, {}).values())


class RedisDirectoryStore:
    """A `DirectoryStore` keeping directories in Redis, where every replica reads them.

    The implementation the gateway ships with once `OPSCONSOLE_REDIS_URL` is set. It differs from `InProcessDirectoryStore` in three ways
    that matter.

    One tenant is one Redis HASH, field `name_field(name)`, value the plaintext
    name (decision B). A hash rather than a set because the hash gives every entry
    a stable handle: a re-push overwrites its own field instead of adding a
    duplicate, and a later plan's delete has something to address
    (`test_the_field_is_the_hashed_name_and_the_value_is_the_plaintext`). No TTL
    is written, unlike a placeholder map's key: a directory is configuration.

    `tenant` is pre-checked before any command is sent, on BOTH
    methods. It is a client-supplied string that shapes the key, and a colon in it
    would forge a segment and address another tenant's namespace - on `get` that
    means compiling another tenant's names into the matcher. The rules are the
    gateway's own `IDENTIFIER_PATTERN` and `MAX_IDENTIFIER_LENGTH`, imported
    rather than restated, so the name a directory accepts and the name
    `gateway/context.py` authenticated cannot disagree
    (`test_put_refuses_a_malformed_tenant_before_touching_redis`,
    `test_get_refuses_a_malformed_tenant_before_touching_redis`).

    Every `redis.RedisError` becomes `DirectoryStoreUnavailable`, chained. Decision
    H is what makes that mandatory on the read path: an empty tuple would be a
    request redacted without the tenant's names and served
    (`test_a_get_against_an_unreachable_redis_raises_the_typed_failure`).

    `key_prefix` exists for the suite: per-test isolation is a namespace rather
    than a `FLUSHDB` that would delete keys this suite does not own (standing rule
    7). The default is the shipped one
    (`test_the_default_prefix_is_the_one_the_key_is_written_under`).

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, so a parametrised annotation fails `mypy --strict` with `type-arg`.
    """

    def __init__(self, client: redis.Redis, *, key_prefix: str = DIRECTORY_KEY_PREFIX) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def key(self, tenant: str) -> str:
        """The Redis key one tenant's directory is stored under.

        Public because it is what a migration, an operator's `SCAN` and the test
        that reads the key back all depend on
        (`test_the_key_carries_the_prefix_and_the_tenant`).
        """
        return f"{self._key_prefix}:{tenant}"

    def put(self, tenant: str, names: Sequence[str]) -> None:
        """Merge `names` into this tenant's hash, or raise having sent no `HSET`.

        The count cap is over the total, so what the tenant already holds is read
        first; that read is the only command a refused push sends, and the tests
        assert `hset` specifically for exactly that reason
        (`test_put_refuses_a_name_outside_the_caps_before_touching_redis`).
        """
        _checked_tenant(tenant)
        entries = checked_entries(names, self.get(tenant))
        if not entries:
            return
        # Widened to the client's own field and value aliases rather than passed as
        # `dict[str, str]`: a `Mapping` key type is invariant, so the narrower dict
        # is not the parameter's type under `mypy --strict`.
        payload: dict[FieldT, EncodableT] = {field: value for field, value in entries.items()}
        try:
            self._client.hset(self.key(tenant), mapping=payload)
        except redis.RedisError as exc:
            raise DirectoryStoreUnavailable("the directory store is unreachable") from exc

    def get(self, tenant: str) -> tuple[str, ...]:
        """Every name this tenant has pushed, decoded, or the empty tuple.

        Decoded here rather than by a `decode_responses` client, because the
        client is built once in `gateway/app.py` and shared with the placeholder
        map store, which stores JSON and decodes its own.
        """
        _checked_tenant(tenant)
        try:
            stored = self._client.hgetall(self.key(tenant))
        except redis.RedisError as exc:
            raise DirectoryStoreUnavailable("the directory store is unreachable") from exc
        return tuple(_as_text(value) for value in stored.values())


def _as_text(value: bytes | str) -> str:
    """One hash value as text, whether the client decoded it or not.

    The client is built once in `gateway/app.py` and shared with the placeholder
    map store, which stores JSON and decodes its own, so it is NOT built with
    `decode_responses`. Handling both spellings keeps this store correct if that
    ever changes, and keeps a `str` out of `bytes.decode`
    (`test_a_pushed_name_round_trips_through_redis`).
    """
    return value if isinstance(value, str) else value.decode("utf-8")


def _checked_tenant(tenant: str) -> None:
    """Refuse a tenant that may not shape a key, before any command is sent.

    The gateway's own identifier rules, imported rather than restated. Inclusive
    at the cap: exactly `MAX_IDENTIFIER_LENGTH` characters is accepted
    (`test_a_tenant_at_the_length_cap_is_accepted`,
    `test_a_tenant_one_character_over_the_cap_is_rejected`).
    """
    if len(tenant) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(
            f"tenant: longer than {MAX_IDENTIFIER_LENGTH} characters", param="tenant"
        )
    if IDENTIFIER_PATTERN.fullmatch(tenant) is None:
        raise IdentifierInvalid("tenant: not a valid identifier", param="tenant")
