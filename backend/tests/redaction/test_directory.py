r"""`InProcessDirectoryStore`: the three caps, the normalisation and the hashed field.

Everything here is in-process and pure CPU. The Redis half of the same contract -
the key shape, the hashed field read back off the server, and the typed
unavailable error - is `tests/redaction/test_redis_directory.py`.

Three claims carry the file.

* The caps. `MIN_DIRECTORY_NAME_CHARS`, `MAX_DIRECTORY_NAME_CHARS` and
  `MAX_DIRECTORY_NAMES` are new values, so each is pinned to its
  exact literal and each has its inclusive boundary pair. The count cap is pinned against the
  directory TOTAL rather
  than against one push, because a per-push reading lets an unbounded directory
  in through pagination and the compiled alternation is the only unbounded thing
  here.
* Nothing is written when anything is refused. Every rejection row reads the
  directory back afterwards, so a store that writes the good names and then
  raises on the bad one - which leaves a tenant's directory half applied and the
  caller with no way to know which half - fails here rather than in production.
* No name reaches the rejection. The message names the field and the cap; the
  value it refused is what this whole subsystem exists to keep out of logs and
  error bodies, so it is asserted absent rather than assumed.
"""

from __future__ import annotations

import hashlib

import pytest

from app.core.errors import GatewayError
from app.redaction.errors import DirectoryNameInvalid, DirectoryStoreUnavailable
from app.redaction.names_store import (
    DIRECTORY_KEY_PREFIX,
    MAX_DIRECTORY_NAME_CHARS,
    MAX_DIRECTORY_NAMES,
    MIN_DIRECTORY_NAME_CHARS,
    DirectoryStore,
    InProcessDirectoryStore,
    name_field,
    normalise_name,
)

TENANT = "nordlicht"
OTHER_TENANT = "cuan"

NAME_A = "Anna Schmidt"
NAME_B = "Bernd Kowalski"

AT_MIN_NAME = "Li"
UNDER_MIN_NAME = "L"
AT_MAX_NAME = "a" * MAX_DIRECTORY_NAME_CHARS
OVER_MAX_NAME = "a" * (MAX_DIRECTORY_NAME_CHARS + 1)


def store() -> InProcessDirectoryStore:
    """A directory nobody else has written to. In-process and free to build."""
    return InProcessDirectoryStore()


def bulk(count: int, prefix: str = "Persona") -> list[str]:
    """`count` distinct legal names, for the rows about the count cap."""
    return [f"{prefix}{index:05d}" for index in range(count)]


# --------------------------------------------------------------------------
# the constants and the exports
# --------------------------------------------------------------------------


def test_the_three_caps_are_the_published_numbers() -> None:
    """Catches a cap moved to a plausible neighbour (a one-character minimum, a 64 or 256
    character maximum): the minimum is the one an operator can break precision with in a
    single typo, so a band would admit exactly the value that must be rejected."""
    assert MIN_DIRECTORY_NAME_CHARS == 2
    assert MAX_DIRECTORY_NAME_CHARS == 128
    assert MAX_DIRECTORY_NAMES == 2000


def test_the_key_prefix_is_the_published_one() -> None:
    """Catches a prefix renamed after data exists under the old one: a tenant's directory
    is addressed by this string, so changing it silently empties every directory an
    operator has pushed."""
    assert DIRECTORY_KEY_PREFIX == "directory"


def test_the_directory_names_are_exported_from_the_package() -> None:
    """Catches an implementation that lands everything in `directory.py` and never
    re-exports it: `gateway/app.py` and the routes import from `app.redaction`, so
    a missing export is an ImportError at process start rather than at a test."""
    from app import redaction

    assert redaction.InProcessDirectoryStore is InProcessDirectoryStore
    assert redaction.DirectoryStore is DirectoryStore
    assert redaction.DirectoryNameInvalid is DirectoryNameInvalid
    assert redaction.DirectoryStoreUnavailable is DirectoryStoreUnavailable
    assert redaction.MAX_DIRECTORY_NAMES == MAX_DIRECTORY_NAMES


def test_the_cap_failure_is_not_a_gateway_error() -> None:
    """Catches a cap failure made a `GatewayError` subclass: `ErrorCode.VALIDATION`
    already belongs to `IdentifierInvalid` and `test_error_code_set_is_exhaustive` allows
    exactly one concrete class per code, so a subclass here reddens a test in another
    directory with no obvious connection to this one."""
    assert issubclass(DirectoryNameInvalid, Exception)
    assert not issubclass(DirectoryNameInvalid, GatewayError)
    assert issubclass(DirectoryStoreUnavailable, Exception)
    assert not issubclass(DirectoryStoreUnavailable, GatewayError)


def test_the_in_process_store_satisfies_the_directory_protocol() -> None:
    """Catches method names or signatures that drift from `DirectoryStore`: the routes
    read whichever store `create_app` chose through this protocol, so a `put` taking its
    arguments in another order is a 500 on the first redacted request, and `mypy --strict`
    is what the annotation below asks."""
    protocol_typed: DirectoryStore = InProcessDirectoryStore()

    protocol_typed.put(TENANT, [NAME_A])

    assert protocol_typed.get(TENANT) == (NAME_A,)


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------


def test_an_empty_directory_answers_an_empty_tuple() -> None:
    """Catches a `get` that raises `KeyError` for a tenant nobody has pushed for: every
    redacted request reads the directory, so a raise here is a 500 on the FIRST request of
    every tenant that has no names, which is most of them."""
    assert store().get(TENANT) == ()


def test_a_pushed_name_is_read_back() -> None:
    """Catches a `put` that stores the hashed field alone: the store keeps a hashed key and
    plaintext value, and the matcher needs the plaintext, so a hash-only store compiles an
    alternation of hex digests that matches nothing."""
    directory = store()

    directory.put(TENANT, [NAME_A, NAME_B])

    assert sorted(directory.get(TENANT)) == sorted([NAME_A, NAME_B])


def test_a_second_push_merges_rather_than_replacing() -> None:
    """Catches replace-whole semantics: a caller that pages a large directory
    would silently delete everything it had already pushed, and the failure is invisible
    until a name stops being redacted."""
    directory = store()

    directory.put(TENANT, [NAME_A])
    directory.put(TENANT, [NAME_B])

    assert sorted(directory.get(TENANT)) == sorted([NAME_A, NAME_B])


def test_a_push_of_no_names_leaves_the_directory_unchanged() -> None:
    """Catches an empty push read as "replace with nothing", and one that raises: an empty
    list is a legal body and the honest answer is that nothing happened."""
    directory = store()
    directory.put(TENANT, [NAME_A])

    directory.put(TENANT, [])

    assert directory.get(TENANT) == (NAME_A,)


def test_one_tenants_names_are_not_readable_under_another_tenant() -> None:
    """Catches a directory kept in one namespace for every tenant: one tenant's staff
    names would then be redacted out of another tenant's traffic, which is both a leak of
    who works where and a precision failure nobody can explain. The closing read proves
    the refusal is not a store that simply lost the names."""
    directory = store()

    directory.put(TENANT, [NAME_A])
    directory.put(OTHER_TENANT, [NAME_B])

    assert directory.get(OTHER_TENANT) == (NAME_B,)
    assert directory.get(TENANT) == (NAME_A,)


# --------------------------------------------------------------------------
# normalisation and the hashed field
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "variant"),
    [
        ("the same spelling", "Anna Schmidt"),
        ("all lowercase", "anna schmidt"),
        ("all uppercase", "ANNA SCHMIDT"),
        ("leading whitespace", "   Anna Schmidt"),
        ("trailing whitespace", "Anna Schmidt   "),
        ("a trailing newline", "Anna Schmidt\n"),
        ("a trailing CRLF", "Anna Schmidt\r\n"),
    ],
)
def test_re_pushing_a_name_the_directory_already_holds_changes_nothing(
    case: str, variant: str
) -> None:
    """Catches a store keyed on the raw string: the field is `sha256(normalised name)`
    exactly so that a re-push from an operator who typed the name slightly differently is
    idempotent rather than a second entry that eats a slot of the 2000 and
    compiles a duplicate branch into the alternation."""
    directory = store()
    directory.put(TENANT, [NAME_A])

    directory.put(TENANT, [variant])

    assert len(directory.get(TENANT)) == 1


def test_normalisation_folds_case_and_strips_surrounding_whitespace() -> None:
    """Catches a normalisation that only strips or only case-folds: the idempotence rows
    above are satisfiable by either half alone in isolation, and this is the row that says
    which transformation the hashed field is over."""
    assert normalise_name("  Anna Schmidt \n") == normalise_name("anna schmidt")
    assert normalise_name("ANNA SCHMIDT") == normalise_name("anna schmidt")
    assert normalise_name("Anna Schmidt") != normalise_name("Anna Schmid")


def test_the_field_is_the_sha256_of_the_normalised_name() -> None:
    """Catches a field that is the plaintext name, a truncated digest, or a hash of the
    RAW string: the key is a hash, and the digest is computed here from
    `hashlib` directly rather than by calling the helper twice, so a helper that answered a
    constant would still fail."""
    expected = hashlib.sha256(normalise_name(NAME_A).encode("utf-8")).hexdigest()

    assert name_field(NAME_A) == expected
    assert name_field("  ANNA SCHMIDT  ") == expected
    assert len(name_field(NAME_A)) == 64
    assert name_field(NAME_B) != expected


# --------------------------------------------------------------------------
# the caps, each with its inclusive boundary pair
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "name"),
    [
        ("exactly the minimum length", AT_MIN_NAME),
        ("exactly the maximum length", AT_MAX_NAME),
    ],
)
def test_a_name_at_a_length_boundary_is_accepted(case: str, name: str) -> None:
    """Catches a comparison written `>` where `>=` belongs and the reverse: a two-letter
    surname is a real name and a 128-character one is what is documented to fit, so
    refusing either would reject a push an operator is entitled to make. The half of each
    pair that a cap alone cannot get right."""
    directory = store()

    directory.put(TENANT, [name])

    assert directory.get(TENANT) == (name,)


@pytest.mark.parametrize(
    ("case", "name", "limit"),
    [
        ("one character under the minimum", UNDER_MIN_NAME, MIN_DIRECTORY_NAME_CHARS),
        ("empty", "", MIN_DIRECTORY_NAME_CHARS),
        ("one character over the maximum", OVER_MAX_NAME, MAX_DIRECTORY_NAME_CHARS),
    ],
)
def test_a_name_outside_the_length_bounds_is_refused_and_nothing_is_stored(
    case: str, name: str, limit: int
) -> None:
    """Catches a cap that was never applied - the one-character name is the whole
    reason, because it redacts every `L` in every request that tenant sends - and a check
    run after the write, which the closing read is what sees."""
    directory = store()

    with pytest.raises(DirectoryNameInvalid) as raised:
        directory.put(TENANT, [name])

    assert raised.value.limit == limit
    assert directory.get(TENANT) == ()


def test_a_name_whose_length_is_only_whitespace_is_refused() -> None:
    """Catches a length check applied to the RAW string rather than to the normalised one:
    `" L "` is three characters raw and one real one, so an unnormalised check stores a
    single-character name through the front door the minimum exists to close."""
    directory = store()

    with pytest.raises(DirectoryNameInvalid):
        directory.put(TENANT, ["  L  "])

    assert directory.get(TENANT) == ()


def test_one_bad_name_in_a_push_stores_none_of_the_good_ones() -> None:
    """Catches a store that validates inside its write loop: the good names before the bad
    one would be stored and the caller would get an error naming a cap, leaving the
    directory in a state neither side believes in. The bad name is LAST, which is the only
    order in which a per-item loop gets as far as writing."""
    directory = store()

    with pytest.raises(DirectoryNameInvalid):
        directory.put(TENANT, [NAME_A, NAME_B, UNDER_MIN_NAME])

    assert directory.get(TENANT) == ()


def test_a_push_at_the_count_cap_is_accepted() -> None:
    """Catches a count cap written `>=`: exactly 2000 names is the largest legal directory, and
    refusing it would make the published number a lie by one."""
    directory = store()

    directory.put(TENANT, bulk(MAX_DIRECTORY_NAMES))

    assert len(directory.get(TENANT)) == MAX_DIRECTORY_NAMES


def test_a_push_one_name_over_the_count_cap_is_refused() -> None:
    """The other half of the boundary pair: catches a count cap that was never applied,
    which the accepted row alone cannot see, and leaves the compiled alternation the only
    unbounded thing here."""
    directory = store()

    with pytest.raises(DirectoryNameInvalid) as raised:
        directory.put(TENANT, bulk(MAX_DIRECTORY_NAMES + 1))

    assert raised.value.limit == MAX_DIRECTORY_NAMES
    assert directory.get(TENANT) == ()


def test_the_count_cap_is_over_the_directory_total_and_not_over_one_push() -> None:
    """Catches a cap compared against `len(names)` alone: two pushes of 1500 are each
    legal on their own, so a per-push reading lets a paged caller build a directory of any
    size at all and the bound means nothing. The first push must survive,
    which is what separates this from a cap that simply refuses everything."""
    directory = store()
    directory.put(TENANT, bulk(1500, prefix="First"))

    with pytest.raises(DirectoryNameInvalid) as raised:
        directory.put(TENANT, bulk(501, prefix="Second"))

    assert raised.value.limit == MAX_DIRECTORY_NAMES
    assert len(directory.get(TENANT)) == 1500


def test_the_count_cap_counts_a_re_push_once() -> None:
    """Catches a total computed as "held plus offered" without noticing the overlap: a
    caller re-pushing the directory it already holds would be refused for a directory that
    is not growing at all, which is exactly what the merge semantics invite."""
    directory = store()
    directory.put(TENANT, bulk(MAX_DIRECTORY_NAMES))

    directory.put(TENANT, bulk(MAX_DIRECTORY_NAMES))

    assert len(directory.get(TENANT)) == MAX_DIRECTORY_NAMES


def test_the_count_cap_is_per_tenant() -> None:
    """Catches a cap applied to the whole store: one tenant filling its directory would
    then refuse every other tenant's first push, and the outage would look like a bug in
    the pushing tenant's own request."""
    directory = store()
    directory.put(TENANT, bulk(MAX_DIRECTORY_NAMES))

    directory.put(OTHER_TENANT, [NAME_A])

    assert directory.get(OTHER_TENANT) == (NAME_A,)


# --------------------------------------------------------------------------
# what a rejection says
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "name", "limit"),
    [
        ("under the minimum", "L", MIN_DIRECTORY_NAME_CHARS),
        ("over the maximum", "Q" * (MAX_DIRECTORY_NAME_CHARS + 1), MAX_DIRECTORY_NAME_CHARS),
    ],
)
def test_a_rejection_names_the_field_and_the_cap_and_never_the_value(
    case: str, name: str, limit: int
) -> None:
    """Catches a message built as `f"{name} is too short"`: the value is a person's name,
    the criterion says it reaches no log line and no error body, and this exception is
    rendered straight into a 422 the caller reads. The cap has to be there for the same
    message to be actionable, so both halves are asserted at once."""
    directory = store()

    with pytest.raises(DirectoryNameInvalid) as raised:
        directory.put(TENANT, [name])

    message = str(raised.value)
    assert name not in message
    assert str(limit) in message
    assert "names" in message
    assert raised.value.param == "names"
