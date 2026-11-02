"""`evals/world.py`: world facts and end states, read in code from the help desk's world.

Contract these tests are written against:

* `Fact` is built from the case file's own form: `Fact.model_validate({"holds": "wiki.editor"})`
  or `{"holds": "wiki.editor", "person": "sales.3"}`; one kind key (or `nothing_changed`),
  and an optional `person`. It has `.kind`, `.value` (a string, or `True` for the flag kinds)
  and `.person` (`None` when the case's own person is meant).
* `START_KINDS` (7 kinds a case may state about the start) and `END_KINDS` (all 13).
* `person_name(seat) -> str`: `"seed:engineering.employee.1"` is `"engineering.1"`.
* `fact_is_true(fact, *, own, before, after, now) -> bool`. State kinds read `after`;
  `open_review` and `nothing_changed` compare `before` with `after`. `own` is the case
  person's world name; a fact with `person` speaks of that person only.
* `false_start_facts(facts, *, own, world, now) -> tuple[Fact, ...]`: the facts that are
  false on the world the help desk returned before the first turn.
* `matching_end_state(end_states, *, own, before, after, now) -> int | None`: the index of the
  first end state whose facts are ALL true, or `None`.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from app.evals.target import EvalWorld
from app.evals.world import (
    END_KINDS,
    START_KINDS,
    Fact,
    fact_is_true,
    false_start_facts,
    matching_end_state,
    person_name,
)

from .builders import NOW, OTHER, OWN, fact, grant, human, req, review, ticket, world

ROLE = "finance-dash.viewer"
EMPTY = world()


def test_the_kinds_are_the_fixed_lists() -> None:
    """Catches a kind added or dropped without the help desk's list: seven kinds describe a
    start, six more only an end, and the loader refuses anything else."""
    assert {
        "holds",
        "lacks",
        "open_request",
        "no_open_request",
        "open_ticket",
        "no_open_ticket",
        "in_department",
    } == START_KINDS
    assert {
        "is_leaver",
        "user_exists",
        "reset_asked",
        "unlocked",
        "open_review",
        "nothing_changed",
    } == END_KINDS - START_KINDS
    assert len(END_KINDS) == 13


def test_a_seat_is_named_the_way_the_world_names_people() -> None:
    """Catches a name built from the whole seat: the world calls the first engineer
    `engineering.1`, so a fact about `seed:engineering.employee.1` would never be found."""
    assert person_name("seed:engineering.employee.1") == "engineering.1"
    assert person_name("seed:finance.it_admin.5") == "finance.5"


def test_an_unknown_kind_is_refused() -> None:
    """Catches a fact with a made-up kind passing as true or false: it must stop the load."""
    with pytest.raises(ValueError):
        Fact.model_validate({"likes_cake": "yes"})


Row = tuple[str, dict[str, Any], EvalWorld, EvalWorld, bool]

LONG_AGO = NOW - timedelta(days=1)
LATER = NOW + timedelta(days=1)

ROWS: list[Row] = [
    # holds: an active grant, for the right person, not expired.
    ("holds, active grant", {"holds": ROLE}, EMPTY, world(grants=[grant(OWN, ROLE)]), True),
    (
        "holds, grant that ends later",
        {"holds": ROLE},
        EMPTY,
        world(grants=[grant(OWN, ROLE, expires_at=LATER)]),
        True,
    ),
    (
        "holds, expired status",
        {"holds": ROLE},
        EMPTY,
        world(grants=[grant(OWN, ROLE, status="expired")]),
        False,
    ),
    (
        "holds, revoked status",
        {"holds": ROLE},
        EMPTY,
        world(grants=[grant(OWN, ROLE, status="revoked")]),
        False,
    ),
    (
        "holds, active status but the end date has passed",
        {"holds": ROLE},
        EMPTY,
        world(grants=[grant(OWN, ROLE, expires_at=LONG_AGO)]),
        False,
    ),
    (
        "holds, someone else holds it",
        {"holds": ROLE},
        EMPTY,
        world(grants=[grant(OTHER, ROLE)]),
        False,
    ),
    (
        "holds, another role",
        {"holds": ROLE},
        EMPTY,
        world(grants=[grant(OWN, "wiki.editor")]),
        False,
    ),
    (
        "holds with person, that person holds it",
        {"holds": ROLE, "person": OTHER},
        EMPTY,
        world(grants=[grant(OTHER, ROLE)]),
        True,
    ),
    (
        "holds with person, only the case person holds it",
        {"holds": ROLE, "person": OTHER},
        EMPTY,
        world(grants=[grant(OWN, ROLE)]),
        False,
    ),
    # lacks is the opposite of holds.
    ("lacks, nothing held", {"lacks": ROLE}, EMPTY, EMPTY, True),
    (
        "lacks, expired grant",
        {"lacks": ROLE},
        EMPTY,
        world(grants=[grant(OWN, ROLE, status="expired")]),
        True,
    ),
    ("lacks, held", {"lacks": ROLE}, EMPTY, world(grants=[grant(OWN, ROLE)]), False),
    (
        "lacks with person, that person holds it",
        {"lacks": ROLE, "person": OTHER},
        EMPTY,
        world(grants=[grant(OTHER, ROLE)]),
        False,
    ),
    (
        "lacks with person, only the case person holds it",
        {"lacks": ROLE, "person": OTHER},
        EMPTY,
        world(grants=[grant(OWN, ROLE)]),
        True,
    ),
    # open_request: a pending request for that person and role.
    (
        "open_request, pending",
        {"open_request": ROLE},
        EMPTY,
        world(requests=[req(1, target=OWN, role=ROLE)]),
        True,
    ),
    (
        "open_request, already decided",
        {"open_request": ROLE},
        EMPTY,
        world(requests=[req(1, target=OWN, role=ROLE, status="approved")]),
        False,
    ),
    (
        "open_request, another role",
        {"open_request": ROLE},
        EMPTY,
        world(requests=[req(1, target=OWN, role="wiki.editor")]),
        False,
    ),
    (
        "open_request, filed for someone else",
        {"open_request": ROLE},
        EMPTY,
        world(requests=[req(1, target=OTHER, requester=OWN, role=ROLE)]),
        False,
    ),
    (
        "open_request with person, filed for that person",
        {"open_request": ROLE, "person": OTHER},
        EMPTY,
        world(requests=[req(1, target=OTHER, requester=OWN, role=ROLE)]),
        True,
    ),
    (
        "open_request with person, only the case person has one",
        {"open_request": ROLE, "person": OTHER},
        EMPTY,
        world(requests=[req(1, target=OWN, role=ROLE)]),
        False,
    ),
    # no_open_request is the opposite.
    ("no_open_request, none", {"no_open_request": ROLE}, EMPTY, EMPTY, True),
    (
        "no_open_request, one pending",
        {"no_open_request": ROLE},
        EMPTY,
        world(requests=[req(1, target=OWN, role=ROLE)]),
        False,
    ),
    (
        "no_open_request, only a decided one",
        {"no_open_request": ROLE},
        EMPTY,
        world(requests=[req(1, target=OWN, role=ROLE, status="rejected")]),
        True,
    ),
    (
        "no_open_request with person, that person has one",
        {"no_open_request": ROLE, "person": OTHER},
        EMPTY,
        world(requests=[req(1, target=OTHER, role=ROLE)]),
        False,
    ),
    # open_ticket / no_open_ticket: by kind and person.
    (
        "open_ticket, open",
        {"open_ticket": "password_reset"},
        EMPTY,
        world(tickets=[ticket(1, person=OWN)]),
        True,
    ),
    (
        "open_ticket, solved",
        {"open_ticket": "password_reset"},
        EMPTY,
        world(tickets=[ticket(1, person=OWN, status="solved")]),
        False,
    ),
    (
        "open_ticket, another kind",
        {"open_ticket": "password_reset"},
        EMPTY,
        world(tickets=[ticket(1, person=OWN, kind="hardware")]),
        False,
    ),
    (
        "open_ticket with person",
        {"open_ticket": "password_reset", "person": OTHER},
        EMPTY,
        world(tickets=[ticket(1, person=OWN)]),
        False,
    ),
    ("no_open_ticket, none", {"no_open_ticket": "password_reset"}, EMPTY, EMPTY, True),
    (
        "no_open_ticket, one open",
        {"no_open_ticket": "password_reset"},
        EMPTY,
        world(tickets=[ticket(1, person=OWN)]),
        False,
    ),
    (
        "no_open_ticket with person, the case person has one",
        {"no_open_ticket": "password_reset", "person": OTHER},
        EMPTY,
        world(tickets=[ticket(1, person=OWN)]),
        True,
    ),
    # in_department
    (
        "in_department, same",
        {"in_department": "Production"},
        EMPTY,
        world(people=[human(OWN, department="Production")]),
        True,
    ),
    (
        "in_department, other",
        {"in_department": "Production"},
        EMPTY,
        world(people=[human(OWN, department="Finance")]),
        False,
    ),
    (
        "in_department, no department",
        {"in_department": "Production"},
        EMPTY,
        world(people=[human(OWN, department=None)]),
        False,
    ),
    (
        "in_department with person",
        {"in_department": "Finance", "person": OTHER},
        EMPTY,
        world(people=[human(OWN, department="Production"), human(OTHER, department="Finance")]),
        True,
    ),
    # is_leaver
    (
        "is_leaver",
        {"is_leaver": True},
        EMPTY,
        world(people=[human(OWN, status="leaver")]),
        True,
    ),
    (
        "is_leaver, still active",
        {"is_leaver": True},
        EMPTY,
        world(people=[human(OWN, status="active")]),
        False,
    ),
    (
        "is_leaver with person, the case person left",
        {"is_leaver": True, "person": OTHER},
        EMPTY,
        world(people=[human(OWN, status="leaver"), human(OTHER, status="active")]),
        False,
    ),
    # user_exists: by email, no person.
    (
        "user_exists",
        {"user_exists": "new.joiner@nordlicht.example"},
        EMPTY,
        world(people=[human("new.joiner@nordlicht.example")]),
        True,
    ),
    ("user_exists, nobody", {"user_exists": "new.joiner@nordlicht.example"}, EMPTY, EMPTY, False),
    # reset_asked / unlocked
    (
        "reset_asked",
        {"reset_asked": True},
        EMPTY,
        world(people=[human(OWN, reset_asked=True)]),
        True,
    ),
    ("reset_asked, not asked", {"reset_asked": True}, EMPTY, world(people=[human(OWN)]), False),
    (
        "reset_asked with person",
        {"reset_asked": True, "person": OTHER},
        EMPTY,
        world(people=[human(OWN, reset_asked=True), human(OTHER)]),
        False,
    ),
    ("unlocked", {"unlocked": True}, EMPTY, world(people=[human(OWN, unlocked=True)]), True),
    ("unlocked, still locked", {"unlocked": True}, EMPTY, world(people=[human(OWN)]), False),
    (
        "unlocked with person",
        {"unlocked": True, "person": OTHER},
        EMPTY,
        world(people=[human(OWN), human(OTHER, unlocked=True)]),
        True,
    ),
    # open_review: more open reviews than before.
    (
        "open_review, one more open",
        {"open_review": "Finance"},
        EMPTY,
        world(reviews=[review("open")]),
        True,
    ),
    (
        "open_review, same number",
        {"open_review": "Finance"},
        world(reviews=[review("open")]),
        world(reviews=[review("open")]),
        False,
    ),
    (
        "open_review, new one is already completed",
        {"open_review": "Finance"},
        EMPTY,
        world(reviews=[review("completed")]),
        False,
    ),
    (
        "open_review, one open became two",
        {"open_review": "Finance"},
        world(reviews=[review("open")]),
        world(reviews=[review("open"), review("open")]),
        True,
    ),
    # nothing_changed: after equals before.
    (
        "nothing_changed, same",
        {"nothing_changed": True},
        world(grants=[grant()]),
        world(grants=[grant()]),
        True,
    ),
    (
        "nothing_changed, a grant added",
        {"nothing_changed": True},
        EMPTY,
        world(grants=[grant()]),
        False,
    ),
    (
        "nothing_changed, a request cancelled",
        {"nothing_changed": True},
        world(requests=[req(1)]),
        world(requests=[req(1, status="cancelled")]),
        False,
    ),
]


@pytest.mark.parametrize(
    ("document", "before", "after", "expected"),
    [row[1:] for row in ROWS],
    ids=[row[0] for row in ROWS],
)
def test_each_fact_kind_reads_the_world(
    document: dict[str, Any], before: EvalWorld, after: EvalWorld, expected: bool
) -> None:
    """Catches a fact kind read from the wrong part of the world: one that ignores the
    person (so a fact about someone else is answered from the case person's rows), one that
    ignores status (a decided request or a solved ticket still counts as open), one that
    counts an expired or revoked grant as held, and a review or change check that looks only
    at `after`. Every one of the 13 kinds has a true row and a false row, and the person
    kinds have a row for each person, so the person is the only thing that flips them."""
    result = fact_is_true(fact(**document), own=OWN, before=before, after=after, now=NOW)

    assert result is expected


def test_every_one_of_the_13_kinds_has_a_row() -> None:
    """Catches a kind added to the list that no row above reads: the table must cover all."""
    covered = {next(k for k in row[1] if k != "person") for row in ROWS}

    assert covered == END_KINDS


def test_a_false_start_fact_makes_the_play_setup_wrong() -> None:
    """Catches a start check that lets a case run over a world that is not the one it was
    written for: `open_request` with no pending row is false, and it is the only fact
    returned. The true fact beside it is not named, so the list is the false facts alone."""
    facts = [fact(lacks=ROLE), fact(open_request=ROLE)]
    start = world()

    false = false_start_facts(facts, own=OWN, world=start, now=NOW)

    assert false == (facts[1],)


def test_a_true_start_leaves_no_false_fact() -> None:
    """Catches a check that reports a fact against a world that satisfies it: the same facts
    over a world with the request pending give an empty answer, so a play is not dropped
    for nothing."""
    facts = [fact(lacks=ROLE), fact(open_request=ROLE)]
    start = world(requests=[req(1, target=OWN, role=ROLE)])

    assert false_start_facts(facts, own=OWN, world=start, now=NOW) == ()


def test_an_end_state_matches_only_when_all_its_facts_are_true() -> None:
    """Catches a match on any one fact: the first end state has two facts and only one is
    true, so it must not match; the second has both true and is the one returned. Also
    catches the index being lost: the answer is the position, 1."""
    before = world()
    after = world(requests=[req(1, target=OWN, role=ROLE)])
    states = [
        [fact(open_request=ROLE), fact(holds=ROLE)],
        [fact(open_request=ROLE), fact(lacks=ROLE)],
    ]

    index = matching_end_state(states, own=OWN, before=before, after=after, now=NOW)

    assert index == 1


def test_a_world_that_fits_no_end_state_matches_none() -> None:
    """Catches a match that defaults to the first end state when nothing fits."""
    states = [[fact(holds=ROLE)], [fact(open_request=ROLE)]]

    assert matching_end_state(states, own=OWN, before=EMPTY, after=EMPTY, now=NOW) is None
