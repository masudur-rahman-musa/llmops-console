"""World facts and end states, read in code from the help desk's world.

A fact is one key from a fixed list, with an optional `person` for someone other than the
case's own person. `holds`, `lacks`, `open_request`, `no_open_request`, `open_ticket`,
`no_open_ticket` and `in_department` may describe the start; `is_leaver`, `user_exists`,
`reset_asked`, `unlocked`, `open_review` and `nothing_changed` only an end. The state
kinds read the world after; `open_review` and `nothing_changed` compare it with the world
before. One branch reads each kind, so a fact never falls through to a default.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, model_validator

from app.evals.target import EvalWorld, WorldPerson

START_KINDS: Final[frozenset[str]] = frozenset(
    {
        "holds",
        "lacks",
        "open_request",
        "no_open_request",
        "open_ticket",
        "no_open_ticket",
        "in_department",
    }
)
END_KINDS: Final[frozenset[str]] = START_KINDS | frozenset(
    {"is_leaver", "user_exists", "reset_asked", "unlocked", "open_review", "nothing_changed"}
)

_SEAT: Final[re.Pattern[str]] = re.compile(r"(?:seed:)?([a-z0-9_-]+)\.[a-z_]+\.([0-9]+)")


def person_name(seat: str) -> str:
    """The name the world gives a seeded person: department lowercased, a dot, the index."""
    found = _SEAT.fullmatch(seat)
    if found is None:
        return seat
    return f"{found.group(1)}.{found.group(2)}"


class Fact(BaseModel):
    """One fact in the case file's own form, e.g. `{"holds": "wiki.editor"}`."""

    model_config = ConfigDict(frozen=True)

    kind: str
    value: str | bool
    person: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_document(cls, data: Any) -> Any:  # noqa: ANN401
        if not isinstance(data, Mapping):
            raise ValueError("a fact is an object")
        if "kind" in data:
            return data
        keys = [key for key in data if key != "person"]
        if len(keys) != 1:
            raise ValueError("a fact holds exactly one kind")
        kind = keys[0]
        if kind not in END_KINDS:
            raise ValueError(f"{kind!r} is not a fact kind")
        return {"kind": kind, "value": data[kind], "person": data.get("person")}


def _name(fact: Fact, own: str) -> str:
    return fact.person if fact.person is not None else own


def _holds(fact: Fact, own: str, after: EvalWorld, now: datetime) -> bool:
    who = _name(fact, own)
    return any(
        g.person == who
        and g.role == fact.value
        and g.status == "active"
        and (g.expires_at is None or g.expires_at > now)
        for g in after.grants
    )


def _open_request(fact: Fact, own: str, after: EvalWorld) -> bool:
    who = _name(fact, own)
    return any(
        r.target == who and r.role == fact.value and r.status == "pending" for r in after.requests
    )


def _open_ticket(fact: Fact, own: str, after: EvalWorld) -> bool:
    who = _name(fact, own)
    return any(
        t.person == who and t.kind == fact.value and t.status == "open" for t in after.tickets
    )


def _person_is(fact: Fact, own: str, after: EvalWorld, test: Callable[[WorldPerson], bool]) -> bool:
    who = _name(fact, own)
    return any(p.person == who and test(p) for p in after.people)


def _open_reviews(world: EvalWorld) -> int:
    return sum(1 for r in world.reviews if r.status == "open")


def fact_is_true(
    fact: Fact, *, own: str, before: EvalWorld, after: EvalWorld, now: datetime
) -> bool:
    """Whether one fact holds, reading `after` (and `before` for the two change kinds)."""
    kind = fact.kind
    if kind == "holds":
        return _holds(fact, own, after, now)
    if kind == "lacks":
        return not _holds(fact, own, after, now)
    if kind == "open_request":
        return _open_request(fact, own, after)
    if kind == "no_open_request":
        return not _open_request(fact, own, after)
    if kind == "open_ticket":
        return _open_ticket(fact, own, after)
    if kind == "no_open_ticket":
        return not _open_ticket(fact, own, after)
    if kind == "in_department":
        return _person_is(fact, own, after, lambda p: p.department == fact.value)
    if kind == "is_leaver":
        return _person_is(fact, own, after, lambda p: p.status == "leaver")
    if kind == "user_exists":
        return any(p.person == fact.value for p in after.people)
    if kind == "reset_asked":
        return _person_is(fact, own, after, lambda p: p.reset_asked)
    if kind == "unlocked":
        return _person_is(fact, own, after, lambda p: p.unlocked)
    if kind == "open_review":
        return _open_reviews(after) > _open_reviews(before)
    if kind == "nothing_changed":
        return after == before
    raise ValueError(f"{kind!r} is not a fact kind")


def false_start_facts(
    facts: Sequence[Fact], *, own: str, world: EvalWorld, now: datetime
) -> tuple[Fact, ...]:
    """The facts that are false on the world the help desk returned before the first turn."""
    return tuple(
        f for f in facts if not fact_is_true(f, own=own, before=world, after=world, now=now)
    )


def matching_end_state(
    end_states: Sequence[Sequence[Fact]],
    *,
    own: str,
    before: EvalWorld,
    after: EvalWorld,
    now: datetime,
) -> int | None:
    """The index of the first end state whose facts are all true, or `None`."""
    for index, facts in enumerate(end_states):
        if all(fact_is_true(f, own=own, before=before, after=after, now=now) for f in facts):
            return index
    return None
