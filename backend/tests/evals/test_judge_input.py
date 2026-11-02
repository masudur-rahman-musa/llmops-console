"""`evals/judge.py`: a long role list in a tool result is cut to the roles the reply names.

Contract these tests are written against:

* In an ok tool result, a list of MORE than 10 items that each carry `system_slug` and
  `role_slug` keeps only the items whose `system_slug.role_slug` or `role_name` (any case)
  appears in that turn's reply. The list is followed by
  `(kept K of N roles: the ones the reply names)`.
* Every other result is rendered whole: a list of 10 or fewer, a list whose items miss a
  key, a refused call, a result that is not a list.
* `render_conversation` and `render_label_conversation` cut the same way.
"""

from __future__ import annotations

import json
from typing import Any

from app.evals.judge import render_conversation, render_label_conversation
from app.evals.labels import LabelToolCall, LabelTurn

from .builders import call, played

LIST_TOOL = "list_requestable_roles"


def role(number: int) -> dict[str, Any]:
    """One role row. Its slugs and name carry the number, so no row names another."""
    return {
        "system_slug": f"s{number:03d}",
        "role_slug": f"r{number:03d}",
        "role_name": f"Role {number:03d}",
        "already_held": False,
    }


def roles(count: int) -> list[dict[str, Any]]:
    return [role(n) for n in range(1, count + 1)]


def two_named() -> list[dict[str, Any]]:
    """200 roles where row 7 is found by its slug pair only and row 42 by its name only."""
    rows = roles(200)
    rows[6] = {**rows[6], "system_slug": "kyc", "role_slug": "analyst"}
    rows[41] = {**rows[41], "role_name": "Wiki Editor"}
    return rows


REPLY_NAMING_TWO = "You can ask for kyc.analyst, or for the WIKI EDITOR role."


def marker(kept: int, total: int) -> str:
    return f"(kept {kept} of {total} roles: the ones the reply names)"


def kept_slugs(text: str) -> list[str]:
    """Every role slug written in the rendered text, in order."""
    return [part.split('"')[0] for part in text.split('"role_slug": "')[1:]]


def test_a_long_role_list_keeps_only_the_roles_the_reply_names() -> None:
    """Catches a judge input that still sends all 200 roles, or one that matches only the
    slug pair or only the name: row 7 is named by `kyc.analyst` alone and row 42 by its
    name `Wiki Editor` alone (reply in capitals), so each match rule is needed. No other role
    slug may appear."""
    turn = played(
        "What can I get?",
        REPLY_NAMING_TWO,
        call(LIST_TOOL, result={"roles": two_named()}),
    )

    text = render_conversation([turn])

    assert kept_slugs(text) == ["analyst", "r042"]
    assert marker(2, 200) in text
    assert text.index(marker(2, 200)) > text.index('"r042"')
    assert '"r001"' not in text
    assert '"r200"' not in text


def test_a_reply_that_names_no_role_keeps_none() -> None:
    """Catches a cut that keeps a default slice (the first few roles) when the reply names
    nothing: no role stays and the marker says 0 of 200."""
    turn = played(
        "What can I get?", "Several roles exist.", call(LIST_TOOL, result={"roles": roles(200)})
    )

    text = render_conversation([turn])

    assert kept_slugs(text) == []
    assert marker(0, 200) in text


def test_a_list_of_10_roles_is_sent_whole() -> None:
    """Catches a threshold off by one: exactly 10 roles are all sent with no marker, and 11
    roles are cut to the one the reply names (the reply names role 3 in both)."""
    reply = "Role 003 is the one."

    ten = render_conversation([played("q", reply, call(LIST_TOOL, result={"roles": roles(10)}))])
    eleven = render_conversation([played("q", reply, call(LIST_TOOL, result={"roles": roles(11)}))])

    assert len(kept_slugs(ten)) == 10
    assert "(kept" not in ten
    assert kept_slugs(eleven) == ["r003"]
    assert marker(1, 11) in eleven


def test_other_results_are_sent_whole() -> None:
    """Catches a cut that fires on the wrong results, one case per row: a long list whose
    items have no `role_slug`, a long list where one item lacks `system_slug`, a refused call
    that still carries a long role list, and a dict result that is not a list. Each renders
    as its plain JSON and carries no marker."""
    no_role_slug = [{"system_slug": f"s{n}", "name": f"n{n}"} for n in range(30)]
    one_without_system = [*roles(29), {"role_slug": "r999", "role_name": "Role 999"}]
    refused_list = {"roles": roles(30)}
    plain_dict = {"count": 30, "note": "x" * 20}
    turn = played(
        "q",
        "Nothing here names a role.",
        call("list_other", result={"items": no_role_slug}),
        call("list_mixed", result={"roles": one_without_system}),
        call(LIST_TOOL, ok=False, result=refused_list, error_code="denied"),
        call("count_things", result=plain_dict),
    )

    text = render_conversation([turn])

    assert "(kept" not in text
    for result in (
        {"items": no_role_slug},
        {"roles": one_without_system},
        refused_list,
        plain_dict,
    ):
        assert f"result={json.dumps(result, ensure_ascii=False)}" in text


def test_a_label_conversation_is_cut_the_same_way() -> None:
    """Catches a calibration render that skips the cut: a labelled turn with the same 200
    roles and reply must render exactly as a run does (world lines aside), so calibration
    tests the text a run sends."""
    rows = two_named()
    run_turn = played(
        "What can I get?",
        REPLY_NAMING_TWO,
        call(LIST_TOOL, {"system": "all"}, result={"roles": rows}),
    )
    label_turn = LabelTurn(
        person="What can I get?",
        reply=REPLY_NAMING_TWO,
        tool_calls=(
            LabelToolCall(
                name=LIST_TOOL,
                args={"system": "all"},
                ok=True,
                error_code=None,
                result={"roles": rows},
            ),
        ),
    )
    expected = render_conversation([run_turn]).split("\nWorld facts before")[0]

    rendered = render_label_conversation([label_turn])

    assert rendered == expected
    assert marker(2, 200) in rendered
    assert '"r001"' not in rendered
