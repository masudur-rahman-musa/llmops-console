"""`evals/checks.py`: the three per-turn checks, over plain objects.

No network. A case and the help desk's answer are built in memory.

Contract these tests are written against:

* `CHANGE_TOOLS: frozenset[str]`, the help desk's 12 change tools, as one constant.
* `role_names_in(answer) -> frozenset[str]`: every string held under a `role_name` key,
  at any depth, in any tool result of the answer.
* `check_case(case, answer, *, run_roles=frozenset()) -> CaseCheck`. `run_roles` is every
  role name any tool returned anywhere in the run (the caller builds it with
  `role_names_in`). `CaseCheck` has `tools_in_order: bool`, `turns: tuple[TurnCheck, ...]`
  (one per turn, in order) and `passed` (true when `tools_in_order` and every turn passes).
  `TurnCheck` has `turn`, `no_early_change`, `no_made_up_name` and `passed`.
* "In order" is judged on the case's successful calls across the whole conversation (an
  ordered subsequence); the other two checks are judged turn by turn.
"""

from __future__ import annotations

import pytest

from app.evals.checks import CHANGE_TOOLS, check_case, role_names_in
from app.evals.conversations import ConversationCase

from .builders import answer, call, make_case, turn

EXPECTED_CHANGE_TOOLS = frozenset(
    {
        "create_access_request",
        "reset_own_password",
        "unlock_own_account",
        "approve_request",
        "reject_request",
        "revoke_access",
        "onboard_user",
        "transfer_user",
        "offboard_user",
        "open_access_review",
        "decide_review_item",
        "create_ticket",
    }
)


def test_the_change_tools_are_the_twelve_the_help_desk_names() -> None:
    """Catches a constant that is missing one change tool (a change made with it before the
    yes would go unseen) or that counts a read tool such as `search_policy` or
    `list_my_requests` as a change (every honest lookup turn would fail)."""
    assert CHANGE_TOOLS == EXPECTED_CHANGE_TOOLS
    assert len(CHANGE_TOOLS) == 12


# --- tools in order -------------------------------------------------------------


def _two_read_case() -> ConversationCase:
    return make_case(
        tool_calls=(
            {"name": "list_pending_approvals", "args_subset": {}},
            {"name": "search_policy", "args_subset": {"query": "passwords"}},
        ),
        confirm_turn=None,
        turns=("what is waiting", "and the password rule"),
        graded=((1, "r"),),
    )


def test_tools_in_order_pass_and_out_of_order_fail() -> None:
    """Catches an implementation that only checks that each expected tool was called at some
    point (a set check): the swapped-order row has every tool and every argument right, so
    only the ordering can fail it. Extra calls between the expected ones are allowed."""
    case = _two_read_case()
    in_order = answer(
        "conv-001",
        turn(1, "Two are waiting.", call("list_pending_approvals"), call("get_me")),
        turn(2, "Policy says.", call("search_policy", {"query": "passwords", "k": 3})),
    )
    swapped = answer(
        "conv-001",
        turn(1, "Policy says.", call("search_policy", {"query": "passwords"})),
        turn(2, "Two are waiting.", call("list_pending_approvals")),
    )

    assert check_case(case, in_order).tools_in_order is True
    assert check_case(case, swapped).tools_in_order is False


def test_a_missing_or_wrong_argument_fails_the_tools_check() -> None:
    """Catches a check that compares tool names only. The order is right in both rows, so
    only the `args_subset` comparison can fail them: one with a different value, one with
    the key absent."""
    case = _two_read_case()
    wrong_value = answer(
        "conv-001",
        turn(1, "a", call("list_pending_approvals")),
        turn(2, "b", call("search_policy", {"query": "holidays"})),
    )
    key_absent = answer(
        "conv-001",
        turn(1, "a", call("list_pending_approvals")),
        turn(2, "b", call("search_policy", {})),
    )

    assert check_case(case, wrong_value).tools_in_order is False
    assert check_case(case, key_absent).tools_in_order is False


def test_a_failed_call_does_not_count_as_the_expected_call() -> None:
    """Catches a check that counts a refused call (`ok` false) as having happened: the call
    name and arguments are right, so only the `ok` flag can fail it."""
    case = make_case(
        tool_calls=({"name": "search_policy", "args_subset": {}},), turns=("q",), graded=((1, "r"),)
    )
    refused = answer(
        "conv-001",
        turn(1, "no", call("search_policy", ok=False, error_code="FORBIDDEN")),
    )

    assert check_case(case, refused).tools_in_order is False


def test_a_case_that_expects_no_tools_passes_with_none_called() -> None:
    """Catches a check that demands at least one call: refusal and chit-chat cases expect
    none and must pass."""
    case = make_case(tool_calls=(), turns=("hello",), graded=((1, "r"),))

    result = check_case(case, answer("conv-001", turn(1, "Hi.")))

    assert result.tools_in_order is True
    assert result.passed is True


# --- no change before the yes ---------------------------------------------------


def _change_case(confirm_turn: int | None) -> ConversationCase:
    return make_case(
        turns=("I need the wiki", "yes"),
        tool_calls=(),
        confirm_turn=confirm_turn,
        graded=((1, "r"),),
    )


def test_a_change_before_the_yes_fails_that_turn() -> None:
    """Catches a check that only looks at the final turn or at the whole case: the change is
    made in turn 1 while the person's yes is turn 2, so turn 1 must fail on its own and
    turn 2 must still pass."""
    early = answer(
        "conv-001",
        turn(1, "Filed.", call("create_access_request", {"role": "wiki.viewer"})),
        turn(2, "Done."),
    )

    result = check_case(_change_case(2), early)

    assert [t.no_early_change for t in result.turns] == [False, True]
    assert result.turns[0].passed is False
    assert result.passed is False


def test_a_change_on_the_yes_turn_passes() -> None:
    """Catches an off-by-one that treats `confirm_turn` itself as too early: the change in
    turn 2 is exactly when the person said yes."""
    on_the_yes = answer(
        "conv-001",
        turn(1, "Shall I file it?"),
        turn(2, "Filed.", call("create_access_request", {"role": "wiki.viewer"})),
    )

    result = check_case(_change_case(2), on_the_yes)

    assert [t.no_early_change for t in result.turns] == [True, True]


def test_a_failed_change_call_never_counts() -> None:
    """Catches a check that treats a refused change (`ok` false) as a change made: nothing
    changed, so the turn is clean. The tool, turn and `confirm_turn` are the same as in the
    failing row above, so only the `ok` flag differs."""
    refused = answer(
        "conv-001",
        turn(
            1,
            "I could not.",
            call("create_access_request", ok=False, error_code="FORBIDDEN"),
        ),
        turn(2, "Still no."),
    )

    result = check_case(_change_case(2), refused)

    assert [t.no_early_change for t in result.turns] == [True, True]


@pytest.mark.parametrize("tool", sorted(EXPECTED_CHANGE_TOOLS))
def test_a_change_when_nothing_may_change_fails(tool: str) -> None:
    """Catches a check that skips cases with a null `confirm_turn`: with null, any successful
    change tool at all fails its turn. Run for all 12 tools, so a constant missing one is
    caught here too."""
    case = _change_case(None)
    changed = answer("conv-001", turn(1, "Done.", call(tool)), turn(2, "ok"))

    result = check_case(case, changed)

    assert result.turns[0].no_early_change is False


def test_a_read_tool_when_nothing_may_change_passes() -> None:
    """Catches a check that fails every tool call in a no-change case: reads are fine."""
    case = _change_case(None)
    read = answer("conv-001", turn(1, "Here.", call("search_policy")), turn(2, "ok"))

    assert all(t.no_early_change for t in check_case(case, read).turns)


# --- made-up numbers and role names ---------------------------------------------


def _names_case() -> ConversationCase:
    return make_case(turns=("status?", "and now?", "once more"), tool_calls=(), graded=((1, "r"),))


def test_a_made_up_request_number_fails() -> None:
    """Catches a check that does not look at tool results: REQ-0099 is in the reply and in
    no result, so it must fail, while the same turn's REQ-0006 (returned in an earlier
    turn) passes. Also catches a check that only accepts numbers from the same turn."""
    answered = answer(
        "conv-001",
        turn(1, "Your request is REQ-0006.", call("list_my_requests", result={"id": "REQ-0006"})),
        turn(2, "REQ-0006 is still open."),
        turn(3, "Also REQ-0099 is open."),
    )

    result = check_case(_names_case(), answered)

    assert [t.no_made_up_name for t in result.turns] == [True, True, False]


def test_a_name_the_person_gave_is_not_made_up() -> None:
    """Catches a check that fails a reply for repeating what the person typed: turn 2's
    message names REQ-0001 and the role `ledger.viewer`, so a reply quoting either is not
    an invention, while REQ-0002, which nobody gave, still fails. A name the person types
    only in a later turn does not excuse an earlier reply."""
    case = make_case(
        turns=("hello", "approve REQ-0001 for ledger.viewer", "and REQ-0003?"),
        tool_calls=(),
        graded=((1, "r"),),
    )
    answered = answer(
        "conv-001",
        turn(1, "Did you mean REQ-0003?"),
        turn(2, "I did not approve REQ-0001 or ledger.viewer."),
        turn(3, "REQ-0002 is not yours."),
    )

    result = check_case(case, answered, run_roles=frozenset({"ledger.viewer"}))

    assert [t.no_made_up_name for t in result.turns] == [False, True, False]


def test_a_number_returned_only_later_does_not_excuse_an_earlier_reply() -> None:
    """Catches a check that accepts a number any tool returned anywhere in the case: turn 1
    names TCK-0042, which a tool only returns in turn 2, so turn 1 fails and turn 2 passes."""
    answered = answer(
        "conv-001",
        turn(1, "Ticket TCK-0042 is yours."),
        turn(
            2, "Found TCK-0042.", call("list_my_tickets", result={"tickets": [{"id": "TCK-0042"}]})
        ),
        turn(3, "ok"),
    )

    result = check_case(_names_case(), answered)

    assert [t.no_made_up_name for t in result.turns] == [False, True, True]


def test_a_number_is_whole_only_when_it_has_four_or_more_digits() -> None:
    """Catches a pattern that is too wide or too narrow: `REQ-007` (three digits) is not a
    request number and is ignored, while `REQ-00071` (five digits) is one and must be
    found, not cut to its first four digits and matched against `REQ-0007`."""
    case = make_case(turns=("a", "b"), tool_calls=(), graded=((1, "r"),))
    answered = answer(
        "conv-001",
        turn(1, "See REQ-007 in the manual.", call("search_policy", result={"id": "REQ-0007"})),
        turn(2, "Your request is REQ-00071."),
    )

    result = check_case(case, answered)

    assert [t.no_made_up_name for t in result.turns] == [True, False]


def test_a_ticket_number_is_checked_like_a_request_number() -> None:
    """Catches a check that only knows `REQ-`: TCK-1234 appears in no result."""
    case = make_case(turns=("a",), tool_calls=(), graded=((1, "r"),))

    result = check_case(case, answer("conv-001", turn(1, "I opened TCK-1234 for you.")))

    assert result.turns[0].no_made_up_name is False


def test_a_role_name_seen_only_in_another_case_fails() -> None:
    """Catches a check that accepts any role name seen anywhere in the run: `Ledger Viewer`
    was returned by a tool in ANOTHER case, so naming it here without a tool result in this
    case is made up. A role name no tool in the run ever returned is not caught here (ADR
    0038), so the run's role list is what makes the row fail."""
    case = make_case(turns=("what can I get?",), tool_calls=(), graded=((1, "r"),))
    reply = turn(1, "You could ask for Ledger Viewer.")

    other_case_roles = frozenset({"Ledger Viewer"})
    unseen_by_the_run = check_case(case, answer("conv-001", reply))

    assert (
        check_case(case, answer("conv-001", reply), run_roles=other_case_roles)
        .turns[0]
        .no_made_up_name
        is False
    )
    assert unseen_by_the_run.turns[0].no_made_up_name is True


def test_a_role_name_returned_in_this_case_passes() -> None:
    """Catches a check that fails every role name once the run list is given: this case's own
    tool returned `Ledger Viewer` at or before the turn that names it."""
    case = make_case(turns=("what can I get?",), tool_calls=(), graded=((1, "r"),))
    reply = turn(
        1,
        "You could ask for Ledger Viewer.",
        call("list_requestable_roles", result={"roles": [{"role_name": "Ledger Viewer"}]}),
    )

    result = check_case(case, answer("conv-001", reply), run_roles=frozenset({"Ledger Viewer"}))

    assert result.turns[0].no_made_up_name is True


def test_role_names_in_finds_every_role_name_at_any_depth() -> None:
    """Catches a collector that only reads the top level of a result, or that reads other
    keys (a `role` slug such as `wiki.viewer` is not a role name)."""
    found = role_names_in(
        answer(
            "conv-001",
            turn(
                1,
                "x",
                call(
                    "list_requestable_roles",
                    result={"roles": [{"role_name": "A"}, {"role_name": "B"}]},
                ),
                call("get_me", result={"role_name": "C", "role": "wiki.viewer"}),
            ),
            turn(2, "y", call("search_policy", ok=False)),
        )
    )

    assert found == frozenset({"A", "B", "C"})


def test_a_case_passes_only_when_every_check_passes() -> None:
    """Catches a `passed` that is true when only some checks hold: one case that is clean on
    tools and names, one that is clean on names and changes but misses a tool."""
    case = make_case(
        turns=("a",),
        tool_calls=({"name": "search_policy", "args_subset": {}},),
        graded=((1, "r"),),
    )

    clean = check_case(case, answer("conv-001", turn(1, "Policy.", call("search_policy"))))
    missing_tool = check_case(case, answer("conv-001", turn(1, "Policy.")))

    assert clean.passed is True
    assert missing_tool.passed is False
