"""The reply judge: one fixed prompt, one check per request, one answer each.

The judge answers each reply check on its own with `yes`, `no` or `does_not_apply`, a quote
from the reply and a one-sentence reason. It sees the whole conversation, the tool results,
the world facts about the people the case names and every row that changed during the play;
a long role list is cut to the roles the reply names. It reaches a
model the way every other client does, through the gateway's `/v1/chat/completions`
naming the alias `judge`: the gateway puts the spend on the `evals` tenant's budget and in
the ledger without this module knowing.

* **One check, one request, one sample.** Checks are never batched: one answer would colour
  the next.
* **The conversation is DATA.** The system message is the prompt file verbatim; the
  conversation travels in the user message between fixed tags. An answer that does not
  validate is a retry and then `None`, never a verdict.
* **A request that cannot be used is retried exactly once**, then the check is unanswered.
  A judge outage is a fact about our own gateway, not about the help desk, so an
  unanswered check is left out of the scores.
* **The prompt is a FILE and its sha256 is the version.**

Nothing here opens a socket of its own: the `httpx.Client` is injected.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from importlib import resources
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from app.evals.cases import ReplyCheck
    from app.evals.labels import LabelTurn
    from app.evals.person import PlayedTurn
    from app.evals.target import EvalWorld

JUDGE_ALIAS: Final[str] = "judge"
"""The alias the judge names. `routing.yaml` decides which provider serves it."""

CHAT_PATH: Final[str] = "/v1/chat/completions"
"""Where the judge and the simulated person speak to the gateway."""

DEFAULT_JUDGE_TIMEOUT_SECONDS: Final[float] = 60.0
"""How long one request may take; a real number, not httpx's `None`."""

MAX_REASON_LENGTH: Final[int] = 500
"""Characters of each reason and quote that are kept, inclusive. Cut, not refused."""

JUDGE_TENANT: Final[str] = "evals"
"""The tenant every judge and person request names in `X-Tenant`. The gateway refuses a
chat request without it; the key the caller passes must be this tenant's."""

JUDGE_DATA_CLASS: Final[str] = "internal"
"""The data class declared: seed replies, never a person's data."""

JUDGE_SESSION: Final[str] = "eval-judge"
"""The `X-Session` every request carries. Opaque to the gateway."""

PROMPT_PACKAGE: Final[str] = "app.evals.prompts"
PROMPT_RESOURCE: Final[str] = "judge-v3.md"

CONVERSATION_OPEN: Final[str] = "<conversation>"
CONVERSATION_CLOSE: Final[str] = "</conversation>"

MAX_SHOWN_UNREADABLE: Final[int] = 4000
"""Characters of an unreadable answer printed to stderr, so a huge one cannot flood the log."""

MAX_WHOLE_ROLE_LIST: Final[int] = 10
"""A role list this long or shorter is sent to the judge whole; a longer one is cut."""

Answer = Literal["yes", "no", "does_not_apply"]


class CheckAnswer(BaseModel):
    """One answer as the model gave it: a word, a quote and a reason, nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: Answer
    quote: str
    reason: str


def prompt_text(resource: str = PROMPT_RESOURCE) -> str:
    """A prompt file's text, which is the system message verbatim."""
    return (resources.files(PROMPT_PACKAGE) / resource).read_text(encoding="utf-8")


def prompt_version(resource: str = PROMPT_RESOURCE) -> str:
    """The sha256 hex of a prompt file's bytes.

    The file name is not the version: an edit to the wording with no rename changes the
    hash, and every score after it is no longer comparable to the ones before.
    """
    return hashlib.sha256((resources.files(PROMPT_PACKAGE) / resource).read_bytes()).hexdigest()


def chat_body(alias: str, system: str, user: str) -> dict[str, Any]:
    """One request body: the alias, a system and a user message.

    No temperature: the judge and person models refuse any value but their default.
    """
    return {
        "model": alias,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def ask_retried[Parsed: BaseModel](
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    *,
    api_key: str,
    timeout: float,
    model: type[Parsed],
) -> Parsed | None:
    """One request, retried EXACTLY once when the first attempt is unusable.

    Once and never twice: giving up on the first bad answer leaves checks unanswered on a
    model's ordinary flakiness, and retrying until it works is unbounded spend on a
    failing gateway. A non-2xx, a body that is not JSON, an answer the model class refuses
    and a transport failure are the same event here; nothing raises out of this function.

    The request names `model`'s own JSON schema as a strict `response_format`, so the
    provider holds the answer to that shape: written by hand, a quote holding double
    quotes broke the JSON. An answer that still cannot be read is printed to stderr,
    so a stopped run shows why.
    """
    shaped = {
        **body,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": model.model_json_schema(),
                "strict": True,
            },
        },
    }
    for _ in range(2):
        try:
            response = client.post(
                url,
                json=shaped,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-Tenant": JUDGE_TENANT,
                    "X-Data-Class": JUDGE_DATA_CLASS,
                    "X-Session": JUDGE_SESSION,
                },
                timeout=timeout,
            )
        except httpx.HTTPError:
            continue
        if not response.is_success:
            continue
        content: object = response.text
        try:
            content = response.json()["choices"][0]["message"]["content"]
            parsed = model.model_validate(json.loads(str(content)))
        except (ValueError, TypeError, KeyError, IndexError, ValidationError) as error:
            shown = str(content)[:MAX_SHOWN_UNREADABLE]
            print(f"unreadable {model.__name__} answer ({error!r}): {shown}", file=sys.stderr)
            continue
        return parsed
    return None


def render_conversation(turns: Sequence[PlayedTurn], people: frozenset[str] | None = None) -> str:
    """The whole conversation for the judge, between the fixed tags.

    Every turn: what the person typed, every tool call with its result, the reply. Then the
    world facts about `people` (the case's own person and anyone its facts name) before the
    first turn, and every row that changed by the last. The rest of the company is left
    out: a seeded tenant has about 200 people, and sending them all made each check cost
    about 100,000 tokens while no check needs them. `None` shows everyone, for a caller
    with a small world and no case at hand; a run always names its people.
    """
    lines = [CONVERSATION_OPEN]
    for number, turn in enumerate(turns, start=1):
        lines.append(f"Turn {number}")
        lines.append(f"Person: {turn.message}")
        lines.extend(
            _tool_line(c.name, c.args, c.ok, c.error_code, c.result, turn.answer.reply)
            for c in turn.answer.tool_calls
        )
        lines.append(f"Help desk: {turn.answer.reply}")
    lines.append(CONVERSATION_CLOSE)
    if turns:
        first, last = turns[0].answer.world_before, turns[-1].answer.world_after
        named = "everyone" if people is None else ", ".join(sorted(people)) or "nobody"
        lines.append(f"World facts before the first turn, about {named}: {_about(first, people)}")
        lines.append(f"Rows that changed by the last turn: {_changes(first, last)}")
    return "\n".join(lines)


def _about(world: EvalWorld, people: frozenset[str] | None) -> str:
    """The rows of `world` that name one of `people` (every row for `None`), as JSON."""
    if people is None:
        return world.model_dump_json(exclude={"reviews"})
    return json.dumps(
        {
            "requests": [
                r.model_dump(mode="json")
                for r in world.requests
                if r.requester in people or r.target in people
            ],
            "grants": [g.model_dump(mode="json") for g in world.grants if g.person in people],
            "tickets": [t.model_dump(mode="json") for t in world.tickets if t.person in people],
            "people": [p.model_dump(mode="json") for p in world.people if p.person in people],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _changes(before: EvalWorld, after: EvalWorld) -> str:
    """Per kind of row, the rows only in `after` (new or changed) and only in `before`
    (gone or changed), as JSON. A kind with no change is left out."""
    shown: dict[str, dict[str, list[Any]]] = {}
    for kind in ("requests", "grants", "tickets", "people", "reviews"):
        old = [row.model_dump_json() for row in getattr(before, kind)]
        new = [row.model_dump_json() for row in getattr(after, kind)]
        old_set, new_set = set(old), set(new)
        added = [json.loads(row) for row in new if row not in old_set]
        gone = [json.loads(row) for row in old if row not in new_set]
        if added or gone:
            shown[kind] = {"now": added, "before": gone}
    return json.dumps(shown, ensure_ascii=False, separators=(",", ":")) if shown else "none"


def render_label_conversation(turns: Sequence[LabelTurn]) -> str:
    """A labelled conversation for the judge, in the same tags and lines a run uses.

    A label row carries no world facts, so none are shown; the turns read exactly as
    `render_conversation` writes them, so a calibration tests the prompt a run uses.
    """
    lines = [CONVERSATION_OPEN]
    for number, turn in enumerate(turns, start=1):
        lines.append(f"Turn {number}")
        lines.append(f"Person: {turn.person}")
        lines.extend(
            _tool_line(c.name, c.args, c.ok, c.error_code, c.result, turn.reply)
            for c in turn.tool_calls
        )
        lines.append(f"Help desk: {turn.reply}")
    lines.append(CONVERSATION_CLOSE)
    return "\n".join(lines)


def _is_role_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > MAX_WHOLE_ROLE_LIST
        and all(isinstance(i, dict) and "system_slug" in i and "role_slug" in i for i in value)
    )


def _names_role(item: dict[str, Any], reply: str) -> bool:
    said = reply.lower()
    pair = f"{item['system_slug']}.{item['role_slug']}".lower()
    name = item.get("role_name")
    return pair in said or (isinstance(name, str) and bool(name) and name.lower() in said)


def _cut_roles(result: object, reply: str) -> tuple[object, list[str]]:
    """The result with each long role list cut to the roles the reply names, and one note
    per cut list.

    A tool can list 200 roles, and every check would send them all again; only the roles
    the reply talks about matter to a check. A fixed cap would be shorter but could drop
    the very role the reply names.
    """
    if not isinstance(result, dict):
        return result, []
    cut: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in result.items():
        if _is_role_list(value):
            kept = [item for item in value if _names_role(item, reply)]
            notes.append(f"(kept {len(kept)} of {len(value)} roles: the ones the reply names)")
            cut[key] = kept
        else:
            cut[key] = value
    return cut, notes


def _tool_line(
    name: str, args: object, ok: bool, error_code: str | None, result: object, reply: str
) -> str:
    shown, notes = _cut_roles(result, reply) if ok else (result, [])
    line = (
        f"Tool {name} args={json.dumps(args, ensure_ascii=False)} "
        f"ok={ok} error={error_code} "
        f"result={json.dumps(shown, ensure_ascii=False)}"
    )
    return " ".join([line, *notes])


class JudgeClient:
    """The `judge` alias, addressed by the base URL of a gateway that serves it."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client,
        api_key: str,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}{CHAT_PATH}"
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    def _ask(
        self, question: str, turns: Sequence[PlayedTurn], people: frozenset[str] | None
    ) -> CheckAnswer | None:
        return self._ask_rendered(question, render_conversation(turns, people))

    def _ask_rendered(self, question: str, conversation: str) -> CheckAnswer | None:
        user = f"{question}\n\n{conversation}"
        body = chat_body(JUDGE_ALIAS, prompt_text(), user)
        answer = ask_retried(
            self._client,
            self._url,
            body,
            api_key=self._api_key,
            timeout=self._timeout,
            model=CheckAnswer,
        )
        if answer is None:
            return None
        return CheckAnswer(
            answer=answer.answer,
            quote=answer.quote[:MAX_REASON_LENGTH],
            reason=answer.reason[:MAX_REASON_LENGTH],
        )

    def check(
        self,
        check: ReplyCheck,
        turns: Sequence[PlayedTurn],
        *,
        people: frozenset[str] | None = None,
    ) -> CheckAnswer | None:
        """Answer one reply check, or `None` when the judge could not be used."""
        question = f"Check: {check.check}\nApplies when: {check.applies_when}"
        return self._ask(question, turns, people)

    def never(
        self, text: str, turns: Sequence[PlayedTurn], *, people: frozenset[str] | None = None
    ) -> CheckAnswer | None:
        """Answer one never rule: `yes` means the agent did the thing it must never do."""
        return self._ask(f"Never rule: the agent {text}", turns, people)

    def label(self, question: str, conversation: Sequence[LabelTurn]) -> CheckAnswer | None:
        """Answer a label row's question about its conversation, or `None` when unusable."""
        return self._ask_rendered(question, render_label_conversation(conversation))
