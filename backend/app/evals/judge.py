"""The reply judge: one fixed prompt, one check per request, one answer each.

The judge answers each reply check on its own with `yes`, `no` or `does_not_apply`, a quote
from the reply and a one-sentence reason. It sees the whole conversation, the tool results
and the world facts. It reaches a model the way every other client does, through the
gateway's `/v1/chat/completions` naming the alias `judge`: the gateway puts the spend on the
`evals` tenant's budget and in the ledger without this module knowing.

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
from collections.abc import Sequence
from importlib import resources
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from app.evals.cases import ReplyCheck
    from app.evals.labels import LabelTurn
    from app.evals.person import PlayedTurn

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
    """
    for _ in range(2):
        try:
            response = client.post(
                url,
                json=body,
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
        try:
            content = response.json()["choices"][0]["message"]["content"]
            parsed = model.model_validate(json.loads(content))
        except (ValueError, TypeError, KeyError, IndexError, ValidationError):
            continue
        return parsed
    return None


def render_conversation(turns: Sequence[PlayedTurn]) -> str:
    """The whole conversation for the judge, between the fixed tags.

    Every turn: what the person typed, every tool call with its result, the reply, and the
    world facts the help desk reported before the first turn and after the last.
    """
    lines = [CONVERSATION_OPEN]
    for number, turn in enumerate(turns, start=1):
        lines.append(f"Turn {number}")
        lines.append(f"Person: {turn.message}")
        lines.extend(
            _tool_line(c.name, c.args, c.ok, c.error_code, c.result) for c in turn.answer.tool_calls
        )
        lines.append(f"Help desk: {turn.answer.reply}")
    lines.append(CONVERSATION_CLOSE)
    if turns:
        first, last = turns[0].answer.world_before, turns[-1].answer.world_after
        lines.append(f"World facts before the first turn: {first.model_dump_json()}")
        lines.append(f"World facts after the last turn: {last.model_dump_json()}")
    return "\n".join(lines)


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
            _tool_line(c.name, c.args, c.ok, c.error_code, c.result) for c in turn.tool_calls
        )
        lines.append(f"Help desk: {turn.reply}")
    lines.append(CONVERSATION_CLOSE)
    return "\n".join(lines)


def _tool_line(name: str, args: object, ok: bool, error_code: str | None, result: object) -> str:
    return (
        f"Tool {name} args={json.dumps(args, ensure_ascii=False)} "
        f"ok={ok} error={error_code} "
        f"result={json.dumps(result, ensure_ascii=False)}"
    )


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

    def _ask(self, question: str, turns: Sequence[PlayedTurn]) -> CheckAnswer | None:
        return self._ask_rendered(question, render_conversation(turns))

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

    def check(self, check: ReplyCheck, turns: Sequence[PlayedTurn]) -> CheckAnswer | None:
        """Answer one reply check, or `None` when the judge could not be used."""
        return self._ask(f"Check: {check.check}\nApplies when: {check.applies_when}", turns)

    def never(self, text: str, turns: Sequence[PlayedTurn]) -> CheckAnswer | None:
        """Answer one never rule: `yes` means the agent did the thing it must never do."""
        return self._ask(f"Never rule: the agent {text}", turns)

    def label(self, question: str, conversation: Sequence[LabelTurn]) -> CheckAnswer | None:
        """Answer a label row's question about its conversation, or `None` when unusable."""
        return self._ask_rendered(question, render_label_conversation(conversation))
