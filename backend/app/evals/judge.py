"""The LLM judge: one fixed prompt, three samples, pass or fail.

The judge grades each `graded` reply of a conversation case against its rubric, and
nothing else. It reaches a model the way every other client does, through the
gateway's `/v1/chat/completions` naming the alias `judge`: the gateway puts the
spend on the `evals` tenant's budget and in the ledger without this module knowing.

Four properties are worth more than the arithmetic.

* **Three samples and the MAJORITY**.
  A single sample of a generative model is a coin toss dressed as a measurement. With
  two classes and three samples there is always a majority, so no tie rule exists.
  The three reasons are kept in the order taken.
* **The rubric and the reply are DATA.** The system message is the prompt file
  verbatim, and both strings travel in the user message under their own labels.
An answer that
  does not validate is a retry and then `None`, never a verdict, and the model's text
  is parsed and never scanned for the word pass.
* **A sample that cannot be used is retried exactly once, then the reply is
  unjudged**. A judge
  outage is a fact about our own gateway, not about the help desk.
* **The prompt is a FILE and its sha256 is the version**.

Nothing here opens a socket of its own: the `httpx.Client` is injected.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any, Final, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

JUDGE_ALIAS: Final[str] = "judge"
"""The alias the judge names, and the only model name this package holds for it.
`routing.yaml` decides which provider serves it."""

JUDGE_PATH: Final[str] = "/v1/chat/completions"
"""Where the judge speaks to the gateway, joined onto `--judge-url`."""

JUDGE_SAMPLES: Final[int] = 3
"""How many times one reply is graded. Three makes a majority of two classes."""


DEFAULT_JUDGE_TIMEOUT_SECONDS: Final[float] = 60.0
"""How long one sample may take; a real number, not httpx's `None`."""

MAX_JUDGE_REASON_LENGTH: Final[int] = 500
"""Characters of each reason that are kept, inclusive. Cut, not refused: a long reason
is not a bad verdict."""

JUDGE_TENANT: Final[str] = "evals"
"""The tenant every judge and label request names in `X-Tenant`. The gateway refuses a
chat request without it, and `evals` is the tenant whose policy lets the judge reach a
US model; the key the caller passes must be this tenant's."""

JUDGE_DATA_CLASS: Final[str] = "internal"
"""The data class the judge declares: seed replies and rubrics, never a person's data."""

JUDGE_SESSION: Final[str] = "eval-judge"
"""The `X-Session` every judge and label request carries. Opaque to the gateway, and the
same for every sample, since the judge holds no conversation of its own."""

PROMPT_PACKAGE: Final[str] = "app.evals.prompts"
"""Where the prompt file lives, read through `importlib.resources`."""

PROMPT_RESOURCE: Final[str] = "judge-v2.md"
"""The prompt file itself."""

RUBRIC_OPEN: Final[str] = "<rubric>"
RUBRIC_CLOSE: Final[str] = "</rubric>"
REPLY_OPEN: Final[str] = "<reply>"
REPLY_CLOSE: Final[str] = "</reply>"
"""The four delimiters the two case-supplied strings travel between."""

Verdict = Literal["pass", "fail"]


class JudgeAnswer(BaseModel):
    """One sample as the model answered it: a reason and a verdict, nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    verdict: Verdict


class JudgeVerdict(BaseModel):
    """What the judge decided about one reply: the majority and the three samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    votes: tuple[Verdict, ...]
    reasons: tuple[str, ...]


class Grader(Protocol):
    """Anything that grades one reply against one rubric. `JudgeClient` is one."""

    def grade(self, rubric: str, reply: str) -> JudgeVerdict | None:
        """The verdict, or `None` when the grader could not be used."""
        ...


def _prompt_bytes() -> bytes:
    return (resources.files(PROMPT_PACKAGE) / PROMPT_RESOURCE).read_bytes()


def judge_prompt_version() -> str:
    """The sha256 hex of the prompt file's bytes.

    The file name is not the version: an edit to the wording with no rename changes
    the hash, and every score after it is no longer comparable to the ones before.
    """
    return hashlib.sha256(_prompt_bytes()).hexdigest()


def system_prompt() -> str:
    """The prompt file's text, which is the system message verbatim."""
    return _prompt_bytes().decode("utf-8")


def user_message(rubric: str, reply: str) -> str:
    """The rubric and the reply, each under its own label, the rubric first."""
    return (
        f"{RUBRIC_OPEN}\n{rubric}\n{RUBRIC_CLOSE}\n\n{REPLY_OPEN}\n{reply}\n{REPLY_CLOSE}\n\n"
        "Grade the reply in the second block against the rubric in the first."
    )


def answer_from(document: object) -> JudgeAnswer | None:
    """One chat answer turned into a sample, or `None` when it is unusable.

    An envelope with no choice, a `content` that is not JSON and a payload the schema
    refuses all end here. The reason is cut at `MAX_JUDGE_REASON_LENGTH`.
    """
    try:
        content = document["choices"][0]["message"]["content"]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return None
    if not isinstance(content, str):
        return None
    try:
        answer = JudgeAnswer.model_validate(json.loads(content))
    except (ValueError, ValidationError):
        return None
    return JudgeAnswer(reason=answer.reason[:MAX_JUDGE_REASON_LENGTH], verdict=answer.verdict)


def chat_body(alias: str, system: str, user: str) -> dict[str, Any]:
    """One request body: the alias, a system and a user message.

    No temperature: the judge and label models (Haiku 5.5, Sonnet 5.5) refuse any value
    but their default, so the samples differ as the model's own sampling makes them.
    """
    return {
        "model": alias,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def sample_once_retried(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    *,
    api_key: str,
    timeout: float,
) -> JudgeAnswer | None:
    """One sample, retried EXACTLY once when the first attempt is unusable.

    Once and never twice: giving up on the first bad answer leaves replies unjudged on
    a model's ordinary flakiness, and retrying until it works is unbounded spend on a
    failing gateway. A non-2xx, a body that is not JSON and a transport failure are the
    same event here; nothing raises out of this function.
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
            document = response.json()
        except ValueError:
            continue
        answer = answer_from(document)
        if answer is not None:
            return answer
    return None


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
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    def grade(self, rubric: str, reply: str) -> JudgeVerdict | None:
        """Grade one reply against one rubric, or `None` when the judge could not be used.

        A sample that failed twice ends the reply there and no further request is
        made: a verdict built from two samples is not a majority of three.
        """
        url = f"{self._base_url}{JUDGE_PATH}"
        body = chat_body(JUDGE_ALIAS, system_prompt(), user_message(rubric, reply))
        answers: list[JudgeAnswer] = []
        for _ in range(JUDGE_SAMPLES):
            answer = sample_once_retried(
                self._client, url, body, api_key=self._api_key, timeout=self._timeout
            )
            if answer is None:
                return None
            answers.append(answer)
        passes = sum(1 for answer in answers if answer.verdict == "pass")
        return JudgeVerdict(
            verdict="pass" if passes * 2 > len(answers) else "fail",
            votes=tuple(answer.verdict for answer in answers),
            reasons=tuple(answer.reason for answer in answers),
        )
