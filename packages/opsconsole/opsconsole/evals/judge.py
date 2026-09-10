"""The LLM judge: one fixed prompt, three samples at temperature zero, the median (#35).

The judge scores the cases a golden set wrote a `judge_rubric` for, and nothing
else - a case with no rubric is never sent and costs nothing
(`test_a_case_with_no_rubric_is_not_judged_at_all`). It reaches a model the way
every other client does, through the gateway's `/v1/chat/completions` naming the
alias `judge` (decision A): the design locks aliases as the only thing a client
names, and going through the gateway puts the judge's spend on the `evals`
tenant's budget, its trace in Langfuse and its cost in the ledger without this
module knowing any of those exist.

Four properties are worth more than the arithmetic.

* **Three samples and the MEDIAN, never the mean and never the first**
  (`test_three_samples_are_taken_and_the_median_is_the_verdict`). A single sample
  of a generative model is a coin toss dressed as a measurement; the mean of
  three is not a score the rubric defines, and the median is one of the three
  answers a model actually gave.
* **The rubric and the reply are DATA.** The system message is the prompt file
  verbatim and nothing is interpolated into it; both case-supplied strings travel
  in a user message under their own labels
  (`test_the_rubric_and_the_reply_are_sent_as_delimited_data`), and an answer
  that does not schema-validate is a retry and then a `None`, never a score - so
  a reply carrying its own `{"score": 5}` is graded rather than obeyed
  (`test_a_reply_that_tries_to_instruct_the_judge_is_scored_not_obeyed`).
* **A sample that cannot be used is retried exactly once, then the case is
  unjudged** (decision E). Not a failed case: a judge outage is a fact about our
  own gateway, and recording it as the target's `provider_error` would make the
  taxonomy lie about which system broke. The count is four requests for three
  samples with one bad answer, and three for a sample abandoned
  (`test_an_unusable_sample_is_retried_exactly_once`,
  `test_a_sample_that_fails_twice_leaves_the_case_unjudged`).
* **The prompt is a FILE and its sha256 is recorded on the run** (decision H,
  `test_the_prompt_version_is_the_sha256_of_the_prompt_file`). The change a
  stored version exists to catch is an edit to the wording with no rename, which
  a file name and a module constant both survive unchanged.

Nothing here opens a socket of its own: the `httpx.Client` is injected, so the
whole contract is pinned against `httpx.MockTransport` with no gateway running,
exactly as `evals/target.py` is.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from opsconsole.evals.cases import EvalCase
from opsconsole.evals.outcome import CaseOutcome

JUDGE_ALIAS: Final[str] = "judge"
"""The alias the judge names, and the only model name this package contains.

`routing.yaml` already defines it with two candidates, so which provider serves a
judged run is a config change and never a code one. Pinned by
`test_the_judge_constants_are_the_published_ones`."""

JUDGE_PATH: Final[str] = "/v1/chat/completions"
"""Where the judge speaks to the gateway. The same OpenAI-compatible path every
other client uses, joined onto `--judge-url`
(`test_the_request_names_the_alias_and_temperature_zero` asserts the whole URL)."""

JUDGE_SAMPLES: Final[int] = 3
"""How many times one case is scored. Three is what makes a median a median, and
a fourth would be a quarter more spend on every rubric case for no tie-break."""

JUDGE_TEMPERATURE: Final[float] = 0.0
"""The temperature every sample is taken at. Zero because a rerun of the same set
on the same commit has to be comparable to the last one; the three samples exist
to catch the residual non-determinism a provider keeps at zero, not to sample a
distribution."""

JUDGE_MIN_SCORE: Final[int] = 1
"""The lowest score the rubric defines, inclusive."""

JUDGE_MAX_SCORE: Final[int] = 5
"""The highest, inclusive. Both bounds are on `JudgeAnswer.score` itself, so an
answer of `0` or `6` is an unusable sample rather than a number that reaches a
metric (`test_a_score_of_zero_and_a_score_of_six_are_refused`)."""

JUDGE_LOW_SCORE: Final[int] = 3
"""The lowest median that does NOT fail a case (decision D): 1 and 2 fail, 3, 4
and 5 do not. Read with `<` and never `<=`, pinned at the boundary by
`test_a_score_of_three_does_not_fail_the_case` in `tests/evals/test_runner.py`
and by `test_a_low_judge_score_is_tagged_judge_low` in `tests/evals/test_taxonomy.py`."""

DEFAULT_JUDGE_TIMEOUT_SECONDS: Final[float] = 60.0
"""How long one sample may take. A real default rather than httpx's `None`, which
means no timeout at all - the same reason `target.DEFAULT_CASE_TIMEOUT_SECONDS`
is one, and the only thing between a hung gateway and a CI job that never ends."""

MAX_JUDGE_REASON_LENGTH: Final[int] = 500
"""Characters of the median sample's reason that are kept, inclusive.

a project rule on this module's one unsized input: the reason is stored in
`eval_cases.details`, which is `jsonb` with no cap of its own, and a model can
answer a page of prose. Truncated rather than refused - a long reason is not a
bad score (`test_the_reason_is_capped_at_its_published_length`)."""

PROMPT_PACKAGE: Final[str] = "opsconsole.evals.prompts"
"""Where the prompt file lives, read through `importlib.resources` so a wheel
install finds it the same way a checkout does."""

PROMPT_RESOURCE: Final[str] = "judge-v1.md"
"""The prompt file itself. Versioned in its NAME for a reader and by its sha256
for a stored run (decision H)."""

RUBRIC_OPEN: Final[str] = "<rubric>"
RUBRIC_CLOSE: Final[str] = "</rubric>"
REPLY_OPEN: Final[str] = "<reply>"
REPLY_CLOSE: Final[str] = "</reply>"
"""The four labelled delimiters the two case-supplied strings travel between.

Labels rather than a bare concatenation: a model handed `rubric + reply` has no
way to tell the standard from the text being judged, and the standard is the half
that must not be scored (`test_the_rubric_and_the_reply_are_sent_as_delimited_data`)."""


class JudgeAnswer(BaseModel):
    """One sample as the model answered it: a score in range and a reason for it.

    `extra="forbid"` and the two bounds are the whole schema check (criterion 3).
    An answer carrying a third field, a missing reason or a score of `0` is an
    unusable sample, which is a retry and then an unjudged case - never a number
    that reaches `judge_mean` or fails a case
    (`test_a_sample_that_fails_twice_leaves_the_case_unjudged` walks the table).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: int = Field(ge=JUDGE_MIN_SCORE, le=JUDGE_MAX_SCORE)
    reason: str


class JudgeVerdict(BaseModel):
    """What the judge decided about one case: the median, its three samples, its reason.

    The samples are carried in the order they were TAKEN rather than sorted, so a
    rerun's report diffs against the last one and a reader can tell a confident
    judgement from three answers that disagreed
    (`test_three_samples_are_taken_and_the_median_is_the_verdict`). The reason is
    the median sample's own, capped at `MAX_JUDGE_REASON_LENGTH`.

    Every field is JSON-native, which is what lets `16d` put this record into the
    `details` `jsonb` column with no adapter of its own
    (`test_the_details_carry_the_judge_verdict` in `tests/evals/test_store.py`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    median: int
    samples: tuple[int, ...]
    reason: str


def _prompt_bytes() -> bytes:
    """The prompt file's bytes, exactly as they sit on disk."""
    return (resources.files(PROMPT_PACKAGE) / PROMPT_RESOURCE).read_bytes()


_PROMPT_VERSION: str | None = None
"""The digest, memoised: the file cannot change under a running process, and
`judge_prompt_version()` is called once per rubric case otherwise."""


def judge_prompt_version() -> str:
    """The sha256 hex of the prompt file's bytes, computed once per process.

    Decision H. The file name is not the version: an edit to the wording with no
    rename is exactly the change that makes every score after it incomparable to
    the ones before, and a hash over the file's BYTES is the only thing that sees
    it (`test_the_prompt_version_is_the_sha256_of_the_prompt_file`).
    """
    global _PROMPT_VERSION
    if _PROMPT_VERSION is None:
        _PROMPT_VERSION = hashlib.sha256(_prompt_bytes()).hexdigest()
    return _PROMPT_VERSION


def _system_prompt() -> str:
    """The prompt file's text, which is the system message verbatim.

    Nothing is formatted into it (criterion 5): the test asserts the message is
    byte-equal to the file, so any interpolation at all is visible
    (`test_the_rubric_and_the_reply_are_sent_as_delimited_data`).
    """
    return _prompt_bytes().decode("utf-8")


def _user_message(rubric: str, reply: str) -> str:
    """The rubric and the reply, each under its own label, as data.

    The rubric comes first because it is the standard the second block is read
    against, and both are delimited so no case text can present itself as an
    instruction that changes the shape the schema then refuses.
    """
    return (
        f"{RUBRIC_OPEN}\n{rubric}\n{RUBRIC_CLOSE}\n\n{REPLY_OPEN}\n{reply}\n{REPLY_CLOSE}\n\n"
        "Grade the reply in the second block against the rubric in the first."
    )


def _answer_from(document: object) -> JudgeAnswer | None:
    """One chat answer turned into a scored sample, or `None` when it is unusable.

    Every way an otherwise-valid 200 can still be unusable arrives here: an
    envelope with no choice, a `content` that is not JSON, a payload the schema
    refuses and a score out of range. The model's text is parsed and NEVER
    scanned for a digit - a reply that reads "I would rate this a 5 out of 5"
    would otherwise become a fabricated verdict
    (`test_a_judge_that_answers_prose_leaves_the_case_unjudged`).
    """
    try:
        choices = document["choices"]  # type: ignore[index]
        content = choices[0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        return None
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    try:
        return JudgeAnswer.model_validate(payload)
    except ValidationError:
        return None


class JudgeClient:
    """The `judge` alias, addressed by the base URL of a gateway that serves it.

    The `httpx.Client` is injected for the reason `target.TargetClient`'s is:
    the whole request contract is pinned against `httpx.MockTransport` with no
    gateway up and no key, and the URL is built from this module's own path
    constant as an absolute URL, so a client quietly carrying its own `base_url`
    cannot pass while sending somewhere else
    (`test_the_request_names_the_alias_and_temperature_zero`).
    """

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

    def _url(self) -> str:
        """The absolute URL of the gateway's chat endpoint on this judge."""
        return f"{self._base_url}{JUDGE_PATH}"

    def _body(self, rubric: str, reply: str) -> dict[str, Any]:
        """One sample's request body: the alias, temperature zero, two messages."""
        return {
            "model": JUDGE_ALIAS,
            "temperature": JUDGE_TEMPERATURE,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_message(rubric, reply)},
            ],
        }

    def _sample(self, rubric: str, reply: str) -> JudgeAnswer | None:
        """One attempt at one sample, or `None` for every way it can be unusable.

        A non-2xx, a body that is not JSON and a transport failure are the same
        event here - the sample cannot be used - and the caller decides what a
        run of them means (decision E). Nothing raises out of this module: a
        judge outage may not abort a run whose deterministic assertions have
        already been paid for
        (`test_a_judge_that_answers_five_hundred_leaves_the_case_unjudged`,
        `test_a_judge_that_times_out_leaves_the_case_unjudged`).
        """
        try:
            response = self._client.post(
                self._url(),
                json=self._body(rubric, reply),
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.HTTPError:
            return None
        if not response.is_success:
            return None
        try:
            document = response.json()
        except ValueError:
            return None
        return _answer_from(document)

    def _sample_once_retried(self, rubric: str, reply: str) -> JudgeAnswer | None:
        """One sample, retried EXACTLY once when the first attempt is unusable.

        Once and never twice (criterion 4): giving up on the first bad answer
        would leave a case unjudged on the ordinary flakiness of a model, and
        retrying until it succeeds is unbounded spend on a gateway that is
        failing (`test_an_unusable_sample_is_retried_exactly_once`).
        """
        answer = self._sample(rubric, reply)
        if answer is not None:
            return answer
        return self._sample(rubric, reply)

    def score_reply(self, rubric: str, reply: str) -> JudgeVerdict | None:
        """Score one reply against one rubric, or `None` when the judge could not be used.

        The whole sampling rule, and the seam 17e's calibration reaches it through
        (17e decision E): `calibrate` has no `EvalCase` and no `CaseOutcome`, and
        a second copy of the retry rule and the median would drift from this one
        (`test_calibrate_reaches_the_judge_through_score_reply_with_the_labels_own_strings`
        in `tests/evals/test_calibration.py`).

        Sequential samples on the caller's own client (decision G): `run_set` is
        synchronous, and three calls on the cases that carry a rubric is not
        worth an async island inside a sync loop.

        A sample that failed twice ends the case there and no further request is
        made - a verdict built from two samples is not the median of three
        (`test_a_sample_that_fails_twice_leaves_the_case_unjudged` counts the
        requests).
        """
        answers: list[JudgeAnswer] = []
        for _ in range(JUDGE_SAMPLES):
            answer = self._sample_once_retried(rubric, reply)
            if answer is None:
                return None
            answers.append(answer)

        median = sorted(answers, key=lambda answer: answer.score)[JUDGE_SAMPLES // 2]
        return JudgeVerdict(
            median=median.score,
            samples=tuple(answer.score for answer in answers),
            reason=median.reason[:MAX_JUDGE_REASON_LENGTH],
        )

    def score(self, case: EvalCase, outcome: CaseOutcome) -> JudgeVerdict | None:
        """Score one rubric case, or answer `None` when the judge could not be used.

        Two lines over `score_reply` and nothing else (17e decision E), which is
        what keeps every 17c case true while the calibration path never learns
        the shape of these two records
        (`test_score_still_delegates_to_score_reply` in
        `tests/evals/test_calibration.py`). Callers hand in a case that HAS a
        rubric; one without is not judged and never reaches here
        (`runner.run_set`).
        """
        if case.judge_rubric is None:
            return None
        return self.score_reply(case.judge_rubric, outcome.reply)
