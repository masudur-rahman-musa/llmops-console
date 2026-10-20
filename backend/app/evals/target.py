"""The target contract: the reset, the turn, and the envelope a target answers.

This is the first module in the package that opens a socket. Everything it sends
and everything it accepts is published in `docs/eval-target.md`, whose worked
example is `tests/evals/fixtures/envelope.json` byte for byte, so an adopter can
make their own workload evaluable without reading this file

Two endpoints, both POST:

    {target}/admin/test/reset   once, before the first case
    {target}/v1/eval/turn       once per case, one attempt

A target answers a turn with an ENVELOPE rather than a chat completion. Reading
an intent, a decision and a risk class back out of prose would put the harness's
own extraction accuracy inside every number it reports, and the Helpdesk has
authored no reply shape yet, so the contract is published here and the first
target that disagrees names the field.

`EvalEnvelope` forbids extras at every level. A target that renames `decision`
would otherwise load cleanly, every assertion would report `skipped`, and the run
would score higher than a target that answered nothing at all.

Two error types, because the two failures are not the same event. A reset that
did not happen aborts the whole run: a run over an unreset seed is not comparable
to the baseline, and comparability is the only thing the number is for. A turn
that failed is ONE case, recorded as failed, one attempt, no retry - a retried
case is a second sample of a non-deterministic system and would quietly lift the
score.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.evals.cases import EvalCase
from app.evals.outcome import CaseOutcome, ObservedToolCall

RESET_PATH: Final[str] = "/admin/test/reset"
"""Where the seed reset lives on a target. A rename here
404s every documented target while this code still looks right."""

TURN_PATH: Final[str] = "/v1/eval/turn"
"""Where one case's turn is sent. Pinned by the same test, and published in
`docs/eval-target.md`."""

PROMPT_VERSIONS_HEADER: Final[str] = "X-Prompt-Versions"
"""Where a target names the prompt each of its steps ran on:
`intent=<sha256>,plan=<sha256>,...`. A header and not an envelope field, because
the envelope refuses extras. Published in `docs/eval-target.md`."""

PROMPT_VERSION_ENTRY: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_]{0,31}=[0-9a-f]{64}")
"""One `step=sha256` entry: a lowercase step name of 1 to 32 characters and a
64-character lowercase hex digest. Matched with `fullmatch`, so a trailing
newline cannot slip past a `$`."""

MAX_PROMPT_STEPS: Final[int] = 16
"""More steps than any agent has; a longer header is a broken target, not data."""

DEFAULT_CASE_TIMEOUT_SECONDS: Final[float] = 60.0
"""How long one turn may take before it is a failed case.

The only thing standing between a hung target and a CI job that never ends, so it
is a real default rather than httpx's `None` (which means no timeout at all)."""


class TargetUnreachable(Exception):
    """The seed could not be reset, so the run may not start.

    Carries the reset URL, because "reset failed" against the wrong host is
    unactionable.
    A plain exception rather than a `GatewayError`: this is an operator's error,
    reported to whoever ran the harness, and it never becomes an answer on a wire.
    """


class TurnFailureKind(StrEnum):
    """Why a turn failed, in the two words the taxonomy tags it by.

    Two and no third: `timeout` is the clock running out on a target that never
    answered, and `provider_error` is everything else - a non-2xx, a body that is
    not JSON, an envelope the contract does not define, a refused connection. A
    third value would silently drop out of the taxonomy, which maps exactly these
    two.
    """

    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"


class TurnFailed(Exception):
    """One case's turn did not produce a usable envelope.

    Every way a turn can fail - a non-2xx, a body that is not JSON, an envelope
    field the contract does not define, a timeout, a refused connection - arrives
    as this one type, so the runner catches one thing to record one failed case
    and a flaky case cannot abort the run. The message names the case id: with two
    hundred cases in a set, "read timeout" names nothing an author can open.

    `kind` is set at the raise site rather than read back out of the message: the
    message is a human sentence naming a case id and a URL, and a taxonomy
    deciding a stored tag by searching it for "timed out" would depend on prose
    that is edited for readability. Each of the five raise sites in `run_case`
    carries the kind its own cause determines.
    """

    def __init__(self, message: str, *, kind: TurnFailureKind) -> None:
        super().__init__(message)
        self.kind = kind


class EnvelopeToolCall(BaseModel):
    """One tool call as the target reports it, arguments and all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args: dict[str, Any] = {}


class EvalEnvelope(BaseModel):
    """What a target answers one turn with. Every field is optional.

    Frozen, because the runner hands the same instance to the residency check and
    to the `CaseOutcome` mapping: a step that normalised `region` in place would
    make the recorded verdict depend on the order those two ran in.

    The defaults read as "nothing observed" and never as an observation - a target
    that answered `{}` is not a refusal with sources. The four measured fields
    are asserted by nothing here: `region` is what the residency check reads, and
    the report module writes the other three.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: str | None = None
    tool_calls: tuple[EnvelopeToolCall, ...] = ()
    decision: str | None = None
    risk_class: str | None = None
    refused: bool = False
    clarified: bool = False
    citations: tuple[str, ...] = ()
    reply: str = ""
    latency_ms: int | None = None
    cost_usd: Decimal | None = None
    provider: str | None = None
    region: str | None = None


def parse_prompt_versions(value: str | None) -> dict[str, str]:
    """The header's step-to-sha256 map, or `ValueError` naming what is wrong.

    No header is an empty map, so a target that predates the header still runs.
    A header that is there but malformed is refused whole: a half-read map would
    put a hash in a report that names no prompt
    (`test_a_malformed_prompt_versions_header_fails_the_turn`).
    """
    if value is None:
        return {}
    entries = value.split(",")
    if len(entries) > MAX_PROMPT_STEPS:
        raise ValueError(f"{len(entries)} steps, more than {MAX_PROMPT_STEPS}")
    versions: dict[str, str] = {}
    for entry in entries:
        if PROMPT_VERSION_ENTRY.fullmatch(entry) is None:
            raise ValueError(f"{entry!r} is not step=<64 lowercase hex>")
        step, digest = entry.split("=", 1)
        if step in versions:
            raise ValueError(f"step {step!r} is named twice")
        versions[step] = digest
    return versions


def to_outcome(
    envelope: EvalEnvelope,
    *,
    other_tenant_markers: tuple[str, ...] = (),
    step_prompt_versions: dict[str, str] | None = None,
) -> CaseOutcome:
    """Map one envelope onto the record `assert_case` reads.

    The markers come from the caller and never from the envelope: they are the
    seed's, not the target's, and a mapping that filled the outcome from the
    envelope alone would leave every `cross_tenant` case's leak assertion
    reporting `skipped`.
    """
    return CaseOutcome(
        intent=envelope.intent,
        tool_calls=tuple(
            ObservedToolCall(name=call.name, args=call.args) for call in envelope.tool_calls
        ),
        decision=envelope.decision,
        risk_class=envelope.risk_class,
        refused=envelope.refused,
        clarified=envelope.clarified,
        citations=envelope.citations,
        reply=envelope.reply,
        other_tenant_markers=other_tenant_markers,
        latency_ms=envelope.latency_ms,
        cost_usd=envelope.cost_usd,
        provider=envelope.provider,
        region=envelope.region,
        step_prompt_versions=step_prompt_versions or {},
    )


class TargetClient:
    """One OpenAI-compatible workload under evaluation, addressed by base URL.

    The `httpx.Client` is injected so the whole contract can be pinned against
    `httpx.MockTransport` with no target running; both requests are sent to the
    base URL joined with this module's own path constant, as an absolute URL, so a
    client that quietly carried its own `base_url` cannot pass while sending
    somewhere else.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client,
        timeout: float = DEFAULT_CASE_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def _url(self, path: str) -> str:
        """The absolute URL of one endpoint on this target."""
        return f"{self._base_url}{path}"

    def reset(self) -> None:
        """Reset the target to its seed, or raise `TargetUnreachable`.

        Anything but a 2xx is a failure, the 404 included: a 404 means the
        endpoint is not enabled in this environment, and a redirect means
        something other than the target answered. Both are runs that would be
        measured against a seed the previous run mutated.
        """
        url = self._url(RESET_PATH)
        try:
            response = self._client.post(url, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise TargetUnreachable(f"{url}: could not be reached: {exc}") from exc
        if not response.is_success:
            raise TargetUnreachable(f"{url}: answered {response.status_code}, not a reset")

    def run_case(
        self, case: EvalCase, *, other_tenant_markers: tuple[str, ...] = ()
    ) -> tuple[EvalEnvelope, CaseOutcome]:
        """Send one case's turn and map the envelope, or raise `TurnFailed`.

        One attempt. The status is checked BEFORE the body is parsed, so a target
        answering a valid envelope with a 500 is a failure rather than a result.
        """
        url = self._url(TURN_PATH)
        payload = {
            "case_id": case.id,
            "tenant": case.tenant,
            "principal_user": case.principal_user,
            "messages": case.messages,
        }
        try:
            response = self._client.post(url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            kind = (
                TurnFailureKind.TIMEOUT
                if isinstance(exc, httpx.TimeoutException)
                else TurnFailureKind.PROVIDER_ERROR
            )
            raise TurnFailed(f"{case.id}: {url} could not be reached: {exc}", kind=kind) from exc
        if not response.is_success:
            raise TurnFailed(
                f"{case.id}: {url} answered {response.status_code}",
                kind=TurnFailureKind.PROVIDER_ERROR,
            )
        try:
            document = response.json()
        except ValueError as exc:
            raise TurnFailed(
                f"{case.id}: the answer is not valid JSON: {exc}",
                kind=TurnFailureKind.PROVIDER_ERROR,
            ) from exc
        try:
            envelope = EvalEnvelope.model_validate(document)
        except ValidationError as exc:
            raise TurnFailed(
                f"{case.id}: the answer is not a valid envelope: {exc}",
                kind=TurnFailureKind.PROVIDER_ERROR,
            ) from exc
        try:
            versions = parse_prompt_versions(response.headers.get(PROMPT_VERSIONS_HEADER))
        except ValueError as exc:
            raise TurnFailed(
                f"{case.id}: the {PROMPT_VERSIONS_HEADER} header is malformed: {exc}",
                kind=TurnFailureKind.PROVIDER_ERROR,
            ) from exc
        return envelope, to_outcome(
            envelope,
            other_tenant_markers=other_tenant_markers,
            step_prompt_versions=versions,
        )
