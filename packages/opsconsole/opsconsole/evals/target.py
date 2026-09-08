"""The target contract: the reset, the turn, and the envelope a target answers (#33).

This is the first module in the package that opens a socket. Everything it sends
and everything it accepts is published in `docs/eval-target.md`, whose worked
example is `tests/evals/fixtures/envelope.json` byte for byte, so an adopter can
make their own workload evaluable without reading this file
(`test_the_worked_envelope_fixture_parses_field_by_field` is the drift check).

Two endpoints, both POST:

    {target}/admin/test/reset   once, before the first case
    {target}/v1/eval/turn       once per case, one attempt

A target answers a turn with an ENVELOPE rather than a chat completion (an earlier change
decision A). Reading an intent, a decision and a risk class back out of prose
would put the harness's own extraction accuracy inside every number it reports,
and the Helpdesk has authored no reply shape yet, so the contract is published
here and the first target that disagrees names the field.

`EvalEnvelope` forbids extras at every level. A target that renames `decision`
would otherwise load cleanly, every assertion would report `skipped`, and the run
would score higher than a target that answered nothing at all
(`test_an_unknown_field_is_refused_on_every_level_of_the_envelope`).

Two error types, because the two failures are not the same event. A reset that
did not happen aborts the whole run (decision F): a run over an unreset seed is
not comparable to the baseline, and comparability is the only thing the number is
for. A turn that failed is ONE case, recorded as failed, one attempt, no retry
(decision E) - a retried case is a second sample of a non-deterministic system
and would quietly lift the score
(`test_a_failing_turn_is_attempted_exactly_once`).
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from opsconsole.evals.cases import EvalCase
from opsconsole.evals.outcome import CaseOutcome, ObservedToolCall

RESET_PATH: Final[str] = "/admin/test/reset"
"""Where the seed reset lives on a target. Pinned by
`test_the_target_paths_and_timeout_are_the_published_ones`, because a rename here
404s every documented target while this code still looks right."""

TURN_PATH: Final[str] = "/v1/eval/turn"
"""Where one case's turn is sent. Pinned by the same test, and published in
`docs/eval-target.md`."""

DEFAULT_CASE_TIMEOUT_SECONDS: Final[float] = 60.0
"""How long one turn may take before it is a failed case.

The only thing standing between a hung target and a CI job that never ends, so it
is a real default rather than httpx's `None` (which means no timeout at all):
`test_the_client_applies_the_default_timeout_when_none_is_given` reads the
signature, and the exact value is pinned by
`test_the_target_paths_and_timeout_are_the_published_ones`."""


class TargetUnreachable(Exception):
    """The seed could not be reset, so the run may not start (decision F).

    Carries the reset URL, because "reset failed" against the wrong host is
    unactionable (`test_a_failed_reset_raises_target_unreachable_naming_the_url`).
    A plain exception rather than a `GatewayError`, for the reason
    `EvalSetInvalid` is one: this is an operator's error, reported to whoever ran
    the harness, and it never becomes an answer on a wire.
    """


class TurnFailureKind(StrEnum):
    """Why a turn failed, in the two words `17b`'s taxonomy tags it by.

    Two and no third: `timeout` is the clock running out on a target that never
    answered, and `provider_error` is everything else - a non-2xx, a body that is
    not JSON, an envelope the contract does not define, a refused connection. A
    third value would silently drop out of the taxonomy, which maps exactly these
    two (`test_the_turn_failure_kind_vocabulary_is_the_published_two`).
    """

    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"


class TurnFailed(Exception):
    """One case's turn did not produce a usable envelope.

    Every way a turn can fail - a non-2xx, a body that is not JSON, an envelope
    field the contract does not define, a timeout, a refused connection - arrives
    as this one type, so the runner catches one thing to record one failed case
    and a flaky case cannot abort the run. The message names the case id: with two
    hundred cases in a set, "read timeout" names nothing an author can open
    (`test_a_failed_turn_raises_turn_failed_naming_the_case`).

    `kind` is set at the raise site rather than read back out of the message: the message is a human sentence naming a case id and a
    URL, and a taxonomy deciding a stored tag by searching it for "timed out"
    would depend on prose that is edited for readability. Each of the four raise
    sites in `run_case` carries the kind its own cause determines, pinned one
    site at a time by `test_a_failed_turn_carries_the_kind_its_raise_site_determined`.
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
    make the recorded verdict depend on the order those two ran in
    (`test_the_envelope_is_frozen`).

    The defaults read as "nothing observed" and never as an observation - a target
    that answered `{}` is not a refusal with sources
    (`test_an_empty_envelope_reads_as_nothing_observed`). The four measured fields
    are asserted by nothing here: `region` is what #38 checks, and `16c` writes
    the other three into a report.
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


def to_outcome(
    envelope: EvalEnvelope, *, other_tenant_markers: tuple[str, ...] = ()
) -> CaseOutcome:
    """Map one envelope onto the record `assert_case` reads.

    The markers come from the caller and never from the envelope (an earlier change
    decision D): they are the seed's, not the target's, and a mapping that filled
    the outcome from the envelope alone would leave every `cross_tenant` case's
    leak assertion reporting `skipped`
    (`test_the_other_tenant_markers_reach_the_outcome`).
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
    )


class TargetClient:
    """One OpenAI-compatible workload under evaluation, addressed by base URL.

    The `httpx.Client` is injected so the whole contract can be pinned against
    `httpx.MockTransport` with no target running; both requests are sent to the
    base URL joined with this module's own path constant, as an absolute URL, so a
    client that quietly carried its own `base_url` cannot pass while sending
    somewhere else
    (`test_the_turn_posts_the_case_and_its_messages_to_the_published_path`).
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
        measured against a seed the previous run mutated
        (`test_a_failed_reset_raises_target_unreachable_naming_the_url`).
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

        One attempt (decision E). The status is checked BEFORE the body is parsed,
        so a target answering a valid envelope with a 500 is a failure rather than
        a result.
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
        return envelope, to_outcome(envelope, other_tenant_markers=other_tenant_markers)
