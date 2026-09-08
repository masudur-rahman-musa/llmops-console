"""`evals/target.py`: the target contract - the reset, the turn and the envelope.

No network and no backing service. Every case here drives a `TargetClient` built
over an `httpx.Client(transport=httpx.MockTransport(...))`, so the whole
request/response contract is pinned without a target existing, exactly as
`tests/evals/test_assertions.py` pins the verdict without a run existing.

The contract these cases are written against, before it exists:

* `EnvelopeToolCall(name=..., args={})` and `EvalEnvelope(...)`, both frozen and
  `extra="forbid"` at every level, every field optional with the defaults listed
  in `docs/eval-target.md`.
* `TargetClient(base_url, client=..., timeout=...)` with `reset() -> None` and
  `run_case(case, *, other_tenant_markers=()) -> tuple[EvalEnvelope, CaseOutcome]`.
* `RESET_PATH`, `TURN_PATH` and `DEFAULT_CASE_TIMEOUT_SECONDS` - the constants
  a project rule requires a test to pin exactly.
* `TargetUnreachable` for a reset that did not happen (the run cannot start,
  decision F) and `TurnFailed` for one case's turn, whose message names the case
  id so an operator learns WHICH case the target disagreed about.

Both requests are posted to `base_url` joined with the module's own path
constant, asserted as the full URL, so a client that quietly uses its own
`httpx.Client(base_url=...)` instead cannot pass while sending somewhere else.

`fixtures/envelope.json` is the worked example `docs/eval-target.md` carries
verbatim; `test_the_worked_envelope_fixture_parses_field_by_field` is the drift
check, mirroring 16a's `test_the_canonical_example_line_loads_field_by_field`.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from opsconsole.evals.cases import EvalCase, EvalCategory, ExpectedOutcome
from opsconsole.evals.outcome import ObservedToolCall
from opsconsole.evals.target import (
    DEFAULT_CASE_TIMEOUT_SECONDS,
    RESET_PATH,
    TURN_PATH,
    EnvelopeToolCall,
    EvalEnvelope,
    TargetClient,
    TargetUnreachable,
    TurnFailed,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

ENVELOPE_PATH = FIXTURE_DIR / "envelope.json"
"""The worked envelope of an earlier change decision A, one complete JSON document.

`docs/eval-target.md` quotes this file byte for byte. The drift it catches is the
whole point of the fixture: an adopter who copies the published envelope and a
model that has since renamed a field disagree HERE, not at the first
`ops-eval run` against a real Helpdesk.
"""

BASE_URL = "https://target.example"

Handler = Callable[[httpx.Request], httpx.Response]


def envelope_payload() -> dict[str, Any]:
    """The worked envelope as a plain dict, so a case can perturb one key of it."""
    loaded: dict[str, Any] = json.loads(ENVELOPE_PATH.read_text(encoding="utf-8"))
    return loaded


def make_client(handler: Handler, *, timeout: float | None = None) -> TargetClient:
    """A client whose every request is answered in-process by `handler`."""
    transport = httpx.MockTransport(handler)
    if timeout is None:
        return TargetClient(BASE_URL, client=httpx.Client(transport=transport))
    return TargetClient(BASE_URL, client=httpx.Client(transport=transport), timeout=timeout)


def make_case(
    case_id: str = "case-1", category: EvalCategory = EvalCategory.ACCESS_AUTO
) -> EvalCase:
    """One case with nothing expected: what the TARGET returned is what is under test."""
    return EvalCase(
        id=case_id,
        category=category,
        tenant="acme",
        principal_user="dana.okafor",
        messages=[{"role": "user", "content": "Grant me read access to finance-reports."}],
        expected=ExpectedOutcome(),
        judge_rubric=None,
    )


def responder(payload: object, *, status: int = 200) -> Handler:
    """Answer every request with one JSON body and one status."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handle


def raiser(error: Exception) -> Handler:
    """Answer every request by raising a transport-level error, as httpx would."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise error

    return handle


# --------------------------------------------------------------------------
# the published constants
# --------------------------------------------------------------------------


def test_the_target_paths_and_timeout_are_the_published_ones() -> None:
    """Catches a rename of either path, which would make every documented target 404 while
    the code still looks right, and a per-case timeout changed from 60s - the one value
    standing between a hung target and a CI job that never ends. Asserted as exact values, so a band or a "roughly a minute" cannot pass."""
    assert RESET_PATH == "/admin/test/reset"
    assert TURN_PATH == "/v1/eval/turn"
    assert DEFAULT_CASE_TIMEOUT_SECONDS == 60.0


def test_the_client_applies_the_default_timeout_when_none_is_given() -> None:
    """Catches a `timeout` parameter defaulting to `None`, which in httpx means NO timeout
    at all: every behavioural case here passes an answer immediately, so none of them can
    see a client that would hang forever against a real target. The signature is read
    because that is where the defect lives."""
    default = inspect.signature(TargetClient.__init__).parameters["timeout"].default

    assert default == DEFAULT_CASE_TIMEOUT_SECONDS


# --------------------------------------------------------------------------
# the envelope model and its worked example
# --------------------------------------------------------------------------


def test_the_worked_envelope_fixture_parses_field_by_field() -> None:
    """Catches a model that drifts from the envelope an adopter is told to copy: a renamed
    field, a `tool_calls` entry that is not `{name, args}`, a `cost_usd` parsed as a float
    rather than a `Decimal`. Every field is read back, so a model that silently drops one
    fails here rather than by reporting an unfilled `CaseOutcome` at run time."""
    envelope = EvalEnvelope.model_validate(envelope_payload())

    assert envelope.intent == "request_access"
    assert envelope.tool_calls == (
        EnvelopeToolCall(
            name="grant_access",
            args={"resource": "finance-reports", "level": "read", "ticket_id": "HD-4821"},
        ),
    )
    assert envelope.decision == "auto_approved"
    assert envelope.risk_class == "low"
    assert envelope.refused is False
    assert envelope.clarified is False
    assert envelope.citations == ("access-policy-v3",)
    assert envelope.reply == "Read access to finance-reports is active, per the access policy."
    assert envelope.latency_ms == 1840
    assert envelope.cost_usd == Decimal("0.0142")
    assert envelope.provider == "anthropic_eu"
    assert envelope.region == "eu"


def test_an_empty_envelope_reads_as_nothing_observed() -> None:
    """Catches defaults that read as an OBSERVATION rather than as silence: `refused=True`
    or a non-empty citations default would turn a target that answered `{}` into a case
    that looks like a refusal with sources. Every default is asserted, because the runner
    hands these straight to `assert_case`."""
    envelope = EvalEnvelope()

    assert envelope.intent is None
    assert envelope.tool_calls == ()
    assert envelope.decision is None
    assert envelope.risk_class is None
    assert envelope.refused is False
    assert envelope.clarified is False
    assert envelope.citations == ()
    assert envelope.reply == ""
    assert envelope.latency_ms is None
    assert envelope.cost_usd is None
    assert envelope.provider is None
    assert envelope.region is None


def test_an_unknown_field_is_refused_on_every_level_of_the_envelope() -> None:
    """Catches `extra="forbid"` missing from either level. A target that renames `decision`
    to `outcome` would otherwise load cleanly, every assertion would report `skipped`, and
    the run would score higher than a target answering nothing. Both payloads carry every
    other field correctly, so only the extras rule can refuse them."""
    top_level = envelope_payload()
    top_level["outcome"] = "auto_approved"
    with pytest.raises(ValidationError):
        EvalEnvelope.model_validate(top_level)

    nested = envelope_payload()
    nested["tool_calls"][0]["arguments"] = {"resource": "finance-reports"}
    with pytest.raises(ValidationError):
        EvalEnvelope.model_validate(nested)


def test_the_envelope_is_frozen() -> None:
    """Catches a mutable envelope: the runner hands the same instance to the residency
    check and to the `CaseOutcome` mapping, so a step that normalised `region` in place
    would make the recorded verdict depend on the order those two ran in."""
    envelope = EvalEnvelope.model_validate(envelope_payload())

    with pytest.raises(ValueError):
        envelope.region = "us"  # type: ignore[misc]


# --------------------------------------------------------------------------
# the turn
# --------------------------------------------------------------------------


def test_the_turn_posts_the_case_and_its_messages_to_the_published_path() -> None:
    """Catches a client that posts somewhere else or sends a body the documented target
    cannot read - the failure an adopter meets on their first run and cannot debug from
    the outside. The full URL is asserted, so a client joining the base URL wrongly
    (a dropped path segment, a doubled slash) fails here."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=envelope_payload())

    case = make_case("access-auto-001")
    make_client(handle).run_case(case)

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"{BASE_URL}{TURN_PATH}"
    assert json.loads(seen[0].content) == {
        "case_id": "access-auto-001",
        "tenant": "acme",
        "principal_user": "dana.okafor",
        "messages": [{"role": "user", "content": "Grant me read access to finance-reports."}],
    }


def test_a_valid_envelope_maps_field_by_field_onto_a_case_outcome() -> None:
    """Catches a mapping that drops a field on the way to the verdict - the defect that
    makes an assertion report `skipped` while the target answered correctly. The four
    MEASURED fields (`latency_ms`, `cost_usd`, `provider`, `region`) are asserted too:
    nothing in `assert_case` reads them, so only this test stops them being forgotten,
    and `16c`'s report columns and the residency check are what read them."""
    case = make_case()

    envelope, outcome = make_client(responder(envelope_payload())).run_case(case)

    assert envelope.decision == "auto_approved"
    assert outcome.intent == "request_access"
    assert outcome.tool_calls == (
        ObservedToolCall(
            name="grant_access",
            args={"resource": "finance-reports", "level": "read", "ticket_id": "HD-4821"},
        ),
    )
    assert outcome.decision == "auto_approved"
    assert outcome.risk_class == "low"
    assert outcome.refused is False
    assert outcome.clarified is False
    assert outcome.citations == ("access-policy-v3",)
    assert outcome.reply == "Read access to finance-reports is active, per the access policy."
    assert outcome.latency_ms == 1840
    assert outcome.cost_usd == Decimal("0.0142")
    assert outcome.provider == "anthropic_eu"
    assert outcome.region == "eu"


def test_the_other_tenant_markers_reach_the_outcome() -> None:
    """Catches a mapping that fills the outcome from the envelope ALONE. The envelope
    cannot carry the markers - they are the seed's, not the target's (16a decision D) - so
    a mapping that ignores the argument leaves `other_tenant_markers` empty and 16a's
    `cross_tenant_leak` assertion reports `skipped` for every cross-tenant case in the
    set, which is the one assertion a leak has to fail."""
    case = make_case(category=EvalCategory.CROSS_TENANT)

    _, outcome = make_client(responder(envelope_payload())).run_case(
        case, other_tenant_markers=("globex", "initech")
    )

    assert outcome.other_tenant_markers == ("globex", "initech")


def test_the_markers_default_to_empty_rather_than_to_the_last_call() -> None:
    """Catches markers defaulted to a mutable module-level value or carried over from a
    previous case - either of which would make an ordinary helpdesk case fail a leak
    assertion it never opted into, and make the run's score depend on case order."""
    _, outcome = make_client(responder(envelope_payload())).run_case(make_case())

    assert outcome.other_tenant_markers == ()


def unknown_field_handler() -> Handler:
    """A target that renamed one envelope field."""
    payload = envelope_payload()
    payload["outcome"] = payload.pop("decision")
    return responder(payload)


def malformed_json_handler() -> Handler:
    """A target that answered 200 with a body that is not JSON at all."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b'{"decision": "auto_appro', headers={"content-type": "application/json"}
        )

    return handle


BAD_TURNS: list[tuple[str, Handler]] = [
    ("an unknown envelope field", unknown_field_handler()),
    ("a malformed JSON body", malformed_json_handler()),
    ("a 500 from the target", responder(envelope_payload(), status=500)),
    ("a 404 from the target", responder(envelope_payload(), status=404)),
    ("a read timeout", raiser(httpx.ReadTimeout("timed out"))),
    ("a refused connection", raiser(httpx.ConnectError("connection refused"))),
]
"""One handler per way a turn can fail, each raising `TurnFailed`.

The two status rows answer a COMPLETE and valid envelope body, so a client that
parses first and never looks at the status cannot pass them; the unknown-field
row carries every other field correctly, so only `extra="forbid"` can refuse it.
"""


@pytest.mark.parametrize(("label", "handler"), BAD_TURNS, ids=[row[0] for row in BAD_TURNS])
def test_a_failed_turn_raises_turn_failed_naming_the_case(label: str, handler: Handler) -> None:
    """Catches a turn failure that surfaces as a raw `httpx` error or a `ValidationError`:
    the runner catches ONE type to record a failed case (decision E), so a leaked type
    aborts the whole run at the first flaky case. Catches a message with no case id too -
    with 200 cases in a set, "read timeout" names nothing an author can open."""
    case = make_case("access-auto-001")

    with pytest.raises(TurnFailed) as raised:
        make_client(handler).run_case(case)

    assert "access-auto-001" in str(raised.value)


def test_a_failing_turn_is_attempted_exactly_once() -> None:
    """Catches a retry (decision E). A retried case is a second sample of a
    non-deterministic system, so a transient failure that succeeds on attempt two quietly
    lifts the score and makes the run unrepeatable. Only a call count can see this: the
    raised error looks identical either way."""
    attempts: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(TurnFailed):
        make_client(handle).run_case(make_case())

    assert len(attempts) == 1


# --------------------------------------------------------------------------
# the reset
# --------------------------------------------------------------------------


def test_reset_posts_to_the_published_reset_path() -> None:
    """Catches a reset sent to the wrong URL or as a GET, which a permissive target would
    answer 200 to while resetting nothing - and every number the run produces would then
    be measured against a seed the previous run mutated. Called for its effect rather than
    compared to `None`."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "reset"})

    make_client(handle).reset()

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"{BASE_URL}{RESET_PATH}"


BAD_RESETS: list[tuple[str, Handler]] = [
    ("the endpoint is not enabled here", responder({"detail": "not found"}, status=404)),
    ("the target failed to reset", responder({"detail": "boom"}, status=500)),
    ("a redirect that never resets", responder({}, status=302)),
    ("a refused connection", raiser(httpx.ConnectError("connection refused"))),
    ("a connect timeout", raiser(httpx.ConnectTimeout("timed out"))),
]
"""Every way the seed can fail to be reset. Each aborts the run (decision F)."""


@pytest.mark.parametrize(("label", "handler"), BAD_RESETS, ids=[row[0] for row in BAD_RESETS])
def test_a_failed_reset_raises_target_unreachable_naming_the_url(
    label: str, handler: Handler
) -> None:
    """Catches a reset that reports success on anything but a 2xx - a 404 means the
    endpoint is not enabled in this environment and a 302 means something else answered,
    and a run over an unreset seed is not comparable to the baseline, which is the only
    thing the number is for. The URL is in the message because "reset failed" against the
    wrong host is unactionable."""
    with pytest.raises(TargetUnreachable) as raised:
        make_client(handler).reset()

    assert f"{BASE_URL}{RESET_PATH}" in str(raised.value)
