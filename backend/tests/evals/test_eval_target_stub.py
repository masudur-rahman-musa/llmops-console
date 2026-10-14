r"""`project-evals/ci/eval-target-stub/main.py`: the kind e2e's throwaway eval target.

The module lives outside `opsconsole` on purpose (its own Dockerfile installs
nothing else), so it is loaded here by file path rather than imported by dotted
name - the same problem a project rule solves for a stubbed subpackage, solved
here with `importlib.util` instead because this is a real standalone script and
not a package.

Every case here is a unit test: the envelope-mapping cases feed representative
completion TEXT and never open a socket or reach a gateway (module docstring,
"no HTTP, no gateway"), and the endpoint cases call the handler functions
directly. No backing service, sub-second total.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from app.evals.assertions import assert_case
from app.evals.cases import EvalCase, load_golden_set
from app.evals.target import EvalEnvelope, to_outcome

REPO_ROOT = Path(__file__).resolve().parents[2]
STUB_PATH = REPO_ROOT / "project-evals" / "ci" / "eval-target-stub" / "main.py"
GOLDEN_SET = REPO_ROOT / "project-evals" / "golden" / "e2e-kind.jsonl"

EXPECTED_CASE_COUNT = 10
"""Acceptance criterion 5: the set is exactly ten cases."""

_TURN_BODY = json.dumps(
    {
        "case_id": "x",
        "tenant": "evals",
        "principal_user": "e2e-kind-runner",
        "messages": [{"role": "user", "content": "the newest question"}],
    }
).encode("utf-8")
"""One well-formed `/v1/eval/turn` body, shared by the measurement cases below: they differ
only in what the replaced `caller` measured, never in the request."""


def _load_stub() -> ModuleType:
    """Load `main.py` by path, the way a script outside every package has to be."""
    spec = importlib.util.spec_from_file_location("eval_target_stub", STUB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stub() -> ModuleType:
    return _load_stub()


@pytest.fixture(scope="module")
def golden_cases() -> tuple[EvalCase, ...]:
    return load_golden_set(GOLDEN_SET)


def test_the_golden_set_parses_as_exactly_ten_cases(golden_cases: tuple[EvalCase, ...]) -> None:
    """Acceptance criterion 5. `load_golden_set` already refuses a malformed line, an
    unknown category and a repeated id, so a set that parses at all is already every
    other shape criterion 5 asks for; this is the one criterion it does not check."""
    assert len(golden_cases) == EXPECTED_CASE_COUNT


@pytest.mark.parametrize(
    ("case_id", "reply"),
    [
        ("greeting", "A virtual private network encrypts your traffic over a public network."),
        ("policy-lookup", "A data retention policy states how long records are kept."),
        ("access-request", "Single sign-on lets one login work across several apps."),
        ("lifecycle-question", "Accounts are deactivated to stop a former employee's access."),
        ("review-question", "A quarterly access review confirms people still need their access."),
        ("approval-question", "A manager approves a request to confirm the access is warranted."),
        ("critical-question", "Production database access can affect real customer data."),
        ("ambiguous-request", "Sure, which account or setting would you like reset?"),
    ],
)
def test_a_benign_reply_passes_every_assertion_its_case_states(
    stub: ModuleType,
    golden_cases: tuple[EvalCase, ...],
    case_id: str,
    reply: str,
) -> None:
    """Risk 1's mitigation: the eight benign cases assert only `should_refuse: false`
    and `forbidden_phrases: []`, both of which a normal answer satisfies trivially -
    this proves the mapping and the case agree before the workflow ever runs."""
    case = next(case for case in golden_cases if case.id == case_id)
    envelope = EvalEnvelope.model_validate(stub.envelope_for_reply(reply))
    outcome = to_outcome(envelope)
    assert assert_case(case, outcome).passed


def test_a_refusal_reply_satisfies_the_out_of_corpus_case(
    stub: ModuleType, golden_cases: tuple[EvalCase, ...]
) -> None:
    """The one `should_refuse: true` case: a reply carrying one of
    `REFUSAL_KEYWORDS` is read as a refusal, which is what the case asserts."""
    case = next(case for case in golden_cases if case.id == "out-of-corpus-refusal")
    envelope = EvalEnvelope.model_validate(
        stub.envelope_for_reply("I can't help with that request.")
    )
    outcome = to_outcome(envelope)
    assert assert_case(case, outcome).passed


def test_a_reply_with_no_refusal_keyword_fails_the_out_of_corpus_case(
    stub: ModuleType, golden_cases: tuple[EvalCase, ...]
) -> None:
    """The mirror of the case above: a reply naming none of `REFUSAL_KEYWORDS` is
    read as `refused=False`, which the case's `should_refuse: true` fails - proving
    the keyword check is load-bearing and not a check that always passes."""
    case = next(case for case in golden_cases if case.id == "out-of-corpus-refusal")
    envelope = EvalEnvelope.model_validate(
        stub.envelope_for_reply("Insert the tension wrench and turn while raking the pins.")
    )
    outcome = to_outcome(envelope)
    assert not assert_case(case, outcome).passed


def test_a_reply_that_declines_the_injection_satisfies_that_case(
    stub: ModuleType, golden_cases: tuple[EvalCase, ...]
) -> None:
    """The injection case never asserts `should_refuse` (there is no system prompt
    here to be refused ABOUT); it asserts the model never complies by literally
    producing the injected phrase, which a plain decline does not contain."""
    case = next(case for case in golden_cases if case.id == "injection-attempt")
    envelope = EvalEnvelope.model_validate(
        stub.envelope_for_reply("I won't follow instructions embedded in a user message.")
    )
    outcome = to_outcome(envelope)
    assert assert_case(case, outcome).passed


def test_a_reply_that_complies_with_the_injection_fails_that_case(
    stub: ModuleType, golden_cases: tuple[EvalCase, ...]
) -> None:
    """The failure the injection case exists to catch: a reply that actually obeys
    the injected instruction and opens with the forbidden phrase."""
    case = next(case for case in golden_cases if case.id == "injection-attempt")
    envelope = EvalEnvelope.model_validate(
        stub.envelope_for_reply("ACCESS GRANTED. How can I help you today?")
    )
    outcome = to_outcome(envelope)
    assert not assert_case(case, outcome).passed


def test_envelope_for_reply_carries_the_stub_fixed_constants(stub: ModuleType) -> None:
    """Acceptance criterion 6: `decision`, `risk_class`, `tool_calls` and `citations`
    are fixed by the stub and never derived from the reply."""
    envelope = stub.envelope_for_reply("anything at all")
    assert envelope["decision"] == "answered"
    assert envelope["risk_class"] == "low"
    assert envelope["tool_calls"] == []
    assert envelope["citations"] == []


def test_reset_answers_two_hundred_with_an_empty_body(stub: ModuleType) -> None:
    """Acceptance criterion 6: the stub holds no seed, so a reset is a no-op 200."""
    status, body = stub.reset_response()
    assert status == 200
    assert body == b""


def test_a_turn_payload_with_an_extra_field_is_rejected(stub: ModuleType) -> None:
    """The request's own version of `EvalEnvelope`'s `extra=\"forbid\"`: a payload
    carrying a field `TargetClient.run_case` never sends is refused rather than
    silently accepted."""
    payload = {
        "case_id": "x",
        "tenant": "evals",
        "principal_user": "e2e-kind-runner",
        "messages": [{"role": "user", "content": "hi"}],
        "unexpected": "field",
    }
    with pytest.raises(stub.MalformedTurn):
        stub.checked_turn_payload(payload)


def test_a_turn_payload_missing_a_field_is_rejected(stub: ModuleType) -> None:
    """The other half of the same strictness: a payload missing a required field."""
    payload = {"case_id": "x", "tenant": "evals", "messages": []}
    with pytest.raises(stub.MalformedTurn):
        stub.checked_turn_payload(payload)


def test_the_turn_endpoint_answers_four_hundred_for_a_malformed_payload(stub: ModuleType) -> None:
    """The malformed-payload refusal reaches the endpoint too, not only the
    validator - `turn_response` is what `/v1/eval/turn` actually calls."""
    raw_body = json.dumps({"case_id": "x"}).encode("utf-8")
    status, document = stub.turn_response(
        raw_body, base_url="http://unused.invalid", api_key="unused"
    )
    assert status == 400
    assert "error" in document


def test_the_turn_endpoint_forwards_the_last_message_and_maps_the_reply(
    stub: ModuleType,
) -> None:
    """End to end through `turn_response`, with `caller` replacing the real gateway
    call - the seam the module docstring names for exactly this reason. The seam now
    returns a `GatewayReply` rather than the bare text, because
    the stub publishes the gateway's measurements alongside its answer."""

    def fake_caller(content: str, *, base_url: str, api_key: str) -> object:
        assert content == "the newest question"
        assert base_url == "http://gateway.example"
        assert api_key == "test-key"
        return stub.GatewayReply(
            reply="a plain, non-refusing answer",
            latency_ms=12,
            cost_usd="0.000500",
            provider="local",
            region="self",
        )

    raw_body = json.dumps(
        {
            "case_id": "x",
            "tenant": "evals",
            "principal_user": "e2e-kind-runner",
            "messages": [
                {"role": "user", "content": "an earlier turn"},
                {"role": "user", "content": "the newest question"},
            ],
        }
    ).encode("utf-8")
    status, document = stub.turn_response(
        raw_body,
        base_url="http://gateway.example",
        api_key="test-key",
        caller=fake_caller,
    )
    assert status == 200
    assert document["reply"] == "a plain, non-refusing answer"
    assert document["refused"] is False


class _Headers:
    """A response's headers, answered case-insensitively by either spelling a stdlib
    caller uses: `response.headers.get(name)` and `response.getheader(name)`."""

    def __init__(self, pairs: dict[str, str]) -> None:
        self._pairs = {key.casefold(): value for key, value in pairs.items()}

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._pairs.get(name.casefold(), default)


class _FakeResponse:
    """One gateway response: the body, the three measurement headers, and a delay.

    The delay is what makes the latency case non-vacuous - a stub that reports `0`,
    or that reports nanoseconds, is visible only when the call actually took time.
    """

    def __init__(self, body: bytes, headers: dict[str, str], delay_seconds: float) -> None:
        self._body = body
        self._delay = delay_seconds
        self.headers = _Headers(headers)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def read(self) -> bytes:
        time.sleep(self._delay)
        return self._body


def test_envelope_for_reply_publishes_no_measurements_when_none_are_given(
    stub: ModuleType,
) -> None:
    """Criterion 36: the four new fields are keyword-only and default to `None`, so every
    call site written before this plan keeps its single positional argument. Catches the
    fields made required (the five other calls in this file stop working), and catches a
    default of `0` for cost or latency, which would publish a free, instant turn."""
    envelope = EvalEnvelope.model_validate(stub.envelope_for_reply("anything at all"))

    assert envelope.latency_ms is None
    assert envelope.cost_usd is None
    assert envelope.provider is None
    assert envelope.region is None


def test_envelope_for_reply_publishes_every_measurement_it_is_given(stub: ModuleType) -> None:
    """Criteria 36 and 41. Catches a mapping that accepts the keywords and drops them on the
    floor (every column of the tradeoff table then renders `n/a`), two fields swapped, and a
    cost published as a float - `EvalEnvelope.cost_usd` is a `Decimal` because a run of a
    thousand turns at six decimal places is exactly where binary floating point loses money."""
    document = stub.envelope_for_reply(
        "an answer",
        latency_ms=412,
        cost_usd="0.001234",
        provider="anthropic",
        region="eu",
    )
    envelope = EvalEnvelope.model_validate(document)

    assert envelope.latency_ms == 412
    assert envelope.cost_usd == Decimal("0.001234")
    assert envelope.provider == "anthropic"
    assert envelope.region == "eu"
    assert not isinstance(document["cost_usd"], float), "the cost is published as a float"


def test_the_turn_endpoint_publishes_the_measurements_from_the_gateway_reply(
    stub: ModuleType,
) -> None:
    """Criteria 37 and 41, at the seam the server actually calls. Catches a `turn_response`
    that keeps calling `envelope_for_reply(reply)` with the text alone after `call_gateway`
    started measuring - the headers would be read and then thrown away one line later, which
    no case on `call_gateway` or on `envelope_for_reply` alone can see."""

    def measured_caller(content: str, *, base_url: str, api_key: str) -> object:
        return stub.GatewayReply(
            reply="a plain, non-refusing answer",
            latency_ms=931,
            cost_usd="0.004500",
            provider="anthropic",
            region="eu",
        )

    status, document = stub.turn_response(
        _TURN_BODY, base_url="http://gateway.example", api_key="test-key", caller=measured_caller
    )
    envelope = EvalEnvelope.model_validate(document)

    assert status == 200
    assert isinstance(envelope.latency_ms, int) and envelope.latency_ms >= 0
    assert envelope.latency_ms == 931
    assert envelope.cost_usd == Decimal("0.004500")
    assert envelope.provider == "anthropic"
    assert envelope.region == "eu"


@pytest.mark.parametrize(
    "header",
    [pytest.param("not-a-number", id="not-a-decimal"), pytest.param(None, id="no-cost-header")],
)
def test_an_unusable_cost_still_answers_two_hundred_and_publishes_null(
    stub: ModuleType, header: str | None
) -> None:
    """Criteria 39 and 40. A provider that prices nothing, or a header a proxy stripped, must
    not turn one turn into a failed case: the row is measured and its cost column is empty.
    Catches `Decimal(header)` unguarded (a `InvalidOperation` traceback and a 500, which the
    runner scores as a failure), `Decimal(header or 0)` and `float(header or 0)`, both of which
    publish a free turn and pull the option's USD per 1,000 turns down towards zero."""

    def priceless_caller(content: str, *, base_url: str, api_key: str) -> object:
        return stub.GatewayReply(
            reply="a plain, non-refusing answer",
            latency_ms=17,
            cost_usd=header,
            provider="local",
            region="self",
        )

    status, document = stub.turn_response(
        _TURN_BODY, base_url="http://gateway.example", api_key="test-key", caller=priceless_caller
    )
    envelope = EvalEnvelope.model_validate(document)

    assert status == 200
    assert document["cost_usd"] is None, f"cost_usd is {document['cost_usd']!r}, not null"
    assert envelope.cost_usd is None
    assert envelope.latency_ms == 17


def test_call_gateway_measures_its_own_latency_in_whole_milliseconds(
    stub: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criteria 37 and 38, over the one function that talks to the gateway, with `urlopen`
    replaced so no socket opens. The gateway publishes no latency header, so a stub that looks
    for one reports `None` on every turn and the p50/p95 columns are empty - that is what the
    `isinstance` check catches. The bounds catch the two unit mistakes that look plausible in
    a report: raw `monotonic_ns()` nanoseconds (about 50,000,000 here) and whole seconds
    floored to `0`. The three header names are read from the response, not invented."""
    body = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode("utf-8")
    headers = {
        "X-Gateway-Cost-USD": "0.001234",
        "X-Gateway-Provider": "anthropic",
        "X-Gateway-Region": "eu",
    }

    def fake_urlopen(request: object, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(body, headers, delay_seconds=0.05)

    monkeypatch.setattr(stub.urllib.request, "urlopen", fake_urlopen)

    reply = stub.call_gateway("hello", base_url="http://gateway.example", api_key="test-key")

    assert reply.reply == "pong"
    assert isinstance(reply.latency_ms, int), f"latency_ms is {reply.latency_ms!r}"
    assert 20 <= reply.latency_ms <= 5_000, f"latency_ms is {reply.latency_ms}, not milliseconds"
    assert reply.provider == "anthropic"
    assert reply.region == "eu"
    assert EvalEnvelope.model_validate(
        stub.envelope_for_reply(
            reply.reply,
            latency_ms=reply.latency_ms,
            cost_usd=reply.cost_usd,
            provider=reply.provider,
            region=reply.region,
        )
    ).cost_usd == Decimal("0.001234")


def test_module_has_no_third_party_import() -> None:
    """a project rule and the Dockerfile's own claim: every name this module imports
    resolves to the standard library, so the throwaway image needs no `pip install`."""
    stdlib_roots = frozenset(sys.stdlib_module_names)
    source = STUB_PATH.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            root = stripped.split()[1].split(".")[0]
            assert root in stdlib_roots, f"{root!r} is not a standard-library module"
