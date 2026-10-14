r"""The kind e2e's eval target: `app.evals.target`'s contract, over the gateway's
own `/v1/chat/completions`.

Stdlib only (`http.server`, `urllib.request`) - no dependency this repo has not already
approved, and none for the throwaway image `Dockerfile` in this directory to install. This module is a CLIENT of the published target contract
(`app.evals.target.TargetClient`), never a change to it, so the two endpoints and
the payload shapes are duplicated here rather than imported (plan decision, "Will NOT
touch `target.py`") - the stub's own image carries none of `opsconsole`.

Two endpoints, both POST, exactly `target.py`'s:

    /admin/test/reset   answered 200 with an empty body - this stub holds no seed to reset
    /v1/eval/turn       the case's payload in, one `EvalEnvelope`-shaped document out

A turn forwards the case's last message to the deployed gateway's own
`POST /v1/chat/completions`, alias `agent-default`, tenant `evals`, with NO
`X-Data-Class` header. That default (`confidential`) is what keeps this tenant's
resolved candidates down to the `self` region alone: `routing.yaml`'s `evals` tenant
names `allow_regions: [eu, self]` for `confidential` regardless of `ANTHROPIC_API_KEY`
being set, `anthropic_eu` ships disabled, and only `local` (region `self`) is left - so
the kind cluster's one real provider is reached with no header, no key and no
`routing.yaml` change (`test_the_alias_and_tenant_resolve_to_the_local_provider_only` in
`infra/tests/infra/test_k8s_ci_kind_overlay.py` pins the routing-policy half of this claim).

The tradeoff measurement run reuses this stub against a DIFFERENT policy:
`dev/scripts/tradeoff_policy.py` merges one option's overlay over the shipped file, and the
two API options widen `confidential` there precisely because this stub sends no header.
That is a change to the policy the gateway loads and none to this module - which is why
the turn publishes what the gateway measured (`GatewayReply`) rather than judging any of
it here.
"""

from __future__ import annotations

import decimal
import http.server
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from typing import Any, Final, NamedTuple

RESET_PATH: Final[str] = "/admin/test/reset"
TURN_PATH: Final[str] = "/v1/eval/turn"

TURN_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"case_id", "tenant", "principal_user", "messages"}
)
"""Exactly the keys `TargetClient.run_case` sends, no more and no fewer - the same
strictness `EvalEnvelope`'s `extra=\"forbid\"` gives the answer, given to the request
(`test_a_turn_payload_with_an_extra_field_is_rejected`,
`test_a_turn_payload_missing_a_field_is_rejected`)."""

ALIAS: Final[str] = "agent-default"
"""The `routing.yaml` alias this stub asks the gateway to resolve, per the module
docstring's routing argument."""

COST_HEADER: Final[str] = "X-Gateway-Cost-USD"
PROVIDER_HEADER: Final[str] = "X-Gateway-Provider"
REGION_HEADER: Final[str] = "X-Gateway-Region"
"""Three of the seven `X-Gateway-*` headers a served answer carries, named here so
this stub reads what the gateway really writes rather than a header it invented
(`test_call_gateway_measures_its_own_latency_in_whole_milliseconds`).

There is no fourth: the gateway publishes no latency header, so `call_gateway`
times itself."""

NANOSECONDS_PER_MILLISECOND: Final[int] = 1_000_000
"""`time.monotonic_ns` counts nanoseconds and `EvalEnvelope.latency_ms` is whole
milliseconds. Both unit mistakes this divisor prevents - raw nanoseconds, and
seconds floored to zero - would be published as a plausible-looking number rather
than an error (`test_call_gateway_measures_its_own_latency_in_whole_milliseconds`)."""

GATEWAY_TIMEOUT_SECONDS: Final[float] = 30.0
"""How long one forwarded chat completion may take. The 3B local model is the whole
budget: a hang here is one case timing out rather than the stub itself."""

REFUSAL_KEYWORDS: Final[tuple[str, ...]] = (
    "i cannot",
    "i can't",
    "i won't",
    "i will not",
    "i'm not able",
    "i am not able",
    "unable to",
    "not able to",
    "cannot help",
    "can't help",
    "cannot provide",
    "can't provide",
    "cannot assist",
    "can't assist",
    "i'm sorry, but",
    "sorry, but i",
)
"""The fixed refusal vocabulary `envelope_for_reply` checks a reply against, casefolded
on both sides. Broad on purpose: this stub scores a REAL model's freeform text, and a
narrow list would read a rephrased refusal as an answer."""

CLARIFY_KEYWORDS: Final[tuple[str, ...]] = (
    "could you clarify",
    "can you clarify",
    "what do you mean",
    "which one do you mean",
    "can you specify",
    "could you specify",
    "could you provide more detail",
)
"""The fixed clarify vocabulary, `REFUSAL_KEYWORDS`'s counterpart. No case in
`project-evals/golden/e2e-kind.jsonl` asserts `should_clarify`, but the mapping still
publishes the field the way `EvalEnvelope` defines it."""


class MalformedTurn(Exception):
    """The `/v1/eval/turn` body does not carry exactly `TURN_REQUIRED_FIELDS`, or its
    `messages` carries no usable text. Answered as one line and a 400, never a
    traceback (`test_a_turn_payload_with_an_extra_field_is_rejected`)."""


def checked_turn_payload(payload: object) -> dict[str, Any]:
    """Refuse a body that is not a JSON object or does not carry exactly the four
    required fields - the request's own version of `EvalEnvelope`'s strictness."""
    if not isinstance(payload, dict):
        raise MalformedTurn("the body is not a JSON object")
    if set(payload) != TURN_REQUIRED_FIELDS:
        raise MalformedTurn(
            f"expected exactly {sorted(TURN_REQUIRED_FIELDS)}, got {sorted(payload)}"
        )
    return payload


def last_user_content(messages: object) -> str:
    """The text this stub forwards: the last message's `content`, whatever its `role`.

    The last message is the one being answered - a golden case's `messages` may carry
    earlier turns for context, and this stub answers the newest one."""
    if not isinstance(messages, list) or not messages:
        raise MalformedTurn("messages is not a non-empty list")
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if not isinstance(content, str) or not content:
        raise MalformedTurn("the last message carries no non-empty string content")
    return content


class GatewayReply(NamedTuple):
    """One served answer: the text, and what the gateway said it cost to produce.

    A tuple rather than four return values because `turn_response` hands the whole
    thing to `envelope_for_reply`, and a positional quartet is exactly where
    `provider` and `region` get swapped
    (`test_the_turn_endpoint_publishes_the_measurements_from_the_gateway_reply`).

    `cost_usd` is the header's RAW STRING, parsed once when the envelope is built.
    Carrying it as text this far means an unpriced turn and an unparseable header
    are the same case with the same answer, and neither becomes a `0`
    (`test_an_unusable_cost_still_answers_two_hundred_and_publishes_null`).
    """

    reply: str
    latency_ms: int | None = None
    cost_usd: str | None = None
    provider: str | None = None
    region: str | None = None


def parsed_cost(value: str | None) -> str | None:
    """The cost header as a published decimal STRING, or `None` when there is no
    usable number.

    `None` and never `0`. A turn the ledger did not price, and a header a proxy
    mangled, are both "nothing measured": publishing `0` would make the option's
    USD per 1,000 turns a real-looking number that no provider charged, and
    `tradeoff.py` deliberately leaves a null metric out of the winner computation
    instead of crowning it (criteria 39 and 40,
    `test_an_unusable_cost_still_answers_two_hundred_and_publishes_null`).

    A string out, not a `Decimal`: the envelope is `json.dumps`ed onto the wire, and
    a float there loses the sixth place the ledger prices in."""
    if value is None:
        return None
    try:
        return str(decimal.Decimal(value))
    except decimal.InvalidOperation:
        return None


def envelope_for_reply(
    reply: str,
    *,
    latency_ms: int | None = None,
    cost_usd: str | None = None,
    provider: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Map one completion onto the published `EvalEnvelope` shape (an earlier change
    acceptance criterion 6, an earlier change criteria 36 and 41).

    `decision`, `risk_class`, `tool_calls` and `citations` are stub constants: this
    target answers no tool calls and no citations, and the two judgements are fixed
    rather than classified, because a bare gateway forwarding to one model has
    nothing to classify intent or risk WITH - the module docstring on `target.py` is
    explicit that an adopter's own target owns that judgement.

    The four measurements are keyword-only and default to `None`, so every call site
    written before the tradeoff run keeps its single positional argument (standing
    rule 44), and an unmeasured turn publishes nothing rather than a zero
    (`test_envelope_for_reply_publishes_no_measurements_when_none_are_given`).
    `region` is published for the same reason as the other three: the residency
    check reads it on every measured run."""
    folded = reply.casefold()
    return {
        "decision": "answered",
        "risk_class": "low",
        "tool_calls": [],
        "citations": [],
        "reply": reply,
        "refused": any(keyword in folded for keyword in REFUSAL_KEYWORDS),
        "clarified": any(keyword in folded for keyword in CLARIFY_KEYWORDS),
        "latency_ms": latency_ms,
        "cost_usd": parsed_cost(cost_usd),
        "provider": provider,
        "region": region,
    }


def call_gateway(content: str, *, base_url: str, api_key: str) -> GatewayReply:
    """POST one chat completion to the deployed gateway and return the assistant's
    reply with what the answer cost. The one seam a test replaces to keep the
    envelope-mapping cases off the network (module docstring, "no HTTP, no gateway").

    The latency is measured HERE, around the whole exchange including reading the
    body, because no response header carries it (criterion 38). `monotonic_ns` and
    not `time.time`: a clock the host steps mid-run would otherwise publish a
    negative turn (`test_call_gateway_measures_its_own_latency_in_whole_milliseconds`).
    """
    body = json.dumps({"model": ALIAS, "messages": [{"role": "user", "content": content}]}).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SECONDS) as response:
        raw = response.read()
        cost = response.headers.get(COST_HEADER)
        provider = response.headers.get(PROVIDER_HEADER)
        region = response.headers.get(REGION_HEADER)
    elapsed_ns = time.monotonic_ns() - started
    document = json.loads(raw)
    return GatewayReply(
        reply=str(document["choices"][0]["message"]["content"]),
        latency_ms=elapsed_ns // NANOSECONDS_PER_MILLISECOND,
        cost_usd=cost,
        provider=provider,
        region=region,
    )


def turn_response(
    raw_body: bytes,
    *,
    base_url: str,
    api_key: str,
    caller: Callable[..., GatewayReply] = call_gateway,
) -> tuple[int, dict[str, Any]]:
    """Answer one `/v1/eval/turn` request: the status code and the JSON body to write.

    `caller` is `call_gateway` by default and replaced in a test that scores the
    mapping without a socket; production code never passes it.

    Every field of the reply reaches the envelope. Reading the measurements in
    `call_gateway` and then publishing the text alone is the failure no case on
    either function by itself can see, so it has one here
    (`test_the_turn_endpoint_publishes_the_measurements_from_the_gateway_reply`)."""
    try:
        payload = checked_turn_payload(json.loads(raw_body))
        content = last_user_content(payload["messages"])
    except (MalformedTurn, json.JSONDecodeError) as exc:
        return 400, {"error": str(exc)}
    answer = caller(content, base_url=base_url, api_key=api_key)
    return 200, envelope_for_reply(
        answer.reply,
        latency_ms=answer.latency_ms,
        cost_usd=answer.cost_usd,
        provider=answer.provider,
        region=answer.region,
    )


def reset_response() -> tuple[int, bytes]:
    """Answer one `/admin/test/reset` request: 200 and an empty body. This stub holds
    no seed state, so there is nothing to reset - unlike a Helpdesk-shaped target,
    which resets a real fixture (`target.py`'s own module docstring)."""
    return 200, b""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Routes the two published paths to the pure functions above, and nothing else."""

    server: _Server

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        if self.path == RESET_PATH:
            status, reset_body = reset_response()
            self._respond(status, reset_body)
            return
        if self.path == TURN_PATH:
            status, document = turn_response(
                body,
                base_url=self.server.gateway_base_url,
                api_key=self.server.gateway_api_key,
            )
            self._respond(status, json.dumps(document).encode("utf-8"))
            return
        self._respond(404, b"")

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:  # noqa: ANN401
        sys.stderr.write(f"{self.address_string()} - {format_string % args}\n")


class _Server(http.server.ThreadingHTTPServer):
    """Carries the two settings the handler needs, read once at startup."""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[http.server.BaseHTTPRequestHandler],
        *,
        gateway_base_url: str,
        gateway_api_key: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.gateway_base_url = gateway_base_url
        self.gateway_api_key = gateway_api_key


def main() -> None:
    port = int(os.environ.get("PORT", "8090"))
    base_url = os.environ["GATEWAY_BASE_URL"]
    api_key = os.environ["GATEWAY_API_KEY"]
    server = _Server(
        ("0.0.0.0", port),
        _Handler,
        gateway_base_url=base_url,
        gateway_api_key=api_key,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
