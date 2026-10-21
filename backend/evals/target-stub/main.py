r"""The kind e2e's eval target: the help desk's conversation contract, played from a small
case file instead of a real agent.

Stdlib only (`http.server`, `urllib.request`), so the throwaway image installs nothing. The contract is duplicated here, never imported: the stub's image carries
none of `opsconsole`.

Two endpoints, both POST:

    /admin/test/reset         answered 204 with an empty body; this stub holds no seed
    /v1/eval/conversation     {case_id, tenant, principal_user, turns} in, one record per
                              user turn out: {case_id, turns: [{turn, reply, tool_calls}]}

The stub reads the case file (`CASES_PATH`, default `/cases/kind-cases.jsonl`, the same
file `ops-eval conversations --cases` plays) and answers each case from its `expected`
block: the case's expected tool calls are reported as made, each one on the turn the
person says yes (`confirm_turn`) or on turn 1 when nothing may change. That is what lets
the kind run pass without a real agent.

The reply text of a turn is one of two things. With `FORWARD_TO_GATEWAY` true (the
default), the turn's message goes to the deployed gateway's `POST /v1/chat/completions`,
alias `agent-default`, tenant `evals`, with no `X-Data-Class` header, so the measurement
run writes real ledger rows and the answer is the model's own text. With it `false`
(the kind cluster, which has no model), the reply is a fixed sentence that carries any
request number the stub's own tool result holds.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, NamedTuple

RESET_PATH: Final[str] = "/admin/test/reset"
CONVERSATION_PATH: Final[str] = "/v1/eval/conversation"

REQUEST_FIELDS: Final[frozenset[str]] = frozenset({"case_id", "tenant", "principal_user", "turns"})
"""Exactly the keys the console sends, no more and no fewer."""

ALIAS: Final[str] = "agent-default"
"""The `routing.yaml` alias this stub asks the gateway to resolve."""

NANOSECONDS_PER_MILLISECOND: Final[int] = 1_000_000

GATEWAY_TIMEOUT_SECONDS: Final[float] = 30.0
"""How long one forwarded chat completion may take. The 3B local model is the whole
budget: a hang here is one case timing out rather than the stub itself."""


CHANGE_RESULTS: Final[dict[str, dict[str, Any]]] = {
    "create_access_request": {"request_number": "REQ-0001", "status": "pending"},
}
"""What the stub says a tool returned. A tool not listed returns an empty result."""


class MalformedRequest(Exception):
    """The body is not exactly the conversation request. Answered as one line and a 400."""


class UnknownCase(Exception):
    """The case id is not in the case file. Answered as one line and a 404."""


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    """Read the case file once: one JSON object per line, keyed by `id`."""
    cases: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            cases[str(case["id"])] = case
    return cases


def checked_request(payload: object) -> dict[str, Any]:
    """Refuse a body that is not a JSON object with exactly the four request fields."""
    if not isinstance(payload, dict):
        raise MalformedRequest("the body is not a JSON object")
    if set(payload) != REQUEST_FIELDS:
        raise MalformedRequest(f"expected exactly {sorted(REQUEST_FIELDS)}, got {sorted(payload)}")
    turns = payload["turns"]
    if not isinstance(turns, list) or not turns or not all(isinstance(t, str) and t for t in turns):
        raise MalformedRequest("turns is not a non-empty list of text")
    return payload


class GatewayReply(NamedTuple):
    """One served answer: the text, and the time the whole exchange took."""

    reply: str
    latency_ms: int


def call_gateway(content: str, *, base_url: str, api_key: str) -> GatewayReply:
    """POST one chat completion to the deployed gateway and return the assistant's reply.
    The one seam a test replaces to keep the conversation cases off the network."""
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
    elapsed_ns = time.monotonic_ns() - started
    document = json.loads(raw)
    return GatewayReply(
        reply=str(document["choices"][0]["message"]["content"]),
        latency_ms=elapsed_ns // NANOSECONDS_PER_MILLISECOND,
    )


def calls_for_turn(case: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    """The expected tool calls the stub reports as made on this turn."""
    expected = case["expected"]
    confirm = expected.get("confirm_turn")
    call_turn = 1 if confirm is None else int(confirm)
    if turn != call_turn:
        return []
    return [
        {
            "name": call["name"],
            "args": dict(call.get("args_subset") or {}),
            "ok": True,
            "result": dict(CHANGE_RESULTS.get(call["name"], {})),
            "error_code": None,
        }
        for call in expected["tool_calls"]
    ]


def canned_reply(calls: list[dict[str, Any]]) -> str:
    """A fixed sentence that repeats only a request number a tool result holds."""
    numbers = [str(c["result"]["request_number"]) for c in calls if "request_number" in c["result"]]
    if numbers:
        return f"Done. Your request number is {numbers[0]}."
    return "Understood."


def conversation_response(
    raw_body: bytes,
    *,
    cases: dict[str, dict[str, Any]],
    base_url: str | None,
    api_key: str,
    caller: Callable[..., GatewayReply] = call_gateway,
) -> tuple[int, dict[str, Any]]:
    """Answer one `/v1/eval/conversation` request: the status code and the JSON body.

    `base_url` is `None` when the gateway is not forwarded to. `caller` is replaced in a
    test that needs no socket; production code never passes it."""
    try:
        payload = checked_request(json.loads(raw_body))
        case = cases.get(str(payload["case_id"]))
        if case is None:
            raise UnknownCase(f"no case {payload['case_id']} in the case file")
    except (MalformedRequest, json.JSONDecodeError) as exc:
        return 400, {"error": str(exc)}
    except UnknownCase as exc:
        return 404, {"error": str(exc)}
    turns: list[dict[str, Any]] = []
    for number, text in enumerate(payload["turns"], start=1):
        calls = calls_for_turn(case, number)
        if base_url is None:
            reply = canned_reply(calls)
        else:
            reply = caller(text, base_url=base_url, api_key=api_key).reply
        turns.append({"turn": number, "reply": reply, "tool_calls": calls})
    return 200, {"case_id": payload["case_id"], "turns": turns}


def reset_response() -> tuple[int, bytes]:
    """Answer one `/admin/test/reset` request: 204 and an empty body."""
    return 204, b""


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
        if self.path == CONVERSATION_PATH:
            try:
                status, document = conversation_response(
                    body,
                    cases=self.server.cases,
                    base_url=self.server.gateway_base_url,
                    api_key=self.server.gateway_api_key,
                )
            except Exception as exc:  # one failed case must not stop the server
                status, document = 502, {"error": f"{type(exc).__name__}: {exc}"}
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
    """Carries the settings the handler needs, read once at startup."""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[http.server.BaseHTTPRequestHandler],
        *,
        cases: dict[str, dict[str, Any]],
        gateway_base_url: str | None,
        gateway_api_key: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.cases = cases
        self.gateway_base_url = gateway_base_url
        self.gateway_api_key = gateway_api_key


def main() -> None:
    port = int(os.environ.get("PORT", "8090"))
    cases = load_cases(Path(os.environ.get("CASES_PATH", "/cases/kind-cases.jsonl")))
    forward = os.environ.get("FORWARD_TO_GATEWAY", "true").lower() == "true"
    server = _Server(
        ("0.0.0.0", port),
        _Handler,
        cases=cases,
        gateway_base_url=os.environ["GATEWAY_BASE_URL"] if forward else None,
        gateway_api_key=os.environ.get("GATEWAY_API_KEY", ""),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
