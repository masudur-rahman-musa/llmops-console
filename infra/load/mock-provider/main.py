r"""The load run's provider: OpenAI-compatible, fixed delays, no model.

k6 drives the real gateway at 50 RPS, and a real local model on one box cannot
answer that fast, so the run would measure the model rather than the gateway.
This service stands where llama.cpp stands, answers the three routes the
gateway's `openai_compatible` adapter calls, and takes a FIXED time for each, so
every millisecond above those delays in the k6 summary is the gateway's own.

It shares no code with the gateway: its image installs FastAPI and uvicorn and
nothing else. The test pinning that scans this file for the gateway package's
name, so the name is not written here (`infra/tests/load/test_mock_provider.py`).

Every delay goes through the module-level `sleep`, looked up at call time, which
is the seam the tests replace to assert the delays exactly without waiting
(`test_non_streaming_chat_sleeps_the_chat_delay_exactly_once`,
`test_a_stream_sleeps_the_chunk_delay_once_per_content_chunk`,
`test_embeddings_sleep_the_embed_delay_once_per_request`).

The eval gate runs the same service in golden-set mode. With `MOCK_GOLDEN_SET`
naming a golden `.jsonl`, a structured call (`response_format` of type
`json_schema`) is answered from the golden case the request is about, so the
help desk's own code runs on answers the set itself expects. Without it, every
call gets the fixed reply as before (`backend/tests/evals/test_mock_golden_mode.py`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import struct
import time
import uuid
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Any, Final

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

CHAT_DELAY_S: Final[float] = 0.4
"""One non-streaming completion. Roughly a short answer from a hosted model."""

STREAM_CHUNKS: Final[int] = 10
"""Content chunks per streamed answer, before the usage chunk and `[DONE]`."""

STREAM_CHUNK_DELAY_S: Final[float] = 0.04
"""The gap before each content chunk: ten of them is the same 0.4 s as a whole
non-streaming answer, so the two chat shapes cost the provider the same."""

EMBED_DELAY_S: Final[float] = 0.02
"""One embeddings request, whatever its batch size."""

EMBED_DIM: Final[int] = 384
"""The width of `bge-small-en-v1.5`, which the semantic cache's collection is
built for; a different width would fail the cache write, not the request."""

PORT: Final[int] = 8090

REPLY: Final[str] = "This is a fixed answer from the load-test mock provider."
PROMPT_TOKENS: Final[int] = 24
COMPLETION_TOKENS: Final[int] = 12

GOLDEN_SET_ENV: Final[str] = "MOCK_GOLDEN_SET"
"""Names the golden `.jsonl` that switches golden-set mode on. Read per request,
so the same image serves the load run (unset) and the eval gate (set)."""

TOOL_RESULTS_LABEL: Final[str] = "tool results so far:"
"""How the help desk's planner hands back what earlier calls returned. A call
named there was made, so the next plan leaves it out."""

GOLDEN_CONFIDENCE: Final[float] = 0.95
"""What a golden-mode intent answer claims: above any sane clarify cut-off, so
the help desk acts on the expected intent rather than asking about it."""

ENTITY_FIELDS: Final[tuple[str, ...]] = (
    "system",
    "role",
    "target_user",
    "department",
    "expiry_days",
    "justification",
)
"""The entities an intent answer carries."""

DAYS: Final[re.Pattern[str]] = re.compile(r"\b(\d{1,3})\s*days?\b", re.IGNORECASE)
"""A length the latest message states ("For 30 days")."""

REASON: Final[re.Pattern[str]] = re.compile(
    r"\b(?:because|so (?:that )?I can|I need it for|it is for|for my)\s+(.+?)[.!?]?$",
    re.IGNORECASE,
)
"""A reason the latest message states ("so I can follow the runbooks")."""

PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"<[a-z_]+_[0-9]+>", re.IGNORECASE)
"""The gateway's redaction placeholder (`<email_1>`). A redacted message still
matches its golden line, with each placeholder standing for any text."""

sleep = asyncio.sleep

app = FastAPI(title="load mock provider")


class ChatBody(BaseModel):
    """Only what this mock reads is required. The adapter sends more
    (`max_tokens`, `stream_options`), and it is ignored rather than refused."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Any]
    stream: bool = False
    response_format: dict[str, Any] | None = None


class EmbedBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[str]


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }


def _vector(text: str) -> list[float]:
    """A unit vector derived from the text alone: the same text always embeds to
    the same vector, so a repeated prompt is a semantic-cache hit and a unique one
    is not (`test_the_same_text_embeds_to_the_same_vector_and_different_texts_do_not`)."""
    raw = b""
    counter = 0
    while len(raw) < EMBED_DIM * 4:
        raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
        counter += 1
    values = [
        (word / 0xFFFFFFFF) * 2.0 - 1.0
        for word in struct.unpack(f"<{EMBED_DIM}I", raw[: EMBED_DIM * 4])
    ]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@cache
def _golden_cases(path: str) -> tuple[dict[str, Any], ...]:
    """Every case in the golden file, in file order, read once per path."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


def _user_texts(messages: list[Any]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]


_last_case: dict[str, Any] = {}
"""The case the last golden answer came from. The eval runs one case at a time,
and a short reply ("Yes, that is right.") appears in many cases, so a tie goes
to the case this conversation was already about."""


def _same_text(golden: str, sent: str) -> bool:
    """True when `sent` is the golden text, or the golden text after redaction."""
    if golden == sent:
        return True
    if PLACEHOLDER.search(sent) is None:
        return False
    parts = PLACEHOLDER.split(sent)
    pattern = ".+?".join(re.escape(part) for part in parts)
    return re.fullmatch(pattern, golden, re.DOTALL) is not None


def _matching_case(cases: tuple[dict[str, Any], ...], messages: list[Any]) -> dict[str, Any] | None:
    """The case whose user messages include the request's last user message.

    Among several, the one with the most of its user messages found anywhere in
    the request (the history the intent step sends), then the case answered
    last, then the first in the file.
    """
    users = _user_texts(messages)
    if not users:
        return None
    last = users[-1]
    whole = "\n".join(users)
    found = [
        case
        for case in cases
        if any(_same_text(text, last) for text in _user_texts(case.get("messages", [])))
    ]
    if not found:
        return None

    def score(case: dict[str, Any]) -> tuple[int, bool]:
        seen = sum(1 for text in _user_texts(case["messages"]) if text in whole)
        return seen, case.get("id") == _last_case.get("id")

    best = max(found, key=score)
    _last_case.clear()
    _last_case.update(best)
    return best


def _labelled(messages: list[Any]) -> dict[str, Any]:
    """The help desk planner's labelled data messages, by label (`label:\n<json>`)."""
    found: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        label, _, rest = str(message.get("content", "")).partition(":\n")
        try:
            found[label] = json.loads(rest)
        except ValueError:
            continue
    return found


def _stand_in(schema: dict[str, Any], latest: str) -> object:
    """A value for a required argument nothing else filled: the enum's first
    word, or the latest message for free text (a ticket's summary), cut to fit."""
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    if schema.get("type") == "string" and "format" not in schema:
        limit = schema.get("maxLength")
        return latest[:limit] if isinstance(limit, int) else latest
    return None


def _plan(case: dict[str, Any], messages: list[Any]) -> list[dict[str, Any]]:
    """The next round of calls: the expected calls not yet made, or none.

    Done once a `tool` message is in the request, or once every expected call
    shows in the planner's tool results. The arguments start from the case's
    `args_subset`, fitted to the tool's schema the way a model would: a
    `system.role` name becomes the resolved `role_id`, the caller's id comes
    from `get_me` (asked for first when the tool needs it), and the length and
    reason come from the extracted entities.
    """
    if any(isinstance(m, dict) and m.get("role") == "tool" for m in messages):
        return []
    data = _labelled(messages)
    results = [r for r in data.get(TOOL_RESULTS_LABEL[:-1], []) if isinstance(r, dict)]
    ran = {r.get("tool") for r in results}
    pending = [call for call in _expected_calls(case) if call.get("name") not in ran]
    schemas = data.get("allowed tools and their argument schemas", {})
    resolved = data.get("resolved directory ids", {}) or {}
    entities = (data.get("intent and entities", {}) or {}).get("entities", {}) or {}
    users = _user_texts(messages)
    latest = users[-1] if users else ""
    me = next(
        (r["payload"].get("user_id") for r in results if r.get("tool") == "get_me" and r.get("ok")),
        None,
    )
    calls: list[dict[str, Any]] = []
    for call in pending:
        args = dict(call.get("args_subset", {}))
        properties = schemas.get(call["name"], {}).get("properties")
        if isinstance(properties, dict):
            if "target_user_id" in properties and me is None and "get_me" in schemas:
                return [{"tool": "get_me", "arguments": "{}"}]
            filled = {
                "role_id": resolved.get("role_id"),
                "system_id": resolved.get("system_id"),
                "target_user_id": me,
                "justification": entities.get("justification"),
                "requested_expiry_days": entities.get("expiry_days"),
            }
            for name, value in filled.items():
                if name in properties and name not in args and value is not None:
                    args[name] = value
            required = schemas[call["name"]].get("required", [])
            for name in required:
                if name not in args:
                    args[name] = _stand_in(properties.get(name, {}), latest)
            args = {
                name: value
                for name, value in args.items()
                if name in properties and value is not None
            }
        calls.append({"tool": call["name"], "arguments": json.dumps(args)})
    return calls


def _expected_calls(case: dict[str, Any]) -> list[dict[str, Any]]:
    calls = case.get("expected", {}).get("tool_calls", [])
    return [call for call in calls if isinstance(call, dict)]


def _entities(case: dict[str, Any], latest: str) -> dict[str, Any]:
    """What a model would read off the latest message, taken from the case.

    The system and role come from the first expected call's `role`
    (`<system>.<role>`) or `system`, so the help desk can look them up; the
    length and the reason are read from the latest message alone, the way
    the help desk's extraction prompt asks for them.
    """
    calls = _expected_calls(case)
    args: dict[str, Any] = calls[0].get("args_subset", {}) if calls else {}
    entities: dict[str, Any] = dict.fromkeys(ENTITY_FIELDS)
    role = args.get("role")
    if isinstance(role, str) and "." in role:
        entities["system"], entities["role"] = role.split(".", 1)
    elif isinstance(args.get("system"), str):
        entities["system"] = args["system"]
    entities["department"] = args.get("department")
    days = DAYS.search(latest)
    entities["expiry_days"] = args.get("requested_expiry_days") or (
        int(days.group(1)) if days else None
    )
    reason = REASON.search(latest)
    entities["justification"] = reason.group(1) if reason else None
    return entities


def _golden_answer(schema: str, case: dict[str, Any], messages: list[Any]) -> dict[str, Any]:
    """The structured answer one schema gets for one case."""
    if schema == "IntentExtraction":
        users = _user_texts(messages)
        return {
            "intent": case.get("expected", {}).get("intent"),
            "entities": _entities(case, users[-1] if users else ""),
            "confidence": GOLDEN_CONFIDENCE,
            "clarifying_question": None,
            "second_request": None,
        }
    if schema == "ToolPlanAnswer":
        return {"calls": _plan(case, messages)}
    return {"text": REPLY, "asks_question": False, "used_passages": [1]}


def _golden_reply(body: ChatBody) -> dict[str, Any] | JSONResponse | None:
    """The golden-mode answer's content, a 400, or `None` for "not golden mode".

    Only `json_schema` calls are answered from the set; a plain chat call keeps
    the fixed reply. The intent and plan steps need the matching case and are a
    400 naming the schema without one, because a canned answer would let the
    gate score something no case asked for. Any other schema (the reply
    writer's) gets a fixed answer of its shape.
    """
    path = os.environ.get(GOLDEN_SET_ENV)
    response_format = body.response_format or {}
    if not path or response_format.get("type") != "json_schema":
        return None
    schema = str(response_format.get("json_schema", {}).get("name", ""))
    case = _matching_case(_golden_cases(path), body.messages)
    if case is None and schema in {"IntentExtraction", "ToolPlanAnswer"}:
        return JSONResponse(
            status_code=400,
            content={"error": f"no golden case matches this {schema} request"},
        )
    return _golden_answer(schema, case or {}, body.messages)


@app.post("/v1/chat/completions", response_model=None)
async def chat(body: ChatBody) -> dict[str, Any] | JSONResponse | StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if body.stream:
        return StreamingResponse(
            _stream(completion_id, created, body.model), media_type="text/event-stream"
        )
    golden = _golden_reply(body)
    if isinstance(golden, JSONResponse):
        return golden
    content = REPLY if golden is None else json.dumps(golden)
    await sleep(CHAT_DELAY_S)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(),
    }


async def _stream(completion_id: str, created: int, model: str) -> AsyncIterator[str]:
    """Ten content chunks, the last carrying `finish_reason`, then the usage chunk
    with empty `choices` that the adapter requires, then `[DONE]`."""
    envelope = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    for position in range(STREAM_CHUNKS):
        await sleep(STREAM_CHUNK_DELAY_S)
        last = position == STREAM_CHUNKS - 1
        chunk = {
            **envelope,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"part {position} "},
                    "finish_reason": "stop" if last else None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield f"data: {json.dumps({**envelope, 'choices': [], 'usage': _usage()})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings")
async def embeddings(body: EmbedBody) -> dict[str, Any]:
    texts = [body.input] if isinstance(body.input, str) else body.input
    await sleep(EMBED_DELAY_S)
    return {
        "object": "list",
        "model": body.model,
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(text)}
            for index, text in enumerate(texts)
        ],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "total_tokens": PROMPT_TOKENS},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
