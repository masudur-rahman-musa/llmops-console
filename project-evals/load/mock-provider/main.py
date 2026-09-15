r"""The load run's provider: OpenAI-compatible, fixed delays, no model.

k6 drives the real gateway at 50 RPS, and a real local model on one box cannot
answer that fast, so the run would measure the model rather than the gateway.
This service stands where llama.cpp stands, answers the three routes the
gateway's `openai_compatible` adapter calls, and takes a FIXED time for each, so
every millisecond above those delays in the k6 summary is the gateway's own.

It shares no code with the gateway: its image installs FastAPI and uvicorn and
nothing else. The test pinning that scans this file for the gateway package's
name, so the name is not written here (`tests/load/test_mock_provider.py`).

Every delay goes through the module-level `sleep`, looked up at call time, which
is the seam the tests replace to assert the delays exactly without waiting
(`test_non_streaming_chat_sleeps_the_chat_delay_exactly_once`,
`test_a_stream_sleeps_the_chunk_delay_once_per_content_chunk`,
`test_embeddings_sleep_the_embed_delay_once_per_request`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Final

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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

sleep = asyncio.sleep

app = FastAPI(title="load mock provider")


class ChatBody(BaseModel):
    """Only what this mock reads is required. The adapter sends more
    (`max_tokens`, `stream_options`), and it is ignored rather than refused."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Any]
    stream: bool = False


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


@app.post("/v1/chat/completions", response_model=None)
async def chat(body: ChatBody) -> dict[str, Any] | StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if body.stream:
        return StreamingResponse(
            _stream(completion_id, created, body.model), media_type="text/event-stream"
        )
    await sleep(CHAT_DELAY_S)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
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
