"""Provider pool: a bounded semaphore that keeps slow provider calls off the shared worker pool.

The chat and embeddings routes are `async def`. FastAPI runs async routes on the
event loop directly, so a blocking call inside one would stall every coroutine
waiting for it - the event loop cannot move to the next request while it is
inside a synchronous provider call.

The routes hand each blocking provider call to `asyncio.to_thread`, which runs
it on the event loop's default executor. In CPython that pool has
`min(32, os.cpu_count() + 4)` threads unless the app sets its own, so on a small
machine it is well under 32. The semaphore caps how many provider calls queue
for that pool at once. The sync routes (`/healthz`, `/readyz`, `/metrics`) do not
use it: FastAPI runs them on anyio's worker threads, a separate pool of 40.

What the semaphore does NOT cover: a streamed answer. The route holds it only
while the pipeline runs up to the first event. It is released before the body
is sent, and Starlette then reads the rest of the stream on anyio's pool, the
same 40 threads the sync routes use. So enough slow streams at once can still
fill that pool and delay `/readyz` and `/metrics`.

The semaphore is an `asyncio.Semaphore`, not a `threading.Semaphore`: the route
holds it while the blocking call runs in a thread, and only an async semaphore
can be awaited by the event loop while the thread runs freely.
"""

from __future__ import annotations

import asyncio
from typing import Final

PROVIDER_POOL_SIZE: Final[int] = 32
"""How many provider calls may run at once.

Decided without asking: enough for a healthy burst of concurrent chat
requests, and small enough that a single stalled provider cannot exhaust the
machine's memory. It is not sized against a thread pool: the default executor
may have fewer threads than this, and then the extra calls wait there instead.
"""


def build_provider_limiter() -> asyncio.Semaphore:
    """A fresh semaphore sized to `PROVIDER_POOL_SIZE`, ready for use by the routes.

    Called once in `create_app` and stored on `app.state`; the typed dep in
    `api/deps.py` reads it per request. One semaphore per app, so a test that
    builds two apps cannot let one app's queue block the other's.
    """
    return asyncio.Semaphore(PROVIDER_POOL_SIZE)
