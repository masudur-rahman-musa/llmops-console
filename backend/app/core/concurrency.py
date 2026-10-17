"""Provider pool: a bounded semaphore that keeps slow provider calls off the shared worker pool.

The chat and embeddings routes are `async def`. FastAPI runs async routes on the
event loop directly, so a blocking call inside one would stall every coroutine
waiting for it - the event loop cannot move to the next request while it is
inside a synchronous provider call.

The routes hand each blocking provider call to `asyncio.to_thread`, which runs
it on `concurrent.futures.ThreadPoolExecutor` threads, and cap concurrency with
a semaphore so a slow or overloaded provider fills only this pool and never the
shared one. One slow provider may hold 32 threads here; it cannot hold the event
loop or the threads that serve `/readyz`, `/metrics` and everything else.

The semaphore is an `asyncio.Semaphore`, not a `threading.Semaphore`: the route
holds it while the blocking call runs in a thread, and only an async semaphore
can be awaited by the event loop while the thread runs freely.
"""

from __future__ import annotations

import asyncio
from typing import Final

PROVIDER_POOL_SIZE: Final[int] = 32
"""How many provider calls may run at once.

Decided without asking: 32 leaves headroom under the shared pool's
default of 40 thread-pool workers, absorbs a healthy burst of concurrent chat
requests, and is small enough that a single stalled provider cannot exhaust the
machine's memory.
"""


def build_provider_limiter() -> asyncio.Semaphore:
    """A fresh semaphore sized to `PROVIDER_POOL_SIZE`, ready for use by the routes.

    Called once in `create_app` and stored on `app.state`; the typed dep in
    `api/deps.py` reads it per request. One semaphore per app, so a test that
    builds two apps cannot let one app's queue block the other's.
    """
    return asyncio.Semaphore(PROVIDER_POOL_SIZE)
