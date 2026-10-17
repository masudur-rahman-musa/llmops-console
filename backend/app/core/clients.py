"""The bounded clients the readiness probes and the semantic cache open.

Every client built here carries `PROBE_TIMEOUT_SECONDS` on both the connect and
the read, because a backing service that accepts a connection and then never
answers is the shape that hangs readiness itself and takes a healthy pod out of
rotation for the wrong reason.

The builders live here rather than beside the probes in `api/routes/health.py`
because the semantic cache opens a Qdrant client over the same bound without
being a probe, so the two timeouts cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Final, Protocol

import httpx
import psycopg
import redis

PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
"""How long any one readiness check may take, connecting and reading, in seconds.

Kubernetes gives a
readiness probe its own timeout as well; this one exists so the process cannot be
the thing that hangs."""


class SupportsPing(Protocol):
    """The one method the Redis probe is allowed to use.

    Narrow on purpose: the probe takes a client rather than a URL so a test can
    hand it a double, and a protocol with one method is what stops the probe
    growing a second call that a double would then have to reimplement.
    """

    def ping(self) -> object: ...


class SupportsExecute(Protocol):
    """A database connection that runs a statement and releases itself.

    Structural, so `psycopg.Connection` satisfies it without this module
    depending on that class in its signatures, and a recording double satisfies
    it without a live server.
    """

    def execute(self, statement: str) -> object: ...

    def __enter__(self) -> SupportsExecute: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


def build_redis_client(url: str) -> redis.Redis:
    """A Redis client for `url`, bounded on both the connect and the read.

    `socket_timeout` is the half that matters and the half a library default
    leaves unset: a server that accepts the connection and never answers is what
    hangs a probe.

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, and a parametrised annotation fails `mypy --strict` with `type-arg`.
    """
    return redis.Redis.from_url(
        url,
        socket_connect_timeout=PROBE_TIMEOUT_SECONDS,
        socket_timeout=PROBE_TIMEOUT_SECONDS,
    )


def build_qdrant_client() -> httpx.Client:
    """An `httpx.Client` whose four timeout phases all carry `PROBE_TIMEOUT_SECONDS`.

    `httpx.Timeout(x)` sets connect, read, write and pool together; the library
    default bounds the connect at 5s and leaves the read unbounded, which is the
    hang this exists to prevent.
    """
    return httpx.Client(timeout=httpx.Timeout(PROBE_TIMEOUT_SECONDS))


def build_postgres_connect(url: str) -> Callable[[], SupportsExecute]:
    """A callable opening one bounded connection to `url`.

    Two bounds, because libpq has two. `connect_timeout` covers the handshake and
    is whole seconds, so the float is floored to an int here rather than silently
    truncated by the driver. `statement_timeout` covers the query, in
    milliseconds, and is what stops `SELECT 1` waiting forever on a server that
    connected and then stalled.
    """
    connect_seconds = max(1, int(PROBE_TIMEOUT_SECONDS))
    statement_ms = int(PROBE_TIMEOUT_SECONDS * 1000)

    def connect() -> SupportsExecute:
        return psycopg.connect(
            url,
            connect_timeout=connect_seconds,
            options=f"-c statement_timeout={statement_ms}",
        )

    return connect
