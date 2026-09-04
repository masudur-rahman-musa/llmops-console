"""Liveness and readiness.

`/healthz` says the process is up and checks nothing else, so a dependency
outage never takes the container down (pinned by `test_registries_are_per_app`,
which asserts 200 from `/healthz` while `/readyz` is 503). `/readyz` runs the registered probes and
returns 503 the moment one of them fails, which is what takes the pod out of
the load balancer. The registry is empty in this plan; the Redis, Postgres,
Qdrant and provider probes are registered by a later plan.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

Probe = Callable[[], bool]

READY_STATUS = 200
NOT_READY_STATUS = 503


class ProbeRegistry:
    """The readiness probes an app checks.

    Iteration follows registration order, and a second registration under an
    existing name replaces the first. Both are pinned by
    `test_registering_the_same_name_replaces_the_probe`.
    """

    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}

    def register(self, name: str, fn: Probe) -> None:
        """Register `fn` under `name`, replacing any probe already at that name."""
        self._probes[name] = fn

    def __iter__(self) -> Iterator[tuple[str, Probe]]:
        return iter(self._probes.items())


def _run(probe: Probe) -> bool:
    """Run one probe. A probe that raises is a probe that failed."""
    try:
        return bool(probe())
    except Exception:
        # `/readyz` only ever reports a bool, so the cause has to be logged here
        # or it is lost to whoever is looking at a pod that will not go ready.
        logger.exception("readiness probe raised")
        return False


router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: 200 as long as the process is serving. Pinned by `test_healthz_ok`."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> Response:
    """Readiness: 200 when every registered probe passes, 503 naming the failures.

    An empty registry is ready. Pinned by `test_readyz_ok_when_all_probes_pass`
    and `test_readyz_503_when_a_probe_fails`.
    """
    registry: ProbeRegistry = request.app.state.probes
    checks = {name: _run(probe) for name, probe in registry}
    ready = all(checks.values())
    return JSONResponse(
        status_code=READY_STATUS if ready else NOT_READY_STATUS,
        content={"checks": checks},
    )
