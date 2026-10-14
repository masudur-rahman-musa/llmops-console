"""Liveness and readiness.

`/healthz` says the process is up and checks nothing else, so a dependency
outage never takes the container down (pinned by `test_registries_are_per_app`,
which asserts 200 from `/healthz` while `/readyz` is 503). `/readyz` runs the registered probes and
returns 503 the moment one of them fails, which is what takes the pod out of
the load balancer.

What goes into the registry is `gateway/probes.py`'s business: this module owns
the registry, the endpoints and the rule that a probe which raises has failed,
and knows nothing about Redis, Postgres, Qdrant or the adapters. `register_probes`
files the configured ones during `create_app`, so an app built with no backing
service configured carries exactly one probe, `provider`
(`test_register_probes_registers_the_configured_stores_and_always_the_provider`).

The body names every probe that RAN, which is the half of the contract an empty
registry makes visible: `{"checks": {}}` is a readiness that checked nothing, and
an operator can see that rather than having to infer it from a green 200
(`test_a_refused_redis_makes_readyz_503_naming_every_probe`).

Beside `checks` it carries `residency_mode`, so a process running the opt-in
relaxed mode cannot be mistaken for a strict one by anything that reads the
endpoint. A top-level key rather than another entry in
`checks`, because `checks` means "a probe ran and this is its verdict" and
readiness is `all(checks.values())` - a string filed in there would read as a
passing probe (`test_readyz_reports_the_residency_mode`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.routing.policy import ResidencyMode

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
    and `test_readyz_503_when_a_probe_fails`. The body also states this process's
    `residency_mode`, which is settings and not a probe, so it never affects the
    status (`test_readyz_reports_the_residency_mode`).
    """
    registry: ProbeRegistry = request.app.state.probes
    mode: ResidencyMode = request.app.state.settings.residency_mode
    checks = {name: _run(probe) for name, probe in registry}
    ready = all(checks.values())
    return JSONResponse(
        status_code=READY_STATUS if ready else NOT_READY_STATUS,
        # The mode is reported at both statuses: a relaxed process that is also
        # unready is exactly the one nobody should mistake for strict.
        content={"checks": checks, "residency_mode": mode.value},
    )
