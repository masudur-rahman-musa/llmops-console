"""ASGI entrypoint.

`uvicorn app.main:app` for a supervisor that owns the port;
`serve()` when the port should come from settings.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from app.core.settings import Settings
from app.gateway.app import create_app

app: FastAPI = create_app()


def serve(reload: bool = False) -> None:
    """Run the gateway on the configured port.

    The listening port comes from `Settings`, so `OPSCONSOLE_PORT` reaches the
    socket. Pinned by `test_serve_uses_the_configured_port`. Clients that dial
    the gateway (`make eval`, the compose healthcheck) carry their own address
    and are not covered by that test.
    """
    uvicorn.run("app.main:app", port=Settings().port, reload=reload)
