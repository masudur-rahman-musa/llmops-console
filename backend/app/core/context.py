"""The request context: the identity and labels every stage of a request reads.

Kept here so the service layer can import it without reaching into `app.api`.
The auth dependency in `app.api.deps` builds and returns it; the services that
use it import it from here, and nothing in them has to know `app.api` exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.model_routing.policy import DataClass


class RequestContext(BaseModel):
    """The identity and labels every later stage of a request reads.

    No user identity, by construction: a tenant, an optional session and a trace
    id are the only things here, which is what keeps telemetry free of it.
    `test_context_carries_the_authenticated_tenant` compares the field set
    exactly, so a field added here goes red rather than quietly reaching a trace;
    that red is the invariant, not a stale expectation to update.
    """

    model_config = ConfigDict(frozen=True)

    tenant: str
    data_class: DataClass
    session: str | None
    trace_id: str
