"""The request context: the identity and labels every stage of a request reads.

Kept here so the service layer can import it without reaching into `app.api`.
The auth dependency in `app.api.deps` builds and returns it; the services that
use it import it from here, and nothing in them has to know `app.api` exists.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DataClass(StrEnum):
    """The sensitivity classes a request can declare. A closed set on purpose.

    Kept in `core` so the request context can name it without importing the
    routing package; `app.model_routing.policy` imports it from here.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"


class RequestContext(BaseModel):
    """The identity and labels every later stage of a request reads.

    No user identity, by construction: a tenant, an optional session, a trace id,
    an optional prompt version and an optional random user id are the only things
    here. The user id is `u_` and 32 hex characters, checked in `app.api.deps`, and
    only the help desk can map it to a person.
    A test compares the field set exactly, so a field added here goes red rather
    than quietly reaching a trace; that red is intended, not a stale expectation
    to update.
    """

    model_config = ConfigDict(frozen=True)

    tenant: str
    data_class: DataClass
    session: str | None
    trace_id: str
    prompt_version: str | None = None
    user_id: str | None = None
