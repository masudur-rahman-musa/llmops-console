"""Process settings, read from the environment at construction time."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROUTING_POLICY_PATH = Path(__file__).resolve().parents[4] / "routing.yaml"
"""The shipped `routing.yaml`, found from this file rather than from the working
directory. A deployment that does not carry the repository layout - the container
image, a wheel install - sets `OPSCONSOLE_ROUTING_POLICY_PATH` instead."""

DEFAULT_PRICES_PATH = Path(__file__).resolve().parents[4] / "prices.yaml"
"""The shipped `prices.yaml`, found the same way and for the same reason, with
`OPSCONSOLE_PRICES_PATH` as the escape hatch."""


class Settings(BaseSettings):
    """Gateway settings.

    Every field has a default, so the process starts with no environment at all.
    Values come from `OPSCONSOLE_`-prefixed variables; presence decides an
    override, not truthiness, so `OPSCONSOLE_ENV=""` yields an empty `env`.
    Pinned by `test_settings_env_override`.
    """

    model_config = SettingsConfigDict(env_prefix="OPSCONSOLE_")

    env: str = "local"
    port: int = 8080

    api_keys: str = ""
    """Comma-separated `tenant:sha256hex` pairs. Digests only: an issued tenant
    key lives in `.env` locally and in a SOPS-encrypted secret in the cluster, and
    no issued key is in this repository (the worthless strings in
    `tests/gateway/conftest.py` are test data, not anyone's key). Empty is a
    running gateway that authorises nobody."""

    routing_policy_path: Path = DEFAULT_ROUTING_POLICY_PATH

    prices_path: Path = DEFAULT_PRICES_PATH
    """Where the price table is read from. Nothing puts it on `app.state` yet:
    an earlier change is what gives the gateway a `PriceStore` and a ledger row to spend
    it on. The field is here now because the loader and the shipped file are,
    and a path setting that arrives with its consumer is one more thing to get
    right in the plan that has the harder work."""

    admin_key_sha256: str = ""
    """The sha256 digest of the one admin key, or empty for a gateway whose
    `/admin/*` routes authorise nobody.

    A digest rather than the key itself: the tenant keys
    above are already hashed at rest, and a plaintext admin key in an environment
    variable is the one form that also leaks through a process dump and a
    `docker inspect`. Issuing a key is sha256-ing it once. This is a SEPARATE key
    space from `api_keys`: a valid tenant key is 401 on `/admin/*`, pinned by
    `test_admin_routes_reject_a_tenant_api_key`, and the shipped empty default
    denies every caller, pinned by
    `test_admin_routes_reject_every_caller_when_no_admin_key_is_configured`."""

    otlp_endpoint: str | None = None
    """Where traces are sent, or `None` for a gateway that builds spans and drops
    them. Langfuse EU is an OTLP/HTTP endpoint, so this is a
    full traces URL rather than a host. Absent by default so a developer, a test
    run and a CI job export nothing without saying so; pinned by
    `test_no_exporter_is_installed_without_an_endpoint`."""

    otlp_headers: str | None = None
    """`key=value,key=value`, the auth the endpoint needs. Langfuse wants a basic
    `Authorization` header built from a project key pair, which is a secret: it
    lives in `.env` locally and in a SOPS-encrypted secret in the cluster, never
    in this repository."""

    redis_url: str | None = None
    """Where the shared Redis lives, or `None` for a gateway that keeps its
    redaction maps, its tenant directories and its response cache in process
    memory and registers no Redis readiness probe.

    Two modules read it, and both read absence as "not configured" rather than as
    a default (`grep -rn "\\.redis_url" packages/ services/`, excluding this
    file, answers `gateway/app.py` and `gateway/probes.py`). `probes.py`
    registers the `redis` probe only when it is set.
    `create_app` builds ONE client from it and hands it to three seams, each of
    which falls back to an in-process twin when it is absent:
    `RedisRedactionMapStore`, `RedisDirectoryStore` (plan
    12a) and `RedisResponseCache`. Absent by default so the
    suite and a bare local run need no backing service; pinned by
    `test_a_backing_service_url_is_absent_until_the_environment_sets_it`,
    `test_create_app_keeps_the_in_process_store_without_a_redis_url`,
    `test_create_app_uses_the_redis_store_when_a_redis_url_is_configured`,
    `test_create_app_keeps_the_in_process_cache_without_a_redis_url` and
    `test_create_app_uses_the_redis_cache_when_a_redis_url_is_configured`."""

    database_url: str | None = None
    """Where PostgreSQL lives, or `None` for no `postgres` readiness probe.

    Nothing else reads it yet: the ledger that will is an earlier change. A probe for a
    store nothing writes to is still worth having, because #5 names Postgres and
    a readiness endpoint that silently omits a configured dependency is a
    readiness endpoint that lies."""

    qdrant_url: str | None = None
    """Where Qdrant lives, or `None` for the in-process semantic cache and no probe.

    Two modules read it now (`grep -rn "\\.qdrant_url" packages/ services/`,
    excluding this file, answers `gateway/app.py` and `gateway/probes.py`).
    Probed over `httpx` rather than through a client library: a probe is one GET against `QDRANT_READY_PATH`. Set,
    it also builds `QdrantSemanticCache` for `app.state.semantic_cache`, so every
    replica searches the same vectors; absent leaves `InProcessSemanticCache`,
    which is what lets the suite and a bare local run take the semantic path with
    no Qdrant anywhere."""

    otel_service_name: str = "opsconsole-gateway"
    """What this process is called in a trace viewer. One trace carries spans from
    the Helpdesk and from the gateway, so the two need different names to be
    told apart at all."""
