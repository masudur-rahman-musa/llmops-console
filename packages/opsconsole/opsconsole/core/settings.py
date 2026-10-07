"""Process settings, read from the environment at construction time."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from opsconsole.routing.policy import ResidencyMode

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

    residency_mode: ResidencyMode = ResidencyMode.STRICT
    """How strictly this process enforces residency: `strict` or `relaxed`.

    `strict` is the default, so a deployment that sets nothing enforces residency
    exactly as the product always has
    (`test_the_residency_mode_defaults_to_strict`). `relaxed` is an opt-in local
    demo mode that lets `resolve()` reach `resolver.RELAXED_REGIONS` for a data
    class that has a rule and is not `pii`.

    A closed enum rather than a string, so an unknown value is refused by
    `Settings()` itself and the process never serves one request under a mode
    nobody meant (`test_an_unknown_residency_mode_is_refused_at_construction`,
    `test_create_app_refuses_an_unknown_residency_mode`). `create_app` logs one
    WARNING when it is relaxed and `/readyz` reports it, because a relaxed
    process that looks strict from outside is the real hazard here
    (`test_create_app_warns_once_under_relaxed_residency`,
    `test_readyz_reports_a_relaxed_residency_mode`). Nothing under `infra/` sets
    it, so the cluster is strict by omission."""

    prices_path: Path = DEFAULT_PRICES_PATH
    """Where the price table is read from.

    Read by `create_app`, which builds the `PriceStore` on `app.state.prices` from
    it at startup, so a malformed file stops the process rather than surfacing on
    a request that then cannot be priced. Every served request is costed against
    that one snapshot (`test_the_price_store_is_built_over_the_configured_path`)."""

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

    budget_override_secret: str = ""
    """The key an `X-Gateway-Budget-Override` token is signed with, or empty for a
    gateway that honours no override at all.

    A SEPARATE secret from `admin_key_sha256` above, which is an earlier change decision A
    and a design note: the admin key opens `/admin/*` and is scoped to nothing else, so
    signing overrides with it would put it in the hands of every eval client that
    ever needs to finish a run over its cap. A token here is bound to one tenant
    and one month instead, and cannot be replayed by another tenant or in the next
    one.

    A secret, not a digest, unlike the admin key: an HMAC needs the key itself to
    verify a signature, where authenticating the admin key only needs its digest.

    The shipped empty default refuses every token outright rather than verifying
    against an empty key, because HMAC over an empty key is still good HMAC and
    anyone who knows the scheme could mint one
    (`test_a_gateway_with_no_override_secret_refuses_a_token_minted_from_the_empty_one`)."""

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

    eval_judge_key: str = ""
    """The tenant API key `ops-eval --judge-url` authenticates the judge with, or
    empty for a harness that judges nothing.

    A key and not a digest, unlike `admin_key_sha256`: the harness is a CLIENT of
    the gateway here and has to present the key, where authenticating one only
    needs its hash. Empty is the shipped default and the common case - `make eval`
    on a laptop with no gateway up runs the deterministic assertions and prints
    `cli.UNJUDGED_LINE` (an earlier change decision B,
    `test_with_no_judge_key_the_run_judges_nothing_and_says_so_once`). It lives in
    `.env` locally and in a SOPS-encrypted secret in the cluster, never in this
    repository."""

    database_url: str | None = None
    """Where PostgreSQL lives, or `None` for no `postgres` readiness probe.

    Read by two things beside the probe: `create_app` builds ONE pool from it when
    it is set and hands that pool to both Postgres writers - the residency log and the cost ledger - and `make migrate`,
    `make purge-residency` and `make refresh-rollup` read it through this same
    field. A probe for a store nothing writes to would still be worth having,
    because #5 names Postgres and a readiness endpoint that silently omits a
    configured dependency is a readiness endpoint that lies."""

    qdrant_url: str | None = None
    """Where Qdrant lives, or `None` for the in-process semantic cache and no probe.

    Two modules read it now (`grep -rn "\\.qdrant_url" packages/ services/`,
    excluding this file, answers `gateway/app.py` and `gateway/probes.py`).
    Probed over `httpx` rather than through a client library: a probe is one GET against `QDRANT_READY_PATH`. Set,
    it also builds `QdrantSemanticCache` for `app.state.semantic_cache`, so every
    replica searches the same vectors; absent leaves `InProcessSemanticCache`,
    which is what lets the suite and a bare local run take the semantic path with
    no Qdrant anywhere."""

    prometheus_url: str | None = None
    """Where the console read routes send their PromQL, or `None` for none at all.

    Read only by `create_app`, which builds one `gateway.prometheus.PrometheusClient`
    over it. Absent leaves `app.state.prometheus` as `None`, and every `/console/*`
    route then answers its metric fields as null or empty with
    `prometheus: unconfigured` rather than failing (a design note,
    `test_with_nothing_configured_every_route_is_200_and_both_sources_unconfigured`).
    No readiness probe reads it: `/readyz` names the stores a served request
    needs, and Prometheus is not one of them."""

    otel_service_name: str = "opsconsole-gateway"
    """What this process is called in a trace viewer. One trace carries spans from
    the Helpdesk and from the gateway, so the two need different names to be
    told apart at all."""
