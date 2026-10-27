"""Process settings, read from the environment at construction time."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROUTING_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"
"""The shipped `routing.yaml`, found from this file rather than from the working
directory. `parents[2]` is `backend/`, so this resolves to `backend/config/`. A
deployment that does not carry the repository layout - the container image, a
wheel install - sets `OPSCONSOLE_ROUTING_POLICY_PATH` instead."""

DEFAULT_PROVIDERS_DIR = Path(__file__).resolve().parents[2] / "config" / "providers"
"""The shipped provider folder, one YAML file per provider, found the same way and
for the same reason, with `OPSCONSOLE_PROVIDERS_DIR` as the escape hatch."""

DEFAULT_EVAL_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "evals" / "conversation-baseline.json"
)
"""The committed eval baseline, found the same way, with
`OPSCONSOLE_EVAL_BASELINE_PATH` as the escape hatch."""


MAX_EMAIL_LENGTH = 254
"""The longest email address `budget_contact` takes, inclusive (RFC 5321's path limit)."""

EMAIL_PATTERN = re.compile(r"[^@\s,]+@[^@\s,]+\.[^@\s,.]+")
"""One address: no space, comma or second `@`, and a dot in the domain. `fullmatch`, so a
trailing newline is refused."""


class Settings(BaseSettings):
    """Gateway settings.

    Every field has a default, so the process starts with no environment at all.
    Values come from `OPSCONSOLE_`-prefixed variables; presence decides an
    override, not truthiness, so `OPSCONSOLE_ENV=""` yields an empty `env`.
    """

    model_config = SettingsConfigDict(env_prefix="OPSCONSOLE_")

    env: str = "local"
    port: int = 8080

    api_keys: str = ""
    """Comma-separated `tenant:sha256hex` pairs. Digests only: an issued tenant
    key lives in `.env` locally and in a SOPS-encrypted secret in the cluster, and
    no issued key is in this repository (the worthless strings in
    the tests are test data, not anyone's key). Empty is a
    running gateway that authorises nobody."""

    routing_policy_path: Path = DEFAULT_ROUTING_POLICY_PATH

    providers_dir: Path = DEFAULT_PROVIDERS_DIR
    """The folder of provider files: providers, their models and their prices.

    Read by `create_app`, which builds the policy and the `PriceStore` from it at
    startup, so a bad file stops the process rather than surfacing on a request
    that then cannot be routed or priced. Read again, with `routing.yaml`, on
    every routing reload."""

    eval_baseline_path: Path = DEFAULT_EVAL_BASELINE_PATH
    """Where the quality page reads the eval gate's thresholds from.

    Read on every `/console/quality` request, never at startup: a missing or
    invalid file leaves the thresholds null and the page still answers."""

    admin_key_sha256: str = ""
    """The sha256 digest of the one admin key, or empty for a gateway whose
    `/admin/*` routes authorise nobody.

    A digest rather than the key itself: the tenant keys
    above are already hashed at rest, and a plaintext admin key in an environment
    variable is the one form that also leaks through a process dump and a
    `docker inspect`. Issuing a key is sha256-ing it once. This is a SEPARATE key
    space from `api_keys`: a valid tenant key is 401 on `/admin/*`, and the
    shipped empty default denies every caller."""

    budget_override_secret: str = ""
    """The key an `X-Gateway-Budget-Override` token is signed with, or empty for a
    gateway that honours no override at all.

    A SEPARATE secret from `admin_key_sha256` above: the admin key opens `/admin/*` and is scoped
    to nothing else, so
    signing overrides with it would put it in the hands of every eval client that
    ever needs to finish a run over its cap. A token here is bound to one tenant
    and one month instead, and cannot be replayed by another tenant or in the next
    one.

    A secret, not a digest, unlike the admin key: an HMAC needs the key itself to
    verify a signature, where authenticating the admin key only needs its digest.

    The shipped empty default refuses every token outright rather than verifying
    against an empty key, because HMAC over an empty key is still good HMAC and
    anyone who knows the scheme could mint one."""

    otlp_endpoint: str | None = None
    """Where traces are sent, or `None` for a gateway that builds spans and drops
    them. Langfuse EU is an OTLP/HTTP endpoint, so this is a
    full traces URL rather than a host. Absent by default so a developer, a test
    run and a CI job export nothing without saying so."""

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
    a default: the app setup and the readiness probes. `probes.py`
    registers the `redis` probe only when it is set.
    `create_app` builds ONE client from it and hands it to three seams, each of
    which falls back to an in-process twin when it is absent:
    `RedisRedactionMapStore`, `RedisDirectoryStore` and `RedisResponseCache`. Absent by default so
    the
    suite and a bare local run need no backing service."""

    eval_judge_key: str = ""
    """The tenant API key `ops-eval --judge-url` authenticates the judge with, or
    empty for a harness that judges nothing.

    A key and not a digest, unlike `admin_key_sha256`: the harness is a CLIENT of
    the gateway here and has to present the key, where authenticating one only
    needs its hash. Empty is the shipped default and the common case - `make eval`
    on a laptop with no gateway up runs the deterministic assertions and prints
    `cli.UNJUDGED_LINE`. It lives in
    `.env` locally and in a SOPS-encrypted secret in the cluster, never in this
    repository."""

    database_url: str | None = None
    """Where PostgreSQL lives, or `None` for no `postgres` readiness probe.

    Read by two things beside the probe: `create_app` builds ONE pool from it when
    it is set and hands that pool to the Postgres writer, the cost ledger,
    and `make migrate` reads it through this same
    field. A probe for a store nothing writes to would still be worth having,
    because #5 names Postgres and a readiness endpoint that silently omits a
    configured dependency is a readiness endpoint that lies."""

    qdrant_url: str | None = None
    """Where Qdrant lives, or `None` for the in-process semantic cache and no probe.

    Two modules read it now: the app setup and the readiness probes.
    Probed over `httpx` rather than through a client library: a probe is one GET against
    `QDRANT_READY_PATH`. Set,
    it also builds `QdrantSemanticCache` for `app.state.semantic_cache`, so every
    replica searches the same vectors; absent leaves `InProcessSemanticCache`,
    which is what lets the suite and a bare local run take the semantic path with
    no Qdrant anywhere."""

    migrate_on_start: bool = False
    """Whether the gateway applies pending ledger migrations as it starts.

    Off by default, so the suite and a bare run never touch a database at start.
    On in compose and `.env.demo`, so a new column never leaves a console page
    failing. Read only by `app.main.lifespan`, which takes a Postgres advisory
    lock first and stops the start if a file fails. `ops-migrate` still works
    on its own."""

    prometheus_url: str | None = None
    """Where the console read routes send their PromQL, or `None` for none at all.

    Read only by `create_app`, which builds one `gateway.prometheus.PrometheusClient`
    over it. Absent leaves `app.state.prometheus` as `None`, and every `/console/*`
    route then answers its metric fields as null or empty with
    `prometheus: unconfigured` rather than failing.
    No readiness probe reads it: `/readyz` names the stores a served request
    needs, and Prometheus is not one of them."""

    langfuse_project_url: str | None = None
    """The Langfuse project page, e.g. `https://cloud.langfuse.com/project/<id>`, or
    `None` for no trace links.

    Read only by `GET /console/requests`, which links a listed request to
    `<this>/traces/<trace_id>` when it is set. Unset on the cluster until the real
    project is known."""

    public_trace_langfuse: bool = False
    """Mark every trace public in Langfuse, so anyone with a trace's link can read it.

    The master switch: off means no trace is public, whatever a tenant's
    `trace_public` says. On, a tenant opts out with `trace_public: false`. A trace
    made public stays public in Langfuse after this is turned off."""

    grafana_url: str | None = None
    """The Grafana page the Overview tab links for health details, or `None` for no link.

    Read only by `GET /console/home`. It must be an `http` or `https` URL with a
    host, checked here so a `javascript:` value never reaches a link; the console
    checks it again before it draws one. Empty counts as unset, which is the local
    value: no Grafana runs locally. On the cluster `gateway-config` points it at the
    `gateway-overview` dashboard. A test pins this."""

    @field_validator("grafana_url")
    @classmethod
    def _grafana_url_is_http(cls, value: str | None) -> str | None:
        if not value:
            return None
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("must be an http or https URL")
        return value

    budget_contact: str = ""
    """Who a tenant over its budget is told to ask for more, or empty for nobody.

    Named in the `BUDGET_EXCEEDED` message. Empty, or one email address of at most 254
    characters; anything else stops the start, so a typo never reaches a refusal message.
    A test pins this."""

    @field_validator("budget_contact")
    @classmethod
    def _budget_contact_is_one_email(cls, value: str) -> str:
        if value and (len(value) > MAX_EMAIL_LENGTH or EMAIL_PATTERN.fullmatch(value) is None):
            raise ValueError("must be empty or one email address")
        return value

    otel_service_name: str = "opsconsole-gateway"
    """What this process is called in a trace viewer. One trace carries spans from
    the Helpdesk and from the gateway, so the two need different names to be
    told apart at all."""
