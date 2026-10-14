# The gateway image. Two stages: dependencies resolved in one, the runtime
# carrying only what serves traffic.
#
# Nothing secret is baked in. Every credential - provider keys, the Langfuse
# pair, the tenant key digests - arrives as an environment variable at run time,
# from `.env` locally and from a SOPS-encrypted secret in the cluster (hard
# rule 5).
FROM python:3.12-slim AS build

# uv from its own published image rather than pip-installed: one pinned binary,
# and the lockfile is the thing being honoured here.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# The dependency layer, cached until the lockfile changes: the project's own
# source is copied afterwards so a code edit does not re-resolve anything.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY app/ ./app/
COPY config/ ./config/
COPY migrations/ ./migrations/
# The eval golden set, so the nightly-eval CronJob can read it from an in-image
# path (`/app/evals/golden/example.jsonl`) rather than a repo-relative one the
# container does not carry. Only `golden/` ships, not the
# baseline, labels or target stub: the nightly run reads the set and nothing else.
COPY evals/golden/ ./evals/golden/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim

# curl is here for the compose and Kubernetes healthchecks and for nothing else.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# A non-root account that owns nothing it does not need: the code is read-only
# to it, and the process writes no files at all.
RUN useradd --create-home --uid 1000 opsconsole

WORKDIR /app
COPY --from=build --chown=root:root /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # The image does not carry the repository layout, so the defaults computed
    # from `core/settings.py`'s file position would point outside /app. Both
    # files are loaded eagerly at startup - the routing policy and the price
    # table are built in `create_app` - so a wrong path here is a container that
    # never answers its first request rather than one that degrades.
    OPSCONSOLE_ROUTING_POLICY_PATH=/app/config/routing.yaml \
    OPSCONSOLE_PRICES_PATH=/app/config/prices.yaml

USER 1000
EXPOSE 8080

# Liveness, not readiness: a dependency outage must leave the container running
# and let `/readyz` take it out of rotation instead (`gateway/health.py`).
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
