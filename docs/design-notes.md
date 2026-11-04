# Design notes

This page holds the background a reviewer may want after the README. It covers the
tools, the main decisions, running and scaling, and the cloud code.

## What it is built with

### The gateway

The gateway is written in Python and checked with `mypy`. It runs on FastAPI. The
routes are async. Tracing is done by plain ASGI middleware. An `asyncio.Semaphore`
in `backend/app/core/concurrency.py` caps how many provider calls run at once. Auth
sits on the parent routers in `backend/app/api/router.py`, so a new route cannot
forget it.

### Data stores

PostgreSQL holds the cost ledger, the eval runs and the releases. Its SQL
migrations are in `backend/migrations/`. Redis holds the exact cache, the budgets,
the rate limit windows and the redaction maps. Qdrant holds the semantic cache, in
`backend/app/cache/vectors.py`.

### Tests

The backend has over 3,600 pytest tests. Markers split out the slow tests and the
tests that need a running service. The console is tested with vitest.

### The console

The console in `web/` is a Next.js 15 app written in TypeScript. Its pages render
on the server.

### Docker and CI

One `docker-compose.yml` runs the whole local stack. The images use pinned base
images and run as a non-root user. There are six GitHub Actions workflows in
`.github/workflows/`. CI calls the same `make` targets you run on a laptop. Images
are pushed to GHCR, scanned with Trivy and shipped with an SBOM.

### Kubernetes and Terraform

The Kubernetes manifests in `infra/k8s/` are built with kustomize and checked with
kubeconform. In CI, a kind cluster deploys the gateway and runs a smoke test.
`infra/terraform/` holds three modules. One is a Hetzner node, one is an AWS EKS
stack and one is an AWS GPU box. CI runs `terraform validate` on all three. The two
AWS modules carry a monthly cost budget. The AWS stack has not been deployed yet.

### Evals and the judge

The evals play multi-turn conversations against an agent. They check that the
agent changes nothing before the user says yes, breaks none of a case's rules, and
leaves the data in a right end state. Pass rates come with Wilson ranges. An LLM
judge answers the reply checks. It can be checked against a person's answers with
Cohen's kappa. The code is in `backend/app/evals/`.

The judge prompt is a file, `backend/app/evals/prompts/judge-v3.md`. Its sha256 is
saved with every run. The rubric and the reply reach the judge as marked data, not
as instructions.

### Safety, cost and tenants

Personal data is replaced before any provider call. Provider refusals become typed
errors. Budgets are hard caps. Every call is priced from versioned price files, and
budgets work per tenant and per user. Keys belong to one tenant, and the tenant is
part of every cache key, redaction map and budget.

### Watching it run

Every request is an OpenTelemetry trace sent to Langfuse. Prometheus collects the
metrics. Grafana shows them in dashboards, and alert rules fire when something is
wrong.

### Many workers

Fallback chains and circuit breakers handle a failing provider. Shared state lives
in Redis and Postgres, so more gateway workers can be added. A k6 load test drives
the gateway against a mock provider.

## The main decisions

### Keys and access

- **Keys are stored as digests.** The gateway keeps only the sha256 of each tenant
  key and of the admin key. A leaked config or process dump gives no working key.
- **The admin key is separate.** A valid tenant key gets a 401 on `/admin/*` and
  `/console/*`.
- **Budget overrides are signed with their own secret.** An override token is an
  HMAC tied to one tenant and one month. Another tenant cannot reuse it, and it
  stops working next month. With no secret set, no override is accepted.

### Providers and models

- **Each provider is one file.** A provider is one YAML file in
  `backend/config/providers/`. It holds the kind, the address, the key name and the
  models with their prices. Prices are a dated list. A price change adds a row, so
  old ledger rows keep the price they were charged. Adding a model takes one file
  and one alias line in `backend/config/routing.yaml`, and no code change. See
  [customizing.md](customizing.md).
- **Embeddings use the routed local provider.** `POST /v1/embeddings` uses the
  OpenAI format and goes through the same routing, limits and pricing as chat.
  There is no second path to keep in step.
- **Embeddings have no fallback.** The `embed` alias has one choice, the local
  llama.cpp server, so embedded text never leaves the machine. A second embedding
  model would write vectors that do not match the first, and search would quietly
  break. It takes up to 64 inputs of 8,192 characters each. It can return base64
  vectors, which the OpenAI SDK asks for by default.
- **Circuit breaker state lives in each process.** It needs no shared store. If a
  breaker opens a little later on another worker, little is lost.

### Evals

- **Eval scores are percentages, with null when there is no evidence.** A run over
  zero cases has no score and fails the gate. It does not read as perfect.
- **The eval gate runs on a real model.** A conversation has no fixed answer for
  each model call. Recordings would go stale with every prompt change. So a mock
  cannot play one.

### Shipping

- **Images are named by commit and scanned after push.** Every image tag points at
  one commit. Trivy scans the image that was actually pushed.
- **The console shows a missing source as a flag, not an error.** If Prometheus or
  the database is missing, that part of the page says "unconfigured". The rest of
  the page still shows.
- **Migrations take a lock.** `ops-migrate` is its own step. The optional
  migrate-on-start takes a Postgres advisory lock first. So two workers starting
  at once never run the same change twice. A failed migration stops the start.

## SLOs, alerts and load

### SLOs

Two goals are worked out from the gateway's own metrics, over a fixed 30 days.

- **Availability.** 99.5% of requests do not end in a 5xx error.
- **Latency.** 95% of API-tier requests finish in under 6 seconds.

The "SLO error budgets" row of the Grafana gateway overview dashboard shows how
much of each allowance is left.

### Alerts

The Prometheus alert rules are in `infra/k8s/monitoring/alerts/rules.yml`. They
fire when the gateway is down, when the error rate or latency goes past its goal,
when a circuit is open, when ledger writes are dropped, on budgets, on the nightly
eval score, and on problems with the platform.

### Load test

`make load` runs k6 at a steady 50 requests a second for ten minutes against the
real gateway. The provider behind it is a mock that answers chat in a fixed
400 ms. So any time above that is the gateway's own work. That work covers auth,
routing, redaction, both caches, limits, budgets and the ledger write.

The traffic mixes plain chat, streamed chat and embeddings. Some prompts repeat, so
the caches take part.

The run fails if chat p95 reaches 6 seconds. It fails if more than 0.5% of requests
fail. It also fails if k6 drops even one iteration because the gateway could not
keep up.

A run writes its summary to `infra/load/results`, which git ignores. No result is
committed yet.

### Scaling out

The gateway keeps no state of its own except the circuit breaker. Redis holds the
caches, rate limit windows and redaction maps. Postgres holds the ledger. So adding
a worker adds throughput without changing either store.

## Cloud and cluster code

The main way to run this project is on a laptop. The cloud code ships too. It is
there to read and check. It is not a step you must take.

- `infra/terraform/aws` is an AWS EKS stack. It has RDS PostgreSQL, ElastiCache
  Redis and an S3 bucket. The pods reach the bucket through IAM roles for service
  accounts.
- `infra/scripts/aws_evidence.sh` applies that stack, deploys the gateway, runs a
  smoke test and saves the result. It always destroys the stack at the end. It has
  not been run against AWS yet, so no result is saved. See
  [aws-evidence/README.md](aws-evidence/README.md).
- `infra/terraform/aws-gpu` is a single GPU box for the self-hosted versus API
  comparison. See [tradeoff-run/README.md](tradeoff-run/README.md).
- `infra/terraform/hetzner` is one small ARM server running k3s, with its DNS.
- `infra/k8s/` holds the kustomize manifests for the gateway, the console, the
  embedding server, the data stores and monitoring. CI builds and checks them. CI
  also deploys the gateway to a kind cluster and runs a smoke test.

Nothing in this repo runs `terraform apply` or `kubectl apply` for you. A person
runs those.

The Azure OpenAI path is described in [azure.md](azure.md). The map of the whole
`infra/` folder is in [infra/README.md](../infra/README.md).
