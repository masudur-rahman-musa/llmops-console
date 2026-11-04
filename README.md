# LLMOps Console

*Designed and built end to end by one engineer. Architecture, gateway, backend, web app, tests, CI and infra.*

A model gateway and an operations console for LLM apps, with an eval gate in CI.

Apps send OpenAI-shaped requests to one gateway and name an alias such as
`agent-default`, never a provider model. The gateway picks the provider, strips
personal data before it leaves, checks the tenant's budget and rate limit, serves
safe repeats from a cache, and records the tokens, cost and latency of every call.
A web console and Grafana show the result. An eval gate plays real agent
conversations on every pull request and fails it when quality drops below a
committed baseline.

Its main client is the companion project
[helpdesk-agentic-flow](https://github.com/masudur-rahman-musa/helpdesk-agentic-flow),
an AI help desk agent. Every model call the help desk makes goes through this
gateway, its RAG documents are redacted here when they are ingested, and its
conversations are what the eval gate scores.

License is Apache-2.0. [Run it on your laptop](docs/local-setup.md) in a few
commands.

### What it does

- **One gateway, model aliases.** Apps ask for `agent-default`, `agent-strong`,
  `judge` or `embed`. Only `backend/config/routing.yaml` and the files in
  `backend/config/providers/` name a real model. Chat runs on Anthropic, with
  OpenAI as the fallback. Embeddings run on a local llama.cpp server.
- **Cost and latency on every call.** Each request writes a row to a PostgreSQL
  ledger with tokens, cached tokens, the price entry used, cost in USD, status and
  latency. The response carries `X-Gateway-Cost-USD`. Prometheus counts requests,
  latency, tokens, fallbacks, open circuits and redactions.
- **Traces.** Every request is one OpenTelemetry trace sent to Langfuse over OTLP,
  with a step for auth, routing, each cache, redaction and the provider call. A
  trace carries the tenant and the session, never a name or an email.
- **PII redaction.** Ten kinds of personal data are replaced with placeholders
  before a provider sees them, and can be put back in the answer. Accuracy is
  measured on a held-out set and published below.
- **Budgets and rate limits per tenant.** A tenant or user over its monthly cap is
  refused before the provider is called. Requests and tokens per minute are
  limited in Redis.
- **Two caches.** An exact cache in Redis, and a semantic cache in Qdrant that
  needs 0.97 similarity and is on only for `agent-default`.
- **Failure handling.** A circuit breaker per provider opens after 5 failures in
  60 seconds and tries again after 30. Calls retry twice with backoff. Fallback
  only goes to providers the alias lists. Every refusal is a typed error code.
- **Eval gate.** The help desk's 42 conversation cases are played three times each
  by a simulated person. Code checks the rules and the end state of the data, and an
  LLM judge answers the reply checks. The pull request fails when safety drops below
  100 or the score falls below `backend/evals/baseline.json`.
- **Ops console.** A Next.js 15 and React 19 app with four tabs (Overview,
  Requests, Cost, Quality) behind a password login. Grafana ships five dashboards
  and Prometheus ships twelve alert rules.

### In numbers

- `make fast` ran 3,683 passing backend tests in about 32 seconds on a laptop, with
  132 skipped because they need Postgres, Redis or Qdrant. `make test` runs those
  too.
- 778 more tests in `infra/tests/` check the CI workflows, compose files,
  Kubernetes manifests, Terraform modules and these docs.
- Coverage must stay at 80% for the whole backend and at 100% for routing, limits
  and input validation, or `make test` fails.
- The latest eval played the help desk's 42 cases three times each, 126 plays.
  Safety is 100, 83% of cases were right in every play, and every case was right
  at least once. The gate holds safety at 100 and that 83% as the baseline.

### Skills and tech shown

- **Python.** The gateway in `backend/app/`, typed and checked with `mypy`.
- **FastAPI.** The gateway API, with auth on parent routers in `backend/app/api/router.py`.
- **Async Python.** Async routes, pure ASGI middleware for tracing, and an
  `asyncio.Semaphore` that caps provider calls in `backend/app/core/concurrency.py`.
- **PostgreSQL.** The cost ledger, eval runs and releases, with SQL migrations in
  `backend/migrations/`.
- **Redis.** Exact cache, budgets, rate limit windows and redaction maps.
- **Vector DBs.** Qdrant holds the semantic cache in `backend/app/cache/vectors.py`.
- **pytest.** Over 3,600 backend tests, with markers for slow and service tests.
- **TypeScript and React.** The console in `web/`, Next.js 15 with server
  components, tested with vitest.
- **Docker.** One `docker-compose.yml` runs the whole local stack. Images use
  pinned base digests and non-root users.
- **CI/CD, GitHub Actions.** Six workflows in `.github/workflows/`. CI calls the
  same `make` targets as a laptop. Images are pushed to GHCR, scanned with Trivy
  and shipped with an SBOM.
- **Kubernetes.** Kustomize manifests in `infra/k8s/`, checked with kubeconform,
  and a kind cluster in CI that deploys the gateway and runs a smoke test.
- **Terraform / IaC.** Modules for a Hetzner node, an AWS EKS stack and an AWS GPU
  box in `infra/terraform/`. CI runs `terraform validate` on all three, and the two
  AWS modules carry a monthly cost budget.
- **AWS.** The `infra/terraform/aws` module (EKS, RDS, ElastiCache, S3 with IAM
  roles for service accounts) and its apply, smoke test and destroy script. It
  has not been deployed to AWS yet.
- **Evals / LLM evaluation.** Multi-turn conversation evals, an LLM judge with
  Cohen's kappa calibration, pass rates with Wilson ranges, and a CI gate in
  `backend/app/evals/`.
- **Agentic systems.** The evals check that an agent changes nothing before the user
  says yes, breaks none of a case's rules, and leaves the data in a right end state.
- **Prompt engineering.** The judge prompt is a file, `backend/app/evals/prompts/judge-v3.md`,
  and its sha256 is recorded on every run. Rubric and reply reach the judge as
  delimited data, not instructions.
- **Guardrails / AI safety.** PII redaction before any provider call, provider
  refusals mapped to typed errors, and hard budget caps.
- **Cost / latency awareness.** Per-call pricing from versioned price files,
  per-tenant and per-user budgets, prompt cache pricing, and a k6 load test.
- **Multi-tenant isolation.** Keys are bound to one tenant, and the tenant is part
  of every cache key, redaction map and budget.
- **Observability for AI.** OpenTelemetry traces to Langfuse, Prometheus metrics,
  Grafana dashboards and alert rules.
- **System design / distributed systems.** Fallback chains, circuit breakers,
  shared state in Redis and Postgres, so gateway workers scale out.
- **LLMOps / MLOps.** The whole repo is one. Model routing, cost tracking, eval
  gates, a guarded baseline, and a measured self-hosted versus API comparison kit.

## The problem

An LLM feature reaches production and three questions have no owner. Is it still
working, or has quality drifted since the prompt changed? What does it cost, per
tenant, this month, against a budget? And what personal data reached the provider?

When each app calls the provider directly, there is no single place to enforce a
rule, price a token or notice a drop in quality. This project puts a gateway in
that place. Every call goes through one OpenAI-compatible endpoint. The console
reads back what the gateway recorded, and the eval gate measures quality before a
change merges.

## Video and screenshots

No recording or screenshots are committed yet.
[docs/media/README.md](docs/media/README.md) lists the file names they will use.

## How it fits together

```mermaid
flowchart LR
    app["Client app<br/>(help desk agent)"] -->|"OpenAI API + alias"| gw
    subgraph gw["Gateway (FastAPI)"]
        direction TB
        auth["Key and tenant check"] --> route["Alias to providers"]
        route --> red["PII redaction"]
        red --> lim["Budget and rate limit"]
        lim --> cache["Exact and semantic cache"]
        cache --> call["Provider call<br/>breaker, retry, fallback"]
    end
    call --> anth["Anthropic"]
    call --> oai["OpenAI (fallback)"]
    call --> emb["llama.cpp embeddings"]
    lim --- redis[("Redis")]
    cache --- qdrant[("Qdrant")]
    gw --> ledger[("Postgres ledger")]
    gw --> prom["Prometheus"]
    gw --> lf["Langfuse traces"]
    ledger --> capi["Console API"]
    prom --> capi
    capi --> web["Console (Next.js)"]
    prom --> graf["Grafana"]
```

A request goes through these steps:

1. **Auth.** The bearer key is hashed and must match the tenant named in
   `X-Tenant`. Admin and console routes use a separate admin key.
2. **Limits.** The monthly budget and the per-minute limits are checked before any
   provider is called.
3. **Routing.** The alias resolves, for that tenant, to an ordered list of
   providers and models. The first is the default and the rest are fallbacks.
4. **Exact cache.** A repeat of the same request is answered from Redis at once.
   The key is a hash scoped by tenant and data class, and the stored answer still
   holds placeholders.
5. **Redaction, then the semantic cache.** If the tenant's rule for the request's
   data class says redact, every message is redacted before the semantic cache or
   any provider sees it. With no `X-Data-Class` header the class is
   `confidential`. For `agent-default`, a close enough question in Qdrant is then
   answered too.
6. **Provider call.** The adapter calls the provider, with a circuit breaker, two
   retries and fallback inside the alias. Streaming is supported.
7. **Record.** Tokens are priced, a ledger row is written, metrics are counted and
   the trace is closed after the last byte is sent.

The console never touches a database or Prometheus from the browser. Its pages
render on the server and read through the gateway's `/console/*` routes. More
detail is in [ARCHITECTURE.md](ARCHITECTURE.md) and
[what a trace shows](docs/tracing.md).

### Providers, one file each

Each provider is one YAML file in `backend/config/providers/`, with its kind,
address, key name, and models with their prices. Prices are a list with dates, so
a price change appends a row and old ledger rows keep the price they were charged.
Adding a model is one file and one alias line in `backend/config/routing.yaml`,
reviewed in a pull request and checked by the eval gate. No code changes. See
[docs/customizing.md](docs/customizing.md).

### Embeddings

`POST /v1/embeddings` speaks the OpenAI shape and goes through the same routing,
limits and pricing as chat. The `embed` alias has one candidate, the local
llama.cpp server, so embedded text never leaves the machine. It has no fallback on
purpose, because a second embedding model writes vectors in a different space and
would quietly break search. It takes up to 64 inputs of 8,192 characters each, and
can return base64 vectors as the OpenAI SDK asks by default.

## The seven guarantees

Seven claims, each with the test that fails if it stops being true.

1. **Fallback stays inside the alias.** When a provider fails, the gateway tries
   only the next provider the alias lists for that tenant:
   [backend/tests/api/test_fallback.py](backend/tests/api/test_fallback.py).
2. **Personal data is replaced before it leaves.** Ten kinds are detected, checked
   with a checksum where the shape alone is not enough, and replaced with stable
   placeholders:
   [backend/tests/redaction/test_redactor.py](backend/tests/redaction/test_redactor.py).
3. **A budget refuses rather than warns.** A tenant over its cap is refused before
   a provider is called, and an override has to be signed:
   [backend/tests/limits/test_budget.py](backend/tests/limits/test_budget.py).
4. **A rate limit is enforced per tenant.** A tenant over its requests or tokens
   per minute gets a 429 that says when to come back:
   [backend/tests/limits/test_ratelimit.py](backend/tests/limits/test_ratelimit.py).
5. **A cache hit is the same answer, for the same tenant.** The tenant, alias,
   provider, data class, model and the request body are all in the key, so two
   tenants cannot share an entry:
   [backend/tests/cache/test_key.py](backend/tests/cache/test_key.py).
6. **A change that makes the agent worse does not merge.** The gate compares the
   run's scores with the committed baseline:
   [backend/tests/evals/test_gate.py](backend/tests/evals/test_gate.py).
7. **Every token is priced at the rate in force.** Ledger rows name the price
   entry that priced them, and old rows are never rewritten:
   [backend/tests/storage/test_cost_ledger.py](backend/tests/storage/test_cost_ledger.py).

## What broke

Failures in the AI and gateway behaviour, how each was found, and the test that
now holds the fix.

- **A refused request could be passed on until some provider said yes.** A
  content-policy refusal looked like an ordinary error, so fallback sent the same
  request to the next provider, which undoes the refusal. Found in a safety review
  of the gateway. A refusal is now recognised from each provider's own codes and is
  never retried, never sent on and never counted against the provider's health.
  Fallback covers only rate limits, server errors and timeouts. Held by
  [backend/tests/api/test_fallback.py](backend/tests/api/test_fallback.py) and
  [backend/tests/providers/test_refusals.py](backend/tests/providers/test_refusals.py).
- **The semantic cache could give one chat another chat's answer.** Only the last
  user message was embedded, so short replies like "yes" in two different chats
  matched perfectly, and the help desk's shared system prompt did not tell them
  apart. Found by working through real multi-turn traffic before it reached users.
  Messages under 16 characters are now never cached, and the lookup embeds the end
  of the previous turn with the final message. Held by
  [backend/tests/cache/test_probe.py](backend/tests/cache/test_probe.py).
- **Redaction scored a perfect 1.000 and still missed real formats.** An
  IPv4-mapped IPv6 address and a spaced IBAN slipped through, because the test set
  held only compact forms. Found in review, before release. Both patterns were
  fixed, scoring now measures what the redactor actually outputs, and a held-out set
  of 600 generated cases replaced the small one. The honest pooled recall is 0.888,
  and every miss is published in
  [docs/redaction-misses.md](docs/redaction-misses.md).
- **A budget could be passed after Redis lost its counters.** Spend was counted in
  a monthly Redis counter that started from zero whenever it was missing, so a
  restart or a new replica forgot what a tenant had spent. Found when the console's
  ledger and the gateway disagreed. A missing counter is now loaded once from the
  ledger, a script stops two replicas loading it twice, and if the ledger cannot be
  read the request is refused rather than counted as free. Held by
  [backend/tests/limits/test_budget_seed.py](backend/tests/limits/test_budget_seed.py).
- **The circuit breaker would not open on a partly failing provider.** Every
  success cleared the failure count, so a provider failing on part of its traffic,
  the most common kind of outage, never tripped it. Recovery also let every waiting
  caller through at once instead of one test request. A success now settles only
  the recovery test, and only the first caller gets it. Held by
  [backend/tests/providers/test_breaker.py](backend/tests/providers/test_breaker.py).
- **Cached OpenAI tokens were billed at the full price.** The gateway ignored the
  cached part of OpenAI's token count, so a fallback to OpenAI cost more on paper
  than it did. Cached tokens are now counted and priced on their own, with no token
  billed twice. Held by
  [backend/tests/providers/test_openai_cache_tokens.py](backend/tests/providers/test_openai_cache_tokens.py).

Process failures get a blameless write-up too, like
[docs/postmortems/2026-09-17-red-tests-went-unseen.md](docs/postmortems/2026-09-17-red-tests-went-unseen.md).

## How the evals work

The engine plays multi-turn conversations against the companion help desk, which keeps
42 cases. Each case says who the person is, what they want, what they know, when they
would say yes, what must never happen, and which end states of the help desk's data
count as right. The format is described in [docs/eval-format.md](docs/eval-format.md).

A simulated person, played by a cheap model, talks to the real help desk agent one turn
at a time, starting from reset data. Code checks every play first: nothing was changed
before a yes, no rule from the case's "never" list was broken, and the data at the end
matches a right end state. A judge model then answers each reply check on its own, with
a quote and a reason. For a play that needs a look it also says whose fault it is: the
agent, the case, or the eval. That answer is shown beside the play and never changes a
score. The report shape is in [docs/eval-report.md](docs/eval-report.md) and the target
it runs against in [docs/eval-target.md](docs/eval-target.md).

Each case is played three times, so chance alone cannot decide it. A play passes, fails,
needs a look, was set up wrong, or hit an eval bug; the last two are not scored.
`backend/evals/baseline.json` holds the committed numbers, and the gate fails a pull
request when safety drops below 100 or the share of cases right in every play falls
below the baseline. The gate is in [docs/eval-gate.md](docs/eval-gate.md). The baseline
changes only through a commit that carries its own decision record, and never to make
a failing run pass.

The latest eval played 42 cases three times each, 126 plays. Safety is 100: no play
changed anything without a yes or broke a case's rules. 83% of cases were right in all
three plays, and every case was right at least once. The committed baseline is safety
100 and 83% right in every play.

### Redaction accuracy

These numbers come from a held-out set of 600 generated cases. That is 50 values
of each kind, written the ways people really write them, plus 100 look-alikes that
must not be found, such as version numbers, MAC addresses and file references. The
templates in `backend/evals/redaction/heldout-templates.json` were never used to
tune a detector, and the set is built from them with a fixed seed. F2 weighs recall
higher than precision, because a missed value reaches a model while a false find
only hides a harmless string. Every miss is listed in
[docs/redaction-misses.md](docs/redaction-misses.md). Both files are written by
`backend/scripts/redaction_report.py`, and a test fails if either drifts.

<!-- redaction-table:start -->
| Kind | Precision | Recall | F2 | Misses |
| --- | --- | --- | --- | --- |
| EMAIL | 1.000 | 1.000 | 1.000 | 0 |
| PHONE | 0.808 | 0.840 | 0.833 | 8 |
| IBAN | 1.000 | 0.960 | 0.968 | 2 |
| CARD | 1.000 | 1.000 | 1.000 | 0 |
| STEUER_ID | 1.000 | 0.700 | 0.745 | 15 |
| PPSN | 1.000 | 0.700 | 0.745 | 15 |
| IPV4 | 0.909 | 1.000 | 0.980 | 0 |
| IPV6 | 1.000 | 1.000 | 1.000 | 0 |
| DOB | 1.000 | 0.680 | 0.726 | 16 |
| PERSON | 1.000 | 1.000 | 1.000 | 0 |
| all | 0.967 | 0.888 | 0.903 | 56 |
<!-- redaction-table:end -->

The ten kinds are email addresses, phone numbers, IBANs, card numbers, German tax
IDs, Irish PPS numbers, IPv4 and IPv6 addresses, dates of birth, and the names a
tenant's operator has pushed to its own directory. Where shape is not enough, the
value must pass its own check. That is Luhn for cards, mod-97 for IBANs, ISO 7064
MOD 11,10 for tax IDs and mod-23 for PPS numbers, so a sixteen-digit order number
is not taken for a card. A date counts as a date of birth only with a cue such as
"born" or "DOB" close in front of it. The weak spots show in the table. Most
misses are phone numbers written with a `(0)`, German tax IDs, PPS numbers and
dates of birth.

Redaction runs in two places:

- **Inline.** When a tenant's rule says redact, every chat message and embedding
  input is redacted before the request leaves. `X-Gateway-Redacted` on the answer
  says whether a value was replaced.
- **On ingestion.** `POST /v1/redact` returns the redacted text and a
  `redaction_map_id`. The map is kept for one hour, for that tenant only. Sending
  the id back as `X-Gateway-Rehydrate` on a chat call puts the real values back in
  the answer, streamed answers included. An unknown or expired id is refused
  before any provider is called.

The same value always gets the same placeholder within a request, so `<EMAIL_1>`
means the same person in every message.

## Self-hosted vs API tradeoff

The repo carries a kit to compare one alias, `agent-default`, served three ways
over the same small conversation file. The options are a cheap API model, a strong
API model, and vLLM on an AWS GPU box that is applied for the run and destroyed
after. Each option reports its check score and its USD per 1,000 turns. API cost
comes from the ledger. Self-hosted cost is the hourly price divided by the
throughput the box delivered.

How the runs are made, and what three cases on one sequential stream can and
cannot tell you, is in [docs/tradeoff-run/README.md](docs/tradeoff-run/README.md).
The table and its winner are written to `docs/tradeoff.md` by `make tradeoff` once
the runs exist.

**Result is pending.** No measured run is committed yet, so there is no table.

## The decisions

The choices that shape the system most, and why.

- **Keys are stored as digests.** The gateway holds only the sha256 of each tenant
  key and of the admin key, so a leaked config or process dump gives no working
  key.
- **The admin key is its own key space.** A valid tenant key is a 401 on
  `/admin/*` and `/console/*`.
- **Budget overrides are signed with their own secret.** A token is an HMAC bound
  to one tenant and one month, so it cannot be reused by another tenant or next
  month. No secret set means no override is honoured.
- **Embeddings use the routed local provider.** They go through the same policy,
  limits and ledger as chat, with no second path to keep in sync.
- **Circuit breaker state is per process.** It needs no shared store, and a
  breaker that opens a little later on another worker costs little.
- **Eval scores are percentages, with null for no evidence.** A run over zero
  cases has no score and fails the gate, rather than reading as perfect.
- **The eval gate runs on a real model.** A conversation has no fixed answer for
  each model call, and recordings drift with every prompt change, so a mock cannot
  play one.
- **Images are named by commit and scanned after push.** Every image tag points at
  one commit, and Trivy scans what was actually pushed.
- **Console sources report as flags, never errors.** A missing Prometheus or
  database shows as "unconfigured" on the page while the rest still renders.
- **Migrations take a lock.** `ops-migrate` is its own step, and the optional
  migrate-on-start takes a Postgres advisory lock first, so two workers starting at
  once never race on the same change. A failed migration stops the start.

## SLOs

Two objectives are computed from the gateway's own metrics over a fixed 30 days.

- **Availability.** 99.5% of requests do not end in a 5xx.
- **Latency.** 95% of API-tier requests finish in under 6 seconds.

The "SLO error budgets" row of the Grafana gateway overview dashboard shows how
much of each allowance is left. The Prometheus alert rules in
`infra/k8s/monitoring/alerts/rules.yml` cover the gateway being down, error rate,
latency over the objective, an open circuit, dropped ledger writes, budgets, the
nightly eval score and the platform.

## Scaling

`make load` drives k6 at a constant 50 requests a second for ten minutes against
the real gateway. The provider behind it is a mock that answers chat in a fixed
400 ms, so any time above that is the gateway's own work. That covers auth,
routing, redaction, both caches, limits, budgets and the ledger write. The traffic
mixes plain chat, streamed chat and embeddings, and some prompts repeat so the
caches take part. The run fails if chat p95 reaches 6 s, if more than 0.5% of
requests fail, or if k6 drops a single iteration because the gateway could not
keep up. No result file is committed. A run writes its summary to
infra/load/results, which git ignores.

The gateway keeps no state of its own except the circuit breaker. Redis holds the
caches, rate limit windows and redaction maps, and Postgres holds the ledger, so
adding a worker adds throughput without changing either.

## Quickstart

Needs Docker, [uv](https://docs.astral.sh/uv/), `make` and `openssl`. The full
walk-through, with the key setup and what to check, is
[docs/local-setup.md](docs/local-setup.md).

    git clone https://github.com/masudur-rahman-musa/llmops-console
    cd llmops-console
    make embed-model                    # the local embedding model, fetched once and checked
    cp backend/.env.example backend/.env
    # add your keys to backend/.env and web/.env.local (see docs/local-setup.md)
    make up                             # gateway, console, Postgres, Redis, Qdrant, Prometheus, Grafana
    # or: docker compose up -d          # the same without the console
    make fast                           # the everyday tests, no services needed

Then open the console at http://localhost:3001 and Grafana at
http://localhost:3002. The gateway is at http://localhost:8080, with `/healthz`
and `/readyz`.

With no model key, the stack still starts. Health checks, redaction, embeddings,
the console, Grafana, the tests and `make load` all work. Only chat needs a key.

To fill every console tab with real numbers, run `make numbers`. It runs the real
eval, sends six chat requests, records a release and checks every tab. It needs the
companion help desk checked out beside this repo and `ANTHROPIC_API_KEY`, and it
spends real tokens.

To run this as your own platform, see [docs/customizing.md](docs/customizing.md).
How the project was built is in [docs/how-it-was-built.md](docs/how-it-was-built.md).

## Azure

The Azure OpenAI path is an adapter plus recorded test fixtures. It ships
disabled, and turning it on is a configuration change, not a code change. See
[docs/azure.md](docs/azure.md).

## AWS

The main path is local. The cloud code ships too, for reading, not as a required
step:

- `infra/terraform/aws` is an EKS stack with RDS PostgreSQL, ElastiCache Redis and
  an S3 bucket reached through IAM roles for service accounts.
  `infra/scripts/aws_evidence.sh` applies it, deploys the gateway, runs a smoke
  test, captures the result and always destroys it. It has not been deployed to
  AWS yet, so no run is captured. See [docs/aws-evidence/](docs/aws-evidence/).
- `infra/terraform/aws-gpu` is the single GPU box for the vLLM comparison above.
- `infra/terraform/hetzner` is one small ARM server running k3s, with its DNS.
- `infra/k8s/` holds the kustomize manifests for the gateway, console, embedding
  server, data stores and monitoring. CI builds and validates them, and deploys the
  gateway to a kind cluster for a smoke test. Nothing in this repo runs
  `terraform apply` or `kubectl apply` for you.

See [infra/README.md](infra/README.md) for the map.
