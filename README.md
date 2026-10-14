# AI Operations Console

The dashboard that tells you whether your AI is working, what it costs, and where your data went.

Underneath: an OpenAI-compatible LLM gateway with per-tenant routing and EU data-residency enforcement, PII redaction, semantic caching, budgets and cost attribution, and an evaluation harness whose CI gate blocks any release that makes the agent worse. Deployed with Terraform and Kubernetes, operated with SLOs, alerts and a runbook.

**Status: under active development.** Measured self-hosted-vs-API numbers and the
walkthrough recording land as the remaining measurement runs complete.

Companion project: [ai-help-desk-governed-access](https://github.com/masudur-rahman-musa/ai-help-desk-governed-access), the agent workload this platform routes, measures and evaluates.

License: Apache-2.0

## The problem

An LLM feature reaches production and three questions have no owner. Is it
working, or has quality quietly drifted since the prompt changed last week? What
does it cost, per tenant, per feature, this month against its budget? And where
did the data go, given a customer whose contract says their text never leaves
the EU?

Most teams answer the first question with a vibe, the second with a provider
invoice that arrives a month late and attributes nothing, and the third with a
policy document nobody can test. The gap is not the model. It is that the calls
leave the application directly, so there is no place to enforce a rule, price a
token, or notice a regression.

This project puts a gateway in that place. Every call goes through one
OpenAI-compatible endpoint that resolves a tenant's alias to a provider under
its residency rules, redacts personal data on the way out, caches what repeats,
prices what it spends, refuses what exceeds the budget, and records the whole
thing. The console reads that record back.

The gateway is one half. The other is an evaluation harness: a golden set, a
scored run, and a CI gate that fails the pull request when the numbers fall
below the committed baseline. A guarantee nobody measures is a preference.

## Video and screenshots

The walkthrough recording is linked here when it is made.

Screenshots of the console and the Grafana dashboards live in
[docs/media/](docs/media/), and the file names each one takes are listed in
[docs/media/README.md](docs/media/README.md). They are captured against the
local stack with seeded traffic, so no real tenant, key or personal value
appears in one.

## How it fits together

A client speaks the OpenAI API and names an alias rather than a model. The
gateway resolves that alias for the calling tenant and the request's data class,
producing an ordered list of candidate providers whose regions the tenant's
residency rule allows. Deny by default: an empty list is a 451, never a widened
region set.

Before a request leaves, personal data is replaced with stable placeholders when
the tenant's rule says so, and the semantic cache is asked whether this prompt
has been answered already for this tenant. After it returns, tokens are priced
against the committed table, the spend is booked to the tenant, and the budget
and rate limits are updated. A trace carrying the session and the tenant, never
a user identity, goes to Langfuse; the counters go to Prometheus.

    client -> gateway -> [ auth | routing + residency | redaction | cache |
                           limits ] -> provider adapter -> provider
                                    -> ledger, traces, metrics -> console

The full component breakdown, written up once the system is complete, is in
[docs/architecture.md](docs/architecture.md). The deployment is Terraform for
the hosts ([infra/terraform/](infra/terraform/)) and kustomize over k3s for the
workloads ([infra/k8s/](infra/k8s/)); the dashboard is a Next.js app
([web/](web/)) that reads through the gateway's
console API and never touches a database or Prometheus from the browser.

## The seven guarantees

Seven claims this system makes, each with the test that would fail if it stopped
being true.

1. **Residency holds under every policy.** A tenant's traffic never reaches a
   provider outside its allowed regions, through no fallback, retry or failure
   path. A property test generates policies and sweeps every tenant, alias and
   data class triple:
   [tests/routing/test_residency_property.py](tests/routing/test_residency_property.py).
2. **Personal data is replaced before it leaves.** Ten kinds are detected,
   checksum-validated where a shape alone is not enough, and replaced with
   stable placeholders:
   [tests/redaction/test_redactor.py](tests/redaction/test_redactor.py).
3. **A budget refuses rather than warns.** A tenant over its cap is refused
   before a provider is called, and an override has to be signed:
   [tests/limits/test_budget.py](tests/limits/test_budget.py).
4. **A rate limit is enforced per tenant.** The window's other traffic is part
   of the test, and so is the reset landing mid-sequence:
   [tests/limits/test_ratelimit.py](tests/limits/test_ratelimit.py).
5. **A cache hit is the same answer, for the same tenant.** The tenant, the
   alias, the model and the normalised body are all part of the key, so two
   tenants cannot collide: [tests/cache/test_key.py](tests/cache/test_key.py).
6. **A release that makes the agent worse does not merge.** The gate scores the
   golden set and compares it to the committed baseline:
   [tests/evals/test_gate.py](tests/evals/test_gate.py).
7. **Every token is attributed and priced at the rate in force.** Ledger rows
   name the price entry that priced them, and history is never rewritten:
   [tests/ledger/test_cost_ledger.py](tests/ledger/test_cost_ledger.py).

## What broke

Five things that went wrong while building this, and what each one changed.
Every standing rule below is one the build paid for.

- Failing tests went unseen for days because no run of the whole suite happened
  in that window. The fix was a blameless incident record, a visible unverified
  list, and a gate that has to be cleared before new work opens:
  [project-obs/postmortems/2026-09-17-red-tests-went-unseen.md](project-obs/postmortems/2026-09-17-red-tests-went-unseen.md).
- An identifier pattern anchored with `$` accepted a value with a trailing
  newline, because in Python `$` also matches before one. Every key and
  identifier pattern now uses `re.fullmatch`, and every rejection table carries
  a trailing newline, a CRLF and a space: the project notes
- A shell gate reported success while its middle step failed, because
  `A && B || C` exits zero when `B` does not. A gate is now reported as passing
  only from the exact command its definition names, never from a variation:
  the project notes
- The formatter removed an import between two edits and the next edit failed on
  the name it had just deleted. An import now lands in the same edit as its
  first use, and removing the last use removes the import in that edit:
  the project notes
- A local load run failed at startup because the companion project's stack was
  already publishing the same Redis, Postgres and Qdrant ports. Host ports are
  now listed and checked free before a stack is started:
  the project notes

## How the evals work

A golden set of cases lives in [project-evals/golden/](project-evals/golden/),
each case carrying its input, what a correct answer must contain, and what it
must refuse. The format is described in
[docs/eval-format.md](docs/eval-format.md).

A run resolves one alias against the real gateway, scores every case, and writes
a report: decision accuracy, refusal recall, citation accuracy, latency per turn
and USD per 1,000 turns. Scoring is deterministic where it can be, and
model-judged where it cannot; the judge's own agreement with human labels is
measured rather than assumed. The report shape is in
[docs/eval-report.md](docs/eval-report.md) and the target it runs against in
[docs/eval-target.md](docs/eval-target.md).

`project-evals/baseline.json` holds the committed numbers. The CI gate compares
a run against them and fails the pull request when a metric falls below its
floor, described in [docs/eval-gate.md](docs/eval-gate.md). The baseline changes
only through a commit that carries its own decision record, and never to make a
failing run pass. Metrics are percentages with an explicit null for "no
evidence", so an empty slice cannot read as a perfect score.

### Redaction accuracy

Accuracy is measured, not claimed. The numbers below are scored against a
fixture of 57 cases, 46 carrying at least one value to find and 11 carrying
look-alikes that must not be found, and regenerated by
`scripts/redaction_report.py`; a test fails if the table and the fixture
disagree. It is a small synthetic fixture, so read it as a floor the detectors
have to keep clearing rather than as a field measurement.

<!-- redaction-table:start -->
| Metric | Value |
| --- | --- |
| Precision | 1.000 |
| Recall | 1.000 |
<!-- redaction-table:end -->

Ten kinds are detected: email addresses, phone numbers, IBANs, card numbers,
German tax identification numbers, Irish PPS numbers, IPv4 and IPv6 addresses,
dates of birth, and the display names a tenant's operator has pushed to its own
directory. Where a shape alone is not enough the value has to pass its own
check, Luhn for cards, mod-97 for IBANs, ISO 7064 MOD 11,10 for tax ids, mod-23
for PPS numbers, so a sixteen-digit order reference is not reported as a card. A
date counts as a date of birth only when a cue such as "born" or "DOB" sits
close enough in front of it.

Redaction runs in two places. `POST /v1/redact` takes a string or a list of
messages and answers the redacted version plus a `redaction_map_id`, which is
the ingestion path: redact once on the way in and keep the id. The map behind it
is held for an hour and scoped to the tenant that made the call. To spend the
id, send it back as `X-Gateway-Rehydrate` on a chat completion and the answer's
text and tool call arguments come back with the original values in place. It is
off unless that header is present, an unknown or expired id is refused before a
provider is called rather than served with placeholders left in, and a streamed
answer is rehydrated too: a placeholder split across two frames is put back
together before it is sent. It also runs inline, so when a tenant's rule for the
request's data class says redact, every chat message and every embedding input
is redacted before the request leaves and the provider is only ever handed
placeholders. Either way, `X-Gateway-Redacted` on the response says whether a
value was actually replaced; it reports the payload, not the setting.

Placeholders are stable within a request: the same value appearing twice gets
the same placeholder and two different values never share one, so `<EMAIL_1>`
names the same person in the first message and the fourth. Text that already
contains a placeholder-shaped token is handled rather than ignored, because the
engine skips any number the input has already used.

## Self-hosted vs API tradeoff

One alias, `agent-default`, resolved four ways over the same golden set: a cheap
API tier, a strong API tier, the local CPU model on the box this project runs
on, and vLLM on a GPU instance applied for the run and destroyed after it. Each
option reports decision accuracy, refusal recall, citation accuracy, p50 and p95
latency per turn, and USD per 1,000 turns. API cost comes from the ledger;
self-hosted cost comes from the instance's hourly price divided by the
throughput it actually delivered.

How the runs are made, and what ten cases on one sequential stream can and
cannot tell you: [docs/tradeoff-run/README.md](docs/tradeoff-run/README.md). The
computed table and the winner it picks are written to `docs/tradeoff.md` by the
same harness once all four runs have landed.

**Conclusion and the cost of the measurement run: pending**, written once the
four runs have been made and the GPU instance's bill has arrived.

### The local model

A small quantised model is wired in as one of the providers the gateway routes
to, and it runs on the project's own ARM server. It exists for residency, not
for speed: a tenant whose data class allows only the box we run ourselves has no
other route, so the alternative to the local model is not a faster provider but
a refusal.

It is slower than a hosted API, and by how much is not stated here yet. Nothing
in CI can measure the machine it runs on, and a number produced anywhere else
would be a guess wearing a decimal point. The measured latency, throughput and
cost per token land in the tradeoff table above.

### Embeddings

`POST /v1/embeddings` speaks the OpenAI shape and answers from the same routing
policy every other request goes through, so an embedding is subject to the same
residency rules as a completion. In the shipped policy the `embed` alias has one
candidate and it is the local model, so for the data classes that allow nothing
else, the text being embedded never leaves the box we run.

A bare string or a list of strings is accepted, up to 64 inputs of 8192
characters each. The answer returns one vector per input, in the order given:
the index is how a caller matches vectors back to its own rows, so it is part of
the contract rather than a convenience. `encoding_format: "base64"` returns each
vector as little-endian float32, which is what the OpenAI SDK asks for by
default; the gateway renders it and always asks the provider for plain floats.

The endpoint needs a running local model server started with embeddings enabled,
because the server exposes its embeddings endpoint only when launched with
`--embedding`. That flag comes with the local stack deployment, so until that
lands this endpoint answers 502: the provider is configured and unreachable.

## The decisions

Every non-obvious choice was recorded when it was made, so a reader can see the
alternative that was rejected and why. The full set is in
[docs/adr/](docs/adr/); these are the ones that shape the system most.

- [Digests, not keys, are what the gateway stores](docs/adr/0001-sha256-for-api-key-digests.md)
- [Embeddings go through the routed local provider rather than a second path](docs/adr/0002-embeddings-through-the-routed-local-provider.md)
- [Circuit breaker state is in-process, not shared](docs/adr/0003-in-process-circuit-breaker-state.md)
- [The residency log write is off the request's critical path](docs/adr/0004-residency-log-write-is-not-on-the-request-s-critical-path.md)
- [Budget overrides are signed with their own scoped secret](docs/adr/0005-budget-overrides-are-signed-with-their-own-scoped-secret.md)
- [Eval metrics are percentages with a null for no evidence](docs/adr/0007-eval-metrics-are-percentages-with-a-null-for-no-evidence.md)
- [The eval gate ships inert without a target](docs/adr/0009-the-eval-gate-ships-inert-without-a-target.md)
- [Terraform Cloud holds the state, not a bucket we operate](docs/adr/0011-terraform-cloud-holds-the-hetzner-state.md)
- [Images are named by commit and scanned after push](docs/adr/0013-images-are-named-by-commit-and-scanned-after-push.md)
- [The metrics stack is hand-written rather than the upstream chart](docs/adr/0015-the-metrics-stack-is-hand-written-rather-than-the-upstream-chart.md)
- [Console sources are flags, never errors](docs/adr/0022-console-sources-are-flags-never-errors.md)
- [Console pages render server-side and degrade to a notice](docs/adr/0024-console-pages-render-server-side-and-degrade-to-a-notice.md)

The threat model that sits behind several of them, STRIDE over the system's
trust boundaries with one row per real threat, is in
[docs/threat-model.md](docs/threat-model.md).

## SLOs

Seven objectives, each with its indicator, target, window, error budget formula
and alert: [project-obs/slo.md](project-obs/slo.md). Three are measured from the
gateway's own metrics (availability, latency, residency) and show their
remaining error budget on the Grafana overview dashboard and on the console's
Reliability page. An error budget is never clamped at zero: how far past the
allowance a month went is the information.

Every alert in [project-obs/alerts/](project-obs/alerts/) has a response section
in [docs/runbook.md](docs/runbook.md). Incidents are recorded on the console's
timeline, each linking its postmortem.

## Scaling

One run of `make load`, which drives k6 against the real gateway at a constant
50 requests a second for ten minutes. The providers behind it are a mock that
answers in a fixed time, so every millisecond above that is the gateway's own
work: auth, routing, residency, redaction, both caches, rate limits, budgets and
the ledger write. The traffic is 80% chat, 10% streamed chat and 10%
embeddings, and one chat request in five repeats a prompt so the caches take
part.

On an Apple M1 (8 cores, Docker Desktop with 10 GB), one gateway worker:

| Metric | Value |
| --- | --- |
| Requests | 30,001 in 10 min |
| Achieved rate | 49.98 req/s |
| Dropped iterations | 0 |
| Error rate | 0.00% |
| Chat p50 / p95 / p99 | 438 / 497 / 583 ms |
| Mock delay, chat | 400 ms (streamed: 10 chunks, 40 ms apart) |
| Mock delay, embeddings | 20 ms |

With 400 ms of mock delay the gateway adds about 38 ms at the median and 97 ms
at p95. Requests answered from the cache come back in about 25 ms. The run fails
if chat p95 reaches 6 s, if more than 0.5% of requests fail, or if k6 drops a
single iteration because the gateway could not keep up.

Horizontally, the gateway is stateless except for the circuit breaker, whose
state is deliberately per-process; Redis holds the caches, the rate limit
windows and the redaction maps, and Postgres holds the ledger. Adding a worker
adds throughput without changing any of those.

## Quickstart

Requires Docker, [uv](https://docs.astral.sh/uv/) and pnpm.

    git clone https://github.com/masudur-rahman-musa/ai-operations-console
    cd ai-operations-console
    cp .env.example .env        # fill in the provider keys you have
    docker compose up -d        # gateway, Redis, Postgres, Qdrant
    uv sync

The local llama.cpp model servers are off by default. To start them too:

    docker compose --profile local-model up -d

Then send it an OpenAI-shaped request, naming an alias rather than a model:

    curl http://localhost:8080/v1/chat/completions \
      -H "Authorization: Bearer $GATEWAY_KEY" \
      -H "X-Tenant: acme" \
      -H "Content-Type: application/json" \
      -d '{"model":"agent-default","messages":[{"role":"user","content":"hello"}]}'

`/healthz` says the process is up and `/readyz` says its dependencies are
reachable. The console is at `http://localhost:3000`. The variables every
service reads are listed with placeholders in [.env.example](.env.example).

To run this as your own platform, see [CUSTOMIZING.md](CUSTOMIZING.md): the only
two files you have to edit are the routing policy and the price table.

## Azure

The Azure OpenAI path is an adapter plus recorded fixtures. It ships disabled,
and enabling it is a configuration change rather than a code change:
[docs/azure.md](docs/azure.md).

## AWS

The Hetzner deployment is the one that runs. An equivalent EKS deployment was
applied once as evidence that the manifests are portable, then destroyed, and
the plan output and its cost are recorded in
[docs/aws-evidence/](docs/aws-evidence/).
