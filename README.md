# AI Operations Console

The dashboard that tells you whether your AI is working, what it costs, and where your data went.

Underneath: an OpenAI-compatible LLM gateway with per-tenant routing and EU data-residency enforcement, PII redaction, semantic caching, budgets, cost attribution — and an evaluation harness whose CI gate blocks any release that makes the agent worse. Deployed with Terraform and Kubernetes, operated with SLOs, alerts, and a runbook.

## The local model

A small quantised model is wired in as one of the providers the gateway routes
to, and it will run on the project's own ARM server; the deployment that starts
it lands with the rest of the local stack. It exists for residency, not for
speed: a tenant whose data class allows only the box we run ourselves has no
other route, so the alternative to the local model is not a faster provider but
a refusal.

It is slower than a hosted API, and by how much is not stated here yet. Nothing
in CI can measure the machine it runs on, and a number produced anywhere else
would be a guess wearing a decimal point. The measured latency, throughput and
cost per token land in the self-hosted-vs-API tradeoff table when that milestone
completes.

## Embeddings

`POST /v1/embeddings` speaks the OpenAI shape and answers from the same routing
policy every other request goes through, so an embedding is subject to the same
residency rules as a completion. In the shipped policy the `embed` alias has one
candidate, and it is the local model on the box we run: for the data classes
that allow nothing else, the text being embedded never leaves it.

A bare string or a list of strings is accepted, up to 64 inputs of 8192
characters each. The answer returns one vector per input, in the order given -
the index is how a caller matches vectors back to its own rows, so it is part of
the contract rather than a convenience. `encoding_format: "base64"` returns each
vector as little-endian float32, which is what the OpenAI SDK asks for by
default; the gateway renders it and always asks the provider for plain floats.

The endpoint needs a running local model server, and one started with embeddings
enabled: the server exposes its embeddings endpoint only when launched with
`--embedding`. That flag comes with the local stack deployment, so until that
lands this endpoint answers 502 - the provider is configured and unreachable.

## Redaction

Free text is scanned for personal data and every value found is replaced with a
stable placeholder such as `<EMAIL_1>`.

It runs in two places. `POST /v1/redact` takes a string or a list of messages and
answers the redacted version plus a `redaction_map_id`, which is the ingestion
path: redact once on the way in and keep the id. The map behind it is held for an
hour and scoped to the tenant that made the call. To spend the id, send it back
as `X-Gateway-Rehydrate` on a chat completion: the answer's text and its tool
call arguments come back with the original values in place of the placeholders.
It is off unless that header is present, an unknown or expired id is refused
before a provider is called rather than served with placeholders left in, and a
streamed answer is rehydrated too - a placeholder split across two frames is put
back together before it is sent. And it runs
inline: when a tenant's rule for the request's data class says `redact`, every
chat message and every embedding input is redacted before the request leaves for
a provider, so the provider is only ever handed placeholders. Either way,
`X-Gateway-Redacted` on the response says whether a value was actually replaced -
it reports the payload, not the setting.

Ten kinds are detected: email addresses, phone numbers, IBANs, card numbers, German
tax identification numbers, Irish PPS numbers, IPv4 and IPv6 addresses, dates
of birth, and the display names a tenant's operator has pushed to its own directory. Where a shape alone is not enough the value has to pass its own check -
Luhn for cards, mod-97 for IBANs, ISO 7064 MOD 11,10 for tax ids, mod-23 for PPS
numbers - so a sixteen-digit order reference is not reported as a card. A date
counts as a date of birth only when a cue such as "born" or "DOB" sits close
enough in front of it.

Placeholders are stable within a request: the same value appearing twice gets the
same placeholder, and two different values never share one, so the redacted text
can be turned back into the original from the map the call returns - one map and
one numbering per request, so `<EMAIL_1>` names the same person in the first
message and the fourth. Text that
already contains a placeholder-shaped token is handled rather than ignored - the
engine skips any number the input has already used, so turning the text back is
exact even when someone pastes redacted output straight back in.

Accuracy is measured, not claimed. The numbers below are scored against a fixture
of 57 cases - 46 carrying at least one value to find, 11 carrying look-alikes that
must not be found - and regenerated by `scripts/redaction_report.py`; a test fails
if the table and the fixture disagree. It is a small synthetic fixture, so read it
as a floor the detectors have to keep clearing rather than as a field measurement.

<!-- redaction-table:start -->
| Metric | Value |
| --- | --- |
| Precision | 1.000 |
| Recall | 1.000 |
<!-- redaction-table:end -->

## Load

`make load` runs k6 against the real gateway at a constant 50 requests a second
for ten minutes. The providers behind it are a mock that answers in a fixed
time, so every millisecond above that is the gateway's own work: auth, routing,
residency, redaction, both caches, rate limits, budgets and the ledger write.
The traffic is 80% chat, 10% streamed chat and 10% embeddings, and one chat
request in five repeats a prompt so the caches take part.

The run fails if chat p95 reaches 6 s, if more than 0.5% of requests fail, or if
k6 drops a single iteration because the gateway could not keep up. The same run
can be started by hand from the `load` workflow in GitHub Actions.

One run, on an Apple M1 (8 cores, Docker Desktop with 10 GB), one gateway worker:

| Metric | Value |
| --- | --- |
| Requests | 30,001 in 10 min |
| Achieved rate | 49.98 req/s |
| Dropped iterations | 0 |
| Error rate | 0.00% |
| Chat p50 / p95 / p99 | 438 / 497 / 583 ms |
| Mock delay, chat | 400 ms (streamed: 10 chunks, 40 ms apart) |
| Mock delay, embeddings | 20 ms |

With 400 ms of mock delay, the gateway adds about 38 ms at the median and 97 ms
at p95. Requests answered from the cache come back in about 25 ms.

## Reliability

Seven SLOs, each with its indicator, target, window, error budget formula and
alert: [project-obs/slo.md](project-obs/slo.md). The three the gateway measures
itself (availability, latency, residency) show their remaining error budget on
the gateway overview dashboard in Grafana and on the console's Reliability page.

Incidents are recorded on the console's timeline, each linking its postmortem.
The first one, blameless, covers failing tests that went unseen for days because
no run of the whole suite happened in that time:
[project-obs/postmortems/2026-09-17-red-tests-went-unseen.md](project-obs/postmortems/2026-09-17-red-tests-went-unseen.md).

**Status: under active development.** Architecture docs, dashboards, and measured self-hosted-vs-API tradeoffs land as milestones complete.

Companion project: [ai-help-desk-governed-access](https://github.com/developermusa/ai-help-desk-governed-access) — the agent workload this platform routes, measures, and evaluates.

License: Apache-2.0

## Self-hosted vs API tradeoff

One alias, `agent-default`, resolved four ways over the same golden set: a cheap
API tier, a strong API tier, the local CPU model on the box this project runs on,
and vLLM on a GPU instance applied for the run and destroyed after it. Each option
reports decision accuracy, refusal recall, citation accuracy, p50 and p95 latency
per turn, and USD per 1,000 turns. API cost comes from the ledger; self-hosted
cost comes from the instance's hourly price divided by the throughput it actually
delivered.

The table, the winner it computes and the methodology behind every row:
[docs/tradeoff.md](docs/tradeoff.md). How the runs are made, and what ten cases on
one sequential stream can and cannot tell you: [docs/tradeoff-run/README.md](docs/tradeoff-run/README.md).

**Conclusion and the cost of the measurement run: PENDING** — written once the
four runs have been made and the GPU instance's bill has landed.
