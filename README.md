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

**Status: under active development.** Architecture docs, dashboards, and measured self-hosted-vs-API tradeoffs land as milestones complete.

Companion project: [ai-help-desk-governed-access](https://github.com/developermusa/ai-help-desk-governed-access) — the agent workload this platform routes, measures, and evaluates.

License: Apache-2.0
