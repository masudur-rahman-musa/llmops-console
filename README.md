# AI Operations Console

The dashboard that tells you whether your AI is working, what it costs, and where your data went.

Underneath: an OpenAI-compatible LLM gateway with per-tenant routing and EU data-residency enforcement, PII redaction, semantic caching, budgets, cost attribution — and an evaluation harness whose CI gate blocks any release that makes the agent worse. Deployed with Terraform and Kubernetes, operated with SLOs, alerts, and a runbook.

**Status: under active development.** Architecture docs, dashboards, and measured self-hosted-vs-API tradeoffs land as milestones complete.

Companion project: [ai-help-desk-governed-access](https://github.com/developermusa/ai-help-desk-governed-access) — the agent workload this platform routes, measures, and evaluates.

License: Apache-2.0
