# project-obs (AI layer)

Runtime visibility into the PRODUCT (coding obs is the work; this is the system).
Filled by /setup from the context; no stub may remain.

- **Traces:** one trace per request, one span per step (retrieve, model call,
  each tool, compose). Attributes: session id, prompt_version, model alias,
  cost, tokens, refused flag. Backend: ADAPT (Langfuse / OpenTelemetry).
- **Metrics:** requests, errors, p50/p95 latency, tokens, cost per request,
  cache hits, refusals. Exposed at `/metrics`. Dashboards: ADAPT.
- **Logs:** structured JSON with the trace id on every line.
- **Health:** `/healthz` (process up) and `/readyz` (dependencies reachable).
- **Identity rule:** telemetry carries session and tenant, never user identity.
- **Alerts:** ADAPT the three that matter (error rate, p95, cost per day).
- **Eval drift:** nightly `make eval` results charted next to release markers.

## Layout
- `README.md` — this checklist: what is traced, measured, logged, alerted. Filled by /setup; no stub may remain.
- `dashboards/` — dashboard definitions as code (e.g. Grafana JSON), provisioned by infra, never hand-edited in a UI only.
- `alerts/` — alert rules as code, tied to the SLOs.
Audit the codebase against this folder with `/project-obs`.

## AI additions
- Spans per model call and per tool call; `prompt_version`, model alias, tokens, cost, and refused flag as attributes.
- Nightly `make eval` results charted next to release markers (eval drift).
- Cost per request and per day on the dashboard; a budget alert.
