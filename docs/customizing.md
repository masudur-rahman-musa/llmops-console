# Customizing

Clone the repo, point it at your own providers, and you have a gateway that
routes, redacts, caches, prices and measures your own traffic.

To make it yours there are only two things you must edit, `backend/config/routing.yaml` and the provider files in `backend/config/providers/`.
Everything else has a working default. The later sections cover the settings you
may want later, not the ones you need now.

## Your providers, their models and what a token costs

The folder is `backend/config/providers/`. It holds one YAML file per provider,
named after the provider. The shipped ones are
`backend/config/providers/anthropic.yaml` and `backend/config/providers/openai.yaml`
for chat, and `backend/config/providers/local_embed.yaml` for the local embedding
server. A new one looks
like this:

    provider: mistral             # must match the file name
    kind: openai_compatible       # anthropic, openai, openai_compatible or azure_openai
    base_url: https://api.mistral.ai/v1
    key_env: MISTRAL_API_KEY      # the name of the variable, never the key
    enabled: true
    models:
      mistral-small:
        prices:
          - effective_from: 2026-10-01
            input_usd_per_1m: "0.20"
            output_usd_per_1m: "0.60"

The prices in this example are made up. Use the ones in your own contract.

To add a model, drop in a file like this one and add an alias line in
`backend/config/routing.yaml`. No code changes. The gateway reads every file at
startup and again on each routing reload. A bad file stops startup with a message
naming the file and the field. On a reload it is refused, and the old set keeps
serving. A provider whose key variable is unset or empty is left out with a
warning, so it is never picked and then fails.

Model changes go through a pull request and the eval gate, never a live settings
page. Azure is the one kind that also needs an endpoint variable, and
`docs/azure.md` shows how.

Prices are in USD per 1,000,000 tokens, written as quoted strings so the digits
stay exact. Each model holds a list of prices, each with its own
`effective_from` date. A price change appends a row and never edits one, so a
request served last month keeps the price it was charged. Cached prompt tokens
can carry their own read and write prices. The cost ledger, the budgets and the
tradeoff table all read their numbers from here, so until these files carry your
prices, every cost the console shows is somebody else's.

## Who answers which alias

`backend/config/routing.yaml` holds the aliases your callers name and the
tenants. Providers are not declared here, and a `providers:` block is refused.

What you change:

- **Aliases.** Your callers ask for an alias, not a model. Give each alias the
  providers to try, in order. The first is the default and the rest are
  fallbacks. The shipped chat aliases list Anthropic, then OpenAI. Each alias has
  a one-line `purpose`, and `semantic_cache: true` turns the semantic cache on for
  it.
- **Tenants.** One entry per tenant, with a redact rule for each data class
  (`public`, `internal`, `confidential`, `pii`), a monthly budget, an optional
  per-user budget, a rate limit in requests and tokens per minute, and the trace
  switches `trace_content` and `trace_public`.

Fallback stays inside the alias. When a provider fails, the gateway tries only the
next candidate the alias lists for that tenant. The test is
`backend/tests/api/test_fallback.py`.

The config files are built into the gateway image, so after an edit run
`make rebuild gateway` locally. The gateway also has an admin route that reloads
these files without a restart, for a deployment that mounts them.

## The rest, in the order you will meet it

You do not have to touch any of these to run the gateway.

- **Keys and settings.** `backend/.env.example` lists every variable the gateway
  reads. Copy it to a `.env` file in `backend/` and fill it in, as `docs/local-setup.md`
  shows. For a cluster, the same values live SOPS-encrypted under
  `infra/secrets/`.
- **Running it.** `docker-compose.yml` brings up the whole local stack, and the
  `Makefile` carries the commands the workflows use.
- **Evaluations.** The eval engine plays conversation cases, meaning what a
  person types, which tools should be called, and when the person says yes. Write
  your cases in the shape `docs/eval-format.md` describes, and serve them through
  the two endpoints in `docs/eval-target.md`.
  `backend/evals/conversation-baseline.json` is the score the CI gate compares
  against. Make it once from a run on your own cases, then change it only with a
  reason. The gate itself is in `docs/eval-gate.md`.
- **Reliability.** The alert rules are in `infra/k8s/monitoring/alerts/`. The two
  SLO numbers, 99.5% availability and 95% of requests under 6 seconds, are in
  `backend/app/services/prometheus_reads.py` and the Grafana gateway dashboard. Set them to what you really
  promise.
- **Infrastructure.** `infra/terraform/` provisions the hosts and `infra/k8s/`
  holds the manifests. Both expect your own domain, state backend and registry.
- **The console.** `web/` is the dashboard. It reads through the gateway's console
  API and holds only its login settings and the admin key.

## Defaults you are taking on

If a default looks wrong for you, read the comment beside it. It says why it is
the default, which is the fastest way to decide whether to change it.
