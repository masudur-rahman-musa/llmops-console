# Customizing

This repository is a GitHub template. Clone it, point it at your own providers,
and you have a gateway that routes, redacts, caches, prices
and measures your own traffic.

To make it yours there are only two things you must edit: `backend/config/routing.yaml` and the provider files in `backend/config/providers/`.
Everything else has a working default. The sections after those two describe the
knobs you may reach for later rather than the ones you have to reach for now.

## `backend/config/providers/` — your providers, their models and what a token costs

One YAML file per provider, named after the provider. The shipped ones are
[anthropic.yaml](backend/config/providers/anthropic.yaml) and
[openai.yaml](backend/config/providers/openai.yaml) for chat, and
[local_embed.yaml](backend/config/providers/local_embed.yaml) for the local
embedding server. Each file holds:

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

To add a model, drop in a file like this one and add an alias line in
`backend/config/routing.yaml`. No code changes. The gateway reads every file at
startup and again on each routing reload. A bad file stops startup with a message
naming the file and the field; on reload it is refused and the old set keeps
serving.
Model changes go through a pull request and the eval gate, never a live
settings page. See [docs/azure.md](docs/azure.md) for the one kind that needs an
endpoint variable too.

Prices are in USD per 1,000,000 tokens. Each model holds a list of prices with
their own `effective_from` date, so re-pricing means appending a row, never
editing one: a request served last month keeps the number it was actually
charged. The cost ledger, the budgets, the savings figures and the tradeoff
table all read their numbers from here, so until these files carry your
contract's prices, every cost the console shows is somebody else's.

## `backend/config/routing.yaml` — who answers which alias

This file holds the aliases your callers name and the tenants. Providers are
not declared here; a `providers:` block is refused.

What you change:

- **Aliases.** Your callers ask for an alias, not a model. Give each alias the
  providers you want tried, in order: the first is the default, the rest are
  fallbacks. The shipped chat aliases list Anthropic, then OpenAI.
- **Tenants.** One row per tenant, each with its rule per data class: whether
  the request is redacted on the way out.

Fallback stays inside the alias: when a provider fails, the gateway tries only
the next candidate the alias lists for that tenant:
[backend/tests/api/test_fallback.py](backend/tests/api/test_fallback.py).

## The rest, in the order you will meet it

You do not have to touch any of these to run the gateway.

- **Credentials.** [backend/.env.example](backend/.env.example) lists every variable the stack
  reads, with placeholders. Copy it to your own env file and fill it in. For
  the cluster, the same values live SOPS-encrypted under
  [infra/secrets/](infra/secrets/).
- **Running it.** `docker-compose.yml` brings the whole local stack up, and the
  [Makefile](Makefile) carries the commands the workflows use.
- **Evaluations.** The eval engine plays conversation cases: what a person types,
  which tools should be called, and when the person says yes. Write your cases in the
  shape [docs/eval-format.md](docs/eval-format.md) describes, and serve them through
  the two endpoints in [docs/eval-target.md](docs/eval-target.md).
  `backend/evals/conversation-baseline.json` is the scoreboard the CI gate compares
  against; make it once from a run on your own cases, then leave it alone. The gate
  itself is in [docs/eval-gate.md](docs/eval-gate.md).
- **Reliability.** The seven service level objectives have their alerts in
  [infra/k8s/monitoring/alerts/](infra/k8s/monitoring/alerts/). Retarget the numbers to what
  you are actually promising.
- **Infrastructure.** [infra/terraform/](infra/terraform/) provisions the hosts
  and [infra/k8s/](infra/k8s/) holds the manifests. Both expect your own domain,
  your own state backend and your own registry.
- **The console.** [web/](web/) is the dashboard. It
  reads through the gateway's console API and holds no credentials of its own.

## Decisions you are inheriting

If a default looks wrong for you, read the comments beside it: they say why
it is the default, which is the fastest way to decide whether to change it.
