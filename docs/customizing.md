# Customizing

This repository is a GitHub template. Clone it, point it at your own providers,
and you have a gateway that routes, enforces residency, redacts, caches, prices
and measures your own traffic.

To make it yours there are only two files you must edit: `routing.yaml` and `prices.yaml`.
Everything else has a working default. The sections after those two describe the
knobs you may reach for later rather than the ones you have to reach for now.

## `routing.yaml` — who your providers are and who may reach them

This file holds the whole policy: the providers that exist, the aliases your
callers name, and which provider answers which alias, per tenant and per data
class. It is also the only shipped file that contains a provider model id.

What you change:

- **Providers.** Replace the shipped rows with your own. A row names a `kind`
  (`anthropic`, `openai_compatible`, `azure_openai`), a `region`, and the
  environment variable holding the key as `key_env`. Keys are never written
  here; only the name of the variable is. See [docs/azure.md](docs/azure.md)
  for the one provider kind that needs an endpoint variable too.
- **Aliases.** Your callers ask for an alias, not a model. Give each alias the
  candidates you want tried, in order.
- **Tenants.** One row per tenant, each with its residency region and its rule
  per data class: which regions are allowed, and whether the request is
  redacted on the way out.
- **`eu_regions`.** The region set an EU tenant is intersected against.

Residency is deny by default, and the resolver never widens a region set on
failure — an empty result is a refusal, not a fallback. That behaviour is not
configurable, and a property test over generated policies keeps it that way:
[tests/routing/test_residency_property.py](tests/routing/test_residency_property.py).

## `prices.yaml` — what a token costs you

The cost ledger, the budgets, the savings figures and the tradeoff table all
read their numbers from here. Until this file carries your contract's prices,
every cost the console shows is somebody else's.

What you change: one entry per provider-and-model key, in USD per 1,000,000
tokens, keyed by the provider name as you spelled it in the routing policy.

Each key holds a list of entries with their own `effective_from` date, so
re-pricing means appending a row, never editing one — a request served last
month keeps the number it was actually charged. Historical ledger rows are
never rewritten.

## The rest, in the order you will meet it

You do not have to touch any of these to run the gateway.

- **Credentials.** [.env.example](.env.example) lists every variable the stack
  reads, with placeholders. Copy it to your own env file and fill it in. For
  the cluster, the same values live SOPS-encrypted under
  [infra/secrets/](infra/secrets/).
- **Running it.** `docker-compose.yml` brings the whole local stack up, and the
  [Makefile](Makefile) carries the commands the workflows use.
- **Evaluations.** Your golden cases go in [project-evals/golden/](project-evals/golden/)
  and the shape they take is described in [docs/eval-format.md](docs/eval-format.md).
  `project-evals/baseline.json` is the scoreboard the CI gate compares against;
  regenerate it once against your own cases, then leave it alone.
- **Reliability.** The seven service level objectives are in
  the project notes their alerts in
  [infra/k8s/monitoring/alerts/](infra/k8s/monitoring/alerts/), and every alert has a response
  section in the project notes Retarget the numbers to what
  you are actually promising.
- **Infrastructure.** [infra/terraform/](infra/terraform/) provisions the hosts
  and [infra/k8s/](infra/k8s/) holds the manifests. Both expect your own domain,
  your own state backend and your own registry.
- **The console.** [web/](web/) is the dashboard. It
  reads through the gateway's console API and holds no credentials of its own.

## Decisions you are inheriting

Every non-obvious choice in here was recorded when it was made, in
the project notes If a default looks wrong for you, the record of why it
is the default is the fastest way to decide whether to change it.
