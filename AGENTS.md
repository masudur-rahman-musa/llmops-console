# AGENTS.md

A short guide for a coding agent, or a person, working on this repo.

## What this is

An LLM gateway and an operations console. Every model call goes through the
gateway, which routes it by alias, caches it, redacts personal data, checks the
budget and rate limit, and records its cost and latency. The console shows cost,
quality and health. An eval gate checks agent quality on every pull request.

## Where things live

- `backend/app/` is the gateway (FastAPI). Routing, providers, cache, redaction,
  limits, storage, evals and the console API each have a folder.
  [ARCHITECTURE.md](ARCHITECTURE.md) says what each one holds.
- `backend/config/` holds `routing.yaml` (aliases and tenants) and one file per
  provider in `providers/`. These are the only files that name a real model.
- `backend/migrations/` holds the numbered SQL migrations for the ledger.
- `backend/evals/` holds the eval baseline, reference labels, the redaction
  held-out templates and a stub eval target.
- `backend/tests/` holds the backend tests, one folder per part of `backend/app/`.
- `web/` is the console, a Next.js app with its tests in `web/tests/`.
- `infra/` holds the Kubernetes manifests, Terraform, the local monitoring config,
  the load test and the tests in `infra/tests/` that check all of it.
- `docs/` holds the guides. Start with [docs/local-setup.md](docs/local-setup.md).

## Commands

- `make setup` installs the Python and web dependencies and the pre-commit hooks.
- `make up` starts the whole stack in Docker. `make down` stops it.
- `make fast` is the everyday test run. It needs no services.
- `make test` runs every backend, infra and web test with the coverage floors. It
  needs `docker compose up -d` first.
- `make lint`, `make typecheck` and `make format` run ruff, mypy and the web
  linters.
- `make verify` runs everything CI checks, in order.
- `make eval` runs the eval gate. It needs the companion help desk checkout and an
  `ANTHROPIC_API_KEY`, and it spends real tokens.
- `make help` lists every other target.

## Rules the code follows

- **Tests prove the change.** Write the test first, from what the change must do,
  including what it must refuse. Never edit a test to make it pass.
- **Local equals CI.** Every CI check is a `make` target, and the workflows call
  the target. A new check lands with its target.
- **Types everywhere.** `mypy --strict` must pass. Data crossing a layer is a
  Pydantic model, not a raw dict.
- **Typed errors.** Every refusal is one of the error codes in
  `backend/app/core/errors.py`, never free text.
- **Aliases only.** Code never names a provider model. Add or change a model in
  `backend/config/` and let the eval gate check it.
- **No secrets in the repo.** Keys arrive through env files git ignores or through
  SOPS-encrypted secrets. Tenant and admin keys are stored as sha256 digests.
- **Identity comes from the key.** Never trust a tenant or user named in a request
  body.
- **The eval baseline is not edited to pass a run.** A change to
  `backend/evals/conversation-baseline.json` needs a commit whose subject starts
  with `baseline:` and a written decision record, and CI checks this.
- **Fast tests.** Logic is tested with fakes and no running service. A test that
  needs Postgres, Redis or Qdrant carries the `db` marker. A slow one carries
  `slow`.
- **Plain comments.** Say what the code does and why. Comments never point at
  private notes or task numbers.
- **Never apply infrastructure.** `terraform apply` and `kubectl apply` against a
  real environment are for a person to run, never an agent.

## Commit style

Small commits with a plain imperative subject that says what the change does.
