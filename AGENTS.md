# AGENTS.md

A short guide for any coding agent, or person, picking up this project.

## What this is

An LLM gateway and operations console. Every model call goes through the
gateway, which routes it, caches it, redacts personal data, counts its cost
against a budget and records it. The console shows cost, quality and health,
and an eval gate checks quality on every pull request.

## Where things live

- `backend/app/`: the gateway (FastAPI): routing, providers, cache, redaction, limits and the console API.
- `backend/migrations/`, `backend/evals/`: the ledger schema and the eval tooling.
- `web/`: the console, a Next.js dashboard.
- `infra/`: Kubernetes manifests, Terraform, monitoring and load tests.
- `docs/`: guides; `ARCHITECTURE.md` has the big picture.

## How to work on it

Every change, a feature or a fix, goes through the same loop:

1. **Plan it.** Write down what changes and why, which files it touches, and
   the acceptance criteria: the checks that prove it works, including what must
   be refused or must fail safely.
2. **Write the tests first.** One test per acceptance criterion. They fail at
   this point, because the code does not exist yet. A coding agent can write
   them straight from the criteria.
3. **Make the smallest change that turns them green.** Touch nothing the plan
   does not name.
4. **Run `make verify`.** It runs the same checks CI runs, so a green run here
   means a green run there.
5. **Commit** with a plain message that says what the change does.

Keep your plan files out of git, for example in a folder git ignores. The
repository holds the code, the tests and the history of each change.

## Commands

- `make setup`: install dependencies and the pre-commit hooks.
- `make up`: start the whole stack in Docker.
- `make fast`: the everyday test run, no backing services needed.
- `make verify`: everything CI checks, in order.
- `make help`: every other target.
