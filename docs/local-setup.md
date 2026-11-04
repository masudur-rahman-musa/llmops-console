# Run it on your laptop

This page takes a fresh clone to a running gateway, console and Grafana, all in
Docker on your machine.

## What you need

Docker with Compose 2.24 or newer, `make`, `curl`, `openssl`, `shasum` and
[uv](https://docs.astral.sh/uv/). Node 22 and pnpm only for the console tests.

These ports must be free. 8080 (gateway), 3001 (console), 3002 (Grafana), 9090
(Prometheus), 9091 (Pushgateway), 8083 (embeddings), 15432 (Postgres), 16379 (Redis)
and 16333 (Qdrant).

Only chat, the judge, `make eval` and `make numbers` need `ANTHROPIC_API_KEY`.
`OPENAI_API_KEY` is the optional fallback. Everything else works with no key.

## 1. Clone and fetch the embedding model

    git clone https://github.com/masudur-rahman-musa/llmops-console
    cd llmops-console
    make embed-model

This downloads the model into `.cache/models/` once and checks its sha256.

## 2. Make the gateway keys

The gateway keeps only the sha256 digest of each key. Make one tenant key and one
admin key.

    tenant_key=$(openssl rand -hex 32)
    admin_key=$(openssl rand -hex 32)
    echo "tenant key  $tenant_key"
    echo "admin key   $admin_key"
    echo "OPSCONSOLE_API_KEYS=nordlicht:$(printf %s "$tenant_key" | shasum -a 256 | cut -d' ' -f1)"
    echo "OPSCONSOLE_ADMIN_KEY_SHA256=$(printf %s "$admin_key" | shasum -a 256 | cut -d' ' -f1)"

`make tenant-key TENANT=nordlicht` does the tenant half for you. The tenant must
be one that `backend/config/routing.yaml` defines (`nordlicht`, `cuan` or `evals`).

## 3. Write the env files

    cp backend/.env.example backend/.env

In `backend/.env`, set `ANTHROPIC_API_KEY` if you have one. Paste the two lines
printed above over the empty `OPSCONSOLE_API_KEYS=` and
`OPSCONSOLE_ADMIN_KEY_SHA256=` lines.

The console has a password login. Write its settings to `web/.env.local`.

    password='pick-a-local-password'
    {
      echo "CONSOLE_PASSWORD_SHA256=$(printf %s "$password" | shasum -a 256 | cut -d' ' -f1)"
      echo "CONSOLE_SESSION_SECRET=$(openssl rand -hex 32)"
      echo "CONSOLE_ADMIN_KEY=$admin_key"
    } > web/.env.local

Git ignores both files. `backend/.env.example` shows how to make the optional
budget override secret and eval judge key. To send traces to Langfuse, make an API
key pair in a Langfuse project and add these two lines to `backend/.env`.

    pk='pk-lf-...'
    sk='sk-lf-...'
    echo "OPSCONSOLE_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces"
    echo "OPSCONSOLE_OTLP_HEADERS=Authorization=Basic $(printf %s "$pk:$sk" | base64 | tr -d '\n')"

## 4. Start everything

    make up

This starts the whole stack. Migrations run when the gateway starts. To run them
by hand (safe to run twice):

    docker compose exec gateway ops-migrate

`docker compose up -d` starts everything except the console. `make run` runs the
gateway on this machine, reloading on save. `make down` stops the stack and keeps
its data. `make rebuild gateway` rebuilds the gateway after a code change.

## 5. Check it

    curl -s localhost:8080/healthz
    curl -s localhost:8080/readyz

Send a chat request. It names an alias, never a model.

    curl -s localhost:8080/v1/chat/completions \
      -H "Authorization: Bearer $tenant_key" \
      -H "X-Tenant: nordlicht" \
      -H "Content-Type: application/json" \
      -d '{"model":"agent-default","messages":[{"role":"user","content":"Hi, I am at jane.doe@example.com"}]}' \
      -D - -o /dev/null

The `X-Gateway-*` response headers show the provider, model, cost, trace id and
whether a value was redacted. Redaction alone needs no model key.

    curl -s localhost:8080/v1/redact \
      -H "Authorization: Bearer $tenant_key" \
      -H "X-Tenant: nordlicht" \
      -H "Content-Type: application/json" \
      -d '{"text":"Mail the report to sam.lee@example.com"}'

Open the console at http://localhost:3001 with your password, Grafana at
http://localhost:3002 and Prometheus at http://localhost:9090. The console fills
from real traffic. `make numbers` fills every tab at once. It needs the help desk
checkout (see below) and `ANTHROPIC_API_KEY`, and it spends real tokens.

## Run the tests

    make fast            # every test not marked slow, no services needed
    docker compose up -d
    make test            # the full suite with the services, plus coverage floors
    make verify          # format, lint, types, tests and security scans, as CI runs them
    pnpm -C web install
    pnpm -C web test     # the console tests
    make load            # k6 against a mock provider, no model key needed

## Run an eval

`make eval` plays the help desk's 42 cases against the real help desk agent. It
needs the companion repo checked out beside this one. The Makefile looks in
`../helpdesk-agentic-flow`, and `HELPDESK_DIR` changes that. It also needs
`ANTHROPIC_API_KEY` and the help desk's ports free (5432, 6379, 6333, 8000, 3000).

    make eval
    RUN_ID=<id> make eval     # continue a stopped run without paying again
    uv run --project backend ops-eval gate \
      --report backend/evals/reports/local/report.json \
      --baseline backend/evals/baseline.json

A run spends real tokens. How it works is in [eval-guide.md](eval-guide.md).

## When something goes wrong

- **The gateway exits at start.** Read `docker compose logs gateway`.
- **Every call is 401.** The key and `X-Tenant` must match `OPSCONSOLE_API_KEYS`.
- **Chat is 503.** Set `ANTHROPIC_API_KEY`, then `make rebuild gateway`.
- **The console says data is unavailable.** `CONSOLE_ADMIN_KEY` must match
  `OPSCONSOLE_ADMIN_KEY_SHA256`.
