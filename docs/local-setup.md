# Run it on your laptop

This page takes a fresh clone to a running gateway, console and Grafana. Everything
runs in Docker on your own machine. Nothing here touches a cloud account.

## What you need

- Docker with Compose v2. The compose file marks its env files as optional, which
  needs Compose 2.24 or newer.
- `make`, `curl`, `openssl` and `shasum`. macOS and most Linux systems have them.
- [uv](https://docs.astral.sh/uv/) for the Python tests and the eval tools. It
  installs Python 3.12 for you.
- Node 22 and pnpm, only if you want to run the console tests outside Docker.

These host ports must be free. They are 8080 (gateway), 3001 (console), 3002 (Grafana), 9090
(Prometheus), 9091 (Pushgateway), 8083 (embedding server), 15432 (Postgres),
16379 (Redis) and 16333 (Qdrant). Postgres, Redis and Qdrant use shifted ports on
purpose, so this stack can run next to the companion help desk, which uses the
standard ones.

## Which keys you need

| Key | Needed for | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` | chat, the judge, `make eval`, `make numbers` | chat answers 503 `ALL_PROVIDERS_FAILED`. Everything else still works. |
| `OPENAI_API_KEY` | the fallback for every chat alias | chat runs on Anthropic alone |
| Langfuse OTLP settings | sending traces to Langfuse | spans are built and dropped |

No key is needed for health checks, `/v1/redact`, `/v1/embeddings` (served by the
local llama.cpp container), `/v1/models`, the console, Grafana, the test suites or
the load test. The load test runs the real gateway against a mock provider, so it
needs no model key at all.

## 1. Clone and fetch the embedding model

    git clone https://github.com/masudur-rahman-musa/llmops-console
    cd llmops-console
    make embed-model

`make embed-model` downloads `bge-small-en-v1.5-q8_0.gguf` from Hugging Face into
`.cache/models/` once, and checks its sha256 every time. `make up` runs it for you
too.

## 2. Make the gateway keys

The gateway stores only the sha256 digest of each key, never the key itself. Make
one tenant key and one admin key:

    tenant_key=$(openssl rand -hex 32)
    admin_key=$(openssl rand -hex 32)
    echo "tenant key  $tenant_key"
    echo "admin key   $admin_key"
    echo "OPSCONSOLE_API_KEYS=nordlicht:$(printf %s "$tenant_key" | shasum -a 256 | cut -d' ' -f1)"
    echo "OPSCONSOLE_ADMIN_KEY_SHA256=$(printf %s "$admin_key" | shasum -a 256 | cut -d' ' -f1)"

`make tenant-key TENANT=nordlicht` does the tenant half for you. It prints the
key and the `OPSCONSOLE_API_KEYS=` line.

Keep the two keys in your shell. You use the tenant key to call the gateway, and the
console uses the admin key.

The tenant must be one that `backend/config/routing.yaml` defines. The shipped file
has three demo tenants, `nordlicht`, `cuan` and `evals`. A key for any other name
is refused.

## 3. Write `backend/.env`

    cp backend/.env.example backend/.env

Then edit `backend/.env`:

- Put your Anthropic key in `ANTHROPIC_API_KEY` if you have one. The line ships
  empty, and an empty key leaves that provider off.
- Paste the two lines the commands above printed over the empty
  `OPSCONSOLE_API_KEYS=` and `OPSCONSOLE_ADMIN_KEY_SHA256=` lines.
- Optional. Set `OPSCONSOLE_GRAFANA_URL=http://localhost:3002/d/gateway-overview`
  so the console's Overview tab links to Grafana.

Compose sets the Redis, Postgres, Qdrant and Prometheus addresses for the gateway
itself, so you do not need to change those lines. Git ignores `backend/.env`.

## 4. Write `web/.env.local`

The console has a shared password login. Pick a password and write its digest, a
session secret and the admin key into `web/.env.local`:

    password='pick-a-local-password'
    {
      echo "CONSOLE_PASSWORD_SHA256=$(printf %s "$password" | shasum -a 256 | cut -d' ' -f1)"
      echo "CONSOLE_SESSION_SECRET=$(openssl rand -hex 32)"
      echo "CONSOLE_ADMIN_KEY=$admin_key"
    } > web/.env.local

Compose points the console at the gateway for you. Git ignores this file too.

## 5. Start everything

    make up

This builds the gateway and console images and starts the gateway, the console,
Postgres, Redis, Qdrant, the llama.cpp embedding server, Prometheus, the
Pushgateway and Grafana. The ledger migrations run when the gateway starts, under a
Postgres advisory lock, so there is no separate migration step. To run them by hand
anyway (safe to run twice):

    docker compose exec gateway ops-migrate

`docker compose up -d` on its own starts everything except the console.
`make run` is for working on the gateway code. It starts the backing services in
Docker and runs the gateway on this machine on port 8080, reloading on every save.
`make down` stops the stack and keeps its data. `make rebuild gateway` rebuilds one
service after a code change.

## 6. Check it

    curl -s localhost:8080/healthz
    curl -s localhost:8080/readyz

`/healthz` says the process is up. `/readyz` says Postgres, Redis and Qdrant
answer.

Send a chat request. The client names an alias, never a model:

    curl -s localhost:8080/v1/chat/completions \
      -H "Authorization: Bearer $tenant_key" \
      -H "X-Tenant: nordlicht" \
      -H "Content-Type: application/json" \
      -d '{"model":"agent-default","messages":[{"role":"user","content":"Hi, I am at jane.doe@example.com"}]}' \
      -D - -o /dev/null

The response headers show what the gateway did. `X-Gateway-Provider` and
`X-Gateway-Model` name who answered, `X-Gateway-Cost-USD` is the price,
`X-Gateway-Redacted` says whether a value was replaced, and `X-Gateway-Trace-Id`
is the trace. With no `X-Data-Class` header the request counts as `confidential`,
which the demo tenants redact, so the email never reaches the provider.

Redaction on its own needs no model key:

    curl -s localhost:8080/v1/redact \
      -H "Authorization: Bearer $tenant_key" \
      -H "X-Tenant: nordlicht" \
      -H "Content-Type: application/json" \
      -d '{"text":"Mail the report to sam.lee@example.com"}'

## 7. Open the pages

| Page | Address | Login |
|---|---|---|
| Console | http://localhost:3001 | the password you picked in step 4 |
| Grafana | http://localhost:3002 | none, read-only viewer, bound to 127.0.0.1 |
| Prometheus | http://localhost:9090 | none |

There is no seed script. The console fills from real traffic, so send a few
requests first. `make numbers` fills every tab in one go. It runs the real eval,
sends six chat requests, records a release and then checks every tab. It needs the
help desk checkout (next section) and `ANTHROPIC_API_KEY`, and it spends real
tokens.

## Run the tests

    make fast            # every test not marked slow, no services needed
    docker compose up -d
    make test            # the full suite with Postgres, Redis and Qdrant, plus coverage floors
    make verify          # format, lint, types, tests and the security scans, as CI runs them

`make fast` is the everyday loop. On a laptop it ran 3,683 passing tests in about
32 seconds, with 132 skipped because they need the backing services. `make test`
points those at the compose ports and fails if a service is missing, rather than
skipping. It holds the whole tree to 80% coverage, and the routing, limits and
validation code to 100%. `make verify` also runs gitleaks in Docker, `pip-audit`
and `pnpm audit`.

The console has its own tests:

    pnpm -C web install
    pnpm -C web test

## Run an eval

The eval gate plays the help desk's 45 conversation cases against the real help
desk agent, with this gateway in front of the real cheap model. It needs:

- the companion repo checked out next to this one, at `../helpdesk-agentic-flow`
  (or set `HELPDESK_DIR`),
- `ANTHROPIC_API_KEY` in your shell or in `backend/.env`,
- the help desk's own ports free (5432, 6379, 6333, 8000 and its web app on 3000).

Then:

    make eval

It makes throwaway gateway keys for the run, starts both stacks, resets and seeds
the help desk, plays every case three times, writes the reports to
`backend/evals/reports/local/` and runs the gate against
`backend/evals/conversation-baseline.json`. When it ends, pass or fail,
`infra/evals/restore_keys.sh` puts both stacks back on their normal keys. A run
spends real tokens. `RUN_ID=<id> make eval` continues a stopped run without paying
again for the cases already saved.

The gate itself is a pure check over two files, so you can run it on any report:

    uv run --project backend ops-eval gate \
      --report backend/evals/reports/local/report.json \
      --baseline backend/evals/conversation-baseline.json

## Run the load test

    make load

This starts the gateway with a mock provider that answers after a fixed 400 ms,
then runs k6 at 50 requests a second for ten minutes. It fails if chat p95 reaches
6 s, if more than 0.5% of requests fail, or if k6 drops an iteration. The summary
lands in `infra/load/results/`, which git ignores. No model key is needed.

## When something goes wrong

- **The gateway exits at start.** Read `docker compose logs gateway`. A bad
  `routing.yaml` or provider file stops the start with the file and field named.
- **Every call is 401.** The key and `X-Tenant` must match a pair in
  `OPSCONSOLE_API_KEYS`, and the tenant must exist in `routing.yaml`.
- **Chat is 503.** No chat provider has a key. Set `ANTHROPIC_API_KEY` and run
  `make rebuild gateway`.
- **The console says data is unavailable.** `CONSOLE_ADMIN_KEY` in
  `web/.env.local` must be the key whose digest is in
  `OPSCONSOLE_ADMIN_KEY_SHA256`.
- **A port is taken.** Stop the other stack, or the companion help desk if it
  still holds a port this one needs.
