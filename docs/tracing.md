# Tracing in Langfuse

Every request to the gateway makes one OpenTelemetry trace. When
`OPSCONSOLE_OTLP_ENDPOINT` and `OPSCONSOLE_OTLP_HEADERS` are set, the trace goes to
Langfuse over OTLP/HTTP. When they are not set, spans are built and dropped. This
page says what you see in Langfuse and what stays empty on purpose.

## What fills

- **Trace name.** The HTTP method and the route, such as
  `POST /v1/chat/completions`. A path no route matches is named `GET unmatched`, so
  a junk URL never becomes a name.
- **Steps.** `auth`, `route`, `cache`, `redact`, `semantic` and `provider:<name>`,
  each with its time.
- **What each step decided.** Click a step to see why the request went the way it
  did. Each note is one plain line, as in these examples.
  - The `auth` step reads input `API key and X-Tenant` and output
    `accepted: tenant nordlicht`.
  - The `route` step reads input `alias agent-default, data class pii` and output
    `picked anthropic claude-haiku-5-5; fallback openai gpt-5-mini`. More fallbacks
    are joined with `, then `, and no fallback reads `fallback none`.
  - The `cache` step reads input `same request answered before, alias agent-default`
    and output `hit` or `miss`.
  - The `redact` step reads input `3 messages` (or `2 texts` for embeddings and
    `/v1/redact`) and output `masked: 1 email, 2 phone` or `nothing masked`. Only
    counts go out, never the values or the placeholders.
  - The `semantic` step reads input `nearest stored answer, alias agent-default` and
    output such as `hit (score 0.981, needed 0.97)` or
    `miss (best match 0.912, needed 0.97)`. Scores are cut to 3 decimals, never
    rounded up, so a miss never looks like it reached the bar.
  - A step that refuses the request says `refused:` and the error code, such as
    `refused: UNAUTHORIZED`. The error message never goes out. Before a step
    decides, it reads `not recorded` and `no decision recorded`.
- **The provider call.** It is a Langfuse generation with the model, input and
  output tokens, and the cost in USD, so the Model, Tokens and Cost columns and the
  dashboards fill. The cost is the same number the ledger and the
  `X-Gateway-Cost-USD` header carry.
- **Streamed answers.** The generation stays open until the stream ends, then
  carries the tokens the stream reported.
- **Cache hits.** A cache hit makes no provider call, so it has no generation. Its
  cost is on the ledger only.
- **Sessions.** The `X-Session` value goes out as `langfuse.session.id`, so the
  Sessions page groups the turns of one chat.
- **Tenant.** The tenant is a label on the trace and a tag, such as `tenant:cuan`,
  so you can filter by tenant.
- **Input and output, switch on.** Each tenant in `backend/config/routing.yaml` has
  a `trace_content` switch. With it on, a chat call's input is the conversation as
  the model got it, after redaction, as a list of role and text. The output is the
  model's answer, with placeholders such as `<EMAIL_1>` still in. Tool call details
  are not sent. Each message is cut at 8,000 characters, and a longer one ends in
  ` [cut]`. The three demo tenants have it on.
- **Input and output, switch off.** A tenant with the switch off, or without the
  line, sends a one-line note instead of text, such as
  `not sent: prompts stay inside the gateway (alias agent-default, 3 messages)`.
  Cache hits and embedding calls always send the note.

## Before you turn content on

- **What redaction misses.** Only text the provider already got goes to Langfuse.
  But redaction does not catch names outside the tenant's directory, street
  addresses or secrets written as free text. A data class the tenant does not
  redact goes as written. With the switch on, that text is in Langfuse too.
- **How long it stays.** Langfuse keeps traces as long as your project's plan
  says. Text that was sent cannot be pulled back before then.

## Users

The help desk may send a random user id in `X-User`. It must be `u_` and 32
lowercase hex characters, never a name or an email. The gateway puts it on the
trace as `langfuse.user.id`, so the Users page shows cost, tokens and traces per
id. Only the help desk can tell which person an id belongs to. A request with no
`X-User` has no user on its trace.

## Public traces

Langfuse shows a trace only to members of the project, unless the trace is public.
A public trace opens for anyone who has its exact link, read-only, with no login.
There is no public list of traces.

- **The switch.** `OPSCONSOLE_PUBLIC_TRACE_LANGFUSE=true` makes every trace public,
  for every tenant. Unset or `false`, no trace is public. It is off by default and
  off in the Kubernetes build.
- **Opting a tenant out.** A tenant with `trace_public: false` stays private while
  the switch is on. A tenant without the line follows the switch.
  `trace_public: true` does nothing while the switch is off.
- **Every span.** The label goes on every span of a public request, because
  Langfuse keeps the value from the span that started last.
- **Refused requests.** A request refused at auth is never public.
- **What a visitor sees.** With `trace_content` on, a public trace shows the
  redacted conversation and the answer to anyone with its link, including anything
  redaction missed.
- **Undoing it.** Turning the switch off stops new traces being public. A trace
  that is already public stays public until you change it in Langfuse, on the
  trace's own page.

## What stays empty on purpose

- **Text for a tenant with the switch off.** It never leaves the gateway.
- **Alerts.** None are set up in Langfuse. Alerts live in Prometheus.
- **Prompts, Playground, Datasets, Experiments, Evaluators, Human Annotation.** Not
  used. Prompts and evals live in git and run in CI with `make eval`.

## What is not traced

`/metrics`, `/healthz` and `/readyz` start no trace. Tracing them would bury real
requests.
