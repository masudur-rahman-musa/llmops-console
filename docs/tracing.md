# Tracing in Langfuse

Every request to the gateway sends one trace to Langfuse. This page says what you will see there and what stays empty on purpose.

## What fills

- **Trace name.** The HTTP method and the route, like `POST /v1/chat/completions`. A path no route matches is named `GET unmatched`, so a junk URL never becomes a name.
- **Steps.** `auth`, `route`, `redact`, `cache` and `provider:<name>`, each with its time.
- **The provider call.** It is a Langfuse generation. It carries the model, the input and output tokens and the cost in USD, so the Model, Tokens and Cost columns and the Dashboards page fill. The cost is the same number the cost ledger and the `X-Gateway-Cost-USD` header carry.
- **Streamed answers.** The generation stays open until the stream ends, then carries the tokens the stream reported. A stream that reports none ends with the model only.
- **Cache hits.** A cache hit makes no provider call, so it has no generation. Its cost is on the ledger only.
- **Sessions.** The `X-Session` value goes out as `langfuse.session.id`, so the Sessions page groups the turns of one chat.
- **Tenant.** The tenant is a label on the trace and a tag, like `tenant:cuan`, so you can filter traces by tenant. A trace carries a tenant and a session, and never a user.
- **Input and Output, switch on.** Each tenant in `backend/config/routing.yaml` has a switch, `trace_content`. With it on, a chat model call's Input is the conversation exactly as the model got it, after redaction: the system prompt, earlier turns and the new message, as a list of role and text. Langfuse shows them as chat bubbles. Output is the model's answer, with placeholders such as `[EMAIL_1]` still in. Tool call details are not sent. Each message is cut at 8000 characters; a longer one ends in ` [cut]`. The trace itself shows the same Input and Output. The finish reason is in the metadata. The demo tenants `nordlicht`, `cuan` and `evals` are on.
- **Input and Output, switch off.** A tenant with the switch off, or without the line, sends a one-line note instead of text. Input says `not sent: prompts stay inside the gateway (alias agent-default, 3 messages)` and Output says `not sent (finish reason stop, 361 tokens out)`. A fact that is not known is left out. Other steps, cache hits and embedding calls always show the bare note, whatever the switch says.

## What to know before turning it on

- **What redaction misses.** Only text the model provider already got goes to Langfuse. But redaction does not catch names outside the tenant's directory, street addresses or secrets written as free text. A data class the tenant does not redact goes as written. With the switch on, that text is in Langfuse too.
- **How long it stays.** Traces are kept 30 days, the most the Langfuse Hobby plan keeps. Text sent cannot be pulled back before then.

## What stays empty on purpose

- **Text for a tenant with the switch off.** It never leaves the gateway.
- **Alerts.** None are set up in Langfuse. Alerts live in Prometheus.
- **Users.** There is no user on any trace.
- **Prompts, Playground, Datasets, Experiments, Evaluators, Human Annotation.** Not used. Prompts and evals live in git and run in CI (`make eval`).

## What is not traced

`/metrics`, `/healthz` and `/readyz` start no trace. They are answered as before. Tracing them would bury real requests and use up the free plan.
