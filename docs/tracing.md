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
- **Input and Output.** Every step shows a short note in place of the text. Input says `not sent: prompts stay inside the gateway`, and Output says `not sent`. The provider step adds safe facts: Input adds the alias and how many messages went out, Output adds the finish reason and the output tokens. For example, Input `{"text": "not sent: prompts stay inside the gateway", "alias": "agent-default", "messages": 3}` and Output `{"text": "not sent", "finish_reason": "stop", "tokens_out": 361}`. The trace itself shows the same notes as the provider call that answered. A fact that is not known is left out.

## What stays empty on purpose

- **The prompt and the answer.** They never leave the gateway. Langfuse sits outside the redaction this system promises, so it only gets labels and the notes above, never text.
- **Alerts.** None are set up in Langfuse. Alerts live in Prometheus.
- **Users.** There is no user on any trace.
- **Prompts, Playground, Datasets, Experiments, Evaluators, Human Annotation.** Not used. Prompts and evals live in git and run in CI (`make eval`).

## What is not traced

`/metrics`, `/healthz` and `/readyz` start no trace. They are answered as before. Tracing them would bury real requests and use up the free plan.
