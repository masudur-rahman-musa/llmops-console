# The eval target contract

`ops-eval run` evaluates any workload that speaks two endpoints. Nothing else
about the workload matters: it may be a FastAPI service, a queue worker behind a
shim, or the gateway itself with a demo app in front of it.

    POST {target}/admin/test/reset   once, before the first case
    POST {target}/v1/eval/turn       once per case, one attempt

Both are plain JSON over HTTP. `{target}` is whatever `--target` names, so a
target may live under a path prefix (`https://helpdesk.example/gateway`).

## The reset

The run resets the workload to its seed before the first case, and once only.
The body is empty and the answer is ignored; anything but a 2xx aborts the run
with exit `2`, the 404 included. A 404 means the endpoint is not enabled in this
environment, and a run over an unreset seed is not comparable to the baseline,
which is the only thing the number is for.

Enable the endpoint in test and staging environments and nowhere else. It exists
to destroy data.

## The turn

One case is one request. The body carries the case as the golden set wrote it:

```json
{
  "case_id": "access-auto-001",
  "tenant": "acme",
  "principal_user": "dana.okafor",
  "messages": [
    {"role": "user", "content": "Grant me read access to finance-reports."}
  ]
}
```

The workload runs the turn as `principal_user` of `tenant` - as that person,
with that person's permissions, not as an administrator. A workload that ignores
the principal will pass its access cases and fail every one that matters.

One attempt per case, under a 60 second timeout. A turn that times out, refuses
the connection or answers a non-2xx is a FAILED case; the run continues and the
case is never retried. A retried case is a second sample of a non-deterministic
system, which would quietly lift the score.

## The prompt versions header

A workload may name the prompt each of its steps ran on in an
`X-Prompt-Versions` response header, because the envelope refuses extra fields:

```
X-Prompt-Versions: intent=<sha256>,plan=<sha256>,compose=<sha256>,system=<sha256>
```

Each entry is a step name (lowercase letters, digits and `_`, starting with a
letter, at most 32 characters) and the 64 lowercase hex characters of that
prompt's `sha256`. At most 16 steps, each named once. The report carries the map
(`step_prompt_versions` in `docs/eval-report.md`). No header is fine: the map is
empty. A header that breaks any of these rules fails that turn as
`provider_error`, because a half-read map would name prompts nobody ran.

## The envelope

The answer is an envelope, not a chat completion. The harness reads the intent,
the decision, the risk class, the tool calls and the citations as fields, because
reading them back out of prose would put the harness's own extraction accuracy
inside every number it reports.

Unknown fields are refused at every level, and the refusal names the case. A
field your workload renames is a failed run, not an assertion that quietly
reports `skipped` while the score goes up.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `intent` | string or null | `null` | The intent the workload classified the turn as. |
| `tool_calls` | list of objects | `[]` | Every tool call it made, in order, each `{"name": ..., "args": {...}}`. The order is asserted exactly; an extra call is a failure. |
| `decision` | string or null | `null` | What it decided (`auto_approved`, `escalated`, ...). Your vocabulary; the golden set names the same words. |
| `risk_class` | string or null | `null` | The risk class it assigned. |
| `refused` | boolean | `false` | Whether the answer refused the request. |
| `clarified` | boolean | `false` | Whether it asked a clarifying question instead of acting. |
| `citations` | list of strings | `[]` | The documents the answer cited, by id. |
| `reply` | string | `""` | The text shown to the user. Forbidden phrases and cross-tenant markers are matched against this, case-insensitively. |
| `latency_ms` | integer or null | `null` | How long the turn took, as the workload measured it. |
| `cost_usd` | string or null | `null` | What the turn cost. A STRING, so it parses as an exact decimal; `"0.0142"`, never `0.0142`. |
| `provider` | string or null | `null` | Which provider served it. |
| `region` | string or null | `null` | Which region served it. Read below. |

Every field is optional, and every default reads as "nothing observed". An
expectation the golden set states about a field the envelope omits fails; an
expectation it does not state reports `skipped`, never `passed`.

## `region`, and why an omission fails

Every turn's `region` is checked against the tenant's allow list for the
`internal` data class, read from the gateway's own `routing.yaml`. A region
outside that list fails the WHOLE run with exit `3`, whatever the assertions
said.

So does an envelope with no `region` at all. Residency is deny-by-default: a
workload that never proved where the data went has not proved it stayed, and a
missing field is the shape a half-implemented target actually produces.

## A complete envelope

This is `tests/evals/fixtures/envelope.json`, byte for byte. A test loads that
file and reads every field back, so this example and the model cannot drift.

```json
{
  "intent": "request_access",
  "tool_calls": [
    {
      "name": "grant_access",
      "args": {
        "resource": "finance-reports",
        "level": "read",
        "ticket_id": "HD-4821"
      }
    }
  ],
  "decision": "auto_approved",
  "risk_class": "low",
  "refused": false,
  "clarified": false,
  "citations": [
    "access-policy-v3"
  ],
  "reply": "Read access to finance-reports is active, per the access policy.",
  "latency_ms": 1840,
  "cost_usd": "0.0142",
  "provider": "anthropic_eu",
  "region": "eu"
}
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Every case passed and every turn stayed in region. |
| `1` | At least one case failed its assertions, or its turn never answered. |
| `2` | A usage error, an unloadable set, or a target that could not be reached or reset. |
| `3` | A residency violation or a cross-tenant leak. |

`1` and `3` are deliberately different. An accuracy regression is a number that
moved; a residency violation or a leak across a tenant boundary is a person's
problem, and a gate that reported both as `1` would let one be merged as the
other.

## What the run writes

Nothing yet, beyond one summary line on stdout naming the counts. `--out` is
validated and created, and the report files that go in it arrive with the next
plan.
