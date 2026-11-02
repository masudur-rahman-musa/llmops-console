# The eval target contract

`ops-eval conversations` evaluates any workload that speaks two endpoints. Nothing
else about the workload matters. It may be a FastAPI service, a queue worker behind
a small adapter, or the stub target in `backend/evals/target-stub/`.

    POST {target}/admin/test/reset          before every play
    POST {target}/v1/eval/conversation      once per turn

Both are plain JSON over HTTP. `{target}` is whatever `--target` names, so a target
may live under a path prefix (`https://helpdesk.example/gateway`).

## The reset

Every play starts from freshly reset data, so no play sees what an earlier one changed.
The body is empty and the answer is ignored; anything but a 2xx aborts the run with exit
`2`, the 404 included. A 404 means the endpoint is not enabled in this environment, and
a run over unreset data cannot be compared to the baseline.

`--reset-command CMD` replaces the endpoint with a command that is run before every play.
`make eval` passes `infra/evals/reset_helpdesk.sh`. A command that exits non-zero aborts
the run the same way.

Enable the endpoint in test and staging environments and nowhere else. It exists to
change data.

## The turn

One turn is one request. The body is exactly these five fields:

```json
{
  "case_id": "case-001",
  "tenant": "nordlicht",
  "principal_user": "seed:production.employee.5",
  "chat_id": null,
  "message": "Could you set me up so I can read the internal wiki?"
}
```

`chat_id` is `null` on the first turn of a play and the id the workload answered with on
every later turn. The workload answers as `principal_user` of `tenant`: as that person,
with that person's permissions, not as an administrator.

One attempt per turn, under a 5 minute timeout. A request that times out, refuses the
connection or answers a non-2xx ends that play, and the turn is never retried. A retried
turn is a second sample of a non-deterministic system, which would quietly lift the
score.

## The answer

Unknown fields are refused at every level.

```json
{
  "chat_id": "chat-9",
  "reply": "I can give you the Internal Wiki viewer role. Shall I file the request?",
  "tool_calls": [],
  "world_before": {"requests": [], "grants": [], "tickets": [], "people": [], "reviews": []},
  "world_after": {"requests": [], "grants": [], "tickets": [], "people": [], "reviews": []},
  "cost_usd": "0.0123"
}
```

| Field | Type | Meaning |
|---|---|---|
| `chat_id` | string | The chat this turn belongs to. |
| `reply` | string | What the workload said back. |
| `tool_calls` | list | Every tool call of the turn, in the order made. Each has `name`, `args`, `ok`, `result` and `error_code`. A refused call has `ok` false and an empty `result`. |
| `world_before`, `world_after` | object | The tenant's rows before and after the turn. |
| `cost_usd` | decimal text, optional | What the turn cost. Effort cost stays empty until it is sent. |

The world holds `requests` (number, requester, target, role, status), `grants` (person,
role, status, expires_at), `tickets` (number, person, kind, status), `people` (person,
department, status, reset_asked, unlocked) and `reviews` (status, items_decided). A
seeded person is named `<department lowercased>.<index>`, a role `<system>.<role>`.

The rules read the tool results and the worlds, so a workload must put in them
everything a reply may name. A number in a reply that appears in neither fails the turn.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | The run finished. Failed plays are the gate's business, not the run's. |
| `2` | The run stopped: a bad argument, an unreadable cases file, or a target that could not be reset. |

## What the run writes

`report.json`, `report.md`, `needs-a-look.md` and `comment.md`. See
[eval-report.md](eval-report.md).
