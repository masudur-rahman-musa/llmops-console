# The eval target contract

`ops-eval conversations` evaluates any workload that speaks two endpoints. Nothing
else about the workload matters. It may be a FastAPI service, a queue worker behind
a small adapter, or the stub target in `backend/evals/target-stub/`.

    POST {target}/admin/test/reset          once, before the first case
    POST {target}/v1/eval/conversation      once per case, one attempt

Both are plain JSON over HTTP. `{target}` is whatever `--target` names, so a target
may live under a path prefix (`https://helpdesk.example/gateway`).

## The reset

The run resets the workload to its seed before the first case, and once only. The
body is empty and the answer is ignored; anything but a 2xx aborts the run with exit
`2`, the 404 included. A 404 means the endpoint is not enabled in this environment,
and a run over an unreset seed cannot be compared to the baseline, which is the only
thing the number is for.

A run that is continued with the same `--run-id` and already has a saved case is not
reset again, because the workload is already part way through its seed.

Enable the endpoint in test and staging environments and nowhere else. It exists to
change data.

## The conversation

One case is one request. The body is exactly these four fields, taken from the case:

```json
{
  "case_id": "conv-001",
  "tenant": "nordlicht",
  "principal_user": "seed:production.employee.5",
  "turns": [
    "Could you set me up so I can read the internal wiki?",
    "yes"
  ]
}
```

The workload plays the person's messages in one new chat, in order, as
`principal_user` of `tenant`. That means as that person, with that person's
permissions, not as an administrator. A workload that ignores the principal will pass its access cases
and fail every one that matters.

One attempt per case, under a 5 minute timeout. A request that times out, refuses
the connection or answers a non-2xx is a failed case with an error line; the run
continues and the case is never retried. A retried case is a second sample of a
non-deterministic system, which would quietly lift the score.

## The answer

One record per user turn, counted from 1, in the order they were played. Unknown
fields are refused at every level.

```json
{
  "case_id": "conv-001",
  "turns": [
    {
      "turn": 1,
      "reply": "I can give you the Internal Wiki viewer role. Shall I file the request?",
      "tool_calls": []
    },
    {
      "turn": 2,
      "reply": "Done. Your request number is REQ-0007.",
      "tool_calls": [
        {
          "name": "create_access_request",
          "args": {"role": "wiki.viewer"},
          "ok": true,
          "result": {"request_number": "REQ-0007"},
          "error_code": null
        }
      ]
    }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `case_id` | string | The id that was sent. An answer for another id fails the case. |
| `turns` | list | Exactly one record per user turn, numbered 1 to n. |
| `turns[].reply` | string | What the workload said back. |
| `turns[].tool_calls` | list | Every tool call of that turn, in the order made. |
| `tool_calls[].name` | string | The tool. |
| `tool_calls[].args` | object | The arguments it was called with. |
| `tool_calls[].ok` | boolean | Whether the call went through. A refused call has `false`. |
| `tool_calls[].result` | object | What the tool returned. Empty for a refused call. Request numbers and role names the checks look for are read from here. |
| `tool_calls[].error_code` | string or null | The typed error of a refused call. |

The checks read the tool results, so a workload must put in `result` everything a
reply may name. A number that appears in a reply and in no result fails the turn.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | The run finished. Failed cases are the gate's business, not the run's. |
| `2` | The run stopped, because of a bad argument, an unreadable cases file, or a target that could not be reset. |

## What the run writes

- `<run-id>.jsonl` in the output folder, one line per case, written the moment the
  case finishes. A restart with the same `--run-id` skips the cases already there,
  so no paid result is lost.
- `report.json` and `report.md` when the run finishes. See [eval-report.md](eval-report.md).
