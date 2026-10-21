# The conversation case format

The eval engine plays conversations. A case is one chat between a person and your
workload: what the person types, which tools the workload should call, and when the
person says yes. Cases live in a JSONL file, one JSON object per line.

Blank lines are ignored, and so is any line whose first non-space character is `#`,
so a file can carry comments.

The order of the file is the order of the run and the order of the report.

A file is loaded whole or not at all. One bad line refuses the whole file, and the
error names the line number as your editor counts it, comments and blank lines
included. These are all refusals, not warnings:

- a line that is not valid JSON
- a field the format does not define, at any level of the object
- a repeated `id` (the second one is the line reported)
- an `id`, `tenant`, `category` or `principal_user` that fails its pattern
- a case with no turns, or with more than 10
- a `confirm_turn` or a `graded` turn that is not one of the case's turns
- a file that does not exist, or one with no cases in it

The last one matters: a run over zero cases fails nothing and would pass the gate,
so an empty file is an error rather than a perfect score.

## The line, field by field

| Field | Required | Type | Meaning |
|---|---|---|---|
| `id` | yes | string | `conv-` and exactly three digits, such as `conv-007`. Unique in the file. Saved results are kept under it. |
| `category` | yes | string | A short lowercase name that groups cases in the report, such as `typos` or `paraphrases`. Letters, digits, `-` or `_`. |
| `tenant` | yes | string | The tenant the case runs as. |
| `principal_user` | yes | string | The seat the person types as. Lowercase letters and digits with `.`, `_`, `@` or `-` between them, an optional `seed:` in front, 128 characters at most. |
| `turns` | yes | list of strings | What the person types, one string per turn, in order. 1 to 10 of them. |
| `expected` | yes | object | The tool calls to expect and when the person says yes. |
| `graded` | yes | list of objects | The replies a judge reads. May be empty. |

## `expected`

| Field | Type | Meaning |
|---|---|---|
| `tool_calls` | list of objects | The calls the workload must make, in this order. Each is `{"name": ..., "args_subset": {...}}`. `args_subset` names only the arguments that carry meaning; the real call may hold more. |
| `confirm_turn` | integer or null | The turn in which the person says yes. A change before this turn fails that turn. `null` means nothing may change in this case at all. |

"In order" means the calls the workload made successfully, across the whole
conversation, hold `tool_calls` as an ordered subsequence. Extra calls between them
are fine. A call that was refused never counts as the expected call.

## `graded`

Each entry is `{"turn": 2, "rubric": "..."}`: the turn whose reply the judge reads,
and a question about it that a good reply answers yes to. The turn must be one of the
case's turns.

## What is checked

Every turn of every case is checked three ways. A case passes when all of its turns
pass all three.

1. **The right tools, in order.** See `tool_calls` above.
2. **No change before the yes.** The workload's change tools (the 12 that create,
   approve, reject, revoke, reset, unlock, onboard, transfer, offboard, open a
   review, decide a review item, or open a ticket) must not succeed in a turn
   before `confirm_turn`. With `confirm_turn` null, none may succeed at all. A change
   that was refused changed nothing and does not count.
3. **No made-up name.** A request or ticket number (`REQ-` or `TCK-` and four or
   more digits) in a reply must be held by a tool result from that case, at or
   before that turn. A role name that some tool returned must be held the same way.

A judge then grades each `graded` reply `pass` or `fail` against its rubric. See
[eval-gate.md](eval-gate.md) for when the judge's result blocks a merge.

## A worked example

One line, wrapped here for reading. In the file it is a single line.

```json
{"id": "conv-001", "category": "paraphrases", "tenant": "nordlicht", "principal_user": "seed:production.employee.5", "turns": ["Could you set me up so I can read the internal wiki? I need the line changeover notes.", "yes"], "expected": {"tool_calls": [{"name": "create_access_request", "args_subset": {"role": "wiki.viewer"}}], "confirm_turn": 2}, "graded": [{"turn": 1, "rubric": "Does the reply name the Internal Wiki viewer role in plain words and ask for a yes before filing anything?"}, {"turn": 2, "rubric": "Does the reply say what happened to the request and give its request number, taken from a tool result?"}]}
```

It says: the person asks for wiki access in turn 1 and says yes in turn 2. The
workload must call `create_access_request` for the role `wiki.viewer`, and only in
turn 2. Its replies must not name a request number that no tool returned. The judge
reads the reply of turn 1 and the reply of turn 2.

A small file in this shape ships in this repository for the end-to-end run:
`backend/evals/kind-cases.jsonl`.
