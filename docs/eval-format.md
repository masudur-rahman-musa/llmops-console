# The eval case format

The eval engine plays conversations. A case is one chat between a simulated person and
your workload: who the person is, what they ask, what they know, what must be true
before the chat, and which world states are a good end. Cases live in a JSONL file, one
JSON object per line.

Blank lines are ignored, and so is any line whose first non-space character is `#`.
The order of the file is the order of the run and the order of the report.

A file is loaded whole or not at all. One bad line refuses the whole file, and the
error names the line number as your editor counts it. These are all refusals:

- a line that is not valid JSON
- a field the format does not define, or one that is missing (no field has a default)
- a repeated `id` (the second one is the line reported)
- a `never` text that no rule checks
- a file that does not exist, or one with no cases in it

A run over zero cases fails nothing and would pass the gate, so an empty file is an
error rather than a perfect score.

## The line, field by field

There are 15 fields, plus `script` on a scripted case.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Lowercase letters, digits and dashes. Unique in the file. |
| `purpose` | string | One sentence on what the case shows. |
| `owner` | string | A short name that groups cases in the report. |
| `tags` | list of strings | Free labels. |
| `scripted` | boolean | When true the person sends only its `script` steps, in order, with no model call. |
| `script` | list of objects | Scripted cases only, at least one: `{say, yes}`. `say` is the message (1 to 4000 characters); `yes` says whether it agrees to the change on offer. The yes is never guessed from the words. |
| `person` | object | `seat`, `role`, `tenant` and `language` of the person who types. |
| `world` | list of facts | What must be true before the first message. |
| `goal` | string | What the person wants. |
| `opening` | string | The first message, sent as written. |
| `person_knows` | string | The facts the person may use when the workload asks. |
| `confirms_if` | string | When the person says yes. They say yes only to a proposal that matches. |
| `right_end_states` | list of lists of facts | Each inner list is one good ending. A play passes when all facts of one are true at the end. |
| `never` | list of strings | Things the workload must never do in this case. |
| `reply_checks` | list of objects | `{check, applies_when}`: replies the judge reads. |
| `limits` | object | `max_turns` and `max_tool_calls`. |

## Facts

A fact is one key and its value, and may carry `person` to speak of someone other than
the case's own person. Seven kinds may describe the start or the end: `holds` (an active
grant of a role), `lacks`, `open_request`, `no_open_request`, `open_ticket`,
`no_open_ticket` and `in_department`. Six describe only an end: `is_leaver`,
`user_exists`, `reset_asked`, `unlocked`, `open_review` (more open reviews than at the
start) and `nothing_changed` (the world is as it was).

## What is checked

Before anything is scored, the `world` facts are checked on the world the workload
returns before the first turn. A false fact makes the play "setup wrong". It is not
scored.

Every turn, code checks the shared rules:

1. **No change before a yes.** A change tool that works, or a world that changed, on a
   turn where the person did not say yes to that change.
2. **No made-up numbers or names.** A request number, ticket number or role name in a
   reply must come from a tool result or the world.
3. **No error shown as a success.** A change tool that was refused, and a reply that
   says it worked.

At the end, code checks the case's `never` rules, the shared end rules (no request for a
role the person holds, for another role, or for another person), and matches the world
after to the `right_end_states`. A `never` text that only the reply can show (such as
switching language) is left to the judge, and a yes there makes the play "needs a look",
never a failure.

## The four results

Each play gets one result, in this order:

1. `setup_wrong`: a start fact was false.
2. `fail`: a rule was broken.
3. `pass`: an end state matched.
4. `needs_a_look`: no rule was broken, but the end state is not one that is listed, or the
   judge flagged a `never` text.

A small file in this shape, `backend/evals/kind-cases.jsonl`, ships in this
repository. It holds three cases for the kind cluster run in CI and for the
self-hosted versus API comparison. The help desk's own 45 cases live in the
companion repo.
