# The eval report

Every `ops-eval conversations` run writes four files into the folder `--out` names:

    report.json        the whole run, for the gate and for storage
    report.md          the same run for a person to read
    needs-a-look.md    the plays no rule caught, each with its conversation
    comment.md         the scores, for a pull request comment

Their names are fixed. A rerun into the same folder overwrites them, so a caller who
wants history keeps folders rather than file names, and the gate reads a known path.

`--plays N` (1 to 10, default 3, the nightly run uses 1) plays every case N times. Each
play starts from freshly reset data. See [eval-target.md](eval-target.md).

## report.json, field by field

### The run

| field | type | what it is |
|---|---|---|
| `id` | string | a `uuid4` hex, 32 characters, fresh per run |
| `started_at` | timestamp | UTC, just before the first play |
| `finished_at` | timestamp | UTC, after the last one |
| `set_name` | string | the cases file's name |
| `set_sha` | string | which version of the cases this was |
| `set_sha_source` | `commit` or `content` | whether `set_sha` is a git commit or the hash of the file's bytes |
| `run_kind` | `ci`, `nightly` or `manual` | why the run happened |
| `git_sha` | string or `null` | this repository's `HEAD`, or `null` when git cannot answer |
| `tenant` | string | `all`, because each case names its own tenant |
| `target` | string | the base URL evaluated |
| `passed` | boolean | true when no play broke a rule |
| `metrics` | object | the scores, below |
| `judge_prompt_version` | string or `null` | the sha256 of the judge prompt used, `null` when nothing was judged |
| `cases` | list | one row per case, below |

### metrics

Scores are in percent with two decimals. A score is `null` when nothing was measured,
never `0`.

| field | what it is |
|---|---|
| `safety` | the share of scored plays that broke no rule. A play set up wrong is not scored. |
| `outcome_every_play` | the share of cases whose scored plays all passed |
| `outcome_at_least_once` | the share of cases with at least one pass |
| `reply_quality` | the share of yes answers among the yes and no answers of the check types the judge calibration trusts. `null` without a calibration file. |
| `mean_turns`, `mean_tool_calls`, `mean_seconds`, `mean_cost_usd` | effort per scored play. Cost stays `null` until the target sends it. |
| `needs_a_look`, `setup_wrong` | how many plays ended each way, counted apart |
| `cases`, `plays` | the counts behind the scores |

### cases

| field | what it is |
|---|---|
| `case_id`, `category` | the case id and its `owner` |
| `passed` | true when every scored play passed |
| `tags` | the rule ids the plays broke, once each, in the order first broken |
| `plays` | each play: `play`, `result` (`setup_wrong`, `fail`, `pass` or `needs_a_look`), `broken`, `flagged`, `turns`, `tool_calls`, `seconds`, `cost_usd`, `checks` and the `conversation` |

A check holds the check type, the judge's answer (`yes`, `no`, `does_not_apply`, or
`null` when the judge was unusable), a quote from the reply and a one-sentence reason.

## The judge

For each reply check the judge gets the check, the whole conversation, the tool results
and the world facts. Each check is its own request, answered once. An answer that cannot
be read is asked again once; if it fails again the check is left unanswered and counts
for nothing. The conversation reaches the model as data between fixed tags. Text in it
that tries to give the judge instructions is read, not obeyed.

## The simulated person

The person runs on the gateway alias `simulated-person` with a fixed prompt file. It
answers from what the case says the person knows, says yes only when the proposal matches
`confirms_if`, and ends with `[done]`, up to the case's `max_turns`.

## Storage

The same run is also stored in PostgreSQL, in two tables the `003_eval_runs.sql`
migration creates: one `eval_runs` row and one `eval_cases` row per case. The rows are
written after the files, in one transaction, so a storage outage costs a run its rows
and never its report. With no database URL configured a run stores nothing, says so in
one line on stderr and keeps its own exit code.

| table | what it holds |
|---|---|
| `eval_runs` | the run's identity, its `metrics` (the console reads them from here), `passed` and the `report_path` |
| `eval_cases` | one row per case: `passed` (every scored play passed), `tags` (the broken rule ids) and `details` (the plays) |

## Metrics for the dashboard

A run can push its scores to a Pushgateway with `--pushgateway-url`:
`gateway_eval_safety`, `gateway_eval_outcome_every_play` and
`gateway_eval_outcome_at_least_once`. A score that is `null` is not pushed. The nightly
alert reads `gateway_eval_outcome_every_play`.

## The tradeoff table

`ops-eval tradeoff --manifest <path> --out <dir>` is the other thing these reports
feed. It reads a manifest naming several published runs of the same cases file and
writes `tradeoff.md` and `tradeoff.json` into `<dir>`: one row per option with its
outcome every play, safety and cost per 1,000 turns, a block naming the report and the
commit behind every row, and one computed conclusion. `make tradeoff` runs it.
