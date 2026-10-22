# The eval report

Every `ops-eval conversations` run writes, into the folder `--out` names:

    <run-id>.jsonl   one line per case, written the moment the case finishes
    report.json      the whole run, for the gate and for storage
    report.md        the same run for a person to read

`report.json` and `report.md` are written when the run finishes, and their names are
fixed. A rerun into the same folder overwrites them, so a caller who wants history
keeps folders rather than file names, and the gate reads a known path instead of
looking for the newest file. A failed run is exactly the one somebody opens the report
for, so a run that finishes always writes both.

## The saved cases

`<run-id>.jsonl` is written and flushed one case at a time. If the run stops (a network
drop, a closed laptop), nothing already paid for is lost. Start it again with the same
`--run-id` and the cases already in the file are skipped. A case that failed because
the target did not answer is a saved result too: its line holds the error. A run that
finds saved cases does not reset the target again.

`--repeats N` (1 to 10, default 1) plays the whole case list N times. Each pass is a
round: it starts with its own reset and plays the cases in file order, because the
cases act on what earlier ones left behind. Each line carries its `round` number. A
continued run picks up inside the round it stopped in, with no reset. A line saved
before rounds existed has no `round` and is read as round 1, so an old file still
continues.

Each line holds the case id and category, whether it passed, the error if the target
failed, the three checks, each graded reply with the judge's verdict and reason, and
the target's own answer (kept so a continued run still knows which role names the run
has seen).

## report.json, field by field

### The run

| field | type | what it is |
|---|---|---|
| `id` | string | a `uuid4` hex, 32 characters, fresh per run |
| `started_at` | timestamp | UTC, just before the first case |
| `finished_at` | timestamp | UTC, after the last one |
| `set_name` | string | the cases file's name |
| `set_sha` | string | which version of the cases this was |
| `set_sha_source` | `commit` or `content` | whether `set_sha` is a git commit or the hash of the file's bytes |
| `run_kind` | `ci`, `nightly` or `manual` | why the run happened |
| `git_sha` | string or `null` | this repository's `HEAD`, or `null` when git cannot answer |
| `tenant` | string | `all`: each case names its own tenant |
| `target` | string | the base URL evaluated |
| `passed` | boolean | true when no stable case failed |
| `metrics` | object | the scores, below |
| `judge_prompt_version` | string or `null` | the sha256 of the judge prompt used, `null` when nothing was judged |
| `calibration` | object or `null` | what the run was told about its judge, below |
| `cases` | list | one row per case, below |

### metrics

| field | what it is |
|---|---|
| `check_score` | percent of stable cases that passed all three checks, two decimals. A case whose target never answered counts as failed. `null` when no case is stable. |
| `judge_score` | percent of graded replies the judge passed, two decimals. `null` when nothing was judged. |
| `cases`, `passed`, `failed`, `unstable` | the counts behind `check_score`; `cases` is the other three added up |

A case played several times passes when every run passed. A case that both passed
and failed is unstable: it is counted in `unstable` and left out of `check_score`, so
chance alone cannot move the score. With one run per case nothing is unstable.

A score is `null` when nothing was measured, never `0`. A `0` would read as a judge
that failed every reply.

### The judge calibration

When the run is given a `calibration.json`, the report carries:

| field | what it is |
|---|---|
| `judge_prompt_version` | the prompt the calibration was made with |
| `labels_name`, `labels_sha` | the label file and its hash |
| `pairs` | how many labelled replies were judged |
| `kappa` | Cohen's kappa between the judge and the labels, or `null` when it cannot be worked out |
| `trusted` | true when `kappa` is 0.7 or better |

### cases

| field | what it is |
|---|---|
| `case_id`, `category` | from the case |
| `passed` | whether every run of the case passed all three checks |
| `runs` | how many times the case was played |
| `pass_rate` | percent of those runs that passed, two decimals |
| `range_low`, `range_high` | the 95% Wilson range of `pass_rate`, in percent. Wilson stays usable at 3 runs, where the plain normal range does not |
| `unstable` | true when the case both passed and failed |
| `error` | why the target did not answer, else `null`; taken from the first failed run |
| `check` | `tools_in_order`, and per turn `no_early_change` and `no_made_up_name`; `null` when the target failed |
| `judged` | each graded reply, from every run: `turn`, `rubric`, `reply`, `verdict` (`pass`, `fail` or `null` when the judge was unusable) and `reason` |
| `tags` | always empty |
| `latency_ms`, `cost_usd`, `provider`, `region` | always `null`: a conversation answer carries none of them |

## The judge

For each graded reply the judge gets the rubric and the reply, and nothing else. It
answers with a one-line reason first, then `pass` or `fail`. It is asked three times
at temperature zero and the majority wins. An answer that cannot be read is asked
again once; if it fails again that sample is dropped, and a reply with no usable
verdict is `null` and is not counted in `judge_score`.

The rubric and the reply reach the model as data, under labelled delimiters. Text in
either that tries to give the judge instructions is graded, not obeyed.

## The reference labels and the calibration

The judge is checked against reference labels, 40 graded replies in
`backend/evals/labels/conversation-labels.jsonl`. Each line:

    {"case_id", "turn", "rubric", "reply", "verdict", "votes", "reason", "source"}

`source` is `model-reference` for a label made by a model, or `human` once a person
has made it. A model label is a placeholder: it is made once by the strong model given
the whole conversation, every tool call and result, and the rubric. It grades each
reply three times, and a reply is kept only when all three agree. A person's labels
replace them later.

`ops-eval label` makes the file, and refuses to overwrite one that exists.
`ops-eval calibrate --labels PATH --judge-url URL --out DIR` plays the judge over the
labelled replies and writes `calibration.json` with the agreement. It refuses with
fewer than 40 labels. Pass that file to `ops-eval gate --calibration` to let the judge
block a merge. Below kappa 0.7 the judge is reported and decides nothing.

## report.md

The same run for a person: which run this was, the metrics, the cases by category, the
replies the judge failed with its reasons, the unstable cases with their pass rate and
range, and the failures last. An unstable case is never listed under the failures. A failed case names
what was wrong: the right tools were not called in order, a change was made before the
yes, a number or role was named that no tool returned, or the error when the target did
not answer. A clean run has no failures section at all.

## Storage

The same run is also stored in PostgreSQL, in two tables the `003_eval_runs.sql`
migration creates: one `eval_runs` row and one `eval_cases` row per case. The rows are
written after both files, in one transaction, so a storage outage costs a run its rows
and never its report.

With no database URL configured a run stores nothing, says so in one line on stderr and
keeps its own exit code.

### eval_runs

| column | what it is |
|---|---|
| `id` | the report's run id |
| `started_at`, `finished_at` | UTC times |
| `set_name`, `set_sha`, `set_sha_source` | which cases, which version |
| `run_kind` | `ci`, `nightly` or `manual` |
| `target`, `tenant`, `git_sha` | where, as whom, which commit |
| `metrics` | the report's `metrics` object; the console reads `check_score` and `judge_score` from here |
| `passed` | true when every case passed |
| `report_path` | the `report.json` this run wrote |
| `judge_kappa` | the calibration's kappa, or `NULL` on a run given no calibration |

### eval_cases

| column | what it is |
|---|---|
| `run_id`, `case_id`, `category` | which run, which case |
| `passed` | whether the case passed |
| `tags` | always empty |
| `latency_ms`, `cost_usd`, `provider`, `region` | `NULL` |
| `details` | the checks, the judged replies and the error |

## Metrics for the dashboard

A run can push its two scores to a Pushgateway with `--pushgateway-url`:
`gateway_eval_check_score` and `gateway_eval_judge_score`. A score that is `null` is
not pushed. The nightly alert reads `gateway_eval_check_score`.

## The tradeoff table

`ops-eval tradeoff --manifest <path> --out <dir>` is the other thing these reports
feed. It reads a manifest naming several published runs of the same cases file and
writes `tradeoff.md` and `tradeoff.json` into `<dir>`: one row per option with its
check score, judge score and cost per 1,000 turns, a block naming the report and the commit
behind every row, and one computed conclusion. `make tradeoff` runs it.
