# The eval gate

A run says what happened. The gate says whether it may merge.

    ops-eval gate --report reports/report.json \
                  --baseline backend/evals/conversation-baseline.json \
                  [--calibration labels/calibration.json] \
                  [--comment comment.md]

It reads the files you name and touches nothing else. There is no database, no
gateway and no clock in it. The same report and the same baseline give the same
answer on a laptop and in CI, so a red gate can be argued about rather than
guessed at.

## The two rules

| Rule | Breaches when |
|---|---|
| `check_score` | below the baseline's `check_score`, or missing |
| `judge_score` | below the baseline's `judge_score`, but only while the judge is trusted |

`check_score` is the percent of stable cases that passed all three checks. There is
no drop allowance. One point under the baseline fails, and an equal score passes.

A real model does not answer the same way twice, so the gate plays each case three
times. A case that both passes and fails is unstable. It is left out of
`check_score` and named in a note on the verdict and the comment, so it stays in
view without deciding anything. The fix for a flaky case is to fix the
conversation, never a wider allowance.

A missing `check_score` breaches the rule. A run over no cases, or one where every
case is unstable, has no score. Reading that as "nothing against us" would merge
anything, so the reason line says `no evidence` instead of a number.

`judge_score` is the percent of graded replies the judge passed. It counts toward
the gate only when a calibration file is given and its Cohen's kappa is 0.7 or
better. Below that, or with no calibration at all, the score is reported and the
comment says the judge decided nothing. No calibration file is committed today,
so today only `check_score` can block a merge. The 40 reference labels are made
by a model of the same family as the judge, so that kappa shows the two agree,
not that either is right, until a person makes the labels.

## The numbers are percentages

Every score is a percentage from `0.0` to `100.0`, at two decimal places. The
baseline uses the same unit as the report, so no conversion sits in the
comparison. A baseline is checked whole before any comparison runs. An unknown key
or a number outside `0.0` to `100.0` refuses the file.

The baseline holds `check_score`, and optionally `judge_score`, and nothing else.
The committed one is a check score of 48.48 and a judge score of 27.42.

## The exit code is the interface

| Code | Meaning |
|---|---|
| `0` | every rule cleared |
| `1` | at least one rule breached |
| `2` | an input could not be read, parsed or validated |

`2` is kept apart from `1` so a workflow can tell "the gate never ran" from "the
run got worse" without reading text.

Stdout carries exactly one line, the verdict. The breached rules go to stderr, one
line each. The markdown that `--comment` writes opens with the same verdict line.

## Changing the baseline

`backend/evals/conversation-baseline.json` is the record of what the system is
known to do. It is never edited to make a run pass. It changes only in a commit
whose subject begins with `baseline:` and which adds a decision record under
docs/adr in that same commit. The folder appears with the first such change.

CI enforces the rule, not a reviewer:

    ops-eval baseline-guard --base <ref> --head <ref>

It reads `base..head` once and gives a verdict over the commits in it. A commit
that does not touch a baseline file is not examined. One that does is clean only
when its subject starts with `baseline:` and it adds a decision record. The guard
also watches the old single-reply baseline file, so that file cannot come back
quietly.

Same commit, not same pull request. A merge and a revert can split a baseline
change from its reason across a pull request. They cannot split them inside one
commit.

| Code | Meaning |
|---|---|
| `0` | the range is clean, and stdout says how many commits were read |
| `1` | at least one commit breaks the rule, and each one's sha is a line on stderr |
| `2` | `--base` or `--head` was refused, or git could not read the range |

A shallow checkout gets `2`. `git log base..head` resolves nothing at depth 1, and
a guard that read no commits must never answer "clean". So the workflow checks out
with `fetch-depth: 0`, and a range it cannot read is a usage error.

## The workflow

`.github/workflows/eval-gate.yml` runs on every pull request into `main`, in two
jobs that fail for different reasons.

`baseline-guard` is the rule above. It always runs.

`eval` also always runs. It checks out the help desk repo beside this one with the
`HELPDESK_REPO_TOKEN` secret, a read-only token, writes the help desk's
`.env.demo` from its `.env.example`, and runs `make eval` with `ANTHROPIC_API_KEY`
from a GitHub secret. Without those secrets the job fails. It never skips.

`make eval` does the whole run, the same way on a laptop. `HELPDESK_DIR` names the
help desk checkout, `../helpdesk-agentic-flow` by default, and the key can sit in
`backend/.env`.

1. It makes throwaway gateway keys, one per help desk tenant and one for the
   judge. The digests go to the gateway and the keys to the help desk. Nothing is
   stored.
2. It starts the gateway with `infra/evals/compose.yml`, clears its response cache
   and applies the migrations. The gateway keeps its shipped routing, so the help
   desk's chat runs on the real cheap model behind `agent-default`.
3. It resets and seeds the help desk's database and loads its policy text with
   `infra/evals/prepare_helpdesk.sh`, then starts the help desk's agent API and
   tool server with the eval routes on, using `infra/evals/helpdesk.compose.yml`.
4. It plays the help desk's 45 conversation cases three times each
   (`EVAL_REPEATS ?= 3`) with `ops-eval conversations --repeats`, writes the
   reports to `backend/evals/reports/local/`, and runs `ops-eval gate` against the
   conversation baseline, with the calibration file when one exists.
5. On exit, pass or fail, `infra/evals/restore_keys.sh` puts both stacks back on
   their normal keys.

A mock cannot play a conversation. A case has no fixed answer for each model call,
and recordings drift with every prompt change. So the gate runs on a real model,
and every run spends real tokens.

`RUN_ID=<id> make eval` continues a stopped run. The cases already saved under that
id are skipped, and the reports folder is kept.

The report folder is uploaded as a build artifact whatever the verdict, because a
failed gate is when somebody opens it. The comment is posted once per pull request
and edited in place on a re-run. On a pull request from a fork the token is
read-only, so the comment is skipped and the artifact is still published.

## What the comment carries

- the verdict line
- the run's `check_score` and `judge_score` against the baseline
- whether the judge counted, and the calibration's kappa when there is one
- the failed cases and why they failed
