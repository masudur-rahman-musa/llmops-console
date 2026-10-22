# The eval gate

A run says what happened. The gate says whether it may merge.

    ops-eval gate --report reports/report.json \
                  --baseline backend/evals/conversation-baseline.json \
                  [--calibration labels/calibration.json] \
                  [--comment comment.md]

It reads the files you name and touches nothing else: no database, no gateway, no
clock. The same report and the same baseline give the same answer on a laptop and in
CI, which is what makes a red gate arguable rather than a mystery.

## The two rules

| rule | breaches when |
|---|---|
| `check_score` | below the baseline's `check_score`, or missing |
| `judge_score` | below the baseline's `judge_score`, but only while the judge is trusted |

`check_score` is the percent of stable cases that passed all three checks. There is
no drop allowance: one point under the baseline fails, and an equal score passes. A
real model does not score the same twice, so the gate plays each case three times. A
case that both passes and fails is unstable: it is left out of `check_score` and named
in a note on the verdict and the comment, so it stays in view without deciding
anything. A run where every case is unstable has no score, and that breaches as `no
evidence`. The fix for a flaky case is to fix the chat, never a wider allowance.

A `check_score` that is missing breaches the rule. A run over no cases has no score,
and reading that as "nothing against us" would merge anything. The reason line says
`no evidence` rather than a number.

`judge_score` is the percent of graded replies the judge passed. It counts toward the
gate only when a calibration file is given and its kappa is 0.7 or better. Below that,
or with no calibration at all, the run goes on, the score is reported, and the
comment says the judge decided nothing. The judge's score is reported on every run.

## The numbers are points

Every score is a percentage between `0.0` and `100.0`, at two decimal places. The
baseline is written in the same unit as the report, so no conversion lives in the
comparison. A baseline is validated whole before any comparison runs: an unknown key
or a number outside `0.0` to `100.0` refuses the file.

The baseline holds `check_score`, and optionally `judge_score`, and nothing else.

## The exit code is the interface

| code | meaning |
|---|---|
| `0` | every rule cleared |
| `1` | at least one rule breached |
| `2` | an input could not be read, parsed or validated |

`2` is distinct from `1` so a workflow can tell "the gate never ran" from "the run
got worse" without reading text.

Stdout carries exactly one line, the verdict. The breached rules go to stderr, one
line each. The markdown `--comment` writes opens with that same verdict line.

## Changing the baseline

`backend/evals/conversation-baseline.json` is the record of what the system is known
to do. It changes only in a commit whose subject begins `baseline:`, with an ADR
saying what moved and why. It is never edited to make a run pass.

The rule is enforced in CI, not by review:

    ops-eval baseline-guard --base <ref> --head <ref>

It reads `base..head` once and answers a verdict over the commits in it. A commit
that does not touch a baseline file is not examined. One that does is clean only when
its subject starts `baseline:` and that same commit carries a file under the project notes.
The guard watches the conversation baseline and also the old single-reply baseline
file, so that file cannot come back quietly.

Same commit, not same pull request. A merge and a revert can separate a baseline
change from its justification across a pull request; they cannot separate them inside
one commit.

| code | meaning |
|---|---|
| `0` | the range is clean; stdout names how many commits were read |
| `1` | at least one commit breaks the rule; each offending sha is one line on stderr |
| `2` | `--base` or `--head` was refused, or git could not read the range |

`2` is what a shallow checkout gets. `git log base..head` resolves nothing at depth 1,
and a guard that read no commits must never answer "clean", so the workflow checks
out with `fetch-depth: 0` and an unresolvable range is a usage error.

## The workflow

`.github/workflows/eval-gate.yml` runs on every pull request into `main`, in two jobs
that fail for different reasons.

`baseline-guard` is the rule above, unconditional.

`eval` runs on every pull request, with no condition. It checks out the help desk
repository beside this one (it is private, so the job reads it with the
`HELPDESK_REPO_TOKEN` secret, a read-only token), writes the help desk's `.env.demo`
from its `.env.example`, and runs `make eval` with `ANTHROPIC_API_KEY` from a GitHub
secret. Until a secret is set the job fails; it never skips.

`make eval` does the whole run, the same on a laptop (`HELPDESK_DIR` names the help
desk checkout, `../ai-help-desk-governed-access` by default; the key can be in
`backend/.env`):

1. It makes two throwaway gateway keys, one per help desk tenant. The digests go to
   the gateway and the keys to the help desk. Nothing is stored.
2. It starts the gateway with `infra/evals/compose.yml`. The gateway keeps its shipped
   routing, so the help desk's chat runs on the real cheap model (`agent-default`).
   A run costs roughly $3 to $6.
3. It resets and seeds the help desk's database and loads its policy text
   (`infra/evals/prepare_helpdesk.sh`), then starts its agent API and tool server with
   the eval routes switched on (`infra/evals/helpdesk.compose.yml`).
4. It plays the help desk's 33 conversation cases three times each
   (`EVAL_REPEATS ?= 3`) with `ops-eval conversations --repeats`,
   writes the reports to `backend/evals/reports/local/`, and runs `ops-eval gate`
   against the conversation baseline (with the calibration file when one exists).

A mock cannot play a conversation: a case has no fixed answer for each model call, and
recordings drift with every prompt change. So the gate runs on a real model.

`RUN_ID=<id> make eval` continues a stopped run: the cases already saved under that id
are skipped, and the reports folder is kept.

The report folder is uploaded as a build artifact whatever the verdict, because a
failed gate is when somebody opens it. The comment is posted once per pull request and
edited in place on a re-run; on a fork's pull request the token is read-only, so the
comment is skipped and the artifact is still published.

## What the comment carries

- the verdict line
- the run's `check_score` and `judge_score` against the baseline
- whether the judge counted, and the calibration's kappa where there is one
- the failed cases and why

The workflow posts that markdown on the pull request and attaches the report as a
build artifact.
