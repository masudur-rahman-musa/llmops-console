# The eval gate

A run says what happened. The gate says whether it may merge.

    ops-eval gate --report reports/report.json \
                  --baseline backend/evals/baseline.json \
                  [--comment gate-comment.md]

It reads the files you name and touches nothing else. There is no database, no
gateway and no clock in it. The same report and the same baseline give the same
answer on a laptop and in CI, so a red gate can be argued about rather than
guessed at.

## The two rules

| Rule | Breaches when |
|---|---|
| `safety` | below 100, or missing |
| `outcome_every_play` | below the baseline's `outcome_every_play`, or missing |

`safety` is the share of scored plays that broke no rule. Full safety is the floor, not
a number from the baseline: one play with a broken rule fails the gate, and a baseline
file written with less cannot let it through.

`outcome_every_play` is the share of cases whose scored plays all reached a listed end
state. There is no drop allowance: one point under the baseline fails, and an equal score
passes. A real model does not score the same twice, so the gate plays each case three
times and a case counts only when every play passed. The fix for a flaky case is to fix
the chat, never a wider allowance.

A score that is missing breaches the rule. A run where every play was set up wrong has
no score, and reading that as "nothing against us" would merge anything. The reason line
says `no evidence` rather than a number.

Plays that need a look and plays that were set up wrong are counted apart. They are named
in a note on the verdict and the comment and decide nothing.

## The numbers are points

Every score is a percentage between `0.0` and `100.0`, at two decimal places. The
baseline is written in the same unit as the report, so no conversion lives in the
comparison. A baseline is validated whole before any comparison runs: an unknown key
or a number outside `0.0` to `100.0` refuses the file.

The baseline holds `safety` and `outcome_every_play`, and nothing else.

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

`backend/evals/baseline.json` is the record of what the system is known
to do. It changes only in a commit whose subject begins `baseline:`, with an ADR
saying what moved and why. It is never edited to make a run pass.

CI enforces the rule, not a reviewer:

    ops-eval baseline-guard --base <ref> --head <ref>

It reads `base..head` once and answers a verdict over the commits in it. A commit
that does not touch a baseline file is not examined. One that does is clean only when
its subject starts `baseline:` and that same commit carries a new decision record.
The guard watches the baseline and also the retired score file of the earlier design, so
that file cannot come back quietly.

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

1. It makes three throwaway gateway keys: one per help desk tenant, and one for the
   `evals` tenant that the simulated person and the judge use. The digests go to the
   gateway and the keys to the help desk. Nothing is stored.
2. It starts the gateway with `infra/evals/compose.yml`. The gateway keeps its shipped
   routing, so the help desk's chat runs on the real cheap model (`agent-default`).
   A run costs roughly $3 to $6.
3. It resets and seeds the help desk's database and loads its policy text
   (`infra/evals/prepare_helpdesk.sh`), then starts its agent API and tool server with
   the eval routes switched on (`infra/evals/helpdesk.compose.yml`).
4. It plays the help desk's 45 conversation cases three times each
   (`EVAL_PLAYS ?= 3`) with `ops-eval conversations --plays`. The help desk data is reset
   before every play (`infra/evals/reset_helpdesk.sh`). The reset keeps the loaded policy
   rows, then checks that every policy in Qdrant still has its row; when one does not,
   policy search would fail, so the run stops there. It writes the reports to
   `backend/evals/reports/local/` and runs `ops-eval gate` against the baseline.

A mock cannot play a conversation. A case has no fixed answer for each model call,
and recordings drift with every prompt change. So the gate runs on a real model,
and every run spends real tokens.

The report folder is uploaded as a build artifact whatever the verdict, because a
failed gate is when somebody opens it. The comment is posted once per pull request
and edited in place on a re-run. On a pull request from a fork the token is
read-only, so the comment is skipped and the artifact is still published.

## What the comment carries

- the verdict line
- the run's `safety` and `outcome_every_play` against what the gate needs
- how many plays need a look or were set up wrong
- the breached rules and why

The workflow posts that markdown on the pull request and attaches the report as a
build artifact.
