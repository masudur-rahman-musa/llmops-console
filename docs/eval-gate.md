# The eval gate

A run says what happened. The gate says whether it may merge.

    ops-eval gate --report reports/report.json \
                  --baseline project-evals/baseline.json \
                  [--comment comment.md]

It reads two files and touches nothing else: no database, no gateway, no clock.
The same report and the same baseline give the same answer on a laptop and in
CI, which is what makes a red gate arguable rather than a mystery.

## The five rules

| rule | breaches when | where the threshold comes from |
|---|---|---|
| `decision_accuracy` | below the reference less the allowed drop | `metrics.decision_accuracy` - `thresholds.max_drop_points` |
| `refusal_recall` | below the floor | `thresholds.refusal_recall_min` |
| `injection_resistance` | below the floor | `thresholds.injection_resistance_min` |
| `cross_tenant_leaks` | above zero | fixed: any leak at all |
| `residency_violations` | above zero | fixed: any violation at all |

Every rule is reported, every time, whether it passed or not. A reviewer who is
shown only the failures cannot tell a rule that passed from a rule the run never
measured.

The boundary is inclusive in the run's favour: a run landing exactly on `95.0`,
or dropping exactly `2.0` points, is green. The comparisons are `<` and `>`,
which is how the spec writes them.

**A metric that is `null` breaches its rule.** a design note makes a metric `null`
when no case stated the expectation, so a golden set that gates refusal recall
and states no refusal case reports `null` - and reading that as "nothing against
us" merges the lost cases green forever. The reason line says `no evidence`
rather than a number, because "this got worse" and "the cases that measured it
are gone" have different fixes.

## The numbers are points

Every threshold and every metric is a percentage between `0.0` and `100.0`, at
the two decimal places `ACCURACY_SCALE` publishes. The baseline is written in
the same unit as the report, so no conversion lives in the comparison. See
`docs/adr/0008-the-baseline-is-written-in-points.md` for why that changed.

A baseline is validated whole before any comparison runs: an unknown key, a
threshold outside `0.0`-`100.0`, or a `golden_set` over 128 characters refuses
the file rather than reading past it. A file half of whose numbers nobody reads
is worse than a file that will not load.

## The exit code is the interface

| code | meaning |
|---|---|
| `0` | every rule cleared |
| `1` | at least one threshold rule breached |
| `2` | a usage error: an input that could not be read, parsed or validated |
| `3` | a residency violation or a cross-tenant leak |

`3` outranks `1`. A boundary breach read as an accuracy dip is a breach somebody
merges over. `2` is distinct from `1` so a workflow can tell "the gate never ran"
from "the run got worse" without parsing text.

Stdout carries exactly one line, the verdict. The breached rules go to stderr,
one line each. The markdown `--comment` writes opens with that same verdict line,
so the summary a workflow reads and the comment a person reads cannot disagree.

## The judge decides nothing

Neither `judge_mean` nor `judge_kappa_vs_human` is a gate rule. An LLM judge
scores against a rubric and is itself calibrated against human labels, so it
reports and never decides: a run whose judge agreed with humans and one whose
kappa was `0.42` get the same verdict from the same metrics. The comment states
the exclusion, and states the run's kappa where there is one, because the comment
is where the merge decision is made.

## Changing the baseline

`project-evals/baseline.json` is the record of what the system is known to do. It
changes only in a commit whose subject begins `baseline:`, with an ADR saying
what moved and why. It is never edited to make a run pass - a gate that can be
edited green is a gate nobody has.

That prefix is the one commit subject in this repository that carries a colon:
the house style has no `type: subject` prefixes, and this rule predates it and
names the literal string.

The rule is enforced in CI, not by review:

    ops-eval baseline-guard --base <ref> --head <ref>

It reads `base..head` once with `git log --format=%H%x00%s --name-only` and
answers a verdict over the commits in it. A commit that does not touch
`project-evals/baseline.json` is not examined. One that does is clean only when
its subject starts `baseline:` **and** that same commit carries a file under
`docs/adr/`.

Same commit, not same pull request. A merge and a revert can separate a baseline
change from its justification across a pull request; they cannot separate them
inside one commit.

The path is matched exactly. `project-evals/baseline.json.bak` is not the
baseline and neither is a file of that name under another directory, so a rename
beside the file does not block a pull request that never touched it.

| code | meaning |
|---|---|
| `0` | the range is clean; stdout carries one line naming how many commits were read |
| `1` | at least one commit breaks the rule; each offending sha is one line on stderr, naming which half it broke |
| `2` | `--base` or `--head` was refused, or git could not read the range |

`2` is what a shallow checkout gets. `git log base..head` resolves nothing at
depth 1, and a guard that read no commits must never answer "clean" - so the
workflow checks out with `fetch-depth: 0` and an unresolvable range is a usage
error rather than an empty verdict. Both refs are checked against a pattern and a
256-character cap before git is asked anything, since both reach a subprocess
argument list.

## The workflow

`.github/workflows/eval-gate.yml` runs on every pull request into `main`, in two
jobs that fail for different reasons.

`baseline-guard` is the rule above, unconditional.

`eval` runs the golden set and compares it to the baseline. **It is skipped
unless the repository variable `EVAL_TARGET_URL` is set**, and today nothing
serves the eval target contract, so it is skipped everywhere. A stub target would
score itself against a baseline this repository wrote, and its green tick would
read like evidence.
`docs/adr/0009-the-eval-gate-ships-inert-without-a-target.md` has the reasoning.
Setting that variable is what turns the eval half on, with no code change.

When it does run: the stack comes up, the run happens against the target with
`project-evals/ci/routing.ci.yaml` merged over `routing.yaml` (which caps the
`evals` tenant at `$2.00` for the run), then `ops-eval gate` produces the verdict
and the comment. The report directory is uploaded as a build artifact whatever
the verdict, because a failed gate is when somebody opens it. The comment is
posted once per pull request and edited in place on a re-run; on a fork's pull
request the token is read-only, so the comment is skipped and the artifact is
still published.

## What the comment carries

- the verdict line, and the run's identity (set, sha, run id)
- one row per rule: actual, threshold, pass or breach
- the run's per-category table, which is what turns "decision accuracy fell two
  points" into "policy_qa fell over"
- what the run's calibration said, and the judge exclusion above

The workflow posts that markdown on the pull request and attaches the report as a
build artifact.
