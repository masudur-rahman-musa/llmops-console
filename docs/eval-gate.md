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
names the literal string. An earlier change enforces the rule in CI; until then it is
enforced by review.

## What the comment carries

- the verdict line, and the run's identity (set, sha, run id)
- one row per rule: actual, threshold, pass or breach
- the run's per-category table, which is what turns "decision accuracy fell two
  points" into "policy_qa fell over"
- what the run's calibration said, and the judge exclusion above

An earlier change posts that markdown on the pull request and attaches the report as a
build artifact.
