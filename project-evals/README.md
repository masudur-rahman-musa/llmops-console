# project-evals (AI layer)

Project evals measure the PRODUCT'S behaviour (model-driven features), unlike
coding evals which judge the coding agent's output. The discipline, distilled
from the portfolio projects:

1. **Golden set** (`golden/*.jsonl`): labelled cases with expected outcomes.
   Grows from production failures. Categories cover the happy path, refusals,
   ambiguity, injection attempts, and any guarded invariant.
2. **Deterministic assertions first** (exact fields, tool calls, forbidden
   phrases); **LLM-as-judge only for free text**, and only after calibration
   against human labels in `labels/` (report agreement; distrust the judge
   below kappa 0.7).
3. **Baseline** (`baseline.json`): the thresholds. Changes only via a commit
   prefixed `baseline:` plus an ADR. Never edited to make a run pass.
4. **Failure taxonomy**: every failed case gets a tag (wrong_output, wrong_tool,
   missed_refusal, false_refusal, injection_followed, timeout, judge_low, ADAPT).
   Read 50 failures by hand before changing anything.
5. **Gate**: `make eval` runs in CI; a regression below baseline blocks merge.
   Nightly runs catch provider drift.
6. Metrics to report: accuracy per category, refusal precision/recall,
   injection_resistance (must be 1.0), p50/p95 latency, cost per case.

Run: `make eval` or `/project-evals`. Reports land in `project-evals/reports/` (ignored).
