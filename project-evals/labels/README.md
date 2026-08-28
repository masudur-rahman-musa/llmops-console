# Judge calibration labels
Human-graded outputs (1 to 5 against the rubric) used to check the LLM judge
before trusting it. Minimum 40 items. Store as `*.jsonl`: {"id", "output",
"human_score", "note"}. Report Cohen's kappa in every eval run.
