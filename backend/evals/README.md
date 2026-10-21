# Evals

The eval engine scores a workload's multi-turn conversations. It plays each case
through the workload's `POST /v1/eval/conversation` and checks every turn three ways:
the right tools were called in order, nothing was changed before the person's yes, and
no request number, ticket number or role name appears that a tool did not return. A
judge then grades the replies a case marks as graded, pass or fail, against a rubric.

What is in this folder:

- `kind-cases.jsonl`: three small cases for the kind end-to-end run and the
  self-hosted versus API comparison.
- `target-stub/`: the stub target those two runs use. It answers the conversation
  route from the case file alone.
- `conversation-baseline.json`: the scores a later run must not fall below. Changes
  only in a commit prefixed `baseline:` with an ADR. Never edited to make a run pass.
- `labels/`: the reference labels (`conversation-labels.jsonl`) and the judge's
  `calibration.json`.
- `reports/`: what a run writes (ignored by git).

The real cases are the companion help desk's, `backend/evals/conversations/cases.jsonl`
in its repository. They are never copied here.

## Running it

    make eval

Starts the gateway and the help desk, plays all of the help desk's cases once on the real
cheap model, and gates the result. It needs `ANTHROPIC_API_KEY` (in your shell or in
`backend/.env`) and costs roughly $1 to $2. Reports land in `backend/evals/reports/local/`.
`RUN_ID=<id> make eval` continues a stopped run without paying for the saved cases again.

By hand, against any target:

    ops-eval conversations --target URL --cases PATH --out DIR [--run-id ID]
    ops-eval gate --report DIR/report.json --baseline backend/evals/conversation-baseline.json

The formats are in `docs/eval-format.md`, `docs/eval-target.md`, `docs/eval-report.md`
and `docs/eval-gate.md`.

## Why a judge number can arrive marked untrusted

`ops-eval calibrate` plays the judge over the reference labels and writes one
`calibration.json`: how well the judge agreed with the labels, as Cohen's kappa. Below
0.7 the judge is not trusted. The run still happens and the judge's score is still
reported, but it blocks nothing and `report.md` says so. Do not quote the judge's score
as a measurement then. Fix the rubric or the judge prompt and calibrate again.
