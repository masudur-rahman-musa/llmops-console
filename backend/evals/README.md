# Evals

The eval engine plays a workload's conversations. Each case is played several times, one
turn per call, each play from freshly reset data. A simulated person answers from what the
case says it knows, and code checks the shared rules and the case's own never rules. The
world after the chat is matched to the case's good end states, and a judge reads the
replies a case marks for checking.

What is in this folder:

- `kind-cases.jsonl`: three small cases for the kind end-to-end run and the
  self-hosted versus API comparison.
- `target-stub/`: the stub target those two runs use. It answers each turn with an
  empty world and no tool call.
- `baseline.json`: the scores a later run must not fall below. Changes
  only in a commit prefixed `baseline:` with an ADR. Never edited to make a run pass.
- `labels/`: the judge's reference labels (`labels.jsonl`, shipped empty: the rows are
  prepared outside the project and reviewed by the owner). A `calibration.json` beside
  them, written by `ops-eval calibrate`, marks which check types the judge is trusted on.
- `reports/`: what a run writes (ignored by git).

The real cases are the companion help desk's, `backend/evals/conversations/cases.jsonl`
in its repository. They are never copied here.

## Running it

    make eval

Starts the gateway and the help desk, plays all of the help desk's cases three times each
on the real cheap model, and gates the result. It needs `ANTHROPIC_API_KEY` (in your shell
or in `backend/.env`). Reports land in `backend/evals/reports/local/`.

By hand, against any target:

    ops-eval conversations --target URL --cases PATH --out DIR [--plays N] \
        [--gateway-url URL] [--reset-command CMD] [--calibration PATH]
    ops-eval calibrate --labels backend/evals/labels/labels.jsonl --gateway-url URL \
        --out backend/evals/labels/calibration.json
    ops-eval gate --report DIR/report.json --baseline backend/evals/baseline.json

The formats are in `docs/eval-guide.md` and `docs/eval-report.md`.

## Reply quality counts only trusted checks

Reply quality counts only the check types `calibration.json` marks trusted, and only while
the judge prompt is the one the calibration was made with. `make eval` passes the file
when it exists. Without it, or after the judge prompt changes, reply quality is empty.
Safety and the outcome scores never depend on the judge: a judge-only answer can make a
play "needs a look", never a failure.
