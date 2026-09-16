# The self-hosted versus API measurement run

Four options, one golden set, one table. This directory holds the evidence: each
option's two report files under `reports/<option id>/`, the GPU host's capture
under a dated directory, and `cost.json` once the spend has landed.

The published table is `docs/tradeoff.md`, built from `project-evals/tradeoff.json`
by `make tradeoff`.

## What you need before you start

- `ANTHROPIC_API_KEY` in `.env`. The two API options send ten real turns each and
  bill a real account. Opus 5 on ten short turns is cents, not dollars, but it is
  not free.
- The local stack's model on disk (`MODELS_DIR`), for the CPU option.
- For the GPU option only: a Terraform Cloud workspace
  `ai-operations-console-aws-gpu` in local execution mode, an AWS G-instance vCPU
  quota of at least 4 in `eu-central-1`, and `infra/terraform/aws-gpu/terraform.tfvars`
  filled in. The instance bills by the hour from the moment apply finishes.
- Nothing else on the machine holding ports 8080, 6379, 5432, 9091, 6333, 8082,
  8083 or 8090. The companion Helpdesk stack publishes four of them.

## The four runs, in order

Each option is measured on its own, with the harness torn down in between, so no
option inherits another's warm cache or loaded model.

| Option | Host it runs on | Command |
| --- | --- | --- |
| `cheap-api` | anywhere | `make tradeoff-option OPTION=cheap-api TARGET=http://localhost:8090` |
| `strong-api` | anywhere | `make tradeoff-option OPTION=strong-api TARGET=http://localhost:8090` |
| `local-cpu` | the Hetzner CAX21 | `make tradeoff-option OPTION=local-cpu TARGET=http://localhost:8090` |
| `gpu-vllm` | the AWS g5.xlarge, via `tradeoff-gpu` | `make tradeoff-gpu SHA=<published commit>` |

The two API options may run anywhere: their cost comes from the ledger's own
per-turn number and is hardware-independent. The CPU option may not. It is priced
at the CAX21's hourly rate, so measuring its throughput anywhere faster would
price ARM inference at another box's rate and publish a CPU option that looks
better than the one this project actually runs.

`tradeoff-gpu` does the whole GPU half in one command: apply, deploy, wait for the
weights and the rollout, measure, capture, destroy. The apply is interactive, and
you read the plan and type yes. Everything after it is covered by an EXIT trap that
destroys the instance on any failure.

About a day later, once Cost Explorer has the spend:

    make tradeoff-gpu-cost START=<the run's date> END=<the next date>

Then build the table and fill in the row below:

    make tradeoff MANIFEST=project-evals/tradeoff.json OUT=docs

## How the numbers were produced, and what they cannot tell you

Read this before quoting anything from the table.

**Ten cases.** The set is `project-evals/golden/e2e-kind.jsonl`, the ten-case
stand-in the kind end-to-end run uses. The Helpdesk golden set lives in the
companion repository and does not exist yet. Ten cases move `decision_accuracy` in
ten-point steps, which is coarser than the table's two-point grouping band: expect
the options to be told apart on cost and latency, rarely on accuracy.

**One sequential single-stream pass.** Every turn is sent, answered and scored
before the next one starts. That measures per-turn latency honestly and measures
throughput under concurrency not at all — which is the one thing vLLM's continuous
batching exists to improve. The GPU option's cost per 1,000 turns is therefore a
ceiling, not the number a batched workload would see.

**A dedicated instance per self-hosted option.** Nothing else runs on the box
while it is measured, so the hourly rate divided by the measured throughput is a
fair price for that box's full attention.

**The GPU turns carry the operator's round trip.** The GPU host opens SSH and
nothing else, so the harness reaches vLLM through an SSH local forward from
wherever the run is started. That adds one network round trip per turn to the
measured latency, and therefore inflates that option's cost per 1,000 turns. It
biases against the GPU, never for it.

**The cache is off.** Every overlay sets `semantic_cache: false` and the run
flushes the `response` key prefix before each option. A second option reading the
first's answers would report a cost no provider charged.

## The run's own cost

Filled in by the `/quick` that commits this directory's evidence.

| Item | USD |
| --- | --- |
| GPU instance (g5.xlarge, applied and destroyed) | PENDING |
| API spend (both API options, from the ledger) | PENDING |
| Total | PENDING |
