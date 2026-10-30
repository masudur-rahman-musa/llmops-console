# The self-hosted versus API measurement run

Three options, one small set of conversation cases, one table. This folder is where
the evidence goes. That is each option's two report files under
`reports/<option id>/`, the GPU host's capture under a dated folder, and `cost.json`
once the spend has landed.

**No run is committed yet.** The folder holds only this page, so there is no table
and no result. The published table would be `docs/tradeoff.md`, built from
`infra/tradeoff/tradeoff.json` by `make tradeoff`.

This is not part of the local setup. The API options spend real tokens and the GPU
option runs a paid AWS instance.

## What you need before you start

- `ANTHROPIC_API_KEY` in `backend/.env`. The two API options play three small
  conversations each and bill a real account.
- For the GPU option only, a Terraform Cloud workspace in local execution mode, an
  AWS G-instance vCPU quota of at least 4 in the module's region, and
  `infra/terraform/aws-gpu/terraform.tfvars` filled in. The instance bills by the
  hour from the moment apply finishes.
- Nothing else on the machine holding ports 8080, 16379, 15432, 9091, 16333, 8083
  or 8090.

## The three runs, in order

Each option is measured on its own, with the harness torn down in between, so no
option inherits another's warm cache or loaded model.

| Option | Model | Where it runs | Command |
| --- | --- | --- | --- |
| `cheap-api` | Anthropic `claude-haiku-5-5` | anywhere | `make tradeoff-option OPTION=cheap-api TARGET=http://localhost:8090` |
| `strong-api` | Anthropic `claude-sonnet-5-5` | anywhere | `make tradeoff-option OPTION=strong-api TARGET=http://localhost:8090` |
| `gpu-vllm` | a 7B open model on vLLM | an AWS g5.xlarge, through `tradeoff-gpu` | `make tradeoff-gpu SHA=<published commit>` |

The models come from the overlays in `infra/tradeoff/`. The two API options may run
anywhere, because their cost comes from the ledger's own per-turn number and does
not depend on the hardware.

`tradeoff-gpu` does the whole GPU half in one command. It applies, deploys, waits
for the weights and the rollout, measures, captures and destroys. The apply is
interactive, so you read the plan and type yes. Everything after it sits behind an
exit trap that destroys the instance on any failure.

About a day later, once Cost Explorer has the spend:

    make tradeoff-gpu-cost START=<the run's date> END=<the next date>

Then build the table:

    make tradeoff MANIFEST=infra/tradeoff/tradeoff.json OUT=docs

## How the numbers are made, and what they cannot tell you

Read this before quoting anything from the table.

**Three cases.** The cases are `backend/evals/kind-cases.jsonl`, the small
conversation file the kind cluster run in CI also uses, played through the stub
target in `backend/evals/target-stub/`. The stub sends each message to the gateway
as a plain chat under the `agent-default` alias, so every option answers the same
messages and the ledger prices them. The stub reports the tool calls the cases
expect, so `check_score` reads 100 for every option and does not tell them apart.
The options are told apart on cost. The help desk's own 45 cases need the help
desk running, not a stub, and are not part of this comparison.

**One sequential single-stream pass.** Every turn is sent, answered and scored
before the next one starts. That measures per-turn latency honestly and measures
throughput under load not at all, which is the one thing vLLM's continuous batching
exists to improve. So the GPU option's cost per 1,000 turns is a ceiling, not what
a batched workload would pay.

**A dedicated instance.** Nothing else runs on the GPU box while it is measured, so
the hourly rate divided by the measured throughput is a fair price for that box's
full attention.

**The GPU turns carry an extra round trip.** The GPU host opens SSH and nothing
else, so the harness reaches vLLM through an SSH tunnel. That adds one network round
trip per turn, which raises that option's cost per 1,000 turns. It biases against
the GPU, never for it.

**The cache is off.** Every overlay sets `semantic_cache: false`, and the run
clears the `response` cache keys before each option. A second option reading the
first one's answers would report a cost no provider charged.

## The run's own cost

Filled in from `cost.json` and the ledger once a run is made.

| Item | USD |
| --- | --- |
| GPU instance (g5.xlarge, applied and destroyed) | not run yet |
| API spend (both API options, from the ledger) | not run yet |
| Total | not run yet |
