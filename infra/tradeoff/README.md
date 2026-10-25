# `infra/tradeoff/`

One partial routing policy per measurement option. `infra/scripts/tradeoff_policy.py`
deep-merges the file named for an option id over the shipped `routing.yaml` and
writes the result somewhere temporary, with a provider folder beside it: the
shipped `backend/config/providers/` plus any file in `providers/` here that the
option's policy names. `infra/tradeoff/compose.yml` mounts both into the gateway
read-only. The shipped policy and provider files are never edited.

Partials rather than whole copies: each file then says only what makes
its option different, and nothing here can drift from the policy the gateway
really runs.

| File | What it pins |
| --- | --- |
| `cheap-api.yaml` | `anthropic` / `claude-haiku-5-5` |
| `strong-api.yaml` | `anthropic` / `claude-sonnet-5-5` |
| `gpu-vllm.yaml` | `vllm` and the 7B model it serves |
| `providers/vllm.yaml` | the `vllm` provider for `gpu-vllm`, switched on, at its NodePort address |

## Why every option turns the semantic cache off

`cache_key` includes the resolved model, so the exact tier is already safe across
options. The semantic tier is not keyed by model. With it on, the second option
measured would answer from the first's replies and publish a near-zero cost at a
latency no provider served. `infra/scripts/tradeoff_run.sh` flushes the `response` key
prefix between options for the same reason.

## What none of these files may contain

No key, no digest, no `key_env`. The providers' own credentials come from `.env`
as they always do, and the run's tenant key is generated per run and
written nowhere. A test scans this directory for every option file.
