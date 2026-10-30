# infra/

Everything around the app. Only `local/`, `evals/` and `load/` are part of the
local path. The rest is deployment code that CI checks but nothing here applies.

| Folder | What it holds | Used by |
|---|---|---|
| `local/` | Prometheus scrape config, Grafana data source and dashboard loading, and the `make numbers` overlay | `docker-compose.yml`, `make numbers` |
| `evals/` | compose overlays and scripts that run the gateway beside the companion help desk for an eval | `make eval` |
| `load/` | the k6 script, a fixed-delay mock provider and its routing policy | `make load` |
| `tradeoff/` | one partial routing policy per option in the self-hosted versus API comparison | `make tradeoff-option`, `make tradeoff` |
| `k8s/` | kustomize manifests for a small k3s cluster, plus an AWS overlay, a GPU build and a kind overlay for CI | `make k8s-check`, CI |
| `terraform/` | modules for one Hetzner node, an AWS EKS stack and one AWS GPU instance | `make infra-check`, CI |
| `secrets/` | example secrets for the cluster. Real ones are SOPS files encrypted with age | `make secrets-check` |
| `scripts/` | the AWS evidence and tradeoff run scripts | `make aws-evidence`, `make tradeoff-gpu` |
| `tests/` | tests that read all of the above as files, and the CI workflows too | `make test-infra` |

## Rules

- `terraform validate` and `kustomize build` piped to kubeconform run in CI on
  every change. `terraform plan` runs when its credentials are set.
- `terraform apply` and `kubectl apply` against a real environment are run by a
  person, never by CI or by an agent. The scripts that apply something stop for
  you to read the plan and type yes, and an exit trap destroys what they made.
- Terraform state lives in Terraform Cloud, never in the repo.
- Secrets are SOPS-encrypted or passed in at run time, never plain in a manifest,
  compose file or image.

See [k8s/README.md](k8s/README.md) for the cluster layout and its resource budget.
