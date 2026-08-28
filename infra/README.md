# infra/
Infrastructure as code. `terraform/` provisions (servers, network, DNS, managed
services); `k8s/` deploys (manifests or kustomize). Rules: `terraform plan` and
`kustomize build` run in CI on every change; `apply` and `kubectl apply` against
a real environment are run by a human, never by the agent. State is remote and
locked. Secrets are encrypted (SOPS) or injected, never plain in manifests.
