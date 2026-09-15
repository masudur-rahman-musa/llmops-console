# aws-gpu

One GPU instance for a vLLM measurement run: applied, used for a few hours,
destroyed. The live system never depends on it, and `routing.yaml` keeps the
`vllm` provider disabled.

- One EC2 instance, `g5.xlarge` by default (one A10G, 24GB), in the account's
  default VPC.
- Amazon's Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), which
  ships the NVIDIA driver and container toolkit.
- cloud-init installs k3s at the same release as the Hetzner cluster and arms a
  shutdown timer. The instance terminates on shutdown.
- A security group with one inbound rule: SSH from `admin_ips`.
- A 100GB encrypted gp3 root volume, deleted with the instance.
- A monthly cost budget of 15 USD that mails you at 80 percent of actual spend.

The state lives in the Terraform Cloud workspace
`ai-operations-console-aws-gpu`, in local execution mode.

## Before the first apply: the GPU quota

A new AWS account's on-demand quota for G and VT instances is 0 vCPUs, and the
apply fails with `VcpuLimitExceeded`. In Service Quotas, for `eu-central-1`,
request "Running On-Demand G and VT instances" at 4 vCPUs (one `g5.xlarge`).
Approval can take a day, so ask before the run is scheduled.

## What it expects

Inputs with no default, so a plan refuses rather than guesses:

- `admin_ips`: the CIDRs allowed to SSH in. Every entry must be a CIDR, and an
  empty list is refused.
- `ssh_public_key`: your ed25519 public key.
- `budget_email`: where the budget alert goes.

`region` (EU only), `name`, `instance_type` (`g5.xlarge` or `g6.xlarge`) and
`auto_terminate_minutes` (30 to 480, default 240) have defaults and are
validated too.

## Running it

    cp terraform.tfvars.example terraform.tfvars   # then fill it in
    terraform init
    terraform plan

Apply and destroy are human steps, run from a human's shell, never by CI and
never by an agent. Destroy even if the timer has already terminated the
instance: the key pair, security group and budget outlive it.

## Reaching the cluster

Nothing but SSH is open. Forward the k3s API over SSH and use the kubeconfig
from the host:

    ssh -L 6443:127.0.0.1:6443 ubuntu@<public_ip>
    ssh ubuntu@<public_ip> sudo cat /etc/rancher/k3s/k3s.yaml > gpu.kubeconfig
    KUBECONFIG=gpu.kubeconfig kubectl get nodes

Then apply `infra/k8s/gpu` (see its README). The file is written with mode 0600
on the host and holds cluster-admin access, so keep your copy out of the repo.

## CI

The `tf-aws-gpu` job runs `fmt -check`, `init -backend=false` and `validate` on
every pull request, and `plan` only when both `TF_API_TOKEN` and
`AWS_PLAN_ROLE_ARN` are set. It reuses the read-only plan role described in
`../aws/README.md`.
