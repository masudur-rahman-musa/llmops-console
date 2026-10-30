# AWS evidence

This folder is for the record of running the platform once on AWS managed
services instead of one small k3s node. The services are EKS, RDS PostgreSQL,
ElastiCache Redis and an S3 bucket reached through IAM roles for service accounts.

**No run is captured here yet.** The folder holds only this page. The Terraform
module, the Kubernetes overlay and the script that would make the record are in
the repo and checked by CI (`terraform validate` and `make k8s-aws-check`), but no
apply output, smoke test result or cost file is committed. Nothing on this page
claims a run happened.

This is not part of the local setup. It costs real money and needs an AWS account.

## What the run does

`make aws-evidence SHA=<commit>` runs `infra/scripts/aws_evidence.sh run`. The
script does these steps:

1. Runs `terraform apply` on `infra/terraform/aws`. You read the plan and type yes.
2. Writes a kubeconfig for the new cluster into a temporary folder that is removed
   on exit. Your own kubeconfig is never touched.
3. Applies the `infra/k8s/aws` overlay, which holds the namespaces and their
   quotas, Qdrant, the gateway, the llama.cpp embedding server and the
   `platform/backups` service account.
4. Builds the gateway's secret at run time from the RDS credential in Secrets
   Manager and freshly made keys, and pipes it to `kubectl`. No credential is
   written to disk or printed.
5. Points the gateway at RDS over TLS and at ElastiCache over `rediss://`, and sets
   its image to the given commit.
6. Waits for the rollouts, runs the ledger migrations, and runs a smoke test.
   `/readyz` must answer 200, then one chat turn is sent as the `evals` tenant with
   `X-Data-Class: pii`, redacted and served by Anthropic.
7. Runs a one-shot Job as `platform/backups` that writes one object to the backups
   bucket, which proves the pod took on the bucket role.
8. Saves the evidence into a dated folder here, with every AWS account ID replaced
   by `ACCOUNT_ID`.
9. Runs `terraform destroy -auto-approve`.

Step 9 is an exit trap. Once apply has started, destroy runs whatever happens next,
whether a step fails, you press Ctrl-C or the run finishes. If destroy itself
fails, the script says so and exits non-zero, and you run
`terraform -chdir=infra/terraform/aws destroy` by hand.

## Before you run it

- An AWS account with no other spend. The cost capture reads the whole account,
  because a cost tag only starts counting after its first resource exists.
- AWS credentials in your shell that can create and destroy everything in
  `infra/terraform/aws`.
- `terraform`, `aws` (v2), `kubectl`, `curl` and `python3` on your `PATH`.
- `infra/terraform/aws/terraform.tfvars` filled in from the example, and
  `terraform -chdir=infra/terraform/aws init` run once against your own Terraform
  Cloud workspace.
- `kubernetes_version` checked against the EKS release calendar. A version in
  extended support costs much more per hour.
- The gateway image for the commit, published to GHCR by the build workflow.
- `ANTHROPIC_API_KEY` in your shell. The script stops before apply if it is
  missing.

## What a run would capture

| File | Contents |
|---|---|
| `terraform-output.json` | the module outputs, such as cluster, database and Redis endpoints and the bucket |
| `nodes.txt` | `kubectl get nodes -o wide` |
| `pods.txt` | `kubectl get pods --all-namespaces -o wide` |
| `smoke.json` | the `/readyz` body and the chat turn's response |
| `irsa.txt` | the IRSA Job's log, with the assumed role and the S3 write |
| `../cost.json` | Cost Explorer, daily, grouped by service |

`make aws-evidence-cost START=<run date> END=<day after>` writes the cost file
once Cost Explorer has the spend, which can take a day.

## What a run would not prove

NetworkPolicies are applied but not enforced. The EKS VPC CNI enforces them only
when its network policy agent is on, and the module leaves it off. The run would
show the platform moves to managed Postgres, Redis and workload identity with no
app change. It would not show the egress rule that only the gateway may reach a
provider.

The Traefik routes, cert-manager, the help desk workloads and the backup CronJobs
are not deployed. EKS carries none of the Traefik or cert-manager CRDs, and the
help desk images are published from their own repo.
