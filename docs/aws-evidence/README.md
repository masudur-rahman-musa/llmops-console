# AWS evidence

The platform normally runs on one Hetzner node with k3s. This folder holds the
record of running it once on AWS managed services instead: EKS, RDS PostgreSQL,
ElastiCache Redis and an S3 bucket reached through IAM roles for service
accounts. The stack was applied, deployed, smoke tested and destroyed in one
sitting, and what it cost is recorded below.

## What the run does

`make aws-evidence SHA=<commit>` runs `infra/scripts/aws_evidence.sh run`, which:

1. Runs `terraform apply` on `infra/terraform/aws`. You read the plan and type yes.
2. Writes a kubeconfig for the new cluster into a temporary directory that is
   removed on exit. Your own kubeconfig is never touched.
3. Applies `infra/k8s/aws`: the namespaces and their quotas, Qdrant, the gateway,
   the two llama.cpp servers and the `platform/backups` service account.
4. Builds the gateway's Secret at run time from the RDS master credential in
   Secrets Manager and freshly generated keys, and pipes it to `kubectl`. No
   credential is written to disk or printed.
5. Points the gateway at RDS over TLS and at ElastiCache over `rediss://`, and
   sets its image to the given commit.
6. Waits for the rollouts, creates the ledger tables, and runs the same smoke
   test as a production deploy: `/readyz` answers 200, then one chat turn as the
   `evals` tenant with `X-Data-Class: pii`, which only the in-cluster model may
   serve.
7. Runs a one-shot Job as `platform/backups` that writes one object to the
   backups bucket, proving the pod assumed the bucket role.
8. Captures the evidence into `docs/aws-evidence/<UTC date>/`, with every AWS
   account ID replaced by `ACCOUNT_ID`.
9. Runs `terraform destroy -auto-approve`.

Step 9 is an exit trap. Once apply has started, destroy runs whatever happens
next: a failed step, Ctrl-C or a clean finish. If destroy itself fails, the
script says so and exits non-zero, and you run
`terraform -chdir=infra/terraform/aws destroy` yourself before walking away.

## Before you run it

- An AWS account with no other spend. The cost capture reads the whole account,
  because a cost allocation tag only starts counting after its first resource
  exists and would miss this run.
- AWS credentials in your shell (for example `aws sso login`) that can create
  and destroy everything in `infra/terraform/aws`.
- `terraform`, `aws` (v2), `kubectl`, `curl` and `python3` on your `PATH`.
- `infra/terraform/aws/terraform.tfvars` filled in from the example, and
  `terraform -chdir=infra/terraform/aws init` run once against the
  `ai-operations-console-aws` workspace in local execution mode.
- `kubernetes_version` checked against the EKS release calendar. A version in
  extended support costs more per hour than the whole budget.
- The gateway image for the commit published to GHCR by the build workflow, and
  the package readable by the cluster.
- About an hour, most of it EKS creation and the 2 GB model download.

## Screenshots to take while the stack is up

Save them in the dated folder, with the account ID cropped or blurred.

1. EKS: the cluster's Compute tab, showing the managed node group with two
   `t4g.medium` nodes Ready.
2. RDS: the database's Connectivity and security tab, showing
   "Publicly accessible: No" and the private subnets.
3. ElastiCache: the Redis cluster details, showing encryption in transit and at
   rest enabled.
4. After destroy: Resource Groups and Tag Editor filtered by the `project` tag,
   showing no resources left.

## What is captured

| File | Contents |
|---|---|
| `terraform-output.json` | The module outputs: region, cluster name and endpoint, RDS and Redis endpoints, bucket |
| `nodes.txt` | `kubectl get nodes -o wide` |
| `pods.txt` | `kubectl get pods --all-namespaces -o wide` |
| `smoke.json` | The `/readyz` body and the chat turn's response |
| `irsa.txt` | The IRSA Job's log: the assumed role and the S3 write |
| `../cost.json` | Cost Explorer, daily, grouped by service |

## What this run does not prove

NetworkPolicies are applied but not enforced. The EKS VPC CNI enforces them only
when its network policy agent is switched on, and the module leaves it off. The
run proves the platform moves to managed Postgres, Redis and workload identity
without an application change. The egress backstop, where a NetworkPolicy
stops anything but the gateway from reaching a provider, is proven on the k3s
cluster, which enforces it.

The Traefik routers, cert-manager, the Helpdesk workloads and the backup CronJobs
are not deployed. EKS carries none of the Traefik or cert-manager CRDs, and the
Helpdesk images are published from their own repository.

## Cost

Cost Explorer lags by up to a day. Once the spend has landed, run
`make aws-evidence-cost START=<run date> END=<day after>` and fill in the table
from `cost.json`.

| Service | USD |
|---|---|
| | |
| **Total** | |
