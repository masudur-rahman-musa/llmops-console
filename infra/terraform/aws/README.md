# aws

The same stack on managed AWS services, built to be applied once for evidence
and destroyed the same day. The live system runs on `../hetzner`; this module
exists to show the platform moves to a hyperscaler without changing the app.

- A VPC across two availability zones, with public and private subnets and one
  NAT gateway.
- EKS with one managed node group of two `t4g.medium` arm64 nodes.
- RDS PostgreSQL 16 on `db.t4g.micro` and ElastiCache Redis 7 on
  `cache.t4g.micro`, both in the private subnets. Each is reachable only from
  the EKS node security group, on its own port.
- An S3 backups bucket with every public-access flag on, encryption and
  versioning, and one IRSA role that can reach that bucket and nothing else.
- A monthly cost budget of 10 USD that mails you at 80 percent of actual spend.

One flat module. The VPC and EKS come from the pinned community modules
(`terraform-aws-modules/vpc/aws` 6.7.2, `terraform-aws-modules/eks/aws`
21.25.0); everything else is plain resources. The state lives in the Terraform
Cloud workspace `ai-operations-console-aws`, in local execution mode.

## What it expects

Inputs with no default, so a plan refuses rather than guesses:

- `admin_ips`: the CIDRs allowed to reach the EKS public API endpoint. Every
  entry is validated as a CIDR, and an empty list is refused.
- `budget_email`: where the budget alert goes.

`region` (EU only, default `eu-central-1`), `name` and `kubernetes_version`
have defaults and are validated too. Check `kubernetes_version` against the
EKS release calendar before an apply: a version in extended support bills
about 0.60 USD an hour extra, more than the whole budget.

Credentials are never variables. The provider reads them from the environment,
for example after `aws sso login`. The database's master credential is created
and held by AWS Secrets Manager (`manage_master_user_password`), so it never
appears in state, a variable or an output.

## Running it

    cp terraform.tfvars.example terraform.tfvars   # then fill it in
    terraform init
    terraform plan

Apply and destroy are human steps, run from a human's shell, never by CI and
never by an agent. Destroy is designed to finish cleanly: no final RDS snapshot,
no deletion protection, and the bucket is emptied with it.

## CI and the read-only plan role

CI runs `fmt -check`, `init -backend=false` and `validate` on every pull
request. `plan` runs only when the repository has both the `TF_API_TOKEN`
secret and the `AWS_PLAN_ROLE_ARN` variable; otherwise the job logs that it
skipped. The plan uses placeholder values for the two required inputs.

The plan role is created once, by hand, outside this module, because this
module is destroyed and the role must outlive it. In the AWS account:

1. Add the GitHub OIDC identity provider: URL
   `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
2. Create a role with the AWS managed policy `ReadOnlyAccess` attached, and
   this trust policy (replace `ACCOUNT_ID`):

        {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Principal": {
                "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
              },
              "Action": "sts:AssumeRoleWithWebIdentity",
              "Condition": {
                "StringEquals": {
                  "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                  "token.actions.githubusercontent.com:sub": "repo:masudur-rahman-musa/ai-operations-console:*"
                }
              }
            }
          ]
        }

3. Set the role's ARN as the repository variable `AWS_PLAN_ROLE_ARN`.

`ReadOnlyAccess` can describe every resource the plan refreshes and cannot
change any of them, and it does not include reading secret values. No
long-lived AWS access key exists for CI at all. The Terraform Cloud workspace
must be set to local execution mode, so the plan runs on the CI runner with
these credentials.

## Service account

The backups role trusts the Kubernetes service account `backups` in the
`platform` namespace. Annotate that service account with the role's ARN
(`eks.amazonaws.com/role-arn`) when the stack is deployed onto this cluster.
