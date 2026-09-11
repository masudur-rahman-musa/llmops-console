# hetzner

One ARM node and the DNS that points at it: an `hcloud_ssh_key`, an
`hcloud_firewall`, one CAX21 (arm64) server in `nbg1` running k3s from
cloud-init, and four Cloudflare `A` records - `api.`, `console.`, `helpdesk.`
and `grafana.` under the zone you pass in.

One flat module, no sub-modules. The state lives in Terraform Cloud.

## What it expects

Inputs with no default, so a plan refuses rather than guesses:

- `admin_ips` - the CIDRs allowed to reach SSH. Every entry is validated as a
  CIDR, and an empty list is refused. There is deliberately no default: a
  default here would open port 22 to the internet the first time someone forgot
  the variable.
- `zone` - the Cloudflare zone, as a bare apex domain.
- `ssh_public_key` - the public half of your admin key, as one OpenSSH line.

`subdomains`, `server_type` and `location` have defaults and are validated too.
`location` accepts EU locations only.

Credentials are never variables. The providers read `HCLOUD_TOKEN` and
`CLOUDFLARE_API_TOKEN` from the environment; in CI the plan runs inside the
Terraform Cloud workspace, which holds them and the three required inputs as
workspace variables.

## Running it

    cp terraform.tfvars.example terraform.tfvars   # then fill it in
    terraform init
    terraform plan

Applying and tearing down are human steps, run from a human's shell, never by
CI and never by an agent. CI runs `fmt -check`, `init -backend=false` and
`validate` on every pull request, and `plan` only when the Terraform Cloud token
is present.

Weekly snapshots are a manual step: take them from the Hetzner console or with
`hcloud server create-image`, and record when you last did.

## What is not here yet

The records resolve to a box serving nothing until the cluster is deployed: the
kustomization, its Ingresses and cert-manager arrive in the next row of the
build order. A 404 or a connection refused on `console.<zone>` before then is
the expected state, not a broken deploy.
