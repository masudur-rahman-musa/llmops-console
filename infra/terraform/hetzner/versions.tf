terraform {
  required_version = ">= 1.9"

  # Decision E, recorded as a design note: the state lives in Terraform Cloud's free
  # tier. It is remote, locked and versioned, and no state file or plan output
  # ever lands in this repository.
  cloud {
    organization = "example-org"

    workspaces {
      name = "ai-operations-console-hetzner"
    }
  }

  # Exact major constraints on both providers. An unconstrained provider picks up
  # the next major on the next `init`, which is where renamed arguments arrive.
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.49"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.40"
    }
  }
}
