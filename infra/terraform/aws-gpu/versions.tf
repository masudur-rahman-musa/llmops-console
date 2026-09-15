terraform {
  required_version = ">= 1.9"

  # A third Terraform Cloud workspace, beside the Hetzner and AWS evidence ones,
  # so a destroy here can only ever plan against this one instance. Local
  # execution mode, as for `../aws`: the plan runs on the CI runner with OIDC
  # credentials, and Terraform Cloud holds only the state and its lock.
  cloud {
    organization = "example-org"

    workspaces {
      name = "ai-operations-console-aws-gpu"
    }
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }
}
