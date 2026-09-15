terraform {
  required_version = ">= 1.9"

  # A second Terraform Cloud workspace beside the Hetzner one, so the
  # two modules never share a state and a destroy here can only plan against this
  # module. The workspace runs in LOCAL execution mode: the plan executes on the CI
  # runner, whose AWS credentials come from OIDC, and Terraform Cloud holds only
  # the state and its lock.
  cloud {
    organization = "example-org"

    workspaces {
      name = "ai-operations-console-aws"
    }
  }

  # A pessimistic constraint on the minor: the next major is where arguments are
  # renamed, and it must arrive as a deliberate edit, not on the next `init`.
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }
}
