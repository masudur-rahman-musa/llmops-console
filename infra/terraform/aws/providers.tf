# Credentials come from the environment only: an assumed role in CI (GitHub OIDC,
# see README.md), a profile or `aws sso login` session locally. No access key is
# ever a variable, a default or a value in a committed file.
provider "aws" {
  region = var.region

  # Every taggable resource carries both tags, so a resource a forgotten destroy
  # left behind is one cost-explorer filter away.
  default_tags {
    tags = {
      project = "ai-operations-console"
      purpose = "evidence"
    }
  }
}
