# Credentials come from the environment only: an assumed role in CI, a profile
# or `aws sso login` session locally. No access key is ever a variable, a
# default or a value in a committed file.
provider "aws" {
  region = var.region

  # A purpose distinct from the evidence stack's, so the GPU hours are one
  # cost-explorer filter away and never filed under the other run.
  default_tags {
    tags = {
      project = "ai-operations-console"
      purpose = "gpu-measurement"
    }
  }
}
