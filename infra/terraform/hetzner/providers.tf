# Both providers read their credential from the environment - HCLOUD_TOKEN for
# hcloud, CLOUDFLARE_API_TOKEN for cloudflare - so no credential is ever a
# terraform variable, a default, or a value in a committed file.
# In CI the plan runs inside the Terraform Cloud workspace, which holds them as
# workspace variables; locally they are exported into the shell.
provider "hcloud" {}

provider "cloudflare" {}
