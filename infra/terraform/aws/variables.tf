# No default: an unset value must fail at plan time. The module's own default for
# the public API endpoint is the whole internet, which is what this variable
# replaces, so a default here would be that same opening under another name.
variable "admin_ips" {
  description = "CIDRs allowed to reach the EKS public API endpoint. Required, never defaulted."
  type        = list(string)

  validation {
    condition     = length(var.admin_ips) > 0
    error_message = "admin_ips must name at least one CIDR; an empty list leaves the API with no source."
  }

  validation {
    condition     = alltrue([for entry in var.admin_ips : can(cidrhost(entry, 0))])
    error_message = "every admin_ips entry must be a CIDR such as 203.0.113.7/32; a bare address is refused."
  }
}

variable "region" {
  description = "AWS region. EU only, because the stack holds EU tenant data."
  type        = string
  default     = "eu-central-1"

  validation {
    condition     = can(regex("^eu-[a-z]+-[0-9]$", var.region))
    error_message = "region must be an EU region such as eu-central-1; anything else moves the data out of the EU."
  }
}

# Prefixes every named resource. The upper bound keeps the longest derived name,
# the backups bucket with its account-id suffix, inside S3's 63 characters.
variable "name" {
  description = "Name prefix for every resource: lowercase, starting with a letter, at most 20 characters."
  type        = string
  default     = "aioc-evidence"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,19}$", var.name))
    error_message = "name must be 1 to 20 lowercase letters, digits or hyphens, starting with a letter."
  }
}

# Pinned to a release in EKS STANDARD support. A version in extended support bills
# about 0.60 USD an hour extra per cluster, which alone exceeds the evidence
# budget, so re-check the AWS calendar before an apply: 1.36 is standard until
# 2027-08-02.
variable "kubernetes_version" {
  description = "EKS Kubernetes minor version, as MAJOR.MINOR. Must be in standard support when applied."
  type        = string
  default     = "1.36"

  validation {
    condition     = can(regex("^1\\.[0-9]{2}$", var.kubernetes_version))
    error_message = "kubernetes_version must be MAJOR.MINOR such as 1.36, with no patch and no v prefix."
  }
}

variable "budget_email" {
  description = "Address the budget alert mails when actual spend passes 80 percent of the limit."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_email))
    error_message = "budget_email must be one email address such as you@example.com."
  }
}
