# Decision F: no default. An unset value must fail at plan time, because the
# alternative - a default of 0.0.0.0/0 - is a firewall that quietly opens SSH to
# the internet the day someone forgets the variable.
variable "admin_ips" {
  description = "CIDRs allowed to reach SSH on the node. Required, never defaulted."
  type        = list(string)

  validation {
    condition     = length(var.admin_ips) > 0
    error_message = "admin_ips must name at least one CIDR; an empty list leaves port 22 with no source."
  }

  validation {
    condition     = alltrue([for entry in var.admin_ips : can(cidrhost(entry, 0))])
    error_message = "every admin_ips entry must be a CIDR such as 203.0.113.7/32; a bare address is refused."
  }
}

variable "zone" {
  description = "The Cloudflare zone the records live in, as a bare apex domain."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$", var.zone))
    error_message = "zone must be a bare apex domain such as example.com, with no scheme and no trailing dot."
  }
}

variable "subdomains" {
  description = "The hostnames that get an A record. Each is one label under the zone."
  type        = list(string)
  default     = ["api", "console", "helpdesk", "grafana"]

  validation {
    condition     = alltrue([for label in var.subdomains : can(regex("^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$", label))])
    error_message = "every subdomain must be one DNS label: lowercase letters, digits and hyphens, no dots."
  }

  validation {
    condition     = length(setsubtract(["api", "console", "helpdesk", "grafana"], var.subdomains)) == 0
    error_message = "subdomains must include api, console, helpdesk and grafana; outputs.tf publishes an FQDN for each."
  }
}

variable "server_type" {
  description = "Hetzner server type. CAX21 is the arm64 box the cluster is sized against."
  type        = string
  default     = "cax21"

  validation {
    condition     = can(regex("^cax[0-9]+$", var.server_type))
    error_message = "server_type must be a CAX type; every image the cluster runs is built for arm64."
  }
}

variable "location" {
  description = "Hetzner location. EU only, because the node holds EU tenant data."
  type        = string
  default     = "nbg1"

  validation {
    condition     = contains(["nbg1", "fsn1", "hel1"], var.location)
    error_message = "location must be one of nbg1, fsn1, hel1; anything else moves the node out of the EU."
  }
}

variable "ssh_public_key" {
  description = "The public half of the admin SSH key, as one OpenSSH line."
  type        = string

  validation {
    condition     = can(regex("^(ssh-ed25519|ssh-rsa|ecdsa-sha2-) ", var.ssh_public_key))
    error_message = "ssh_public_key must be an OpenSSH public key line, not a path and not a private key."
  }
}
