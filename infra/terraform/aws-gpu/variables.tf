# No default: an unset value must fail at plan time rather than open SSH to
# whatever a default would have said.
variable "admin_ips" {
  description = "CIDRs allowed to reach the instance on SSH. Required, never defaulted."
  type        = list(string)

  validation {
    condition     = length(var.admin_ips) > 0
    error_message = "admin_ips must name at least one CIDR; an empty list leaves the host with no way in."
  }

  validation {
    condition     = alltrue([for cidr in var.admin_ips : can(cidrhost(cidr, 0))])
    error_message = "every admin_ips entry must be a CIDR such as 203.0.113.7/32; a bare address is refused."
  }
}

variable "region" {
  description = "AWS region. EU only, because the measurement runs the Helpdesk golden set."
  type        = string
  default     = "eu-central-1"

  validation {
    condition     = can(regex("^eu-[a-z]+-[0-9]$", var.region))
    error_message = "region must be an EU region such as eu-central-1."
  }
}

variable "name" {
  description = "Name prefix for every named resource: lowercase, starting with a letter, at most 20 characters."
  type        = string
  default     = "aioc-gpu"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,19}$", var.name))
    error_message = "name must be 1 to 20 lowercase letters, digits or hyphens, starting with a letter."
  }
}

# The two single-GPU sizes with 24GB of GPU memory: an A10G (g5) or an L4 (g6).
# Anything larger multiplies the hourly bill for a GPU the model does not need.
variable "instance_type" {
  description = "EC2 instance type: g5.xlarge or g6.xlarge."
  type        = string
  default     = "g5.xlarge"

  validation {
    condition     = can(regex("^g[56]\\.xlarge$", var.instance_type))
    error_message = "instance_type must be g5.xlarge or g6.xlarge."
  }
}

variable "ssh_public_key" {
  description = "The admin's ed25519 public key, one line as in id_ed25519.pub."
  type        = string

  validation {
    condition     = startswith(var.ssh_public_key, "ssh-ed25519 ")
    error_message = "ssh_public_key must be an ed25519 public key starting with \"ssh-ed25519 \"."
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

# The forgotten-destroy guard that does not lag. cloud-init schedules a shutdown
# this many minutes after boot, and the instance terminates on shutdown. The
# floor leaves time to fetch the weights and run the benchmark; the ceiling
# keeps a forgotten host inside the budget.
variable "auto_terminate_minutes" {
  description = "Minutes after boot at which the instance shuts down and terminates."
  type        = number
  default     = 240

  validation {
    condition     = var.auto_terminate_minutes >= 30 && var.auto_terminate_minutes <= 480
    error_message = "auto_terminate_minutes must be between 30 and 480 inclusive."
  }
}
