resource "hcloud_ssh_key" "admin" {
  name       = "ai-operations-console-admin"
  public_key = var.ssh_public_key
}

# hcloud firewalls are an allow-list on inbound traffic, so everything not named
# below is denied. A test pins this rule set by count as well as
# by content: the cheap mistake here is a fourth rule opening 6443 or a database
# port "just for a minute", and the expensive one is port 22 inheriting the
# world-readable source list its neighbours carry.
resource "hcloud_firewall" "edge" {
  name = "ai-operations-console-edge"

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "22"
    # The admin CIDRs plus the CD runner's, concatenated rather than split into a
    # fourth rule: the rule COUNT is what pins this firewall's default-deny, so one
    # port-22 rule with two sources says the same thing without loosening the guard.
    source_ips = concat(var.admin_ips, var.ci_ips)
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

resource "hcloud_server" "node" {
  name         = "ai-operations-console"
  server_type  = var.server_type
  location     = var.location
  image        = "ubuntu-24.04"
  ssh_keys     = [hcloud_ssh_key.admin.id]
  firewall_ids = [hcloud_firewall.edge.id]

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }

  # The k3s install is a shell script, so it lives in a file that
  # reads as one. The only value rendered into it is the API hostname, which k3s
  # needs as a TLS SAN before anything can talk to the cluster by name.
  user_data = templatefile("${path.module}/cloud-init.yaml", {
    api_fqdn = "api.${var.zone}"
  })
}
