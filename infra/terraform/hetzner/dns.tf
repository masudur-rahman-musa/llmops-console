data "cloudflare_zone" "this" {
  name = var.zone
}

# DNS-only, never proxied. cert-manager solves
# the HTTP-01 challenge on the origin itself, so Cloudflare must hand the visitor
# the node's address rather than terminate TLS in front of it.
resource "cloudflare_record" "hosts" {
  for_each = toset(var.subdomains)

  zone_id = data.cloudflare_zone.this.id
  name    = each.value
  type    = "A"
  value   = hcloud_server.node.ipv4_address
  proxied = false
  ttl     = 300
}
