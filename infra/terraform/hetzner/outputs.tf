# What a human and row 20's deploy need, and nothing else. No credential and no
# cluster admin file is published here: Terraform Cloud stores every output in
# state and prints it in the run log, so an output is the widest place a secret
# can land.
output "server_ipv4" {
  description = "The node's public IPv4. Every A record points at it."
  value       = hcloud_server.node.ipv4_address
}

output "server_id" {
  description = "The hcloud server id, for snapshots and support requests."
  value       = hcloud_server.node.id
}

output "api_fqdn" {
  description = "The gateway hostname."
  value       = cloudflare_record.hosts["api"].hostname
}

output "console_fqdn" {
  description = "The operations console hostname."
  value       = cloudflare_record.hosts["console"].hostname
}

output "helpdesk_fqdn" {
  description = "The Helpdesk workload hostname."
  value       = cloudflare_record.hosts["helpdesk"].hostname
}

output "grafana_fqdn" {
  description = "The Grafana hostname."
  value       = cloudflare_record.hosts["grafana"].hostname
}
