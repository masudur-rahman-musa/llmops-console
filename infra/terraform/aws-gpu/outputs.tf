# What the operator needs to reach the host, and nothing else. Terraform Cloud
# stores every output and prints it in the run log, so no credential and no
# cluster access file is ever published here; the cluster's admin file stays on
# the host and is read over SSH. A test scans this file for the credential words.
output "region" {
  description = "The region the instance runs in."
  value       = var.region
}

output "instance_id" {
  description = "The instance id, for the AWS console and the cost capture."
  value       = aws_instance.gpu.id
}

output "public_ip" {
  description = "The address to SSH to, from admin_ips only."
  value       = aws_instance.gpu.public_ip
}
