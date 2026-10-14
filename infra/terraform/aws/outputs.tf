# What a human and the evidence deploy need, and nothing else. No credential and
# no cluster admin file is published: Terraform Cloud stores every output in state
# and prints it in the run log, so an output is the widest place a credential can land
#. The database credential is read from the AWS-managed store the
# RDS instance names, never from here. The scan in infra/tests/infra/test_aws_module.py
# refuses the credential words anywhere in this file, comments included.
output "region" {
  description = "The region every resource was created in."
  value       = var.region
}

output "cluster_name" {
  description = "The EKS cluster name, for the aws eks CLI."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "The EKS API endpoint, reachable from admin_ips only."
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "PostgreSQL host:port, private to the VPC."
  value       = aws_db_instance.postgres.endpoint
}

output "redis_endpoint" {
  description = "Redis primary endpoint, private to the VPC, TLS only."
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "backup_bucket" {
  description = "The S3 bucket the backups CronJobs write to."
  value       = aws_s3_bucket.backups.id
}
