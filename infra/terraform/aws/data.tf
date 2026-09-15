# Both stores sit in the private subnets behind a security group of their own,
# and each group admits exactly one source: the EKS node security group, on the
# store's one port. Ingress is written as standalone rule resources rather than
# inline blocks so a rule is one reviewable resource with one source.

resource "aws_security_group" "postgres" {
  name        = "${var.name}-postgres"
  description = "PostgreSQL, reachable from the EKS nodes only."
  vpc_id      = module.vpc.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "postgres_from_nodes" {
  security_group_id            = aws_security_group.postgres.id
  referenced_security_group_id = module.eks.node_security_group_id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
  description                  = "PostgreSQL from the EKS nodes."
}

resource "aws_security_group" "redis" {
  name        = "${var.name}-redis"
  description = "Redis, reachable from the EKS nodes only."
  vpc_id      = module.vpc.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "redis_from_nodes" {
  security_group_id            = aws_security_group.redis.id
  referenced_security_group_id = module.eks.node_security_group_id
  from_port                    = 6379
  to_port                      = 6379
  ip_protocol                  = "tcp"
  description                  = "Redis from the EKS nodes."
}

resource "aws_db_subnet_group" "postgres" {
  name       = "${var.name}-postgres"
  subnet_ids = module.vpc.private_subnets
}

# The master credential is generated and held by Secrets Manager, so it never
# enters state, a variable or an output. The last two flags exist for the destroy:
# no final snapshot left billing afterwards, no protection failing it halfway.
resource "aws_db_instance" "postgres" {
  identifier     = "${var.name}-postgres"
  engine         = "postgres"
  engine_version = "16"
  instance_class = "db.t4g.micro"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name                     = "opsconsole"
  username                    = "opsconsole"
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  publicly_accessible    = false
  multi_az               = false

  backup_retention_period = 0
  skip_final_snapshot     = true
  deletion_protection     = false
  apply_immediately       = true
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.name}-redis"
  subnet_ids = module.vpc.private_subnets
}

# A replication group of one node rather than a bare cluster: only the replication
# group carries transit encryption. There is no auth credential; encryption both
# ways and the node-only security group are the controls.
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "${var.name}-redis"
  description          = "Gateway cache, rate limits and routing state."
  engine               = "redis"
  engine_version       = "7.1"
  node_type            = "cache.t4g.micro"
  port                 = 6379

  num_cache_clusters         = 1
  automatic_failover_enabled = false

  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]

  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
  apply_immediately          = true
}
