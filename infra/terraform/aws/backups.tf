data "aws_caller_identity" "current" {}

# The backups CronJobs run as this Kubernetes service account.
locals {
  backups_service_account = "system:serviceaccount:platform:backups"
}

# The account id makes the globally unique bucket name deterministic. Objects
# never block destroy: an evidence run's backups are disposable by design.
resource "aws_s3_bucket" "backups" {
  bucket        = "${var.name}-backups-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

# IRSA: only the backups service account, through the cluster's own OIDC
# provider, can assume this role. No node-wide role gets bucket access.
resource "aws_iam_role" "backups" {
  name = "${var.name}-backups"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Action    = "sts:AssumeRoleWithWebIdentity"
        Principal = { Federated = module.eks.oidc_provider_arn }
        Condition = {
          StringEquals = {
            "${module.eks.oidc_provider}:sub" = local.backups_service_account
            "${module.eks.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      },
    ]
  })
}

# The whole reach of the role: list this bucket, read and write its objects.
resource "aws_iam_role_policy" "backups" {
  name = "${var.name}-backups"
  role = aws_iam_role.backups.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.backups.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${aws_s3_bucket.backups.arn}/*"]
      },
    ]
  })
}
