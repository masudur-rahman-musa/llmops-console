data "aws_availability_zones" "available" {
  state = "available"
}

# Two zones, not every zone in the region: RDS and ElastiCache subnet groups need
# at least two, and each further zone is more subnets for a run that lives hours.
# One NAT gateway for both private subnets: a zone outage takes the nodes' egress
# with it, which an evidence run can afford and a per-zone NAT bill cannot.
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.7.2"

  name = var.name
  cidr = "10.40.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = ["10.40.0.0/20", "10.40.16.0/20"]
  public_subnets  = ["10.40.128.0/24", "10.40.129.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  # The tags the AWS load balancer integration discovers subnets by.
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}
