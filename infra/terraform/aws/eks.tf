# The community module rather than hand-written EKS, IAM and node group.
# Input names are the v21 ones: that major dropped the `cluster_` prefix from them.
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.25.0"

  name               = var.name
  kubernetes_version = var.kubernetes_version

  # The public endpoint is open to the admin CIDRs only; the module's default for
  # the source list is the whole internet. Nodes reach the API privately.
  endpoint_public_access       = true
  endpoint_public_access_cidrs = var.admin_ips
  endpoint_private_access      = true

  # The identity that applies the module can administer the cluster it created.
  enable_cluster_creator_admin_permissions = true

  # The OIDC provider the backups role in backups.tf trusts.
  enable_irsa = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  addons = {
    coredns    = {}
    kube-proxy = {}
    vpc-cni = {
      before_compute = true
    }
  }

  # One group of two arm64 nodes, fixed size: every image the stack runs is built
  # for arm64, and a max above 2 would let the autoscaler outrun the budget.
  eks_managed_node_groups = {
    default = {
      ami_type       = "AL2023_ARM_64_STANDARD"
      instance_types = ["t4g.medium"]
      min_size       = 2
      max_size       = 2
      desired_size   = 2
    }
  }
}
