# One GPU instance running k3s, in the account's default VPC. Nothing else: no
# EKS control plane, no NAT gateway, no database. `infra/k8s/gpu` is applied to
# it over an SSH tunnel, the measurement runs, and the whole module is destroyed.

# Amazon's own Deep Learning Base AMI carries the NVIDIA driver and the NVIDIA
# container toolkit, so cloud-init installs neither. `owners` is Amazon alone:
# anyone can publish an AMI under the same name.
data "aws_ami" "gpu" {
  owners      = ["amazon"]
  most_recent = true

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_vpc" "default" {
  default = true
}

resource "aws_key_pair" "admin" {
  key_name   = "${var.name}-admin"
  public_key = var.ssh_public_key
}

# SSH from the admin CIDRs is the only way in. vLLM's 8000 and the k3s API's
# 6443 are reached through an SSH tunnel, never opened to any address.
resource "aws_security_group" "gpu" {
  name        = "${var.name}-host"
  description = "SSH from admin_ips only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from the admin CIDRs"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.admin_ips
  }

  # The host pulls k3s, three images and 15GB of weights.
  egress {
    description = "all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "gpu" {
  ami                         = data.aws_ami.gpu.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.admin.key_name
  vpc_security_group_ids      = [aws_security_group.gpu.id]
  associate_public_ip_address = true

  # With the cloud-init timer, a shutdown is a termination: a stopped instance
  # would keep billing for its volume.
  instance_initiated_shutdown_behavior = "terminate"

  # Sized for the image layers (vLLM's alone is about 10GB), the 40Gi model
  # claim on k3s's local-path provisioner, and headroom.
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 100
    delete_on_termination = true
    encrypted             = true
  }

  metadata_options {
    http_tokens = "required"
  }

  user_data = templatefile("${path.module}/cloud-init.yaml", {
    auto_terminate_minutes = var.auto_terminate_minutes
  })

  tags = {
    Name = "${var.name}-host"
  }
}
