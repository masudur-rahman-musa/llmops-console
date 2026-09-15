# The guard against a forgotten destroy. Two t4g.medium nodes, the EKS control
# plane and a NAT gateway pass 10 USD within a day or two, so the mail at 80
# percent of ACTUAL spend arrives while the damage is still small. A forecast
# alert would be silent in the first hours of a month, which is when it matters.
resource "aws_budgets_budget" "evidence" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = "10"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}
