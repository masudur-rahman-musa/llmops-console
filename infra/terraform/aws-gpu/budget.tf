# The second forgotten-destroy guard. A g5.xlarge bills about 1.2 USD an hour in
# eu-central-1, so the mail at 80 percent of ACTUAL spend arrives after roughly
# ten hours. A budget alert lags spend by hours, which is why the shutdown timer
# in cloud-init is the first guard and this is the backstop.
resource "aws_budgets_budget" "gpu" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = "15"
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
