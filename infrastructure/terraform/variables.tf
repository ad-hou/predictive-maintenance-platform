variable "project" {
  description = "Prefix for resource names."
  type        = string
  default     = "pmp"
}

variable "region" {
  type    = string
  default = "eu-west-3"
}

variable "instance_type" {
  description = "EC2 type for the API. t3.small is enough for the demo."
  type        = string
  default     = "t3.small"
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to reach the API on port 8000. Empty = not reachable from the internet (use SSM port forwarding)."
  type        = list(string)
  default     = []
}

variable "create_rds" {
  description = "Create the RDS PostgreSQL instance (costs money while it exists). The API can also run on CSV data from S3."
  type        = bool
  default     = false
}

variable "alert_email" {
  description = "E-mail for the monthly budget alert. Empty = no budget."
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  type    = number
  default = 15
}

variable "image_tag" {
  description = "Initial API image tag (the deploy workflow updates it afterwards)."
  type        = string
  default     = "latest"
}
