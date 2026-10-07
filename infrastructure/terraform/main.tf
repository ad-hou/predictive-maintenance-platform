data "aws_caller_identity" "current" {}
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  bucket_name = "${var.project}-${data.aws_caller_identity.current.account_id}-${var.region}"
  log_group   = "/${var.project}/api"
  model_uri   = "s3://${local.bucket_name}/models/champion.joblib"
}
