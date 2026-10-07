resource "aws_security_group" "api" {
  name        = "${var.project}-api"
  description = "API access"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = length(var.allowed_cidrs) > 0 ? [1] : []

    content {
      description = "API"
      from_port   = 8000
      to_port     = 8000
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "api" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.api.name

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size = 20
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region      = var.region
    project     = var.project
    ecr_url     = aws_ecr_repository.api.repository_url
    log_group   = local.log_group
    bucket      = aws_s3_bucket.data.bucket
    model_uri   = local.model_uri
    image_tag   = var.image_tag
    data_source = var.create_rds ? "postgres" : "csv"
    db_host     = var.create_rds ? aws_db_instance.postgres[0].address : ""
  })
  user_data_replace_on_change = true

  tags = { Name = "${var.project}-api" }
}
