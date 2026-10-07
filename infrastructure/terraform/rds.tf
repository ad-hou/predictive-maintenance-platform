resource "aws_security_group" "rds" {
  count       = var.create_rds ? 1 : 0
  name        = "${var.project}-rds"
  description = "PostgreSQL from the API only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}

resource "aws_db_instance" "postgres" {
  count                       = var.create_rds ? 1 : 0
  identifier                  = "${var.project}-postgres"
  engine                      = "postgres"
  engine_version              = "16"
  instance_class              = "db.t4g.micro"
  allocated_storage           = 20
  db_name                     = "pmp"
  username                    = "pmp"
  manage_master_user_password = true # stored in Secrets Manager, never in the Terraform state
  storage_encrypted           = true
  publicly_accessible         = false
  vpc_security_group_ids      = [aws_security_group.rds[0].id]
  backup_retention_period     = 1
  skip_final_snapshot         = true # demo project
  deletion_protection         = false
}
