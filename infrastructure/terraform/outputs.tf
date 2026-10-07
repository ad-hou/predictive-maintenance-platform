output "bucket" {
  value = aws_s3_bucket.data.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "instance_id" {
  description = "Use it for SSM port forwarding and by the deploy workflow."
  value       = aws_instance.api.id
}

output "api_private_ip" {
  value = aws_instance.api.private_ip
}

output "rds_endpoint" {
  value = var.create_rds ? aws_db_instance.postgres[0].address : null
}
