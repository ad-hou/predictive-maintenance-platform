# Terraform (AWS)

Creates: S3 bucket (data, models), ECR repository, EC2 instance running the API container, IAM role with least privilege (bucket prefix, ECR pull, SSM parameters), CloudWatch log group and alarms, optional RDS PostgreSQL, optional monthly budget with email alert.

## Use

```bash
cp terraform.tfvars.example terraform.tfvars   # edit allowed_cidrs, alert_email...
terraform init
terraform plan
terraform apply
```

Run these yourself with your own AWS credentials. Nothing in this repository stores keys.

## Cost

A `t3.small` plus S3 and CloudWatch is a few dollars per month if left on. RDS is off by default (`create_rds = false`) and costs more. Set `monthly_budget_usd` and `alert_email` to get a budget alert, and run `terraform destroy` when you are done.

## Access

`allowed_cidrs = []` keeps the API closed to the internet; reach it through SSM port forwarding. Put only your own IP (`x.x.x.x/32`) if you open it.

## GitHub Actions deploy

`deploy.yml` assumes an IAM role trusted by GitHub OIDC (not created here: it depends on your GitHub org and repository). Set the repository variables `AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`, `ECR_REPOSITORY`, `EC2_INSTANCE_ID`.

## Status

The configuration has been formatted but **not validated or applied** by the author's tooling. Run `terraform validate` first; CI does it on every push.
