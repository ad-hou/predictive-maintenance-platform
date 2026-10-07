# Architecture

## Data flow

1. **Ingestion** (`src/ingestion`): CSV files (synthetic generator or Azure PdM adapter) are loaded into PostgreSQL `raw` tables. Loads are idempotent (truncate + append).
2. **Quality gate** (`src/monitoring/data_quality.py`, DBT tests): schema, nulls, duplicates, ranges, freshness. A failure stops the DAG before any model or prediction is produced.
3. **DBT** (`dbt/predictive_maintenance`): `staging` views (typing, renaming), `intermediate` (sensor features), `marts` tables (`dim_machine`, `fact_machine_measurements`, `fact_machine_risk`). Custom generic tests: `unique_combination`, `accepted_range`, `recent_enough`.
4. **Features** (`src/features`): one feature definition for training (pandas) and one fast numpy implementation for serving. A test asserts they agree.
5. **Training and comparison** (`src/training`): baseline, logistic regression, random forest, HistGradientBoosting, LightGBM, IsolationForest, on chronological blocks, all with the same calibration and threshold procedure.
6. **Registry** (`src/registry`): MLflow model registry with aliases `champion`, `challenger`, `previous_champion`. Promotion rule and rollback live in `registry.py`; the drift cycle in `cycle.py`.
7. **Serving** (`api/`): FastAPI loads `models/champion.joblib` (optionally synced from S3). `/admin/reload` swaps the model without restart.
8. **Dashboard** (`dashboard/`): Streamlit. Runs on a snapshot in `dashboard/demo_data` so it can be hosted without any backend; an optional `API_URL` enables a live scoring form.

## Orchestration

`predictive_maintenance_pipeline` (10 tasks): start → check_source → load_postgres → dbt_staging → dbt_tests → dbt_marts → train_and_compare → generate_predictions → dbt_risk_mart → end.
`drift_retrain_pipeline`: drift check, retrain on the recent tail window, compare on the same unseen window, promote or refuse.
Tasks call `python -m src.cli ...` so the same code runs in Airflow, in a terminal and in CI.

## Deployment

- Local: `docker compose` (PostgreSQL, MLflow, Airflow, API, dashboard).
- AWS (Terraform): S3 (data and models), ECR (API image), EC2 (API container), IAM least privilege, CloudWatch logs and alarms, optional RDS, optional budget.
- CI/CD (GitHub Actions): `ci.yml` (lint, tests with PostgreSQL, DBT build, Docker builds, Terraform fmt/validate) and `deploy.yml` (OIDC, push to ECR, deploy through SSM, health check, rollback).
