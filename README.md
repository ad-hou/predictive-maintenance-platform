# Predictive Maintenance Platform

End-to-end predictive maintenance platform combining Data Engineering, Machine Learning and MLOps on AWS.

![Demo: dashboard, drift detection, Airflow pipeline](docs/demo.gif)

> **Read this first.** The data is **synthetic** and the euro costs are **hypothetical**. Nothing here says anything about real machines. The project demonstrates the engineering around a model (pipelines, tests, monitoring, safe promotion, deployment), not a model that works on a real fleet.

## What it does

Every machine gets a daily risk score: *will it fail in the next 24 hours?* The maintenance team can inspect only a few machines per day (default 5), so the useful output is a **ranking**, plus a threshold chosen to minimise a business cost (missed failure vs. useless inspection).

```
sensor/error/maintenance CSV ──► PostgreSQL (raw) ──► DBT (staging → intermediate → marts, tested)
                                                          │
                                  Airflow orchestrates    ▼
                                  features ► model comparison ► MLflow registry (champion / challenger)
                                                          │
                       drift (PSI + KS) ► retrain on recent window ► promote only if cost drops ≥ 2 %
                                                          │
                       FastAPI (/predict, /machines, explanations) ◄── Streamlit dashboard
                                                          │
                              Docker Compose locally · Terraform + GitHub Actions on AWS
```

## Design choices that matter

- **Chronological evaluation only.** Train / calibration / threshold / test blocks follow time. No random split, no leakage across blocks.
- **Calibrated probabilities.** Isotonic calibration on a separate block, then a cost-optimal threshold on another block, identical procedure for every model.
- **Metrics that fit the problem.** PR-AUC against the base rate, precision@N inspections per day, and total cost in euros against "do nothing" and "inspect everything".
- **Safe promotion.** A challenger trained after drift is compared with the champion on the *same unseen window*. It replaces the champion only if the cost drops by at least 2 %. Every decision is written to `logs/promotion_decisions.jsonl`; `rollback` restores the previous champion.
- **No train/serve skew.** The API computes features with a fast numpy path; a test checks it matches the pandas training features.
- **Idempotent pipelines.** Loads truncate and append; predictions are replaced per `run_date`. A failing DBT test stops the DAG before any new prediction or model is produced.

## Demo scenario (synthetic data)

`python -m src.cli demo` runs: train v1 → inject drift on a share of machines → detect it (PSI on voltage and vibration far above 0.2, pressure below the threshold and not flagged) → retrain → compare → promote. A control run without drift triggers no action. Exact figures are in `dashboard/demo_data/` and the Model / Monitoring pages of the dashboard.

## Quick start (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.12
pip install -r requirements-dev.txt
make demo         # data → train → drift → retrain → promote → dashboard snapshot
make test
make dashboard    # http://localhost:8501 (reads the snapshot, no backend needed)
make api          # http://localhost:8000/docs
```

## Full stack with Docker

```bash
cp .env.example .env        # change the password
docker compose up --build
```

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

Trigger the `predictive_maintenance_pipeline` DAG in Airflow, then `drift_retrain_pipeline`.

## AWS

`infrastructure/terraform/` creates S3, ECR, an EC2 instance for the API, least-privilege IAM, CloudWatch alarms and an optional budget and RDS. See its README. **Apply it yourself with your own credentials; the repository never contains keys.** GitHub Actions deploys through OIDC (`deploy.yml`) with a health check and automatic rollback on the instance.

## Layout

```
src/            ingestion, features, training, monitoring, registry, CLI
api/            FastAPI service
dbt/            DBT project (staging, intermediate, marts, custom tests)
airflow/dags/   orchestration
dashboard/      Streamlit app + demo snapshot
infrastructure/ Terraform
load_tests/     Locust
tests/          unit, integration, train/serve skew
docs/           architecture, data dictionary, decisions, model card, cost assumptions
```

## Known limits

- Synthetic data; results (low PR-AUC on a deliberately hard task) say nothing about real equipment. An adapter exists for the Azure PdM Kaggle dataset, but its column names still need to be checked against the real files.
- Costs are assumptions, see `docs/cost_assumptions.md`.
- The champion is selected on calibration-block PR-AUC; in some runs another model has a lower test cost. See `docs/decisions.md`.
- Local load-test figures (`docs/results.md`) come from a small 2-core machine and do not represent AWS performance.
- Terraform and the GitHub Actions workflows were written but not run by the author's tooling; CI will be their first real execution.

## License

MIT
