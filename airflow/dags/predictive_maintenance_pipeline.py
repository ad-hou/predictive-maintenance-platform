"""Daily pipeline: ingest -> DBT (tests gate everything) -> train challenger -> compare -> predict.

If any DBT test fails the DAG stops: no new model and no new predictions are produced.
Every step is idempotent, so a task can be retried and past days can be back-filled
(`airflow dags backfill`): `{{ ds }}` is passed as the "as of" date.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from airflow import DAG

PROJECT = os.getenv("PMP_HOME", "/opt/pmp")
PY = os.getenv("PMP_PYTHON", "python")  # in Docker: a dedicated venv, isolated from Airflow's own dependencies
DBT_BIN = os.getenv("PMP_DBT", "dbt")
DBT = f"{DBT_BIN} {{cmd}} --project-dir {PROJECT}/dbt/predictive_maintenance --profiles-dir {PROJECT}/dbt/predictive_maintenance"
CLI = f"cd {PROJECT} && {PY} -m src.cli"

default_args = {
    "owner": "data",
    "retries": int(os.getenv("PMP_RETRIES", "2")),
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="predictive_maintenance_pipeline",
    description="Ingestion, DBT with tests, model comparison and daily predictions",
    start_date=datetime(2025, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["data", "ml", "maintenance"],
) as dag:
    start = EmptyOperator(task_id="start")

    check_source = BashOperator(
        task_id="check_source",
        bash_command=f"{CLI} quality --structure-only",  # files present with expected columns; values are tested by DBT
    )
    load_postgres = BashOperator(task_id="load_postgres", bash_command=f"{CLI} load-postgres")
    dbt_staging = BashOperator(task_id="dbt_staging", bash_command=DBT.format(cmd="run --select staging"))
    # The tests are the gate: if one fails, nothing downstream runs.
    dbt_tests = BashOperator(task_id="dbt_tests", bash_command=DBT.format(cmd="test --select staging"))
    # Marts are built (and tested) only once the staging tests are green.
    dbt_marts = BashOperator(
        task_id="dbt_marts", bash_command=DBT.format(cmd="build --select int_sensor_features dim_machine fact_machine_measurements")
    )
    train_and_compare = BashOperator(
        task_id="train_and_compare",
        bash_command=f"{CLI} train --as-of {{{{ ds }}}} --trigger schedule",  # registers a challenger, promotes only if cheaper
        env={"DATA_SOURCE": "postgres", **os.environ},
        execution_timeout=timedelta(hours=1),
    )
    generate_predictions = BashOperator(
        task_id="generate_predictions",
        bash_command=f"{CLI} predict --as-of {{{{ ds }}}} --run-date {{{{ ds }}}}",  # replaces that day's rows: idempotent
        env={"DATA_SOURCE": "postgres", **os.environ},
    )
    dbt_risk_mart = BashOperator(task_id="dbt_risk_mart", bash_command=DBT.format(cmd="build --select fact_machine_risk"))
    end = EmptyOperator(task_id="end")

    (
        start
        >> check_source
        >> load_postgres
        >> dbt_staging
        >> dbt_tests
        >> dbt_marts
        >> train_and_compare
        >> generate_predictions
        >> dbt_risk_mart
        >> end
    )
