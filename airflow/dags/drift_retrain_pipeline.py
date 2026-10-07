"""Weekly drift check. If sensor distributions moved, retrain a challenger, compare it with the
champion on the same unseen window, and promote it only if the total cost drops enough.
The decision (and why) is written to logs/promotion_decisions.jsonl and to MLflow."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from airflow import DAG

PROJECT = os.getenv("PMP_HOME", "/opt/pmp")
PY = os.getenv("PMP_PYTHON", "python")
CLI = f"cd {PROJECT} && {PY} -m src.cli"

with DAG(
    dag_id="drift_retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="0 7 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "drift"],
) as dag:
    start = EmptyOperator(task_id="start")
    drift_cycle = BashOperator(
        task_id="drift_check_retrain_compare_promote",
        # REFERENCE_END = last day of the data the champion was trained on (set as an Airflow Variable or env var).
        bash_command=f"{CLI} drift-cycle --as-of {{{{ ds }}}} --reference-end ${{REFERENCE_END}}",
        env={"DATA_SOURCE": "postgres", "REFERENCE_END": os.getenv("REFERENCE_END", "2025-03-19"), **os.environ},
    )
    end = EmptyOperator(task_id="end")
    start >> drift_cycle >> end
