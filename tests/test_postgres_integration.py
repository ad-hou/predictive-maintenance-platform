"""Integration tests against a real PostgreSQL (skipped unless PMP_TEST_DATABASE_URL is set).

WARNING: these tests replace the raw/staging/marts/ml schemas of that database.
Use a throwaway database (the CI workflow starts one).
Example: PMP_TEST_DATABASE_URL=postgresql+psycopg2://postgres@localhost:5432/pmp pytest tests/test_postgres_integration.py
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

URL = os.getenv("PMP_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="PMP_TEST_DATABASE_URL not set")

DBT_DIR = Path(__file__).resolve().parents[1] / "dbt" / "predictive_maintenance"


def dbt(*args):
    u = make_url(URL)
    env = {
        **os.environ,
        "DBT_PROFILES_DIR": str(DBT_DIR),
        "POSTGRES_HOST": u.host or "localhost",
        "POSTGRES_PORT": str(u.port or 5432),
        "POSTGRES_USER": u.username or "postgres",
        "POSTGRES_PASSWORD": u.password or "",
        "POSTGRES_DB": u.database,
    }
    profiles = DBT_DIR / "profiles.yml"
    if not profiles.exists():
        profiles.write_text((DBT_DIR / "profiles.yml.example").read_text())
    return subprocess.run(["dbt", *args, "--project-dir", str(DBT_DIR)], capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def loaded(data_dir):
    from src.ingestion import load_postgres

    load_postgres.load(data_dir, URL)
    return data_dir


def test_dbt_build_passes_on_clean_data(loaded):
    r = dbt("build")
    assert r.returncode == 0, r.stdout[-2000:]


def test_null_machine_id_makes_dbt_fail_and_fixing_it_recovers(loaded, tables, data_dir):
    from src.ingestion import load_postgres

    engine = create_engine(URL)
    with engine.begin() as conn:
        conn.execute(text("update raw.raw_telemetry set machine_id = null where ctid in (select ctid from raw.raw_telemetry limit 1)"))
    broken = dbt("build", "--select", "staging")
    assert broken.returncode != 0 and "not_null_stg_telemetry_machine_id" in broken.stdout
    load_postgres.load(data_dir, URL)  # idempotent reload = fixed data
    assert dbt("build", "--select", "staging").returncode == 0


def test_postgres_reader_matches_csv_reader(loaded):
    from src import config
    from src.ingestion.reader import read_tables

    assert dbt("build").returncode == 0
    old = config.DATABASE_URL
    config.DATABASE_URL = URL
    try:
        pg = read_tables("postgres")
    finally:
        config.DATABASE_URL = old
    csv = read_tables("csv", loaded)
    for t in csv:
        assert len(pg[t]) == len(csv[t]), t
    assert np.allclose(pg["telemetry"]["vibration"].to_numpy(), csv["telemetry"]["vibration"].to_numpy())


def test_sql_features_match_python_features(loaded, dataset):
    assert dbt("build").returncode == 0
    sql = pd.read_sql("select * from intermediate.int_sensor_features", create_engine(URL))
    sql["timestamp"] = pd.to_datetime(sql["timestamp"])
    cols = [c for c in sql.columns if c not in ("machine_id", "timestamp", "voltage", "rotation", "pressure", "vibration")]
    merged = dataset.merge(sql, on=["machine_id", "timestamp"], suffixes=("_py", "_sql"))
    assert len(merged) == len(dataset)
    for c in cols:
        assert np.allclose(merged[f"{c}_py"], merged[f"{c}_sql"], atol=1e-6), c


def test_publish_predictions_is_idempotent(trained, tables):
    from src.ingestion import load_postgres
    from src.registry.cycle import score_latest

    scores = score_latest(trained["model"], tables, with_factors=False)
    scores["top_factor"] = ""
    for _ in range(2):
        load_postgres.publish_predictions(scores, "2025-03-15", URL)
    with create_engine(URL).connect() as conn:
        n = conn.execute(text("select count(*) from ml.machine_risk_predictions where run_date = '2025-03-15'")).scalar()
    assert n == len(scores)
    assert dbt("build", "--select", "fact_machine_risk").returncode == 0
