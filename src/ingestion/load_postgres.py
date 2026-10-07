"""PostgreSQL I/O: load raw CSV tables (schema `raw`) and publish predictions (schema `ml`).

Both operations are idempotent: raw tables are truncated and reloaded (never dropped, DBT views
depend on them), predictions are replaced per run_date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from src import config
from src.ingestion.synthetic import TABLES

PREDICTIONS_DDL = """
create table if not exists ml.machine_risk_predictions (
    run_date date not null,
    machine_id text not null,
    "timestamp" timestamp not null,
    failure_probability double precision not null,
    risk_level text not null,
    model_version text not null,
    top_factor text,
    primary key (run_date, machine_id)
)
"""


def _engine(url: str | None = None):
    return create_engine(url or config.DATABASE_URL)


def ensure_ml_schema(url: str | None = None) -> None:
    with _engine(url).begin() as conn:
        conn.execute(text("create schema if not exists ml"))
        conn.execute(text(PREDICTIONS_DDL))


def load(directory: Path | None = None, url: str | None = None) -> dict[str, int]:
    engine = _engine(url)
    d = Path(directory or config.DATA_DIR)
    counts = {}
    with engine.begin() as conn:
        conn.execute(text("create schema if not exists raw"))
        for t in TABLES:
            df = pd.read_csv(d / f"{t}.csv")
            name = f"raw_{t}"
            # Never DROP: DBT views depend on these tables, so a second run would fail.
            # Truncate + append keeps the objects and makes the load idempotent.
            if inspect(conn).has_table(name, schema="raw"):
                conn.execute(text(f"truncate table raw.{name}"))
            df.to_sql(name, conn, schema="raw", if_exists="append", index=False, chunksize=20000)
            counts[t] = len(df)
    ensure_ml_schema(url)
    return counts


def publish_predictions(df: pd.DataFrame, run_date, url: str | None = None) -> int:
    """Replace the predictions of `run_date` (re-running a day never duplicates rows)."""
    ensure_ml_schema(url)
    out = df[["machine_id", "timestamp", "failure_probability", "risk_level", "model_version", "top_factor"]].copy()
    out.insert(0, "run_date", pd.Timestamp(run_date).date())
    with _engine(url).begin() as conn:
        conn.execute(text("delete from ml.machine_risk_predictions where run_date = :d"), {"d": pd.Timestamp(run_date).date()})
        out.to_sql("machine_risk_predictions", conn, schema="ml", if_exists="append", index=False)
    return len(out)
