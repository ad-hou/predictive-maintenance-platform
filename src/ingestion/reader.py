"""Read the five source tables from CSV files or from the PostgreSQL staging schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.ingestion.synthetic import TABLES

TIME_TABLES = {"telemetry", "errors", "maintenance", "failures"}


def _clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name in TIME_TABLES:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df


def read_tables(source: str | None = None, directory: Path | None = None, as_of=None) -> dict[str, pd.DataFrame]:
    """Return the tables, optionally truncated at `as_of` (inclusive) for backfills."""
    source = source or config.DATA_SOURCE
    if source == "csv":
        d = Path(directory or config.DATA_DIR)
        tables = {t: pd.read_csv(d / f"{t}.csv") for t in TABLES}
    elif source == "postgres":
        from sqlalchemy import create_engine

        engine = create_engine(config.DATABASE_URL)
        tables = {t: pd.read_sql(f"select * from staging.stg_{t}", engine) for t in TABLES}
    else:
        raise ValueError(f"unknown data source: {source}")

    tables = {name: _clean(name, df) for name, df in tables.items()}
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        if cut == cut.normalize():  # a bare date means "end of that day"
            cut = cut + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        for name in TIME_TABLES:
            df = tables[name]
            tables[name] = df[df["timestamp"] <= cut].reset_index(drop=True)
    return tables
