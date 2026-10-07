"""Lightweight data-quality checks run before training (complements the DBT tests)."""

from __future__ import annotations

import pandas as pd

from src.features.build_features import SENSORS

PLAUSIBLE = {"voltage": (0, 400), "rotation": (0, 1500), "pressure": (0, 300), "vibration": (0, 200)}


def check_tables(tables: dict[str, pd.DataFrame], max_age_hours: float | None = None, now=None) -> list[str]:
    """Return a list of problems (empty list = OK)."""
    problems = []
    tel = tables["telemetry"]
    if tel["machine_id"].isna().any():
        problems.append("telemetry: null machine_id")
    if tel["timestamp"].isna().any():
        problems.append("telemetry: null timestamp")
    if tel.duplicated(["machine_id", "timestamp"]).any():
        problems.append("telemetry: duplicated (machine_id, timestamp)")
    for col, (lo, hi) in PLAUSIBLE.items():
        bad = ((tel[col] < lo) | (tel[col] > hi)).sum()
        if bad:
            problems.append(f"telemetry: {bad} implausible values in {col}")
        if tel[col].isna().mean() > 0.05:
            problems.append(f"telemetry: more than 5% missing in {col}")
    unknown = set(tables["failures"]["machine_id"]) - set(tables["machines"]["machine_id"])
    if unknown:
        problems.append(f"failures: unknown machines {sorted(unknown)[:3]}")
    if max_age_hours is not None:
        ref = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
        age = (ref - tel["timestamp"].max()).total_seconds() / 3600
        if age > max_age_hours:
            problems.append(f"telemetry: stale data ({age:.0f} h old)")
    return problems


def summary(tables: dict[str, pd.DataFrame]) -> dict:
    tel = tables["telemetry"]
    return {
        "rows": int(len(tel)),
        "machines": int(tel["machine_id"].nunique()),
        "missing_rate": float(tel[SENSORS].isna().mean().mean()),
        "duplicate_rate": float(tel.duplicated(["machine_id", "timestamp"]).mean()),
        "last_timestamp": str(tel["timestamp"].max()),
    }


REQUIRED_COLUMNS = {
    "telemetry": ["machine_id", "timestamp", *SENSORS],
    "errors": ["machine_id", "timestamp", "error_code"],
    "maintenance": ["machine_id", "timestamp", "component"],
    "failures": ["machine_id", "timestamp", "component"],
    "machines": ["machine_id", "model", "age"],
}


def check_structure(directory) -> list[str]:
    """Availability check on the raw files (exist, non-empty, expected columns). Value checks are
    deliberately left to the DBT tests, which are the gate of the pipeline."""
    from pathlib import Path

    problems = []
    for name, cols in REQUIRED_COLUMNS.items():
        path = Path(directory) / f"{name}.csv"
        if not path.exists():
            problems.append(f"{name}.csv is missing")
            continue
        head = pd.read_csv(path, nrows=5)
        missing = [c for c in cols if c not in head.columns]
        if missing:
            problems.append(f"{name}.csv: missing columns {missing}")
        if head.empty:
            problems.append(f"{name}.csv is empty")
    return problems
