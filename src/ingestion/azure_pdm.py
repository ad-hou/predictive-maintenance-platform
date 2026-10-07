"""Adapter for the public "Azure Predictive Maintenance" dataset (Kaggle).

Column names below come from the dataset description as recalled when this file was
written. CHECK THEM against the files you download (and the licence) before use.
Expected files: PDM_telemetry.csv, PDM_errors.csv, PDM_maint.csv, PDM_failures.csv,
PDM_machines.csv (the exact capitalisation may differ).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find(directory: Path, stem: str) -> Path:
    for p in directory.iterdir():
        if p.stem.lower().replace("_", "") == stem.lower().replace("_", ""):
            return p
    raise FileNotFoundError(f"{stem}.csv not found in {directory}")


def load(directory: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(directory)
    ts = {"datetime": "timestamp", "machineID": "machine_id"}

    tel = pd.read_csv(_find(d, "PdM_telemetry"), parse_dates=["datetime"]).rename(columns={**ts, "volt": "voltage", "rotate": "rotation"})
    err = pd.read_csv(_find(d, "PdM_errors"), parse_dates=["datetime"]).rename(columns={**ts, "errorID": "error_code"})
    mnt = pd.read_csv(_find(d, "PdM_maint"), parse_dates=["datetime"]).rename(columns={**ts, "comp": "component"})
    fai = pd.read_csv(_find(d, "PdM_failures"), parse_dates=["datetime"]).rename(columns={**ts, "failure": "component"})
    mac = pd.read_csv(_find(d, "PdM_machines")).rename(columns={"machineID": "machine_id"})

    for df in (tel, err, mnt, fai, mac):
        df["machine_id"] = "M" + df["machine_id"].astype(str).str.zfill(3)
    return {"telemetry": tel, "errors": err, "maintenance": mnt, "failures": fai, "machines": mac}
