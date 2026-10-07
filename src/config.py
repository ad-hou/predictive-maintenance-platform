"""Central configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


DATA_DIR = _path("DATA_DIR", "data/raw")
MODELS_DIR = _path("MODELS_DIR", "models")
LOGS_DIR = _path("LOGS_DIR", "logs")
REPORTS_DIR = _path("REPORTS_DIR", "reports")
DEMO_DIR = _path("DEMO_DIR", "dashboard/demo_data")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MODEL_NAME = os.getenv("MODEL_NAME", "predictive-maintenance")

# Data source: "csv" (files in DATA_DIR) or "postgres" (staging tables built by DBT).
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Business setting: horizon of the prediction target, in hours.
HORIZON_HOURS = int(os.getenv("HORIZON_HOURS", "24"))
# Capacity: how many machines the maintenance team can inspect per day.
INSPECTIONS_PER_DAY = int(os.getenv("INSPECTIONS_PER_DAY", "5"))


@dataclass(frozen=True)
class Costs:
    """Hypothetical costs in euros. NOT real data: see docs/cost_assumptions.md."""

    missed_failure: float = float(os.getenv("COST_MISSED_FAILURE", "10000"))
    useless_inspection: float = float(os.getenv("COST_USELESS_INSPECTION", "500"))
    useful_inspection: float = float(os.getenv("COST_USEFUL_INSPECTION", "500"))


COSTS = Costs()

# Drift: PSI above this value is flagged (common rule of thumb, see docs/decisions.md).
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
# A challenger replaces the champion only if its total cost is lower by this fraction.
MIN_COST_GAIN = float(os.getenv("MIN_COST_GAIN", "0.02"))
