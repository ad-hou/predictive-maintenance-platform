"""Data drift: compare production sensor distributions with the training reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.features.build_features import SENSORS


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with bins taken from the reference quantiles.

    Rule of thumb (industry convention, not a law): < 0.1 stable, 0.1-0.2 moderate, > 0.2 major.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=None, threshold: float | None = None) -> dict:
    """Per-variable PSI and Kolmogorov-Smirnov test; `drift_detected` if any PSI > threshold."""
    threshold = config.PSI_THRESHOLD if threshold is None else threshold
    columns = list(columns or SENSORS)
    rows = []
    for col in columns:
        ks = stats.ks_2samp(reference[col].to_numpy(), current[col].to_numpy())
        value = psi(reference[col].to_numpy(), current[col].to_numpy())
        rows.append(
            {
                "variable": col,
                "psi": round(value, 4),
                "ks_statistic": round(float(ks.statistic), 4),
                "ks_pvalue": float(ks.pvalue),
                "drift": bool(value > threshold),
            }
        )
    table = pd.DataFrame(rows)
    return {
        "threshold": threshold,
        "drift_detected": bool(table["drift"].any()),
        "drifted_variables": table.loc[table["drift"], "variable"].tolist(),
        "table": table,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }


def psi_timeline(telemetry: pd.DataFrame, reference_end, last_days: int = 40, columns=None) -> pd.DataFrame:
    """Daily PSI of each sensor against the reference period (used by the dashboard)."""
    columns = list(columns or SENSORS)
    ref = telemetry[telemetry["timestamp"] <= pd.Timestamp(reference_end)]
    end = telemetry["timestamp"].max().normalize()
    rows = []
    for d in pd.date_range(end - pd.Timedelta(days=last_days - 1), end, freq="D"):
        day = telemetry[(telemetry["timestamp"] >= d) & (telemetry["timestamp"] < d + pd.Timedelta(days=1))]
        if len(day) < 50:
            continue
        for col in columns:
            rows.append({"day": d, "variable": col, "psi": psi(ref[col].to_numpy(), day[col].to_numpy())})
    return pd.DataFrame(rows)
