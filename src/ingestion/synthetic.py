"""Synthetic multi-table dataset used for development, tests and the public demo.

It mimics the shape of a real predictive-maintenance dataset (telemetry, errors,
maintenance, failures, machines) so that the real data can replace it through
`azure_pdm.py` without touching the rest of the code.

IMPORTANT: results obtained on this data say nothing about real-world performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TABLES = ("telemetry", "errors", "maintenance", "failures", "machines")

BASE = {"voltage": 170.0, "rotation": 446.0, "pressure": 100.0, "vibration": 40.0}
NOISE = {"voltage": 8.0, "rotation": 50.0, "pressure": 10.0, "vibration": 5.0}
# How much each sensor moves at the peak of a degradation episode.
RAMP = {"voltage": 6.0, "rotation": -45.0, "pressure": 9.0, "vibration": 11.0}
RAMP_HOURS = 36


@dataclass
class SyntheticConfig:
    n_machines: int = 60
    days: int = 120
    seed: int = 7
    start: str = "2025-01-01"
    failures_per_machine: float = 3.0  # on average over the period
    sudden_share: float = 0.25  # failures with no visible degradation
    near_miss_share: float = 0.30  # degradation episodes that do not end in failure


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.6) -> np.ndarray:
    eps = rng.normal(size=n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + np.sqrt(1 - phi**2) * eps[i]
    return out


def generate(cfg: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n_hours = cfg.days * 24
    grid = pd.date_range(cfg.start, periods=n_hours, freq="h")

    machines = pd.DataFrame(
        {
            "machine_id": [f"M{i:03d}" for i in range(1, cfg.n_machines + 1)],
            "model": rng.choice(["model1", "model2", "model3", "model4"], cfg.n_machines),
            "age": rng.integers(0, 20, cfg.n_machines),
        }
    )

    telemetry_parts, error_rows, maint_rows, fail_rows = [], [], [], []
    for _, m in machines.iterrows():
        health = np.zeros(n_hours)  # 0 = healthy, 1 = peak of a degradation episode
        rate = cfg.failures_per_machine * (1 + m["age"] / 40) / n_hours
        n_fail = rng.poisson(rate * n_hours)
        fail_idx = np.sort(rng.integers(24, n_hours - 2, n_fail))
        keep = np.ones(len(fail_idx), dtype=bool)
        keep[1:] = np.diff(fail_idx) > RAMP_HOURS + 6
        fail_idx = fail_idx[keep]

        for f in fail_idx:
            if rng.random() > cfg.sudden_share:
                lo = max(0, f - RAMP_HOURS)
                ramp = ((np.arange(lo, f + 1) - lo) / RAMP_HOURS) ** 1.5
                health[lo : f + 1] = np.maximum(health[lo : f + 1], ramp)
            fail_rows.append({"machine_id": m["machine_id"], "timestamp": grid[f], "component": f"comp{rng.integers(1, 5)}"})
            if f + 2 < n_hours:
                maint_rows.append({"machine_id": m["machine_id"], "timestamp": grid[f + 2], "component": "repair"})
            health[f + 1 : f + 3] = 0.0

        # Near misses: a degradation episode that ends with a maintenance, not a failure.
        for _ in range(rng.poisson(len(fail_idx) * cfg.near_miss_share / max(1 - cfg.near_miss_share, 0.01))):
            end = int(rng.integers(RAMP_HOURS, n_hours - 4))
            if any(abs(end - f) < RAMP_HOURS + 6 for f in fail_idx):
                continue
            lo = end - RAMP_HOURS
            ramp = 0.8 * ((np.arange(lo, end + 1) - lo) / RAMP_HOURS) ** 1.5
            health[lo : end + 1] = np.maximum(health[lo : end + 1], ramp)
            maint_rows.append({"machine_id": m["machine_id"], "timestamp": grid[end + 1], "component": "preventive"})

        # Routine maintenance, unrelated to degradation.
        for idx in rng.integers(0, n_hours, rng.poisson(cfg.days / 45)):
            maint_rows.append({"machine_id": m["machine_id"], "timestamp": grid[idx], "component": "routine"})

        sig = {}
        for s in BASE:
            offset = rng.normal(0, NOISE[s] * 0.4)
            sig[s] = BASE[s] + offset + NOISE[s] * _ar1(rng, n_hours) + RAMP[s] * health
        telemetry_parts.append(pd.DataFrame({"machine_id": m["machine_id"], "timestamp": grid, **sig}))

        err_rate = 0.01 + 0.22 * health
        for idx in np.flatnonzero(rng.random(n_hours) < err_rate):
            error_rows.append({"machine_id": m["machine_id"], "timestamp": grid[idx], "error_code": f"E{rng.integers(1, 6)}"})

    def frame(rows, cols):
        df = pd.DataFrame(rows, columns=cols)
        return df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    return {
        "telemetry": pd.concat(telemetry_parts, ignore_index=True).round({k: 3 for k in BASE}),
        "errors": frame(error_rows, ["machine_id", "timestamp", "error_code"]),
        "maintenance": frame(maint_rows, ["machine_id", "timestamp", "component"]),
        "failures": frame(fail_rows, ["machine_id", "timestamp", "component"]),
        "machines": machines,
    }


def apply_drift(
    tables: dict[str, pd.DataFrame],
    start: str | pd.Timestamp,
    machine_share: float = 0.6,
    shifts: dict[str, float] | None = None,
    seed: int = 11,
) -> dict[str, pd.DataFrame]:
    """Return a copy where some machines drift after `start` (sensor level shift).

    Example: a sensor recalibration or a new operating environment. Only the
    telemetry changes; failures keep following the original degradation process.
    """
    shifts = shifts or {"vibration": 9.0, "voltage": 12.0, "pressure": 8.0}
    rng = np.random.default_rng(seed)
    ids = tables["machines"]["machine_id"].to_numpy()
    drifted = set(rng.choice(ids, int(len(ids) * machine_share), replace=False))
    out = {k: v.copy() for k, v in tables.items()}
    tel = out["telemetry"]
    mask = tel["machine_id"].isin(drifted) & (tel["timestamp"] >= pd.Timestamp(start))
    days_in = (tel.loc[mask, "timestamp"] - pd.Timestamp(start)).dt.total_seconds() / 86400
    ramp = np.minimum(days_in / 5.0, 1.0)  # progressive over 5 days
    for col, delta in shifts.items():
        tel.loc[mask, col] = tel.loc[mask, col] + delta * ramp
    out["telemetry"] = tel
    return out


def save(tables: dict[str, pd.DataFrame], directory) -> None:
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(d / f"{name}.csv", index=False)
