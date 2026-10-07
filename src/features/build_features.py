"""Feature engineering shared by training AND serving.

Rule: a feature at time t only uses information available at or before t.
The same `build_features` function is used by the training pipeline and by the API,
and `tests/test_train_serve_skew.py` checks that both paths give identical values.

Telemetry is assumed to be on a regular hourly grid per machine (rolling windows are
expressed in rows = hours).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

SENSORS = ["voltage", "rotation", "pressure", "vibration"]
CAP_HOURS = 24.0 * 30  # "never happened" is encoded as 30 days
HOUR_NS = 3_600_000_000_000

FEATURE_COLUMNS: list[str] = (
    [f"{s}_mean_3h" for s in SENSORS]
    + [f"{s}_mean_24h" for s in SENSORS]
    + [f"{s}_std_24h" for s in SENSORS]
    + [f"{s}_delta_3h" for s in SENSORS]
    + ["vibration_max_6h", "pressure_max_6h"]
    + ["errors_last_24h", "errors_last_72h", "hours_since_last_error"]
    + ["hours_since_last_maintenance", "hours_since_last_failure"]
    + ["age", "model_code"]
)

FEATURE_LABELS = {
    "vibration_mean_3h": "vibration (3 h average)",
    "vibration_mean_24h": "vibration (24 h average)",
    "vibration_max_6h": "vibration (6 h peak)",
    "vibration_delta_3h": "vibration change over 3 h",
    "vibration_std_24h": "vibration variability (24 h)",
    "pressure_mean_3h": "pressure (3 h average)",
    "pressure_mean_24h": "pressure (24 h average)",
    "pressure_max_6h": "pressure (6 h peak)",
    "pressure_delta_3h": "pressure change over 3 h",
    "pressure_std_24h": "pressure variability (24 h)",
    "rotation_mean_3h": "rotation speed (3 h average)",
    "rotation_mean_24h": "rotation speed (24 h average)",
    "rotation_delta_3h": "rotation speed change over 3 h",
    "rotation_std_24h": "rotation variability (24 h)",
    "voltage_mean_3h": "voltage (3 h average)",
    "voltage_mean_24h": "voltage (24 h average)",
    "voltage_delta_3h": "voltage change over 3 h",
    "voltage_std_24h": "voltage variability (24 h)",
    "errors_last_24h": "errors in the last 24 h",
    "errors_last_72h": "errors in the last 72 h",
    "hours_since_last_error": "hours since last error",
    "hours_since_last_maintenance": "hours since last maintenance",
    "hours_since_last_failure": "hours since last failure",
    "age": "machine age",
    "model_code": "machine model",
}


def model_code(model: str) -> int:
    digits = "".join(ch for ch in str(model) if ch.isdigit())
    return int(digits) if digits else 0


def _event_features(machine_ts: np.ndarray, event_ts: np.ndarray, windows_h: tuple[int, ...]):
    """For each timestamp t: counts of events in (t-w, t] and hours since the last event <= t."""
    ev = np.sort(event_ts)
    if len(ev) == 0:  # machine without any event of this kind
        return {w: np.zeros(len(machine_ts)) for w in windows_h}, np.full(len(machine_ts), CAP_HOURS)
    upto = np.searchsorted(ev, machine_ts, side="right")
    counts = {}
    for w in windows_h:
        before = np.searchsorted(ev, machine_ts - w * HOUR_NS, side="right")
        counts[w] = (upto - before).astype(float)
    since = np.full(len(machine_ts), CAP_HOURS)
    has = upto > 0
    last = ev[np.maximum(upto - 1, 0)]
    since[has] = np.minimum((machine_ts[has] - last[has]) / HOUR_NS, CAP_HOURS)
    return counts, since


def build_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maintenance: pd.DataFrame,
    failures: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """One row per telemetry row: machine_id, timestamp, raw sensors and FEATURE_COLUMNS."""
    tel = telemetry.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    g = tel.groupby("machine_id", sort=False)
    out = tel[["machine_id", "timestamp", *SENSORS]].copy()

    def roll(col, window, fn, min_periods=1):
        r = g[col].rolling(window, min_periods=min_periods)
        return getattr(r, fn)().reset_index(level=0, drop=True)

    for s in SENSORS:
        out[f"{s}_mean_3h"] = roll(s, 3, "mean")
        out[f"{s}_mean_24h"] = roll(s, 24, "mean")
        out[f"{s}_std_24h"] = roll(s, 24, "std", min_periods=2).fillna(0.0)
        out[f"{s}_delta_3h"] = (tel[s] - g[s].shift(3)).fillna(0.0)
    out["vibration_max_6h"] = roll("vibration", 6, "max")
    out["pressure_max_6h"] = roll("pressure", 6, "max")

    for name in (
        "errors_last_24h",
        "errors_last_72h",
        "hours_since_last_error",
        "hours_since_last_maintenance",
        "hours_since_last_failure",
    ):
        out[name] = CAP_HOURS if name.startswith("hours") else 0.0

    ev_groups = {
        "err": {k: v["timestamp"].to_numpy("datetime64[ns]").astype("int64") for k, v in errors.groupby("machine_id")},
        "mnt": {k: v["timestamp"].to_numpy("datetime64[ns]").astype("int64") for k, v in maintenance.groupby("machine_id")},
        "fai": {k: v["timestamp"].to_numpy("datetime64[ns]").astype("int64") for k, v in failures.groupby("machine_id")},
    }
    empty = np.array([], dtype="int64")
    for machine_id, idx in tel.groupby("machine_id", sort=False).indices.items():
        ts = tel["timestamp"].to_numpy("datetime64[ns]").astype("int64")[idx]
        counts, since_err = _event_features(ts, ev_groups["err"].get(machine_id, empty), (24, 72))
        _, since_mnt = _event_features(ts, ev_groups["mnt"].get(machine_id, empty), ())
        _, since_fai = _event_features(ts, ev_groups["fai"].get(machine_id, empty), ())
        out.loc[idx, "errors_last_24h"] = counts[24]
        out.loc[idx, "errors_last_72h"] = counts[72]
        out.loc[idx, "hours_since_last_error"] = since_err
        out.loc[idx, "hours_since_last_maintenance"] = since_mnt
        out.loc[idx, "hours_since_last_failure"] = since_fai

    mac = machines.assign(model_code=machines["model"].map(model_code))[["machine_id", "age", "model_code"]]
    out = out.merge(mac, on="machine_id", how="left")
    out["age"] = out["age"].fillna(0)
    out["model_code"] = out["model_code"].fillna(0)
    return out


def add_target(features: pd.DataFrame, failures: pd.DataFrame, horizon_h: int | None = None) -> pd.DataFrame:
    """failure_next_24h = 1 if a failure happens in (t, t + horizon]; `label_known` flags rows
    whose whole horizon is covered by the data (the last `horizon` hours are not labelled)."""
    horizon_h = horizon_h or config.HORIZON_HOURS
    df = features.copy()
    df["failure_next_24h"] = 0
    fails = {k: np.sort(v["timestamp"].to_numpy("datetime64[ns]").astype("int64")) for k, v in failures.groupby("machine_id")}
    ts_all = df["timestamp"].to_numpy("datetime64[ns]").astype("int64")
    for machine_id, idx in df.groupby("machine_id", sort=False).indices.items():
        f = fails.get(machine_id)
        if f is None or len(f) == 0:
            continue
        ts = ts_all[idx]
        nxt = np.searchsorted(f, ts, side="right")
        has_next = nxt < len(f)
        gap = np.where(has_next, f[np.minimum(nxt, len(f) - 1)] - ts, np.iinfo("int64").max)
        df.loc[idx, "failure_next_24h"] = (gap <= horizon_h * HOUR_NS).astype(int)
    df["label_known"] = df["timestamp"] <= df["timestamp"].max() - pd.Timedelta(hours=horizon_h)
    return df


def make_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    feats = build_features(tables["telemetry"], tables["errors"], tables["maintenance"], tables["failures"], tables["machines"])
    return add_target(feats, tables["failures"])
