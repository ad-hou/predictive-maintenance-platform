"""Online path: features for ONE new sensor reading.

`features_for_reading` is a lean numpy implementation (about 100x faster than running the whole
pandas pipeline for a single row). It must give EXACTLY the same values as the training code:
`tests/test_train_serve_skew.py` compares both on hundreds of random (machine, time) points, and
`features_for_reading_reference` keeps the original "reuse build_features" version for that test.

Only the last HISTORY_HOURS of telemetry are needed; events older than CAP_HOURS cannot change any
feature (they are capped), so they are ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import CAP_HOURS, FEATURE_COLUMNS, HOUR_NS, SENSORS, build_features, model_code

HISTORY_HOURS = 72


def _ns(series: pd.Series) -> np.ndarray:
    return series.to_numpy("datetime64[ns]").astype("int64")


def _since_and_count(events_ns: np.ndarray, t_ns: int, windows_h=()) -> tuple[float, list[float]]:
    if len(events_ns) == 0:
        return CAP_HOURS, [0.0] * len(windows_h)
    ev = np.sort(events_ns)
    upto = int(np.searchsorted(ev, t_ns, side="right"))
    since = CAP_HOURS if upto == 0 else min((t_ns - ev[upto - 1]) / HOUR_NS, CAP_HOURS)
    counts = [float(upto - np.searchsorted(ev, t_ns - w * HOUR_NS, side="right")) for w in windows_h]
    return float(since), counts


def features_for_reading(
    telemetry_m: pd.DataFrame,
    errors_m: pd.DataFrame,
    maintenance_m: pd.DataFrame,
    failures_m: pd.DataFrame,
    machine_row: pd.DataFrame,
    reading: dict,
    history_hours: int = HISTORY_HOURS,
) -> pd.DataFrame:
    """`*_m` are the tables restricted to one machine; `reading` = timestamp + the four sensors."""
    ts = pd.Timestamp(reading["timestamp"])
    t_ns = int(ts.value) if hasattr(ts, "value") else int(ts.to_datetime64().astype("int64"))
    machine_id = machine_row["machine_id"].iloc[0]

    tel_ns = _ns(telemetry_m["timestamp"])
    end = int(np.searchsorted(tel_ns, t_ns, side="left"))  # rows strictly before the reading
    start = max(0, end - history_hours)
    f: dict[str, float] = {}
    for s in SENSORS:
        x = np.append(telemetry_m[s].to_numpy(dtype=float)[start:end], float(reading[s]))
        w24, w3, w6 = x[-24:], x[-3:], x[-6:]
        f[f"{s}_mean_3h"] = float(w3.mean())
        f[f"{s}_mean_24h"] = float(w24.mean())
        f[f"{s}_std_24h"] = float(w24.std(ddof=1)) if len(w24) >= 2 else 0.0
        f[f"{s}_delta_3h"] = float(x[-1] - x[-4]) if len(x) > 3 else 0.0
        if s in ("vibration", "pressure"):
            f[f"{s}_max_6h"] = float(w6.max())

    since_err, (n24, n72) = _since_and_count(_ns(errors_m["timestamp"]), t_ns, (24, 72))
    since_mnt, _ = _since_and_count(_ns(maintenance_m["timestamp"]), t_ns)
    since_fai, _ = _since_and_count(_ns(failures_m["timestamp"]), t_ns)
    f.update(
        errors_last_24h=n24,
        errors_last_72h=n72,
        hours_since_last_error=since_err,
        hours_since_last_maintenance=since_mnt,
        hours_since_last_failure=since_fai,
        age=float(machine_row["age"].iloc[0]) if pd.notna(machine_row["age"].iloc[0]) else 0.0,
        model_code=float(model_code(machine_row["model"].iloc[0])),
    )
    out = pd.DataFrame([{c: f[c] for c in FEATURE_COLUMNS}])
    out.insert(0, "timestamp", ts)
    out.insert(0, "machine_id", machine_id)
    return out


def features_for_reading_reference(telemetry_m, errors_m, maintenance_m, failures_m, machine_row, reading, history_hours=HISTORY_HOURS):
    """Slow version that reuses `build_features` as is. Kept as the oracle for the skew tests."""
    ts = pd.Timestamp(reading["timestamp"])
    machine_id = machine_row["machine_id"].iloc[0]
    hist = telemetry_m[telemetry_m["timestamp"] < ts].tail(history_hours)
    new = pd.DataFrame([{"machine_id": machine_id, "timestamp": ts, **{s: float(reading[s]) for s in SENSORS}}])
    tel = pd.concat([hist[["machine_id", "timestamp", *SENSORS]], new], ignore_index=True)
    horizon = ts - pd.Timedelta(hours=CAP_HOURS)

    def recent(df):
        return df[(df["timestamp"] <= ts) & (df["timestamp"] >= horizon)]

    feats = build_features(tel, recent(errors_m), recent(maintenance_m), recent(failures_m), machine_row)
    return feats.iloc[[-1]][["machine_id", "timestamp", *FEATURE_COLUMNS]].reset_index(drop=True)
