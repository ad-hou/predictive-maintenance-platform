"""Training path (full-table features) and serving path (one reading + 72 h of history)
must produce identical features."""

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.features.serving import features_for_reading, features_for_reading_reference


def _machine_tables(tables, m):
    return {t: tables[t][tables[t].machine_id == m] for t in ("telemetry", "errors", "maintenance", "failures")}, tables["machines"][
        tables["machines"].machine_id == m
    ]


def test_serving_features_equal_training_features(tables, dataset):
    rng = np.random.default_rng(0)
    machines = tables["machines"]["machine_id"].to_numpy()
    checked = 0
    for _ in range(300):
        m = rng.choice(machines)
        sub = dataset[dataset.machine_id == m]
        row = sub.iloc[int(rng.integers(0, len(sub)))]
        mt, mrow = _machine_tables(tables, m)
        reading = {"timestamp": row["timestamp"], **{s: row[s] for s in ("voltage", "rotation", "pressure", "vibration")}}
        online = features_for_reading(mt["telemetry"], mt["errors"], mt["maintenance"], mt["failures"], mrow, reading)
        a = online[FEATURE_COLUMNS].to_numpy(dtype=float)[0]
        b = row[FEATURE_COLUMNS].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-9), (m, row["timestamp"], dict(zip(FEATURE_COLUMNS, zip(a, b, strict=True), strict=True)))
        checked += 1
    assert checked == 300


def test_serving_with_short_history_does_not_crash(tables):
    m = tables["machines"]["machine_id"].iloc[0]
    mt, mrow = _machine_tables(tables, m)
    first = mt["telemetry"].iloc[0]
    reading = {"timestamp": first["timestamp"], "voltage": 170, "rotation": 450, "pressure": 100, "vibration": 40}
    out = features_for_reading(mt["telemetry"], mt["errors"], mt["maintenance"], mt["failures"], mrow, reading)
    assert out[FEATURE_COLUMNS].notna().all().all()
    assert isinstance(out["timestamp"].iloc[0], pd.Timestamp)


def test_fast_path_equals_the_reference_path(tables, dataset):
    rng = np.random.default_rng(1)
    for _ in range(40):
        m = rng.choice(tables["machines"]["machine_id"].to_numpy())
        sub = dataset[dataset.machine_id == m]
        row = sub.iloc[int(rng.integers(0, len(sub)))]  # includes the very first rows (short history)
        mt, mrow = _machine_tables(tables, m)
        reading = {"timestamp": row["timestamp"], **{s: row[s] for s in ("voltage", "rotation", "pressure", "vibration")}}
        args = (mt["telemetry"], mt["errors"], mt["maintenance"], mt["failures"], mrow, reading)
        a = features_for_reading(*args)[FEATURE_COLUMNS].to_numpy(dtype=float)
        b = features_for_reading_reference(*args)[FEATURE_COLUMNS].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-9), (m, row["timestamp"])
