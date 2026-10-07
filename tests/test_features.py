import numpy as np
import pandas as pd

from src.features.build_features import CAP_HOURS, FEATURE_COLUMNS, add_target, build_features


def _row(df, machine, hour):
    ts = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=hour)
    return df[(df.machine_id == machine) & (df.timestamp == ts)].iloc[0]


def test_event_features_exact(tiny):
    f = build_features(**{k: tiny[k] for k in ("telemetry", "errors", "maintenance", "failures", "machines")})
    r = _row(f, "M001", 35)
    assert r.errors_last_24h == 1  # only the error at hour 30 is in (11, 35]
    assert r.errors_last_72h == 2
    assert r.hours_since_last_error == 5
    assert r.hours_since_last_maintenance == 15
    assert r.hours_since_last_failure == CAP_HOURS  # failure at hour 50 is in the future
    assert _row(f, "M001", 55).hours_since_last_failure == 5
    assert r.age == 5 and r.model_code == 2


def test_machine_without_events_does_not_crash(tiny):
    f = build_features(**{k: tiny[k] for k in ("telemetry", "errors", "maintenance", "failures", "machines")})
    r = _row(f, "M002", 30)
    assert r.errors_last_72h == 0 and r.hours_since_last_failure == CAP_HOURS


def test_no_leakage_from_the_future(tiny):
    """Features at time t must not change when everything after t is removed."""
    full = build_features(**{k: tiny[k] for k in ("telemetry", "errors", "maintenance", "failures", "machines")})
    t = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=40)
    cut = {k: (v[v.timestamp <= t] if "timestamp" in v else v) for k, v in tiny.items()}
    part = build_features(**{k: cut[k] for k in ("telemetry", "errors", "maintenance", "failures", "machines")})
    a = full[(full.machine_id == "M001") & (full.timestamp == t)][FEATURE_COLUMNS].to_numpy(dtype=float)
    b = part[(part.machine_id == "M001") & (part.timestamp == t)][FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(a, b)


def test_target_definition(tiny):
    f = build_features(**{k: tiny[k] for k in ("telemetry", "errors", "maintenance", "failures", "machines")})
    d = add_target(f, tiny["failures"], horizon_h=24)
    y = d[d.machine_id == "M001"].set_index("timestamp")["failure_next_24h"]
    base = pd.Timestamp("2025-01-01")
    assert y[base + pd.Timedelta(hours=26)] == 1  # failure at hour 50 is within 24 h
    assert y[base + pd.Timedelta(hours=25)] == 0  # 25 h away
    assert y[base + pd.Timedelta(hours=50)] == 0  # the failure hour itself is not "next"
    assert (d[d.machine_id == "M002"]["failure_next_24h"] == 0).all()
    assert not d.iloc[-1]["label_known"]  # last hours cannot be labelled


def test_dataset_shape_and_no_nan(dataset):
    assert dataset[FEATURE_COLUMNS].notna().all().all()
    assert 0.005 < dataset[dataset.label_known]["failure_next_24h"].mean() < 0.15
