import numpy as np
import pandas as pd

from src.ingestion.synthetic import apply_drift
from src.monitoring.drift import drift_report, psi


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert psi(a, b) < 0.02


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0.3, 1, 20000)) < psi(a, rng.normal(1.0, 1, 20000))
    assert psi(a, rng.normal(1.0, 1, 20000)) > 0.2


def test_report_flags_only_the_drifted_variable():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"vibration": rng.normal(40, 5, 10000), "pressure": rng.normal(100, 10, 10000)})
    cur = pd.DataFrame({"vibration": rng.normal(52, 5, 10000), "pressure": rng.normal(100, 10, 10000)})
    r = drift_report(ref, cur, columns=["vibration", "pressure"])
    assert r["drift_detected"] and r["drifted_variables"] == ["vibration"]


def test_apply_drift_changes_only_late_telemetry(tables):
    start = pd.Timestamp("2025-02-15")
    d = apply_drift(tables, start, machine_share=1.0)
    before = tables["telemetry"]["timestamp"] < start
    assert np.allclose(d["telemetry"].loc[before, "vibration"], tables["telemetry"].loc[before, "vibration"])
    assert d["telemetry"].loc[~before, "vibration"].mean() > tables["telemetry"].loc[~before, "vibration"].mean() + 3
    assert d["failures"].equals(tables["failures"])
