import numpy as np
import pandas as pd

from src.config import Costs
from src.training import metrics

COSTS = Costs(missed_failure=10000, useless_inspection=500, useful_inspection=500)


def test_total_cost_exact():
    y = np.array([1, 1, 0, 0, 0])
    flagged = np.array([1, 0, 1, 0, 0])
    # 1 TP (500) + 1 FN (10000) + 1 FP (500)
    assert metrics.total_cost(y, flagged, COSTS) == 11000


def test_best_threshold_is_not_beaten_by_other_candidates():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=20000) < 0.03).astype(int)
    p = np.clip(0.03 + 0.4 * y + rng.normal(0, 0.1, 20000), 0, 1)
    thr, cost = metrics.best_threshold(y, p, COSTS)
    for t in np.linspace(0.05, 0.9, 40):
        assert cost <= metrics.total_cost(y, p >= t, COSTS) + 1e-6
    assert cost < metrics.total_cost(y, np.zeros_like(y), COSTS)


def test_higher_miss_cost_lowers_the_threshold():
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=20000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.12, 20000), 0, 1)
    cheap, _ = metrics.best_threshold(y, p, Costs(missed_failure=2000, useless_inspection=500, useful_inspection=500))
    dear, _ = metrics.best_threshold(y, p, Costs(missed_failure=50000, useless_inspection=500, useful_inspection=500))
    assert dear <= cheap


def test_precision_at_n_exact():
    ts = pd.to_datetime(["2025-01-01 10:00"] * 3 + ["2025-01-02 10:00"] * 3)
    df = pd.DataFrame({"machine_id": ["A", "B", "C"] * 2, "timestamp": ts, "failure_next_24h": [1, 0, 0, 0, 0, 1]})
    good = np.array([0.9, 0.5, 0.1, 0.1, 0.2, 0.8])  # ranks the failing machine first every day
    bad = np.array([0.1, 0.5, 0.9, 0.9, 0.2, 0.1])
    assert metrics.precision_at_n(df, good, n=1) == {"precision_at_n": 1.0, "failures_caught_share": 1.0, "n_per_day": 1}
    assert metrics.precision_at_n(df, bad, n=1)["precision_at_n"] == 0.0
    assert metrics.precision_at_n(df, bad, n=3)["failures_caught_share"] == 1.0
