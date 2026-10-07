import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.training.train import chronological_blocks, tail_blocks


def test_blocks_are_chronological_and_disjoint(dataset):
    b = chronological_blocks(dataset)
    order = ["train", "calibration", "threshold", "test"]
    for a, c in zip(order, order[1:], strict=False):
        assert b[a]["timestamp"].max() < b[c]["timestamp"].min()
    assert sum(len(v) for v in b.values()) == int(dataset["label_known"].sum())


def test_tail_blocks_end_with_evaluation_window(dataset):
    b = tail_blocks(dataset, eval_days=7, calibration_days=5, threshold_days=5)
    assert b["train"]["timestamp"].max() < b["calibration"]["timestamp"].min()
    assert b["threshold"]["timestamp"].max() < b["test"]["timestamp"].min()
    assert (b["test"]["timestamp"].max() - b["test"]["timestamp"].min()) <= pd.Timedelta(days=7)


def test_model_beats_the_never_fails_baseline(trained):
    final = trained["final_test"]
    assert final["pr_auc"] > 3 * final["baseline_pr_auc"]
    assert final["total_cost"] < final["cost_do_nothing"]
    assert trained["model"].name in ("logistic_regression", "random_forest", "hist_gradient_boosting", "lightgbm")


def test_all_models_reported(trained):
    names = set(trained["test_report"]["model"])
    assert {"baseline_never_fails", "logistic_regression", "random_forest", "lightgbm"} <= names
    assert any(n.startswith("isolation_forest") for n in names)


def test_model_predictions_are_probabilities(trained, dataset):
    m = trained["model"]
    p = m.predict_proba(dataset[FEATURE_COLUMNS].head(500))
    assert p.shape == (500,) and np.all((p >= 0) & (p <= 1))
    assert m.risk_level(1.0) == "HIGH" and m.risk_level(0.0) == "LOW"
