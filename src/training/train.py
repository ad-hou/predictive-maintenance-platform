"""Model comparison with strictly chronological blocks, calibration and cost-based threshold.

Blocks (by time, never random):  train | calibration | threshold | test
 - train:       fit every candidate model
 - calibration: fit the isotonic calibrator of the selected model
 - threshold:   choose the decision threshold that minimises the (hypothetical) cost
 - test:        final report, untouched until the end
"""

from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS
from src.training import metrics
from src.training.model import ChampionModel

FRACTIONS = (0.55, 0.15, 0.15, 0.15)
SUPERVISED = ("logistic_regression", "random_forest", "hist_gradient_boosting", "lightgbm")


def chronological_blocks(df: pd.DataFrame, fractions=FRACTIONS) -> dict[str, pd.DataFrame]:
    """Split labelled rows into consecutive time blocks (same cut dates for all machines)."""
    d = df[df["label_known"]]
    ts = np.sort(d["timestamp"].unique())
    edges = np.cumsum(fractions)[:-1]
    cuts = [ts[int(len(ts) * e)] for e in edges]
    names = ["train", "calibration", "threshold", "test"]
    bounds = [None, *cuts, None]
    out = {}
    for i, name in enumerate(names):
        lo, hi = bounds[i], bounds[i + 1]
        m = np.ones(len(d), dtype=bool)
        if lo is not None:
            m &= d["timestamp"].to_numpy() >= lo
        if hi is not None:
            m &= d["timestamp"].to_numpy() < hi
        out[name] = d[m].reset_index(drop=True)
    return out


def candidate_models(seed: int = 42) -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=200, class_weight="balanced", random_state=seed
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def _isolation_scores(model: IsolationForest, ref: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    raw = -model.score_samples(X[FEATURE_COLUMNS])
    lo, hi = ref.min(), ref.max()
    return np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)


def tail_blocks(df: pd.DataFrame, eval_days: int, calibration_days: int, threshold_days: int) -> dict[str, pd.DataFrame]:
    """Recent-window blocks used when retraining after drift: the training block runs up to
    the calibration window, so it contains the most recent regime. `test` is the last window."""
    d = df[df["label_known"]]
    end = d["timestamp"].max()
    t_test = end - pd.Timedelta(days=eval_days)
    t_thr = t_test - pd.Timedelta(days=threshold_days)
    t_cal = t_thr - pd.Timedelta(days=calibration_days)
    ts = d["timestamp"]
    return {
        "train": d[ts < t_cal].reset_index(drop=True),
        "calibration": d[(ts >= t_cal) & (ts < t_thr)].reset_index(drop=True),
        "threshold": d[(ts >= t_thr) & (ts < t_test)].reset_index(drop=True),
        "test": d[ts >= t_test].reset_index(drop=True),
    }


def run_comparison(df: pd.DataFrame, seed: int = 42, costs=None, blocks: dict | None = None) -> dict:
    """Fit all candidates on `train`, rank them on `calibration`, report on `test`."""
    blocks = blocks if blocks is not None else chronological_blocks(df)
    tr, ca, th, te = (blocks[k] for k in ("train", "calibration", "threshold", "test"))
    ytr = tr["failure_next_24h"].to_numpy()
    if ytr.sum() < 5:
        raise ValueError("not enough failures in the training block")

    fitted, table = {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Baseline: never predicts a failure.
        for name, est in candidate_models(seed).items():
            t0 = time.time()
            est.fit(tr[FEATURE_COLUMNS], ytr)
            fitted[name] = est
            p = est.predict_proba(ca[FEATURE_COLUMNS])[:, 1]
            table.append(
                {"model": name, "pr_auc_calibration": metrics.evaluate(ca, p, None)["pr_auc"], "fit_seconds": round(time.time() - t0, 1)}
            )
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1)
        iso.fit(tr[FEATURE_COLUMNS])
        ref = -iso.score_samples(tr[FEATURE_COLUMNS])
        p_iso = _isolation_scores(iso, ref, ca)
        table.append(
            {
                "model": "isolation_forest (unsupervised)",
                "pr_auc_calibration": metrics.evaluate(ca, p_iso, None)["pr_auc"],
                "fit_seconds": 0.0,
            }
        )

    ranking = pd.DataFrame(table).sort_values("pr_auc_calibration", ascending=False).reset_index(drop=True)
    best = next(m for m in ranking["model"] if m in SUPERVISED)
    est = fitted[best]

    # Calibration on its own block, then threshold on a separate block.
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(est.predict_proba(ca[FEATURE_COLUMNS])[:, 1], ca["failure_next_24h"].to_numpy())
    p_th = calibrator.predict(est.predict_proba(th[FEATURE_COLUMNS])[:, 1])
    threshold, cost_on_threshold_block = metrics.best_threshold(th["failure_next_24h"].to_numpy(), p_th, costs)

    model = ChampionModel(
        name=best,
        estimator=est,
        calibrator=calibrator,
        threshold=threshold,
        feature_names=list(FEATURE_COLUMNS),
        meta={
            "train_period": [str(tr["timestamp"].min()), str(tr["timestamp"].max())],
            "calibration_period": [str(ca["timestamp"].min()), str(ca["timestamp"].max())],
            "threshold_period": [str(th["timestamp"].min()), str(th["timestamp"].max())],
            "test_period": [str(te["timestamp"].min()), str(te["timestamp"].max())],
            "seed": seed,
            "reference_medians": {c: float(v) for c, v in tr[FEATURE_COLUMNS].median().items()},
        },
    )

    # Final report on the test block. Every model goes through the SAME procedure (isotonic calibration on
    # the calibration block, cost-optimal threshold on the threshold block) so the rows are comparable.
    def report_row(name, score_fn):
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score_fn(ca), ca["failure_next_24h"].to_numpy())
        thr, _ = metrics.best_threshold(th["failure_next_24h"].to_numpy(), cal.predict(score_fn(th)), costs)
        ev = metrics.evaluate(te, cal.predict(score_fn(te)), thr, costs)
        return {"model": name, **{k: ev.get(k) for k in ("pr_auc", "roc_auc", "precision_at_n", "recall", "precision", "total_cost")}}

    ypos = te["failure_next_24h"].to_numpy()
    base = metrics.evaluate(te, np.zeros(len(te)), 0.5, costs)
    report = [{"model": "baseline_never_fails", **{k: base.get(k) for k in ("pr_auc", "recall", "precision", "total_cost")}}]
    for name, e in fitted.items():
        report.append(report_row(name, lambda X, e=e: e.predict_proba(X[FEATURE_COLUMNS])[:, 1]))
    report.append(report_row("isolation_forest (unsupervised)", lambda X: _isolation_scores(iso, ref, X)))

    p_cal_test = model.predict_proba(te)
    final = metrics.evaluate(te, p_cal_test, threshold, costs)
    model.meta["test_metrics"] = final
    return {
        "model": model,
        "blocks": {k: {"rows": len(v), "positives": int(v["failure_next_24h"].sum())} for k, v in blocks.items()},
        "ranking_calibration": ranking,
        "test_report": pd.DataFrame(report),
        "final_test": final,
        "calibration_table": metrics.calibration_table(ypos, p_cal_test),
        "cost_curve": metrics.cost_curve(th["failure_next_24h"].to_numpy(), p_th, costs),
        "threshold_block_cost": cost_on_threshold_block,
        "iso": (iso, ref),
    }
