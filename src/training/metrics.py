"""Evaluation: ranking quality, calibration, operational value and cost."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config
from src.config import Costs


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    y = y.astype(bool)
    flagged = flagged.astype(bool)
    return {
        "tp": int((y & flagged).sum()),
        "fp": int((~y & flagged).sum()),
        "fn": int((y & ~flagged).sum()),
        "tn": int((~y & ~flagged).sum()),
    }


def total_cost(y: np.ndarray, flagged: np.ndarray, costs: Costs | None = None) -> float:
    """Hypothetical cost in euros of one decision per observation (see docs/cost_assumptions.md)."""
    costs = costs or config.COSTS
    c = confusion(y, flagged)
    return c["fn"] * costs.missed_failure + c["fp"] * costs.useless_inspection + c["tp"] * costs.useful_inspection


def best_threshold(y: np.ndarray, p: np.ndarray, costs: Costs | None = None) -> tuple[float, float]:
    """Threshold on calibrated probabilities that minimises the total cost."""
    candidates = np.unique(np.concatenate([np.quantile(p, np.linspace(0.5, 0.999, 120)), [0.5]]))
    best = (0.5, np.inf)
    for t in candidates:
        c = total_cost(y, p >= t, costs)
        if c < best[1]:
            best = (float(t), float(c))
    return best


def cost_curve(y: np.ndarray, p: np.ndarray, costs: Costs | None = None, points: int = 60) -> pd.DataFrame:
    ts = np.unique(np.quantile(p, np.linspace(0.5, 0.999, points)))
    return pd.DataFrame({"threshold": ts, "cost": [total_cost(y, p >= t, costs) for t in ts]})


def precision_at_n(df: pd.DataFrame, p: np.ndarray, n: int | None = None) -> dict[str, float]:
    """Each day the team inspects the n riskiest machines. Returns precision and failures caught."""
    n = n or config.INSPECTIONS_PER_DAY
    d = pd.DataFrame(
        {
            "day": df["timestamp"].dt.floor("D").to_numpy(),
            "machine_id": df["machine_id"].to_numpy(),
            "p": p,
            "y": df["failure_next_24h"].to_numpy(),
        }
    )
    daily = d.groupby(["day", "machine_id"]).agg(p=("p", "max"), y=("y", "max")).reset_index()
    hits, picked, total_pos = 0, 0, int(daily["y"].sum())
    for _, day in daily.groupby("day"):
        top = day.nlargest(n, "p")
        hits += int(top["y"].sum())
        picked += len(top)
    return {
        "precision_at_n": hits / picked if picked else float("nan"),
        "failures_caught_share": hits / total_pos if total_pos else float("nan"),
        "n_per_day": n,
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency per quantile bin of the predicted probability."""
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    t = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return t


def evaluate(df: pd.DataFrame, p: np.ndarray, threshold: float | None, costs: Costs | None = None) -> dict:
    y = df["failure_next_24h"].to_numpy()
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "baseline_pr_auc": float(y.mean()),
    }
    out.update(precision_at_n(df, p))
    if threshold is not None:
        flagged = p >= threshold
        c = confusion(y, flagged)
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out.update(
            {
                "threshold": float(threshold),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "confusion": c,
                "total_cost": total_cost(y, flagged, costs),
                "cost_do_nothing": total_cost(y, np.zeros_like(flagged), costs),
                "cost_inspect_all": total_cost(y, np.ones_like(flagged), costs),
            }
        )
        pr, rc = out["precision"], out["recall"]
        out["f1"] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    return out
