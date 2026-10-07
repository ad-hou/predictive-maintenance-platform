"""Per-prediction explanations (top factors pushing the risk up or down)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_LABELS
from src.training.model import ChampionModel


def _contributions(model: ChampionModel, X: pd.DataFrame) -> np.ndarray | None:
    """Signed contributions in the model's raw score space; None if the model is unsupported."""
    est = model.estimator
    Xf = X[model.feature_names]
    kind = type(est).__name__
    if kind in ("LGBMClassifier", "RandomForestClassifier"):
        import shap

        values = shap.TreeExplainer(est).shap_values(Xf)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return values
    if kind == "Pipeline" and hasattr(est[-1], "coef_"):
        scaled = est[:-1].transform(Xf)
        return scaled * est[-1].coef_[0]
    return _occlusion(model, Xf)


def _occlusion(model: ChampionModel, Xf: pd.DataFrame) -> np.ndarray | None:
    """Model-agnostic fallback (e.g. HistGradientBoosting): how much does the raw score drop when one
    feature is replaced by its reference (training-median) value? An approximation, not SHAP."""
    ref = model.meta.get("reference_medians")
    if not ref:
        return None
    row = Xf.iloc[[0]]
    variants = pd.concat([row] * (len(model.feature_names) + 1), ignore_index=True).astype(float)
    for i, f in enumerate(model.feature_names):
        variants.loc[i + 1, f] = ref[f]
    scores = model.estimator.predict_proba(variants[model.feature_names])[:, 1]
    return (scores[0] - scores[1:])[None, :]


def top_factors(model: ChampionModel, X: pd.DataFrame, k: int = 3) -> list[dict]:
    """Top-k features that increase the risk for the (single) row in X."""
    contrib = _contributions(model, X.iloc[[0]])
    if contrib is None:
        return []
    row = X.iloc[0]
    order = np.argsort(-contrib[0])[:k]
    return [
        {
            "feature": model.feature_names[i],
            "label": FEATURE_LABELS.get(model.feature_names[i], model.feature_names[i]),
            "value": round(float(row[model.feature_names[i]]), 3),
            "contribution": round(float(contrib[0][i]), 4),
        }
        for i in order
        if contrib[0][i] > 0
    ]
