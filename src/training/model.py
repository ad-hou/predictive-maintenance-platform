"""The deployable model: estimator + probability calibrator + decision threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ChampionModel:
    name: str
    estimator: Any
    calibrator: Any  # IsotonicRegression fitted on a separate chronological block
    threshold: float  # calibrated probability above which a machine is flagged HIGH
    feature_names: list[str]
    version: str = "unregistered"
    medium_threshold: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.medium_threshold:
            self.medium_threshold = self.threshold / 2.0

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.feature_names])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw_score(X)), 0.0, 1.0)

    def risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "HIGH"
        if p >= self.medium_threshold:
            return "MEDIUM"
        return "LOW"
