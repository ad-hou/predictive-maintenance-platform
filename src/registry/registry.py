"""MLflow model registry with champion / challenger aliases, promotion and rollback."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src import config
from src.training.model import ChampionModel

EXPERIMENT = "predictive-maintenance"


@dataclass
class Decision:
    promoted: bool
    reason: str
    champion_version: str | None
    challenger_version: str
    champion_cost: float | None
    challenger_cost: float
    window: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None, name: str | None = None, models_dir: Path | None = None):
        self.uri = tracking_uri or config.MLFLOW_TRACKING_URI
        self.name = name or config.MODEL_NAME
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        mlflow.set_tracking_uri(self.uri)
        self.client = MlflowClient(self.uri)

    # -- registration -------------------------------------------------------------------
    def register_challenger(self, model: ChampionModel, metrics: dict, params: dict | None = None, tags: dict | None = None) -> str:
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"{model.name}"):
            mlflow.log_params({"model": model.name, "threshold": round(model.threshold, 5), **(params or {})})
            mlflow.log_params({"n_features": len(model.feature_names)})
            flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
            mlflow.log_metrics(flat)
            if tags:
                mlflow.set_tags(tags)
            mlflow.log_dict({"features": model.feature_names, "meta": model.meta}, "model_info.json")
            info = mlflow.sklearn.log_model(model, name="model", registered_model_name=self.name, serialization_format="cloudpickle")
        version = str(info.registered_model_version)
        self.client.set_registered_model_alias(self.name, "challenger", version)
        model.version = version
        return version

    # -- aliases ----------------------------------------------------------------------------
    def version_of(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(self.name, alias).version)
        except Exception:
            return None

    def load(self, alias_or_version: str) -> ChampionModel:
        ref = alias_or_version
        uri = f"models:/{self.name}/{ref}" if ref.isdigit() else f"models:/{self.name}@{ref}"
        model = mlflow.sklearn.load_model(uri)
        version = ref if ref.isdigit() else self.version_of(ref)
        model.version = str(version)
        return model

    def metrics_of(self, version: str) -> dict:
        mv = self.client.get_model_version(self.name, version)
        return self.client.get_run(mv.run_id).data.metrics

    # -- promotion / rollback --------------------------------------------------------------
    def promote(self, version: str) -> None:
        current = self.version_of("champion")
        if current and current != version:
            self.client.set_registered_model_alias(self.name, "previous_champion", current)
        self.client.set_registered_model_alias(self.name, "champion", version)
        self.export_champion(version)

    def rollback(self) -> str:
        previous = self.version_of("previous_champion")
        if previous is None:
            raise RuntimeError("no previous champion to roll back to")
        current = self.version_of("champion")
        self.client.set_registered_model_alias(self.name, "champion", previous)
        if current:
            self.client.set_registered_model_alias(self.name, "previous_champion", current)
        self.export_champion(previous)
        return previous

    def export_champion(self, version: str) -> Path:
        """Write the champion to models/champion.joblib so the API can serve it without MLflow."""
        model = self.load(version)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.models_dir / "champion.joblib.tmp"
        joblib.dump(model, tmp)
        shutil.move(tmp, self.models_dir / "champion.joblib")
        (self.models_dir / "champion.json").write_text(
            json.dumps(
                {"version": str(version), "name": model.name, "threshold": model.threshold, "meta": model.meta}, indent=2, default=str
            )
        )
        uri = os.getenv("MODEL_S3_URI")
        if uri:  # make the new champion available to the API running elsewhere (EC2)
            from api import s3_sync

            s3_sync.upload(self.models_dir / "champion.joblib", uri)
        return self.models_dir / "champion.joblib"


def decide(champion_cost: float | None, challenger_cost: float, min_gain: float | None = None) -> tuple[bool, str]:
    """Promote only if the challenger lowers the total cost by at least `min_gain` (relative)."""
    min_gain = config.MIN_COST_GAIN if min_gain is None else min_gain
    if champion_cost is None:
        return True, "no champion yet: first model is promoted"
    gain = (champion_cost - challenger_cost) / champion_cost if champion_cost > 0 else 0.0
    if gain >= min_gain:
        return True, f"challenger lowers the total cost by {gain:.1%} (required: {min_gain:.1%})"
    return False, f"challenger gain is {gain:.1%}, below the required {min_gain:.1%}: champion kept"


def log_decision(decision: Decision, extra: dict | None = None, logs_dir: Path | None = None) -> None:
    d = Path(logs_dir or config.LOGS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    row = {"at": pd.Timestamp.now().isoformat(timespec="seconds"), **decision.as_dict(), **(extra or {})}
    with open(d / "promotion_decisions.jsonl", "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
