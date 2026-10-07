from __future__ import annotations

import os
import warnings

import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

from src.features.build_features import make_dataset  # noqa: E402
from src.ingestion.synthetic import SyntheticConfig, generate, save  # noqa: E402
from src.registry.registry import ModelRegistry  # noqa: E402
from src.training.train import run_comparison  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    return generate(SyntheticConfig(n_machines=24, days=75, seed=3))


@pytest.fixture(scope="session")
def dataset(tables):
    return make_dataset(tables)


@pytest.fixture(scope="session")
def trained(dataset):
    return run_comparison(dataset)


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("pmp")


@pytest.fixture(scope="session")
def registry(workdir, trained):
    reg = ModelRegistry(f"sqlite:///{workdir}/mlflow.db", models_dir=workdir / "models")
    version = reg.register_challenger(trained["model"], {"total_cost": trained["final_test"]["total_cost"]})
    reg.promote(version)
    return reg


@pytest.fixture(scope="session")
def data_dir(workdir, tables):
    d = workdir / "raw"
    save(tables, d)
    return d


@pytest.fixture()
def tiny():
    """Hand-made tables for exact assertions."""
    ts = pd.date_range("2025-01-01", periods=60, freq="h")
    tel = pd.DataFrame({"machine_id": "M001", "timestamp": ts, "voltage": 170.0, "rotation": 450.0, "pressure": 100.0, "vibration": 40.0})
    tel["vibration"] = [40.0 + i for i in range(60)]
    tel2 = tel.assign(machine_id="M002")
    errors = pd.DataFrame({"machine_id": ["M001", "M001"], "timestamp": [ts[10], ts[30]], "error_code": ["E1", "E2"]})
    maint = pd.DataFrame({"machine_id": ["M001"], "timestamp": [ts[20]], "component": ["x"]})
    fail = pd.DataFrame({"machine_id": ["M001"], "timestamp": [ts[50]], "component": ["c"]})
    machines = pd.DataFrame({"machine_id": ["M001", "M002"], "model": ["model2", "model3"], "age": [5, 7]})
    return {
        "telemetry": pd.concat([tel, tel2], ignore_index=True),
        "errors": errors,
        "maintenance": maint,
        "failures": fail,
        "machines": machines,
    }
