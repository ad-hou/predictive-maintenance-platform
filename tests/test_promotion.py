import pandas as pd

from src.ingestion.synthetic import SyntheticConfig, apply_drift, generate
from src.registry import cycle
from src.registry.registry import ModelRegistry, decide


def test_decide_rules():
    assert decide(None, 100)[0]
    assert decide(1000, 900, min_gain=0.05)[0]
    ok, reason = decide(1000, 990, min_gain=0.05)
    assert not ok and "champion kept" in reason
    assert not decide(1000, 1200)[0]


def test_registry_promotion_and_rollback(trained, workdir):
    reg = ModelRegistry(f"sqlite:///{workdir}/rollback.db", models_dir=workdir / "rb_models")
    v1 = reg.register_challenger(trained["model"], {"total_cost": 1.0})
    reg.promote(v1)
    v2 = reg.register_challenger(trained["model"], {"total_cost": 0.5})
    assert reg.version_of("champion") == v1 and reg.version_of("challenger") == v2
    reg.promote(v2)
    assert reg.version_of("champion") == v2 and reg.version_of("previous_champion") == v1
    assert reg.rollback() == v1
    assert reg.version_of("champion") == v1
    assert (workdir / "rb_models" / "champion.joblib").exists()
    assert reg.load("champion").version == v1


def test_no_drift_means_no_action(tables, registry, workdir):
    ref_end = tables["telemetry"]["timestamp"].max() - pd.Timedelta(days=10)
    out = cycle.drift_cycle(tables, registry, reference_end=ref_end, logs_dir=workdir / "logs")
    assert out["action"] == "none" and not out["drift"]["drift_detected"]


def test_drift_triggers_retraining_and_a_logged_decision(workdir):
    tb = generate(SyntheticConfig(n_machines=24, days=110, seed=5))
    start = pd.Timestamp("2025-01-01")
    early = {k: (v[v.timestamp <= start + pd.Timedelta(days=55)] if "timestamp" in v else v) for k, v in tb.items()}
    reg = ModelRegistry(f"sqlite:///{workdir}/cycle.db", models_dir=workdir / "cycle_models")
    first = cycle.train_initial(early, reg)
    assert reg.version_of("champion") == first["version"]

    drifted = apply_drift(tb, start + pd.Timedelta(days=60))
    out = cycle.drift_cycle(drifted, reg, reference_end=start + pd.Timedelta(days=55), logs_dir=workdir / "cycle_logs")
    assert out["drift"]["drift_detected"]
    assert out["action"] in ("promoted", "refused")
    # Whatever the outcome, the decision is explained and journalled, and the champion is consistent with it.
    assert out["decision"].reason
    log = (workdir / "cycle_logs" / "promotion_decisions.jsonl").read_text().strip().splitlines()
    assert len(log) == 1
    expected = out["challenger_version"] if out["action"] == "promoted" else first["version"]
    assert reg.version_of("champion") == expected
