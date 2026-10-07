"""Write training reports and the dashboard snapshot (everything the demo shows, no AWS needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.features.build_features import make_dataset
from src.monitoring import data_quality
from src.monitoring.drift import drift_report, psi_timeline
from src.registry.cycle import score_latest
from src.training.model import ChampionModel


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_run(result: dict, out_dir: Path | None = None) -> Path:
    """Reports of one training run: model comparison, calibration, cost curve, summary."""
    d = Path(out_dir or config.REPORTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    result["test_report"].to_csv(d / "comparison.csv", index=False)
    result["ranking_calibration"].to_csv(d / "ranking_calibration.csv", index=False)
    result["calibration_table"].to_csv(d / "calibration.csv", index=False)
    result["cost_curve"].to_csv(d / "cost_curve.csv", index=False)
    model: ChampionModel = result["model"]
    _json(
        d / "summary.json",
        {
            "model": model.name,
            "version": model.version,
            "threshold": model.threshold,
            "blocks": result["blocks"],
            "final_test": result["final_test"],
            "periods": model.meta,
            "costs_are_hypothetical": True,
            "costs": {
                "missed_failure": config.COSTS.missed_failure,
                "useless_inspection": config.COSTS.useless_inspection,
                "useful_inspection": config.COSTS.useful_inspection,
            },
        },
    )
    return d


def export_demo(
    tables: dict, model: ChampionModel, run: dict, reference_end, out_dir: Path | None = None, decisions_log: Path | None = None
) -> Path:
    """Static snapshot read by the Streamlit dashboard when no live API / database is reachable."""
    d = Path(out_dir or config.DEMO_DIR)
    d.mkdir(parents=True, exist_ok=True)
    write_run(run, d)

    risk = score_latest(model, tables)
    risk.to_csv(d / "risk_table.csv", index=False)

    feats = make_dataset(tables)
    feats["failure_probability"] = model.predict_proba(feats)
    last = feats["timestamp"].max() - pd.Timedelta(days=7)
    hist = feats[feats["timestamp"] > last][
        ["machine_id", "timestamp", "voltage", "rotation", "pressure", "vibration", "failure_probability"]
    ]
    hist.round({"voltage": 3, "rotation": 3, "pressure": 3, "vibration": 3, "failure_probability": 5}).to_csv(
        d / "history.csv", index=False
    )
    fl = tables["failures"]
    fl[fl["timestamp"] > feats["timestamp"].max() - pd.Timedelta(days=30)].to_csv(d / "failures.csv", index=False)
    tables["machines"].to_csv(d / "machines.csv", index=False)

    tel = tables["telemetry"]
    ref = tel[tel["timestamp"] <= pd.Timestamp(reference_end)]
    cur = tel[tel["timestamp"] > tel["timestamp"].max() - pd.Timedelta(days=7)]
    rep = drift_report(ref, cur)
    rep["table"].to_csv(d / "drift.csv", index=False)
    psi_timeline(tel, reference_end).to_csv(d / "drift_timeline.csv", index=False)

    decisions = []
    log = Path(decisions_log or config.LOGS_DIR / "promotion_decisions.jsonl")
    if log.exists():
        decisions = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    _json(d / "decisions.json", decisions)
    _json(
        d / "overview.json",
        {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "synthetic_data": True,
            "model_name": model.name,
            "model_version": model.version,
            "threshold": model.threshold,
            "data": data_quality.summary(tables),
            "quality_problems": data_quality.check_tables(tables),
            "drift_detected": rep["drift_detected"],
            "drifted_variables": rep["drifted_variables"],
            "high_risk": int((risk["risk_level"] == "HIGH").sum()),
            "medium_risk": int((risk["risk_level"] == "MEDIUM").sum()),
        },
    )
    return d
