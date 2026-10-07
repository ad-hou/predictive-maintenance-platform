"""Operational cycles: scoring, drift check -> retrain -> compare -> promote or refuse."""

from __future__ import annotations

import pandas as pd

from src import config
from src.features.build_features import make_dataset
from src.monitoring.drift import drift_report
from src.registry.registry import Decision, ModelRegistry, decide, log_decision
from src.training import metrics
from src.training.explain import top_factors
from src.training.model import ChampionModel
from src.training.train import chronological_blocks, run_comparison, tail_blocks


def evaluate_on_window(model: ChampionModel, window: pd.DataFrame, costs=None) -> dict:
    return metrics.evaluate(window, model.predict_proba(window), model.threshold, costs)


def retrain_and_compare(
    tables: dict,
    registry: ModelRegistry,
    blocks_fn=None,
    seed: int = 42,
    trigger: str = "schedule",
    logs_dir=None,
    extra: dict | None = None,
) -> dict:
    """Train a challenger, compare it with the current champion on the same unseen window
    (the `test` block), promote only if the total cost drops enough. The very first model is promoted."""
    df = make_dataset(tables)
    blocks = blocks_fn(df) if blocks_fn else None
    cand = run_comparison(df, seed=seed, blocks=blocks)
    challenger = cand["model"]
    window = (blocks or chronological_blocks(df))["test"]

    version = registry.register_challenger(
        challenger, cand["final_test"], params={"seed": seed, "horizon_h": config.HORIZON_HOURS}, tags={"trigger": trigger}
    )
    champion_version = registry.version_of("champion")
    champion = registry.load("champion") if champion_version and champion_version != version else None
    challenger_eval = evaluate_on_window(challenger, window)
    champion_eval = evaluate_on_window(champion, window) if champion else None
    promoted, reason = decide(champion_eval["total_cost"] if champion_eval else None, challenger_eval["total_cost"])
    decision = Decision(
        promoted=promoted,
        reason=reason,
        champion_version=champion_version if champion else None,
        challenger_version=version,
        champion_cost=champion_eval["total_cost"] if champion_eval else None,
        challenger_cost=challenger_eval["total_cost"],
        window=f"{window['timestamp'].min()} -> {window['timestamp'].max()}",
    )
    if promoted:
        registry.promote(version)
    log_decision(decision, {"trigger": trigger, **(extra or {})}, logs_dir)
    return {
        "action": "promoted" if promoted else "refused",
        "decision": decision,
        "challenger_version": version,
        "champion_eval": champion_eval,
        "challenger_eval": challenger_eval,
        "candidate": cand,
        "version": version,
        "final_test": cand["final_test"],
    }


def train_initial(tables: dict, registry: ModelRegistry, seed: int = 42, logs_dir=None) -> dict:
    """Train on everything available; promoted if there is no champion yet, otherwise compared with it."""
    return retrain_and_compare(tables, registry, seed=seed, trigger="initial", logs_dir=logs_dir)


def score_latest(model: ChampionModel, tables: dict, with_factors: bool = True) -> pd.DataFrame:
    """Risk of every machine at its most recent observation."""
    feats = make_dataset(tables)
    last = feats.sort_values("timestamp").groupby("machine_id").tail(1).reset_index(drop=True)
    p = model.predict_proba(last)
    out = pd.DataFrame(
        {
            "machine_id": last["machine_id"],
            "timestamp": last["timestamp"],
            "failure_probability": p,
            "risk_level": [model.risk_level(x) for x in p],
            "model_version": model.version,
        }
    )
    if with_factors:
        out["top_factor"] = [(lambda f: f[0]["label"] if f else "")(top_factors(model, last.iloc[[i]], 1)) for i in range(len(last))]
    return out.sort_values("failure_probability", ascending=False).reset_index(drop=True)


def drift_cycle(
    tables: dict,
    registry: ModelRegistry,
    reference_end,
    current_days: int = 7,
    eval_days: int = 14,
    calibration_days: int = 10,
    threshold_days: int = 10,
    seed: int = 42,
    logs_dir=None,
) -> dict:
    """1) detect drift  2) if any: retrain a challenger on recent data  3) compare with the champion
    on the same unseen window  4) promote only if the total cost drops enough; otherwise keep the champion."""
    tel = tables["telemetry"]
    end = tel["timestamp"].max()
    reference = tel[tel["timestamp"] <= pd.Timestamp(reference_end)]
    current = tel[tel["timestamp"] > end - pd.Timedelta(days=current_days)]
    drift = drift_report(reference, current)
    result: dict = {"drift": drift, "action": "none"}
    if not drift["drift_detected"]:
        return result

    out = retrain_and_compare(
        tables,
        registry,
        blocks_fn=lambda df: tail_blocks(df, eval_days, calibration_days, threshold_days),
        seed=seed,
        trigger="drift",
        logs_dir=logs_dir,
        extra={"drifted_variables": drift["drifted_variables"]},
    )
    result.update(out)
    return result
