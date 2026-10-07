"""Command line used by Airflow, the Makefile and by hand: `python -m src.cli <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src import config


def _registry():
    from src.registry.registry import ModelRegistry

    return ModelRegistry()


def cmd_generate(a):
    from src.ingestion.synthetic import SyntheticConfig, generate, save

    tables = generate(SyntheticConfig(n_machines=a.machines, days=a.days, seed=a.seed))
    save(tables, a.out)
    print(f"wrote {sum(len(t) for t in tables.values())} rows to {a.out}")


def cmd_quality(a):
    from src.ingestion.reader import read_tables
    from src.monitoring.data_quality import check_structure, check_tables

    if a.structure_only:
        problems = check_structure(config.DATA_DIR)
    else:
        problems = check_tables(read_tables(as_of=a.as_of), max_age_hours=a.max_age_hours)
    for p in problems:
        print("PROBLEM:", p)
    print("data quality: OK" if not problems else f"data quality: {len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_load_postgres(a):
    from src.ingestion import load_postgres

    print(load_postgres.load(Path(a.dir) if a.dir else None))


def cmd_train(a):
    from src.ingestion.reader import read_tables
    from src.monitoring.data_quality import check_tables
    from src.registry.cycle import retrain_and_compare
    from src.reports import write_run

    tables = read_tables(as_of=a.as_of)
    problems = check_tables(tables)
    if problems:
        print("refusing to train on bad data:", *problems, sep="\n  ")
        return 1
    out = retrain_and_compare(tables, _registry(), seed=a.seed, trigger=a.trigger)
    write_run(out["candidate"])
    d = out["decision"]
    print(f"challenger v{d.challenger_version}: {d.reason}")
    print(
        json.dumps(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in out["final_test"].items() if not isinstance(v, dict)}, indent=2
        )
    )


def cmd_predict(a):
    from src.ingestion.reader import read_tables
    from src.registry.cycle import score_latest

    model = _registry().load("champion")
    scores = score_latest(model, read_tables(as_of=a.as_of))
    run_date = a.run_date or str(scores["timestamp"].max().date())
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(config.REPORTS_DIR / "predictions_latest.csv", index=False)
    if config.DATA_SOURCE == "postgres":
        from src.ingestion.load_postgres import publish_predictions

        print(f"published {publish_predictions(scores, run_date)} predictions for {run_date}")
    print(scores.head(10).to_string(index=False))


def cmd_drift_cycle(a):
    from src.ingestion.reader import read_tables
    from src.registry.cycle import drift_cycle

    out = drift_cycle(read_tables(as_of=a.as_of), _registry(), reference_end=a.reference_end)
    print(out["drift"]["table"].to_string(index=False))
    print("drift detected:", out["drift"]["drift_detected"], "| action:", out["action"])
    if out["action"] != "none":
        print(out["decision"].reason)


def cmd_rollback(a):
    print("champion is now version", _registry().rollback())


def cmd_demo(a):
    """Whole story on synthetic data: train v1, inject drift, detect, retrain, compare, decide."""
    import shutil

    from src.ingestion.synthetic import SyntheticConfig, apply_drift, generate, save
    from src.registry.cycle import drift_cycle, train_initial
    from src.registry.registry import ModelRegistry
    from src.reports import export_demo

    work = Path(a.workdir)
    for leftover in ("mlflow.db", "models", "logs", "raw", "raw_drifted"):  # a demo run always starts clean
        target = work / leftover
        shutil.rmtree(target) if target.is_dir() else target.unlink(missing_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2025-01-01")
    full = generate(SyntheticConfig(n_machines=a.machines, days=a.days, seed=a.seed))
    save(full, work / "raw")
    t1 = start + pd.Timedelta(days=a.train_day)
    early = {k: (v[v["timestamp"] <= t1] if "timestamp" in v else v) for k, v in full.items()}
    reg = ModelRegistry(f"sqlite:///{work / 'mlflow.db'}", models_dir=work / "models")
    logs = work / "logs"
    print(f"[1/4] training v1 on data up to day {a.train_day}")
    first = train_initial(early, reg, logs_dir=logs)
    print("      champion:", reg.version_of("champion"), first["candidate"]["model"].name)

    drifted = apply_drift(full, start + pd.Timedelta(days=a.drift_day))
    save(drifted, work / "raw_drifted")
    print(f"[2/4] drift injected from day {a.drift_day}; checking at day {a.days}")
    out = drift_cycle(drifted, reg, reference_end=t1, logs_dir=logs)
    print(out["drift"]["table"].to_string(index=False))
    print(f"[3/4] action: {out['action']}")
    if out["action"] != "none":
        print("      ", out["decision"].reason)
        print(
            f"       champion cost {out['decision'].champion_cost:,.0f} EUR vs challenger {out['decision'].challenger_cost:,.0f} EUR (hypothetical costs)"
        )
    champion = reg.load("champion")
    run = out["candidate"] if out["action"] == "promoted" else first["candidate"]
    print("[4/4] exporting dashboard snapshot ->", config.DEMO_DIR)
    export_demo(drifted, champion, run, reference_end=t1, decisions_log=logs / "promotion_decisions.jsonl")
    print("done. champion version:", champion.version)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("generate-data")
    s.set_defaults(fn=cmd_generate)
    s.add_argument("--machines", type=int, default=60)
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default=str(config.DATA_DIR))

    s = sub.add_parser("quality")
    s.set_defaults(fn=cmd_quality)
    s.add_argument("--as-of")
    s.add_argument("--max-age-hours", type=float)
    s.add_argument("--structure-only", action="store_true", help="only check files and columns (value checks belong to DBT)")

    s = sub.add_parser("load-postgres")
    s.set_defaults(fn=cmd_load_postgres)
    s.add_argument("--dir")

    s = sub.add_parser("train")
    s.set_defaults(fn=cmd_train)
    s.add_argument("--as-of")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--trigger", default="schedule")

    s = sub.add_parser("predict")
    s.set_defaults(fn=cmd_predict)
    s.add_argument("--as-of")
    s.add_argument("--run-date")

    s = sub.add_parser("drift-cycle")
    s.set_defaults(fn=cmd_drift_cycle)
    s.add_argument("--as-of")
    s.add_argument("--reference-end", required=True)

    s = sub.add_parser("rollback")
    s.set_defaults(fn=cmd_rollback)

    s = sub.add_parser("demo")
    s.set_defaults(fn=cmd_demo)
    s.add_argument("--workdir", default=str(config.ROOT / "reports" / "demo-run"))
    s.add_argument("--machines", type=int, default=60)
    s.add_argument("--days", type=int, default=150)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--train-day", type=int, default=78)
    s.add_argument("--drift-day", type=int, default=80)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
