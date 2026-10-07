"""FastAPI service: predictive maintenance scores and explanations."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException

from api.model_loader import ModelHolder
from api.schemas import Factor, Health, MachineRisk, Prediction, Reading
from api.store import Store
from src.features.serving import features_for_reading
from src.training.explain import top_factors


def create_app(store: Store | None = None, holder: ModelHolder | None = None) -> FastAPI:
    store = store or Store()
    holder = holder or ModelHolder()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            store.refresh()
        except Exception as exc:  # the API still starts: /health reports the problem
            app.state.startup_error = str(exc)
        if os.getenv("MODEL_S3_URI"):
            from api import s3_sync

            s3_sync.download(os.environ["MODEL_S3_URI"], holder.path)
        model = holder.get()
        if model is not None and store.loaded:
            scores(model)  # warm the cache so the first /machines call is fast
        yield

    app = FastAPI(title="Predictive Maintenance API", version="1.0.0", lifespan=lifespan)
    app.state.store, app.state.holder, app.state.startup_error = store, holder, None
    cache: dict = {"key": None, "last": None, "by_machine": None}

    def require_key(x_api_key: str | None = Header(default=None)):
        expected = os.getenv("API_KEY")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    def model_or_503():
        model = holder.get()
        if model is None:
            raise HTTPException(status_code=503, detail="no champion model available yet")
        return model

    score_lock = threading.Lock()

    def scores(model):
        """Probability for every stored observation, computed once per (model version, data load).
        The lock stops concurrent requests from all recomputing it at the same time."""
        key = (model.version, id(store.features))
        if cache["key"] != key:
            with score_lock:
                if cache["key"] != key:
                    f = store.features
                    s = f[["machine_id", "timestamp"]].assign(p=model.predict_proba(f))
                    cache["by_machine"] = {m: g.reset_index(drop=True) for m, g in s.groupby("machine_id", sort=False)}
                    cache["last"] = s.sort_values("timestamp").groupby("machine_id").tail(1).sort_values("p", ascending=False)
                    cache["key"] = key
        return cache

    @app.get("/health", response_model=Health)
    def health():
        model = holder.get()
        ok = model is not None and store.loaded
        return Health(
            status="ok" if ok else "degraded",
            model_version=model.version if model else None,
            model_name=model.name if model else None,
            model_loaded_at=holder.loaded_at,
            data_last_timestamp=store.last_timestamp(),
            machines=len(store.by_machine),
        )

    @app.post("/predict", response_model=Prediction)
    def predict(reading: Reading, explain: bool = True):
        model = model_or_503()
        hist = store.by_machine.get(reading.machine_id)
        if hist is None:
            raise HTTPException(status_code=404, detail=f"unknown machine {reading.machine_id}")
        tel = hist["telemetry"]
        ts = reading.timestamp or (tel["timestamp"].max() + pd.Timedelta(hours=1) if len(tel) else pd.Timestamp.now().floor("h"))
        payload = reading.model_dump()
        payload["timestamp"] = ts
        X = features_for_reading(tel, hist["errors"], hist["maintenance"], hist["failures"], hist["machine"], payload)
        p = float(model.predict_proba(X)[0])
        return Prediction(
            machine_id=reading.machine_id,
            timestamp=ts,
            failure_probability=p,
            risk_level=model.risk_level(p),
            model_version=model.version,
            threshold=model.threshold,
            top_factors=[Factor(**f) for f in top_factors(model, X, 3)] if explain else [],
        )

    @app.get("/machines", response_model=list[MachineRisk])
    def machines(limit: int = 200, level: str | None = None):
        model = model_or_503()
        last = scores(model)["last"]
        rows = []
        for r in last.itertuples():
            lvl = model.risk_level(r.p)
            if level and lvl != level.upper():
                continue
            rows.append(MachineRisk(machine_id=r.machine_id, timestamp=r.timestamp, failure_probability=float(r.p), risk_level=lvl))
        return rows[:limit]

    @app.get("/machines/{machine_id}")
    def machine(machine_id: str, hours: int = 72):
        model = model_or_503()
        if machine_id not in store.by_machine:
            raise HTTPException(status_code=404, detail=f"unknown machine {machine_id}")
        recent = scores(model)["by_machine"][machine_id].tail(hours)
        tel = store.by_machine[machine_id]["telemetry"].tail(hours)
        last = recent.iloc[-1]
        return {
            "machine_id": machine_id,
            "latest_timestamp": last["timestamp"],
            "failure_probability": float(last["p"]),
            "risk_level": model.risk_level(float(last["p"])),
            "model_version": model.version,
            "latest_measurements": tel.iloc[-1][["voltage", "rotation", "pressure", "vibration"]].to_dict(),
            "history": [{"timestamp": t, "failure_probability": float(p)} for t, p in zip(recent["timestamp"], recent["p"], strict=True)],
            "last_failure": store.by_machine[machine_id]["failures"]["timestamp"].max()
            if len(store.by_machine[machine_id]["failures"])
            else None,
        }

    @app.get("/machines/{machine_id}/explain", response_model=list[Factor])
    def explain(machine_id: str, k: int = 5):
        model = model_or_503()
        if machine_id not in store.by_machine:
            raise HTTPException(status_code=404, detail=f"unknown machine {machine_id}")
        f = store.features
        row = f[f["machine_id"] == machine_id].tail(1)
        return [Factor(**x) for x in top_factors(model, row, k)]

    @app.post("/admin/reload", dependencies=[Depends(require_key)])
    def reload():
        """Reload data (and the model file if it changed)."""
        store.refresh()
        cache["key"] = None
        holder.get()
        return {"status": "reloaded", "model_version": holder.model.version if holder.model else None}

    return app


app = create_app()
