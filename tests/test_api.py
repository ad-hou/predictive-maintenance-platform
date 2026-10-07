import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.model_loader import ModelHolder
from api.store import Store


@pytest.fixture(scope="module")
def client(registry, data_dir, workdir):
    app = create_app(Store(source="csv", directory=data_dir), ModelHolder(workdir / "models" / "champion.joblib"))
    with TestClient(app) as c:
        yield c


def body(machine="M001", **over):
    return {"machine_id": machine, "voltage": 170, "rotation": 450, "pressure": 100, "vibration": 40, **over}


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_version"] and r["machines"] == 24


def test_predict_returns_calibrated_probability(client):
    r = client.post("/predict", json=body())
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["failure_probability"] <= 1 and j["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert j["model_version"] == client.get("/health").json()["model_version"]


def test_online_prediction_matches_the_batch_score(client):
    """Replaying a stored reading through /predict gives the same probability as the batch path."""
    store = client.app.state.store
    tel = store.by_machine["M001"]["telemetry"]
    row = tel.iloc[-30]
    history = {h["timestamp"]: h["failure_probability"] for h in client.get("/machines/M001", params={"hours": 200}).json()["history"]}
    online = client.post(
        "/predict",
        json=body(
            timestamp=row["timestamp"].isoformat(),
            voltage=row["voltage"],
            rotation=row["rotation"],
            pressure=row["pressure"],
            vibration=row["vibration"],
        ),
    ).json()["failure_probability"]
    batch = next(v for k, v in history.items() if k.startswith(row["timestamp"].isoformat()[:19]))
    assert online == pytest.approx(batch, abs=1e-9)


def test_invalid_input_is_rejected_with_422(client):
    assert client.post("/predict", json=body(vibration=-5)).status_code == 422
    assert client.post("/predict", json={"machine_id": "M001"}).status_code == 422
    assert client.post("/predict", json=body(vibration="abc")).status_code == 422


def test_unknown_machine_is_404(client):
    assert client.post("/predict", json=body(machine="NOPE")).status_code == 404
    assert client.get("/machines/NOPE").status_code == 404
    assert client.get("/machines/NOPE/explain").status_code == 404


def test_machines_are_sorted_by_risk(client):
    rows = client.get("/machines").json()
    assert len(rows) == 24
    probs = [r["failure_probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    high = client.get("/machines", params={"level": "HIGH"}).json()
    assert all(r["risk_level"] == "HIGH" for r in high)


def test_machine_detail_and_explanation(client):
    d = client.get("/machines/M001").json()
    assert d["machine_id"] == "M001" and len(d["history"]) > 10 and "vibration" in d["latest_measurements"]
    ex = client.get("/machines/M001/explain").json()
    assert isinstance(ex, list) and all({"feature", "label", "contribution"} <= set(x) for x in ex)


def test_service_without_model_answers_503(data_dir, tmp_path):
    app = create_app(Store(source="csv", directory=data_dir), ModelHolder(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=body()).status_code == 503


def test_admin_reload_requires_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-API-Key": "secret"}).status_code == 200


def test_promotion_is_picked_up_without_restart(registry, data_dir, trained, workdir):
    holder = ModelHolder(workdir / "models" / "champion.joblib")
    app = create_app(Store(source="csv", directory=data_dir), holder)
    with TestClient(app) as c:
        before = c.get("/health").json()["model_version"]
        v = registry.register_challenger(trained["model"], {"total_cost": 0.0})
        registry.promote(v)
        after = c.get("/health").json()["model_version"]
        assert after == v != before
        registry.rollback()


def test_s3_sync_with_a_fake_client(tmp_path):
    from api import s3_sync

    calls = []

    class Fake:
        def download_file(self, bucket, key, dest):
            calls.append(("down", bucket, key))
            if key.endswith("missing.joblib"):
                raise RuntimeError("404")
            open(dest, "wb").write(b"x")

        def upload_file(self, src, bucket, key):
            calls.append(("up", bucket, key))

    assert s3_sync.parse_uri("s3://my-bucket/models/champion.joblib") == ("my-bucket", "models/champion.joblib")
    with pytest.raises(ValueError):
        s3_sync.parse_uri("http://nope")
    assert s3_sync.download("s3://b/models/champion.joblib", tmp_path / "c.joblib", Fake())
    assert not s3_sync.download("s3://b/models/missing.joblib", tmp_path / "m.joblib", Fake())
    s3_sync.upload(tmp_path / "c.joblib", "s3://b/models/champion.joblib", Fake())
    assert ("up", "b", "models/champion.joblib") in calls
