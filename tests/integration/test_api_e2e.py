"""FastAPI end-to-end via TestClient.

Trains on synthetic data, then probes /health, /predict, /predict_batch, /clusters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sensorcluster.api.main import create_app
from sensorcluster.config import Settings
from sensorcluster.pipeline.train import train


@pytest.fixture()
def trained_app(synthetic_csv: Path, tmp_artifacts: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = Settings(
        artifacts_dir=tmp_artifacts,
        data={"path": synthetic_csv, "sensor_min": -1.0, "sensor_max": 1.0},
        hdbscan={"min_cluster_size": 10, "min_samples": 3},
        evaluation={"cv_folds": 3, "bootstrap_n_runs": 3, "bootstrap_sample_frac": 0.8},
        mlflow={"enabled": False},
    )
    train(cfg)
    monkeypatch.setenv("SENSORCLUSTER_ARTIFACTS_DIR", str(tmp_artifacts))
    app = create_app()
    # Entering the TestClient as a context manager triggers the lifespan events
    # so the pipeline is loaded onto app.state before any requests fire.
    with TestClient(app) as client:
        yield client


@pytest.mark.integration()
def test_health_returns_ok(trained_app: TestClient) -> None:
    r = trained_app.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "trained_at" in body
    assert body["n_clusters"] >= 2


@pytest.mark.integration()
def test_predict_returns_well_formed_response(trained_app: TestClient) -> None:
    r = trained_app.post("/predict", json={"sensors": [0.0] * 20})
    assert r.status_code == 200
    body = r.json()
    assert "predicted_label" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["novelty_score"] <= 1.0


@pytest.mark.integration()
def test_predict_batch_round_trip(trained_app: TestClient) -> None:
    r = trained_app.post(
        "/predict_batch",
        json={"rows": [[0.1] * 20, [-0.2] * 20, [0.5] * 20]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3


@pytest.mark.integration()
def test_clusters_endpoint(trained_app: TestClient) -> None:
    r = trained_app.get("/clusters")
    assert r.status_code == 200
    body = r.json()
    assert "entries" in body and len(body["entries"]) >= 1


@pytest.mark.integration()
def test_predict_rejects_wrong_dimensionality(trained_app: TestClient) -> None:
    r = trained_app.post("/predict", json={"sensors": [0.0] * 5})  # too few
    assert r.status_code == 422


@pytest.mark.integration()
def test_healthz_does_not_depend_on_pipeline(trained_app: TestClient) -> None:
    """Liveness must succeed even if the pipeline isn't loaded."""
    r = trained_app.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok"}


@pytest.mark.integration()
def test_readyz_returns_model_identity(trained_app: TestClient) -> None:
    """Readiness exposes model identity for ops dashboards."""
    r = trained_app.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_version"]
    assert body["trained_at"]
    assert body["n_clusters"] >= 2


@pytest.mark.integration()
def test_version_endpoint_carries_schema_and_api_versions(trained_app: TestClient) -> None:
    r = trained_app.get("/version")
    assert r.status_code == 200
    body = r.json()
    # Schema and API versions follow semver and are non-empty.
    assert all(part.isdigit() for part in body["schema_version"].split("."))
    assert all(part.isdigit() for part in body["api_version"].split("."))


@pytest.mark.integration()
def test_predict_batch_stream_emits_one_ndjson_line_per_row(trained_app: TestClient) -> None:
    rows = [[0.1] * 20, [-0.2] * 20, [0.5] * 20]
    with trained_app.stream("POST", "/predict_batch/stream", json={"rows": rows}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        body = b"".join(resp.iter_bytes()).decode("utf-8")
    lines = [ln for ln in body.split("\n") if ln.strip()]
    assert len(lines) == len(rows)
    import json as _json

    for line in lines:
        payload = _json.loads(line)
        assert "predicted_label" in payload
        assert 0.0 <= payload["confidence"] <= 1.0
