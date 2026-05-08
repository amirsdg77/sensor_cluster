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
