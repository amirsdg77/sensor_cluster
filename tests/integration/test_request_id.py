"""X-Request-ID middleware: echo client value, otherwise mint a uuid4."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sensorcluster.api.main import REQUEST_ID_HEADER, create_app
from sensorcluster.config import Settings
from sensorcluster.pipeline.train import train

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@pytest.fixture()
def client(synthetic_csv: Path, tmp_artifacts: Path, monkeypatch: pytest.MonkeyPatch):
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
    with TestClient(app) as c:
        yield c


@pytest.mark.integration()
def test_response_carries_minted_request_id_when_client_omits_one(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    rid = r.headers.get(REQUEST_ID_HEADER)
    assert rid is not None and UUID_RE.match(rid), rid


@pytest.mark.integration()
def test_response_echoes_valid_client_supplied_request_id(client: TestClient) -> None:
    incoming = "12345678-1234-1234-1234-123456789012"
    r = client.get("/health", headers={REQUEST_ID_HEADER: incoming})
    assert r.headers[REQUEST_ID_HEADER] == incoming


@pytest.mark.integration()
def test_invalid_client_request_id_is_replaced_with_minted_uuid(client: TestClient) -> None:
    r = client.get("/health", headers={REQUEST_ID_HEADER: "not-a-uuid"})
    rid = r.headers[REQUEST_ID_HEADER]
    assert UUID_RE.match(rid) and rid != "not-a-uuid"
