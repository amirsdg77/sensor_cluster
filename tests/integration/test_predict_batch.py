"""Batch prediction integration test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sensorcluster.config import Settings
from sensorcluster.pipeline.predict_batch import predict_batch
from sensorcluster.pipeline.train import train


@pytest.mark.integration()
def test_predict_batch_round_trip(synthetic_csv: Path, tmp_artifacts: Path, tmp_path: Path) -> None:
    cfg = Settings(
        artifacts_dir=tmp_artifacts,
        data={"path": synthetic_csv, "sensor_min": -1.0, "sensor_max": 1.0},
        hdbscan={"min_cluster_size": 10, "min_samples": 3},
        evaluation={"cv_folds": 3, "bootstrap_n_runs": 3, "bootstrap_sample_frac": 0.8},
        mlflow={"enabled": False},
    )
    train(cfg)

    out = tmp_path / "preds.parquet"
    df = predict_batch(
        input_path=synthetic_csv,
        output_path=out,
        artifacts_dir=tmp_artifacts,
        sensor_min=-1.0,
        sensor_max=1.0,
    )
    assert out.exists()
    assert len(df) > 0
    assert {"predicted_label", "confidence", "novelty_score", "cluster_id", "is_novel"}.issubset(
        df.columns
    )
    # sanity: at least one CLASS_ assignment for synthetic blobs
    assert df["predicted_label"].str.startswith("CLASS_").any()


@pytest.mark.integration()
def test_predict_batch_csv_output(synthetic_csv: Path, tmp_artifacts: Path, tmp_path: Path) -> None:
    cfg = Settings(
        artifacts_dir=tmp_artifacts,
        data={"path": synthetic_csv, "sensor_min": -1.0, "sensor_max": 1.0},
        hdbscan={"min_cluster_size": 10, "min_samples": 3},
        evaluation={"cv_folds": 3, "bootstrap_n_runs": 3, "bootstrap_sample_frac": 0.8},
        mlflow={"enabled": False},
    )
    train(cfg)

    out = tmp_path / "preds.csv"
    predict_batch(
        input_path=synthetic_csv,
        output_path=out,
        artifacts_dir=tmp_artifacts,
        sensor_min=-1.0,
        sensor_max=1.0,
    )
    df = pd.read_csv(out)
    assert "predicted_label" in df.columns
