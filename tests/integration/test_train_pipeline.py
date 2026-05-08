"""End-to-end training: synthetic blob data -> trained pipeline -> ARI > 0.6."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensorcluster.config import Settings
from sensorcluster.config import load_settings as _ls  # noqa: F401  (re-export check)
from sensorcluster.pipeline.train import train


@pytest.mark.integration()
def test_train_on_synthetic_data_produces_artifacts_and_high_ari(
    synthetic_csv: Path, tmp_artifacts: Path
) -> None:
    cfg = Settings(
        artifacts_dir=tmp_artifacts,
        data={
            "path": synthetic_csv,
            "sensor_min": -1.0,
            "sensor_max": 1.0,
        },
        hdbscan={"min_cluster_size": 10, "min_samples": 3},
        evaluation={"cv_folds": 3, "bootstrap_n_runs": 5, "bootstrap_sample_frac": 0.8},
        mlflow={"enabled": False},
    )

    result = train(cfg)

    # Artifacts exist
    for fname in [
        "scaler.joblib",
        "pca.joblib",
        "hdbscan.joblib",
        "cluster_label_map.json",
        "evaluation_report.md",
        "model_card.md",
    ]:
        assert (tmp_artifacts / fname).exists(), f"missing {fname}"

    # Cluster discovery worked
    assert result.internal.n_clusters >= 2
    # CV ARI on synthetic blobs should be well above chance
    assert result.cv is not None
    assert result.cv.mean_ari > 0.5


@pytest.mark.integration()
def test_train_then_load_pipeline_round_trip(synthetic_csv: Path, tmp_artifacts: Path) -> None:
    cfg = Settings(
        artifacts_dir=tmp_artifacts,
        data={"path": synthetic_csv, "sensor_min": -1.0, "sensor_max": 1.0},
        hdbscan={"min_cluster_size": 10, "min_samples": 3},
        evaluation={"cv_folds": 3, "bootstrap_n_runs": 3, "bootstrap_sample_frac": 0.8},
        mlflow={"enabled": False},
    )
    res = train(cfg)

    from sensorcluster.models.pipeline_model import SensorClusterPipeline

    loaded = SensorClusterPipeline.load(tmp_artifacts)
    assert loaded.label_map.entries.keys() == res.label_map.entries.keys()
