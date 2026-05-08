"""Stratified K-fold CV must be deterministic for a fixed seed."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensorcluster.config import Settings
from sensorcluster.pipeline.train import train


@pytest.mark.integration()
def test_cv_fold_aris_match_for_identical_seed(synthetic_csv: Path, tmp_path: Path) -> None:
    """Two training runs with the same seed must produce the same per-fold ARIs."""

    def _train(artifacts_dir: Path):
        cfg = Settings(
            artifacts_dir=artifacts_dir,
            data={"path": synthetic_csv, "sensor_min": -1.0, "sensor_max": 1.0},
            hdbscan={"min_cluster_size": 10, "min_samples": 3},
            evaluation={"cv_folds": 3, "bootstrap_n_runs": 2, "bootstrap_sample_frac": 0.8},
            mlflow={"enabled": False},
            random_seed=123,
        )
        return train(cfg)

    a = _train(tmp_path / "a")
    b = _train(tmp_path / "b")

    assert a.cv is not None and b.cv is not None
    assert a.cv.fold_aris == b.cv.fold_aris
    assert a.cv.mean_ari == b.cv.mean_ari
