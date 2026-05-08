"""Performance regression test for vectorized batch prediction.

Predict was historically a per-row Python loop with a per-row k-NN call.
The vectorized rewrite must score 1000 synthetic rows in well under a second
on commodity hardware. The threshold is generous (5 s) so we're testing
"is the implementation still vectorized" rather than micro-benchmarking.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensorcluster.config import Settings
from sensorcluster.pipeline.train import train


@pytest.mark.integration()
def test_predict_1000_rows_is_fast(synthetic_csv: Path, tmp_artifacts: Path) -> None:
    cfg = Settings(
        artifacts_dir=tmp_artifacts,
        data={"path": synthetic_csv, "sensor_min": -1.0, "sensor_max": 1.0},
        hdbscan={"min_cluster_size": 10, "min_samples": 3},
        evaluation={"cv_folds": 3, "bootstrap_n_runs": 3, "bootstrap_sample_frac": 0.8},
        mlflow={"enabled": False},
    )
    pipeline = train(cfg).pipeline

    rng = np.random.default_rng(0)
    rows = rng.uniform(-0.9, 0.9, size=(1000, 20))
    X = pd.DataFrame(rows, columns=[f"Sensor {i}" for i in range(20)])

    # Warm any lazy imports / caches.
    pipeline.predict(X.iloc[:1])

    start = time.perf_counter()
    results = pipeline.predict(X)
    elapsed = time.perf_counter() - start

    assert len(results) == 1000
    assert elapsed < 5.0, f"predict(1000 rows) took {elapsed:.2f}s — vectorization regressed"
