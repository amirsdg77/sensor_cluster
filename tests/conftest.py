"""Shared fixtures: a small synthetic dataset that *does* have density structure
(unlike the real sample), so unit tests can assert on cluster properties."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_SENSORS = 20


def _make_synthetic(
    n_per_cluster: int = 60,
    n_clusters: int = 3,
    n_noise: int = 20,
    n_labeled_per_cluster: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate well-separated Gaussian blobs in 20-D, scaled into [-1, 1].

    Returns a DataFrame with columns Sensor 0..Sensor 19 + Label, where Label
    is populated for `n_labeled_per_cluster` random members of each cluster.
    """
    rng = np.random.default_rng(seed)
    cluster_centers = rng.uniform(-0.5, 0.5, size=(n_clusters, N_SENSORS))
    rows = []
    labels = []
    cluster_ids = []
    for c in range(n_clusters):
        pts = cluster_centers[c] + rng.normal(0, 0.05, size=(n_per_cluster, N_SENSORS))
        rows.append(pts)
        labels.extend([float(c + 1)] * n_per_cluster)
        cluster_ids.extend([c] * n_per_cluster)
    # Add diffuse noise points
    rows.append(rng.uniform(-0.95, 0.95, size=(n_noise, N_SENSORS)))
    labels.extend([float("nan")] * n_noise)
    cluster_ids.extend([-1] * n_noise)

    X = np.vstack(rows)
    X = np.clip(X, -0.999, 0.999)

    df = pd.DataFrame(X, columns=[f"Sensor {i}" for i in range(N_SENSORS)])
    df["Label"] = labels
    df["_true_cluster"] = cluster_ids  # for tests, drop before passing to loader

    # Mask all but `n_labeled_per_cluster` labeled samples to NaN per class
    keep_mask = np.zeros(len(df), dtype=bool)
    for c in range(n_clusters):
        candidates = df.index[(df["Label"] == float(c + 1))].to_numpy()
        rng.shuffle(candidates)
        keep_mask[candidates[:n_labeled_per_cluster]] = True
    keep_mask |= df["Label"].isna().to_numpy()
    df.loc[~keep_mask, "Label"] = np.nan

    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic()


@pytest.fixture()
def synthetic_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    p = tmp_path / "synth.csv"
    synthetic_df.drop(columns=["_true_cluster"]).to_csv(p, index=False)
    return p


@pytest.fixture()
def tmp_artifacts(tmp_path: Path) -> Path:
    out = tmp_path / "artifacts"
    out.mkdir()
    return out


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Send MLflow runs to a per-test tmp dir so tests don't pollute ./mlruns."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    yield
