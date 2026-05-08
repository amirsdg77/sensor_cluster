"""Evaluation metrics, grouped by what they require:

- Internal  (label-free) : silhouette, Davies-Bouldin, noise fraction.
- External  (need labels): ARI on the labeled subset, per-cluster purity.
- Stability (label-free) : bootstrap-resample, fit, compare runs via ARI.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


@dataclass(frozen=True)
class InternalMetrics:
    """Label-free quality summary for a clustering.

    Attributes:
        silhouette: Mean silhouette over non-noise points (NaN if <2 clusters).
        davies_bouldin: Davies-Bouldin index over non-noise points (NaN if <2).
        noise_fraction: Share of points assigned to the noise group.
        n_clusters: Number of clusters discovered, excluding noise.
        n_points: Total number of points scored.
    """

    silhouette: float
    davies_bouldin: float
    noise_fraction: float
    n_clusters: int
    n_points: int


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> InternalMetrics:
    """Internal cluster-quality metrics. Silhouette/DB computed on non-noise points only."""
    labels = np.asarray(labels)
    n_points = len(labels)
    noise_mask = labels == -1
    noise_fraction = float(noise_mask.mean()) if n_points else 0.0

    non_noise_labels = labels[~noise_mask]
    n_clusters = len(set(non_noise_labels.tolist()))

    if n_clusters >= 2:
        silhouette = float(silhouette_score(X[~noise_mask], non_noise_labels))
        db = float(davies_bouldin_score(X[~noise_mask], non_noise_labels))
    else:
        silhouette = float("nan")
        db = float("nan")

    return InternalMetrics(
        silhouette=silhouette,
        davies_bouldin=db,
        noise_fraction=noise_fraction,
        n_clusters=n_clusters,
        n_points=n_points,
    )


def labeled_ari(predicted_label_ids: np.ndarray, y_true: pd.Series) -> float:
    """ARI between predicted labels (string names mapped to ints) and ground truth.

    Only the labeled positions count. predicted_label_ids should already be
    integer-encoded (e.g. via pandas.factorize or a consistent string->int map).
    """
    y = y_true.to_numpy()
    mask = ~pd.isna(y)
    if mask.sum() == 0:
        return float("nan")
    return float(adjusted_rand_score(y[mask].astype(float), predicted_label_ids[mask]))


def cluster_purity(cluster_labels: np.ndarray, y_true: pd.Series) -> dict[int, float]:
    """Per-cluster purity = max class count / total labeled points in that cluster.

    Clusters with no labeled members are absent from the result dict.
    """
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    y = y_true.to_numpy()
    mask = ~pd.isna(y)

    purity: dict[int, float] = {}
    for cid in set(cluster_labels[mask].tolist()):
        in_cluster = (cluster_labels == cid) & mask
        if not in_cluster.any():
            continue
        counts = Counter(y[in_cluster].tolist())
        purity[int(cid)] = max(counts.values()) / sum(counts.values())
    return purity


def bootstrap_stability(
    X: np.ndarray,
    fit_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_runs: int = 30,
    sample_frac: float = 0.8,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate clustering stability via repeated subsample-and-refit.

    Each run draws a random subsample of `X`, fits a fresh clustering with
    `fit_fn`, and records the resulting labels. All C(n_runs, 2) pairs are
    compared via ARI on the indices they share, and the mean / std of those
    pairwise ARIs is returned. A higher mean indicates the clustering is
    robust to which points the model sees.

    Returns a dict with ``stability_mean_ari``, ``stability_std_ari``, and
    ``n_pairs`` (the number of run pairs with enough overlap to score).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sample_size = max(2, int(sample_frac * n))

    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_runs):
        idx = rng.choice(n, size=sample_size, replace=False)
        labels = fit_fn(X[idx])
        runs.append((idx, np.asarray(labels)))

    aris: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            i1, l1 = runs[i]
            i2, l2 = runs[j]
            common, p1, p2 = np.intersect1d(i1, i2, return_indices=True)
            if len(common) < 2:
                continue
            aris.append(float(adjusted_rand_score(l1[p1], l2[p2])))

    if not aris:
        # No pair had enough overlap to score; report NaN explicitly rather
        # than letting numpy emit a "Mean of empty slice" warning.
        return {
            "stability_mean_ari": float("nan"),
            "stability_std_ari": float("nan"),
            "n_pairs": 0,
        }
    arr = np.array(aris)
    return {
        "stability_mean_ari": float(arr.mean()),
        "stability_std_ari": float(arr.std()),
        "n_pairs": len(arr),
    }
