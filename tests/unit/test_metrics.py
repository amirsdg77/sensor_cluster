"""Metrics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorcluster.evaluation.metrics import (
    cluster_purity,
    compute_internal_metrics,
    labeled_ari,
)


def test_compute_internal_metrics_handles_all_noise() -> None:
    X = np.random.default_rng(0).normal(size=(20, 3))
    labels = np.full(20, -1, dtype=int)
    out = compute_internal_metrics(X, labels)
    assert out.n_clusters == 0
    assert out.noise_fraction == 1.0
    assert np.isnan(out.silhouette)


def test_compute_internal_metrics_two_blobs() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, size=(20, 2)), rng.normal(5, 0.1, size=(20, 2))])
    labels = np.array([0] * 20 + [1] * 20)
    out = compute_internal_metrics(X, labels)
    assert out.n_clusters == 2
    assert out.silhouette > 0.5
    assert out.noise_fraction == 0.0


def test_cluster_purity_perfect_assignment() -> None:
    cluster_labels = np.array([0, 0, 0, 1, 1])
    y = pd.Series([1.0, 1.0, np.nan, 2.0, 2.0])
    p = cluster_purity(cluster_labels, y)
    assert p == {0: 1.0, 1: 1.0}


def test_labeled_ari_perfect_match() -> None:
    pred = np.array([1, 1, 2, 2])
    truth = pd.Series([1.0, 1.0, 2.0, 2.0])
    assert labeled_ari(pred, truth) == 1.0


def test_labeled_ari_ignores_unlabeled() -> None:
    pred = np.array([1, 1, 99, 2])
    truth = pd.Series([1.0, 1.0, np.nan, 2.0])
    # unlabeled index is masked out -> should still be perfect on labeled subset
    assert labeled_ari(pred, truth) == 1.0
