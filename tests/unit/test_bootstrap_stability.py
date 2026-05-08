"""Unit tests for `bootstrap_stability` independent of training."""

from __future__ import annotations

import math

import numpy as np

from sensorcluster.evaluation.metrics import bootstrap_stability


def _stable_fitter(seed: int = 0):
    """Return a clusterer that always assigns labels by row index modulo 3.

    Two runs over the same overlap will agree perfectly, so ARI must be 1.
    """

    def fit(X: np.ndarray) -> np.ndarray:
        # Use the first column's value as a stable per-row label so any overlap
        # between subsamples produces identical labels (deterministic).
        return np.asarray(np.round(X[:, 0]).astype(np.int64))

    return fit


def test_stability_perfect_agreement_when_fitter_is_index_dependent() -> None:
    rng = np.random.default_rng(0)
    X = np.zeros((100, 2))
    # Encode a deterministic per-row label in column 0 so the fitter recovers
    # it deterministically regardless of which subsample it sees.
    X[:, 0] = np.arange(100) % 3
    X[:, 1] = rng.normal(size=100)

    out = bootstrap_stability(X, _stable_fitter(), n_runs=5, sample_frac=0.7, seed=42)
    assert out["stability_mean_ari"] == 1.0
    assert out["n_pairs"] == 10  # 5 runs -> C(5,2) = 10 all-pairs


def test_stability_returns_nan_when_no_pair_overlaps_enough() -> None:
    """A 1-run call cannot form any pair and must report NaN."""
    X = np.zeros((50, 2))
    out = bootstrap_stability(X, _stable_fitter(), n_runs=1, sample_frac=0.5, seed=42)
    assert math.isnan(out["stability_mean_ari"])
    assert out["n_pairs"] == 0


def test_stability_deterministic_for_fixed_seed() -> None:
    """Two calls with identical seed must produce identical summaries."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))

    a = bootstrap_stability(X, _stable_fitter(), n_runs=4, sample_frac=0.6, seed=7)
    b = bootstrap_stability(X, _stable_fitter(), n_runs=4, sample_frac=0.6, seed=7)
    assert a == b
