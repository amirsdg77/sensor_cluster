"""HDBSCAN wrapper tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensorcluster.models.hdbscan_model import HDBSCANModel


@pytest.fixture()
def blob_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10], [-10, 5]])
    return np.vstack([c + rng.normal(0, 0.4, size=(40, 2)) for c in centers])


def test_fit_finds_blobs(blob_data: np.ndarray) -> None:
    m = HDBSCANModel(min_cluster_size=10, min_samples=3).fit(blob_data)
    assert m.n_clusters_ == 3
    assert (m.labels_ != -1).sum() > 100  # most points clustered


def test_predict_with_strength_returns_aligned_arrays(blob_data: np.ndarray) -> None:
    m = HDBSCANModel(min_cluster_size=10, min_samples=3, prediction_data=True).fit(blob_data)
    new = blob_data[:5] + np.random.default_rng(1).normal(0, 0.1, size=(5, 2))
    labels, strengths = m.predict_with_strength(new)
    assert labels.shape == (5,)
    assert strengths.shape == (5,)
    assert ((strengths >= 0) & (strengths <= 1)).all()


def test_novelty_score_high_for_far_points(blob_data: np.ndarray) -> None:
    m = HDBSCANModel(min_cluster_size=10, min_samples=3, prediction_data=True).fit(blob_data)
    far = np.array([[1000.0, 1000.0]])
    score = m.novelty_score(far)
    assert score[0] >= 0.5  # far points get high novelty


def test_novelty_score_matches_training_glosh_within_tolerance(blob_data: np.ndarray) -> None:
    """Inference-time novelty must approximate training-time GLOSH on the training set.

    This is the parity guarantee that the README + model_card promise: the score
    returned for new points uses the same GLOSH machinery as the training-time
    outlier_scores_, so a threshold tuned on training generalizes at inference.
    """
    m = HDBSCANModel(min_cluster_size=10, min_samples=3, prediction_data=True).fit(blob_data)
    inference_scores = m.novelty_score(blob_data)
    training_scores = m.training_outlier_scores_

    # The two are not bit-identical (approximate_predict_scores uses the
    # condensed tree built for new-point prediction), but they should agree
    # closely on which points are outliers.
    rho = np.corrcoef(inference_scores, training_scores)[0, 1]
    assert rho > 0.7, f"inference vs training GLOSH correlation {rho:.3f} below 0.7"


def test_novelty_score_floors_noise_to_one(blob_data: np.ndarray) -> None:
    """Points predicted as cluster -1 always get novelty_score == 1.0."""
    m = HDBSCANModel(min_cluster_size=10, min_samples=3, prediction_data=True).fit(blob_data)
    # A point very far from every blob is overwhelmingly likely to be predicted as noise.
    far = np.array([[5000.0, -5000.0]])
    labels, _ = m.predict_with_strength(far)
    if labels[0] == -1:
        assert m.novelty_score(far)[0] == 1.0


def test_save_load_roundtrip(blob_data: np.ndarray, tmp_path: Path) -> None:
    m = HDBSCANModel(min_cluster_size=10, min_samples=3, prediction_data=True).fit(blob_data)
    p = tmp_path / "hdb.joblib"
    m.save(p)
    m2 = HDBSCANModel.load(p)
    np.testing.assert_array_equal(m.labels_, m2.labels_)


def test_unfitted_model_raises_on_predict() -> None:
    m = HDBSCANModel()
    with pytest.raises(RuntimeError):
        m.predict_with_strength(np.zeros((1, 2)))
