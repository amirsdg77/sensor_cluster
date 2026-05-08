"""HDBSCAN clustering model.

Exposes hard cluster labels, per-point membership strength (used as
inference-time confidence), and GLOSH outlier scores (used as novelty).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hdbscan
import joblib
import numpy as np

from sensorcluster.models.base import ClusterModel


def _normalize_glosh(scores: np.ndarray) -> np.ndarray:
    """Clip raw GLOSH scores into [0, 1].

    GLOSH (Global-Local Outlier Score from Hierarchies) is defined in [0, 1]
    by the paper, but the hdbscan implementation can return values marginally
    above 1.0 on degenerate clusters. Clipping keeps downstream thresholds
    operating on a bounded score.
    """
    return np.clip(scores.astype(np.float64), 0.0, 1.0)


class HDBSCANModel(ClusterModel):
    """HDBSCAN with `prediction_data=True` so new points can be scored."""

    def __init__(
        self,
        *,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        prediction_data: bool = True,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.prediction_data = prediction_data
        self._model: hdbscan.HDBSCAN | None = None

    # ---- training ------------------------------------------------------------
    def fit(self, X: np.ndarray) -> HDBSCANModel:
        self._model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=self.prediction_data,
        )
        self._model.fit(X)
        return self

    # ---- inference -----------------------------------------------------------
    def _ensure_fitted(self) -> hdbscan.HDBSCAN:
        if self._model is None:
            raise RuntimeError("HDBSCANModel.fit must be called before inference")
        return self._model

    def score_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cluster ids, membership strengths, and GLOSH novelty in one pass.

        Calling this once per batch avoids two redundant traversals of the
        prediction tree that would happen if the caller invoked
        :meth:`predict_with_strength` and :meth:`novelty_score` separately.

        Returns:
            labels: int64 array of cluster ids; -1 for the noise group.
            strengths: float64 array in [0, 1]; HDBSCAN's membership strength.
            novelty: float64 array in [0, 1]; GLOSH outlier score with the
                noise group floored to 1.0.
        """
        model = self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        labels_raw, strengths_raw = hdbscan.approximate_predict(model, X)
        labels = np.asarray(labels_raw, dtype=np.int64)
        strengths = np.asarray(strengths_raw, dtype=np.float64)

        novelty = np.asarray(hdbscan.approximate_predict_scores(model, X), dtype=np.float64)
        # Degenerate near-empty inputs can yield NaN; treat as maximally novel.
        novelty = np.where(np.isnan(novelty), 1.0, novelty)
        novelty = np.where(labels == -1, 1.0, novelty)
        return labels, strengths, _normalize_glosh(novelty)

    def predict_with_strength(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cluster id and membership strength for each row of `X`.

        Returns labels in {-1, 0, 1, ...} and strengths in [0, 1]. Inputs are
        coerced to float64 because the prediction tree is built in float64;
        feeding float32 produces silent scoring drift.
        """
        labels, strengths, _ = self.score_batch(X)
        return labels, strengths

    def novelty_score(self, X: np.ndarray) -> np.ndarray:
        """Per-point GLOSH novelty score in [0, 1] for new points.

        Backed by ``hdbscan.approximate_predict_scores``, which evaluates the
        same GLOSH score exposed at training time but against the fitted
        clustering. Points assigned to noise are forced to 1.0 so the noise
        cluster always reads as maximally novel.
        """
        _, _, novelty = self.score_batch(X)
        return novelty

    # ---- training-time accessors --------------------------------------------
    @property
    def labels_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HDBSCANModel not fitted")
        return np.asarray(self._model.labels_, dtype=np.int64)

    @property
    def training_outlier_scores_(self) -> np.ndarray:
        """GLOSH outlier scores for the training points, clipped to [0, 1]."""
        if self._model is None:
            raise RuntimeError("HDBSCANModel not fitted")
        return _normalize_glosh(np.asarray(self._model.outlier_scores_))

    @property
    def training_probabilities_(self) -> np.ndarray:
        """Cluster-membership probability for each training point, in [0, 1]."""
        if self._model is None:
            raise RuntimeError("HDBSCANModel not fitted")
        return np.asarray(self._model.probabilities_, dtype=np.float64)

    @property
    def n_clusters_(self) -> int:
        """Number of clusters discovered at fit time, excluding the noise group."""
        labels = self.labels_
        return len(set(labels.tolist()) - {-1})

    # ---- persistence ---------------------------------------------------------
    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save unfitted HDBSCANModel")
        payload: dict[str, Any] = {
            "model": self._model,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "prediction_data": self.prediction_data,
        }
        joblib.dump(payload, Path(path), compress=3)

    @classmethod
    def load(cls, path: Path | str) -> HDBSCANModel:
        payload: dict[str, Any] = joblib.load(Path(path))
        obj = cls(
            min_cluster_size=payload["min_cluster_size"],
            min_samples=payload["min_samples"],
            metric=payload["metric"],
            cluster_selection_method=payload["cluster_selection_method"],
            prediction_data=payload["prediction_data"],
        )
        obj._model = payload["model"]
        return obj
