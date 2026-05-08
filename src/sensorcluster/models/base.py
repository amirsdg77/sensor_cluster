"""Abstract contract for clustering models.

Defines the surface the pipeline orchestration depends on (fit, prediction
with confidence, novelty scoring, training-time labels, save/load). A new
clustering backend can be plugged in by satisfying this interface without
touching `pipeline/train.py` or the serving layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ClusterModel(ABC):
    """Minimal contract every clustering model must satisfy."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> ClusterModel: ...

    @abstractmethod
    def score_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(labels, membership_strengths, novelty_scores)`` for `X`.

        Implementations should compute all three in a single pass so that
        callers wiring the inference pipeline never have to traverse the
        underlying clustering structure twice.
        """

    @abstractmethod
    def predict_with_strength(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, membership_strengths) for new points."""

    @abstractmethod
    def novelty_score(self, X: np.ndarray) -> np.ndarray:
        """Return per-point novelty score in [0, 1] (higher = more novel)."""

    @property
    @abstractmethod
    def labels_(self) -> np.ndarray:
        """Cluster labels assigned during fit (-1 = noise)."""

    @abstractmethod
    def save(self, path: Path | str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> ClusterModel: ...
