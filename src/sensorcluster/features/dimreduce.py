"""Dimensionality reduction.

PCA is deterministic and picklable, with a clean ``transform()`` for new
points, so it sits in the inference path. UMAP has no parametric inverse
out of the box and is used only for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    """PCA wrapper that auto-picks `n_components` to hit a variance target."""

    def __init__(self, variance_target: float = 0.95, random_state: int = 42) -> None:
        if not 0.0 < variance_target <= 1.0:
            raise ValueError(f"variance_target must be in (0, 1], got {variance_target}")
        self.variance_target = variance_target
        self.random_state = random_state
        self._pca: PCA | None = None
        self._n_components_: int | None = None

    def fit(self, X: np.ndarray) -> PCAReducer:
        # Passing a float in (0, 1) to sklearn's PCA picks the smallest k whose
        # cumulative explained variance reaches that target.
        self._pca = PCA(
            n_components=self.variance_target,
            svd_solver="full",
            random_state=self.random_state,
        ).fit(X)
        self._n_components_ = int(self._pca.n_components_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAReducer.fit must be called before transform")
        return np.asarray(self._pca.transform(X), dtype=np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_components_(self) -> int:
        if self._n_components_ is None:
            raise RuntimeError("PCAReducer not fitted")
        return self._n_components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAReducer not fitted")
        return np.asarray(self._pca.explained_variance_ratio_)

    def save(self, path: Path | str) -> None:
        if self._pca is None:
            raise RuntimeError("Cannot save unfitted PCAReducer")
        payload: dict[str, Any] = {
            "pca": self._pca,
            "variance_target": self.variance_target,
            "random_state": self.random_state,
            "n_components_": self._n_components_,
        }
        joblib.dump(payload, Path(path), compress=3)

    @classmethod
    def load(cls, path: Path | str) -> PCAReducer:
        payload: dict[str, Any] = joblib.load(Path(path))
        obj = cls(
            variance_target=payload["variance_target"],
            random_state=payload["random_state"],
        )
        obj._pca = payload["pca"]
        obj._n_components_ = payload["n_components_"]
        return obj


class UMAPVisualizer:
    """2-D UMAP for plots only.

    Centralizing the call here keeps notebook and report-generation code
    pulling from a single configured embedder.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self._reducer: Any | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Lazy import: umap-learn lives in the `train` extra, not in `serve`.
        import umap

        # n_jobs=1 keeps the embedding deterministic given random_state;
        # parallel pynndescent passes otherwise introduce nondeterminism that
        # survives any seed.
        self._reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            n_jobs=1,
        )
        return np.asarray(self._reducer.fit_transform(X), dtype=np.float64)
