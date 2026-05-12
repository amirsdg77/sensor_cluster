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
    """PCA wrapper that picks `n_components` either by an integer count
    (``n_components`` set) or by a variance target (``variance_target`` set).

    The two parameters are mutually exclusive at fit time. Setting both is
    accepted for backwards-compatibility but ``n_components`` wins.

    Why both: the project originally used a variance target (~0.95 → 19 PCs
    on the bundled 20-sensor data). Notebook 04 / the PCA sweep showed that
    on near-uniform tabular data, the curse of dimensionality kicks in well
    before the variance target is hit, and a small fixed ``n_components``
    (~5) gives meaningfully better HDBSCAN ARI and lower noise. The integer
    knob lets the YAML config pin a known-good dimensionality directly.
    """

    def __init__(
        self,
        variance_target: float | None = 0.95,
        n_components: int | None = None,
        random_state: int = 42,
    ) -> None:
        if n_components is None and variance_target is None:
            raise ValueError("Either n_components or variance_target must be set")
        if n_components is not None:
            if n_components < 1:
                raise ValueError(f"n_components must be >= 1, got {n_components}")
        elif not 0.0 < variance_target <= 1.0:
            raise ValueError(f"variance_target must be in (0, 1], got {variance_target}")
        self.variance_target = variance_target
        self.n_components = n_components
        self.random_state = random_state
        self._pca: PCA | None = None
        self._n_components_: int | None = None

    def fit(self, X: np.ndarray) -> PCAReducer:
        # n_components takes precedence; otherwise pass the float variance
        # target straight to sklearn (it picks the smallest k whose cumulative
        # explained variance reaches that target).
        if self.n_components is not None:
            n_arg: int | float = min(self.n_components, X.shape[1])
        else:
            n_arg = self.variance_target
        self._pca = PCA(
            n_components=n_arg,
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
            "n_components": self.n_components,
            "random_state": self.random_state,
            "n_components_": self._n_components_,
        }
        joblib.dump(payload, Path(path), compress=3)

    @classmethod
    def load(cls, path: Path | str) -> PCAReducer:
        payload: dict[str, Any] = joblib.load(Path(path))
        obj = cls(
            variance_target=payload.get("variance_target"),
            n_components=payload.get("n_components"),
            random_state=payload["random_state"],
        )
        obj._pca = payload["pca"]
        obj._n_components_ = payload["n_components_"]
        return obj


class AggregatesReducer:
    """Per-row statistical-aggregates feature transformer.

    Computes a fixed set of summary statistics across the input columns for
    each row — mean, std, min, max, range, skewness, kurtosis, q25, q75,
    median by default. This collapses an n-sensor row into a small set of
    per-row distributional features.

    Why this exists: notebook 04's multi-view experiment showed that on the
    bundled near-uniform sensor data, HDBSCAN clusters per-row aggregates
    much better than per-sensor PCA — multi-seed CV ARI ~0.40 vs ~0.09 for
    PCA-5. The mechanism: PCA finds linear combinations of *sensor columns*;
    aggregates compute per-row *moments* (e.g. ``max - min`` is non-linear).
    Failure modes on this dataset are distinguishable by the row's overall
    distribution character, not by individual sensor values, which PCA
    cannot capture.

    Public surface matches :class:`PCAReducer`: ``fit / transform /
    fit_transform / save / load / n_components_``, so callers (train.py,
    pipeline_model.py) treat the two interchangeably.
    """

    # Names match the ``feature_names_`` order; keep in sync with _compute().
    FEATURE_NAMES: tuple[str, ...] = (
        "mean",
        "std",
        "min",
        "max",
        "range",
        "skewness",
        "kurtosis",
        "q25",
        "q75",
        "median",
    )

    def __init__(self, random_state: int = 42) -> None:
        # random_state is not used internally — aggregates are deterministic —
        # but we keep the parameter so the public API mirrors PCAReducer.
        self.random_state = random_state
        self._n_input_features: int | None = None

    def fit(self, X: np.ndarray) -> AggregatesReducer:
        """No-op fit: we record the input dimensionality for sanity checks
        but the transform itself has no fitted state."""
        self._n_input_features = int(X.shape[1])
        return self

    @staticmethod
    def _compute(X: np.ndarray) -> np.ndarray:
        """Compute the row-level aggregates without using any sklearn fit
        state. Pure function of the input row."""
        # Lazy import — scipy is already a transitive dep but keep the import
        # local so the import order is clear from this method alone.
        from scipy import stats as sp_stats

        return np.column_stack(
            [
                X.mean(axis=1),
                X.std(axis=1),
                X.min(axis=1),
                X.max(axis=1),
                X.max(axis=1) - X.min(axis=1),
                sp_stats.skew(X, axis=1),
                sp_stats.kurtosis(X, axis=1),
                np.quantile(X, 0.25, axis=1),
                np.quantile(X, 0.75, axis=1),
                np.median(X, axis=1),
            ]
        ).astype(np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._n_input_features is None:
            raise RuntimeError("AggregatesReducer.fit must be called before transform")
        if X.shape[1] != self._n_input_features:
            raise ValueError(f"Expected {self._n_input_features} input features, got {X.shape[1]}")
        return self._compute(np.asarray(X, dtype=np.float64))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_components_(self) -> int:
        return len(self.FEATURE_NAMES)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """For interface parity with PCAReducer. Aggregates don't decompose
        variance, so we return an empty array — callers that branch on the
        feature mode know not to interpret this number."""
        return np.array([], dtype=np.float64)

    def save(self, path: Path | str) -> None:
        if self._n_input_features is None:
            raise RuntimeError("Cannot save unfitted AggregatesReducer")
        payload: dict[str, Any] = {
            "kind": "aggregates",
            "n_input_features": self._n_input_features,
            "random_state": self.random_state,
            "feature_names": list(self.FEATURE_NAMES),
        }
        joblib.dump(payload, Path(path), compress=3)

    @classmethod
    def load(cls, path: Path | str) -> AggregatesReducer:
        payload: dict[str, Any] = joblib.load(Path(path))
        obj = cls(random_state=payload["random_state"])
        obj._n_input_features = int(payload["n_input_features"])
        return obj


def load_reducer(path: Path | str):
    """Polymorphic loader — detects whether ``path`` holds a PCA or an
    aggregates artifact and returns the right object. This keeps
    :class:`SensorClusterPipeline.load` agnostic about the feature mode."""
    payload: dict[str, Any] = joblib.load(Path(path))
    if payload.get("kind") == "aggregates":
        obj = AggregatesReducer(random_state=payload["random_state"])
        obj._n_input_features = int(payload["n_input_features"])
        return obj
    # Default: legacy PCA payload. Reconstruct via PCAReducer.load to keep
    # the round-trip path identical to before this refactor.
    obj_pca = PCAReducer(
        variance_target=payload.get("variance_target"),
        n_components=payload.get("n_components"),
        random_state=payload["random_state"],
    )
    obj_pca._pca = payload["pca"]
    obj_pca._n_components_ = payload["n_components_"]
    return obj_pca


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
