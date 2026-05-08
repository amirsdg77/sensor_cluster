"""Imputer + StandardScaler bundle.

Always re-fit at training time so the saved scaler reflects the actual
training distribution; this protects inference against upstream rescaling
or distribution drift between training runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Fit/transform/save/load wrapper around `Imputer -> StandardScaler`."""

    def __init__(self, imputer_strategy: str = "median") -> None:
        self.imputer_strategy = imputer_strategy
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        self._feature_names = list(X.columns)
        self._pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.imputer_strategy)),
                ("scaler", StandardScaler()),
            ]
        )
        self._pipeline.fit(X.to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Preprocessor.fit must be called before transform")
        if self._feature_names is not None and list(X.columns) != self._feature_names:
            # tolerate column reordering, fail on mismatch
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing expected columns at transform time: {sorted(missing)}")
            X = X[self._feature_names]
        return np.asarray(self._pipeline.transform(X.to_numpy()), dtype=np.float64)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Preprocessor not fitted")
        return list(self._feature_names)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        if self._pipeline is None:
            raise RuntimeError("Cannot save unfitted Preprocessor")
        payload: dict[str, Any] = {
            "pipeline": self._pipeline,
            "feature_names": self._feature_names,
            "imputer_strategy": self.imputer_strategy,
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: Path | str) -> Preprocessor:
        payload: dict[str, Any] = joblib.load(Path(path))
        obj = cls(imputer_strategy=payload["imputer_strategy"])
        obj._pipeline = payload["pipeline"]
        obj._feature_names = payload["feature_names"]
        return obj
