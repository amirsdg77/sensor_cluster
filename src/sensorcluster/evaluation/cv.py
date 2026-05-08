"""Stratified K-fold cross-validation over the labeled subset.

For each fold a stratified slice of labeled rows is held out, the entire
pipeline (preprocess + PCA + HDBSCAN + label map) is refit on the rest, and
the held-out rows are scored to produce an ARI per fold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from sensorcluster.data.splits import labeled_stratified_kfold


@dataclass(frozen=True)
class CVResult:
    """Per-fold and aggregate ARI from a CV run.

    Attributes:
        fold_aris: ARI scored on each held-out fold, in fold order.
        mean_ari: Mean over ``fold_aris``.
        std_ari: Standard deviation over ``fold_aris``.
        n_folds: Number of folds actually executed.
    """

    fold_aris: list[float]
    mean_ari: float
    std_ari: float
    n_folds: int


def cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    pipeline_factory: Callable[[pd.DataFrame, pd.Series], Callable[[pd.DataFrame], np.ndarray]],
    n_splits: int = 5,
    seed: int = 42,
) -> CVResult:
    """Run stratified cross-validation over the labeled subset of `y`.

    Args:
        X: Feature frame for all rows (labeled + unlabeled).
        y: Label series; NaN entries are treated as unlabeled and used only
            for training, never scored.
        pipeline_factory: Callable that takes ``(X_train, y_train)`` and
            returns a ``predict_label_id(X_test)`` function. Yielding a
            closure keeps this generic and prevents the evaluation layer
            from depending on any concrete pipeline class.
        n_splits: Number of stratified folds.
        seed: Random seed forwarded to ``StratifiedKFold``.
    """
    fold_aris: list[float] = []

    for train_idx, test_idx in labeled_stratified_kfold(y, n_splits=n_splits, seed=seed):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx].to_numpy().astype(float)

        predict_fn = pipeline_factory(X_train, y_train)
        y_pred_ids = predict_fn(X_test)

        ari = float(adjusted_rand_score(y_test, y_pred_ids))
        fold_aris.append(ari)

    arr = np.array(fold_aris)
    return CVResult(
        fold_aris=fold_aris,
        mean_ari=float(arr.mean()) if len(arr) else float("nan"),
        std_ari=float(arr.std()) if len(arr) else float("nan"),
        n_folds=len(fold_aris),
    )
