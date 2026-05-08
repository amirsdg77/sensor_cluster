"""CV split tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensorcluster.data.splits import labeled_stratified_kfold


def test_labeled_stratified_kfold_only_holds_out_labeled_indices() -> None:
    n = 100
    y = pd.Series([np.nan] * n)
    labeled_idx = list(range(30))
    for i, idx in enumerate(labeled_idx):
        y.iloc[idx] = float((i % 3) + 1)  # 3 classes, 10 each

    splits = list(labeled_stratified_kfold(y, n_splits=5))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        # Test indices must all be labeled
        assert y.iloc[test_idx].notna().all()
        # Train + test = full range
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert sorted(np.union1d(train_idx, test_idx).tolist()) == list(range(n))


def test_labeled_stratified_kfold_raises_when_too_few_labels() -> None:
    y = pd.Series([1.0, 2.0, np.nan, np.nan])  # only 2 labels, asking for 5 folds
    with pytest.raises(ValueError):
        list(labeled_stratified_kfold(y, n_splits=5))
