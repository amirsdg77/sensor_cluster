"""Cross-validation splits over the labeled subset only.

The labeled rows are the only ground truth available, so each fold holds out
a stratified slice of labeled indices and trains on the remainder (other
labeled rows + all unlabeled rows). Callers score the held-out labels and
report ARI per fold.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def labeled_stratified_kfold(
    y: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) over the *full* dataframe length.

    `train_idx` includes all unlabeled points and the labeled points NOT in this fold.
    `test_idx` is the held-out labeled slice.

    Raises ValueError if any class has fewer members than `n_splits`.
    """
    labeled_mask = y.notna().to_numpy()
    labeled_positions = np.where(labeled_mask)[0]
    labels = y.dropna().to_numpy()

    if len(labeled_positions) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} labeled points for {n_splits}-fold CV; got {len(labeled_positions)}"
        )
    class_counts = Counter(labels.tolist())
    min_class_count = min(class_counts.values())
    if min_class_count < n_splits:
        smallest_class = min(class_counts, key=class_counts.get)
        raise ValueError(
            f"Each class needs at least {n_splits} labeled samples for {n_splits}-fold CV; "
            f"class {smallest_class!r} has only {min_class_count}."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(len(y))
    for _, test_pos_in_labeled in skf.split(labeled_positions, labels):
        test_idx = labeled_positions[test_pos_in_labeled]
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        yield train_idx, test_idx
