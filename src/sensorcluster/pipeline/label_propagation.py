"""Build a ``ClusterLabelMap`` from training-time clusters and labels.

Naming rule per cluster c:
  - c == -1 (HDBSCAN noise)  -> "NOISE"
  - no labeled members in c  -> "UNKNOWN_<c>"   (candidate undiscovered mode)
  - otherwise                -> weighted majority vote over the labeled members,
                                producing "CLASS_<n>" for the winning class
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sensorcluster.models.label_map import ClusterLabelEntry, ClusterLabelMap


def _label_to_name(label: float) -> str:
    """Render a numeric class label as a stable string (e.g. 1.0 -> ``CLASS_1``)."""
    return f"CLASS_{int(label)}"


def build_label_map(
    cluster_labels: np.ndarray,
    y: pd.Series,
    *,
    weighted: bool = True,
    purity_warning: float = 0.6,
) -> ClusterLabelMap:
    """Construct a ``ClusterLabelMap`` from training-time clusters and labels.

    Args:
        cluster_labels: HDBSCAN ``labels_`` array (length N, -1 = noise).
        y: Ground-truth labels (length N, NaN for unlabeled rows).
        weighted: When True, scale each labeled vote by 1 / class_count so
            rare classes are not crowded out by frequent ones.
        purity_warning: Per-cluster purity below this value triggers a
            warning surfaced via :meth:`ClusterLabelMap.warnings`.
    """
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    if len(cluster_labels) != len(y):
        raise ValueError("cluster_labels and y must be same length")

    y_arr = y.to_numpy()
    labeled_mask = ~pd.isna(y_arr)

    class_counts = Counter(float(v) for v in y_arr[labeled_mask])
    weights = {c: (1.0 / n if weighted and n > 0 else 1.0) for c, n in class_counts.items()}

    per_cluster_labels: dict[int, list[float]] = defaultdict(list)
    per_cluster_total: Counter[int] = Counter()
    for idx, cid in enumerate(cluster_labels):
        per_cluster_total[int(cid)] += 1
        if labeled_mask[idx]:
            per_cluster_labels[int(cid)].append(float(y_arr[idx]))

    entries: dict[int, ClusterLabelEntry] = {}
    for cid, total in per_cluster_total.items():
        labels_in_cluster = per_cluster_labels.get(cid, [])
        n_labeled = len(labels_in_cluster)
        label_dist = {str(c): int(n) for c, n in Counter(labels_in_cluster).items()}

        if cid == -1:
            entries[cid] = ClusterLabelEntry(
                cluster_id=cid,
                name="NOISE",
                is_known=False,
                is_noise=True,
                purity=float("nan"),
                n_labeled=n_labeled,
                n_total=total,
                label_distribution=label_dist,
            )
            continue

        if n_labeled == 0:
            entries[cid] = ClusterLabelEntry(
                cluster_id=cid,
                name=f"UNKNOWN_{cid}",
                is_known=False,
                is_noise=False,
                purity=float("nan"),
                n_labeled=0,
                n_total=total,
                label_distribution={},
            )
            continue

        # Sort key: vote score descending, then class id ascending. Without
        # the secondary key, ties resolve by dict-insertion order, which
        # depends on data ordering.
        scores: dict[float, float] = defaultdict(float)
        for c in labels_in_cluster:
            scores[c] += weights.get(c, 1.0)
        winner = max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        # Purity is computed on raw counts so it stays interpretable as a
        # fraction even when `weighted` is True.
        purity = max(label_dist.values()) / float(n_labeled)

        entries[cid] = ClusterLabelEntry(
            cluster_id=cid,
            name=_label_to_name(winner),
            is_known=True,
            is_noise=False,
            purity=purity,
            n_labeled=n_labeled,
            n_total=total,
            label_distribution=label_dist,
        )

    return ClusterLabelMap(entries=entries, purity_warning=purity_warning)
