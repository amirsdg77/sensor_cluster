"""Label propagation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sensorcluster.pipeline.label_propagation import build_label_map


def test_pure_cluster_named_after_majority_label() -> None:
    cluster_labels = np.array([0, 0, 0, 0, 1, 1])
    y = pd.Series([1.0, 1.0, 1.0, np.nan, 2.0, np.nan])
    m = build_label_map(cluster_labels, y, weighted=False)
    assert m.entries[0].name == "CLASS_1"
    assert m.entries[0].is_known
    assert m.entries[0].purity == 1.0
    assert m.entries[1].name == "CLASS_2"


def test_empty_cluster_named_unknown() -> None:
    cluster_labels = np.array([0, 0, 1, 1, 1])
    y = pd.Series([1.0, 1.0, np.nan, np.nan, np.nan])
    m = build_label_map(cluster_labels, y)
    assert m.entries[1].name == "UNKNOWN_1"
    assert not m.entries[1].is_known


def test_noise_cluster_named_noise() -> None:
    cluster_labels = np.array([-1, -1, 0, 0])
    y = pd.Series([1.0, np.nan, 2.0, np.nan])
    m = build_label_map(cluster_labels, y)
    assert m.entries[-1].name == "NOISE"
    assert m.entries[-1].is_noise


def test_weighted_voting_corrects_imbalance() -> None:
    # cluster 0: 3x class_1, 2x class_2.
    # If global counts are (1: 30, 2: 5), unweighted majority = CLASS_1.
    # Weighted (1/30 vs 1/5) = CLASS_2 wins.
    cluster_labels = np.array([0, 0, 0, 0, 0])
    y_in_cluster = [1.0, 1.0, 1.0, 2.0, 2.0]
    y = pd.Series(y_in_cluster + [1.0] * 27 + [2.0] * 3 + [np.nan] * 5)
    cluster_labels = np.concatenate([cluster_labels, np.full(35, -1, dtype=int)])
    m = build_label_map(cluster_labels, y, weighted=True)
    assert m.entries[0].name == "CLASS_2"  # rarer class wins despite fewer in-cluster votes


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cluster_labels = np.array([0, 0, 1, -1])
    y = pd.Series([1.0, np.nan, 2.0, np.nan])
    m = build_label_map(cluster_labels, y)

    p = tmp_path / "map.json"
    m.save(p)
    from sensorcluster.models.label_map import ClusterLabelMap

    m2 = ClusterLabelMap.load(p)
    assert {cid: e.name for cid, e in m2.entries.items()} == {
        cid: e.name for cid, e in m.entries.items()
    }


def test_tied_vote_breaks_to_smaller_class_id() -> None:
    """An exact tie must resolve to the smaller class id, not to data order."""
    cluster_labels = np.array([0, 0])
    # Reverse data order so insertion would put CLASS_2 first; smaller-id rule
    # should still pick CLASS_1.
    y = pd.Series([2.0, 1.0])
    m = build_label_map(cluster_labels, y, weighted=False)
    assert m.entries[0].name == "CLASS_1"

    # Same outcome regardless of ordering.
    y2 = pd.Series([1.0, 2.0])
    m2 = build_label_map(cluster_labels, y2, weighted=False)
    assert m2.entries[0].name == "CLASS_1"


def test_warnings_flag_low_purity() -> None:
    cluster_labels = np.array([0, 0, 0, 0])
    y = pd.Series([1.0, 1.0, 2.0, 2.0])  # purity 0.5
    m = build_label_map(cluster_labels, y, purity_warning=0.6)
    warnings = m.warnings()
    assert len(warnings) == 1
    assert "low purity" in warnings[0]
