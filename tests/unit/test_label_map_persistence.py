"""Round-trip and version-handling tests for ClusterLabelMap on disk."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sensorcluster.models.label_map import (
    LABEL_MAP_SCHEMA_VERSION,
    ClusterLabelEntry,
    ClusterLabelMap,
)


def _entry(cid: int, *, name: str = "CLASS_1", purity: float = 1.0) -> ClusterLabelEntry:
    return ClusterLabelEntry(
        cluster_id=cid,
        name=name,
        is_known=True,
        is_noise=False,
        purity=purity,
        n_labeled=3,
        n_total=10,
        label_distribution={"1.0": 3},
    )


def test_save_emits_json_null_for_nan_purity(tmp_path: Path) -> None:
    """NaN purities must serialize as JSON ``null``, not the non-standard ``NaN`` literal."""
    m = ClusterLabelMap(
        entries={
            0: _entry(0),
            -1: ClusterLabelEntry(
                cluster_id=-1,
                name="NOISE",
                is_known=False,
                is_noise=True,
                purity=float("nan"),
                n_labeled=2,
                n_total=50,
                label_distribution={"1.0": 1, "2.0": 1},
            ),
        }
    )
    p = tmp_path / "label_map.json"
    m.save(p)

    raw = p.read_text(encoding="utf-8")
    assert "NaN" not in raw
    parsed = json.loads(raw)
    assert parsed["entries"]["-1"]["purity"] is None


def test_round_trip_preserves_nan_purity(tmp_path: Path) -> None:
    m = ClusterLabelMap(
        entries={
            -1: ClusterLabelEntry(
                cluster_id=-1,
                name="NOISE",
                is_known=False,
                is_noise=True,
                purity=float("nan"),
                n_labeled=0,
                n_total=4,
                label_distribution={},
            )
        }
    )
    p = tmp_path / "label_map.json"
    m.save(p)
    loaded = ClusterLabelMap.load(p)
    assert math.isnan(loaded.entries[-1].purity)


def test_round_trip_preserves_known_entries(tmp_path: Path) -> None:
    m = ClusterLabelMap(
        entries={0: _entry(0, name="CLASS_1", purity=0.75)},
        purity_warning=0.5,
    )
    p = tmp_path / "label_map.json"
    m.save(p)
    loaded = ClusterLabelMap.load(p)
    assert loaded.entries[0].name == "CLASS_1"
    assert loaded.entries[0].purity == 0.75
    assert loaded.purity_warning == 0.5


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "bad_version.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "999.0.0",
                "purity_warning": 0.6,
                "entries": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        ClusterLabelMap.load(p)


def test_save_includes_current_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "label_map.json"
    ClusterLabelMap().save(p)
    assert json.loads(p.read_text())["schema_version"] == LABEL_MAP_SCHEMA_VERSION
