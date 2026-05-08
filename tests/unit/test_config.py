"""Settings loader tests."""

from __future__ import annotations

from pathlib import Path

from sensorcluster.config import load_settings


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        """
random_seed: 7
hdbscan:
  min_cluster_size: 25
""",
        encoding="utf-8",
    )
    cfg = load_settings(p)
    assert cfg.random_seed == 7
    assert cfg.hdbscan.min_cluster_size == 25
    # other knobs default
    assert cfg.pca.variance_target == 0.95


def test_load_settings_overrides_take_precedence(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("random_seed: 1\n", encoding="utf-8")
    cfg = load_settings(p, overrides={"random_seed": 99})
    assert cfg.random_seed == 99
