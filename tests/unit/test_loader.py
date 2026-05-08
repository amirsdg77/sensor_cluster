"""Loader / split-features tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sensorcluster.data.loader import (
    DataValidationError,
    load_sensor_data,
    split_features_label,
)


def test_load_sensor_data_round_trips_synthetic_csv(synthetic_csv: Path) -> None:
    df = load_sensor_data(synthetic_csv)
    assert df.shape[1] == 21  # 20 sensors + Label
    assert "Label" in df.columns
    assert df["Label"].notna().sum() > 0


def test_load_sensor_data_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sensor_data(tmp_path / "nope.csv")


def test_load_sensor_data_rejects_bad_schema(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Sensor 0": [99.0], "Label": [1.0]}).to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        load_sensor_data(p, n_sensors=1, sensor_min=-1.0, sensor_max=1.0)


def test_split_features_label_preserves_column_order(synthetic_csv: Path) -> None:
    df = load_sensor_data(synthetic_csv)
    X, y = split_features_label(df)
    assert list(X.columns) == [f"Sensor {i}" for i in range(20)]
    assert len(y) == len(X) == len(df)
