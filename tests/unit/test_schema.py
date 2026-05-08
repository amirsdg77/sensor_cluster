"""Schema validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensorcluster.data.schema import build_sensor_schema


def _good_row() -> dict[str, float]:
    return {f"Sensor {i}": 0.1 for i in range(20)} | {"Label": 1.0}


def test_build_sensor_schema_validates_valid_dataframe() -> None:
    schema = build_sensor_schema()
    df = pd.DataFrame([_good_row(), {**_good_row(), "Label": float("nan")}])
    out = schema.validate(df, lazy=True)
    assert out.shape == (2, 21)


def test_build_sensor_schema_rejects_out_of_range_sensor() -> None:
    schema = build_sensor_schema(sensor_min=-1.0, sensor_max=1.0)
    bad = _good_row() | {"Sensor 0": 1.5}
    with pytest.raises(Exception):
        schema.validate(pd.DataFrame([bad]), lazy=True)


def test_build_sensor_schema_rejects_invalid_label() -> None:
    schema = build_sensor_schema(valid_labels=(1.0, 2.0))
    bad = _good_row() | {"Label": 99.0}
    with pytest.raises(Exception):
        schema.validate(pd.DataFrame([bad]), lazy=True)


def test_build_sensor_schema_allows_missing_label() -> None:
    schema = build_sensor_schema()
    row = _good_row() | {"Label": np.nan}
    out = schema.validate(pd.DataFrame([row]), lazy=True)
    assert pd.isna(out["Label"].iloc[0])
