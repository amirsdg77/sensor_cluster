"""Pandera schema for the raw sensor CSV.

Validates:
    - 20 numeric sensor columns named "Sensor 0".."Sensor 19"
    - All sensor values within [sensor_min, sensor_max]
    - Optional Label column ∈ valid_labels ∪ {NaN}
"""

from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema


def _sensor_columns(prefix: str, n_sensors: int, lo: float, hi: float) -> dict[str, Column]:
    return {
        f"{prefix}{i}": Column(
            float,
            checks=[pa.Check.in_range(lo, hi, include_min=True, include_max=True)],
            nullable=False,
            coerce=True,
        )
        for i in range(n_sensors)
    }


def build_sensor_schema(
    *,
    prefix: str = "Sensor ",
    n_sensors: int = 20,
    sensor_min: float = -1.05,
    sensor_max: float = 1.05,
    label_column: str = "Label",
    valid_labels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> DataFrameSchema:
    """Construct the DataFrameSchema for the sensor CSV.

    The Label column is nullable; non-null values must be one of `valid_labels`.
    """
    columns = _sensor_columns(prefix, n_sensors, sensor_min, sensor_max)
    columns[label_column] = Column(
        float,
        checks=[pa.Check.isin(list(valid_labels))],
        nullable=True,
        coerce=True,
    )
    return DataFrameSchema(
        columns=columns,
        strict=False,  # tolerate extra trailing columns (e.g. row index)
        ordered=False,
        coerce=True,
    )
