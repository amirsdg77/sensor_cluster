"""Load and validate the sensor CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors as pae

from sensorcluster.data.schema import build_sensor_schema
from sensorcluster.logging_setup import get_logger

log = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when the input CSV fails schema validation."""


def load_sensor_data(
    path: Path | str,
    *,
    prefix: str = "Sensor ",
    n_sensors: int = 20,
    sensor_min: float = -1.05,
    sensor_max: float = 1.05,
    label_column: str = "Label",
    valid_labels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Read a sensor CSV and validate it against the schema.

    Empty Label cells parse to NaN — those are the unlabeled samples.
    Returns a DataFrame guaranteed to satisfy the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor data not found: {path}")

    df = pd.read_csv(path)
    schema = build_sensor_schema(
        prefix=prefix,
        n_sensors=n_sensors,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        label_column=label_column,
        valid_labels=valid_labels,
    )
    try:
        validated = schema.validate(df, lazy=True)
    except pae.SchemaErrors as exc:
        raise DataValidationError(
            f"Schema validation failed for {path}:\n{exc.failure_cases}"
        ) from exc

    n_rows = len(validated)
    n_labeled = int(validated[label_column].notna().sum())
    label_hist = (
        validated[label_column].dropna().astype(float).value_counts().sort_index().to_dict()
    )

    log.info(
        "data_loaded",
        path=str(path),
        n_rows=n_rows,
        n_labeled=n_labeled,
        n_unlabeled=n_rows - n_labeled,
        label_histogram=label_hist,
    )
    return validated


def split_features_label(
    df: pd.DataFrame,
    *,
    prefix: str = "Sensor ",
    n_sensors: int = 20,
    label_column: str = "Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X holds sensor columns in order and y is the Label column."""
    cols = [f"{prefix}{i}" for i in range(n_sensors)]
    return df[cols].copy(), df[label_column].copy()
