"""Data ingestion: schema validation, loading, CV splits."""

from sensorcluster.data.loader import DataValidationError, load_sensor_data
from sensorcluster.data.schema import build_sensor_schema
from sensorcluster.data.splits import labeled_stratified_kfold

__all__ = [
    "DataValidationError",
    "build_sensor_schema",
    "labeled_stratified_kfold",
    "load_sensor_data",
]
