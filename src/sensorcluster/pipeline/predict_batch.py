"""Batch scoring: read a CSV of sensor rows, write predictions to a parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensorcluster.data.loader import load_sensor_data, split_features_label
from sensorcluster.logging_setup import get_logger
from sensorcluster.models.pipeline_model import SensorClusterPipeline

log = get_logger(__name__)


def predict_batch(
    *,
    input_path: Path | str,
    output_path: Path | str,
    artifacts_dir: Path | str,
    sensor_columns_prefix: str = "Sensor ",
    n_sensors: int = 20,
    label_column: str = "Label",
    sensor_min: float = -1.05,
    sensor_max: float = 1.05,
    valid_labels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Score every row of `input_path` and write predictions to `output_path`.

    The output format is parquet when ``output_path`` has the ``.parquet``
    suffix, CSV otherwise. Returns the same predictions in-memory as a
    DataFrame for callers that want them without round-tripping through disk.
    """
    df = load_sensor_data(
        input_path,
        prefix=sensor_columns_prefix,
        n_sensors=n_sensors,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        label_column=label_column,
        valid_labels=valid_labels,
    )
    X, _y = split_features_label(
        df,
        prefix=sensor_columns_prefix,
        n_sensors=n_sensors,
        label_column=label_column,
    )
    pipeline = SensorClusterPipeline.load(artifacts_dir)

    results = pipeline.predict(X)
    out = pd.DataFrame(
        {
            "row_index": range(len(results)),
            "predicted_label": [r.predicted_label for r in results],
            "confidence": [r.confidence for r in results],
            "novelty_score": [r.novelty_score for r in results],
            "cluster_id": [r.cluster_id for r in results],
            "is_novel": [r.is_novel for r in results],
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)

    log.info(
        "batch_predict_complete",
        n_rows=len(out),
        output=str(output_path),
        n_novel=int(out["is_novel"].sum()),
    )
    return out
