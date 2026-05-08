"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    model_version: str
    trained_at: str
    n_clusters: int
    n_unknown_clusters: int


# Loose upper bound on vector length; the route handler runs the exact
# pipeline-specific length check. Decoupling the schema from the pipeline's
# feature count keeps the OpenAPI surface stable when the model changes shape.
_MAX_SENSORS = 1024
# In-band sensor value range, mirrored from the pandera training schema.
_SENSOR_MIN = -1.05
_SENSOR_MAX = 1.05


class PredictRequest(BaseModel):
    """Single sensor reading."""

    model_config = ConfigDict(extra="forbid")
    sensors: list[float] = Field(
        min_length=1,
        max_length=_MAX_SENSORS,
        description="Vector of sensor readings, one per sensor channel.",
    )

    @field_validator("sensors")
    @classmethod
    def _validate_values(cls, v: list[float]) -> list[float]:
        for x in v:
            if x != x or x in (float("inf"), float("-inf")):  # NaN: x != x.
                raise ValueError("sensors must not contain NaN or Inf")
            if not (_SENSOR_MIN <= x <= _SENSOR_MAX):
                raise ValueError(
                    f"sensor values must be in [{_SENSOR_MIN}, {_SENSOR_MAX}]; got {x}"
                )
        return v


class NeighborInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_index: int
    label: str
    distance: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    cluster_id: int
    is_novel: bool
    top_neighbors: list[NeighborInfo]


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[list[float]] = Field(min_length=1, max_length=10_000)

    @field_validator("rows")
    @classmethod
    def _validate_rows(cls, v: list[list[float]]) -> list[list[float]]:
        # Reuse the per-row validator to keep the rules in one place.
        for i, row in enumerate(v):
            if not (1 <= len(row) <= _MAX_SENSORS):
                raise ValueError(f"row {i}: length {len(row)} outside [1, {_MAX_SENSORS}]")
            for x in row:
                if x != x or x in (float("inf"), float("-inf")):
                    raise ValueError(f"row {i}: NaN/Inf forbidden")
                if not (_SENSOR_MIN <= x <= _SENSOR_MAX):
                    raise ValueError(
                        f"row {i}: sensor value {x} outside [{_SENSOR_MIN}, {_SENSOR_MAX}]"
                    )
        return v


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: list[PredictResponse]


class ClusterEntryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_id: int
    name: str
    is_known: bool
    is_noise: bool
    purity: float | None
    n_labeled: int
    n_total: int
    label_distribution: dict[str, int]


class ClustersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purity_warning: float
    entries: list[ClusterEntryResponse]


def to_neighbors(raw: list[dict[str, Any]]) -> list[NeighborInfo]:
    return [NeighborInfo(**r) for r in raw]
