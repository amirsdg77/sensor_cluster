"""HTTP route handlers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from sensorcluster.api.dependencies import get_pipeline
from sensorcluster.api.schemas import (
    ClusterEntryResponse,
    ClustersResponse,
    HealthResponse,
    NeighborInfo,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from sensorcluster.models.pipeline_model import PredictionResult, SensorClusterPipeline

router = APIRouter()


def _build_dataframe(rows: list[list[float]], pipeline: SensorClusterPipeline) -> pd.DataFrame:
    """Materialize incoming rows into a frame with the pipeline's expected columns.

    Raises 422 when any row has the wrong number of sensors for the loaded
    pipeline; this is the exact-length check that the schema cannot do
    (the count is pipeline-dependent rather than schema-static).
    """
    expected_cols = pipeline.preprocessor.feature_names
    n = len(expected_cols)
    for i, row in enumerate(rows):
        if len(row) != n:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Row {i}: expected {n} sensors, got {len(row)}",
            )
    return pd.DataFrame(rows, columns=expected_cols, dtype=float)


@router.get("/health", response_model=HealthResponse)
def health(pipeline: SensorClusterPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Liveness probe that also reports model identity and cluster counts."""
    n_clusters = sum(1 for e in pipeline.label_map.entries.values() if not e.is_noise)
    n_unknown = sum(
        1 for e in pipeline.label_map.entries.values() if (not e.is_known) and (not e.is_noise)
    )
    return HealthResponse(
        status="ok",
        model_version=pipeline.model_version,
        trained_at=pipeline.trained_at,
        n_clusters=n_clusters,
        n_unknown_clusters=n_unknown,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    pipeline: SensorClusterPipeline = Depends(get_pipeline),
) -> PredictResponse:
    """Score a single sensor reading."""
    df = _build_dataframe([body.sensors], pipeline)
    result = pipeline.predict(df)[0]
    return _to_response(result)


@router.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(
    body: PredictBatchRequest,
    pipeline: SensorClusterPipeline = Depends(get_pipeline),
) -> PredictBatchResponse:
    """Score many sensor readings in a single vectorized call."""
    df = _build_dataframe(body.rows, pipeline)
    results = pipeline.predict(df)
    return PredictBatchResponse(predictions=[_to_response(r) for r in results])


@router.get("/clusters", response_model=ClustersResponse)
def clusters(pipeline: SensorClusterPipeline = Depends(get_pipeline)) -> ClustersResponse:
    """Return the cluster->label map for inspection (purity, distribution, totals)."""
    entries: list[ClusterEntryResponse] = []
    for e in sorted(pipeline.label_map.entries.values(), key=lambda x: x.cluster_id):
        purity = None if (isinstance(e.purity, float) and math.isnan(e.purity)) else e.purity
        entries.append(
            ClusterEntryResponse(
                cluster_id=e.cluster_id,
                name=e.name,
                is_known=e.is_known,
                is_noise=e.is_noise,
                purity=purity,
                n_labeled=e.n_labeled,
                n_total=e.n_total,
                label_distribution=e.label_distribution,
            )
        )
    return ClustersResponse(
        purity_warning=pipeline.label_map.purity_warning,
        entries=entries,
    )


def _to_response(result: PredictionResult) -> PredictResponse:
    """Convert an internal `PredictionResult` into the wire-format response.

    Confidence and novelty are clipped to [0, 1] so that bounds drift in any
    upstream component never produces a 500 from the pydantic response model.
    """
    confidence = float(np.clip(result.confidence, 0.0, 1.0))
    novelty = float(np.clip(result.novelty_score, 0.0, 1.0))
    neighbors = [NeighborInfo(**n) for n in result.top_neighbors]
    return PredictResponse(
        predicted_label=result.predicted_label,
        confidence=confidence,
        novelty_score=novelty,
        cluster_id=result.cluster_id,
        is_novel=result.is_novel,
        top_neighbors=neighbors,
    )
