"""HTTP route handlers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from sensorcluster.api.dependencies import get_pipeline
from sensorcluster.api.schemas import (
    ClusterEntryResponse,
    ClustersResponse,
    HealthResponse,
    LivenessResponse,
    NeighborInfo,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
    VersionResponse,
)
from sensorcluster.models.label_map import LABEL_MAP_SCHEMA_VERSION
from sensorcluster.models.pipeline_model import (
    PredictionResult,
    SensorClusterPipeline,
)

router = APIRouter()

# Bumped when the wire format of the API changes incompatibly.
API_VERSION = "1.0.0"


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


@router.get("/healthz", response_model=LivenessResponse, tags=["meta"])
def healthz() -> LivenessResponse:
    """Liveness probe — process is responsive.

    No dependency on the pipeline so this stays green during model reload
    or any transient downstream failure. Container orchestrators (k8s,
    Cloud Run, ECS) restart the pod only when liveness fails.
    """
    return LivenessResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse, tags=["meta"])
def readyz(pipeline: SensorClusterPipeline = Depends(get_pipeline)) -> ReadinessResponse:
    """Readiness probe — pipeline loaded and able to serve predictions.

    Returns model identity for ops dashboards. If the pipeline failed to
    load, the dependency raises and the load balancer pulls this replica
    out of rotation without restarting it.
    """
    n_clusters = sum(1 for e in pipeline.label_map.entries.values() if not e.is_noise)
    n_unknown = sum(
        1 for e in pipeline.label_map.entries.values() if (not e.is_known) and (not e.is_noise)
    )
    return ReadinessResponse(
        status="ready",
        model_version=pipeline.model_version,
        trained_at=pipeline.trained_at,
        n_clusters=n_clusters,
        n_unknown_clusters=n_unknown,
    )


@router.get("/version", response_model=VersionResponse, tags=["meta"])
def version(pipeline: SensorClusterPipeline = Depends(get_pipeline)) -> VersionResponse:
    """Build + model version metadata. Useful for audit logs and incident triage."""
    return VersionResponse(
        model_version=pipeline.model_version,
        trained_at=pipeline.trained_at,
        schema_version=LABEL_MAP_SCHEMA_VERSION,
        api_version=API_VERSION,
    )


@router.get("/health", response_model=HealthResponse, tags=["meta"], include_in_schema=False)
def health(pipeline: SensorClusterPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Backwards-compatible combined health endpoint.

    Kept for the existing Docker HEALTHCHECK and docker-compose health probe;
    new clients should call /healthz (liveness) and /readyz (readiness)
    separately. Hidden from the OpenAPI schema to discourage new use.
    """
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


@router.post("/predict", response_model=PredictResponse, tags=["analyze"])
def predict(
    body: PredictRequest,
    pipeline: SensorClusterPipeline = Depends(get_pipeline),
) -> PredictResponse:
    """Score a single sensor reading."""
    df = _build_dataframe([body.sensors], pipeline)
    result = pipeline.predict(df)[0]
    return _to_response(result)


@router.post("/predict_batch", response_model=PredictBatchResponse, tags=["analyze"])
def predict_batch(
    body: PredictBatchRequest,
    pipeline: SensorClusterPipeline = Depends(get_pipeline),
) -> PredictBatchResponse:
    """Score many sensor readings in a single vectorized call."""
    df = _build_dataframe(body.rows, pipeline)
    results = pipeline.predict(df)
    return PredictBatchResponse(predictions=[_to_response(r) for r in results])


@router.post("/predict_batch/stream", tags=["analyze"])
def predict_batch_stream(
    body: PredictBatchRequest,
    request: Request,
    pipeline: SensorClusterPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """Score many sensor readings, streamed as NDJSON (one prediction per line).

    Lets a client start consuming results before the full batch finishes
    rendering — useful for very large batches where the synchronous
    /predict_batch payload would either time out or pin server memory.
    The validation + scoring still happens up-front in one vectorized
    call; streaming is purely a wire-format choice.
    """
    df = _build_dataframe(body.rows, pipeline)
    results = pipeline.predict(df)

    def _emit() -> Iterator[bytes]:
        for r in results:
            payload = _to_response(r).model_dump()
            yield (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")

    return StreamingResponse(_emit(), media_type="application/x-ndjson")


@router.get("/clusters", response_model=ClustersResponse, tags=["meta"])
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
