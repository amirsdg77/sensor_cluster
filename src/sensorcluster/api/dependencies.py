"""FastAPI dependency that returns the singleton pipeline.

The pipeline is loaded once at startup by the lifespan handler in
``main.py`` and stored on ``app.state.pipeline``. Route handlers fetch it
through this dependency, which keeps the load off the request hot path
and removes any runtime caching concerns.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, Request, status

from sensorcluster.models.pipeline_model import SensorClusterPipeline


def resolve_artifacts_dir() -> Path:
    """Return the configured artifacts directory (env-driven)."""
    return Path(os.environ.get("SENSORCLUSTER_ARTIFACTS_DIR", "artifacts"))


def get_pipeline(request: Request) -> SensorClusterPipeline:
    """Return the pipeline cached on the app state.

    Raises 503 if the pipeline is missing; this should not normally happen
    because the lifespan handler refuses to start the server when the
    pipeline cannot be loaded.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not loaded; check server startup logs.",
        )
    return pipeline
