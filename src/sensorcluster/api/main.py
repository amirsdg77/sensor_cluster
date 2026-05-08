"""FastAPI inference service entrypoint.

Builds and exposes the ``app`` object that uvicorn serves. The pipeline is
loaded once at startup via the lifespan handler, requests are tagged with
an X-Request-ID via middleware, and Prometheus metrics are served at
``/metrics``.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from sensorcluster.api.dependencies import resolve_artifacts_dir
from sensorcluster.api.routes import router
from sensorcluster.logging_setup import configure as configure_logging
from sensorcluster.logging_setup import get_logger
from sensorcluster.models.pipeline_model import SensorClusterPipeline

# An incoming X-Request-ID is echoed back only when it matches the RFC 4122
# UUID shape; anything else is replaced with a fresh uuid4 to keep the field
# safe to feed into log lines and downstream tracing.
_REQUEST_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
REQUEST_ID_HEADER = "X-Request-ID"

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the pipeline before the server accepts traffic.

    A failure here propagates out of uvicorn's startup, which is the
    behaviour we want: a server with a missing or corrupt pipeline must
    refuse to bind rather than 500 on the first request.
    """
    artifacts_dir = resolve_artifacts_dir()
    log.info("pipeline_loading", artifacts_dir=str(artifacts_dir))
    app.state.pipeline = SensorClusterPipeline.load(artifacts_dir)
    log.info(
        "pipeline_loaded",
        model_version=app.state.pipeline.model_version,
        trained_at=app.state.pipeline.trained_at,
    )
    yield
    # The pipeline holds only in-memory state; nothing to release on shutdown.
    log.info("pipeline_unloaded")


def create_app() -> FastAPI:
    """Construct the FastAPI app with logging, lifespan, middleware, and routes."""
    configure_logging(
        level=os.environ.get("SENSORCLUSTER_LOG_LEVEL", "INFO"),
        json=os.environ.get("SENSORCLUSTER_LOG_JSON", "false").lower() == "true",
    )

    app = FastAPI(
        title="sensorcluster",
        version="0.1.0",
        description=(
            "Inference API for the sensorcluster model: returns cluster assignment, "
            "confidence (HDBSCAN membership strength), and novelty score (GLOSH) per "
            "sensor reading."
        ),
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Tag every request with an X-Request-ID and bind it to log context.

        A valid client-supplied header is echoed; otherwise a fresh uuid4 is
        minted. The id is bound on the structlog contextvars for the duration
        of the request so every log line emitted while handling it carries
        ``request_id`` automatically.
        """
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.include_router(router)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
