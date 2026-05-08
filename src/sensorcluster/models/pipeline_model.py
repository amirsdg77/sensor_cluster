"""Bundled inference pipeline: scaler + PCA + HDBSCAN + label_map + neighbor index.

The API loads this as a single object. Persistence is atomic via ``os.replace``
of a staging directory, so a crashed save can never leave a half-written
artifact directory behind.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sensorcluster.features.dimreduce import PCAReducer
from sensorcluster.features.preprocess import Preprocessor
from sensorcluster.models.hdbscan_model import HDBSCANModel
from sensorcluster.models.label_map import ClusterLabelMap


@dataclass
class PredictionResult:
    """One row of inference output."""

    predicted_label: str
    confidence: float
    novelty_score: float
    cluster_id: int
    is_novel: bool
    top_neighbors: list[dict[str, Any]]


class SensorClusterPipeline:
    """End-to-end inference object. One pickle bundle per model version."""

    SCALER_FILE = "scaler.joblib"
    PCA_FILE = "pca.joblib"
    HDBSCAN_FILE = "hdbscan.joblib"
    LABELMAP_FILE = "cluster_label_map.json"
    NEIGHBORS_FILE = "neighbors.joblib"
    META_FILE = "pipeline_meta.json"

    # Bumped when the on-disk artifact layout changes. Loaders refuse other
    # versions outright rather than silently misinterpreting fields.
    META_SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        preprocessor: Preprocessor,
        pca: PCAReducer,
        hdbscan_model: HDBSCANModel,
        label_map: ClusterLabelMap,
        *,
        labeled_neighbors: NearestNeighbors | None = None,
        labeled_neighbor_meta: pd.DataFrame | None = None,
        glosh_threshold: float = 0.7,
        top_neighbors: int = 3,
        trained_at: str | None = None,
        model_version: str = "0.1.0",
    ) -> None:
        self.preprocessor = preprocessor
        self.pca = pca
        self.hdbscan_model = hdbscan_model
        self.label_map = label_map
        self.labeled_neighbors = labeled_neighbors
        self.labeled_neighbor_meta = labeled_neighbor_meta
        self.glosh_threshold = glosh_threshold
        self.top_neighbors = top_neighbors
        self.trained_at = trained_at or datetime.now(UTC).isoformat()
        self.model_version = model_version

    # ---- inference -----------------------------------------------------------
    def _project(self, X: pd.DataFrame) -> np.ndarray:
        scaled = self.preprocessor.transform(X)
        return self.pca.transform(scaled)

    def predict(self, X: pd.DataFrame) -> list[PredictionResult]:
        """Score every row of `X`, returning one `PredictionResult` per row.

        Projection, clustering, novelty, and neighbor lookup all run vectorized
        over the full batch; the trailing loop only assembles dataclass
        instances from the precomputed arrays.
        """
        proj = self._project(X)
        labels, strengths, novelty = self.hdbscan_model.score_batch(proj)
        neighbors_per_row = self._lookup_neighbors_batch(proj)

        results: list[PredictionResult] = []
        for i in range(proj.shape[0]):
            cluster_id = int(labels[i])
            name = self.label_map.name_for(cluster_id)
            is_novel = bool(novelty[i] >= self.glosh_threshold) or cluster_id == -1
            results.append(
                PredictionResult(
                    predicted_label=name,
                    confidence=float(strengths[i]),
                    novelty_score=float(novelty[i]),
                    cluster_id=cluster_id,
                    is_novel=is_novel,
                    top_neighbors=neighbors_per_row[i],
                )
            )
        return results

    def _lookup_neighbors_batch(self, proj: np.ndarray) -> list[list[dict[str, Any]]]:
        """Find the nearest labeled training rows for each projected query row.

        Distances are Euclidean distances in the *PCA-reduced* feature space,
        not raw sensor space; this matches the space the clustering uses.
        Returns a length-N list whose i-th element is a list of neighbor
        descriptors ``{row_index, label, distance}`` for the i-th input row,
        or an empty list when the pipeline has no neighbor index attached.
        """
        if self.labeled_neighbors is None or self.labeled_neighbor_meta is None:
            return [[] for _ in range(proj.shape[0])]

        k = min(self.top_neighbors, self.labeled_neighbor_meta.shape[0])
        if k == 0:
            return [[] for _ in range(proj.shape[0])]

        dists, idxs = self.labeled_neighbors.kneighbors(proj, n_neighbors=k)
        meta = self.labeled_neighbor_meta
        meta_row_index = meta["row_index"].to_numpy()
        meta_label_name = meta["label_name"].to_numpy()

        out: list[list[dict[str, Any]]] = []
        for row in range(proj.shape[0]):
            out.append(
                [
                    {
                        "row_index": int(meta_row_index[int(j)]),
                        "label": str(meta_label_name[int(j)]),
                        "distance": float(d),
                    }
                    for d, j in zip(dists[row], idxs[row], strict=True)
                ]
            )
        return out

    # ---- persistence ---------------------------------------------------------
    def save(self, directory: Path | str) -> None:
        """Atomically persist all sub-artifacts to `directory`.

        Writes happen first into a sibling staging directory; the live
        directory is then swapped in via a single ``os.replace`` (atomic on
        both POSIX and Windows). A crash before the final swap leaves the
        previous artifacts intact; a crash after it leaves the new state.
        """
        directory = Path(directory)
        parent = directory.parent
        parent.mkdir(parents=True, exist_ok=True)

        # Stage on the same filesystem as the target so the rename is a true
        # rename rather than a cross-device copy.
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.staging.", dir=str(parent)))

        try:
            self.preprocessor.save(staging / self.SCALER_FILE)
            self.pca.save(staging / self.PCA_FILE)
            self.hdbscan_model.save(staging / self.HDBSCAN_FILE)
            self.label_map.save(staging / self.LABELMAP_FILE)

            if self.labeled_neighbors is not None and self.labeled_neighbor_meta is not None:
                joblib.dump(
                    {
                        "neighbors": self.labeled_neighbors,
                        "meta": self.labeled_neighbor_meta,
                    },
                    staging / self.NEIGHBORS_FILE,
                    compress=3,
                )

            meta = {
                "schema_version": self.META_SCHEMA_VERSION,
                "glosh_threshold": self.glosh_threshold,
                "top_neighbors": self.top_neighbors,
                "trained_at": self.trained_at,
                "model_version": self.model_version,
            }
            (staging / self.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Carry over files that this save() does not own (e.g. the
            # evaluation_report.md written elsewhere into the same directory).
            # Without this, the rename would erase them.
            if directory.is_dir():
                for existing in directory.iterdir():
                    if not (staging / existing.name).exists():
                        if existing.is_dir():
                            shutil.copytree(existing, staging / existing.name)
                        else:
                            shutil.copy2(existing, staging / existing.name)

            # On Windows, os.replace will not replace a non-empty directory
            # directly, so move the live state aside, swap in the staging
            # directory, and delete the backup last.
            backup: Path | None = None
            if directory.exists():
                backup = parent / f".{directory.name}.old.{os.getpid()}"
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                os.replace(directory, backup)
            os.replace(staging, directory)
            staging = directory
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
        except BaseException:
            if staging.exists() and staging != directory:
                shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def load(cls, directory: Path | str) -> SensorClusterPipeline:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Pipeline directory not found: {directory}")

        preprocessor = Preprocessor.load(directory / cls.SCALER_FILE)
        pca = PCAReducer.load(directory / cls.PCA_FILE)
        hdb = HDBSCANModel.load(directory / cls.HDBSCAN_FILE)
        label_map = ClusterLabelMap.load(directory / cls.LABELMAP_FILE)

        neighbors: NearestNeighbors | None = None
        neighbor_meta: pd.DataFrame | None = None
        n_path = directory / cls.NEIGHBORS_FILE
        if n_path.exists():
            payload = joblib.load(n_path)
            neighbors = payload["neighbors"]
            neighbor_meta = payload["meta"]

        meta_path = directory / cls.META_FILE
        meta: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )
        meta_version = meta.get("schema_version")
        # Tolerate the no-meta case (older bundles); reject unknown schemas.
        if meta_version is not None and meta_version != cls.META_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported pipeline_meta schema_version {meta_version!r}; "
                f"this build expects {cls.META_SCHEMA_VERSION!r}."
            )

        return cls(
            preprocessor=preprocessor,
            pca=pca,
            hdbscan_model=hdb,
            label_map=label_map,
            labeled_neighbors=neighbors,
            labeled_neighbor_meta=neighbor_meta,
            glosh_threshold=float(meta.get("glosh_threshold", 0.7)),
            top_neighbors=int(meta.get("top_neighbors", 3)),
            trained_at=str(meta.get("trained_at", datetime.now(UTC).isoformat())),
            model_version=str(meta.get("model_version", "0.1.0")),
        )
