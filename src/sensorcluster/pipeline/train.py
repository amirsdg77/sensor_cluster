"""End-to-end training entrypoint.

Orchestration:
    load CSV  ->  validate  ->  preprocess  ->  PCA  ->  HDBSCAN  ->  build label map
              ->  evaluate (internal + CV + stability)  ->  generate report  ->  save artifacts
              ->  (optional) MLflow log

Returns a `TrainResult` object so the CLI / tests can introspect the run.
"""

from __future__ import annotations

import os
import random as _random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sensorcluster.config import Settings
from sensorcluster.data.loader import load_sensor_data, split_features_label
from sensorcluster.evaluation.cv import CVResult, cv_evaluate
from sensorcluster.evaluation.metrics import (
    InternalMetrics,
    bootstrap_stability,
    compute_internal_metrics,
)
from sensorcluster.evaluation.report import generate_report
from sensorcluster.features.dimreduce import PCAReducer
from sensorcluster.features.preprocess import Preprocessor
from sensorcluster.logging_setup import configure as configure_logging
from sensorcluster.logging_setup import get_logger
from sensorcluster.models.hdbscan_model import HDBSCANModel
from sensorcluster.models.label_map import ClusterLabelMap
from sensorcluster.models.pipeline_model import SensorClusterPipeline
from sensorcluster.pipeline.label_propagation import build_label_map

log = get_logger(__name__)

# Aliases for the closures threaded through CV and bootstrap-stability.
PredictLabelIdFn = Callable[[pd.DataFrame], np.ndarray]
PipelineFactoryFn = Callable[[pd.DataFrame, pd.Series], PredictLabelIdFn]
StabilityFitFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class TrainResult:
    """Output of :func:`train`.

    Attributes:
        pipeline: The bundled inference pipeline that was fit and persisted.
        label_map: Cluster-id -> label-name map produced from the training data.
        internal: Label-free quality metrics for the discovered clustering.
        cv: Cross-validated ARI on the labeled subset, or None if CV was
            skipped (e.g. too few labeled rows).
        stability: Bootstrap stability summary, or None if it was skipped.
        artifacts_dir: Directory where all artifacts were written.
        report_path: Path to the generated ``evaluation_report.md``.
        extras: Run-specific extras (chosen ``n_components``, row counts,
            seed) included verbatim in the report.
    """

    pipeline: SensorClusterPipeline
    label_map: ClusterLabelMap
    internal: InternalMetrics
    cv: CVResult | None
    stability: dict[str, float] | None
    artifacts_dir: Path
    report_path: Path
    extras: dict[str, Any] = field(default_factory=dict)


def _set_seed(seed: int) -> None:
    """Seed Python and NumPy global RNGs plus PYTHONHASHSEED for reproducibility."""
    _random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _label_to_int(name: str) -> int:
    """Encode a label name as a stable integer for ARI computation.

    Mapping: ``NOISE`` -> ``-1``; ``CLASS_n`` -> ``n``; ``UNKNOWN_k`` ->
    ``1000 + k``. Anything else raises ``ValueError`` — the upstream label
    map only ever produces those three shapes, so an unknown name signals a
    bug rather than data we should silently fold into a hashed bucket.
    """
    if name == "NOISE":
        return -1
    if name.startswith("CLASS_"):
        try:
            return int(name.split("_", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Malformed CLASS_* label name: {name!r}") from exc
    if name.startswith("UNKNOWN_"):
        try:
            return 1000 + int(name.split("_", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Malformed UNKNOWN_* label name: {name!r}") from exc
    raise ValueError(f"Unrecognized cluster label name: {name!r}")


def _build_neighbor_index(
    proj: np.ndarray, y: pd.Series, *, top_neighbors: int
) -> tuple[NearestNeighbors | None, pd.DataFrame | None]:
    """Build a k-NN index over labeled rows for the API's ``top_neighbors``
    explainability field. Returns ``(None, None)`` if no rows are labeled."""
    y_arr = y.to_numpy()
    mask = ~pd.isna(y_arr)
    if mask.sum() == 0:
        return None, None
    labeled_proj = proj[mask]
    nn = NearestNeighbors(n_neighbors=min(top_neighbors, labeled_proj.shape[0]))
    nn.fit(labeled_proj)
    meta = pd.DataFrame(
        {
            "row_index": np.where(mask)[0],
            "label_name": [f"CLASS_{int(v)}" for v in y_arr[mask].astype(float)],
        }
    )
    return nn, meta


def _factory_for_cv(cfg: Settings) -> PipelineFactoryFn:
    """Build the ``pipeline_factory`` callable required by :func:`cv_evaluate`.

    The returned callable takes ``(X_train, y_train)``, fits a fresh pipeline
    on those rows, and returns a ``predict_label_id(X_test)`` closure that
    encodes test-point predictions into integer labels suitable for ARI.
    """

    def factory(X_train: pd.DataFrame, y_train: pd.Series) -> PredictLabelIdFn:
        pre = Preprocessor(imputer_strategy=cfg.preprocess.imputer_strategy).fit(X_train)
        Xs = pre.transform(X_train)
        pca = PCAReducer(
            variance_target=cfg.pca.variance_target,
            n_components=cfg.pca.n_components,
            random_state=cfg.random_seed,
        ).fit(Xs)
        Xp = pca.transform(Xs)
        model = HDBSCANModel(
            min_cluster_size=cfg.hdbscan.min_cluster_size,
            min_samples=cfg.hdbscan.min_samples,
            metric=cfg.hdbscan.metric,
            cluster_selection_method=cfg.hdbscan.cluster_selection_method,
            prediction_data=True,
        ).fit(Xp)
        train_clusters = model.labels_
        label_map = build_label_map(
            train_clusters,
            y_train,
            weighted=cfg.label_propagation.weighted,
            purity_warning=cfg.label_propagation.purity_warning,
        )

        def predict_label_id(X_test: pd.DataFrame) -> np.ndarray:
            Xs_te = pre.transform(X_test)
            Xp_te = pca.transform(Xs_te)
            cluster_ids, _ = model.predict_with_strength(Xp_te)
            names = [label_map.name_for(int(c)) for c in cluster_ids]
            return np.array([_label_to_int(n) for n in names], dtype=np.int64)

        return predict_label_id

    return factory


def _stability_fit_fn(cfg: Settings) -> StabilityFitFn:
    """Build the per-bootstrap fitter for :func:`bootstrap_stability`.

    The returned callable fits a fresh HDBSCAN on a subsample of the data
    and returns the cluster labels assigned to that subsample.
    """

    def fit(X_sub: np.ndarray) -> np.ndarray:
        m = HDBSCANModel(
            min_cluster_size=cfg.hdbscan.min_cluster_size,
            min_samples=cfg.hdbscan.min_samples,
            metric=cfg.hdbscan.metric,
            cluster_selection_method=cfg.hdbscan.cluster_selection_method,
            prediction_data=False,
        ).fit(X_sub)
        return m.labels_

    return fit


def train(cfg: Settings) -> TrainResult:
    """Run the full training pipeline end-to-end.

    Loads and validates the input CSV, fits the preprocessor / PCA / HDBSCAN
    chain, builds the cluster->label map, computes evaluation metrics, writes
    all artifacts to ``cfg.artifacts_dir``, generates the report, and (when
    enabled) logs a run to MLflow. Idempotent for a fixed seed and fixed
    input data.
    """
    configure_logging(level=cfg.logging.level, json=cfg.logging.json_logs)
    _set_seed(cfg.random_seed)
    log.info(
        "training_started",
        artifacts_dir=str(cfg.artifacts_dir),
        seed=cfg.random_seed,
    )

    df = load_sensor_data(
        cfg.data.path,
        prefix=cfg.data.sensor_columns_prefix,
        n_sensors=cfg.data.n_sensors,
        sensor_min=cfg.data.sensor_min,
        sensor_max=cfg.data.sensor_max,
        label_column=cfg.data.label_column,
        valid_labels=tuple(cfg.data.valid_labels),
    )
    X, y = split_features_label(
        df,
        prefix=cfg.data.sensor_columns_prefix,
        n_sensors=cfg.data.n_sensors,
        label_column=cfg.data.label_column,
    )

    pre = Preprocessor(imputer_strategy=cfg.preprocess.imputer_strategy).fit(X)
    X_scaled = pre.transform(X)

    pca = PCAReducer(
        variance_target=cfg.pca.variance_target,
        n_components=cfg.pca.n_components,
        random_state=cfg.random_seed,
    ).fit(X_scaled)
    X_proj = pca.transform(X_scaled)
    log.info(
        "pca_fitted",
        n_components=pca.n_components_,
        variance_target=cfg.pca.variance_target,
        n_components_config=cfg.pca.n_components,
    )

    model = HDBSCANModel(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric=cfg.hdbscan.metric,
        cluster_selection_method=cfg.hdbscan.cluster_selection_method,
        prediction_data=True,
    ).fit(X_proj)
    cluster_labels = model.labels_
    log.info(
        "hdbscan_fitted",
        n_clusters=model.n_clusters_,
        noise_count=int((cluster_labels == -1).sum()),
    )

    label_map = build_label_map(
        cluster_labels,
        y,
        weighted=cfg.label_propagation.weighted,
        purity_warning=cfg.label_propagation.purity_warning,
    )

    internal = compute_internal_metrics(X_proj, cluster_labels)
    log.info(
        "internal_metrics",
        silhouette=internal.silhouette,
        davies_bouldin=internal.davies_bouldin,
        noise_fraction=internal.noise_fraction,
        n_clusters=internal.n_clusters,
    )

    cv: CVResult | None = None
    try:
        cv = cv_evaluate(
            X,
            y,
            pipeline_factory=_factory_for_cv(cfg),
            n_splits=cfg.evaluation.cv_folds,
            seed=cfg.random_seed,
        )
        log.info("cv_completed", mean_ari=cv.mean_ari, std_ari=cv.std_ari, n_folds=cv.n_folds)
    except ValueError as exc:
        log.warning("cv_skipped", reason=str(exc))

    stability: dict[str, float] | None = None
    try:
        stability = bootstrap_stability(
            X_proj,
            _stability_fit_fn(cfg),
            n_runs=cfg.evaluation.bootstrap_n_runs,
            sample_frac=cfg.evaluation.bootstrap_sample_frac,
            seed=cfg.random_seed,
        )
        log.info("stability_computed", **stability)
    except (ValueError, RuntimeError, MemoryError) as exc:
        # Bootstrap stability is decorative; failures here must not abort
        # training. ValueError covers degenerate resamples, RuntimeError
        # covers HDBSCAN numeric failures, MemoryError covers oversized runs.
        log.warning("stability_skipped", reason=str(exc))

    nn, nn_meta = _build_neighbor_index(X_proj, y, top_neighbors=cfg.inference.top_neighbors)

    pipeline = SensorClusterPipeline(
        preprocessor=pre,
        pca=pca,
        hdbscan_model=model,
        label_map=label_map,
        labeled_neighbors=nn,
        labeled_neighbor_meta=nn_meta,
        glosh_threshold=cfg.novelty.glosh_threshold,
        top_neighbors=cfg.inference.top_neighbors,
        trained_at=datetime.now(UTC).isoformat(),
        model_version=cfg.model_version,
    )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(cfg.artifacts_dir)

    extras = {
        "pca_n_components": pca.n_components_,
        "pca_variance_target": cfg.pca.variance_target,
        "min_cluster_size": cfg.hdbscan.min_cluster_size,
        "min_samples": cfg.hdbscan.min_samples,
        "n_rows": len(df),
        "n_labeled": int(y.notna().sum()),
        "random_seed": cfg.random_seed,
    }

    umap_plot_path: Path | None = None
    try:
        from sensorcluster.visualization.plots import save_cluster_umap

        umap_plot_path = save_cluster_umap(
            X_embedded=X_proj,
            cluster_labels=cluster_labels,
            y=y,
            label_map=label_map,
            out_path=cfg.artifacts_dir / "evaluation_umap.png",
            seed=cfg.random_seed,
        )
        log.info("umap_plot_saved", path=str(umap_plot_path))
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        # The plot is optional. ImportError covers serve-only installs that
        # lack umap/matplotlib; ValueError/RuntimeError cover unembeddable
        # inputs; OSError covers a non-writable artifacts directory.
        log.warning("umap_plot_skipped", reason=str(exc))

    report_path = generate_report(
        out_dir=cfg.artifacts_dir,
        label_map=label_map,
        internal=internal,
        cv=cv,
        stability=stability,
        extra=extras,
        umap_plot=umap_plot_path,
    )
    log.info("report_written", path=str(report_path))

    if cfg.mlflow.enabled:
        try:
            _log_to_mlflow(cfg, internal, cv, stability, label_map, extras)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            # Tracking is optional; cover the plausible failure modes
            # (missing dependency, unreachable URI, invalid run state) but
            # leave anything more exotic to propagate so it stays visible.
            log.warning("mlflow_log_failed", reason=str(exc))

    return TrainResult(
        pipeline=pipeline,
        label_map=label_map,
        internal=internal,
        cv=cv,
        stability=stability,
        artifacts_dir=cfg.artifacts_dir,
        report_path=report_path,
        extras=extras,
    )


def _log_to_mlflow(
    cfg: Settings,
    internal: InternalMetrics,
    cv: CVResult | None,
    stability: dict[str, float] | None,
    label_map: ClusterLabelMap,
    extras: dict[str, Any],
) -> None:
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "pca_variance_target": cfg.pca.variance_target,
                "hdbscan_min_cluster_size": cfg.hdbscan.min_cluster_size,
                "hdbscan_min_samples": cfg.hdbscan.min_samples,
                "hdbscan_metric": cfg.hdbscan.metric,
                "novelty_glosh_threshold": cfg.novelty.glosh_threshold,
                "label_propagation_weighted": cfg.label_propagation.weighted,
                "random_seed": cfg.random_seed,
                "n_rows": extras.get("n_rows"),
                "n_labeled": extras.get("n_labeled"),
            }
        )
        mlflow.log_metric("internal_silhouette", internal.silhouette)
        mlflow.log_metric("internal_davies_bouldin", internal.davies_bouldin)
        mlflow.log_metric("internal_noise_fraction", internal.noise_fraction)
        mlflow.log_metric("internal_n_clusters", internal.n_clusters)
        if cv is not None:
            mlflow.log_metric("cv_mean_ari", cv.mean_ari)
            mlflow.log_metric("cv_std_ari", cv.std_ari)
            # Per-fold ARI logged as a stepped metric so the MLflow UI shows
            # the distribution across folds rather than only the summary.
            for fold_idx, ari in enumerate(cv.fold_aris):
                mlflow.log_metric("cv_fold_ari", ari, step=fold_idx)
        if stability is not None:
            mlflow.log_metric(
                "stability_mean_ari", stability.get("stability_mean_ari", float("nan"))
            )
        n_unknown = sum(
            1 for e in label_map.entries.values() if (not e.is_known) and (not e.is_noise)
        )
        mlflow.log_metric("n_unknown_clusters", n_unknown)
        mlflow.log_artifacts(str(cfg.artifacts_dir))
