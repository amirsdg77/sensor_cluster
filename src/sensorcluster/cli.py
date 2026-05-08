"""Top-level CLI: `sensorcluster {train,predict,evaluate,serve}`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sensorcluster.config import load_settings
from sensorcluster.logging_setup import configure as configure_logging
from sensorcluster.logging_setup import get_logger

app = typer.Typer(
    name="sensorcluster",
    help="Semi-supervised clustering of sensor data with HDBSCAN.",
    no_args_is_help=True,
    add_completion=False,
)

log = get_logger(__name__)

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        readable=True,
        help="Path to the YAML configuration file.",
    ),
]


@app.command("train")
def train_cmd(
    config: ConfigOption = Path("configs/base.yaml"),
    artifacts_dir: Annotated[
        Path | None,
        typer.Option("--artifacts-dir", help="Override artifacts_dir from config."),
    ] = None,
) -> None:
    """Run the end-to-end training pipeline."""
    overrides: dict[str, object] = {}
    if artifacts_dir is not None:
        overrides["artifacts_dir"] = str(artifacts_dir)

    cfg = load_settings(config, overrides=overrides or None)
    configure_logging(level=cfg.logging.level, json=cfg.logging.json_logs)

    from sensorcluster.pipeline.train import train as run_train

    result = run_train(cfg)
    typer.echo("")
    typer.echo(f"[OK] Training complete. Artifacts at: {result.artifacts_dir}")
    typer.echo(f"   Report: {result.report_path}")
    if result.cv is not None:
        typer.echo(f"   CV mean ARI: {result.cv.mean_ari:.3f} ± {result.cv.std_ari:.3f}")
    typer.echo(
        f"   Clusters: {result.internal.n_clusters} "
        f"(silhouette={result.internal.silhouette:.3f}, "
        f"noise={result.internal.noise_fraction:.2%})"
    )


@app.command("predict")
def predict_cmd(
    input: Annotated[Path, typer.Option("--input", "-i", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "data/processed/predictions.parquet"
    ),
    artifacts_dir: Annotated[Path, typer.Option("--artifacts-dir")] = Path("artifacts"),
    config: ConfigOption = Path("configs/base.yaml"),
) -> None:
    """Batch-score a CSV and write predictions to parquet/CSV."""
    cfg = load_settings(config)
    configure_logging(level=cfg.logging.level, json=cfg.logging.json_logs)

    from sensorcluster.pipeline.predict_batch import predict_batch

    out = predict_batch(
        input_path=input,
        output_path=output,
        artifacts_dir=artifacts_dir,
        sensor_columns_prefix=cfg.data.sensor_columns_prefix,
        n_sensors=cfg.data.n_sensors,
        label_column=cfg.data.label_column,
        sensor_min=cfg.data.sensor_min,
        sensor_max=cfg.data.sensor_max,
        valid_labels=tuple(cfg.data.valid_labels),
    )
    typer.echo(
        f"[OK] Wrote {len(out)} predictions to {output} "
        f"(novel: {int(out['is_novel'].sum())}, "
        f"unknown clusters: {int(out['predicted_label'].str.startswith('UNKNOWN_').sum())})"
    )


@app.command("evaluate")
def evaluate_cmd(
    config: ConfigOption = Path("configs/base.yaml"),
) -> None:
    """Show the evaluation report from the most recent training run."""
    cfg = load_settings(config)
    report = cfg.artifacts_dir / "evaluation_report.md"
    if not report.exists():
        raise typer.BadParameter(f"No report found at {report}; run `sensorcluster train` first.")
    typer.echo(report.read_text(encoding="utf-8"))


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
    artifacts_dir: Annotated[Path, typer.Option("--artifacts-dir")] = Path("artifacts"),
) -> None:
    """Start the FastAPI inference service."""
    import os

    import uvicorn

    os.environ.setdefault("SENSORCLUSTER_ARTIFACTS_DIR", str(artifacts_dir))
    uvicorn.run(
        "sensorcluster.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
