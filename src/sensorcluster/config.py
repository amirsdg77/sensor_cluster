"""Pydantic-settings configuration loaded from YAML + env vars.

Precedence (highest first):
    1. Constructor kwargs (tests)
    2. Environment variables prefixed SENSORCLUSTER_ (double underscore = nesting)
    3. YAML file (configs/base.yaml by default)
    4. Field defaults defined here
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrictBaseModel(BaseModel):
    """Base for nested config blocks. Rejects unknown keys so typos in YAML or
    env-var names raise rather than being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictBaseModel):
    path: Path = Path("data/raw/data_sensors.csv")
    sensor_columns_prefix: str = "Sensor "
    n_sensors: int = 20
    label_column: str = "Label"
    sensor_min: float = -1.05
    sensor_max: float = 1.05
    valid_labels: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])


class PreprocessConfig(StrictBaseModel):
    imputer_strategy: str = "median"


class PCAConfig(StrictBaseModel):
    variance_target: float = 0.95


class HDBSCANConfig(StrictBaseModel):
    # Defaults track configs/base.yaml so constructing Settings() without a
    # YAML file produces the same operating point reviewers see when they
    # run `sensorcluster train --config configs/base.yaml`.
    min_cluster_size: int = 8
    min_samples: int = 3
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True


class NoveltyConfig(StrictBaseModel):
    glosh_threshold: float = 0.7


class LabelPropagationConfig(StrictBaseModel):
    purity_warning: float = 0.6
    weighted: bool = True


class EvaluationConfig(StrictBaseModel):
    cv_folds: int = 5
    bootstrap_n_runs: int = 30
    bootstrap_sample_frac: float = 0.8


class InferenceConfig(StrictBaseModel):
    top_neighbors: int = 3


class MLflowConfig(StrictBaseModel):
    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "sensorcluster"


class LoggingConfig(StrictBaseModel):
    level: str = "INFO"
    json_logs: bool = False


class Settings(BaseSettings):
    """Top-level configuration. See `configs/base.yaml` for the canonical defaults."""

    model_config = SettingsConfigDict(
        env_prefix="SENSORCLUSTER_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    random_seed: int = 42
    artifacts_dir: Path = Path("artifacts")
    model_version: str = "0.1.0"
    data: DataConfig = Field(default_factory=DataConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    pca: PCAConfig = Field(default_factory=PCAConfig)
    hdbscan: HDBSCANConfig = Field(default_factory=HDBSCANConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    label_propagation: LabelPropagationConfig = Field(default_factory=LabelPropagationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return data


def load_settings(
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a `Settings` object from a YAML file plus optional overrides.

    Args:
        config_path: Path to a YAML file containing a mapping of config keys.
            When omitted, only field defaults and environment variables apply.
        overrides: Nested dict deep-merged onto the YAML mapping before
            validation. Values here take precedence over the YAML.
    """
    yaml_dict: dict[str, Any] = {}
    if config_path is not None:
        yaml_dict = _load_yaml(Path(config_path))

    if overrides:
        merged_cfg = OmegaConf.merge(OmegaConf.create(yaml_dict), OmegaConf.create(overrides))
        merged: dict[str, Any] = OmegaConf.to_container(merged_cfg, resolve=True)  # type: ignore[assignment]
    else:
        merged = yaml_dict

    return Settings(**merged)
