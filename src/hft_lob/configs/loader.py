"""YAML 实验配置加载。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hft_lob.configs.experiment import (
    BaselineConfig,
    CleaningConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureConfig,
    LoaderConfig,
    ModelConfig,
    NormalizationConfig,
    SessionConfig,
    SplitConfig,
    TargetConfig,
    TaskConfig,
    TrainingConfig,
    WalkForwardConfig,
    WindowConfig,
)

_SECTIONS = {
    "task", "data", "cleaning", "target", "sessions", "window", "features",
    "normalization", "loader", "model", "baselines", "training", "evaluation",
    "split", "walk_forward", "seed",
}


def load_config(config_path: str, *, experiment_id: str) -> ExperimentConfig:
    """读取、校验 YAML 并组装 ExperimentConfig。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 结构、字段或取值不符合配置契约。
    """
    source = Path(config_path)
    if not source.is_file():
        raise FileNotFoundError(config_path)
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {source}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("config root must be a mapping")
    raw = dict(loaded)
    unknown = sorted(set(raw).difference(_SECTIONS))
    if unknown:
        raise ValueError(f"unknown config sections: {unknown}")
    if "task" not in raw:
        raise ValueError("config.task is required")

    def section(name: str) -> dict[str, Any]:
        value = raw.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"config.{name} must be a mapping")
        return dict(value)

    sessions = section("sessions")
    _tuple_field(sessions, "morning")
    _tuple_field(sessions, "afternoon")
    features = section("features")
    _tuple_field(features, "derived_features")
    baselines = section("baselines")
    _tuple_field(baselines, "names")
    training = section("training")
    _tuple_field(training, "betas")
    evaluation = section("evaluation")
    _tuple_field(evaluation, "metrics")
    split = section("split")
    for name in ("train_dates", "validation_dates", "test_dates"):
        _tuple_field(split, name, allow_none=True)

    try:
        config = ExperimentConfig(
            experiment_id=experiment_id,
            task=TaskConfig(**section("task")),
            data=DataConfig(**section("data")),
            cleaning=CleaningConfig(**section("cleaning")),
            target=TargetConfig(**section("target")),
            sessions=SessionConfig(**sessions),
            window=WindowConfig(**section("window")),
            features=FeatureConfig(**features),
            normalization=NormalizationConfig(**section("normalization")),
            loader=LoaderConfig(**section("loader")),
            model=ModelConfig(**section("model")),
            baselines=BaselineConfig(**baselines),
            training=TrainingConfig(**training),
            evaluation=EvaluationConfig(**evaluation),
            split=SplitConfig(**split),
            walk_forward=WalkForwardConfig(**section("walk_forward")),
            seed=raw.get("seed", 42),
        )
    except TypeError as exc:
        raise ValueError(f"invalid config field: {exc}") from exc
    _validate_config(config)
    return config


def _tuple_field(values: dict[str, Any], name: str, *, allow_none: bool = False) -> None:
    if name not in values or (allow_none and values[name] is None):
        return
    value = values[name]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config field {name} must be a sequence")
    values[name] = tuple(value)


def _validate_config(config: ExperimentConfig) -> None:
    if not config.ticker.strip():
        raise ValueError("task.ticker must not be empty")
    if config.task.task_type != "regression":
        raise ValueError("task.task_type must be 'regression'")
    positive = {
        "data.levels": config.data.levels,
        "data.snapshot_interval_seconds": config.data.snapshot_interval_seconds,
        "cleaning.max_ffill_gap_seconds": config.cleaning.max_ffill_gap_seconds,
        "target.horizon_seconds": config.target.horizon_seconds,
        "window.history_snapshots": config.window.history_snapshots,
        "loader.batch_size": config.loader.batch_size,
        "training.epochs": config.training.epochs,
    }
    invalid = [name for name, value in positive.items() if not isinstance(value, int) or value <= 0]
    if invalid:
        raise ValueError(f"config integer fields must be > 0: {invalid}")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ValueError("seed must be an integer in [0, 2**32)")
    if config.seed < 0 or config.seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if config.target.type not in {"log_mid_return", "simple_mid_return"}:
        raise ValueError("target.type must be 'log_mid_return' or 'simple_mid_return'")
    if config.target.tolerance_seconds < 0:
        raise ValueError("target.tolerance_seconds must be >= 0")
    if config.model.output_dim != 1:
        raise ValueError("model.output_dim must be 1 for regression")
    if config.training.monitor_mode not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be 'min' or 'max'")
    if len(config.training.betas) != 2 or any(
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not 0 <= beta < 1
        for beta in config.training.betas
    ):
        raise ValueError("training.betas must contain two values in [0, 1)")
    for field, value in (
        ("data.raw_dir", config.data.raw_dir),
        ("data.processed_dir", config.data.processed_dir),
        ("data.manifest_dir", config.data.manifest_dir),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty path")
