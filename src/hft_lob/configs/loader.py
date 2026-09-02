"""YAML 实验配置加载。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hft_lob.configs.experiment import (
    BaselineConfig,
    BaselineRunConfig,
    CleaningConfig,
    DataBuildConfig,
    DataConfig,
    EvaluationConfig,
    FeatureConfig,
    FoldSelectionConfig,
    LoaderConfig,
    ModelConfig,
    ModelRunConfig,
    NormalizationConfig,
    SessionConfig,
    SplitConfig,
    TargetConfig,
    TaskConfig,
    TrainingConfig,
    WalkForwardConfig,
    WindowConfig,
)

_DATA_SECTIONS = {
    "task", "data", "cleaning", "target", "sessions", "window", "features",
    "normalization", "split", "walk_forward",
}
_MODEL_SECTIONS = {"loader", "model", "training", "evaluation", "folds", "seed"}
_BASELINE_SECTIONS = {"loader", "baselines", "evaluation", "folds", "seed"}

def load_data_config(config_path: str) -> DataBuildConfig:
    """加载阶段一配置；拒绝混入任何训练字段。"""
    raw = _load_mapping(config_path, _DATA_SECTIONS)
    if "task" not in raw:
        raise ValueError("config.task is required")
    sessions = _section(raw, "sessions")
    _tuple_field(sessions, "morning")
    _tuple_field(sessions, "afternoon")
    features = _section(raw, "features")
    _tuple_field(features, "derived_features")
    target = _section(raw, "target")
    split = _section(raw, "split")
    for name in ("train_dates", "validation_dates", "test_dates"):
        _tuple_field(split, name, allow_none=True)
    try:
        config = DataBuildConfig(
            task=TaskConfig(**_section(raw, "task")),
            data=DataConfig(**_section(raw, "data")),
            cleaning=CleaningConfig(**_section(raw, "cleaning")),
            target=TargetConfig(**target),
            sessions=SessionConfig(**sessions),
            window=WindowConfig(**_section(raw, "window")),
            features=FeatureConfig(**features),
            normalization=NormalizationConfig(**_section(raw, "normalization")),
            split=SplitConfig(**split),
            walk_forward=WalkForwardConfig(**_section(raw, "walk_forward")),
        )
    except TypeError as exc:
        raise ValueError(f"invalid config field: {exc}") from exc
    _validate_data_config(config)
    return config


def load_model_config(config_path: str, *, experiment_id: str) -> ModelRunConfig:
    """加载阶段二模型配置；不包含 baseline 生成配置。"""
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    raw = _load_mapping(config_path, _MODEL_SECTIONS)
    training = _section(raw, "training")
    _tuple_field(training, "betas")
    evaluation = _section(raw, "evaluation")
    _tuple_field(evaluation, "metrics")
    try:
        config = ModelRunConfig(
            experiment_id=experiment_id,
            loader=LoaderConfig(**_section(raw, "loader")),
            model=ModelConfig(**_section(raw, "model")),
            training=TrainingConfig(**training),
            evaluation=EvaluationConfig(**evaluation),
            folds=FoldSelectionConfig(**_section(raw, "folds")),
            seed=raw.get("seed", 42),
        )
    except TypeError as exc:
        raise ValueError(f"invalid config field: {exc}") from exc
    _validate_model_config(config)
    return config


def load_baseline_config(config_path: str, *, experiment_id: str) -> BaselineRunConfig:
    """加载独立 baseline 实验配置。"""
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    raw = _load_mapping(config_path, _BASELINE_SECTIONS)
    baselines = _section(raw, "baselines")
    _tuple_field(baselines, "names")
    evaluation = _section(raw, "evaluation")
    _tuple_field(evaluation, "metrics")
    try:
        config = BaselineRunConfig(
            experiment_id=experiment_id,
            loader=LoaderConfig(**_section(raw, "loader")),
            baselines=BaselineConfig(**baselines),
            evaluation=EvaluationConfig(**evaluation),
            folds=FoldSelectionConfig(**_section(raw, "folds")),
            seed=raw.get("seed", 42),
        )
    except TypeError as exc:
        raise ValueError(f"invalid config field: {exc}") from exc
    _validate_baseline_config(config)
    return config
    _validate_model_config(config)
    return config


def _load_mapping(config_path: str, allowed: set[str]) -> dict[str, Any]:
    source = Path(config_path)
    if not source.is_file():
        raise FileNotFoundError(config_path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {source}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("config root must be a mapping")
    raw = dict(loaded)
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unknown config sections: {unknown}")
    return raw


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{name} must be a mapping")
    return dict(value)


def _tuple_field(values: dict[str, Any], name: str, *, allow_none: bool = False) -> None:
    if name not in values or (allow_none and values[name] is None):
        return
    value = values[name]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config field {name} must be a sequence")
    values[name] = tuple(value)


def _validate_data_config(config: DataBuildConfig) -> None:
    if not config.ticker.strip():
        raise ValueError("task.ticker must not be empty")
    if config.task.task_type != "regression":
        raise ValueError("task.task_type must be 'regression'")
    positive = {
        "data.levels": config.data.levels,
        "data.snapshot_interval_seconds": config.data.snapshot_interval_seconds,
        "target.label_count": config.target.target_count,
        "window.history_snapshots": config.window.history_snapshots,
    }
    invalid = [name for name, value in positive.items() if not isinstance(value, int) or value <= 0]
    if invalid:
        raise ValueError(f"config integer fields must be > 0: {invalid}")
    if config.target.type not in {"log_mid_return", "simple_mid_return"}:
        raise ValueError("target.type must be 'log_mid_return' or 'simple_mid_return'")
    if config.target.tolerance_seconds < 0:
        raise ValueError("target.tolerance_seconds must be >= 0")
    if not isinstance(config.data.raw_dir, str) or not config.data.raw_dir.strip():
        raise ValueError("data.raw_dir must be a non-empty path")


def _validate_model_config(config: ModelRunConfig) -> None:
    if config.loader.batch_size <= 0 or config.training.epochs <= 0:
        raise ValueError("loader.batch_size and training.epochs must be > 0")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if config.model.name.strip() == "":
        raise ValueError("model.name must not be empty")
    if config.model.output_dim != 1:
        raise ValueError("model.output_dim must be 1 for scalar model contract")
    if config.training.monitor_mode not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be 'min' or 'max'")
    if len(config.training.betas) != 2 or any(
        isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0 <= beta < 1
        for beta in config.training.betas
    ):
        raise ValueError("training.betas must contain two values in [0, 1)")
    if len(config.evaluation.metrics) == 0:
        raise ValueError("evaluation.metrics must not be empty")

def _validate_baseline_config(config: BaselineRunConfig) -> None:
    if config.loader.batch_size <= 0:
        raise ValueError("loader.batch_size must be > 0")
    if not config.baselines.names:
        raise ValueError("baselines.names must not be empty")
    if len(set(config.baselines.names)) != len(config.baselines.names):
        raise ValueError("baselines.names must be unique")
    allowed = {"zero", "imbalance", "ridge"}
    unknown = sorted(set(config.baselines.names).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported baseline names: {unknown}")
    if config.baselines.ridge_alpha <= 0:
        raise ValueError("baselines.ridge_alpha must be > 0")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if len(config.evaluation.metrics) == 0:
        raise ValueError("evaluation.metrics must not be empty")
