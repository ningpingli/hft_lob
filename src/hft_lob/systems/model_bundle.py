"""Self-contained trained-model directory contract."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath

import torch
import yaml

from hft_lob.configs import load_model_config
from hft_lob.configs.experiment import ModelRunConfig
from hft_lob.data_pipeline.dataset_validator import DatasetPackageMetadata
from hft_lob.utils._yaml_io import atomic_dump_yaml

MODEL_BUNDLE_SCHEMA_VERSION = 1
MODEL_CONFIG_FILENAME = "model_config.yaml"
MODEL_METADATA_FILENAME = "model_metadata.yaml"


@dataclass(frozen=True)
class ModelDataContract:
    """Dataset fields that must match before loading model weights."""

    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_dtype: str
    target_dtype: str
    snapshot_interval_seconds: int
    history_snapshots: int
    normalization_mode: str
    normalization_window: int
    labels: tuple[int, ...]
    dataset_schema_version: int

    def __post_init__(self) -> None:
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("model feature_columns must be non-empty and unique")
        if not self.target_columns or len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("model target_columns must be non-empty and unique")
        if not self.feature_dtype.strip() or not self.target_dtype.strip():
            raise ValueError("model data dtypes must not be empty")
        if not self.normalization_mode.strip():
            raise ValueError("model normalization_mode must not be empty")
        if self.snapshot_interval_seconds <= 0 or self.history_snapshots <= 0:
            raise ValueError("model snapshot interval and history must be > 0")
        if self.normalization_window < 2:
            raise ValueError("model normalization_window must be >= 2")
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("model labels must be non-empty and unique")
        if any(
            not isinstance(label, int) or isinstance(label, bool) or label <= 0
            for label in self.labels
        ):
            raise ValueError("model labels must contain positive integers")
        if self.dataset_schema_version <= 0:
            raise ValueError("model dataset_schema_version must be > 0")

    @classmethod
    def from_dataset_metadata(cls, metadata: DatasetPackageMetadata) -> ModelDataContract:
        return cls(
            feature_columns=metadata.feature_columns,
            target_columns=metadata.target_columns,
            feature_dtype=metadata.feature_dtype,
            target_dtype=metadata.target_dtype,
            snapshot_interval_seconds=metadata.snapshot_interval_seconds,
            history_snapshots=metadata.history_snapshots,
            normalization_mode=metadata.normalization_mode,
            normalization_window=metadata.normalization_window,
            labels=metadata.labels,
            dataset_schema_version=metadata.schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["feature_columns"] = list(self.feature_columns)
        value["target_columns"] = list(self.target_columns)
        value["labels"] = list(self.labels)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ModelDataContract:
        _require_exact_fields(value, cls.__dataclass_fields__, context="model data contract")
        feature_columns = _string_list(value["feature_columns"], field="feature_columns")
        target_columns = _string_list(value["target_columns"], field="target_columns")
        labels = _integer_list(value["labels"], field="labels")
        try:
            return cls(
                feature_columns=feature_columns,
                target_columns=target_columns,
                feature_dtype=str(value["feature_dtype"]),
                target_dtype=str(value["target_dtype"]),
                snapshot_interval_seconds=_integer(
                    value["snapshot_interval_seconds"], field="snapshot_interval_seconds"
                ),
                history_snapshots=_integer(value["history_snapshots"], field="history_snapshots"),
                normalization_mode=str(value["normalization_mode"]),
                normalization_window=_integer(
                    value["normalization_window"], field="normalization_window"
                ),
                labels=labels,
                dataset_schema_version=_integer(
                    value["dataset_schema_version"], field="dataset_schema_version"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid model data contract: {exc}") from exc


@dataclass(frozen=True)
class ModelBundleMetadata:
    """Identity and checkpoint location for one trained fold/model."""

    schema_version: int
    model_name: str
    model_version: str
    fold_index: int
    checkpoint_file: str
    training_dataset_version: str
    data_contract: ModelDataContract

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported model bundle schema: {self.schema_version}")
        for field, value in (
            ("model_name", self.model_name),
            ("model_version", self.model_version),
            ("training_dataset_version", self.training_dataset_version),
        ):
            _validate_component(value, field=field)
        if self.fold_index <= 0:
            raise ValueError("model fold_index must be > 0")
        checkpoint = PurePosixPath(self.checkpoint_file)
        if (
            checkpoint.is_absolute()
            or not checkpoint.parts
            or any(part in {"", ".", ".."} for part in checkpoint.parts)
            or "\\" in self.checkpoint_file
            or any(":" in part for part in checkpoint.parts)
            or checkpoint.suffix.lower() != ".ckpt"
        ):
            raise ValueError("checkpoint_file must be a safe relative .ckpt path")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "fold_index": self.fold_index,
            "checkpoint_file": self.checkpoint_file,
            "training_dataset_version": self.training_dataset_version,
            "data_contract": self.data_contract.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ModelBundleMetadata:
        _require_exact_fields(value, cls.__dataclass_fields__, context="model metadata")
        raw_contract = value["data_contract"]
        if not isinstance(raw_contract, dict):
            raise TypeError("model metadata data_contract must be a mapping")
        return cls(
            schema_version=_integer(value["schema_version"], field="schema_version"),
            model_name=str(value["model_name"]),
            model_version=str(value["model_version"]),
            fold_index=_integer(value["fold_index"], field="fold_index"),
            checkpoint_file=str(value["checkpoint_file"]),
            training_dataset_version=str(value["training_dataset_version"]),
            data_contract=ModelDataContract.from_dict(raw_contract),
        )


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    metadata: ModelBundleMetadata
    config: ModelRunConfig
    checkpoint_path: Path


def save_model_bundle(
    model_dir: str | Path,
    *,
    config: ModelRunConfig,
    dataset_metadata: DatasetPackageMetadata,
    checkpoint_path: str | Path,
    model_version: str,
    fold_index: int,
) -> ModelBundle:
    """Persist sidecars beside an existing Lightning checkpoint."""
    root = Path(model_dir).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        relative_checkpoint = checkpoint.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("checkpoint must be contained in model_dir") from exc

    config_value = asdict(config)
    config_value.pop("experiment_id")
    atomic_dump_yaml(root / MODEL_CONFIG_FILENAME, config_value)
    metadata = ModelBundleMetadata(
        schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
        model_name=config.model.name,
        model_version=model_version,
        fold_index=fold_index,
        checkpoint_file=relative_checkpoint,
        training_dataset_version=dataset_metadata.dataset_id,
        data_contract=ModelDataContract.from_dataset_metadata(dataset_metadata),
    )
    atomic_dump_yaml(root / MODEL_METADATA_FILENAME, metadata.to_dict())
    # experiment_id 是训练运行上下文，不写入 model_config.yaml；返回的
    # ModelBundle 与后续 load_model_bundle 读到的一致（无实验标识）。
    return ModelBundle(root, metadata, replace(config, experiment_id=""), checkpoint)


def load_model_bundle(model_dir: str | Path) -> ModelBundle:
    """Load and strictly validate one self-contained trained-model directory."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    metadata_path = root / MODEL_METADATA_FILENAME
    config_path = root / MODEL_CONFIG_FILENAME
    if not metadata_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"model directory must contain {MODEL_METADATA_FILENAME} and {MODEL_CONFIG_FILENAME}"
        )
    raw_metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_metadata, dict):
        raise TypeError(f"{MODEL_METADATA_FILENAME} must contain a mapping")
    metadata = ModelBundleMetadata.from_dict(raw_metadata)
    config = load_model_config(str(config_path))
    if config.model.name != metadata.model_name:
        raise ValueError("model_config.yaml model name does not match model_metadata.yaml")
    checkpoint_path = root.joinpath(*PurePosixPath(metadata.checkpoint_file).parts)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    _validate_checkpoint_identity(checkpoint_path, metadata)
    return ModelBundle(root, metadata, config, checkpoint_path)


def validate_model_data_contract(
    expected: ModelDataContract,
    actual: DatasetPackageMetadata,
) -> None:
    """Reject a test dataset whose model-facing fields differ from training."""
    received = ModelDataContract.from_dataset_metadata(actual)
    mismatches = {
        field: (getattr(expected, field), getattr(received, field))
        for field in expected.__dataclass_fields__
        if getattr(expected, field) != getattr(received, field)
    }
    if mismatches:
        raise ValueError(f"test dataset is incompatible with model data contract: {mismatches}")


def _validate_checkpoint_identity(
    checkpoint_path: Path,
    metadata: ModelBundleMetadata,
) -> None:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a mapping")
    hyper_parameters = payload.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise TypeError("checkpoint must contain hyper_parameters")
    expected: dict[str, object] = {
        "dataset_version": metadata.training_dataset_version,
        "model_version": metadata.model_version,
        "fold_index": metadata.fold_index,
    }
    mismatches = {
        field: (expected_value, hyper_parameters.get(field))
        for field, expected_value in expected.items()
        if hyper_parameters.get(field) != expected_value
    }
    if mismatches:
        raise ValueError(f"checkpoint identity does not match model metadata: {mismatches}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint must contain a non-empty state_dict")


def _require_exact_fields(
    value: dict[str, object], expected_fields: Iterable[str], *, context: str
) -> None:
    expected = set(expected_fields)
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise ValueError(f"invalid {context} fields: missing={missing}, unknown={unknown}")


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return tuple(value)


def _integer_list(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{field} must be a list of integers")
    return tuple(value)


def _integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _validate_component(value: str, *, field: str) -> None:
    if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be one path component")
