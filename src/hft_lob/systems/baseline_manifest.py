"""Dataset-level manifest for reusable baseline experiments."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from hft_lob.datasets.dataset_validator import DatasetPackage
from hft_lob.systems.artifact import load_prediction_artifact

_RESULTS_ROOT = Path("loggers") / "results"


@dataclass(frozen=True)
class BaselineArtifactReference:
    """One baseline/fold artifact registered in the default manifest."""

    fold_index: int
    baseline_name: str
    predictions_path: str
    evaluation_path: str
    overall: dict[str, float]
    mean_daily_ic: float


@dataclass(frozen=True)
class BaselineManifest:
    """Authoritative default baseline reference for one dataset."""

    dataset_id: str
    experiment_id: str
    config_hash: str
    fold_indices: tuple[int, ...]
    baseline_names: tuple[str, ...]
    artifacts: tuple[BaselineArtifactReference, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["fold_indices"] = list(self.fold_indices)
        value["baseline_names"] = list(self.baseline_names)
        value["artifacts"] = [asdict(item) for item in self.artifacts]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BaselineManifest:
        required = {
            "dataset_id",
            "experiment_id",
            "config_hash",
            "fold_indices",
            "baseline_names",
            "artifacts",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"baseline manifest missing fields: {missing}")
        artifacts = tuple(
            BaselineArtifactReference(
                fold_index=int(item["fold_index"]),
                baseline_name=str(item["baseline_name"]),
                predictions_path=str(item["predictions_path"]),
                evaluation_path=str(item["evaluation_path"]),
                overall={str(key): float(metric) for key, metric in item["overall"].items()},
                mean_daily_ic=float(item["mean_daily_ic"]),
            )
            for item in value["artifacts"]
        )
        return cls(
            dataset_id=str(value["dataset_id"]),
            experiment_id=str(value["experiment_id"]),
            config_hash=str(value["config_hash"]),
            fold_indices=tuple(int(index) for index in value["fold_indices"]),
            baseline_names=tuple(str(name) for name in value["baseline_names"]),
            artifacts=artifacts,
        )


def baseline_space(dataset_id: str) -> Path:
    """Return the dataset-level baseline experiment space."""
    _validate_component(dataset_id, field="dataset_id")
    return _RESULTS_ROOT / dataset_id / "baseline"


def default_manifest_path(dataset_id: str) -> Path:
    return baseline_space(dataset_id) / "manifest.yaml"


def baseline_run_root(dataset_id: str, experiment_id: str) -> Path:
    _validate_component(experiment_id, field="experiment_id")
    return baseline_space(dataset_id) / "runs" / experiment_id


def load_default_manifest(dataset_id: str) -> BaselineManifest:
    path = default_manifest_path(dataset_id)
    if not path.is_file():
        raise FileNotFoundError(f"default baseline manifest not found: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid baseline manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"baseline manifest root must be a mapping: {path}")
    return BaselineManifest.from_dict(value)


def save_default_manifest(manifest: BaselineManifest) -> Path:
    destination = default_manifest_path(manifest.dataset_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as temporary:
            temporary_path = Path(temporary.name)
            yaml.safe_dump(manifest.to_dict(), temporary, allow_unicode=True, sort_keys=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def validate_default_manifest(
    package: DatasetPackage,
    *,
    fold_indices: tuple[int, ...],
) -> BaselineManifest:
    """Validate the default manifest and every referenced artifact before training."""
    manifest = load_default_manifest(package.metadata.dataset_id)
    if manifest.dataset_id != package.metadata.dataset_id:
        raise ValueError("baseline manifest dataset_id does not match the dataset package")
    missing_folds = sorted(set(fold_indices).difference(manifest.fold_indices))
    if missing_folds:
        raise ValueError(f"baseline manifest is missing requested folds: {missing_folds}")
    if not manifest.baseline_names:
        raise ValueError("baseline manifest must contain at least one baseline")
    expected = {
        (fold_index, baseline_name)
        for fold_index in fold_indices
        for baseline_name in manifest.baseline_names
    }
    references = {(item.fold_index, item.baseline_name): item for item in manifest.artifacts}
    missing = sorted(expected.difference(references))
    if missing:
        raise ValueError(f"baseline manifest is missing artifacts: {missing}")
    root = baseline_space(manifest.dataset_id).resolve()
    for key in expected:
        reference = references[key]
        prediction_path = (root / reference.predictions_path).resolve()
        try:
            prediction_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("baseline manifest contains a path outside its dataset space") from exc
        artifact = load_prediction_artifact(str(prediction_path))
        if artifact.dataset_version != manifest.dataset_id:
            raise ValueError(f"baseline artifact dataset mismatch: {prediction_path}")
        if artifact.fold_index != reference.fold_index or artifact.model_name != reference.baseline_name:
            raise ValueError(f"baseline artifact identity mismatch: {prediction_path}")
    return manifest


def _validate_component(value: str, *, field: str) -> None:
    if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be one path component")
