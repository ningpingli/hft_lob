"""Dataset-level manifest and scalar comparisons for reusable baselines."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
import yaml

from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.data_pipeline.writer import DatasetPackage
from hft_lob.metrics.metrics import EvaluationReport, build_evaluation_report
from hft_lob.reporting.artifact import load_prediction_artifact
from hft_lob.reporting.reporter import load_evaluation_report
from hft_lob.utils._yaml_io import atomic_dump_yaml

_RESULTS_ROOT = Path("output")
BASELINE_MANIFEST_SCHEMA_VERSION = 2
_COMPARISON_METRICS = ("mse", "mae", "mean_daily_ic", "positive_ic_day_ratio")


class FoldEvaluationResult(Protocol):
    """Read-only view of one fold's evaluation report."""

    @property
    def fold_index(self) -> int: ...

    @property
    def evaluation(self) -> EvaluationReport: ...


@dataclass(frozen=True)
class BaselineArtifactReference:
    """One immutable baseline/fold artifact pair registered in the manifest."""

    fold_index: int
    baseline_name: str
    predictions_path: str
    predictions_sha256: str
    evaluation_path: str
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or self.fold_index <= 0:
            raise ValueError("baseline fold_index must be a positive integer")
        _validate_component(self.baseline_name, field="baseline_name")
        _validate_relative_path(self.predictions_path, field="predictions_path")
        _validate_relative_path(self.evaluation_path, field="evaluation_path")
        _validate_sha256(self.predictions_sha256, field="predictions_sha256")
        _validate_sha256(self.evaluation_sha256, field="evaluation_sha256")


@dataclass(frozen=True)
class BaselineManifest:
    """Authoritative default baseline references for one dataset."""

    dataset_id: str
    experiment_id: str
    config_hash: str
    fold_indices: tuple[int, ...]
    baseline_names: tuple[str, ...]
    artifacts: tuple[BaselineArtifactReference, ...]
    schema_version: int = BASELINE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BASELINE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported baseline manifest schema: {self.schema_version}")
        _validate_component(self.dataset_id, field="dataset_id")
        _validate_component(self.experiment_id, field="experiment_id")
        _validate_sha256(self.config_hash, field="config_hash")
        if (
            not self.fold_indices
            or any(isinstance(index, bool) or index <= 0 for index in self.fold_indices)
            or len(set(self.fold_indices)) != len(self.fold_indices)
        ):
            raise ValueError("baseline fold_indices must be non-empty, unique positive integers")
        if not self.baseline_names or len(set(self.baseline_names)) != len(self.baseline_names):
            raise ValueError("baseline_names must be non-empty and unique")
        for name in self.baseline_names:
            _validate_component(name, field="baseline_name")
        expected = {
            (fold_index, baseline_name)
            for fold_index in self.fold_indices
            for baseline_name in self.baseline_names
        }
        actual = {(item.fold_index, item.baseline_name) for item in self.artifacts}
        if len(actual) != len(self.artifacts):
            raise ValueError("baseline manifest contains duplicate artifact references")
        if actual != expected:
            raise ValueError("baseline artifacts must exactly cover every fold and baseline")

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
            "schema_version",
        }
        if set(value) != required:
            missing = sorted(required.difference(value))
            unknown = sorted(set(value).difference(required))
            raise ValueError(f"invalid baseline manifest fields: missing={missing}, unknown={unknown}")
        folds = value["fold_indices"]
        names = value["baseline_names"]
        artifacts = value["artifacts"]
        if not isinstance(folds, list) or not isinstance(names, list) or not isinstance(artifacts, list):
            raise ValueError("baseline fold_indices, baseline_names and artifacts must be lists")
        return cls(
            dataset_id=str(value["dataset_id"]),
            experiment_id=str(value["experiment_id"]),
            config_hash=str(value["config_hash"]),
            fold_indices=tuple(_positive_integer(index, field="fold index") for index in folds),
            baseline_names=tuple(_string(name, field="baseline name") for name in names),
            artifacts=tuple(_artifact_reference(item) for item in artifacts),
            schema_version=_positive_integer(value["schema_version"], field="schema_version"),
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


def artifact_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one persisted baseline artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    atomic_dump_yaml(destination, manifest.to_dict())
    return destination


def validate_default_manifest(
    package: DatasetPackage,
    *,
    fold_indices: tuple[int, ...],
) -> BaselineManifest:
    """Validate the manifest, file hashes, identities and evaluation contents."""
    manifest, _ = _validate_and_collect_reports(package, fold_indices=fold_indices)
    return manifest


def load_validated_reference_reports(
    package: DatasetPackage,
    *,
    fold_indices: tuple[int, ...],
) -> tuple[BaselineManifest, dict[tuple[int, str], EvaluationReport]]:
    """Validate the manifest once and return every persisted reference report.

    The returned mapping is keyed by ``(fold_index, baseline_name)`` and can be
    passed to :func:`build_baseline_comparison` to avoid re-reading and
    re-verifying the reference artifacts.
    """
    return _validate_and_collect_reports(package, fold_indices=fold_indices)


def _validate_and_collect_reports(
    package: DatasetPackage,
    *,
    fold_indices: tuple[int, ...],
) -> tuple[BaselineManifest, dict[tuple[int, str], EvaluationReport]]:
    manifest = load_default_manifest(package.metadata.dataset_id)
    if manifest.dataset_id != package.metadata.dataset_id:
        raise ValueError("baseline manifest dataset_id does not match the dataset package")
    missing_folds = sorted(set(fold_indices).difference(manifest.fold_indices))
    if missing_folds:
        raise ValueError(f"baseline manifest is missing requested folds: {missing_folds}")

    root = baseline_space(manifest.dataset_id).resolve()
    reports: dict[tuple[int, str], EvaluationReport] = {}
    for reference in manifest.artifacts:
        prediction_path = _reference_path(root, reference.predictions_path)
        evaluation_path = _reference_path(root, reference.evaluation_path)
        _verify_hash(prediction_path, reference.predictions_sha256)
        _verify_hash(evaluation_path, reference.evaluation_sha256)
        artifact = load_prediction_artifact(str(prediction_path))
        if artifact.dataset_version != manifest.dataset_id:
            raise ValueError(f"baseline artifact dataset mismatch: {prediction_path}")
        if (
            artifact.fold_index != reference.fold_index
            or artifact.model_name != reference.baseline_name
            or artifact.split != "test"
        ):
            raise ValueError(f"baseline artifact identity mismatch: {prediction_path}")
        persisted_report = load_evaluation_report(evaluation_path)
        expected_report = build_evaluation_report(
            artifact,
            EvaluationConfig(prediction_bins=len(persisted_report.prediction_bins)),
        )
        if not _reports_equal(persisted_report, expected_report):
            raise ValueError(f"baseline evaluation does not match predictions: {evaluation_path}")
        reports[(reference.fold_index, reference.baseline_name)] = persisted_report
    return manifest, reports


def build_baseline_comparison(
    model_fold_results: Sequence[FoldEvaluationResult],
    manifest: BaselineManifest,
    reference_reports: Mapping[tuple[int, str], EvaluationReport] | None = None,
) -> dict[str, dict[str, object]]:
    """Compare the four scalar model metrics with each registered baseline by fold."""
    if not model_fold_results:
        raise ValueError("model_fold_results must not be empty")
    model_folds = [result.fold_index for result in model_fold_results]
    if len(set(model_folds)) != len(model_folds):
        raise ValueError("model fold results must be unique")
    missing_folds = sorted(set(model_folds).difference(manifest.fold_indices))
    if missing_folds:
        raise ValueError(f"baseline manifest is missing model folds: {missing_folds}")

    references = {(item.fold_index, item.baseline_name): item for item in manifest.artifacts}
    comparison: dict[str, dict[str, object]] = {}
    for baseline_name in manifest.baseline_names:
        deltas: dict[str, list[float]] = {metric: [] for metric in _COMPARISON_METRICS}
        wins: dict[str, list[bool]] = {metric: [] for metric in _COMPARISON_METRICS}
        for fold_result in model_fold_results:
            key = (fold_result.fold_index, baseline_name)
            if reference_reports is not None:
                try:
                    baseline_report = reference_reports[key]
                except KeyError as exc:
                    raise ValueError(
                        "reference_reports must cover every manifest fold and baseline"
                    ) from exc
            else:
                reference = references[key]
                baseline_report = _load_reference_report(manifest.dataset_id, reference)
            model_metrics = _report_metrics(fold_result.evaluation)
            baseline_metrics = _report_metrics(baseline_report)
            for metric in _COMPARISON_METRICS:
                model_value = model_metrics[metric]
                baseline_value = baseline_metrics[metric]
                if _finite(model_value, baseline_value):
                    deltas[metric].append(model_value - baseline_value)
                    wins[metric].append(_is_better(metric, model_value, baseline_value))
        comparison[baseline_name] = {
            "model_fold_count": len(model_fold_results),
            "manifest_fold_count": len(manifest.fold_indices),
            "matched_fold_count": {metric: len(values) for metric, values in deltas.items()},
            "fold_delta": {
                metric: float(np.mean(values)) if values else float("nan")
                for metric, values in deltas.items()
            },
            "fold_win_ratio": {
                metric: float(np.mean(values)) if values else float("nan")
                for metric, values in wins.items()
            },
        }
    return comparison


def _artifact_reference(value: object) -> BaselineArtifactReference:
    fields = {
        "fold_index",
        "baseline_name",
        "predictions_path",
        "predictions_sha256",
        "evaluation_path",
        "evaluation_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("baseline artifact reference has an invalid schema")
    return BaselineArtifactReference(
        fold_index=_positive_integer(value["fold_index"], field="fold_index"),
        baseline_name=_string(value["baseline_name"], field="baseline_name"),
        predictions_path=_string(value["predictions_path"], field="predictions_path"),
        predictions_sha256=_string(value["predictions_sha256"], field="predictions_sha256"),
        evaluation_path=_string(value["evaluation_path"], field="evaluation_path"),
        evaluation_sha256=_string(value["evaluation_sha256"], field="evaluation_sha256"),
    )


def _load_reference_report(
    dataset_id: str, reference: BaselineArtifactReference
) -> EvaluationReport:
    path = _reference_path(baseline_space(dataset_id).resolve(), reference.evaluation_path)
    _verify_hash(path, reference.evaluation_sha256)
    return load_evaluation_report(path)


def _report_metrics(report: EvaluationReport) -> dict[str, float]:
    if set(report.overall) != {"mse", "mae"}:
        raise ValueError("evaluation report must contain exactly mse and mae")
    return {
        "mse": report.overall["mse"],
        "mae": report.overall["mae"],
        "mean_daily_ic": report.mean_daily_ic,
        "positive_ic_day_ratio": report.positive_ic_day_ratio,
    }


def _reports_equal(left: EvaluationReport, right: EvaluationReport) -> bool:
    if (
        left.sample_count != right.sample_count
        or left.valid_sample_count != right.valid_sample_count
        or left.valid_day_count != right.valid_day_count
        or set(left.overall) != set(right.overall)
        or len(left.daily_ic) != len(right.daily_ic)
        or len(left.prediction_bins) != len(right.prediction_bins)
    ):
        return False
    numbers = [
        (left.mean_daily_ic, right.mean_daily_ic),
        (left.positive_ic_day_ratio, right.positive_ic_day_ratio),
        *((left.overall[name], right.overall[name]) for name in left.overall),
    ]
    if not all(_same_number(first, second) for first, second in numbers):
        return False
    for first, second in zip(left.daily_ic, right.daily_ic, strict=True):
        if (
            first.trade_date != second.trade_date
            or first.sample_count != second.sample_count
            or not _same_number(first.ic, second.ic)
        ):
            return False
    for left_bin, right_bin in zip(left.prediction_bins, right.prediction_bins, strict=True):
        if (
            left_bin.bin_index != right_bin.bin_index
            or left_bin.sample_count != right_bin.sample_count
            or not _same_number(left_bin.lower_quantile, right_bin.lower_quantile)
            or not _same_number(left_bin.upper_quantile, right_bin.upper_quantile)
            or not _same_number(left_bin.mean_prediction, right_bin.mean_prediction)
            or not _same_number(left_bin.mean_realized_return, right_bin.mean_realized_return)
        ):
            return False
    return True


def _reference_path(root: Path, relative: str) -> Path:
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("baseline manifest contains a path outside its dataset space") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_hash(path: Path, expected: str) -> None:
    if artifact_file_sha256(path) != expected:
        raise ValueError(f"baseline artifact hash mismatch: {path}")


def _finite(left: float, right: float) -> bool:
    return bool(np.isfinite(left) and np.isfinite(right))


def _is_better(metric: str, model_value: float, baseline_value: float) -> bool:
    if metric in {"mse", "mae"}:
        return model_value < baseline_value
    return model_value > baseline_value


def _same_number(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-15
    )


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _validate_relative_path(value: str, *, field: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{field} must be a safe relative POSIX path")


def _validate_component(value: str, *, field: str) -> None:
    if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be one path component")
