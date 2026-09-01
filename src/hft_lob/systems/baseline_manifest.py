"""Dataset-level manifest for reusable baseline experiments."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
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
    daily_metrics: tuple[dict[str, object], ...] = ()
    horizon_decay: tuple[dict[str, object], ...] = ()


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
                daily_metrics=tuple(
                    {str(key): value for key, value in record.items()}
                    for record in item.get("daily_metrics", [])
                ),
                horizon_decay=tuple(
                    {str(key): value for key, value in record.items()}
                    for record in item.get("horizon_decay", [])
                ),
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
        prediction_path = _manifest_file(
            root, reference.predictions_path, field="predictions_path"
        )
        evaluation_path = _manifest_file(root, reference.evaluation_path, field="evaluation_path")
        artifact = load_prediction_artifact(str(prediction_path))
        _validate_evaluation_file(evaluation_path, reference)
        if artifact.dataset_version != manifest.dataset_id:
            raise ValueError(f"baseline artifact dataset mismatch: {prediction_path}")
        if artifact.fold_index != reference.fold_index or artifact.model_name != reference.baseline_name:
            raise ValueError(f"baseline artifact identity mismatch: {prediction_path}")
        if artifact.split != "test" or artifact.metadata[0].ticker != package.metadata.ticker:
            raise ValueError(f"baseline artifact split/ticker mismatch: {prediction_path}")
    return manifest


def build_baseline_comparison(
    model_fold_results: Sequence[Any],
    manifest: BaselineManifest,
) -> dict[str, dict[str, object]]:
    """Compare model fold/day metrics against the registered baselines."""
    references = {(item.fold_index, item.baseline_name): item for item in manifest.artifacts}
    comparison: dict[str, dict[str, object]] = {}
    for baseline_name in manifest.baseline_names:
        fold_deltas: dict[str, list[float]] = {}
        fold_wins: dict[str, list[bool]] = {}
        day_wins: dict[str, list[bool]] = {}
        horizon_fold_deltas: dict[str, list[float]] = {}
        horizon_day_wins: dict[str, list[bool]] = {}
        for fold_result in model_fold_results:
            reference = references[(fold_result.fold_index, baseline_name)]
            evaluation = fold_result.evaluation
            model_metrics = {
                **evaluation.overall,
                "mean_daily_ic": evaluation.mean_daily_ic,
            }
            baseline_metrics = {
                **reference.overall,
                "mean_daily_ic": reference.mean_daily_ic,
            }
            for metric, model_value in model_metrics.items():
                baseline_value = baseline_metrics.get(metric)
                if isinstance(baseline_value, (int, float)):
                    baseline_numeric = float(baseline_value)
                    if _finite(model_value, baseline_numeric):
                        fold_deltas.setdefault(metric, []).append(
                            model_value - baseline_numeric
                        )
                        fold_wins.setdefault(metric, []).append(
                            _is_better(metric, model_value, baseline_numeric)
                        )
            baseline_daily = {
                str(item["trade_date"]): item["metrics"]
                for item in reference.daily_metrics
            }
            for daily_record in evaluation.daily:
                daily_baseline_metrics = cast(
                    dict[str, object], baseline_daily.get(daily_record.trade_date, {})
                )
                for metric, model_value in daily_record.metrics.items():
                    raw_baseline_value = daily_baseline_metrics.get(metric)
                    if not isinstance(raw_baseline_value, (int, float)):
                        continue
                    baseline_value = float(raw_baseline_value)
                    if _finite(model_value, baseline_value):
                        day_wins.setdefault(metric, []).append(
                            _is_better(metric, model_value, baseline_value)
                        )
            baseline_horizon = {
                int(cast(int, item["horizon_seconds"])): cast(dict[str, object], item)
                for item in reference.horizon_decay
            }
            for horizon_record in evaluation.horizon_decay:
                baseline_record = baseline_horizon.get(horizon_record.horizon_seconds)
                if baseline_record is None:
                    continue
                baseline_value = float(cast(float, baseline_record["mean_daily_pearson_corr"]))
                if _finite(horizon_record.mean_daily_pearson_corr, baseline_value):
                    key = str(horizon_record.horizon_seconds)
                    horizon_fold_deltas.setdefault(key, []).append(
                        horizon_record.mean_daily_pearson_corr - baseline_value
                    )
                    horizon_day_values = {
                        str(date): float(cast(float, value))
                        for date, value in cast(
                            list[tuple[object, object]], baseline_record.get("daily_values", [])
                        )
                    }
                    for date, model_value in horizon_record.daily_values:
                        baseline_day_value = horizon_day_values.get(date)
                        if isinstance(baseline_day_value, (int, float)) and _finite(
                            model_value, float(baseline_day_value)
                        ):
                            horizon_day_wins.setdefault(key, []).append(
                                model_value > float(baseline_day_value)
                            )
        comparison[baseline_name] = {
            "fold_delta": {
                metric: float(np.mean(values)) for metric, values in fold_deltas.items()
            },
            "fold_win_ratio": {
                metric: float(np.mean(values)) for metric, values in fold_wins.items()
            },
            "day_win_ratio": {
                metric: float(np.mean(values)) for metric, values in day_wins.items()
            },
            "horizon_fold_delta": {
                horizon: float(np.mean(values))
                for horizon, values in horizon_fold_deltas.items()
            },
            "horizon_day_win_ratio": {
                horizon: float(np.mean(values))
                for horizon, values in horizon_day_wins.items()
            },
        }
    return comparison


def _finite(left: float, right: float) -> bool:
    return bool(np.isfinite(left) and np.isfinite(right))


def _is_better(metric: str, model_value: float, baseline_value: float) -> bool:
    if metric in {"mae", "rmse"}:
        return model_value < baseline_value
    return model_value > baseline_value


def _manifest_file(root: Path, relative_path: str, *, field: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"baseline manifest {field} is outside its dataset space") from exc
    if not path.is_file():
        raise ValueError(f"baseline manifest {field} does not exist: {path}")
    return path


def _validate_evaluation_file(
    path: Path, reference: BaselineArtifactReference
) -> None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid baseline evaluation YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"baseline evaluation root must be a mapping: {path}")
    overall = value.get("overall")
    if not isinstance(overall, dict) or set(overall) != set(reference.overall):
        raise ValueError(f"baseline evaluation overall metrics mismatch: {path}")
    for metric, reference_value in reference.overall.items():
        raw_value = overall.get(metric)
        if not isinstance(raw_value, (int, float)) or not np.isclose(
            float(raw_value), reference_value, equal_nan=True
        ):
            raise ValueError(f"baseline evaluation overall metrics mismatch: {path}")
    mean_daily_ic = value.get("mean_daily_ic")
    if not isinstance(mean_daily_ic, (int, float)) or not np.isclose(
        float(mean_daily_ic), reference.mean_daily_ic, equal_nan=True
    ):
        raise ValueError(f"baseline evaluation mean_daily_ic mismatch: {path}")
    if not _daily_records_match(value.get("daily"), reference.daily_metrics):
        raise ValueError(f"baseline evaluation daily metrics mismatch: {path}")
    if not _horizon_records_match(value.get("horizon_decay"), reference.horizon_decay):
        raise ValueError(f"baseline evaluation horizon decay mismatch: {path}")


def _daily_records_match(
    raw_records: object, expected_records: tuple[dict[str, object], ...]
) -> bool:
    if not isinstance(raw_records, list) or len(raw_records) != len(expected_records):
        return False
    for raw, expected in zip(raw_records, expected_records, strict=True):
        if not isinstance(raw, Mapping) or raw.get("trade_date") != expected.get("trade_date"):
            return False
        raw_metrics = raw.get("metrics")
        expected_metrics = expected.get("metrics")
        if not isinstance(raw_metrics, Mapping) or not isinstance(expected_metrics, Mapping):
            return False
        if set(raw_metrics) != set(expected_metrics):
            return False
        if any(
            not _same_float(raw_metrics[name], expected_metrics[name]) for name in expected_metrics
        ):
            return False
    return True


def _horizon_records_match(
    raw_records: object, expected_records: tuple[dict[str, object], ...]
) -> bool:
    if not isinstance(raw_records, list) or len(raw_records) != len(expected_records):
        return False
    for raw, expected in zip(raw_records, expected_records, strict=True):
        if not isinstance(raw, Mapping):
            return False
        for name in ("horizon_seconds", "valid_day_count", "valid_sample_count"):
            if raw.get(name) != expected.get(name):
                return False
        if not _same_float(
            raw.get("mean_daily_pearson_corr"), expected.get("mean_daily_pearson_corr")
        ):
            return False
        raw_daily = raw.get("daily_values")
        expected_daily = expected.get("daily_values")
        if not isinstance(raw_daily, list) or not isinstance(expected_daily, (list, tuple)):
            return False
        if len(raw_daily) != len(expected_daily):
            return False
        for raw_value, expected_value in zip(raw_daily, expected_daily, strict=True):
            if (
                not isinstance(raw_value, (list, tuple))
                or not isinstance(expected_value, (list, tuple))
                or len(raw_value) != 2
                or len(expected_value) != 2
                or raw_value[0] != expected_value[0]
                or not _same_float(raw_value[1], expected_value[1])
            ):
                return False
    return True


def _same_float(raw: object, expected: object) -> bool:
    return (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and bool(np.isclose(float(raw), float(expected), equal_nan=True))
    )


def _validate_component(value: str, *, field: str) -> None:
    if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be one path component")
