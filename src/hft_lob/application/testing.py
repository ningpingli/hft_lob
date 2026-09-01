"""Standalone evaluation from a recorded model experiment."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from hft_lob.configs import load_model_config
from hft_lob.datasets.dataset_validator import load_dataset_package
from hft_lob.models import build_model
from hft_lob.systems.artifact import save_prediction_artifact
from hft_lob.systems.evaluation_plots import save_evaluation_outputs
from hft_lob.systems.executor import build_trainer, run_test
from hft_lob.systems.lob_data_module import LOBDataModule
from hft_lob.systems.lob_module import LOBLightningModule
from hft_lob.systems.metrics import build_evaluation_report
from hft_lob.systems.walk_forward import select_package_folds
from hft_lob.utils.experiment_manager import write_experiment_log


@dataclass(frozen=True)
class TestRequest:
    """Standalone test request identified by an experiment directory."""

    experiment_dir: str


@dataclass(frozen=True)
class TestResult:
    experiment_id: str
    fold_count: int
    output_dir: str


def run_standalone_test(request: TestRequest) -> TestResult:
    """Evaluate every recorded fold without fitting or selecting a checkpoint."""
    experiment_dir = Path(request.experiment_dir).resolve()
    manifest_path = experiment_dir / "experiment.yaml"
    config_path = experiment_dir / "config_used.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("experiment directory must contain experiment.yaml and config_used.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("experiment.yaml must contain a mapping")
    required = {
        "experiment_id",
        "dataset_dir",
        "dataset_version",
        "model_name",
        "fold_indices",
        "folds",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"experiment.yaml is missing fields: {missing}")
    experiment_id = _manifest_text(manifest, "experiment_id")
    dataset_dir = _manifest_text(manifest, "dataset_dir")
    dataset_version = _manifest_text(manifest, "dataset_version")
    model_name = _manifest_text(manifest, "model_name")
    package = load_dataset_package(dataset_dir)
    if dataset_version != package.metadata.dataset_id:
        raise ValueError("experiment dataset_version does not match the dataset package")
    config = load_model_config(str(config_path), experiment_id=experiment_id)
    if model_name != config.model.name:
        raise ValueError("experiment model_name does not match model configuration")
    raw_fold_indices = manifest["fold_indices"]
    if not isinstance(raw_fold_indices, (list, tuple)):
        raise ValueError("experiment fold_indices must be a sequence")
    fold_indices = tuple(
        _manifest_fold_index(value, field="fold_indices") for value in raw_fold_indices
    )
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError("experiment fold_indices must be unique")
    if fold_indices != select_package_folds(package.root, config.folds):
        raise ValueError("experiment fold metadata does not match model configuration")
    raw_folds = manifest["folds"]
    if not isinstance(raw_folds, list):
        raise ValueError("experiment folds must be a list")
    checkpoint_by_fold: dict[int, str] = {}
    for item in raw_folds:
        if not isinstance(item, Mapping) or "fold_index" not in item or "checkpoint_path" not in item:
            raise ValueError("experiment fold entries must contain fold_index and checkpoint_path")
        fold_index = _manifest_fold_index(item["fold_index"], field="fold_index")
        if fold_index in checkpoint_by_fold:
            raise ValueError(f"experiment contains duplicate fold metadata: {fold_index}")
        checkpoint_by_fold[fold_index] = _resolve_checkpoint_path(
            item["checkpoint_path"], experiment_dir
        )
    if set(checkpoint_by_fold) != set(fold_indices):
        raise ValueError("experiment fold metadata does not match fold checkpoint metadata")
    for fold_index in fold_indices:
        checkpoint_path = checkpoint_by_fold[fold_index]
        if not checkpoint_path.strip() or not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"missing checkpoint for fold {fold_index}")
    output_root = experiment_dir / "standalone_test"
    staging_root = Path(
        tempfile.mkdtemp(prefix=".standalone_test-", dir=experiment_dir)
    )
    backup_root = experiment_dir / f".standalone_test-backup-{uuid.uuid4().hex}"
    fold_results: list[dict[str, object]] = []
    published = False
    try:
        for fold_index in fold_indices:
            checkpoint_path = checkpoint_by_fold[fold_index]
            output_dir = staging_root / f"fold_{fold_index:03d}" / config.model.name
            datamodule = LOBDataModule(
                package,
                fold_index=fold_index,
                loader=config.loader,
                seed=config.seed,
            )
            trainer = build_trainer(
                str(output_dir), 1, 0, callbacks=[], accelerator="auto", devices=1
            )
            module = LOBLightningModule(
                build_model(
                    config,
                    feature_columns=package.metadata.feature_columns,
                    history_snapshots=package.metadata.history_snapshots,
                ),
                config,
                dataset_version=package.metadata.dataset_id,
                model_version=f"{experiment_id}-standalone-test-fold{fold_index}",
                fold_index=fold_index,
            )
            artifact = run_test(trainer, module, datamodule, checkpoint_path)
            report = build_evaluation_report(
                artifact, config.evaluation, seed=config.seed + fold_index
            )
            predictions_path = output_dir / "predictions.parquet"
            saved_predictions = save_prediction_artifact(
                artifact=artifact, path=str(predictions_path)
            )
            outputs = save_evaluation_outputs(report, output_dir)
            fold_results.append(
                {
                    "fold_index": fold_index,
                    "predictions_path": saved_predictions,
                    "evaluation": outputs,
                    "mean_daily_ic": report.mean_daily_ic,
                }
            )
        if output_root.exists():
            output_root.replace(backup_root)
        staging_root.replace(output_root)
        published = True
    finally:
        if not published:
            if staging_root.exists():
                shutil.rmtree(staging_root)
            if backup_root.exists() and not output_root.exists():
                backup_root.replace(output_root)
        elif backup_root.exists():
            shutil.rmtree(backup_root)
    published_results: list[dict[str, object]] = []
    for result in fold_results:
        evaluation = cast(dict[str, str], result["evaluation"])
        published_results.append(
            {
                **result,
                "predictions_path": _published_path(
                    str(result["predictions_path"]), staging_root, output_root
                ),
                "evaluation": {
                    name: _published_path(path, staging_root, output_root)
                    for name, path in evaluation.items()
                },
            }
        )
    fold_results = published_results
    write_experiment_log(
        experiment_id,
        "standalone_test",
        {"fold_count": len(fold_results), "folds": fold_results},
    )
    return TestResult(experiment_id, len(fold_results), str(output_root.resolve()))


def _manifest_text(manifest: Mapping[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"experiment.yaml field must be a non-empty string: {field}")
    return value


def _published_path(path: str, staging_root: Path, output_root: Path) -> str:
    relative = Path(path).resolve().relative_to(staging_root.resolve())
    return str((output_root / relative).resolve())


def _manifest_fold_index(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"experiment.yaml {field} entries must be positive integers")
    return value


def _resolve_checkpoint_path(value: object, experiment_dir: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("experiment.yaml checkpoint_path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = experiment_dir / path
    return str(path.resolve())
