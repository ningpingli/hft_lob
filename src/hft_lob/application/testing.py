"""Standalone evaluation from a recorded model experiment."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from hft_lob.configs import load_model_config
from hft_lob.datasets.dataset_validator import load_dataset_package
from hft_lob.models import build_model
from hft_lob.systems.baseline_manifest import validate_default_manifest
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
    if not isinstance(manifest, dict):
        raise ValueError("experiment.yaml must contain a mapping")
    experiment_id = str(manifest["experiment_id"])
    dataset_dir = str(manifest["dataset_dir"])
    package = load_dataset_package(dataset_dir)
    config = load_model_config(str(config_path), experiment_id=experiment_id)
    fold_indices = tuple(int(value) for value in manifest["fold_indices"])
    if fold_indices != select_package_folds(package.root, config.folds):
        raise ValueError("experiment fold metadata does not match model configuration")
    validate_default_manifest(package, fold_indices=fold_indices)

    output_root = experiment_dir / "standalone_test"
    if output_root.exists():
        shutil.rmtree(output_root)
    fold_results: list[dict[str, object]] = []
    checkpoint_by_fold = {int(item["fold_index"]): str(item["checkpoint_path"]) for item in manifest["folds"]}
    for fold_index in fold_indices:
        checkpoint_path = checkpoint_by_fold.get(fold_index)
        if checkpoint_path is None or not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"missing checkpoint for fold {fold_index}")
        output_dir = output_root / f"fold_{fold_index:03d}" / config.model.name
        datamodule = LOBDataModule(
            package,
            fold_index=fold_index,
            loader=config.loader,
            seed=config.seed,
        )
        trainer = build_trainer(str(output_dir), 1, 0, callbacks=[], accelerator="auto", devices=1)
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
        report = build_evaluation_report(artifact, config.evaluation, seed=config.seed + fold_index)
        predictions_path = output_dir / "predictions.parquet"
        from hft_lob.systems.artifact import save_prediction_artifact

        saved_predictions = save_prediction_artifact(artifact=artifact, path=str(predictions_path))
        outputs = save_evaluation_outputs(report, output_dir)
        fold_results.append(
            {
                "fold_index": fold_index,
                "predictions_path": saved_predictions,
                "evaluation": outputs,
                "mean_daily_ic": report.mean_daily_ic,
            }
        )
    write_experiment_log(
        experiment_id,
        "standalone_test",
        {"fold_count": len(fold_results), "folds": fold_results},
    )
    return TestResult(experiment_id, len(fold_results), str(output_root.resolve()))
