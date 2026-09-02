"""Standalone evaluation of one trained model on a compatible test dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StandaloneTestRequest:
    """Inputs required to evaluate one self-contained trained-model directory."""

    test_data_dir: str
    model_name: str
    model_dir: str
    output_dir: str | None = None


@dataclass(frozen=True)
class StandaloneTestResult:
    model_name: str
    model_version: str
    dataset_version: str
    sample_count: int
    output_dir: str
    predictions_path: str
    evaluation_path: str


def run_standalone_test(request: StandaloneTestRequest) -> StandaloneTestResult:
    """Load a recorded checkpoint and evaluate it without fitting or checkpoint selection."""
    if not request.model_name.strip():
        raise ValueError("model_name must not be empty")
    from hft_lob.datasets.dataset_validator import fold_index_path, load_dataset_package
    from hft_lob.models import build_model
    from hft_lob.systems.artifact import save_prediction_artifact
    from hft_lob.systems.evaluation_plots import save_evaluation_outputs
    from hft_lob.systems.executor import build_test_trainer, run_test
    from hft_lob.systems.lob_data_module import LOBDataModule
    from hft_lob.systems.lob_module import LOBLightningModule
    from hft_lob.systems.metrics import build_evaluation_report
    from hft_lob.systems.model_bundle import (
        load_model_bundle,
        validate_model_data_contract,
    )

    bundle = load_model_bundle(request.model_dir)
    if request.model_name != bundle.metadata.model_name:
        raise ValueError(
            f"requested model {request.model_name!r} does not match model directory "
            f"{bundle.metadata.model_name!r}"
        )

    package = load_dataset_package(request.test_data_dir)
    validate_model_data_contract(bundle.metadata.data_contract, package.metadata)
    fold_index = bundle.metadata.fold_index
    test_index = fold_index_path(package.root, fold_index, "test")
    if not test_index.is_file():
        raise FileNotFoundError(f"test dataset does not contain fold {fold_index}: {test_index}")

    output_dir = _resolve_output_dir(
        request, bundle.metadata.model_version, package.metadata.dataset_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    datamodule = LOBDataModule(
        package,
        fold_index=fold_index,
        loader=bundle.config.loader,
        seed=bundle.config.seed,
    )
    module = LOBLightningModule(
        build_model(
            bundle.config,
            feature_columns=package.metadata.feature_columns,
            history_snapshots=package.metadata.history_snapshots,
        ),
        bundle.config,
        dataset_version=package.metadata.dataset_id,
        model_version=bundle.metadata.model_version,
        fold_index=fold_index,
    )
    trainer = build_test_trainer(str(output_dir), accelerator="auto", devices=1)
    artifact = run_test(
        trainer,
        module,
        datamodule,
        str(bundle.checkpoint_path),
    )
    predictions_path = save_prediction_artifact(
        artifact=artifact,
        path=str(output_dir / "predictions.parquet"),
    )
    report = build_evaluation_report(
        artifact,
        bundle.config.evaluation,
        seed=bundle.config.seed + fold_index,
    )
    outputs = save_evaluation_outputs(report, output_dir)
    return StandaloneTestResult(
        model_name=bundle.metadata.model_name,
        model_version=bundle.metadata.model_version,
        dataset_version=package.metadata.dataset_id,
        sample_count=report.sample_count,
        output_dir=str(output_dir),
        predictions_path=predictions_path,
        evaluation_path=outputs["evaluation_report"],
    )


def _resolve_output_dir(
    request: StandaloneTestRequest,
    model_version: str,
    dataset_version: str,
) -> Path:
    if request.output_dir is not None:
        if not request.output_dir.strip():
            raise ValueError("output_dir must be None or a non-empty path")
        return Path(request.output_dir).resolve()
    return (Path("loggers/results/standalone_test") / model_version / dataset_version).resolve()
