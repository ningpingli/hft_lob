"""Application service for dataset-level shared baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from hft_lob.configs import BaselineRunConfig, load_baseline_config
from hft_lob.data_pipeline.dataset_validator import (
    load_dataset_package,
    stable_config_hash,
)
from hft_lob.metrics.metrics import build_evaluation_report
from hft_lob.reporting.artifact import save_prediction_artifact
from hft_lob.reporting.reporter import save_evaluation_outputs
from hft_lob.systems.baseline_manifest import (
    BaselineArtifactReference,
    BaselineManifest,
    artifact_file_sha256,
    baseline_run_root,
    baseline_space,
    default_manifest_path,
    load_default_manifest,
    save_default_manifest,
    validate_default_manifest,
)
from hft_lob.systems.executor import DefaultWalkForwardExecutor
from hft_lob.systems.walk_forward import select_package_folds
from hft_lob.utils.experiment_manager import resolve_experiment_id
from hft_lob.utils.seed import set_seed


@dataclass(frozen=True)
class BaselineRunRequest:
    """CLI/API request for one shared baseline run."""

    config_path: str
    dataset_dir: str
    experiment_id: str | None = None
    seed: int | None = None
    replace_default: bool = False


@dataclass(frozen=True)
class BaselineRunResult:
    """Minimal result returned after publishing a baseline manifest."""

    experiment_id: str
    dataset_version: str
    artifact_count: int
    manifest_path: str


def run_baseline_application(request: BaselineRunRequest) -> BaselineRunResult:
    """Generate all configured baseline/fold artifacts and publish the default manifest."""
    provisional_id = request.experiment_id or "baseline-pending"
    config = load_baseline_config(request.config_path, experiment_id=provisional_id)
    package = load_dataset_package(request.dataset_dir)
    selected_folds = select_package_folds(package.root, config.folds)
    config = replace(config, seed=config.seed if request.seed is None else request.seed)
    config_hash = _baseline_config_hash(config)
    manifest_path = default_manifest_path(package.metadata.dataset_id)

    if manifest_path.is_file() and not request.replace_default:
        existing = load_default_manifest(package.metadata.dataset_id)
        if existing.config_hash != config_hash:
            raise ValueError(
                "default baseline config hash differs; rerun with --replace-default to publish a new default"
            )
        validate_default_manifest(package, fold_indices=selected_folds)
        return BaselineRunResult(
            experiment_id=existing.experiment_id,
            dataset_version=existing.dataset_id,
            artifact_count=len(existing.artifacts),
            manifest_path=str(manifest_path.resolve()),
        )

    experiment_id = resolve_experiment_id(
        model_name="baseline",
        ticker=package.metadata.ticker,
        override_id=request.experiment_id,
    )
    config = replace(config, experiment_id=experiment_id)
    set_seed(config.seed)
    output_root = baseline_run_root(package.metadata.dataset_id, experiment_id)
    executor = DefaultWalkForwardExecutor(str(output_root))
    references: list[BaselineArtifactReference] = []
    for fold_index in selected_folds:
        for baseline_name in config.baselines.names:
            run = executor.run_baseline_candidate(
                package=package,
                config=config,
                fold_index=fold_index,
                candidate_name=baseline_name,
            )
            artifact = run.artifact
            if (
                artifact.dataset_version != package.metadata.dataset_id
                or artifact.fold_index != fold_index
                or artifact.model_name != baseline_name
                or artifact.split != "test"
            ):
                raise ValueError("baseline executor returned an artifact with mismatched identity")
            predictions_path = save_prediction_artifact(
                artifact=artifact,
                path=run.predictions_path,
            )
            report = build_evaluation_report(
                artifact,
                config.evaluation,
            )
            outputs = save_evaluation_outputs(report, Path(predictions_path).parent)
            root = baseline_space(package.metadata.dataset_id).resolve()
            prediction_file = Path(predictions_path).resolve()
            evaluation_file = Path(outputs["evaluation_report"]).resolve()
            references.append(
                BaselineArtifactReference(
                    fold_index=fold_index,
                    baseline_name=baseline_name,
                    predictions_path=prediction_file.relative_to(root).as_posix(),
                    predictions_sha256=artifact_file_sha256(prediction_file),
                    evaluation_path=evaluation_file.relative_to(root).as_posix(),
                    evaluation_sha256=artifact_file_sha256(evaluation_file),
                )
            )

    manifest = BaselineManifest(
        dataset_id=package.metadata.dataset_id,
        experiment_id=experiment_id,
        config_hash=config_hash,
        fold_indices=selected_folds,
        baseline_names=config.baselines.names,
        artifacts=tuple(references),
    )
    save_default_manifest(manifest)
    return BaselineRunResult(
        experiment_id=experiment_id,
        dataset_version=manifest.dataset_id,
        artifact_count=len(manifest.artifacts),
        manifest_path=str(manifest_path.resolve()),
    )


def _baseline_config_hash(config: BaselineRunConfig) -> str:
    values = asdict(config)
    values.pop("experiment_id", None)
    return stable_config_hash(values)
