from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hft_lob.baselines.manifest import (
    BASELINE_MANIFEST_SCHEMA_VERSION,
    BaselineArtifactReference,
    BaselineManifest,
    artifact_file_sha256,
    baseline_space,
    build_baseline_comparison,
    default_manifest_path,
    load_default_manifest,
    load_validated_reference_reports,
    save_default_manifest,
    validate_default_manifest,
)
from hft_lob.configs.experiment import EvaluationConfig
from hft_lob.data_pipeline.writer import (
    DatasetPackage,
    DatasetPackageMetadata,
    compute_dataset_id,
)
from hft_lob.data_types import SampleMeta
from hft_lob.metrics.metrics import build_evaluation_report
from hft_lob.reporting.artifact import PredictionArtifact, save_prediction_artifact
from hft_lob.reporting.reporter import save_evaluation_outputs

_CONFIG_HASH = "a" * 64


def test_default_manifest_round_trips_and_validates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, manifest = _published_manifest(tmp_path, monkeypatch)

    loaded = load_default_manifest(package.metadata.dataset_id)
    validated = validate_default_manifest(package, fold_indices=(1,))

    assert loaded == manifest
    assert validated == manifest
    assert default_manifest_path(package.metadata.dataset_id).is_file()


def test_baseline_comparison_covers_four_scalar_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest = _published_manifest(tmp_path, monkeypatch)
    model_report = build_evaluation_report(
        _artifact(_metadata(), model_name="model", predictions=np.array([1.0, 2.0, 1.0, 2.0])),
        EvaluationConfig(prediction_bins=2),
    )

    result = build_baseline_comparison(
        (SimpleNamespace(fold_index=1, evaluation=model_report),),
        manifest,
    )["zero"]

    assert result["model_fold_count"] == 1
    assert result["manifest_fold_count"] == 1
    assert result["matched_fold_count"] == {
        "mse": 1,
        "mae": 1,
        "mean_daily_ic": 1,
        "positive_ic_day_ratio": 1,
    }
    assert result["fold_delta"]["mse"] < 0
    assert result["fold_delta"]["mae"] < 0
    assert result["fold_delta"]["mean_daily_ic"] > 0
    assert result["fold_delta"]["positive_ic_day_ratio"] > 0
    assert set(result["fold_win_ratio"].values()) == {1.0}


def test_load_validated_reference_reports_collects_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, _ = _published_manifest(tmp_path, monkeypatch)

    manifest, reports = load_validated_reference_reports(package, fold_indices=(1,))

    assert set(reports) == {(1, "zero")}
    assert reports[(1, "zero")].sample_count == 4
    assert reports[(1, "zero")].overall == pytest.approx({"mse": 0.5625, "mae": 0.625})
    assert manifest.schema_version == BASELINE_MANIFEST_SCHEMA_VERSION




def test_baseline_comparison_rejects_incomplete_reference_reports() -> None:
    manifest = BaselineManifest(
        dataset_id="dataset",
        experiment_id="baseline",
        config_hash=_CONFIG_HASH,
        fold_indices=(1,),
        baseline_names=("zero",),
        artifacts=(_reference("predictions.parquet", "evaluation.yaml"),),
    )
    model_report = build_evaluation_report(
        _artifact(_metadata(), model_name="model", predictions=np.array([1.0, 2.0, 1.0, 2.0])),
        EvaluationConfig(prediction_bins=2),
    )

    with pytest.raises(ValueError, match="must cover every manifest fold and baseline"):
        build_baseline_comparison(
            (SimpleNamespace(fold_index=1, evaluation=model_report),),
            manifest,
            reference_reports={},
        )


def test_manifest_rejects_old_schema_and_duplicate_references() -> None:
    value = {
        "dataset_id": "dataset",
        "experiment_id": "baseline",
        "config_hash": _CONFIG_HASH,
        "fold_indices": [1],
        "baseline_names": ["zero"],
        "artifacts": [],
    }
    with pytest.raises(ValueError, match="missing=.*schema_version"):
        BaselineManifest.from_dict(value)

    reference = _reference("predictions.parquet", "evaluation.yaml")
    with pytest.raises(ValueError, match="duplicate artifact references"):
        BaselineManifest(
            dataset_id="dataset",
            experiment_id="baseline",
            config_hash=_CONFIG_HASH,
            fold_indices=(1,),
            baseline_names=("zero",),
            artifacts=(reference, reference),
        )


def test_validation_rejects_tampered_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, manifest = _published_manifest(tmp_path, monkeypatch)
    root = baseline_space(package.metadata.dataset_id)
    evaluation_path = root / manifest.artifacts[0].evaluation_path
    evaluation_path.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_default_manifest(package, fold_indices=(1,))


def _published_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DatasetPackage, BaselineManifest]:
    monkeypatch.setattr("hft_lob.baselines.manifest._RESULTS_ROOT", tmp_path / "results")
    metadata = _metadata()
    package = DatasetPackage(root=tmp_path / metadata.dataset_id, metadata=metadata)
    root = baseline_space(metadata.dataset_id)
    output = root / "runs" / "baseline-1" / "fold_001" / "zero"
    artifact = _artifact(
        metadata,
        model_name="zero",
        predictions=np.array([1.0, 1.5, 2.0, 1.0]),
    )
    prediction_path = Path(
        save_prediction_artifact(artifact=artifact, path=str(output / "predictions.parquet"))
    )
    report = build_evaluation_report(artifact, EvaluationConfig(prediction_bins=2))
    evaluation_path = Path(save_evaluation_outputs(report, output)["evaluation_report"])
    reference = BaselineArtifactReference(
        fold_index=1,
        baseline_name="zero",
        predictions_path=prediction_path.relative_to(root).as_posix(),
        predictions_sha256=artifact_file_sha256(prediction_path),
        evaluation_path=evaluation_path.relative_to(root).as_posix(),
        evaluation_sha256=artifact_file_sha256(evaluation_path),
    )
    manifest = BaselineManifest(
        dataset_id=metadata.dataset_id,
        experiment_id="baseline-1",
        config_hash=_CONFIG_HASH,
        fold_indices=(1,),
        baseline_names=("zero",),
        artifacts=(reference,),
        schema_version=BASELINE_MANIFEST_SCHEMA_VERSION,
    )
    save_default_manifest(manifest)
    return package, manifest


def _reference(predictions_path: str, evaluation_path: str) -> BaselineArtifactReference:
    return BaselineArtifactReference(
        fold_index=1,
        baseline_name="zero",
        predictions_path=predictions_path,
        predictions_sha256="b" * 64,
        evaluation_path=evaluation_path,
        evaluation_sha256="c" * 64,
    )


def _artifact(
    metadata: DatasetPackageMetadata,
    *,
    model_name: str,
    predictions: np.ndarray,
) -> PredictionArtifact:
    targets = np.array([[1.0], [2.0], [1.0], [2.0]])
    samples = tuple(
        SampleMeta(
            ticker=metadata.ticker,
            trade_date="2026-01-05" if index < 2 else "2026-01-06",
            session_id="AM",
            anchor_timestamp=(
                f"2026-01-0{5 if index < 2 else 6}T09:30:0{index % 2}"
            ),
            mid_t=10.0,
            bid1=9.9,
            ask1=10.1,
            spread=0.2,
        )
        for index in range(4)
    )
    return PredictionArtifact(
        predictions=np.asarray(predictions).reshape(4, 1),
        targets=targets,
        labels=(60,),
        metadata=samples,
        model_name=model_name,
        model_version=f"baseline-1-fold1-{model_name}",
        dataset_version=metadata.dataset_id,
        fold_index=1,
        split="test",
    )


def _metadata() -> DatasetPackageMetadata:
    identity = {
        "ticker": "TEST",
        "source_hash": "source",
        "processing_config_hash": "processing",
        "fold_plan_hash": "fold-plan",
    }
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(**identity),
        feature_columns=("f0", "f1", "f2", "f3"),
        target_columns=("Target_60s_log",),
        labels=(60,),
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=3,
        history_snapshots=2,
        normalization_mode="causal_rolling",
        normalization_window=2,
        **identity,
    )
