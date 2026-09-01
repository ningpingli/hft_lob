from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hft_lob.datasets.dataset_validator import (
    DatasetPackage,
    DatasetPackageMetadata,
    compute_dataset_id,
)
from hft_lob.systems.artifact import PredictionArtifact, save_prediction_artifact
from hft_lob.systems.baseline_manifest import (
    BaselineArtifactReference,
    BaselineManifest,
    baseline_space,
    build_baseline_comparison,
    default_manifest_path,
    load_default_manifest,
    save_default_manifest,
    validate_default_manifest,
)
from hft_lob.systems.contracts import SampleMeta


def test_default_manifest_round_trips_and_validates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hft_lob.systems.baseline_manifest._RESULTS_ROOT", tmp_path / "results")
    metadata = _metadata()
    package = DatasetPackage(root=tmp_path / metadata.dataset_id, metadata=metadata)
    root = baseline_space(metadata.dataset_id)
    prediction_path = root / "runs" / "baseline-1" / "fold_001" / "zero" / "predictions.parquet"
    evaluation_path = prediction_path.with_name("evaluation.yaml")
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        "overall:\n  ts_ic: 0.2\nmean_daily_ic: 0.1\ndaily: []\nhorizon_decay: []\n",
        encoding="utf-8",
    )
    save_prediction_artifact(artifact=_artifact(metadata), path=str(prediction_path))
    manifest = BaselineManifest(
        dataset_id=metadata.dataset_id,
        experiment_id="baseline-1",
        config_hash="config-hash",
        fold_indices=(1,),
        baseline_names=("zero",),
        artifacts=(
            BaselineArtifactReference(
                fold_index=1,
                baseline_name="zero",
                predictions_path=prediction_path.relative_to(root).as_posix(),
                evaluation_path=evaluation_path.relative_to(root).as_posix(),
                overall={"ts_ic": 0.2},
                mean_daily_ic=0.1,
            ),
        ),
    )

    save_default_manifest(manifest)
    loaded = load_default_manifest(metadata.dataset_id)
    validated = validate_default_manifest(package, fold_indices=(1,))

    assert loaded == manifest
    assert validated == manifest
    assert default_manifest_path(metadata.dataset_id).is_file()


def test_baseline_comparison_uses_metric_direction_and_mean_daily_ic() -> None:
    manifest = BaselineManifest(
        dataset_id="dataset",
        experiment_id="baseline",
        config_hash="config",
        fold_indices=(1,),
        baseline_names=("zero",),
        artifacts=(
            BaselineArtifactReference(
                fold_index=1,
                baseline_name="zero",
                predictions_path="predictions.parquet",
                evaluation_path="evaluation.yaml",
                overall={"mae": 1.0, "ts_ic": 0.2},
                mean_daily_ic=0.1,
            ),
        ),
    )
    evaluation = SimpleNamespace(
        overall={"mae": 0.5, "ts_ic": 0.3},
        mean_daily_ic=0.2,
        daily=(),
        horizon_decay=(),
    )
    result = build_baseline_comparison(
        (SimpleNamespace(fold_index=1, evaluation=evaluation),),
        manifest,
    )["zero"]

    assert result["fold_delta"]["mae"] == pytest.approx(-0.5)
    assert result["fold_delta"]["ts_ic"] == pytest.approx(0.1)
    assert result["fold_delta"]["mean_daily_ic"] == pytest.approx(0.1)
    assert result["fold_win_ratio"] == {"mae": 1.0, "ts_ic": 1.0, "mean_daily_ic": 1.0}


def test_manifest_rejects_missing_requested_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hft_lob.systems.baseline_manifest._RESULTS_ROOT", tmp_path / "results")
    metadata = _metadata()
    package = DatasetPackage(root=tmp_path / metadata.dataset_id, metadata=metadata)
    save_default_manifest(
        BaselineManifest(
            dataset_id=metadata.dataset_id,
            experiment_id="baseline-1",
            config_hash="config-hash",
            fold_indices=(1,),
            baseline_names=("zero",),
            artifacts=(),
        )
    )

    with pytest.raises(ValueError, match="missing requested folds"):
        validate_default_manifest(package, fold_indices=(1, 2))


def _artifact(metadata: DatasetPackageMetadata) -> PredictionArtifact:
    return PredictionArtifact(
        predictions=np.array([0.1]),
        targets=np.array([0.2]),
        metadata=(
            SampleMeta(
                ticker=metadata.ticker,
                trade_date="2026-01-05",
                session_id="AM",
                anchor_timestamp="2026-01-05T09:30:00",
                mid_t=10.0,
                future_mid=10.1,
                bid1=9.9,
                ask1=10.1,
                spread=0.2,
            ),
        ),
        model_name="zero",
        model_version="baseline-1-fold1-zero",
        dataset_version=metadata.dataset_id,
        fold_index=1,
        split="test",
    )


def _metadata() -> DatasetPackageMetadata:
    ticker = "TEST"
    source_hash = "source"
    processing_config_hash = "processing"
    fold_plan_hash = "fold-plan"
    return DatasetPackageMetadata(
        dataset_id=compute_dataset_id(
            ticker=ticker,
            source_hash=source_hash,
            processing_config_hash=processing_config_hash,
            fold_plan_hash=fold_plan_hash,
        ),
        ticker=ticker,
        feature_columns=("f0", "f1", "f2", "f3"),
        target_column="target",
        feature_dtype="float32",
        target_dtype="float32",
        snapshot_interval_seconds=3,
        history_snapshots=2,
        normalization_mode="causal_rolling",
        normalization_window=2,
        source_hash=source_hash,
        processing_config_hash=processing_config_hash,
        fold_plan_hash=fold_plan_hash,
    )
