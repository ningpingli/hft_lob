from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from hft_lob.configs.experiment import (
    BaselineConfig,
    EvaluationConfig,
    FoldSelectionConfig,
    LoaderConfig,
    ModelConfig,
    ModelRunConfig,
    TrainingConfig,
)
from hft_lob.datasets.dataset_validator import (
    DatasetPackage,
    DatasetPackageMetadata,
    compute_dataset_id,
)
from hft_lob.systems.artifact import PredictionArtifact
from hft_lob.systems.contracts import SampleMeta
from hft_lob.systems.executor import DefaultWalkForwardExecutor


def test_baseline_artifact_is_reused_across_model_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata = _metadata()
    package = DatasetPackage(root=tmp_path / metadata.dataset_id, metadata=metadata)
    config = _config("model-a")
    source_artifact = PredictionArtifact(
        predictions=np.array([0.1]),
        targets=np.array([0.2]),
        metadata=(
            SampleMeta(
                ticker="TEST",
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
        model_version="model-a-fold1-zero",
        dataset_version=metadata.dataset_id,
        fold_index=1,
        split="test",
    )
    calls = 0

    def fake_run_baseline(**_: object) -> PredictionArtifact:
        nonlocal calls
        calls += 1
        return source_artifact

    monkeypatch.setattr(DefaultWalkForwardExecutor, "_run_baseline", staticmethod(fake_run_baseline))
    executor = DefaultWalkForwardExecutor(
        str(tmp_path / "run-a"),
        baseline_cache_root=str(tmp_path / "baseline-cache"),
    )
    first = executor.run_candidate(
        package=package,
        config=config,
        fold_index=1,
        candidate_name="zero",
    )
    second = DefaultWalkForwardExecutor(
        str(tmp_path / "run-b"),
        baseline_cache_root=str(tmp_path / "baseline-cache"),
    ).run_candidate(
        package=package,
        config=replace(config, experiment_id="model-b"),
        fold_index=1,
        candidate_name="zero",
    )

    cache_path, _ = executor._baseline_cache_path(
        metadata=metadata,
        config=config,
        fold_index=1,
        candidate_name="zero",
    )
    assert calls == 1
    assert cache_path.is_file()
    assert first.artifact.predictions.tolist() == [0.1]
    assert second.artifact.predictions.tolist() == [0.1]
    assert second.artifact.model_version == "model-b-fold1-zero"


def _config(experiment_id: str) -> ModelRunConfig:
    return ModelRunConfig(
        experiment_id=experiment_id,
        loader=LoaderConfig(),
        model=ModelConfig(name="cnn1"),
        baselines=BaselineConfig(names=("zero",)),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
        folds=FoldSelectionConfig(),
        seed=42,
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
