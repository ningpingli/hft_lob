from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from hft_lob.application.baseline import BaselineRunRequest, run_baseline_application
from hft_lob.configs.experiment import (
    RAW_FEATURE_COLUMNS,
    CleaningConfig,
    DataBuildConfig,
    DataConfig,
    EvaluationConfig,
    FeatureConfig,
    FoldSelectionConfig,
    LoaderConfig,
    ModelConfig,
    ModelRunConfig,
    NormalizationConfig,
    SessionConfig,
    SplitConfig,
    TargetConfig,
    TaskConfig,
    TrainingConfig,
    WalkForwardConfig,
    WindowConfig,
)
from hft_lob.datasets.builder import build_dataset_package
from hft_lob.datasets.dataset_validator import open_dataset_package
from hft_lob.systems.baseline_manifest import load_default_manifest
from hft_lob.systems.executor import DefaultWalkForwardExecutor
from hft_lob.systems.walk_forward import run_walk_forward


def test_default_executor_trains_cnn_and_writes_prediction_artifact(tmp_path: Path) -> None:
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    data_config, model_config = _configs(tmp_path)
    for day, date in enumerate(dates):
        _write_raw(tmp_path, date, day)
    dataset_dir = build_dataset_package(data_config, tmp_path / "prebuilt")

    report = run_walk_forward(
        open_dataset_package(dataset_dir),
        model_config,
        executor=DefaultWalkForwardExecutor(
            str(tmp_path / "results"), accelerator="cpu"
        ),
    )

    assert len(report.fold_results) == 1
    result = report.fold_results[0]
    assert Path(result.checkpoint_path or "").is_file()
    assert Path(result.predictions_path).is_file()
    assert result.evaluation.sample_count > 0
    output_dir = Path(result.predictions_path).parent
    assert (output_dir / "evaluation.yaml").is_file()
    assert (output_dir / "daily_ic_curve.png").is_file()
    assert (output_dir / "time_series_grouped_return_curve.png").is_file()
    assert "mean_daily_ic_mean" in report.summary["cnn1"]


def test_baseline_run_publishes_dataset_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    data_config, _ = _configs(tmp_path)
    for day, date in enumerate(dates):
        _write_raw(tmp_path, date, day)
    dataset_dir = build_dataset_package(data_config, tmp_path / "prebuilt")
    monkeypatch.setattr(
        "hft_lob.systems.baseline_manifest._RESULTS_ROOT",
        tmp_path / "results",
    )

    result = run_baseline_application(
        BaselineRunRequest(
            config_path="configs/baselines.yaml",
            dataset_dir=str(dataset_dir),
            experiment_id="baseline-smoke",
        )
    )
    manifest = load_default_manifest(result.dataset_version)

    assert result.artifact_count == 3
    assert result.experiment_id == "baseline-smoke"
    assert manifest.experiment_id == "baseline-smoke"
    assert manifest.baseline_names == ("zero", "imbalance", "ridge")
    assert len(manifest.artifacts) == 3


def _configs(tmp_path: Path) -> tuple[DataBuildConfig, ModelRunConfig]:
    data_config = DataBuildConfig(
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
        ),
        cleaning=CleaningConfig(),
        target=TargetConfig(label=(6, 12), tolerance_seconds=0),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=8),
        features=FeatureConfig(),
        normalization=NormalizationConfig(normalize_window=2),
        split=SplitConfig(),
        walk_forward=WalkForwardConfig(
            train_window_days=1,
            validation_window_days=1,
            test_window_days=1,
            step_days=1,
        ),
    )
    model_config = ModelRunConfig(
        experiment_id="executor-smoke",
        loader=LoaderConfig(batch_size=8),
        model=ModelConfig(name="cnn1"),
        training=TrainingConfig(epochs=1, patience=0),
        evaluation=EvaluationConfig(
            metrics=("mae", "ts_ic"),
            prediction_bins=2,
            bootstrap_samples=2,
            bootstrap_block_size=2,
        ),
        folds=FoldSelectionConfig(),
        seed=7,
    )
    return data_config, model_config


def _write_raw(tmp_path: Path, trade_date: str, day_offset: int) -> None:
    start = datetime.fromisoformat(f"{trade_date}T09:30:00")
    rows: list[dict[str, object]] = []
    for index in range(30):
        timestamp = start + timedelta(seconds=3 * index)
        row: dict[str, object] = {"timestamp": timestamp}
        mid = 10.0 + day_offset * 0.01 + index * 0.001
        for level in range(1, 6):
            row[f"ASKp{level}"] = mid + level * 0.01
            row[f"ASKs{level}"] = 100.0 + index + level
            row[f"BIDp{level}"] = mid - level * 0.01
            row[f"BIDs{level}"] = 100.0 + index + level
        row.update(last=mid, volume=1000.0 + index, amount=10000.0 + index)
        rows.append(row)
    path = tmp_path / "raw" / "TEST" / f"{trade_date.replace('-', '')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).select("timestamp", *RAW_FEATURE_COLUMNS).write_parquet(path)
