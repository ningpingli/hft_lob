from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from hft_lob.configs.experiment import (
    RAW_FEATURE_COLUMNS,
    BaselineConfig,
    CleaningConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureConfig,
    LoaderConfig,
    ModelConfig,
    NormalizationConfig,
    SessionConfig,
    SplitConfig,
    TargetConfig,
    TaskConfig,
    TrainingConfig,
    WalkForwardConfig,
    WindowConfig,
)
from hft_lob.preprocessing.manifest import build_manifest, write_manifest
from hft_lob.preprocessing.pipeline import PreparedDataset
from hft_lob.preprocessing.split import Fold, WalkForwardPlan
from hft_lob.systems.executor import DefaultWalkForwardExecutor
from hft_lob.systems.walk_forward import run_walk_forward


def test_default_executor_trains_cnn_and_writes_prediction_artifact(tmp_path: Path) -> None:
    version = "smoke-dataset-v1"
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    processed_files = [_write_processed(tmp_path, date, day) for day, date in enumerate(dates)]
    records = [
        _manifest_record(path, date, version)
        for path, date in zip(processed_files, dates, strict=True)
    ]
    manifest_path = tmp_path / "manifest.parquet"
    write_manifest(build_manifest(ticker="TEST", records=records), str(manifest_path))
    quality_path = tmp_path / "quality.parquet"
    pl.DataFrame({"trade_date": dates}).write_parquet(quality_path)
    plan = WalkForwardPlan(
        version,
        (Fold(1, [dates[0]], [dates[1]], [dates[2]]),),
    )
    dataset = PreparedDataset(
        dataset_version=version,
        feature_columns=RAW_FEATURE_COLUMNS,
        feature_version="features-v1",
        label_version="label-v1",
        manifest_path=str(manifest_path),
        quality_report_path=str(quality_path),
        walk_forward_plan=plan,
    )
    config = _config(tmp_path)

    report = run_walk_forward(
        dataset,
        config,
        executor=DefaultWalkForwardExecutor(
            str(tmp_path / "results"), accelerator="cpu"
        ),
    )

    assert len(report.fold_results) == 1
    result = report.fold_results[0]
    assert Path(result.checkpoint_path or "").is_file()
    assert Path(result.predictions_path).is_file()
    assert result.evaluation.sample_count > 0


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="executor-smoke",
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(manifest_dir=str(tmp_path)),
        cleaning=CleaningConfig(),
        target=TargetConfig(),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=8),
        features=FeatureConfig(),
        normalization=NormalizationConfig(normalize_window=2),
        loader=LoaderConfig(batch_size=8),
        model=ModelConfig(name="cnn1"),
        baselines=BaselineConfig(names=()),
        training=TrainingConfig(epochs=1, patience=0),
        evaluation=EvaluationConfig(
            metrics=("mae", "ts_ic"),
            prediction_bins=2,
            bootstrap_samples=2,
            bootstrap_block_size=2,
        ),
        split=SplitConfig(),
        walk_forward=WalkForwardConfig(
            train_window_days=1,
            validation_window_days=1,
            test_window_days=1,
            step_days=1,
        ),
        seed=7,
    )


def _write_processed(tmp_path: Path, trade_date: str, day_offset: int) -> str:
    start = datetime.fromisoformat(f"{trade_date}T09:30:00")
    rows: list[dict[str, object]] = []
    for index in range(30):
        timestamp = start + timedelta(seconds=3 * index)
        row: dict[str, object] = {
            "ticker": "TEST",
            "trade_date": trade_date,
            "session_id": "AM",
            "timestamp": timestamp,
            "seconds": timestamp.time(),
            "book_valid": True,
            "feature_valid": True,
            "target_valid": True,
            "mid_price": 10.0 + 0.001 * index,
            "future_mid": 10.1 + 0.001 * index,
            "Target_60s_log": (day_offset + 1) * 0.001 + index * 0.0001,
        }
        for feature_index, name in enumerate(RAW_FEATURE_COLUMNS):
            row[name] = 1.0 + feature_index * 0.1 + index * 0.01
        row["ASKp1"] = 10.1 + index * 0.001
        row["BIDp1"] = 9.9 + index * 0.001
        rows.append(row)
    path = tmp_path / f"{trade_date}_AM.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return str(path.resolve())


def _manifest_record(path: str, trade_date: str, version: str) -> dict[str, object]:
    start = datetime.fromisoformat(f"{trade_date}T09:30:00")
    return {
        "trade_date": trade_date,
        "session_id": "AM",
        "source_file": path,
        "processed_file": path,
        "raw_hash": "raw-hash",
        "processing_config_hash": "config-hash",
        "dataset_version": version,
        "row_count": 30,
        "valid_row_count": 30,
        "data_start": start,
        "data_end": start + timedelta(seconds=87),
        "feature_version": "features-v1",
        "label_version": "label-v1",
        "quality_status": "passed",
    }
