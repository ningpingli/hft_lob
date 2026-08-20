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
from hft_lob.preprocessing.manifest import read_manifest
from hft_lob.preprocessing.pipeline import prepare_dataset


def _raw_row(timestamp: datetime, price_offset: float) -> dict[str, object]:
    row: dict[str, object] = {"timestamp": timestamp}
    for level in range(1, 6):
        row[f"ASKp{level}"] = 10.01 + price_offset + 0.01 * (level - 1)
        row[f"ASKs{level}"] = 100.0
        row[f"BIDp{level}"] = 9.99 + price_offset - 0.01 * (level - 1)
        row[f"BIDs{level}"] = 100.0
    row.update(last=10.0 + price_offset, volume=1_000.0, amount=10_000.0)
    return row


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="pipeline-test",
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            manifest_dir=str(tmp_path / "datasets"),
        ),
        cleaning=CleaningConfig(max_ffill_gap_seconds=6),
        target=TargetConfig(horizon_seconds=60, tolerance_seconds=3),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=2),
        features=FeatureConfig(use_derived=False),
        normalization=NormalizationConfig(normalize_window=2),
        loader=LoaderConfig(),
        model=ModelConfig(),
        baselines=BaselineConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
        split=SplitConfig(),
        walk_forward=WalkForwardConfig(
            train_window_days=2,
            validation_window_days=1,
            test_window_days=1,
            step_days=1,
        ),
    )


def test_prepare_dataset_builds_content_addressed_session_artifacts(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "TEST"
    raw_root.mkdir(parents=True)
    for day in range(1, 6):
        start = datetime(2026, 1, day, 9, 30)
        frame = pl.DataFrame(
            [
                _raw_row(start, price_offset=day * 0.01),
                _raw_row(start + timedelta(seconds=63), price_offset=day * 0.02),
            ]
        ).select("timestamp", *RAW_FEATURE_COLUMNS)
        frame.write_parquet(raw_root / f"202601{day:02d}.parquet")

    prepared = prepare_dataset(_config(tmp_path))
    manifest = read_manifest(prepared.manifest_path)
    quality = pl.read_parquet(prepared.quality_report_path)

    assert manifest.height == 5
    assert manifest.get_column("session_id").unique().to_list() == ["AM"]
    assert manifest.get_column("dataset_version").unique().to_list() == [
        prepared.dataset_version
    ]
    assert manifest.get_column("valid_row_count").min() > 0
    assert quality.height == 5
    assert len(prepared.walk_forward_plan.folds) == 2
    assert prepared.feature_columns == RAW_FEATURE_COLUMNS

    processed_path = Path(manifest.get_column("processed_file").item(0))
    processed = pl.read_parquet(processed_path)
    assert processed_path.is_file()
    assert processed.get_column("session_id").unique().to_list() == ["AM"]
    assert "Target_60s_log" in processed.columns
    assert not any(name.startswith("normalized__") for name in processed.columns)

    repeated = prepare_dataset(_config(tmp_path))
    assert repeated.dataset_version == prepared.dataset_version
