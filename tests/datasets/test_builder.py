from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch

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
from hft_lob.datasets.builder import build_dataset_package
from hft_lob.datasets.validation import validate_dataset_package


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="builder-test",
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            manifest_dir=str(tmp_path / "manifests"),
        ),
        cleaning=CleaningConfig(),
        target=TargetConfig(horizon_seconds=6, tolerance_seconds=0),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=2),
        features=FeatureConfig(),
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


def _write_raw_data(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "TEST"
    raw.mkdir(parents=True)
    for day in range(1, 6):
        start = datetime(2026, 1, day, 9, 30)
        rows: list[dict[str, object]] = []
        for step in range(12):
            mid = 10 + day * 0.01 + step * 0.001
            row: dict[str, object] = {"timestamp": start + timedelta(seconds=step * 3)}
            for level in range(1, 6):
                row[f"ASKp{level}"] = mid + level * 0.01
                row[f"ASKs{level}"] = 100.0 + step
                row[f"BIDp{level}"] = mid - level * 0.01
                row[f"BIDs{level}"] = 100.0 + step
            row.update(last=mid, volume=1000.0 + step, amount=10000.0 + step)
            rows.append(row)
        pl.DataFrame(rows).select("timestamp", *RAW_FEATURE_COLUMNS).write_parquet(
            raw / f"202601{day:02d}.parquet"
        )


def test_build_dataset_package_is_complete_and_idempotent(tmp_path: Path) -> None:
    _write_raw_data(tmp_path)
    config = _config(tmp_path)

    first = build_dataset_package(config, tmp_path / "prebuilt")
    second = build_dataset_package(config, tmp_path / "prebuilt")
    metadata = validate_dataset_package(first)

    assert second == first
    assert metadata.ticker == "TEST"
    assert len(list((first / "sessions").glob("*.pt"))) == 5
    assert len(list((first / "folds").glob("fold_*"))) == 2
    index = pl.read_parquet(first / "folds" / "fold_001" / "train.parquet")
    assert index.height > 0
    payload = torch.load(first / index.get_column("session_file").item(0), weights_only=True)
    assert payload["features"].shape[1] == len(RAW_FEATURE_COLUMNS)
    assert not any(path.name.startswith(".") for path in (tmp_path / "prebuilt").iterdir())
