from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_lob.configs.experiment import (
    RAW_FEATURE_COLUMNS,
    CleaningConfig,
    DataBuildConfig,
    DataConfig,
    FeatureConfig,
    LoaderConfig,
    NormalizationConfig,
    SessionConfig,
    SplitConfig,
    TargetConfig,
    TaskConfig,
    WalkForwardConfig,
    WindowConfig,
)
from hft_lob.data_pipeline.builder import build_dataset_package
from hft_lob.data_pipeline.dataset_validator import open_dataset_package, validate_dataset_package
from hft_lob.datasets.datamodule import LOBDataModule
from hft_lob.datasets.lob_dataset import PrebuiltLOBDataset


def _config(tmp_path: Path) -> DataBuildConfig:
    return DataBuildConfig(
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
        ),
        cleaning=CleaningConfig(),
        target=TargetConfig(label=[12, 6], tolerance_seconds=0),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=2),
        features=FeatureConfig(),
        normalization=NormalizationConfig(normalize_window=2),
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


def test_build_dataset_package_is_complete_and_idempotent(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    _write_raw_data(tmp_path)
    config = _config(tmp_path)

    first = build_dataset_package(config, tmp_path / "prebuilt")
    second = build_dataset_package(config, tmp_path / "prebuilt")
    metadata = validate_dataset_package(first)

    assert second == first
    assert metadata.ticker == "TEST"
    stored = np.load(first / "features.npy", mmap_mode="r")
    assert stored.shape[1] == len(RAW_FEATURE_COLUMNS)
    assert len(list((first / "folds").glob("fold_*"))) == 2
    index = pl.read_parquet(first / "folds" / "fold_001" / "train.parquet")
    assert index.height > 0
    dataset = PrebuiltLOBDataset(first, metadata, fold_index=1, split="train")
    features, target, targets_by_horizon, sample = dataset[0]
    assert features.shape == (config.window.history_snapshots, len(RAW_FEATURE_COLUMNS))
    assert target.shape == (1,)
    assert set(targets_by_horizon) == {12, 6}
    assert target.item() == targets_by_horizon[12].item()
    assert sample.ticker == "TEST"
    module = LOBDataModule(
        open_dataset_package(first), fold_index=1, loader=LoaderConfig(), seed=42
    )
    module.setup("fit")
    assert next(iter(module.train_dataloader())).features.ndim == 3
    assert not any(path.name.startswith(".") for path in (tmp_path / "prebuilt").iterdir())
    assert not (first / "anchors.parquet").exists()
    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "manifests").exists()
    assert "dataset_build.start" in caplog.text
    assert "dataset_build.progress" in caplog.text
    assert "dataset_build.validate_complete" in caplog.text
    assert "dataset_build.complete" in caplog.text
