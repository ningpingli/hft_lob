from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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
from hft_lob.datasets.prebuilt_dataset import PrebuiltLOBDataset
from hft_lob.datasets.validation import validate_dataset_package
from hft_lob.systems.lob_data_module import LOBDataModule, _seed_worker
from hft_lob.utils.seed import set_seed


class RandomizedLOBWindowDataset(PrebuiltLOBDataset):
    """测试专用包装：让每个 worker 同时消费三套随机流。"""

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        features, target, metadata = super().__getitem__(index)
        noise = random.random() + float(np.random.random()) + float(torch.rand(()))
        return features + noise, target, metadata


def _raw_row(timestamp: datetime, *, step: int, day: int) -> dict[str, object]:
    mid = 10.0 + day * 0.01 + step * 0.0001
    row: dict[str, object] = {"timestamp": timestamp}
    for level in range(1, 6):
        row[f"ASKp{level}"] = mid + 0.01 * level
        row[f"ASKs{level}"] = 100.0 + step + level
        row[f"BIDp{level}"] = mid - 0.01 * level
        row[f"BIDs{level}"] = 100.0 + step + level
    row.update(last=mid, volume=1_000.0 + step, amount=10_000.0 + step)
    return row


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="seed-integration",
        task=TaskConfig(ticker="TEST"),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            manifest_dir=str(tmp_path / "datasets"),
        ),
        cleaning=CleaningConfig(max_ffill_gap_seconds=6),
        target=TargetConfig(horizon_seconds=6, tolerance_seconds=0),
        sessions=SessionConfig(),
        window=WindowConfig(history_snapshots=2),
        features=FeatureConfig(use_derived=False),
        normalization=NormalizationConfig(normalize_window=2),
        loader=LoaderConfig(batch_size=4, num_workers=0),
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
        seed=31415,
    )


def _load_once(config: ExperimentConfig, dataset_dir: Path) -> tuple[torch.Tensor, tuple[str, ...]]:
    metadata = validate_dataset_package(dataset_dir)
    module = LOBDataModule(
        dataset_dir, fold_index=1, loader=config.loader, seed=config.seed
    )
    dataset = RandomizedLOBWindowDataset(
        dataset_dir,
        metadata,
        fold_index=1,
        split="train",
    )
    batches = list(module._make_loader(dataset, shuffle=True))
    features = torch.cat([batch.features for batch in batches])
    anchors = tuple(meta.anchor_timestamp for batch in batches for meta in batch.metadata)
    return features, anchors


def test_virtual_raw_lob_pipeline_and_loader_are_reproducible(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw" / "TEST"
    raw_root.mkdir(parents=True)
    for day in range(1, 6):
        start = datetime(2026, 1, day, 9, 30)
        frame = pl.DataFrame(
            [
                _raw_row(start + timedelta(seconds=step * 3), step=step, day=day)
                for step in range(12)
            ]
        ).select("timestamp", *RAW_FEATURE_COLUMNS)
        frame.write_parquet(raw_root / f"202601{day:02d}.parquet")

    config = _config(tmp_path)
    dataset_dir = build_dataset_package(config, tmp_path / "prebuilt")

    set_seed(config.seed)
    first_features, first_anchors = _load_once(config, dataset_dir)
    set_seed(config.seed)
    second_features, second_anchors = _load_once(config, dataset_dir)

    assert first_anchors == second_anchors
    assert torch.equal(first_features, second_features)
    assert first_features.shape[0] > 0


def test_worker_seed_replays_python_numpy_and_torch_streams() -> None:
    _seed_worker(worker_id=3, base_seed=31415)
    first = (random.random(), float(np.random.random()), torch.rand(3))

    _seed_worker(worker_id=3, base_seed=31415)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
