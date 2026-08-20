"""LOBDataModule（需求文档 §2/§12/§15/§33）：装配 train/val/test/predict DataLoader。

职责单一：
- 不执行 ETL（raw→processed 由 ``prepare_dataset`` 完成）；
- 不做切分决策（文件清单由 walk-forward runner 按 fold 注入）；
- 只负责：按阶段构造 ``LOBWindowDataset``、因果滚动标准化（§12）、按 loader
  配置装配 ``DataLoader``。processed parquet 是唯一缓存层。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.lob_dataset import LOBBatch, LOBWindowDataset, SampleMeta
from hft_lob.preprocessing.features import FeatureTransformer
from hft_lob.preprocessing.manifest import read_manifest
from hft_lob.preprocessing.normalize import FrameStandardizer
from hft_lob.preprocessing.pipeline import PreparedDataset


@dataclass(frozen=True)
class StageFiles:
    """train/val/test 三段的 processed parquet 文件清单（来自 split manifest）。"""

    training_files: list[str]
    validation_files: list[str]
    test_files: list[str]
    dataset_version: str
    fold_index: int


def resolve_stage_files(dataset: PreparedDataset, *, fold_index: int) -> StageFiles:
    """从 PreparedDataset 固定的版本和 fold plan 解析文件清单。

    Args:
        dataset: prepare-data 的唯一交付对象。

    Returns:
        三段文件清单。

    Raises:
        ValueError: fold_index 不在计划内。
        FileNotFoundError: manifest 中引用的 processed 文件不存在。
    """
    if dataset.walk_forward_plan.dataset_version != dataset.dataset_version:
        raise ValueError("prepared dataset and walk-forward plan versions do not match")
    fold = next(
        (item for item in dataset.walk_forward_plan.folds if item.index == fold_index),
        None,
    )
    if fold is None:
        raise ValueError(f"fold_index {fold_index} is not in the walk-forward plan")

    manifest = read_manifest(dataset.manifest_path)
    versions = manifest.get_column("dataset_version").unique().to_list()
    if versions != [dataset.dataset_version]:
        raise ValueError(
            "manifest dataset_version does not match the prepared dataset: "
            f"expected {dataset.dataset_version!r}, got {versions}"
        )

    def files_for(dates: list[str], *, stage: str) -> list[str]:
        selected = manifest.filter(manifest["trade_date"].is_in(dates)).sort(
            "trade_date", "session_id"
        )
        present_dates = set(selected.get_column("trade_date").to_list())
        missing_dates = sorted(set(dates).difference(present_dates))
        if missing_dates:
            raise ValueError(f"manifest is missing {stage} dates: {missing_dates}")
        paths = selected.get_column("processed_file").to_list()
        missing_files = [path for path in paths if not Path(path).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"manifest references missing {stage} processed files: {missing_files}"
            )
        return paths

    return StageFiles(
        training_files=files_for(fold.train_dates, stage="training"),
        validation_files=files_for(fold.validation_dates, stage="validation"),
        test_files=files_for(fold.test_dates, stage="test"),
        dataset_version=dataset.dataset_version,
        fold_index=fold.index,
    )


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """DataLoader worker 确定性种子（§29 全种子；``num_workers > 0`` 时作
    ``worker_init_fn``）。按 ``(base_seed, worker_id)`` 派生独立种子。"""
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, SampleMeta]],
) -> LOBBatch:
    """构造强类型 batch，并校验 features/targets 分别为 [B,T,F]/[B,1]。"""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    features, targets, metadata = zip(*batch, strict=True)
    feature_batch = torch.stack(features)
    target_batch = torch.stack(targets)
    if feature_batch.ndim != 3:
        raise ValueError(
            f"features must collate to [B, T, F], got {tuple(feature_batch.shape)}"
        )
    if target_batch.ndim != 2 or target_batch.shape[1] != 1:
        raise ValueError(f"targets must collate to [B, 1], got {tuple(target_batch.shape)}")
    if feature_batch.shape[0] != target_batch.shape[0]:
        raise ValueError("features and targets must have the same batch size")
    return LOBBatch(feature_batch, target_batch, tuple(metadata))


class LOBDataModule(pl.LightningDataModule):
    """装配 train/val/test/predict 的 DataLoader（纯装配职责）。

    Lightning 2.x 约定：
    - processed parquet 是唯一数据缓存层，不缓存/序列化整个 Dataset；
    - ``setup(stage)``：每进程执行，赋值 ``self.{train,val,test,predict}_dataset``；
      回归语义下 predict 与 test 共用同一数据集（推理 = 测试窗口数据，§33）；
    - 标准化：所有 Dataset 共享同一个因果滚动标准化配置，在单-session 文件
      加载后、随机采样前应用；
    - 确定性：train shuffle 使用根 ``config.seed`` 播种的 generator，
      ``_seed_worker`` 逐 worker 重播种（§29）。
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        stage_files: StageFiles,
        standardizer: FrameStandardizer,
    ) -> None:
        """初始化数据模块。

        Args:
            config: 实验配置根（data/loader/window/features/target/normalization 段）。
            stage_files: 已绑定 dataset_version/fold 的三段文件清单。
            standardizer: 严格因果的滚动标准化器，不包含全训练集统计量。
        """
        super().__init__()
        self.config = config
        self.stage_files = stage_files
        # DataModule 配置由实验目录备份；checkpoint 仅保留安全基础标识，避免
        # PyTorch weights_only 加载反序列化 ExperimentConfig/StageFiles。
        self.save_hyperparameters(
            {
                "dataset_version": stage_files.dataset_version,
                "fold_index": stage_files.fold_index,
            }
        )

        self.train_dataset: LOBWindowDataset | None = None
        self.val_dataset: LOBWindowDataset | None = None
        self.test_dataset: LOBWindowDataset | None = None
        self.predict_dataset: LOBWindowDataset | None = None
        self.standardizer = standardizer

    @property
    def dataset_version(self) -> str:
        """当前固定数据集版本标识。"""
        return self.stage_files.dataset_version

    def prepare_data(self) -> None:
        """processed parquet 已由 prepare-data 生成，因此固定为 no-op。"""
        return None

    def setup(self, stage: str | None = None) -> None:
        """按 Lightning 阶段（fit / validate / test / predict）构造或加载数据集。

        Args:
            stage: Lightning 阶段标识。
        """
        supported = {None, "fit", "validate", "test", "predict"}
        if stage not in supported:
            raise ValueError(f"unsupported Lightning stage: {stage!r}")
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(self.stage_files.training_files)
            self.val_dataset = self._make_dataset(self.stage_files.validation_files)
        elif stage == "validate":
            self.val_dataset = self._make_dataset(self.stage_files.validation_files)
        if stage in (None, "test"):
            self.test_dataset = self._make_dataset(self.stage_files.test_files)
        if stage in (None, "predict"):
            self.predict_dataset = self._make_dataset(self.stage_files.test_files)

    def train_dataloader(self) -> DataLoader:
        """训练 DataLoader：shuffle=True + 根 ``config.seed`` 确定性种子。"""
        return self._make_loader(self._require_dataset("train_dataset"), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """验证 DataLoader：按时间序，shuffle=False。"""
        return self._make_loader(self._require_dataset("val_dataset"), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """测试 DataLoader：按时间序，shuffle=False。"""
        return self._make_loader(self._require_dataset("test_dataset"), shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """预测 DataLoader：按时间序，shuffle=False（与 test 共用数据）。"""
        return self._make_loader(self._require_dataset("predict_dataset"), shuffle=False)

    def teardown(self, stage: str | None = None) -> None:
        """释放阶段资源：清空数据集引用，便于 GC。

        Args:
            stage: Lightning 阶段标识。
        """
        supported = {None, "fit", "validate", "test", "predict"}
        if stage not in supported:
            raise ValueError(f"unsupported Lightning stage: {stage!r}")
        if stage in (None, "fit"):
            self.train_dataset = None
            self.val_dataset = None
        elif stage == "validate":
            self.val_dataset = None
        if stage in (None, "test"):
            self.test_dataset = None
        if stage in (None, "predict"):
            self.predict_dataset = None

    # -- 内部装配（实装阶段实现）--------------------------------------------

    def _make_dataset(
        self,
        files: Sequence[str],
        *,
        standardizer: FrameStandardizer | None = None,
    ) -> LOBWindowDataset:
        """构造 Dataset；所有 split 使用同一套严格因果滚动标准化语义。"""
        return LOBWindowDataset(
            files,
            ticker=self.config.ticker,
            window_size=self.config.window.history_snapshots,
            feature_cols=self.config_feature_columns,
            target_col=self.config.target_column,
            cache_size=self.config.loader.cache_size,
            standardizer=self.standardizer if standardizer is None else standardizer,
        )

    def _make_loader(self, dataset: LOBWindowDataset, *, shuffle: bool) -> DataLoader:
        """从 ``config.loader`` 构造 DataLoader（含确定性种子与工程参数）。"""
        loader = self.config.loader
        if loader.batch_size <= 0:
            raise ValueError("loader.batch_size must be > 0")
        if loader.num_workers < 0:
            raise ValueError("loader.num_workers must be >= 0")
        if loader.persistent_workers and loader.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if loader.prefetch_factor is not None and loader.num_workers == 0:
            raise ValueError("prefetch_factor requires num_workers > 0")

        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=shuffle,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers,
            prefetch_factor=loader.prefetch_factor,
            collate_fn=_collate,
            worker_init_fn=partial(_seed_worker, base_seed=self.config.seed),
            generator=generator,
        )

    @property
    def config_feature_columns(self) -> tuple[str, ...]:
        """从统一 FeatureConfig 推导 Dataset 输入列。"""
        return tuple(FeatureTransformer(self.config.features).feature_columns())

    def _require_dataset(self, name: str) -> LOBWindowDataset:
        dataset = cast(LOBWindowDataset | None, getattr(self, name))
        if dataset is None:
            raise RuntimeError(f"{name} is not initialized; call setup() first")
        return dataset
