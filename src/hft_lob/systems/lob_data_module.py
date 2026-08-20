"""LOBDataModule（需求文档 §2/§12/§15/§33）：装配 train/val/test/predict DataLoader。

职责单一：
- 不执行 ETL（raw→processed 由 ``prepare_dataset`` 完成）；
- 不做切分决策（文件清单由 walk-forward runner 按 fold 注入）；
- 只负责：按阶段构造 ``LOBWindowDataset``、因果滚动标准化（§12）、按 loader
  配置装配 ``DataLoader``。processed parquet 是唯一缓存层。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.lob_dataset import LOBBatch, LOBWindowDataset, SampleMeta
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
    raise NotImplementedError("resolve_stage_files not implemented")


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """DataLoader worker 确定性种子（§29 全种子；``num_workers > 0`` 时作
    ``worker_init_fn``）。按 ``(base_seed, worker_id)`` 派生独立种子。"""
    raise NotImplementedError("_seed_worker not implemented")


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, SampleMeta]],
) -> LOBBatch:
    """构造强类型 batch，并校验 features/targets 分别为 [B,1,T,F]/[B,1]。"""
    raise NotImplementedError("_collate not implemented")


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
        self.save_hyperparameters()

        self.train_dataset: LOBWindowDataset | None = None
        self.val_dataset: LOBWindowDataset | None = None
        self.test_dataset: LOBWindowDataset | None = None
        self.predict_dataset: LOBWindowDataset | None = None
        self.standardizer = standardizer

    @property
    def dataset_version(self) -> str | None:
        """当前固定数据集版本标识。"""
        raise NotImplementedError("LOBDataModule.dataset_version not implemented")

    def prepare_data(self) -> None:
        """processed parquet 已由 prepare-data 生成，因此固定为 no-op。"""
        raise NotImplementedError("LOBDataModule.prepare_data not implemented")

    def setup(self, stage: str) -> None:
        """按 Lightning 阶段（fit / validate / test / predict）构造或加载数据集。

        Args:
            stage: Lightning 阶段标识。
        """
        raise NotImplementedError("LOBDataModule.setup not implemented")

    def train_dataloader(self) -> DataLoader:
        """训练 DataLoader：shuffle=True + 根 ``config.seed`` 确定性种子。"""
        raise NotImplementedError("LOBDataModule.train_dataloader not implemented")

    def val_dataloader(self) -> DataLoader:
        """验证 DataLoader：按时间序，shuffle=False。"""
        raise NotImplementedError("LOBDataModule.val_dataloader not implemented")

    def test_dataloader(self) -> DataLoader:
        """测试 DataLoader：按时间序，shuffle=False。"""
        raise NotImplementedError("LOBDataModule.test_dataloader not implemented")

    def predict_dataloader(self) -> DataLoader:
        """预测 DataLoader：按时间序，shuffle=False（与 test 共用数据）。"""
        raise NotImplementedError("LOBDataModule.predict_dataloader not implemented")

    def teardown(self, stage: str) -> None:
        """释放阶段资源：清空数据集引用，便于 GC。

        Args:
            stage: Lightning 阶段标识。
        """
        raise NotImplementedError("LOBDataModule.teardown not implemented")

    # -- 内部装配（实装阶段实现）--------------------------------------------

    def _make_dataset(
        self,
        files: Sequence[str],
        *,
        standardizer: FrameStandardizer | None = None,
    ) -> LOBWindowDataset:
        """构造 Dataset；所有 split 使用同一套严格因果滚动标准化语义。"""
        raise NotImplementedError("LOBDataModule._make_dataset not implemented")

    def _make_loader(self, dataset: LOBWindowDataset, *, shuffle: bool) -> DataLoader:
        """从 ``config.loader`` 构造 DataLoader（含确定性种子与工程参数）。"""
        raise NotImplementedError("LOBDataModule._make_loader not implemented")
