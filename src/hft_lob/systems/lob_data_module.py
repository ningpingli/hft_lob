"""LOBDataModule（需求文档 §2/§12/§15/§33）：装配 train/val/test/predict DataLoader。

职责单一：
- 不执行 ETL（raw→processed 由 ``preprocessing.run_pipeline`` 在独立阶段完成）；
- 不做切分决策（文件清单来自 split manifest，经 ``StageFiles`` 注入或
  ``resolve_stage_files`` 解析）；
- 只负责：按阶段构造 ``LOBWindowDataset``、train-only 归一化（§12）、按 loader
  配置装配 ``DataLoader``、可选 .pt 缓存。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from hft_lob.configs.experiment import ExperimentConfig
from hft_lob.datasets.lob_dataset import LOBWindowDataset


@dataclass(frozen=True)
class StageFiles:
    """train/val/test 三段的 processed parquet 文件清单（来自 split manifest）。"""

    training_files: list[str]
    validation_files: list[str]
    test_files: list[str]


def resolve_stage_files(config: ExperimentConfig) -> StageFiles:
    """从 ``config.data.manifest_dir`` 下最新数据集的 split manifest 解析文件清单。

    Args:
        config: 实验配置根。

    Returns:
        三段文件清单。

    Raises:
        FileNotFoundError: 尚未运行 data_processing（无数据集版本目录）。
    """
    raise NotImplementedError("resolve_stage_files not implemented")


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """DataLoader worker 确定性种子（§29 全种子；``num_workers > 0`` 时作
    ``worker_init_fn``）。按 ``(base_seed, worker_id)`` 派生独立种子。"""
    raise NotImplementedError("_seed_worker not implemented")


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, object]]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[object]]]:
    """自定义 collate：拆分 (X, y, meta) 三元组；meta 转 dict-of-lists
    （字符串字段不会被 default_collate 处理）。"""
    raise NotImplementedError("_collate not implemented")


class LOBDataModule(pl.LightningDataModule):
    """装配 train/val/test/predict 的 DataLoader（纯装配职责）。

    Lightning 2.x 约定：
    - ``prepare_data()``：单进程一次性执行（仅磁盘副作用）；有 ``cache_dir`` 时
      构建三段数据集并 ``torch.save``（幂等），无则 no-op；
    - ``setup(stage)``：每进程执行，赋值 ``self.{train,val,test,predict}_dataset``；
      回归语义下 predict 与 test 共用同一数据集（推理 = 测试窗口数据，§33）；
    - 归一化：``train_only`` 模式在 ``setup`` 内 fit(train) → transform(all)
      （§12，参数只来自训练段）；``causal`` 模式暂不在此做行级归一化
      （``CausalRollingNormalizer`` 已在 preprocessing 层提供，待接入）；
    - 确定性：train shuffle 使用 ``loader.seed`` 播种的 generator，
      ``_seed_worker`` 逐 worker 重播种（§29）。
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        stage_files: StageFiles | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """初始化数据模块。

        Args:
            config: 实验配置根（data/loader/window/features/target/normalization 段）。
            stage_files: 三段文件清单；None 时 ``setup`` 经 ``resolve_stage_files``
                从最新数据集版本解析。
            cache_dir: 可选 .pt 数据集缓存目录（None 表示不缓存）。
        """
        super().__init__()
        self.config = config
        self.stage_files = stage_files
        self.cache_dir = cache_dir
        self.save_hyperparameters()

        self.train_dataset: LOBWindowDataset | None = None
        self.val_dataset: LOBWindowDataset | None = None
        self.test_dataset: LOBWindowDataset | None = None
        self.predict_dataset: LOBWindowDataset | None = None

    @property
    def dataset_version(self) -> str | None:
        """当前数据集版本标识（§31；来自 manifest 目录名；注入模式下为 None）。"""
        raise NotImplementedError("LOBDataModule.dataset_version not implemented")

    def prepare_data(self) -> None:
        """单进程一次性准备：有 ``cache_dir`` 时构建三段数据集并 ``torch.save``
        （幂等：目标 .pt 已存在则跳过）。无 ``cache_dir`` 时为 no-op。"""
        raise NotImplementedError("LOBDataModule.prepare_data not implemented")

    def setup(self, stage: str) -> None:
        """按 Lightning 阶段（fit / validate / test / predict）构造或加载数据集。

        Args:
            stage: Lightning 阶段标识。
        """
        raise NotImplementedError("LOBDataModule.setup not implemented")

    def train_dataloader(self) -> DataLoader:
        """训练 DataLoader：shuffle=True + ``loader.seed`` 确定性种子。"""
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
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> LOBWindowDataset:
        """从 ``config`` 构造单阶段 ``LOBWindowDataset``（含 train-only 归一化参数）。"""
        raise NotImplementedError("LOBDataModule._make_dataset not implemented")

    def _make_loader(self, dataset: LOBWindowDataset, *, shuffle: bool) -> DataLoader:
        """从 ``config.loader`` 构造 DataLoader（含确定性种子与工程参数）。"""
        raise NotImplementedError("LOBDataModule._make_loader not implemented")
