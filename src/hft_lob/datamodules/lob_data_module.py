"""LightningDataModule：装配 train/val/test/predict 的 DataLoader。"""

from __future__ import annotations

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from hft_lob.configs.experiment import ExperimentConfig


class LOBDataModule(pl.LightningDataModule):
    """装配 train/val/test/predict DataLoader；落盘与加载分离。"""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        persist_root: str,
    ) -> None:
        """初始化数据模块。

        Args:
            config: 实验配置根（含 loader/data 段超参数）。
            persist_root: 处理后数据持久化根目录。
        """
        super().__init__()
        self.config = config
        self.persist_root = persist_root

    def prepare_data(self) -> None:
        """落盘侧钩子：执行一次性数据准备（下载/预处理落盘）。"""
        raise NotImplementedError("LOBDataModule.prepare_data not implemented")

    def setup(self, stage: str) -> None:
        """加载侧钩子：按阶段构造数据集与 DataLoader 配置。

        Args:
            stage: Lightning 阶段标识（fit / validate / test / predict）。
        """
        raise NotImplementedError("LOBDataModule.setup not implemented")

    def train_dataloader(self) -> DataLoader:
        """返回训练 DataLoader。"""
        raise NotImplementedError("LOBDataModule.train_dataloader not implemented")

    def val_dataloader(self) -> DataLoader:
        """返回验证 DataLoader。"""
        raise NotImplementedError("LOBDataModule.val_dataloader not implemented")

    def test_dataloader(self) -> DataLoader:
        """返回测试 DataLoader。"""
        raise NotImplementedError("LOBDataModule.test_dataloader not implemented")

    def predict_dataloader(self) -> DataLoader:
        """返回预测 DataLoader。"""
        raise NotImplementedError("LOBDataModule.predict_dataloader not implemented")

    def teardown(self, stage: str) -> None:
        """释放阶段资源。

        Args:
            stage: Lightning 阶段标识。
        """
        raise NotImplementedError("LOBDataModule.teardown not implemented")
