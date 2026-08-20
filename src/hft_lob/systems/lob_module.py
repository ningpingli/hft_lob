from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch
from torch import nn


class LOBLightningModule(L.LightningModule):
    """回归任务的 LightningModule 接口。"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 10,
    ) -> None:
        """初始化 LightningModule。"""
        super().__init__()
        # 此处不写实现，仅声明接口
        # 实际实现应：self.save_hyperparameters(ignore=["model", "criterion"])
        # 并保存 self.model 和 self.criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        纯前向传播方法。

        Args:
            x: 输入张量。

        Returns:
            模型原始输出（未经过损失函数处理）。
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """可选设置钩子。"""
        ...

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """训练步骤。"""
        ...

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """验证步骤。"""
        ...

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """测试步骤。"""
        ...

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """预测步骤。"""
        ...

    def configure_optimizers(self) -> dict[str, Any]:
        """配置优化器与调度器。"""
        ...

    def on_validation_epoch_end(self) -> None:
        """验证轮次结束钩子。"""
        ...

    def on_test_epoch_end(self) -> None:
        """测试轮次结束钩子。"""
        ...