"""Lightning 训练管线：回归 LightningModule 与批量梯度下降管理器。

回归任务：模型预测连续的未来收益，损失可配置（mse / mae / huber），
IC（预测与已实现收益的 Pearson 相关）为监控指标。wandb 集成为尽力而为：
初始化失败时降级为 offline 模式，最终降级为无 logger 的 Trainer。
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn


class LOBLightningModule(pl.LightningModule):
    """回归任务的 LightningModule：封装模型、损失与训练/验证/测试钩子。"""

    def __init__(
        self,
        model: nn.Module,
        experiment_id: str,
        general_hyperparameters: dict[str, Any],
        model_hyperparameters: dict[str, Any],
    ) -> None:
        """初始化 LightningModule。

        Args:
            model: 待训练的模型。
            experiment_id: 实验 ID。
            general_hyperparameters: 通用超参数。
            model_hyperparameters: 模型超参数（loss / 学习率等）。
        """
        raise NotImplementedError("LOBLightningModule.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("LOBLightningModule.forward not implemented")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ANN401
        """训练步：forward + loss 计算与收集。

        Args:
            batch: (输入, 目标) 批次。
            batch_idx: 批次下标。

        Returns:
            损失张量。
        """
        raise NotImplementedError("LOBLightningModule.training_step not implemented")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ANN401
        """验证步：forward + loss 计算与收集。

        Args:
            batch: (输入, 目标) 批次。
            batch_idx: 批次下标。

        Returns:
            损失张量。
        """
        raise NotImplementedError("LOBLightningModule.validation_step not implemented")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ANN401
        """测试步：forward + loss 计算与收集。

        Args:
            batch: (输入, 目标) 批次。
            batch_idx: 批次下标。

        Returns:
            损失张量。
        """
        raise NotImplementedError("LOBLightningModule.test_step not implemented")

    def on_test_epoch_end(self) -> None:
        """测试期结束钩子：记录测试损失/IC 并持久化预测。"""
        raise NotImplementedError("LOBLightningModule.on_test_epoch_end not implemented")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """配置优化器（AdamW，参数来自 model_hyperparameters）。"""
        raise NotImplementedError("LOBLightningModule.configure_optimizers not implemented")

    def on_validation_epoch_end(self) -> None:
        """验证期结束钩子：记录训练/验证损失与 IC、写 metrics.csv 并打印日志。"""
        raise NotImplementedError("LOBLightningModule.on_validation_epoch_end not implemented")


class BatchGDManager:
    """批量梯度下降训练管理器：构建 Trainer、执行训练与测试。"""

    def __init__(
        self,
        experiment_id: str,
        model: nn.Module,
        train_loader: Any,  # noqa: ANN401
        val_loader: Any,  # noqa: ANN401
        test_loader: Any,  # noqa: ANN401
        epochs: int,
        patience: int,
        general_hyperparameters: dict[str, Any],
        model_hyperparameters: dict[str, Any],
    ) -> None:
        """初始化训练管理器。

        Args:
            experiment_id: 实验 ID。
            model: 待训练的模型。
            train_loader: 训练 DataLoader。
            val_loader: 验证 DataLoader。
            test_loader: 测试 DataLoader。
            epochs: 最大训练轮数。
            patience: 早停耐心值。
            general_hyperparameters: 通用超参数。
            model_hyperparameters: 模型超参数。
        """
        raise NotImplementedError("BatchGDManager.__init__ not implemented")

    def train(self) -> None:
        """执行训练：构建 Trainer（best-effort wandb）并 fit。"""
        raise NotImplementedError("BatchGDManager.train not implemented")

    def test(self) -> None:
        """加载最佳检查点在测试集上评估（logger-less Trainer）。"""
        raise NotImplementedError("BatchGDManager.test not implemented")
