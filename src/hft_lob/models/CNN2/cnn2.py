"""CNN2：五层卷积神经网络 LOB 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class CNN2(nn.Module):
    """CNN2：五层卷积（含 BatchNorm/PReLU）+ 全连接回归模型。"""

    def __init__(
        self,
        num_features: int = 20,
        num_classes: int = 1,
        history_length: int = 100,
        temp: int | None = None,
    ) -> None:
        """初始化 CNN2。

        Args:
            num_features: 每快照特征数。
            num_classes: 输出类别数（回归为 1）。
            history_length: 历史窗口长度。
            temp: 卷积池化堆叠后的时间长度；None 时由 history_length 推导。
        """
        raise NotImplementedError("CNN2.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("CNN2.forward not implemented")
