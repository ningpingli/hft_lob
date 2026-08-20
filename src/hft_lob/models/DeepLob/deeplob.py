"""DeepLOB：双流 CNN + Inception + LSTM 的 LOB 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class DeepLOB(nn.Module):
    """DeepLOB：卷积块 + Inception 模块 + LSTM 回归模型。"""

    def __init__(
        self,
        num_features: int | None = None,
        levels: int | None = None,
    ) -> None:
        """初始化 DeepLOB。

        Args:
            num_features: 每快照特征数（None 时为 40）。
            levels: 盘口档位数（None 时为 10）。
        """
        raise NotImplementedError("DeepLOB.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("DeepLOB.forward not implemented")
