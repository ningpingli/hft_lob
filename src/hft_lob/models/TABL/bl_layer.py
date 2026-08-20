"""BL 层：双线性映射 + ReLU 激活层。"""

from __future__ import annotations

import torch
from torch import nn


class BL_layer(nn.Module):
    """BL_layer：``ReLU(W1 @ X @ W2 + B)`` 的双线性映射层。"""

    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        """初始化 BL 层。

        Args:
            d2: 特征维输出尺寸。
            d1: 特征维输入尺寸。
            t1: 时间维输入尺寸。
            t2: 时间维输出尺寸。
        """
        super().__init__()
        weight1 = torch.empty(d2, d1)
        self.W1 = nn.Parameter(weight1)
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")

        weight2 = torch.empty(t1, t2)
        self.W2 = nn.Parameter(weight2)
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")

        bias1 = torch.zeros((d2, t2))
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, d1, t1)``。

        Returns:
            映射后的张量 ``(N, d2, t2)``。
        """
        x = self.activation(self.W1 @ x @ self.W2 + self.B)

        return x
