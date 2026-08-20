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
        raise NotImplementedError("BL_layer.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            映射后的张量。
        """
        raise NotImplementedError("BL_layer.forward not implemented")
