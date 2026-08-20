"""BiN：双向归一化层（沿时间维与特征维同时归一化）。"""

from __future__ import annotations

import lightning.pytorch as pl
import torch


class BiN(pl.LightningModule):
    """BiN：时间维与特征维归一化后按可学习权重混合的归一化层。"""

    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        """初始化 BiN 层。

        Args:
            d2: 特征维输出尺寸。
            d1: 特征维输入尺寸。
            t1: 时间维输入尺寸。
            t2: 时间维输出尺寸。
        """
        raise NotImplementedError("BiN.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            归一化后的张量。
        """
        raise NotImplementedError("BiN.forward not implemented")
