"""TABL 层：时间感知双向线性（temporal-aware bilinear）注意力层。"""

from __future__ import annotations

import lightning.pytorch as pl
import torch


class TABL_layer(pl.LightningModule):
    """TABL_layer：软注意力加权的时间依赖建模层。"""

    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        """初始化 TABL 层。

        Args:
            d2: 特征维输出尺寸。
            d1: 特征维输入尺寸。
            t1: 时间维输入尺寸。
            t2: 时间维输出尺寸。
        """
        raise NotImplementedError("TABL_layer.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            映射后的张量。
        """
        raise NotImplementedError("TABL_layer.forward not implemented")
