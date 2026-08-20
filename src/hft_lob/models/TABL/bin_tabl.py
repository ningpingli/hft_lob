"""BiN_TABL / BiN_CTABL：双向归一化 + TABL 注意力层组合模型。"""

from __future__ import annotations

import torch
from torch import nn


class BiN_BTABL(nn.Module):
    """BiN_BTABL：BiN + BL 层 + TABL 层的 B(TABL) 架构。"""

    def __init__(self, d2: int, d1: int, t1: int, t2: int, d3: int, t3: int) -> None:
        """初始化 BiN_BTABL。

        Args:
            d2: BiN 特征维输出尺寸。
            d1: BiN 特征维输入尺寸。
            t1: BiN 时间维输入尺寸。
            t2: BiN 时间维输出尺寸。
            d3: TABL 特征维输出尺寸。
            t3: TABL 时间维输出尺寸。
        """
        raise NotImplementedError("BiN_BTABL.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("BiN_BTABL.forward not implemented")


class BiN_CTABL(nn.Module):
    """BiN_CTABL：BiN + 两个 BL 层 + TABL 层的 C(TABL) 架构。"""

    def __init__(
        self, d2: int, d1: int, t1: int, t2: int, d3: int, t3: int, d4: int, t4: int
    ) -> None:
        """初始化 BiN_CTABL。

        Args:
            d2: BiN 特征维输出尺寸。
            d1: BiN 特征维输入尺寸。
            t1: BiN 时间维输入尺寸。
            t2: BiN 时间维输出尺寸。
            d3: 第一个 BL 层特征维输出尺寸。
            t3: 第一个 BL 层时间维输出尺寸。
            d4: TABL 特征维输出尺寸。
            t4: TABL 时间维输出尺寸。
        """
        raise NotImplementedError("BiN_CTABL.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            模型输出。
        """
        raise NotImplementedError("BiN_CTABL.forward not implemented")
