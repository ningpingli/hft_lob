"""BiN_TABL / BiN_CTABL：双向归一化 + TABL 注意力层组合模型。"""

from __future__ import annotations

import torch
from torch import nn

from hft_lob.models.TABL.bin_nn import BiN
from hft_lob.models.TABL.bl_layer import BL_layer
from hft_lob.models.TABL.tabl_layer import TABL_layer


def _enforce_max_norm(w: torch.Tensor) -> None:
    """将权重矩阵范数钳制到 10（TABL 系列前向的稳定性约束，两模型共用）。"""
    with torch.no_grad():
        if torch.linalg.matrix_norm(w) > 10.0:
            norm = torch.linalg.matrix_norm(w)
            desired = torch.clamp(norm, min=0.0, max=10.0)
            w *= desired / (1e-8 + norm)


class BiN_BTABL(nn.Module):
    """BiN_BTABL：BiN + BL 层 + TABL 层的 B(TABL) 架构。"""

    def __init__(
        self,
        d2: int,
        d1: int,
        t1: int,
        t2: int,
        d3: int,
        t3: int,
    ) -> None:
        """初始化 BiN_BTABL。

        Args:
            d2: BiN 特征维输出尺寸。
            d1: BiN 特征维输入尺寸。
            t1: BiN 时间维输入尺寸。
            t2: BiN 时间维输出尺寸。
            d3: TABL 特征维输出尺寸。
            t3: TABL 时间维输出尺寸。
        """
        super().__init__()

        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.TABL = TABL_layer(d3, d2, t2, t3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, 1, t1, d1)``。

        Returns:
            模型输出 ``(N, d3)``。
        """
        x = x.squeeze(1)
        # 先过 BiN 层，再使用 B(TABL) 架构
        x = torch.permute(x, (0, 2, 1))

        x = self.BiN(x)

        _enforce_max_norm(self.BL.W1.data)
        _enforce_max_norm(self.BL.W2.data)
        x = self.BL(x)
        x = self.dropout(x)

        _enforce_max_norm(self.TABL.W1.data)
        _enforce_max_norm(self.TABL.W.data)
        _enforce_max_norm(self.TABL.W2.data)
        x = self.TABL(x)
        x = torch.squeeze(x, 2)
        return x


class BiN_CTABL(nn.Module):
    """BiN_CTABL：BiN + 两个 BL 层 + TABL 层的 C(TABL) 架构。"""

    def __init__(
        self, d2: int, d1: int, t1: int, t2: int, d3: int, t3: int, d4: int,
        t4: int,
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
        super().__init__()

        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        self.TABL = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, 1, t1, d1)``。

        Returns:
            模型输出 ``(N, d4)``。
        """
        x = x.squeeze(1)
        # 先过 BiN 层，再使用 C(TABL) 架构
        x = torch.permute(x, (0, 2, 1))

        x = self.BiN(x)

        _enforce_max_norm(self.BL.W1.data)
        _enforce_max_norm(self.BL.W2.data)
        x = self.BL(x)
        x = self.dropout(x)

        _enforce_max_norm(self.BL2.W1.data)
        _enforce_max_norm(self.BL2.W2.data)
        x = self.BL2(x)
        x = self.dropout(x)

        _enforce_max_norm(self.TABL.W1.data)
        _enforce_max_norm(self.TABL.W.data)
        _enforce_max_norm(self.TABL.W2.data)
        x = self.TABL(x)
        x = torch.squeeze(x, 2)
        return x
