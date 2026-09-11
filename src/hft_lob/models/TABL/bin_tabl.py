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


def enforce_weight_constraints(model: nn.Module) -> None:
    """对模型内全部 BL / TABL 层权重施加范数上限（训练循环在每个优化器步之后调用）。

    约束是权重的性质、与输入无关，故每个更新步之后投影一次即可：两次前向之间权重不变，
    前向看到的取值与「每次前向都检查」的旧实现相同（见 `tests/test_model_contracts.py`
    的等价性用例）。放在 ``forward`` 内会让每次前向都做 ``matrix_norm`` 并触发
    device→CPU 同步——binctabl 单样本延迟 +230 μs（+28%），见 issue #16。
    """
    for module in model.modules():
        if isinstance(module, BL_layer):
            _enforce_max_norm(module.W1.data)
            _enforce_max_norm(module.W2.data)
        elif isinstance(module, TABL_layer):
            _enforce_max_norm(module.W1.data)
            _enforce_max_norm(module.W.data)
            _enforce_max_norm(module.W2.data)


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
        super().__init__()

        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.TABL = TABL_layer(d3, d2, t2, t3)
        self.dropout = nn.Dropout(0.1)

        # 初始化后立即满足范数约束：旧实现在第一次前向内钳制，构造时钳制与之逐位等价。
        enforce_weight_constraints(self)

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

        # 权重范数约束不在前向内执行：见 enforce_weight_constraints（训练循环按步调用）。
        x = self.BL(x)
        x = self.dropout(x)

        x = self.TABL(x)
        x = torch.squeeze(x, 2)
        return x


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
        super().__init__()

        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        self.TABL = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(0.1)

        # 初始化后立即满足范数约束：旧实现在第一次前向内钳制，构造时钳制与之逐位等价。
        enforce_weight_constraints(self)

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

        # 权重范数约束不在前向内执行：见 enforce_weight_constraints（训练循环按步调用）。
        x = self.BL(x)
        x = self.dropout(x)

        x = self.BL2(x)
        x = self.dropout(x)

        x = self.TABL(x)
        x = torch.squeeze(x, 2)
        return x
