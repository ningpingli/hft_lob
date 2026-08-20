"""TABL 层：时间感知双向线性（temporal-aware bilinear）注意力层。"""

from __future__ import annotations

import torch
from torch import nn


class TABL_layer(nn.Module):
    """TABL_layer：软注意力加权的时间依赖建模层。"""

    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        """初始化 TABL 层。

        Args:
            d2: 特征维输出尺寸。
            d1: 特征维输入尺寸。
            t1: 时间维输入尺寸。
            t2: 时间维输出尺寸。
        """
        super().__init__()
        self.t1 = t1

        weight = torch.empty(d2, d1)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")

        weight2 = torch.empty(t1, t1)
        self.W = nn.Parameter(weight2)
        nn.init.constant_(self.W, 1 / t1)

        weight3 = torch.empty(t1, t2)
        self.W2 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")

        bias1 = torch.empty(d2, t2)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        l_init = torch.empty(1,)
        self.l = nn.Parameter(l_init)
        nn.init.constant_(self.l, 0.5)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            X: 输入张量 ``(N, d1, t1)``。

        Returns:
            映射后的张量 ``(N, d2, t2)``。
        """
        # 原地钳制混合标量到 [0, 1]：替换 Parameter（旧实现）会使其脱离优化器，
        # 且使用普通 CPU 张量会在 CUDA 模型上产生设备不匹配。
        with torch.no_grad():
            self.l.clamp_(min=0.0, max=1.0)

        device = X.device

        # 沿 X 的第一维（特征）建模依赖，保持时间顺序不变（论文式 7）
        X = self.W1 @ X

        # 对角线强制为常量 1（device 跟随输入；旧实现硬编码 CUDA 破坏 CPU 前向）
        eye = torch.eye(self.t1, dtype=torch.float32, device=device)
        W = self.W - self.W * eye + eye / self.t1

        # 注意力：第二步学习各时间实例对彼此的重要性
        E = X @ W

        # 注意力掩码
        A = torch.softmax(E, dim=-1)

        # 软注意力机制：注意力掩码 A 用于削弱不重要元素的影响
        X = self.l[0] * (X) + (1.0 - self.l[0]) * X * A

        # 最后一步估计时间映射 W2，加偏置平移
        y = X @ self.W2 + self.B
        return y
