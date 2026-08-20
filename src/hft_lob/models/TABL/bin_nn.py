"""BiN：双向归一化层（沿时间维与特征维同时归一化）。"""

from __future__ import annotations

import torch
from torch import nn


class BiN(nn.Module):
    """BiN：时间维与特征维归一化后按可学习权重混合的归一化层。"""

    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        """初始化 BiN 层。

        Args:
            d2: 特征维输出尺寸。
            d1: 特征维输入尺寸。
            t1: 时间维输入尺寸。
            t2: 时间维输出尺寸。
        """
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.t2 = t2
        self.d2 = d2

        bias1 = torch.empty(t1, 1)
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)

        l1 = torch.empty(t1, 1)
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)

        bias2 = torch.empty(d1, 1)
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        l2 = torch.empty(d1, 1)
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)

        y1 = torch.empty(1,)
        self.y1 = nn.Parameter(y1)
        nn.init.constant_(self.y1, 0.5)

        y2 = torch.empty(1,)
        self.y2 = nn.Parameter(y2)
        nn.init.constant_(self.y2, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 ``(N, d1, t1)``。

        Returns:
            归一化后的张量。
        """
        # 原地钳制两个混合标量保持非负：重建为新张量（旧实现）既硬编码设备
        # （破坏 CPU 前向），又使原 Parameter 脱离优化器（冻结）。
        with torch.no_grad():
            self.y1.clamp_(min=0.01)
            self.y2.clamp_(min=0.01)

        device = x.device

        # 沿时间维归一化
        T2 = torch.ones([self.t1, 1], device=device)
        x2 = torch.mean(x, dim=2)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))

        std = torch.std(x, dim=2)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))
        # 部分时间切片 std 为 0 会产生 inf；置为 1。
        std[std < 1e-4] = 1

        diff = x - (x2 @ (T2.T))
        Z2 = diff / (std @ (T2.T))

        X2 = self.l2 @ T2.T
        X2 = X2 * Z2
        X2 = X2 + (self.B2 @ T2.T)

        # 沿特征维归一化
        T1 = torch.ones([self.d1, 1], device=device)
        x1 = torch.mean(x, dim=1)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))

        std = torch.std(x, dim=1)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))
        # 与时间维相同的零 std 保护（缺失会产生 NaN/inf）。
        std[std < 1e-4] = 1

        op1 = x1 @ T1.T
        op1 = torch.permute(op1, (0, 2, 1))

        op2 = std @ T1.T
        op2 = torch.permute(op2, (0, 2, 1))

        z1 = (x - op1) / (op2)
        X1 = (T1 @ self.l1.T)
        X1 = X1 * z1
        X1 = X1 + (T1 @ self.B1.T)

        # 时间维与特征维归一化的加权融合
        x = self.y1 * X1 + self.y2 * X2

        return x
