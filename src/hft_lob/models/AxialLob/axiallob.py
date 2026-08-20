"""AxialLOB：门控轴向注意力 LOB 模型。"""

from __future__ import annotations

import torch
from torch import nn


class GatedAxialAttention(nn.Module):
    """多头上轴自注意力（沿 LOB 帧的高度或宽度方向），带门控与相对位置嵌入。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        dim: int,
        flag: bool,
    ) -> None:
        """初始化门控轴向注意力层。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            heads: 注意力头数。
            dim: 轴向维度长度。
            flag: 为 True 时沿宽度方向计算注意力，否则沿高度方向。
        """
        raise NotImplementedError("GatedAxialAttention.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            轴向注意力输出张量。
        """
        raise NotImplementedError("GatedAxialAttention.forward not implemented")

    def reset_parameters(self) -> None:
        """重置相对位置嵌入参数。"""
        raise NotImplementedError("GatedAxialAttention.reset_parameters not implemented")


class AxialLOB(nn.Module):
    """AxialLOB：CNN 卷积 + 门控轴向注意力 + 残差 + 池化回归模型。"""

    def __init__(
        self,
        W: int = 40,
        H: int = 100,
        c_in: int = 32,
        c_out: int = 32,
        c_final: int = 4,
        n_heads: int = 4,
        pool_kernel: tuple[int, int] = (1, 4),
        pool_stride: tuple[int, int] = (1, 4),
    ) -> None:
        """初始化 AxialLOB。

        Args:
            W: 输入帧宽度（特征数，40 或 20）。
            H: 输入帧高度（时间快照数）。
            c_in: CNN 输入通道数。
            c_out: 轴向层输出通道数。
            c_final: 最终输出通道数。
            n_heads: 注意力头数。
            pool_kernel: 平均池化核。
            pool_stride: 平均池化步长。
        """
        raise NotImplementedError("AxialLOB.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量（形状 ``(N, 1, H, W)``）。

        Returns:
            模型输出。
        """
        raise NotImplementedError("AxialLOB.forward not implemented")
