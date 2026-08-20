"""MVP baseline 模型接口（需求文档 §17）。"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn


class ZeroBaseline(nn.Module):
    """Baseline 0：对每个样本恒定预测零收益。"""

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """无参数拟合，返回自身。"""
        raise NotImplementedError("ZeroBaseline.fit not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, 1]`` 的全零预测。"""
        raise NotImplementedError("ZeroBaseline.forward not implemented")


class ImbalanceBaseline(nn.Module):
    """Baseline 1：使用 anchor 快照盘口不平衡度预测收益。

    ``bid_volume_indices`` / ``ask_volume_indices`` 显式注入，避免模型依赖
    特征列的隐式位置；可使用 L1 或 L1-L5 聚合不平衡度。
    """

    def __init__(
        self,
        *,
        bid_volume_indices: tuple[int, ...],
        ask_volume_indices: tuple[int, ...],
    ) -> None:
        raise NotImplementedError("ImbalanceBaseline.__init__ not implemented")

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """在训练段拟合 imbalance 到收益的线性映射。"""
        raise NotImplementedError("ImbalanceBaseline.fit not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用 anchor 帧不平衡度返回 ``[B, 1]`` 预测。"""
        raise NotImplementedError("ImbalanceBaseline.forward not implemented")


class RidgeBaseline(nn.Module):
    """Baseline 2：基于窗口特征的 Ridge 回归。

    窗口展平策略属于模型内部；``fit`` 只能使用训练段，拟合参数作为 module
    state 保存，以便复现实验和离线推理。
    """

    def __init__(self, *, num_features: int, history_snapshots: int, alpha: float = 1.0) -> None:
        raise NotImplementedError("RidgeBaseline.__init__ not implemented")

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """拟合带 L2 正则的线性回归。"""
        raise NotImplementedError("RidgeBaseline.fit not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``[B, 1]`` Ridge 预测。"""
        raise NotImplementedError("RidgeBaseline.forward not implemented")
