"""Baseline 统一契约（需求文档 §17）。"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import torch


@runtime_checkable
class BaselineModel(Protocol):
    """可训练/推理 baseline 的最小公共协议。

    所有 baseline 接收与神经网络相同的 ``[B, 1, T, F]`` 输入，并输出
    ``[B, 1]``。无需拟合的 baseline 也实现 ``fit``，直接返回自身。
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """使用训练段数据拟合；不得读取 validation/test 数据。"""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """预测未来收益：``[B, 1, T, F] -> [B, 1]``。"""
        ...
