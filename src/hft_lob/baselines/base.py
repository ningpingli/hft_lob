"""Baseline 统一契约（需求文档 §17）。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, Self, runtime_checkable

import torch


@runtime_checkable
class BaselineModel(Protocol):
    """非梯度 baseline 的最小公共协议。

    Zero/Imbalance/Ridge/MLP 均接收与主模型相同的输入输出，并由
    ``BaselineRunner`` 统一适配为 PredictionArtifact。
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """使用训练段数据拟合；不得读取 validation/test 数据。"""
        ...

    def fit_batches(
        self,
        batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    ) -> Self:
        """流式拟合，训练数据不得整体物化到内存。"""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """预测未来收益：``[B, T, F] -> [B, 1]``。"""
        ...
