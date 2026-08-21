"""Baseline 统一契约（需求文档 §17）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, Self, runtime_checkable

import torch
from torch import nn


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


class ValidatedBaseline(nn.Module, ABC):
    """Baseline 模板：统一输入校验和空 batch 契约。

    子类只实现已经验证的数据上的模型算法，避免每个模型重复调用验证函数。
    """

    def __init__(
        self,
        *,
        num_features: int | None = None,
        history_snapshots: int | None = None,
    ) -> None:
        super().__init__()
        if (num_features is None) != (history_snapshots is None):
            raise ValueError("num_features and history_snapshots must be configured together")
        if num_features is not None and (
            num_features <= 0 or history_snapshots is None or history_snapshots <= 0
        ):
            raise ValueError("num_features and history_snapshots must be > 0")
        self.num_features = num_features
        self.history_snapshots = history_snapshots

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        features, targets = self._validate_xy(x, y)
        self._fit_validated(features, targets)
        return self

    def fit_batches(
        self,
        batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    ) -> Self:
        self._fit_batches_validated(lambda: self._validated_batches(batches()))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_validated(self._validate_x(x))

    @abstractmethod
    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """拟合一个已经验证的完整 tensor。"""

    @abstractmethod
    def _fit_batches_validated(
        self,
        batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        """流式拟合已经验证的 batches。"""

    @abstractmethod
    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        """对已经验证的输入执行预测。"""

    def _validated_batches(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        seen = False
        for x, y in batches:
            seen = True
            yield self._validate_xy(x, y)
        if not seen:
            raise ValueError("training batches must not be empty")

    def _validate_x(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or x.ndim != 3:
            raise ValueError(
                f"x must be a floating tensor with shape [B,T,F], got {tuple(x.shape)}"
            )
        if any(size == 0 for size in x.shape):
            raise ValueError("x dimensions must be non-empty")
        if self.num_features is not None and x.shape[2] != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {x.shape[2]}")
        if self.history_snapshots is not None and x.shape[1] != self.history_snapshots:
            raise ValueError(f"expected {self.history_snapshots} snapshots, got {x.shape[1]}")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        return x

    def _validate_xy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._validate_x(x)
        if not y.is_floating_point() or y.ndim not in (1, 2):
            raise ValueError("y must be a floating tensor with shape [B] or [B,1]")
        targets = y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("x and y must have matching batch sizes and y shape [B] or [B,1]")
        if not torch.isfinite(targets).all():
            raise ValueError("y must contain only finite values")
        return features, targets
