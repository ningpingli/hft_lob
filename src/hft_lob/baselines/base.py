"""Baseline 统一契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, Self, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class BaselineModel(Protocol):
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self: ...

    def fit_batches(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> Self: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class ValidatedBaseline(nn.Module, ABC):
    """Baseline 模板：统一输入校验和多标签输出契约。"""

    def __init__(
        self,
        *,
        num_features: int | None = None,
        history_snapshots: int | None = None,
        target_count: int = 1,
    ) -> None:
        super().__init__()
        if (num_features is None) != (history_snapshots is None):
            raise ValueError("num_features and history_snapshots must be configured together")
        if num_features is not None and (
            num_features <= 0 or history_snapshots is None or history_snapshots <= 0
        ):
            raise ValueError("num_features and history_snapshots must be > 0")
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        self.num_features = num_features
        self.history_snapshots = history_snapshots
        self.target_count = target_count

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        features, targets = self._validate_xy(x, y)
        self._fit_validated(features, targets)
        return self

    def fit_batches(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> Self:
        self._fit_batches_validated(lambda: self._validated_batches(batches()))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_validated(self._validate_x(x))

    @abstractmethod
    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None: ...

    @abstractmethod
    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None: ...

    @abstractmethod
    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor: ...

    def _validated_batches(
        self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        seen = False
        for x, y in batches:
            seen = True
            yield self._validate_xy(x, y)
        if not seen:
            raise ValueError("training batches must not be empty")

    def _validate_x(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or x.ndim != 3:
            raise ValueError(f"x must be a floating tensor with shape [B,T,F], got {tuple(x.shape)}")
        if any(size == 0 for size in x.shape):
            raise ValueError("x dimensions must be non-empty")
        if self.num_features is not None and x.shape[2] != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {x.shape[2]}")
        if self.history_snapshots is not None and x.shape[1] != self.history_snapshots:
            raise ValueError(f"expected {self.history_snapshots} snapshots, got {x.shape[1]}")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        return x

    def _validate_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._validate_x(x)
        if not y.is_floating_point() or y.ndim != 2:
            raise ValueError("y must be a floating tensor with shape [B,L]")
        if y.shape != (features.shape[0], self.target_count):
            raise ValueError(
                f"x and y must have matching batch sizes and y shape [B,{self.target_count}]"
            )
        if not torch.isfinite(y).all():
            raise ValueError("y must contain only finite values")
        return features, y
