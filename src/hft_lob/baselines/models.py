"""多标签 baseline 模型。"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from hft_lob.baselines.base import ValidatedBaseline


class ZeroBaseline(ValidatedBaseline):
    """对每个样本、每个配置标签恒定预测零收益。"""

    def __init__(self, *, target_count: int = 1) -> None:
        super().__init__(target_count=target_count)

    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        return None

    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        for _features, _targets in batches():
            return

    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros((features.shape[0], self.target_count))


class ImbalanceBaseline(ValidatedBaseline):
    """使用 anchor 快照盘口不平衡度预测多标签收益。"""

    slope: torch.Tensor
    intercept: torch.Tensor
    fitted: torch.Tensor

    def __init__(
        self,
        *,
        bid_volume_indices: tuple[int, ...],
        ask_volume_indices: tuple[int, ...],
        target_count: int = 1,
    ) -> None:
        super().__init__(target_count=target_count)
        if not bid_volume_indices or not ask_volume_indices:
            raise ValueError("bid/ask volume indices must not be empty")
        if len(bid_volume_indices) != len(ask_volume_indices):
            raise ValueError("bid/ask volume indices must have the same length")
        indices = (*bid_volume_indices, *ask_volume_indices)
        if min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError("volume indices must be non-negative and unique")
        self.bid_volume_indices = tuple(bid_volume_indices)
        self.ask_volume_indices = tuple(ask_volume_indices)
        self.register_buffer("slope", torch.zeros(target_count))
        self.register_buffer("intercept", torch.zeros(target_count))
        self.register_buffer("fitted", torch.tensor(False))

    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        imbalance = self._imbalance(features)
        mean_x = imbalance.mean()
        denominator = torch.sum((imbalance - mean_x).square())
        for position in range(self.target_count):
            target = targets[:, position]
            mean_y = target.mean()
            slope = (
                torch.sum((imbalance - mean_x) * (target - mean_y)) / denominator
                if denominator > 0
                else denominator.new_zeros(())
            )
            self.slope[position] = slope
            self.intercept[position] = mean_y - slope * mean_x
        self.fitted.fill_(True)

    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        features_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for features, targets in batches():
            features_list.append(features)
            targets_list.append(targets)
        self._fit_validated(torch.cat(features_list), torch.cat(targets_list))

    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("ImbalanceBaseline must be fitted before prediction")
        return self._imbalance(features).unsqueeze(1).mul(self.slope).add(self.intercept)

    def _imbalance(self, x: torch.Tensor) -> torch.Tensor:
        max_index = max(*self.bid_volume_indices, *self.ask_volume_indices)
        if max_index >= x.shape[-1]:
            raise ValueError(f"volume index {max_index} exceeds feature dimension {x.shape[-1]}")
        anchor = x[:, -1, :]
        bid = anchor[:, self.bid_volume_indices].sum(dim=1)
        ask = anchor[:, self.ask_volume_indices].sum(dim=1)
        total = bid + ask
        return torch.where(total.abs() > torch.finfo(x.dtype).eps, (bid - ask) / total, 0.0)


class RidgeBaseline(ValidatedBaseline):
    """基于窗口特征的多标签 Ridge 回归。"""

    weight: torch.Tensor
    intercept: torch.Tensor
    fitted: torch.Tensor
    num_features: int
    history_snapshots: int

    def __init__(
        self,
        *,
        num_features: int,
        history_snapshots: int,
        alpha: float = 1.0,
        target_count: int = 1,
    ) -> None:
        super().__init__(
            num_features=num_features,
            history_snapshots=history_snapshots,
            target_count=target_count,
        )
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.num_features = num_features
        self.history_snapshots = history_snapshots
        self.alpha = float(alpha)
        dimension = num_features * history_snapshots
        self.register_buffer("weight", torch.zeros(dimension, target_count))
        self.register_buffer("intercept", torch.zeros(target_count))
        self.register_buffer("fitted", torch.tensor(False))

    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        design = features.reshape(features.shape[0], -1).to(torch.float64)
        mean_x = design.mean(dim=0)
        centered_x = design - mean_x
        gram = centered_x.T @ centered_x
        gram += self.alpha * torch.eye(gram.shape[0], dtype=gram.dtype)
        for position in range(self.target_count):
            response = targets[:, position].to(torch.float64)
            mean_y = response.mean()
            rhs = centered_x.T @ (response - mean_y)
            try:
                weight = torch.linalg.solve(gram, rhs)
            except torch.linalg.LinAlgError:
                weight = torch.linalg.lstsq(gram, rhs).solution
            self.weight[:, position].copy_(weight.to(self.weight))
            self.intercept[position] = (mean_y - mean_x @ weight).to(self.intercept)
        self.fitted.fill_(True)

    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        features_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for features, targets in batches():
            features_list.append(features)
            targets_list.append(targets)
        self._fit_validated(torch.cat(features_list), torch.cat(targets_list))

    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("RidgeBaseline must be fitted before prediction")
        return features.reshape(features.shape[0], -1) @ self.weight + self.intercept
