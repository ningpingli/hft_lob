"""MVP baseline 模型接口（需求文档 §17）。"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from hft_lob.baselines.base import ValidatedBaseline


class ZeroBaseline(ValidatedBaseline):
    """Baseline 0：对每个样本恒定预测零收益。"""

    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        return None

    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        for _features, _targets in batches():
            return

    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, 1]`` 的全零预测。"""
        return features.new_zeros((features.shape[0], 1))


class ImbalanceBaseline(ValidatedBaseline):
    """Baseline 1：使用 anchor 快照盘口不平衡度预测收益。

    ``bid_volume_indices`` / ``ask_volume_indices`` 显式注入，避免模型依赖
    特征列的隐式位置；可使用 L1 或 L1-L5 聚合不平衡度。
    """

    slope: torch.Tensor
    intercept: torch.Tensor
    fitted: torch.Tensor

    def __init__(
        self,
        *,
        bid_volume_indices: tuple[int, ...],
        ask_volume_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not bid_volume_indices or not ask_volume_indices:
            raise ValueError("bid/ask volume indices must not be empty")
        if len(bid_volume_indices) != len(ask_volume_indices):
            raise ValueError("bid/ask volume indices must have the same length")
        indices = (*bid_volume_indices, *ask_volume_indices)
        if min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError("volume indices must be non-negative and unique")
        self.bid_volume_indices = tuple(bid_volume_indices)
        self.ask_volume_indices = tuple(ask_volume_indices)
        self.register_buffer("slope", torch.zeros(1))
        self.register_buffer("intercept", torch.zeros(1))
        self.register_buffer("fitted", torch.tensor(False))

    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """在训练段拟合 imbalance 到收益的线性映射。"""
        imbalance = self._imbalance(features)
        mean_x = imbalance.mean()
        mean_y = targets.mean()
        denominator = torch.sum((imbalance - mean_x).square())
        slope = (
            torch.sum((imbalance - mean_x) * (targets - mean_y)) / denominator
            if denominator > 0
            else denominator.new_zeros(())
        )
        self.slope.copy_(slope.reshape(1))
        self.intercept.copy_((mean_y - slope * mean_x).reshape(1))
        self.fitted.fill_(True)

    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        count = 0
        sum_x = torch.zeros((), dtype=torch.float64)
        sum_y = torch.zeros((), dtype=torch.float64)
        sum_xx = torch.zeros((), dtype=torch.float64)
        sum_xy = torch.zeros((), dtype=torch.float64)
        for features, targets in batches():
            imbalance = self._imbalance(features).to(torch.float64)
            targets = targets.to(torch.float64)
            count += targets.numel()
            sum_x += imbalance.sum()
            sum_y += targets.sum()
            sum_xx += imbalance.square().sum()
            sum_xy += (imbalance * targets).sum()
        denominator = sum_xx - sum_x.square() / count
        slope = (
            (sum_xy - sum_x * sum_y / count) / denominator
            if denominator > 0
            else denominator.new_zeros(())
        )
        intercept = sum_y / count - slope * sum_x / count
        self.slope.copy_(slope.to(self.slope).reshape(1))
        self.intercept.copy_(intercept.to(self.intercept).reshape(1))
        self.fitted.fill_(True)

    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        """使用 anchor 帧不平衡度返回 ``[B, 1]`` 预测。"""
        if not bool(self.fitted.item()):
            raise RuntimeError("ImbalanceBaseline must be fitted before prediction")
        return self._imbalance(features).mul(self.slope).add(self.intercept).unsqueeze(1)

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
    """Baseline 2：基于窗口特征的 Ridge 回归。

    窗口展平策略属于模型内部；``fit`` 只能使用训练段，拟合参数作为 module
    state 保存，以便复现实验和离线推理。
    """

    weight: torch.Tensor
    intercept: torch.Tensor
    fitted: torch.Tensor
    num_features: int
    history_snapshots: int

    def __init__(self, *, num_features: int, history_snapshots: int, alpha: float = 1.0) -> None:
        super().__init__(num_features=num_features, history_snapshots=history_snapshots)
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.num_features = num_features
        self.history_snapshots = history_snapshots
        self.alpha = float(alpha)
        self.register_buffer("weight", torch.zeros(num_features * history_snapshots, 1))
        self.register_buffer("intercept", torch.zeros(1))
        self.register_buffer("fitted", torch.tensor(False))

    def _fit_validated(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """拟合带 L2 正则的线性回归。"""
        design = features.reshape(features.shape[0], -1).to(torch.float64)
        response = targets.to(torch.float64)
        mean_x = design.mean(dim=0, keepdim=True)
        mean_y = response.mean()
        centered_x = design - mean_x
        centered_y = response - mean_y
        response_column = centered_y.unsqueeze(1)
        use_dual = design.shape[1] > design.shape[0]
        matrix_size = design.shape[0] if use_dual else design.shape[1]
        identity = torch.eye(matrix_size, dtype=design.dtype, device=design.device)
        gram = (
            centered_x @ centered_x.T if use_dual else centered_x.T @ centered_x
        ) + self.alpha * identity
        rhs = response_column if use_dual else centered_x.T @ response_column
        try:
            solution = torch.linalg.solve(gram, rhs)
        except torch.linalg.LinAlgError:
            solution = torch.linalg.lstsq(gram, rhs).solution
        weight = centered_x.T @ solution if use_dual else solution
        intercept = mean_y - (mean_x @ weight).squeeze()
        self.weight.copy_(weight.to(self.weight))
        self.intercept.copy_(intercept.reshape(1).to(self.intercept))
        self.fitted.fill_(True)

    def _fit_batches_validated(
        self, batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        dimension = self.num_features * self.history_snapshots
        count = 0
        sum_x = torch.zeros(dimension, dtype=torch.float64)
        sum_y = torch.zeros((), dtype=torch.float64)
        xtx = torch.zeros((dimension, dimension), dtype=torch.float64)
        xty = torch.zeros((dimension, 1), dtype=torch.float64)
        for features, targets in batches():
            design = features.reshape(features.shape[0], -1).to(torch.float64)
            response = targets.to(torch.float64)
            count += response.numel()
            sum_x += design.sum(dim=0)
            sum_y += response.sum()
            xtx += design.T @ design
            xty += design.T @ response.unsqueeze(1)
        mean_x = sum_x / count
        mean_y = sum_y / count
        gram = xtx - sum_x.unsqueeze(1) @ sum_x.unsqueeze(0) / count
        gram += self.alpha * torch.eye(dimension, dtype=torch.float64)
        rhs = xty - sum_x.unsqueeze(1) * mean_y
        try:
            weight = torch.linalg.solve(gram, rhs)
        except torch.linalg.LinAlgError:
            weight = torch.linalg.lstsq(gram, rhs).solution
        intercept = mean_y - mean_x @ weight[:, 0]
        self.weight.copy_(weight.to(self.weight))
        self.intercept.copy_(intercept.to(self.intercept).reshape(1))
        self.fitted.fill_(True)

    def _forward_validated(self, features: torch.Tensor) -> torch.Tensor:
        """返回 ``[B, 1]`` Ridge 预测。"""
        if not bool(self.fitted.item()):
            raise RuntimeError("RidgeBaseline must be fitted before prediction")
        return features.reshape(features.shape[0], -1) @ self.weight + self.intercept
