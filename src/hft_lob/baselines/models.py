"""MVP baseline 模型接口（需求文档 §17）。"""

from __future__ import annotations

from typing import Self, cast

import torch
from torch import nn


class ZeroBaseline(nn.Module):
    """Baseline 0：对每个样本恒定预测零收益。"""

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """无参数拟合，返回自身。"""
        _validate_xy(x, y)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, 1]`` 的全零预测。"""
        _validate_x(x)
        return x.new_zeros((x.shape[0], 1))


class ImbalanceBaseline(nn.Module):
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

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """在训练段拟合 imbalance 到收益的线性映射。"""
        features, targets = _validate_xy(x, y)
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
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用 anchor 帧不平衡度返回 ``[B, 1]`` 预测。"""
        features = _validate_x(x)
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


class RidgeBaseline(nn.Module):
    """Baseline 2：基于窗口特征的 Ridge 回归。

    窗口展平策略属于模型内部；``fit`` 只能使用训练段，拟合参数作为 module
    state 保存，以便复现实验和离线推理。
    """

    weight: torch.Tensor
    intercept: torch.Tensor
    fitted: torch.Tensor

    def __init__(self, *, num_features: int, history_snapshots: int, alpha: float = 1.0) -> None:
        super().__init__()
        _validate_dimensions(num_features, history_snapshots)
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.num_features = num_features
        self.history_snapshots = history_snapshots
        self.alpha = float(alpha)
        self.register_buffer("weight", torch.zeros(num_features * history_snapshots, 1))
        self.register_buffer("intercept", torch.zeros(1))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """拟合带 L2 正则的线性回归。"""
        features, targets = _validate_xy(x, y, self.num_features, self.history_snapshots)
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
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``[B, 1]`` Ridge 预测。"""
        features = _validate_x(x, self.num_features, self.history_snapshots)
        if not bool(self.fitted.item()):
            raise RuntimeError("RidgeBaseline must be fitted before prediction")
        return features.reshape(features.shape[0], -1) @ self.weight + self.intercept


class MLPBaseline(nn.Module):
    """Baseline 3：展平历史窗口后的轻量 MLP 回归模型。"""

    fitted: torch.Tensor

    def __init__(
        self,
        *,
        num_features: int,
        history_snapshots: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        _validate_dimensions(num_features, history_snapshots)
        if hidden_dim <= 0 or epochs <= 0 or learning_rate <= 0:
            raise ValueError("hidden_dim, epochs and learning_rate must be > 0")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_features = num_features
        self.history_snapshots = history_snapshots
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * history_snapshots, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """仅使用当前 fold training split 拟合 MLP。"""
        features, targets = _validate_xy(x, y, self.num_features, self.history_snapshots)
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.HuberLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.network(features), targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
        self.fitted.fill_(True)
        self.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回严格 ``[B, 1]`` 的预测。"""
        features = _validate_x(x, self.num_features, self.history_snapshots)
        if not bool(self.fitted.item()):
            raise RuntimeError("MLPBaseline must be fitted before prediction")
        return cast(torch.Tensor, self.network(features))


def _validate_dimensions(num_features: int, history_snapshots: int) -> None:
    if num_features <= 0 or history_snapshots <= 0:
        raise ValueError("num_features and history_snapshots must be > 0")


def _validate_x(
    x: torch.Tensor,
    num_features: int | None = None,
    history_snapshots: int | None = None,
) -> torch.Tensor:
    if not x.is_floating_point() or x.ndim != 3:
        raise ValueError(f"x must be a floating tensor with shape [B,T,F], got {tuple(x.shape)}")
    if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError("x dimensions must be non-empty")
    if num_features is not None and x.shape[2] != num_features:
        raise ValueError(f"expected {num_features} features, got {x.shape[2]}")
    if history_snapshots is not None and x.shape[1] != history_snapshots:
        raise ValueError(f"expected {history_snapshots} snapshots, got {x.shape[1]}")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    return x


def _validate_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    num_features: int | None = None,
    history_snapshots: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = _validate_x(x, num_features, history_snapshots)
    if not y.is_floating_point() or y.ndim not in (1, 2):
        raise ValueError("y must be a floating tensor with shape [B] or [B,1]")
    targets = y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("x and y must have matching batch sizes and y shape [B] or [B,1]")
    if not torch.isfinite(targets).all():
        raise ValueError("y must contain only finite values")
    return features, targets
