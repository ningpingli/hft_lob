"""多标签 baseline 模型。"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from hft_lob.baselines.base import ValidatedBaseline


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
