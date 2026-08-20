"""MVP 轻量 MLP 回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """展平 LOB 历史窗口的轻量 MLP，输出严格为 [B,1]。"""

    def __init__(
        self,
        *,
        num_features: int,
        history_snapshots: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        raise NotImplementedError("MLP.__init__ not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：``[B,1,T,F] -> [B,1]``。"""
        raise NotImplementedError("MLP.forward not implemented")
